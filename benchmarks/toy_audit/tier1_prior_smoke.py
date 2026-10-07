"""Bounded, CUDA-only prior diagnosis; never fills ordinary Forge cells.

The protocol freezes the selected BCAP recipe and original task budgets/gates.
Only the declared prior changes. Raw states, arrays and logs stay in runs/api.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash, stable_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models
from .api_vectors import _bounds
from .reproducibility import reproducible_execution

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "reports/forge/tier1-prior-smoke/protocol.json"


def declaration():
    protocol = json.loads(PROTOCOL.read_text())
    for binding in protocol["tasks"].values():
        if file_hash(ROOT / binding["path"]) != binding["sha256"]:
            # Current tasks can advance; diagnostics retain the exact original
            # bytes, never the revised conditions. The frozen protocol hash is
            # unchanged and the archive must match its original task digest.
            archive = ROOT / "reports/forge/tier1-prior-smoke/frozen-tasks" / Path(binding["path"]).name
            if not archive.is_file() or file_hash(archive) != binding["sha256"]:
                raise ValueError("task changed after protocol freeze; declare a new study")
            binding["path"] = str(archive.relative_to(ROOT))
    if file_hash(ROOT / protocol["candidate_path"]) != protocol["candidate_sha256"]:
        raise ValueError("selected trainer changed after protocol freeze")
    return protocol


def scorer(task_id):
    if task_id == "gaussian1d_acquisition":
        from .gaussian1d_quality import sample_target, score_samples
    else:
        from benchmarks.transfer_suite.vector_tasks import sample_target
        from .ring16_quality import score_samples
    return sample_target, score_samples


def suffix(rows, field):
    count = 0
    for row in reversed(rows):
        if not row[field]:
            break
        count += 1
    return count


@reproducible_execution
def controls(*, device):
    """Oracle/destructive laws, with all input samples resident on CUDA.

    Frozen scorers and target/reference RNGs retain their declared CPU law.
    The reduced smoke question intentionally accepts some shape impostors.
    """
    protocol = declaration()
    result = []
    for task_id, binding in protocol["tasks"].items():
        task = json.loads((ROOT / binding["path"]).read_text())
        spec = task["execution"]["host_definition"]
        target, score = scorer(task_id)
        oracle = target(spec, 4096, torch.Generator(device="cpu").manual_seed(78013), 0).to(device)
        if task_id == "gaussian1d_acquisition":
            laws = {"oracle": (oracle, True, True),
                    "point_collapse": (torch.full_like(oracle, 2.), False, False),
                    "shift": (oracle + .5, False, False),
                    "double_width": (2. + 2. * (oracle - 2.), False, False),
                    "same_moment_uniform": ((2. + 3.**.5 * .5 * torch.linspace(-1., 1., 4096, device=device))[:, None], False, True),
                    "same_moment_two_atoms": (oracle.new_tensor([1.5, 2.5]).repeat(2048)[:, None], False, True)}
        else:
            centers = oracle.new_tensor(spec["means"])
            laws = {"oracle": (oracle, True, True),
                    "single_mode": (oracle[:256].repeat(16, 1) * 0 + centers[0], False, False),
                    "all_mode_atoms": (centers.repeat(256, 1), False, True),
                    "double_radius": (oracle * 2., False, False)}
        for name, (points, expected_full, expected_smoke) in laws.items():
            assert points.device.type == "cuda"
            metrics = score(points.cpu(), spec, 0)
            full = not _bounds(metrics, task["evaluation"]["thresholds"])
            smoke = not _bounds(metrics, protocol["smoke_projection"][task_id])
            result.append(dict(task=task_id, control=name, full_pass=full, smoke_pass=smoke,
                               expected_full=expected_full, expected_smoke=expected_smoke,
                               passed=(full == expected_full and smoke == expected_smoke), metrics=metrics,
                               input_device=str(points.device)))
    return dict(training_updates=0, passed=all(row["passed"] for row in result), controls=result)


@reproducible_execution
def trial(arm, task_id, output, *, device):
    protocol = declaration()
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("this study requires a GPU; CPU fallback is forbidden")
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
    task = deepcopy(original)
    task["id"] = task_id + "_prior_diagnostic_" + arm
    condition = protocol["arms"][arm]
    task["execution"]["prior"] = condition["prior"]
    task["execution"]["host_definition"]["particles"] = condition["particles"]
    task["requires_capabilities"] = [name for name in task["requires_capabilities"] if name != "mog_prior"]
    task["requires_capabilities"].append("particle_cloud" if condition["prior"]["kind"] == "particle_cloud" else "mog_prior")
    candidate = json.loads((ROOT / protocol["candidate_path"]).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    spec = task["execution"]["host_definition"]
    generator, critic = build_vector_models(context, spec)
    trainer = context.build_trainer(generator, critic, max_steps=spec["steps"])
    assert all(p.device.type == "cuda" for model in (generator, critic, trainer.prior) for p in model.parameters())
    target, score = scorer(task_id)
    data = context.streams.generator("data", component="target", purpose="training", device="cpu")
    evaluation = context.streams.generator("eval", component="live", purpose="samples")
    data_digest = hashlib.sha256()
    initial = dict(context.initialization)
    initial["prior_locations_sha256"] = state_digest(trainer.prior.z)
    atomic_json(output / "source.json", inspect_source(ROOT, extra_paths=(str(PROTOCOL.relative_to(ROOT)), protocol["candidate_path"])))
    torch.save(context.state_dict(), output / "initial-state.pt")
    rows, snapshots = [], []
    steps = sorted({math.ceil(i * spec["steps"] / 24) for i in range(1, 25)})
    started = time.monotonic()

    def observe(step):
        before = context.streams.audit()
        points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
        metrics = score(points, spec, step)
        full_failures = _bounds(metrics, task["evaluation"]["thresholds"])
        smoke_failures = _bounds(metrics, protocol["smoke_projection"][task_id])
        row = dict(step=step, metrics=metrics, full_pass=not full_failures, smoke_pass=not smoke_failures,
                   full_failed_bounds=full_failures, smoke_failed_bounds=smoke_failures)
        after = context.streams.audit()
        allowed = [key for key, binding in context.streams.manifest()["bindings"].items() if binding["family"] == "eval"]
        audit = context.streams.compare(before, after, allowed=allowed)
        if audit["unintended_rng_deviations"]:
            raise RuntimeError("evaluation consumed a training stream")
        snapshots.append(dict(step=step, samples=points, metrics=metrics))
        print(json.dumps(dict(event="observation", arm=arm, task=task_id, **row)), flush=True)
        return row

    # Initial illustration does not advance the frozen post-update scoring law.
    with context.streams.preserve():
        initial_row = observe(0)
    for step in range(1, spec["steps"] + 1):
        real = target(spec, trainer.recipe.batch_size, data, step - 1)
        data_digest.update(real.numpy().tobytes())
        trainer.step(real.to(device))
        if step in steps:
            rows.append(observe(step))
            if time.monotonic() - started > original["resources"]["timeout_seconds"]:
                raise TimeoutError("declared task allowance exceeded")
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    torch.save(context.state_dict(), output / "state.pt")
    torch.save(snapshots, output / "observations.pt")
    counts = {}
    for role, optimizer, parameters in (("generator", trainer.opt_g, trainer.G.parameters()),
                                        ("discriminator", trainer.opt_d, trainer.D.parameters()),
                                        ("prior", trainer.opt_g, trainer.prior.parameters())):
        counts[role] = min(int(optimizer.state[p]["step"]) for p in parameters)
    assert all(count == spec["steps"] for count in counts.values())
    result = dict(schema_version=1, scope=protocol["scope"], qualification_input=False, arm=arm, task=task_id,
                  protocol_sha256=file_hash(PROTOCOL), original_task_sha256=protocol["tasks"][task_id]["sha256"],
                  candidate_id=protocol["candidate_id"], particles=condition["particles"], prior=condition["prior"],
                  device=str(device), gpu=torch.cuda.get_device_name(device), runtime=runtime_manifest(),
                  deterministic=True, tf32=False, completed_updates=trainer.completed_steps,
                  optimizer_updates=counts, elapsed_seconds=elapsed, initial=initial, initial_metrics=initial_row["metrics"],
                  data_sequence_sha256=data_digest.hexdigest(), recipe=context.recipe.to_dict(),
                  recipe_sha256=stable_hash(context.recipe.to_dict()), rng=context.streams.manifest(),
                  full_terminal_suffix=suffix(rows, "full_pass"), smoke_terminal_suffix=suffix(rows, "smoke_pass"),
                  full_verdict="PASS" if suffix(rows, "full_pass") >= 5 else "FAIL",
                  smoke_verdict="PASS" if suffix(rows, "smoke_pass") >= 5 else "FAIL",
                  final_metrics=rows[-1]["metrics"], terminal_observations=rows[-5:],
                  final_failed_bounds=rows[-1]["full_failed_bounds"],
                  artifacts={name: file_hash(output / name) for name in ("initial-state.pt", "state.pt", "observations.pt", "source.json")})
    atomic_json(output / "receipt.json", result)
    atomic_json(output / "curve.json", rows)
    print(json.dumps(dict(event="completed", arm=arm, task=task_id, full=result["full_verdict"], smoke=result["smoke_verdict"], seconds=elapsed)), flush=True)
    return result


def publish(raw, destination):
    """Verify, summarize and render saved draws; no training or new model draws."""
    from .api_run import render_gif
    import numpy as np
    protocol = declaration()
    results = []
    destination.mkdir(parents=True, exist_ok=True)
    for arm in protocol["arms"]:
        for task_id in protocol["tasks"]:
            directory = raw / arm / task_id
            receipt = json.loads((directory / "receipt.json").read_text())
            for name, digest in receipt["artifacts"].items():
                if file_hash(directory / name) != digest:
                    raise ValueError("raw artifact hash mismatch")
            observations = torch.load(directory / "observations.pt", weights_only=True)
            card = json.loads((ROOT / protocol["tasks"][task_id]["path"]).read_text())
            target, _ = scorer(task_id)
            reference = target(card["execution"]["host_definition"], 4096, torch.Generator(device="cpu").manual_seed(78013), 0)
            frames = []
            for index in np.linspace(0, len(observations) - 1, 9).round().astype(int):
                observed = observations[index]
                samples = observed["samples"]
                metrics = observed["metrics"]
                if samples.shape[1] == 1:
                    edges = np.linspace(-2., 5., 57)
                    def density(values):
                        return np.histogram(values[:, 0].numpy(), edges)[0] / len(values) / np.diff(edges)
                    view = dict(kind="bar", title="Target and actual Gaussian histograms", target=density(reference), samples=density(samples),
                                bin_centers=(edges[:-1] + edges[1:]) / 2, bin_width=float(edges[1] - edges[0]), xlim=[-2., 5.], ylim=[0., 8.5], xlabel="x", ylabel="density")
                else:
                    view = dict(kind="scatter", title="Target and actual sixteen-mode ring", target=reference, samples=samples, xlim=[-4., 4.], ylim=[-4., 4.], xlabel="x", ylabel="y")
                view["caption"] = "Clean live public prior/G draws. Full original shape gate; reduced smoke result is reported separately."
                scalar = {k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))}
                frames.append(dict(step=observed["step"], metrics=scalar, passed=not _bounds(metrics, card["evaluation"]["thresholds"]), views=[view]))
            media = arm + "-" + task_id + ".gif"
            render_gif(dict(id=arm + "/" + task_id, goal=card["description"], default_steps=card["execution"]["steps"]), frames, destination / media,
                       full_budget=True, requested_steps=card["execution"]["steps"], final_verdict=receipt["full_verdict"])
            results.append({**receipt, "raw_receipt_sha256": file_hash(directory / "receipt.json"),
                            "source": json.loads((directory / "source.json").read_text())["digest"],
                            "source_commit": json.loads((directory / "source.json").read_text())["origin_commit"],
                            "gif": media, "gif_sha256": file_hash(destination / media)})
    for task_id in protocol["tasks"]:
        peers = [row for row in results if row["task"] == task_id]
        for key in ("data_sequence_sha256",):
            assert len({row[key] for row in peers}) == 1, key
        for key in ("generator", "discriminator"):
            assert len({stable_hash(row["initial"][key]) for row in peers}) == 1, key
        for count in {row["particles"] for row in peers}:
            assert len({row["initial"]["prior_locations_sha256"] for row in peers if row["particles"] == count}) == 1
        # Recipe differs only in the explicitly studied task-owned prior fields.
        assert len({stable_hash({k:v for k,v in row["recipe"].items() if k not in ("prior_kind", "num_particles")}) for row in peers}) == 1
    assert len({row["source"] for row in results}) == 1
    atomic_json(destination / "results.json", dict(protocol=protocol["id"], qualification_input=False, paired_checks_passed=True, runs=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("controls", "run", "trial", "publish"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/api/tier1-prior-smoke-v1")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.action in ("controls", "run", "trial") and torch.device(args.device).type != "cuda":
        parser.error("GPU execution is required")
    if args.action == "controls":
        result = controls(device=args.device)
        atomic_json(args.output, result)
        return 0 if result["passed"] else 1
    if args.action == "trial":
        trial(args.arm, args.task, args.output, device=args.device)
    elif args.action == "publish":
        publish(args.raw, args.output)
    else:
        protocol = declaration()
        args.output.mkdir(parents=True, exist_ok=False)
        check = controls(device=args.device)
        atomic_json(args.output / "controls.json", check)
        if not check["passed"]:
            raise RuntimeError("scorer controls failed; no training authorized")
        errors = []
        for arm in protocol["arms"]:
            for task_id, binding in protocol["tasks"].items():
                directory = args.output / arm / task_id
                directory.parent.mkdir(parents=True, exist_ok=True)
                log = directory.parent / (task_id + ".log")
                command = [sys.executable, "-u", "-m", "benchmarks.toy_audit.tier1_prior_smoke", "trial", "--arm", arm, "--task", task_id,
                           "--device", args.device, "--output", str(directory)]
                print(json.dumps(dict(event="start", arm=arm, task=task_id, log=str(log))), flush=True)
                with log.open("w") as stream:
                    try:
                        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                timeout=binding["timeout_seconds"], env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
                        if result.returncode:
                            errors.append(dict(arm=arm, task=task_id, returncode=result.returncode))
                    except subprocess.TimeoutExpired:
                        errors.append(dict(arm=arm, task=task_id, status="TIMEOUT"))
                print(json.dumps(dict(event="finish", arm=arm, task=task_id, errors=len(errors))), flush=True)
        atomic_json(args.output / "completion.json", dict(errors=errors, training_runs=len(protocol["arms"]) * len(protocol["tasks"]), retries=0))
        return 1 if errors else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
