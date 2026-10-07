"""Prospective CUDA-only, fresh-live Ring16 intervention diagnostics.

No checkpoint is loaded into a candidate. Boundary-only acts at update401,
after the untouched live400 prefix; every-step acts from update1. Research
patches are confined to this process and never change package defaults.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.sources import inspect_source, runtime_manifest
from experiments.forge.state import state_digest, require_same_formulation, require_optimizer_steps
from experiments.forge.vectorprofiles import build_vector_models
from benchmarks.transfer_suite.vector_tasks import sample_target
from .api_vectors import _bounds
from .reproducibility import reproducible_execution
from .ring16_quality import score_samples

ROOT = Path(__file__).resolve().parents[2]
MODULE = "benchmarks.toy_audit.ring16_interventions"
CHECKS = sorted({math.ceil(i * 400 / 24) + block
                 for block in range(0, 1600, 400) for i in range(1, 25)})


def read(path):
    return json.loads(path.read_text())


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return deepcopy(value)


def declaration(path):
    protocol = read(path)
    combined = protocol.get("intervention_kind") == "serialized_truncation"
    arm_count = 1 if combined else 2
    schedules = (["every_step"] if combined else
                 ["every_step", "boundary_only"] if protocol.get("intervention_kind") == "autograd_serialization"
                 else ["boundary_only", "every_step"])
    required = {"schema_version": 1, "status": "ready", "seed": 0,
                "scope": "cuda_ring16_intervention_diagnostic", "steps": 1600,
                "recipe_horizon": 400, "boundary_update": 401, "max_attempts": arm_count,
                "max_new_host_updates": 1600 * arm_count, "max_reserved_seconds": 300 * arm_count,
                "timeout_seconds_per_arm": 300, "scientific_retries": 0,
                "max_new_scoring_draws": 97 * arm_count, "qualification_input": False,
                "schedules": schedules}
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"unsupported or unfrozen protocol field: {key}")
    if protocol.get("intervention_kind", "polar") not in ("polar", "autograd_serialization", "serialized_truncation"):
        raise ValueError("unsupported intervention kind")
    helper = protocol["mechanism_module"].replace(".", "/") + ".py"
    for key in (protocol["task_path"], protocol["candidate_path"], helper,
                "benchmarks/toy_audit/ring16_interventions.py"):
        if key not in protocol["bindings"]:
            raise ValueError("missing source binding: " + key)
    for name, expected in protocol["bindings"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("frozen source binding differs: " + name)
    task = read(ROOT / protocol["task_path"])
    spec = task["execution"]["host_definition"]
    if (task["execution"]["initializer"] != "deterministic_orthogonal"
            or task["execution"]["prior"] != {"kind": "mog", "sigma": .1,
                                               "standardize": False, "learnable": True}
            or (spec["batch"], spec["particles"], spec["z_dim"], spec["steps"]) != (128, 256, 4, 1600)
            or task["evaluation"]["observations"] != 96):
        raise ValueError("baseline task contract differs")
    return protocol


def require_cuda(device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback and no training reservation consumed")


def suffix(rows):
    n = 0
    for row in reversed(rows):
        if not row["full_pass"]:
            break
        n += 1
    return n


def baseline_projection(state):
    """Remove only the explicitly declared extra stream registrations."""
    projected = cpu(state)
    # Only two deliberately new streams are absent in the historical registry.
    for key in list(projected["streams"]["states"]):
        binding = projected["streams"]["manifest"]["bindings"][key]
        if (binding["component"], binding["purpose"]) in {
                ("ring16_intervention", "weak_directions"), ("live", "confirmation")}:
            del projected["streams"]["states"][key]
            del projected["streams"]["manifest"]["bindings"][key]
    return projected


def prefix_identity(state, baseline_path, expected_sha):
    """Compare saved values only; NEVER restore the reference into the learner."""
    if file_hash(baseline_path) != expected_sha:
        raise ValueError("archived baseline checkpoint bytes differ")
    baseline = torch.load(baseline_path, weights_only=True, map_location="cpu")
    projected = baseline_projection(state)
    return {"exact_except_new_stream_registration": state_digest(projected) == state_digest(baseline),
            "reference_checkpoint_sha256": expected_sha,
            "reference_state_digest": state_digest(baseline),
            "candidate_projected_state_digest": state_digest(projected)}


@reproducible_execution
def trial(protocol_path, output, arm, baseline_path, *, device):
    require_cuda(device)
    protocol = declaration(protocol_path)
    if arm not in protocol["schedules"]:
        raise ValueError("arm not declared")
    if read(output / "frozen-protocol.json") != protocol:
        raise ValueError("campaign protocol changed")
    directory = output / arm
    directory.mkdir(exist_ok=False)  # Exclusive attempt; interrupted arms cannot be retried.
    atomic_json(directory / "reservation.json", {"updates": 1600, "seconds": 300, "scoring_draws": 97})
    started, completed, draws = time.monotonic(), 0, 0
    context = trainer = None
    observations, snapshots, confirmation = [], [], None
    status, error, prefix = "INCOMPLETE", None, None
    boundary = None
    from particlegan.optim import dualnorm
    original = dualnorm.polar_factor
    try:
        if not torch.autograd.is_multithreading_enabled():
            raise ValueError("caller autograd multithreading must be enabled")
        helper = importlib.import_module(protocol["mechanism_module"])
        task = read(ROOT / protocol["task_path"])
        candidate = read(ROOT / protocol["candidate_path"])
        if protocol.get("intervention_kind") in ("autograd_serialization", "serialized_truncation"):
            candidate = helper.trainer_candidate(candidate, arm)
        context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
        g, d = build_vector_models(context, task["execution"]["host_definition"])
        trainer = context.build_trainer(g, d, max_steps=400)
        if any(p.device.type != "cuda" or p.dtype != torch.float32
               for model in (g, d, trainer.prior) for p in model.parameters()):
            raise ValueError("all model/prior parameters must be CUDA float32")
        data = context.streams.generator("data", component="target", purpose="training", device="cpu")
        evaluation = context.streams.generator("eval", component="live", purpose="samples")
        confirm_rng = context.streams.generator("eval", component="live", purpose="confirmation")
        perturb_rng = context.streams.generator("noise", component="ring16_intervention", purpose="weak_directions")
        initial = cpu(context.state_dict())
        torch.save(initial, directory / "initial-state.pt")
        paths = [str(protocol_path.relative_to(ROOT)), protocol["candidate_path"],
                 protocol["mechanism_module"].replace(".", "/") + ".py", "benchmarks/toy_audit/ring16_interventions.py"]
        source = inspect_source(ROOT, extra_paths=tuple(paths))
        atomic_json(directory / "source.json", source)
        atomic_json(directory / "applied.json", {"candidate": candidate, "context": context.receipt(),
                    "serial_backward": trainer.serial_backward, "schedule": arm})
        ambient = {"initial": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}}
        digest = hashlib.sha256()
        calls = 0

        def polar(matrix):
            nonlocal calls
            update = trainer.completed_steps + 1
            if arm == "every_step" or update == protocol["boundary_update"]:
                calls += 1
                result = helper.polar(matrix, perturb_rng)
                if result.shape != matrix.shape or result.dtype != matrix.dtype or result.device != matrix.device:
                    raise ValueError("intervention changed tensor contract")
                if not bool(torch.isfinite(result).all()):
                    raise FloatingPointError("nonfinite intervention direction")
                return result
            return original(matrix)

        if protocol.get("intervention_kind", "polar") in ("polar", "serialized_truncation"):
            dualnorm.polar_factor = polar
        while trainer.completed_steps < 1600:
            if time.monotonic() - started > 300:
                raise TimeoutError("declared arm timeout exceeded")
            step = trainer.completed_steps + 1
            real = sample_target(task["execution"]["host_definition"], 128, data, step - 1)
            digest.update(real.numpy().tobytes())
            active = arm == "every_step" or step == protocol["boundary_update"]
            scope = (helper.step_context(arm, step)
                     if protocol.get("intervention_kind") in ("autograd_serialization", "serialized_truncation") else nullcontext())
            with scope:
                update = trainer.step(real.to(device), collect_stats=False)
            if protocol.get("intervention_kind") == "autograd_serialization" and active:
                calls += 1
            if not torch.autograd.is_multithreading_enabled():
                raise ValueError("update leaked serialized autograd into caller")
            completed = trainer.completed_steps
            if any(not bool(torch.isfinite(v).all()) for v in update.values() if isinstance(v, torch.Tensor)):
                raise FloatingPointError("nonfinite update")
            if step in CHECKS:
                points = trainer.sample(4096, generator=evaluation, output_noise=False).detach().cpu()
                draws += 1
                metrics = score_samples(points, task["execution"]["host_definition"], step)
                failed = _bounds(metrics, task["evaluation"]["thresholds"])
                row = {"step": step, "metrics": metrics, "full_pass": not failed, "failed_bounds": failed}
                observations.append(row)
                snapshots.append({**row, "samples": points})
                print(json.dumps({"event": "observation", "arm": arm, "step": step,
                                  "pass": not failed, "covariance": metrics["component_covariance_error"],
                                  "hq": metrics["hq"]}), flush=True)
                if not failed and confirmation is None:
                    # Separate stream and unchanged state: one independent check, no best-of retry.
                    witness = cpu(context.state_dict())
                    torch.save(witness, directory / "acquisition-state.pt")
                    confirmed = trainer.sample(4096, generator=confirm_rng, output_noise=False).detach().cpu()
                    draws += 1
                    cm = score_samples(confirmed, task["execution"]["host_definition"], step)
                    cf = _bounds(cm, task["evaluation"]["thresholds"])
                    torch.save(confirmed, directory / "confirmation-samples.pt")
                    confirmation = {"step": step, "full_pass": not cf, "metrics": cm, "failed_bounds": cf,
                                    "state_digest_before_draw": state_digest(witness)}
            if step == 400:
                state = cpu(context.state_dict())
                torch.save(state, directory / "prefix-state.pt")
                prefix = prefix_identity(state, baseline_path, protocol["baseline_checkpoint_sha256"])
                if arm == "boundary_only" and not prefix["exact_except_new_stream_registration"]:
                    raise ValueError("fresh live400 does not match archived baseline; no attribution possible")
                trainer.extend_execution(1600)
            if step == 401 and protocol.get("intervention_kind") in ("autograd_serialization", "serialized_truncation"):
                state = cpu(context.state_dict())
                torch.save(state, directory / "state401.pt")
                ambient["401"] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
                if arm == "boundary_only":
                    projected = baseline_projection(state)
                    # The earlier causal experiment stopped at401; only the
                    # external allowance differs. Recipe horizon stays400.
                    projected["trainer"]["max_steps"] = 401
                    observed = state_digest(projected)
                    boundary = {"exact_except_extra_stream_registration_and_execution_allowance":
                                observed == protocol["serialized401_state_digest"],
                                "projected_state_digest": observed,
                                "reference_state_digest": protocol["serialized401_state_digest"]}
                    if not boundary["exact_except_extra_stream_registration_and_execution_allowance"]:
                        raise ValueError("serialized401 differs from archived third trajectory; stop without retry")
        torch.cuda.synchronize(device)
        if time.monotonic() - started > 300:
            raise TimeoutError("arm exceeded reservation after final synchronization")
        final = cpu(context.state_dict())
        require_same_formulation(initial, final)
        require_optimizer_steps(final, 1600)
        status = "COMPLETE"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        dualnorm.polar_factor = original
        if context is not None and trainer is not None:
            # Save the full actual state even after timeout/nonfinite/metadata failure.
            try:
                torch.save(cpu(context.state_dict()), directory / "state.pt")
            except Exception as exc:
                error = {"primary": error, "checkpoint_error": repr(exc)}
                status = "INCOMPLETE"
        if "ambient" in locals():
            ambient["final"] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}
            torch.save(cpu(ambient), directory / "ambient-states.pt")
        torch.save(snapshots, directory / "observations.pt")
        atomic_json(directory / "curve.json", observations)
        elapsed = time.monotonic() - started
        if elapsed > 300:
            status = "INCOMPLETE"
            error = {"primary": error, "budget_error": "elapsed time exceeded 300-second reservation"}
        atomic_json(directory / "receipt.json", {
            "schema_version": 1, "scope": protocol["scope"], "qualification_input": False,
            "arm": arm, "status": status, "error": error, "device": str(device),
            "runtime": runtime_manifest(), "gpu_model": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)), "cuda_runtime": torch.version.cuda,
            "protocol_sha256": file_hash(protocol_path),
            "completed_updates": completed, "new_updates": completed, "elapsed_seconds": elapsed,
            "scoring_draws": draws, "prefix_identity": prefix,
            "boundary_identity": boundary,
            "batch_sequence_sha256": locals().get("digest").hexdigest() if "digest" in locals() else None,
            "intervention_calls": locals().get("calls", 0),
            "any_full_pass": any(p["full_pass"] for p in observations),
            "confirmed_smoke": status == "COMPLETE" and confirmation is not None and confirmation["full_pass"],
            "first_full_pass": next((p["step"] for p in observations if p["full_pass"]), None),
            "confirmation": confirmation, "terminal_suffix": suffix(observations),
            "five_terminal_verdict": "INCOMPLETE" if status != "COMPLETE" else "PASS" if suffix(observations) >= 5 else "FAIL",
            "final_metrics": observations[-1]["metrics"] if observations else None,
            "recipe": context.recipe.to_dict() if context is not None else None,
            "artifacts": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}})
        print(json.dumps({"event": "arm_complete", "arm": arm, "status": status,
                          "updates": completed, "seconds": elapsed, "error": error}), flush=True)
    return status == "COMPLETE"


def summarize(output):
    protocol = read(output / "frozen-protocol.json")
    controller_path = output / "controller-attempts.json"
    attempts = read(controller_path) if controller_path.exists() else []
    admitted = {row["arm"]: row for row in attempts}
    rows = []
    for arm in protocol["schedules"]:
        directory = output / arm
        if (directory / "receipt.json").exists():
            r = read(directory / "receipt.json")
            rows.append({k: r[k] for k in ("arm", "status", "error", "completed_updates", "elapsed_seconds",
                         "scoring_draws", "prefix_identity", "batch_sequence_sha256", "any_full_pass",
                         "first_full_pass", "confirmed_smoke", "terminal_suffix", "five_terminal_verdict", "final_metrics")})
        else:
            rows.append({"arm": arm, "status": "INTERRUPTED" if directory.exists() or arm in admitted else "UNMEASURED",
                         "controller_attempt": admitted.get(arm),
                         "conservative_seconds_debit": 300 if directory.exists() or arm in admitted else 0})
    result = {"schema_version": 1, "qualification_input": False, "protocol_id": protocol["id"], "arms": rows,
              "reserved_updates": 1600 * sum((output / a).exists() or a in admitted for a in protocol["schedules"]),
              "reserved_seconds": 300 * sum((output / a).exists() or a in admitted for a in protocol["schedules"])}
    atomic_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)


def render(output):
    # CPU saved-output rendering only; never constructs or samples a model.
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    entries = []
    for directory in (output / a for a in read(output / "frozen-protocol.json")["schedules"]):
        source = directory / "observations.pt"
        if not source.exists():
            continue
        rows = torch.load(source, weights_only=True, map_location="cpu")
        if not rows:
            continue
        frames, steps = [], []
        for i in sorted(set(np.linspace(0, len(rows) - 1, 9).round().astype(int))):
            row = rows[i]
            steps.append(row["step"])
            fig, ax = plt.subplots(figsize=(5, 5))
            samples = row["samples"].numpy()
            ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=.25)
            for theta in np.arange(16) * 2 * np.pi / 16:
                ax.add_patch(Circle((3*np.cos(theta), 3*np.sin(theta)), .3, fill=False, color="orange"))
            ax.set(xlim=(-6, 6), ylim=(-6, 6), aspect="equal", xlabel="x", ylabel="y")
            ax.set_title(f"{directory.name}, update {row['step']}: {'PASS' if row['full_pass'] else 'FAIL'}\n"
                         f"cov={row['metrics']['component_covariance_error']:.3f}, HQ={row['metrics']['hq']:.3f}")
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            plt.close(fig)
            buffer.seek(0)
            with Image.open(buffer) as im:
                frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE))
        gif = directory / "actual-training.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=600, loop=0, optimize=False)
        entries.append({"arm": directory.name, "frame_steps": steps, "source_sha256": file_hash(source),
                        "gif_sha256": file_hash(gif), "source_commit": read(directory / "source.json")["origin_commit"]})
    atomic_json(output / "media-index.json", {"training_updates": 0, "model_forwards": 0, "sampling_draws": 0, "entries": entries})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run", "trial", "summarize", "render"))
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("boundary_only", "every_step"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--baseline", type=Path, default=Path("/home/martyn/dev/ParticleGAN/runs/api/ring16-restart-diagnostic-v1/live/prefix-state.pt"))
    args = p.parse_args()
    protocol_path, output = args.protocol.resolve(), args.output.resolve()
    protocol = declaration(protocol_path)
    if args.action == "plan":
        print(json.dumps({"id": protocol["id"], "schedules": protocol["schedules"], "updates": protocol["max_new_host_updates"],
                          "training_reservation_seconds": protocol["max_reserved_seconds"], "cuda_available": torch.cuda.is_available(),
                          "training_launched": False}), flush=True)
    elif args.action in ("summarize", "render"):
        if read(output / "frozen-protocol.json") != protocol:
            raise ValueError("campaign protocol changed")
        (summarize if args.action == "summarize" else render)(output)
    elif args.action == "trial":
        require_cuda(args.device)  # Guard before RNG setup or creating an attempt.
        if not trial(protocol_path, output, args.arm, args.baseline.resolve(), device=args.device):
            raise SystemExit(1)
    else:
        require_cuda(args.device)
        if file_hash(args.baseline) != protocol["baseline_checkpoint_sha256"]:
            raise ValueError("baseline artifact unavailable or changed")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "frozen-protocol.json", protocol)
        failures, attempts = [], []
        for arm in protocol["schedules"]:
            # Admission precedes child imports, so an early process failure
            # cannot erase the attempt from campaign reservation accounting.
            attempt = {"arm": arm, "status": "STARTED", "reserved_seconds": 300,
                       "reserved_updates": 1600, "reserved_scoring_draws": 97}
            attempts.append(attempt)
            atomic_json(output / "controller-attempts.json", attempts)
            launched = time.monotonic()
            print(json.dumps({"event": "arm_start", "arm": arm}), flush=True)
            try:
                result = subprocess.run([sys.executable, "-u", "-m", MODULE, "trial", "--protocol", str(protocol_path),
                                         "--output", str(output), "--arm", arm, "--baseline", str(args.baseline.resolve()),
                                         "--device", args.device], timeout=300, check=False)
                failed = result.returncode != 0
            except subprocess.TimeoutExpired:
                # Reservation survives a killed child. Never retry it or prevent
                # the other independent, already reserved schedule from running.
                directory = output / arm
                directory.mkdir(exist_ok=True)
                atomic_json(directory / "interruption.json", {"status": "INTERRUPTED", "charged_seconds": 300,
                            "reserved_updates": 1600, "reserved_scoring_draws": 97, "retry_permitted": False})
                failed = True
            attempt["status"] = "FAILED" if failed else "FINISHED"
            attempt["child_wall_seconds"] = time.monotonic() - launched
            atomic_json(output / "controller-attempts.json", attempts)
            if failed:
                failures.append(arm)
        summarize(output)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
