"""Bounded, exact-source coverage of four shipped conditional toy entries.

CPU refusal and obsolete public API calls are retained, never adapted. Each
entry has one 120-second allowance, including its short observer parity probe.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

REVISION = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
CASES = {
    "source-family-00": dict(label="Sparse mixed identity symbols", family="sparse", config=None),
    "source-family-01": dict(label="Sparse mixed split symbols", family="sparse", config="configs/sparse/discrete/gst_split_s1.yaml"),
    "source-family-02": dict(label="Analytic denoising grid, one class", family="denoising", config="configs/denoising/diagnostics/ddgan_class_free_28k_s24002.yaml"),
    "source-family-03": dict(label="Analytic denoising grid, four classes", family="denoising", config="configs/denoising/default.toml"),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __package__:
    from .source_demos import digest, isolated, read_progress, state_hash, write
    from . import source_conditional_scoring as scoring
else:
    helper = load_module("_conditional_helpers", Path(__file__).with_name("observer-utils.py"))
    digest, isolated, read_progress, state_hash, write = (getattr(helper, name) for name in
                                                        ("digest", "isolated", "read_progress", "state_hash", "write"))
    scoring = load_module("_conditional_scoring", Path(__file__).with_name("scoring.py"))


class PrefixStop(Exception):
    pass


class WallTimeout(Exception):
    pass


def owned(local):
    """Complete observable model/optimizer/RNG owners, including gradients."""
    result = dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state())
    for name, value in local.items():
        if isinstance(value, torch.nn.Module):
            result[name] = dict(state=value.state_dict(),
                                modes={k: part.training for k, part in value.named_modules()},
                                gradients={k: parameter.grad for k, parameter in value.named_parameters()},
                                requires_grad={k: parameter.requires_grad for k, parameter in value.named_parameters()})
        elif isinstance(value, torch.optim.Optimizer):
            result[name] = value.state_dict()
        elif isinstance(value, torch.Generator):
            result[name] = value.get_state()
        elif isinstance(value, torch.Tensor):
            result[name] = value
    return result


def optimizer_receipt(optimizer):
    groups = []
    for group in optimizer.param_groups:
        values = {key: value for key, value in group.items()
                  if key != "params" and isinstance(value, (str, float, int, bool, type(None), tuple, list))}
        values["parameter_shapes"] = [list(parameter.shape) for parameter in group["params"]]
        groups.append(values)
    return dict(class_name=type(optimizer).__qualname__, groups=groups)


def effective(local):
    prior = local["prior"]
    return dict(recipe=local["recipe"].to_dict(),
                actual_prior=dict(class_name=type(prior).__qualname__, locations=list(prior.z.shape),
                                  buffers={name: value.detach().cpu().tolist() for name, value in prior.named_buffers()
                                           if value.numel() <= 32}),
                optimizers={name: optimizer_receipt(local[name]) for name in ("opt_G", "opt_D")},
                sampling_law="The source sample_z closure: class-block or shared indices followed by pr(idx); generator eval hard-symbol law",
                initializer="Source deterministic_orthogonal_ on the recipe prior; G seed0 and D seed1 after constructors",
                training_seed=int(local["cfg"]["seed"]))


class Capture:
    def __init__(self, module, output, *, observe=True, stop=None):
        self.module, self.output, self.observe, self.stop = module, Path(output), observe, stop
        self.rows, self.completed, self.prefix, self.effective = [], 0, None, None
        self.started = time.monotonic()
        tree = ast.parse(Path(module["train"].__code__.co_filename).read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
        self.loop_line = next(node.lineno for node in ast.walk(train) if isinstance(node, ast.For)
                              and isinstance(node.target, ast.Name) and node.target.id == "step")

    def trace(self, frame, event, argument):
        if event == "call" and frame.f_code is self.module["train"].__code__:
            return self.local_trace
        return None

    def local_trace(self, frame, event, argument):
        local = frame.f_locals
        if event == "line" and frame.f_lineno == self.loop_line:
            completed = local.get("done", 0)
            self.completed = completed
            if self.effective is None:
                self.effective = effective(local)
                if self.observe:
                    write(self.output / "effective.json", self.effective)
            if self.stop is not None and completed >= self.stop:
                self.prefix = state_hash(owned(local))
                raise PrefixStop()
            if self.observe and (completed in (0, 1, 10) or completed % 100 == 0):
                if not self.rows or self.rows[-1]["step"] != completed:
                    self.capture(local, completed)
            if self.observe:
                write(self.output / "progress.json", dict(completed_update_pairs=completed,
                      last_scored_step=self.rows[-1]["step"] if self.rows else None, observations=len(self.rows)))
        elif event == "return" and isinstance(argument, dict):
            self.completed = argument["steps"]
            if self.observe and (not self.rows or self.rows[-1]["step"] != self.completed):
                self.capture(local, self.completed)
        return self.local_trace

    def capture(self, local, step):
        before = state_hash(owned(local))
        arrays, metrics = {}, {}
        toy = local["toy"]
        requested = torch.arange(scoring.COUNT) % toy.n_classes
        for cohort, prefix in (("live", ""), ("ema", "ema_")):
            generator, prior = local[prefix + "G"], local[prefix + "prior"]
            with isolated((generator, prior)):
                stream = torch.Generator().manual_seed(scoring.EVAL_SEED)
                generated, _ = local["draw_fakes"](generator, prior, requested, stream)
                emitted = generated["logits"].argmax(1)
                values = scoring.sparse_metrics(toy, generated["x"], emitted, requested)
                values["original_bar"] = self.module["convergence_bar"](values, toy.n_modes)
                metrics[cohort] = values
                arrays.update({cohort + "_x": generated["x"].numpy(), cohort + "_symbol": emitted.numpy()})
        with isolated(()):
            real, symbols, _ = toy.sample_given_class(requested, torch.Generator().manual_seed(scoring.EVAL_SEED + 1))
        arrays.update(requested=requested.numpy(), reference_x=real.numpy(), reference_symbol=symbols.numpy(),
                      centers=toy.centers.numpy(), active=toy.active.numpy())
        after = state_hash(owned(local))
        if before != after:
            raise AssertionError("Observer changed the original model/optimizer/gradient/RNG owners")
        index = len(self.rows)
        self.output.mkdir(parents=True, exist_ok=True)
        cloud = self.output / f"cloud-{index:03d}.npz"
        checkpoint = self.output / f"state-{index:03d}.pt"
        np.savez_compressed(cloud, **arrays)
        torch.save(owned(local), checkpoint)
        row = dict(step=step, seconds=time.monotonic() - self.started, **metrics,
                   observer_state_and_rng_pure=True, training_owner_hash=before,
                   cloud_sha256=digest(cloud), state_sha256=digest(checkpoint))
        self.rows.append(row)
        with (self.output / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(dict(event="CONDITIONAL_SOURCE_OBSERVATION", step=step,
                             live_pass=metrics["live"]["passed"], ema_pass=metrics["ema"]["passed"])), flush=True)


def run_sparse(module, cfg, output, *, observe, stop=None):
    cfg = dict(cfg, out_dir=str(output / "original-output"))
    capture = Capture(module, output / "raw", observe=observe, stop=stop)
    previous = sys.gettrace()
    try:
        sys.settrace(capture.trace)
        result = module["train"](cfg, torch.device("cpu"))
    except PrefixStop:
        result = None
    finally:
        sys.settrace(previous)
    return capture, result


def parity(module, cfg, output):
    hashes = []
    started = time.monotonic()
    for observe in (False, True):
        random.seed(5101); np.random.seed(5101)
        capture, _ = run_sparse(module, cfg, output / ("observed" if observe else "baseline"), observe=observe, stop=2)
        hashes.append(capture.prefix)
    result = dict(status="PASS" if hashes[0] == hashes[1] else "FAIL", completed_update_pairs=2,
                  source_budget_unchanged=cfg["total_steps"], training_owner_hashes=hashes,
                  matched="Model/optimizer/gradient/local-input tensors, module modes, requires-grad flags and global/owned RNGs",
                  extra_software_probe_update_pairs=4, seconds=time.monotonic() - started)
    write(output / "parity.json", result)
    if result["status"] != "PASS":
        raise AssertionError("Exact observer prefix parity failed")
    return result


def terminal(rows, cohort, field="passed"):
    suffix = 0
    for row in reversed(rows):
        passed = row[cohort][field] if field == "passed" else row[cohort]["original_bar"]["bar_all"]
        if not passed:
            break
        suffix += 1
    return dict(passing_suffix=suffix, passed=suffix >= 5)


def prepare(output, identifier):
    entry = CASES[identifier]
    snapshot = output / "source"
    sys.path.insert(0, str(snapshot))
    script = snapshot / "experiments" / ("train_sparse.py" if entry["family"] == "sparse" else "train_denoising.py")
    module = runpy.run_path(str(script))
    config = snapshot / entry["config"] if entry["config"] else None
    if entry["family"] == "sparse":
        cfg = module["load_config"](str(config) if config else None)
        from particlegan import get_recipe
        recipe = get_recipe(z_dim=cfg["z_dim"], num_particles=cfg["num_particles"], batch_size=cfg["batch_size"],
                            total_steps=cfg["total_steps"], lr=cfg["lr"], d_lr_mult=cfg["d_lr_mult"],
                            prior_lr_mult=cfg["prior_lr_mult"], betas=(cfg["beta1"], .999),
                            reg_coeff=cfg["coeff"], reg_kappa=cfg["kappa"], ema_decay=cfg["ema_decay"],
                            lr_anneal_start=cfg["lr_anneal_start"], lr_floor=cfg["lr_floor"])
        original_gate = "lib.sparse_metrics.convergence_bar at the original EMA final endpoint"
        budget = cfg["total_steps"]
        argv = [str(script), "--device", "cpu", "--out_dir", str(output / "full/original-output")]
        if config:
            argv.extend(["--config", str(config)])
    else:
        cfg = {**module["DEFAULTS"], **module["read_config"](str(config))}
        recipe = module["training_recipe"](cfg)
        original_gate = "No single final scientific acceptance gate is declared in the source"
        budget = cfg["steps"]
        argv = [str(script), "--config", str(config)]
    protocol = dict(catalog_id=identifier, label=entry["label"], family=entry["family"], source_revision=REVISION,
                    script=str(script.relative_to(snapshot)), script_sha256=digest(script),
                    config=entry["config"], config_sha256=digest(config) if config else None,
                    resolved_config=cfg, resolved_recipe=recipe.to_dict(), exact_original_cli_argv=argv,
                    original_update_budget=budget, original_gate=original_gate,
                    wall_cap_seconds=120, runtime="CPU, one Torch/OpenMP/MKL/BLAS thread, CUDA hidden",
                    changed_scientific_factors=[], output_relocation_only=entry["family"] == "sparse",
                    added_gate_version=scoring.VERSION, added_gate_thresholds=scoring.GATES,
                    observation=dict(count=scoring.COUNT, evaluation_seed=scoring.EVAL_SEED,
                                     live_ema_separate=True, source_sampling_law_preserved=True),
                    selection="One capped attempt; no seed/profile/budget changes, retries or partial PASS")
    write(output / "protocol.json", protocol)
    return module, cfg, argv, protocol


def worker(args):
    torch.set_num_threads(1)
    started, capture, result, probe = time.monotonic(), None, None, None
    status, error, phase = "COMPLETE", None, "PREPARE"
    def timeout(*_):
        raise WallTimeout("The fixed 120-second conditional-source allowance expired")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 117)
    try:
        module, cfg, argv, protocol = prepare(args.output, args.case)
        if protocol["family"] == "denoising":
            phase = "ORIGINAL_CLI_PREREQUISITE"
            # The exact original CLI owns device refusal; never patch CUDA or port it.
            saved_argv = sys.argv
            try:
                sys.argv = argv
                runpy.run_path(str(args.output / "source" / protocol["script"]), run_name="__main__")
            finally:
                sys.argv = saved_argv
        else:
            phase = "EXACT_PREFIX_PARITY"
            probe = parity(module, cfg, args.output / "parity")
            phase = "FULL_ORIGINAL_TRAINING"
            capture, result = run_sparse(module, cfg, args.output / "full", observe=True)
    except WallTimeout:
        status, error = "TIMEOUT", traceback.format_exc()
    except BaseException:
        status, error = ("BLOCKED" if phase != "FULL_ORIGINAL_TRAINING" else "ERROR"), traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    # Interrupted capture objects remain available through the progress/raw receipts.
    protocol = json.loads((args.output / "protocol.json").read_text()) if (args.output / "protocol.json").exists() else None
    summary = dict(catalog_id=args.case, label=CASES[args.case]["label"], status=status, phase=phase,
                   error=error, seconds=time.monotonic()-started, original_summary=result,
                   parity=probe, protocol_sha256=digest(args.output / "protocol.json") if protocol else None)
    if capture is not None:
        summary.update(completed_update_pairs=capture.completed, observations=len(capture.rows),
                       final=capture.rows[-1] if capture.rows else None)
    else:
        progress = read_progress(args.output / "full/raw/progress.json")
        summary.update(completed_update_pairs=progress.get("completed_update_pairs", 0),
                       observations=progress.get("observations", 0))
    summary["measurement_complete"] = bool(status == "COMPLETE" and protocol and
        summary["completed_update_pairs"] == protocol["original_update_budget"] and summary.get("final") and
        summary["final"]["step"] == protocol["original_update_budget"])
    write(args.output / "execution-summary.json", summary)
    print(json.dumps(dict(event="CONDITIONAL_SOURCE_END", **summary)), flush=True)
    return 0 if status == "COMPLETE" else 1


def export(output):
    import io
    import tarfile
    repo = Path(__file__).resolve().parents[2]
    data = subprocess.check_output(["git", "archive", REVISION, "particlegan", "experiments", "lib", "configs"], cwd=repo)
    snapshot = output / "source"
    snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(snapshot, filter="data")
    names = [path for path in snapshot.rglob("*") if path.is_file() and path.suffix in (".py", ".toml", ".yaml")]
    return {str(path.relative_to(snapshot)): digest(path) for path in sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = export(args.output)
    for filename, source in (("observer-source.py", __file__), ("observer-utils.py", Path(__file__).with_name("source_demos.py")),
                             ("scoring.py", scoring.__file__)):
        (args.output / filename).write_bytes(Path(source).read_bytes())
    write(args.output / "source-receipt.json", dict(source_revision=REVISION, source_sha256=sources,
          runner_sha256=digest(__file__), helper_sha256=digest(args.output / "observer-utils.py"),
          evaluator_sha256=digest(args.output / "scoring.py"), python=sys.version, torch=torch.__version__, numpy=np.__version__))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/toy-conditional-source-mpl")
    command = [sys.executable, str(args.output / "observer-source.py"), "--worker", "--case", args.case,
               "--output", str(args.output)]
    launched, next_progress = time.monotonic(), 0
    with (args.output / "execution.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                   cwd=str(args.output / "source"))
        while process.poll() is None:
            elapsed = time.monotonic() - launched
            if elapsed >= 120:
                process.kill(); process.wait()
                write(args.output / "hard-timeout.json", dict(wall_cap_seconds=120, killed_pid=process.pid))
                break
            if elapsed >= next_progress:
                progress = read_progress(args.output / "full/raw/progress.json")
                print(json.dumps(dict(case=args.case, elapsed_seconds=elapsed, **progress)), flush=True)
                next_progress = elapsed + 10
            time.sleep(.5)
    write(args.output / "launcher.json", dict(pid=process.pid, supervisor_pid=os.getpid(),
          returncode=process.returncode, command=command, elapsed_seconds=time.monotonic()-launched,
          child_ownership="Only this Popen child PID is signaled", hard_wall_cap_seconds=120))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
