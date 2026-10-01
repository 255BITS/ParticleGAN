"""Observe the original five-word and Gaussian demos without changing training.

Raw checkpoints, clouds, dashboards and execution logs belong in --output,
outside Git. Scientific sources are exported from one immutable develop SHA.
The stronger evaluator is independently pinned; historical demos had no gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import inspect
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
from unittest.mock import patch

import numpy as np
import torch


SOURCE_SHA = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
DRAW_COUNT = 4096
EVALUATION_SEED = 901
VERSION = "source-demo-convergence-v1"
CPU128_PROTOCOL = {
    "version": "source-quickstart-cpu128-v1",
    "claim": "A bounded Gaussian distribution diagnostic using the original public example",
    "changed_factor": {"batch_size": {"original": 2048, "profile": 128}},
    "preserved": "Original architecture, initializer, seed, data law, public trainer and 1000-update budget",
    "factory_override": "particlegan.get_recipe(..., batch_size=128); original source calls and loop stay unchanged",
    "sampling": "4096 actual clean prior draws with isolated evaluation seed901; live and EMA separate",
    "gates": {"mean_error_sigma_max": .10, "covariance_eigenvalues": [.85, 1.15],
              "radial_ks_max": .075, "max_projection_ks_max": .06,
              "minimum_terminal_observations": 5, "full_update_budget": 1000},
    "wall_cap_seconds": 120,
    "selection": "One frozen profile; no retry, threshold tuning or best-checkpoint substitution",
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_progress(path):
    try:
        return json.loads(Path(path).read_text()) if Path(path).exists() else {}
    except (OSError, json.JSONDecodeError):
        return {"progress_read": "pending atomic receipt"}


def evaluator(path=None):
    if path is None:
        return importlib.import_module("benchmarks.toy_audit.definition_quality")
    spec = importlib.util.spec_from_file_location("_source_demo_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def isolated(modules):
    """Preserve caller RNGs and exact module modes around observational reads."""
    modes = [(child, child.training) for module in modules for child in module.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            for module in modules:
                module.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


def word_draw(encoder, generator, prior, canonical, count=DRAW_COUNT):
    """Sample the public prior law; never replace a MoG draw by prior.z."""
    with isolated((encoder, generator, prior)):
        stream = torch.Generator(device=prior.z.device).manual_seed(EVALUATION_SEED)
        latent, _ = prior.sample(count, generator=stream)
        generated = generator(latent).detach().cpu().numpy()
        reconstructed = generator(encoder(canonical)).detach().cpu().numpy()
        encoded = encoder(canonical).detach().cpu().numpy()
    return generated, reconstructed, encoded


def state_hash(value):
    """Stable tensor/RNG comparison for software parity, not training ranking."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            h.update(item.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((str(item.dtype), tuple(item.shape))).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


@contextmanager
def recipe_profile(profile):
    """One explicit test-profile override; no copied trainer or source loop."""
    if profile == "original":
        yield
        return
    if profile != "cpu128":
        raise ValueError("unknown source-demo profile")
    import particlegan
    original = particlegan.get_recipe
    def factory(*args, **kwargs):
        if "batch_size" in kwargs:
            raise ValueError("cpu128 profile requires an original source without a batch override")
        return original(*args, **kwargs, batch_size=128)
    with patch.object(particlegan, "get_recipe", factory):
        yield


class Observer:
    def __init__(self, problem, output, scoring, *, count=DRAW_COUNT):
        self.problem, self.output, self.scoring, self.count = problem, Path(output), scoring, count
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows, self.completed = [], 0
        self.started = time.monotonic()
        self.recipe, self.refs = None, None

    def append(self, step, arrays, live, ema, checkpoint):
        index = len(self.rows)
        np.savez_compressed(self.output / f"cloud-{index:03d}.npz", **arrays)
        torch.save(checkpoint, self.output / f"state-{index:03d}.pt")
        row = dict(step=step, seconds=time.monotonic() - self.started, live=live, ema=ema,
                   cloud_sha256=digest(self.output / f"cloud-{index:03d}.npz"),
                   state_sha256=digest(self.output / f"state-{index:03d}.pt"))
        self.rows.append(row)
        with (self.output / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        write(self.output / "progress.json", dict(completed_update_pairs=self.completed,
              last_scored_step=step, observations=len(self.rows)))
        print(json.dumps(dict(event="SOURCE_DEMO_OBSERVATION", problem=self.problem,
                              completed_updates=self.completed, live=live, ema=ema)), flush=True)

    def five(self, local):
        canonical = local["str_to_tensor"](local["WORDS"]) if "str_to_tensor" in local else None
        # The source defines the conversion at module scope, not train locals.
        if canonical is None:
            chars = self.scoring.CHARS
            tokens = [[chars.index(c) for c in word + "_"] for word in self.scoring.WORDS]
            canonical = torch.nn.functional.one_hot(torch.tensor(tokens), len(chars)).permute(0, 2, 1).float()
        canonical = canonical.to(local["device"])
        arrays, metrics = {}, {}
        for cohort, prefix in (("live", ""), ("ema", "ema_")):
            encoded, generated, prior = local[prefix + "E"], local[prefix + "G"], local[prefix + "prior"]
            logits, recon, positions = word_draw(encoded, generated, prior, canonical, self.count)
            probabilities, reconstruction = self.scoring.word_probabilities(logits), self.scoring.word_probabilities(recon)
            metrics[cohort] = self.scoring.five_word_metrics(probabilities, reconstruction)
            arrays.update({cohort + "_logits": logits, cohort + "_reconstruction_logits": recon,
                           cohort + "_encoded": positions})
        step = local["step"] + 1
        self.recipe = local["recipe"].to_dict()
        checkpoint = dict(modules={name: local[name].state_dict() for name in
                          ("E", "G", "D", "prior", "ema_E", "ema_G", "ema_prior")},
                          optimizers=[opt.state_dict() for opt in local["optimizers"]],
                          torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                          source_step_label=local["step"], completed_update_pairs=step)
        self.append(step, arrays, metrics["live"], metrics["ema"], checkpoint)

    def quick(self, trainer, data_rng):
        self.recipe = trainer.recipe.to_dict()
        arrays, metrics = {}, {}
        for cohort in ("live", "ema"):
            stream = torch.Generator(device=trainer.device).manual_seed(EVALUATION_SEED)
            cloud = trainer.sample(self.count, ema=cohort == "ema", generator=stream, output_noise=False).cpu().numpy()
            arrays[cohort] = cloud
            metrics[cohort] = self.scoring.gaussian_metrics(cloud)
        checkpoint = dict(trainer=trainer.state_dict(), data_rng=data_rng.get_state())
        self.append(trainer.completed_steps, arrays, metrics["live"], metrics["ema"], checkpoint)


@contextmanager
def five_observer(observer, module, *, baseline_refs=None):
    """Read only at the source's existing frame saves, after its EMA update."""
    from matplotlib.figure import Figure
    recipe_type = module["get_recipe"]().__class__
    original_factory, original_save = recipe_type.make_optimizers, Figure.savefig
    handles = []
    def factory(recipe, generator, critic, prior=None, **kwargs):
        result = original_factory(recipe, generator, critic, prior, **kwargs)
        if baseline_refs is not None:
            baseline_refs["optimizers"] = result
            baseline_refs["training_inputs"] = []
            encoder = kwargs.get("encoder")
            def input_read(module, inputs):
                if module is encoder and torch.is_grad_enabled():
                    baseline_refs["training_inputs"].append(state_hash(inputs))
            if encoder is not None:
                handles.append(encoder.register_forward_pre_hook(input_read))
        if observer is not None:
            observer.recipe = recipe.to_dict()
            def counted(*_):
                observer.completed += 1
            handles.append(result[0].register_step_post_hook(counted))
        return result
    def save(figure, *args, **kwargs):
        frame = inspect.currentframe().f_back
        local = dict(frame.f_locals)
        result = original_save(figure, *args, **kwargs)
        if observer is not None and frame.f_code is module["train"].__code__:
            observer.five(local)
        del frame
        return result
    try:
        with patch.object(recipe_type, "make_optimizers", factory), patch.object(Figure, "savefig", save):
            yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def quick_observer(observer, trainer_type):
    original = trainer_type.step
    def step(trainer, *args, **kwargs):
        frame = inspect.currentframe().f_back
        data_rng = frame.f_locals["data_rng"]
        if not observer.rows:
            observer.quick(trainer, data_rng)
        del frame
        result = original(trainer, *args, **kwargs)
        observer.completed = trainer.completed_steps
        write(observer.output / "progress.json", dict(completed_update_pairs=observer.completed,
              last_scored_step=observer.rows[-1]["step"], observations=len(observer.rows)))
        if trainer.completed_steps == 1 or trainer.completed_steps % 20 == 0:
            observer.quick(trainer, data_rng)
        if trainer.completed_steps % 10 == 0:
            print(json.dumps(dict(event="SOURCE_UPDATE", step=trainer.completed_steps)), flush=True)
        return result
    with patch.object(trainer_type, "step", step):
        yield


class WallTimeout(Exception):
    pass


def worker(args):
    # No live checkout/module from the audit is used for scientific training.
    snapshot = args.output / "source"
    sys.path.insert(0, str(snapshot))
    if "particlegan" in sys.modules:
        raise RuntimeError("worker must import the pinned scientific package first")
    from particlegan import GANTrainer
    scoring = evaluator(args.output / "evaluator.py")
    torch.set_num_threads(1)
    observer = Observer(args.problem, args.output / "raw", scoring)
    started, error, status = time.monotonic(), None, "COMPLETE"
    def timeout(*_):
        raise WallTimeout("declared wall-clock allowance expired")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(1, args.wall_seconds - 2))
    try:
        if args.problem == "five_modes":
            module = runpy.run_path(str(snapshot / "examples/five_modes.py"))
            with five_observer(observer, module):
                module["train"](out_dir=str(args.output / "original-dashboard"))
        else:
            source = snapshot / "examples/quickstart_gan.py"
            argv = [str(source), "--output", str(args.output / "original-quickstart.pt")]
            with patch.object(sys, "argv", argv), recipe_profile(args.profile), quick_observer(observer, GANTrainer):
                runpy.run_path(str(source), run_name="__main__")
    except WallTimeout:
        status, error = "TIMEOUT", traceback.format_exc()
    except BaseException:
        status, error = "BLOCKED" if observer.completed == 0 else "ERROR", traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    summary = dict(version=VERSION, problem=args.problem, profile=args.profile, source_sha=SOURCE_SHA, status=status,
                   error=error, seconds=time.monotonic() - started, wall_cap_seconds=args.wall_seconds,
                   declared_budget=20000 if args.problem == "five_modes" else 1000,
                   original_loop_update_count=20001 if args.problem == "five_modes" else 1000,
                   completed_update_pairs=observer.completed, recipe=observer.recipe,
                   sampling=dict(count=DRAW_COUNT, evaluation_seed=EVALUATION_SEED,
                                 law="actual prior.sample with isolated evaluation RNG; clean outputs; separate live/EMA weights"),
                   observations=len(observer.rows), final=observer.rows[-1] if observer.rows else None,
                   live_terminal=terminal(scoring, observer.rows, "live"),
                   ema_terminal=terminal(scoring, observer.rows, "ema"))
    expected = summary["original_loop_update_count"]
    summary["measurement_complete"] = bool(status == "COMPLETE" and observer.completed == expected
        and observer.rows and observer.rows[-1]["step"] == expected)
    summary["live_pass"] = summary["measurement_complete"] and summary["live_terminal"]["passed"]
    summary["ema_pass"] = summary["measurement_complete"] and summary["ema_terminal"]["passed"]
    write(args.output / "summary.json", summary)
    print(json.dumps(dict(event="SOURCE_DEMO_END", problem=args.problem, status=status,
                          completed_update_pairs=observer.completed, observations=len(observer.rows))), flush=True)
    return 0 if status == "COMPLETE" else 1


def terminal(scoring, rows, cohort):
    return scoring.terminal_window([r[cohort]["passed"] for r in rows], minimum=5)


def export_source(output):
    repo = Path(__file__).resolve().parents[2]
    archive = subprocess.check_output(["git", "archive", SOURCE_SHA, "particlegan", "examples"], cwd=repo)
    import io
    import tarfile
    snapshot = output / "source"
    snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(snapshot, filter="data")
    return {str(path.relative_to(snapshot)): digest(path) for path in sorted(snapshot.rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=("five_modes", "quickstart"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, help="Exact review evaluator for a worktree before integration")
    parser.add_argument("--profile", choices=("original", "cpu128"), default="original")
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.profile != "original" and args.problem != "quickstart":
        parser.error("cpu128 is a distinct quickstart diagnostic only")
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = export_source(args.output)
    scoring = args.evaluator or Path(__file__).with_name("definition_quality.py")
    (args.output / "evaluator.py").write_bytes(scoring.read_bytes())
    (args.output / "observer-source.py").write_bytes(Path(__file__).read_bytes())
    source_receipt = dict(version=VERSION, source_sha=SOURCE_SHA, source_sha256=sources,
                          evaluator_sha256=digest(scoring), runner_sha256=digest(__file__),
                          problem=args.problem, profile=args.profile, wall_cap_seconds=args.wall_seconds,
                          python=sys.version, torch=torch.__version__, numpy=np.__version__)
    write(args.output / "source-receipt.json", source_receipt)
    if args.profile == "cpu128":
        sys.path.insert(0, str(args.output / "source"))
        from particlegan import get_recipe
        resolved = get_recipe(total_steps=1000, batch_size=128).to_dict()
        write(args.output / "protocol.json", dict(**CPU128_PROTOCOL, resolved_recipe=resolved,
              source_sha=SOURCE_SHA, evaluator_sha256=digest(scoring),
              source_receipt_sha256=digest(args.output / "source-receipt.json")))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(args.output / "mpl-cache"))
    command = [sys.executable, str(args.output / "observer-source.py"), "--worker", "--problem", args.problem,
               "--profile", args.profile, "--output", str(args.output), "--wall-seconds", str(args.wall_seconds)]
    launched = time.monotonic()
    with (args.output / "execution.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        write(args.output / "process.json", dict(pid=process.pid, supervisor_pid=os.getpid(),
              child_ownership="one Popen PID; no process-group signaling", launched_at=time.time()))
        next_progress = launched
        while process.poll() is None:
            now = time.monotonic()
            if now - launched >= args.wall_seconds:
                process.kill()
                process.wait()
                write(args.output / "timeout.json", dict(status="HARD_TIMEOUT", wall_cap_seconds=args.wall_seconds))
                break
            if now >= next_progress:
                progress = args.output / "raw/progress.json"
                data = read_progress(progress)
                print(json.dumps(dict(event="SOURCE_DEMO_PROGRESS", profile=args.profile,
                                      elapsed_seconds=now - launched, **data)), flush=True)
                next_progress = now + 10
            time.sleep(.5)
        returncode = 124 if (args.output / "timeout.json").exists() else process.returncode
    write(args.output / "launcher.json", dict(returncode=returncode, command=command,
          pid=process.pid, elapsed_seconds=time.monotonic() - launched))
    print(json.dumps(dict(problem=args.problem, returncode=returncode, output=str(args.output))), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
