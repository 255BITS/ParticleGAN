"""Audit the exact routed examples and continuous ring protocol on CPU.

This observer never repairs a source fixture or selects training parameters.
Raw states, traces and checkpoints go to --artifacts. Only compact receipts
and real captured-state GIFs go to --output. The stronger quality evaluator is
a separate source identity supplied with --quality-module when necessary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VERSION = "source-family-training-v1"
FIXTURES = {
    "paired": dict(catalog_id="source-family-14", source="examples/e22_routed_paired.py",
                   updates=160, cadence=20, wall_cap=180),
    "support": dict(catalog_id="source-family-14", source="examples/e22_routed_support.py",
                    updates=1200, cadence=100, wall_cap=180),
    "moving": dict(catalog_id="source-family-14", source="examples/e22_routed_moving.py",
                   updates=1500, cadence=50, wall_cap=180),
    "replay": dict(catalog_id="source-family-14", source="examples/e22_routed_replay.py",
                   updates=8, cadence=1, wall_cap=180),
    "ring": dict(catalog_id="source-family-10", source="benchmarks/toy100/continuous_probe.py",
                 updates=1200, cadence=50, wall_cap=120),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_digest(value):
    """Stable hash including BF16 tensors, optimizer state and explicit RNGs."""
    digest = hashlib.sha256()
    def feed(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(str((str(array.dtype), tuple(array.shape))).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda x: repr(x)):
                feed(key)
                feed(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for sub in item:
                feed(sub)
        else:
            digest.update(repr((type(item).__name__, item)).encode())
    feed(value)
    return digest.hexdigest()


class WallLimit(RuntimeError):
    pass


@contextmanager
def wall_cap(seconds):
    def expired(signum, frame):
        raise WallLimit(f"declared audit wall cap {seconds}s reached")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def runtime():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                torch_git_revision=torch.version.git_version, torch_build=torch.__config__.show(),
                numpy=np.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability(),
                machine=platform.machine(), threads=torch.get_num_threads(), device="cpu")


def source_binding(quality_path):
    paths = [*sorted((ROOT / "particlegan").rglob("*.py")),
             *(ROOT / row["source"] for row in FIXTURES.values()),
             ROOT / "examples/e22_routed_sites.py", ROOT / "examples/e22_routed_game.py",
             ROOT / "benchmarks/toy100/warm_equilibrium_probe.py",
             Path(__file__)]
    return dict(source_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
                quality_evaluator=dict(version=VERSION, path=str(quality_path), sha256=sha(quality_path)),
                runtime=runtime())


def suffix(flags):
    n = 0
    for okay in reversed(flags):
        if not okay:
            break
        n += 1
    return dict(passing_checks=sum(flags), checks=len(flags), passing_suffix=n,
                minimum_checks=5, passed=n >= 5)


def routed_modules(name):
    if str(ROOT / "examples") not in sys.path:
        sys.path.insert(0, str(ROOT / "examples"))
    module = load_module(ROOT / FIXTURES[name]["source"], f"audit_routed_{name}")
    if name == "paired":
        return module, module.make_loop, module.update, module.evaluate, module.checkpoint, \
            lambda loop: loop.policy.G.host(loop.test_context.bfloat16()).float()
    if name == "moving":
        paired = module.paired
        return module, paired["make_loop"], paired["update"], paired["evaluate"], paired["checkpoint"], \
            lambda loop: module.host(loop.test_context)
    if name == "replay":
        import e22_routed_sites as sites
        return module, lambda: module.make_loop(initialization="api", penalty_units="token"), \
            lambda loop: module.update(loop, generator_forward=module.checkpointed_generate), \
            module.evaluate, sites.checkpoint, lambda loop: sites.neutral(loop.test_context)
    return module, module.make_loop, module.update, module.evaluate, module.checkpoint, \
        lambda loop: module.neutral(loop.test_context)


class RoutedCapture:
    """Clean deterministic held-out pairs; never a training observation."""
    def __init__(self, evaluate, neutral, quality):
        self.evaluate, self.neutral, self.quality = evaluate, neutral, quality
        self.frames, self.metrics = [], []

    @torch.no_grad()
    def __call__(self, loop, *, event="check", angle=0.):
        # served_model is an independent frozen snapshot; the fork also protects
        # constructor RNG against future source implementations.
        with torch.random.fork_rng(devices=[]):
            served = loop.policy.served_model()
            prediction = served.routed_forward(loop.test_context, perturb=False, output_noise=False)
            neutral = self.neutral(loop)
            frame = dict(step=loop.policy.completed_steps,
                         prediction=prediction.detach().float().cpu().numpy().copy(),
                         target=loop.test_targets.detach().float().cpu().numpy().copy(),
                         neutral=neutral.detach().float().cpu().numpy().copy())
            metric = dict(step=frame["step"], event=event, angle=angle,
                          original=self.evaluate(loop),
                          correspondence=self.quality.paired_edit_metrics(
                              frame["prediction"], frame["target"], frame["neutral"]))
        self.frames.append(frame)
        self.metrics.append(metric)
        return metric


def routed_state(loop, checkpoint):
    p = loop.policy
    modules = [*p._training_modules().values(), *p._average_modules().values(), p.opt_d.ema_critic]
    return dict(checkpoint=checkpoint(loop), modes=[[m.training for m in root.modules()] for root in modules],
                gradients=[None if param.grad is None else param.grad.detach().clone()
                           for root in modules for param in root.parameters()],
                table_gradient=None if p.table.grad is None else p.table.grad.detach().clone())


def routed_parity(make_loop, update, evaluate, checkpoint, neutral, quality, *, steps=2):
    """Observer-on/off equality covers parameters, averages, Adam, modes and RNG."""
    initial_rng = torch.get_rng_state().clone()
    runs = []
    for observed in (False, True):
        torch.set_rng_state(initial_rng)
        loop = make_loop()
        capture = RoutedCapture(evaluate, neutral, quality)
        if observed:
            capture(loop)
        trace = []
        for _ in range(steps):
            with torch.autograd.set_multithreading_enabled(False):
                trace.append(update(loop))
            if observed:
                capture(loop)
        runs.append(dict(state=state_digest(routed_state(loop, checkpoint)), trace=state_digest(trace)))
    torch.set_rng_state(initial_rng)
    matched = runs[0] == runs[1]
    if not matched:
        raise RuntimeError(f"observer altered training state or update trace: {runs}")
    return dict(updates=steps, state_and_trace_exact=True, control=runs[0], observed=runs[1],
                scope="Short observer protocol control; not a scientific convergence gate")


def save_routed_capture(capture, path):
    if not capture.frames:
        return None
    np.savez_compressed(path, steps=np.array([r["step"] for r in capture.frames]),
                        **{key: np.stack([r[key] for r in capture.frames])
                           for key in ("prediction", "target", "neutral")})
    return dict(path=str(path), sha256=sha(path), captured_frames=len(capture.frames), interpolation=False)


def run_routed(name, quality, artifacts):
    spec = FIXTURES[name]
    artifacts.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    loop, capture, parity = None, None, None
    report = dict(fixture=name, **spec, status="UNRUN", historical_binary_gate="UNDECLARED",
                  original_evaluation="heldout_rmse and max error, without an absolute trained pass threshold",
                  sampling_policy="All 180 held-out contexts; clean deterministic selected served routed bank; no DV12 or output noise",
                  heldout_not_a_training_signal=True, optimizer_parameters_changed=False,
                  scientific_claim="paired context-to-edit correspondence")
    try:
        with wall_cap(spec["wall_cap"]):
            module, make_loop, update, evaluate, checkpoint, neutral = routed_modules(name)
            parity = routed_parity(make_loop, update, evaluate, checkpoint, neutral, quality)
            loop = make_loop()
            capture = RoutedCapture(evaluate, neutral, quality)
            report["initial_state_sha256"] = state_digest(checkpoint(loop))
            report["recipe"] = loop.policy.recipe.to_dict()
            report["config"] = getattr(loop, "config", dict(batch_size=32, particles=16, z_dim=2))
            report["data"] = {key: dict(shape=list(getattr(loop, key).shape),
                                        sha256=state_digest(getattr(loop, key)))
                              for key in ("fit_context", "fit_targets", "guard_context", "guard_targets", "test_context", "test_targets")}
            report["context_population"] = len(loop.test_context)
            capture(loop, event="initial")
            trace_path = artifacts / "updates.jsonl"
            trace = hashlib.sha256()
            with trace_path.open("w") as out:
                moving = module.MovingTarget(loop) if name == "moving" else None
                for index in range(spec["updates"]):
                    if moving is not None and index and index % 500 == 0:
                        moving.turn_to(math.radians(30. * (index // 500)))
                        capture(loop, event="target_turn", angle=30. * (index // 500))
                    with torch.autograd.set_multithreading_enabled(False):
                        row = update(loop)
                    encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    out.write(encoded)
                    out.flush()
                    trace.update(encoded.encode())
                    if (index + 1) % spec["cadence"] == 0 or index == spec["updates"] - 1:
                        metric = capture(loop, angle=30. * (index // 500) if moving else 0.)
                        print(json.dumps({"event": "check", "fixture": name, **metric}), flush=True)
            report["trace_sha256"] = trace.hexdigest()
            report["status"] = "PASS" if capture.metrics[-1]["correspondence"]["passed"] else "FAIL"
            if name == "replay":
                report["protocol_status"] = "PASS"
                report["scientific_claim"] = "activation recomputation with private DV12 replay; eight updates are a software smoke, not full convergence"
                report["convergence_qualified"] = False
            if name == "moving":
                report["config"].update(turn_every=500, turns=2, degrees=30., r1=False)
    except Exception as error:
        report.update(status="CAPPED" if isinstance(error, WallLimit) else "BLOCKED",
                      error=repr(error), traceback_artifact=str(artifacts / "error.txt"))
        (artifacts / "error.txt").write_text(traceback.format_exc())
        print(json.dumps({"event": "error", "fixture": name, "error": repr(error)}), flush=True)
    report["seconds"] = time.perf_counter() - started
    report["observer_control"] = parity
    report["completed_updates"] = None if loop is None else loop.policy.completed_steps
    if loop is not None:
        # State dumps are deliberately outside Git, even for a partial run.
        try:
            saved = checkpoint(loop)
            torch.save(saved, artifacts / "checkpoint.pt")
            report["final_state_sha256"] = state_digest(saved)
            report["checkpoint"] = dict(path=str(artifacts / "checkpoint.pt"), sha256=sha(artifacts / "checkpoint.pt"))
        except Exception as error:
            report["checkpoint_error"] = repr(error)
        report["row_diagnostics"] = module.row_diagnostics(loop.policy) if name == "paired" else \
            module.paired["row_diagnostics"](loop.policy) if name == "moving" else module.diagnostics(loop.policy)
    if capture is not None and capture.frames:
        report["observations"] = save_routed_capture(capture, artifacts / "observations.npz")
        write(artifacts / "captured-metrics.json", capture.metrics)
        report["initial"] = capture.metrics[0]
        report["best"] = min(capture.metrics, key=lambda row: row["correspondence"]["relative_mse"])
        report["final"] = capture.metrics[-1]
        checks = [r for r in capture.metrics if r["event"] == "check"]
        if name == "moving":
            checks = [r for r in checks if r["step"] > 1000]
        report["stronger_gate_window"] = suffix([r["correspondence"]["passed"] for r in checks])
    write(artifacts / "receipt.json", report)
    return report


def ring_state(values):
    """Training-affecting state only, including policy streams and Adam moments."""
    from benchmarks.toy100.warm_equilibrium_probe import training_state_sha256
    return training_state_sha256(values)


def run_ring(quality, artifacts, *, baseline_receipt=None):
    from benchmarks.toy100 import continuous_probe as probe
    from benchmarks.locked_shared import mode_hold
    from benchmarks.toy100.device import apply_device_policy
    from benchmarks.init_research.init_registry import use_init
    spec = FIXTURES["ring"]
    artifacts.mkdir(parents=True, exist_ok=False)
    config = json.loads(probe.DEFAULT_CONFIG.read_text())
    apply_device_policy("cpu")
    use_init(None)
    captures, metrics, parity, final_state = [], [], [], None
    original = mode_hold.checkpoint
    policy_holder = {}
    original_noise_policy = probe._noise_policy

    def noise_policy(*args, **kwargs):
        value = original_noise_policy(*args, **kwargs)
        policy_holder["policy"] = value
        return value

    def observe(step, measure):
        original(step, measure)
        if step % 50:
            return
        frame = inspect.currentframe().f_back
        # _run_extended.observe delegates here; measure closes over the real
        # host. No source replacement or training-forward interception occurs.
        host = frame.f_back
        if host.f_code.co_name != "train_mode_hold":
            raise RuntimeError("ring observer lost the declared frozen host frame")
        values = host.f_locals
        policy = policy_holder["policy"]
        state = {**values, "noise_policy": policy}
        before = ring_state(state)
        with torch.no_grad(), policy.evaluation(step):
            latent, _ = values["prior"].sample(mode_hold.EVAL_N, generator=torch.Generator().manual_seed(values["seed"] + 9))
            samples = values["generator"](latent).detach().cpu().numpy().copy()
        after = ring_state(state)
        parity.append(before == after)
        if before != after:
            raise RuntimeError("ring evaluator changed training-affecting state")
        captures.append(samples)
        metric = dict(step=step, original=mode_hold.diversity(torch.from_numpy(samples), values["means"], detailed=True),
                      gaussian_law=quality.ring_metrics(samples))
        metrics.append(metric)
        if step == spec["updates"]:
            final_state = dict(generator=deepcopy(values["generator"].state_dict()),
                               critic=deepcopy(values["critic"].state_dict()),
                               prior=deepcopy(values["prior"].state_dict()),
                               opt_g=deepcopy(values["opt_g"].state_dict()),
                               opt_d=deepcopy(values["opt_d"].state_dict()),
                               ema_g=deepcopy(values["ema_g"]), ema_z=values["ema_z"].clone(),
                               stream=values["stream"].get_state(), cpu_rng=torch.get_rng_state(),
                               input_stream=policy.input_stream.get_state())
            torch.save(final_state, artifacts / "checkpoint.pt")

    report = dict(fixture="ring", **spec, mode="constant", noise_horizon=1200,
                  input_config=dict(path=str(probe.DEFAULT_CONFIG.relative_to(ROOT)), sha256=sha(probe.DEFAULT_CONFIG)),
                  initialization="source PyTorch initialization, seed0 host; --init omitted",
                  sampling_policy="live host 4096 draws; prior indices fixed seed9; declared NoisePolicy.evaluation(step) output-noisy law",
                  heldout_not_a_training_signal=True, optimizer_parameters_changed=False)
    started = time.perf_counter()
    try:
        with wall_cap(spec["wall_cap"]), patch.object(mode_hold, "checkpoint", observe), \
             patch.object(probe, "_noise_policy", noise_policy):
            result = probe.run_probe(config, mode="constant", steps=1200, diagnostic_every=50,
                                     log=lambda row: print(json.dumps(row), flush=True))
        write(artifacts / "original-result.json", result)
        report.update(status=result["status"], completed_updates=result["steps"],
                      source_recipe=result["source_recipe"], effective_config=result["effective_config"],
                      config_sha256=result["config_sha256"], source_sha256=result["source_sha256"],
                      stationary=result["stationary"], original_terminal_grade=result["terminal_grade"],
                      original_convergence=result["convergence"], final=result["final"], ema=result["ema"],
                      optimizer_final=result["optimizer_final"], noise=result["noise"],
                      rate_ranges=result["rate_ranges"], acquisition_passed=result["status"] == "PASS")
        if baseline_receipt is not None:
            baseline = json.loads(Path(baseline_receipt).read_text())
            keys = ("final", "ema", "diagnostic", "stationary", "source_sha256", "config_sha256", "noise", "optimizer_final", "rate_ranges")
            matched = all(result[key] == baseline[key] for key in keys)
            report["full_metric_observer_control"] = dict(path=str(baseline_receipt), sha256=sha(baseline_receipt),
                                                         compared_keys=list(keys), exact=matched)
            if not matched:
                raise RuntimeError("observed ring result differs from direct source default control")
    except Exception as error:
        report.update(status="CAPPED" if isinstance(error, WallLimit) else "BLOCKED", error=repr(error))
        (artifacts / "error.txt").write_text(traceback.format_exc())
    report["seconds"] = time.perf_counter() - started
    report["observer_state_parity"] = dict(checks=len(parity), state_preserved=bool(parity) and all(parity))
    if captures:
        np.savez_compressed(artifacts / "observations.npz", live=np.stack(captures), steps=np.array([r["step"] for r in metrics]))
        write(artifacts / "captured-metrics.json", metrics)
        report["observations"] = dict(path=str(artifacts / "observations.npz"), sha256=sha(artifacts / "observations.npz"),
                                      captured_frames=len(captures), interpolation=False)
        report["final_gaussian_law"] = metrics[-1]["gaussian_law"]
        report["stronger_gate_window"] = suffix([r["gaussian_law"]["passed"] for r in metrics])
    report["phases"] = {
        "constant_cold_acquisition": report["status"],
        "uninterrupted_hold_2400": "NOT_RUN_FAILED_ACQUISITION" if not report.get("acquisition_passed") else "NOT_RUN",
        "shift_3600_at_2400_and_frozen_control": "NOT_RUN_FAILED_ACQUISITION" if not report.get("acquisition_passed") else "NOT_RUN",
        "warm_checkpoint_1000": "SOURCE_ONLY: warm_equilibrium_probe.run_warm_variants defaults to a scheduled prefix, Linux fork, identity/cold state parity and 200 per-update checks; source helper does not enforce an all-eight passing prerequisite. No warm variant executed in this constant-acquisition audit.",
    }
    if (artifacts / "checkpoint.pt").exists():
        report["checkpoint"] = dict(path=str(artifacts / "checkpoint.pt"), sha256=sha(artifacts / "checkpoint.pt"))
    write(artifacts / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=FIXTURES)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-module", type=Path, default=ROOT / "benchmarks/toy_audit/definition_quality.py")
    parser.add_argument("--ring-control", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    quality = load_module(args.quality_module, "audit_definition_quality")
    binding = source_binding(args.quality_module)
    report = run_ring(quality, args.artifacts, baseline_receipt=args.ring_control) if args.fixture == "ring" else \
        run_routed(args.fixture, quality, args.artifacts)
    report["binding"] = binding
    if binding != source_binding(args.quality_module):
        raise RuntimeError("executed source or runtime changed during this audit")
    write(args.output, report)
    print(json.dumps({"event": "complete", "fixture": args.fixture, "status": report["status"],
                      "updates": report.get("completed_updates"), "receipt": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
