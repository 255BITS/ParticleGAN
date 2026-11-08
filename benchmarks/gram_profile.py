"""Bounded synthetic numerical-kernel benchmark, never training/Sana evidence.

Caller MUST enforce a wall timeout and host/device/output memory caps, including
CPU64 oracle and reporting. This module does not admit a CUDA job. Example CPU:
timeout 60 python -m benchmarks.gram_profile --device cpu --size 32 --output out.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from particlegan.init import deterministic_orthogonal_
from particlegan.optim import GramGuardError, NormalizedOptimizer


PROFILE = "gram_fp64_guarded_v1"


def _hash(value):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def _identity(values):
    return [dict(shape=list(v.shape), dtype=str(v.dtype), sha256=_hash(v)) for v in values]


def _settings(device):
    result = dict(default_dtype=str(torch.get_default_dtype()), default_device=str(torch.get_default_device()),
        threads=torch.get_num_threads(), matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(), cpu_autocast=torch.is_autocast_enabled("cpu"))
    if device.type == "cuda":
        result.update(tf32=torch.backends.cuda.matmul.allow_tf32, cuda_autocast=torch.is_autocast_enabled("cuda"))
    return result


def _fixtures(size, matrices, biases, case):
    # Meta construction consumes no caller CPU/CUDA random stream. A fixed
    # nonconstant scratch pattern lets the PUBLIC initializer own the weights.
    layer = torch.nn.Linear(size, size, bias=biases, dtype=torch.float32, device="meta").to_empty(device="cpu")
    with torch.no_grad():
        layer.weight.copy_(torch.arange(size * size, dtype=torch.float32, device="cpu").reshape(size, size).sin())
        if biases:
            layer.bias.zero_()
    deterministic_orthogonal_(layer, seed=0)
    if case == "full_rank":
        gradient = layer.weight.detach() @ torch.diag(torch.linspace(1., .1, size, dtype=torch.float32, device="cpu"))
    else:
        singular = torch.linspace(1., .1, size, dtype=torch.float32, device="cpu")
        singular[-1] = 0. if case == "rank_deficient" else size * torch.finfo(torch.float32).eps
        gradient = torch.diag(singular)
    weights, gradients = [], []
    for _ in range(matrices):
        weights.append(layer.weight.detach().clone()); gradients.append(gradient.clone())
        if biases:
            weights.append(layer.bias.detach().clone())
            gradients.append(torch.linspace(-.3, .2, size, dtype=torch.float32, device="cpu"))
    return weights, gradients


def _oracle(weights, gradients):
    expected = []
    for old, gradient in zip(weights, gradients):
        g = gradient.double()
        if g.ndim == 2:
            u, s, vh = torch.linalg.svd(g, full_matrices=False)
            keep = s > max(g.shape) * torch.finfo(torch.float32).eps * s[0]
            direction = (u * (keep * s / torch.hypot(s, s.new_tensor(.001)))) @ vh
        else:
            direction = g / torch.hypot(g.norm(), g.new_tensor(.001))
        new = old.clone().add_(direction.float(), alpha=-8e-5)
        expected.append(new - old)  # Oracle also rounds the actual FP32 write/subtract.
    return expected


def _errors(actual, expected):
    squared_error = squared_reference = maximum = 0.
    for a, b in zip(actual, expected):
        error = a.double() - b.double()
        squared_error += float(error.square().sum()); squared_reference += float(b.double().square().sum())
        maximum = max(maximum, float(error.abs().max()))
    return dict(frobenius_absolute=squared_error ** .5, maximum_absolute=maximum,
        relative_frobenius=(squared_error / squared_reference) ** .5 if squared_reference else None)


def run_benchmark(*, device="cpu", size=32, matrices=2, repeats=1, warmup=0, biases=False, case="full_rank"):
    """Fresh identical public steps; PASS means execution, not native parity."""
    for name, value, low, high in (("size", size, 2, 1152), ("matrices", matrices, 1, 112),
                                  ("repeats", repeats, 1, 3), ("warmup", warmup, 0, 1)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if type(biases) is not bool or case not in ("full_rank", "rank_deficient", "near_cutoff"):
        raise ValueError("explicit bool biases and known numerical case required")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and device.index is None:
        raise ValueError("device must be cpu or explicitly indexed cuda:N")
    started = time.monotonic(); before = _settings(device); cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    tick = time.monotonic(); weights, gradients = _fixtures(size, matrices, biases, case)
    inputs = dict(weights=_identity(weights), gradients=_identity(gradients))
    fixture_seconds = time.monotonic() - tick
    tick = time.monotonic(); expected = _oracle(weights, gradients); oracle_seconds = time.monotonic() - tick
    rows = []
    for profile in ("default", PROFILE):
        for iteration in range(-warmup, repeats):
            tick = time.monotonic()
            originals = [w.to(device=device, copy=True) for w in weights]
            parameters = [torch.nn.Parameter(w.clone()) for w in originals]
            opt = NormalizedOptimizer(parameters, family="dualnorm", momentum=0., smoothing=.001,
                lr=8e-5, svd_backend="native", dense_profile=profile)
            for p, gradient in zip(parameters, gradients):
                p.grad = gradient.to(device=device, copy=True)
            sync(); preparation = time.monotonic() - tick
            begin = end = None
            if device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(device))
            error = None; tick = time.monotonic()
            try:
                opt.step()
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc), numerical_guard=isinstance(exc, GramGuardError))
            if end is not None:
                end.record(torch.cuda.current_stream(device))
            sync(); elapsed = time.monotonic() - tick
            tick = time.monotonic()
            actual = [(p.detach() - old).cpu() for p, old in zip(parameters, originals)]
            finite = all(bool(torch.isfinite(value).all()) for value in actual)
            if not finite and error is None:
                error = dict(type="NonfiniteUpdate", message="nonfinite actual rounded displacement", numerical_guard=False)
            row = dict(profile=profile, repeat=iteration, warmup=iteration < 0,
                status="HOLD" if error else "PASS", exception=error, input_identity=inputs,
                complete_optimizer_step_wall_seconds=elapsed,
                complete_optimizer_step_cuda_ms=begin.elapsed_time(end) if begin is not None else None,
                preparation_seconds=preparation, actual_FP32_displacements=_identity(actual),
                CPU64_oracle_error=_errors(actual, expected) if finite else None,
                clocks=[opt.state[p].get("step", 0) for p in parameters], partial_state_discard_required=error is not None)
            row["untimed_reporting_seconds"] = time.monotonic() - tick; rows.append(row)
            del opt, parameters, originals, actual
            if error:
                break  # No hidden retries, fallback, or continuation after partial mutation.
    unchanged = before == _settings(device) and torch.equal(cpu_rng, torch.get_rng_state())
    unchanged &= cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    unchanged &= inputs == dict(weights=_identity(weights), gradients=_identity(gradients))
    return dict(schema="public-gram-profile-synthetic-benchmark-v1", status="PASS" if unchanged and all(r["status"] == "PASS" for r in rows) else "HOLD",
        case=case, device=str(device), size=size, matrices=matrices, biases=biases, repeats=repeats, warmup=warmup,
        seed=0, input_identity=inputs, records=rows, runtime_settings=before, inputs_RNG_settings_unchanged=bool(unchanged),
        fixture_seconds=fixture_seconds, untimed_CPU64_oracle_seconds=oracle_seconds, total_wall_seconds=time.monotonic()-started,
        source_sha256={str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, __import__("particlegan.init", fromlist=["x"]).__file__, __import__("particlegan.optim.dualnorm", fromlist=["x"]).__file__, __import__("particlegan.optim.gram_polar", fromlist=["x"]).__file__)},
        numerical_acceptance="Measured errors only; PASS is completed execution, not default FP32 parity or native qualification",
        fixture_scope="seed0 public initializer; explicit synthetic repeated gradient bank; not actual Sana/CP/capsule/training",
        timing_scope="complete public optimizer.step including selected guard; copies, constructors, oracle and hashes excluded and separately reported",
        caller_must_enforce="whole-worker wall timeout and host/device/output memory caps; CUDA requires separate admission")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True); parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--matrices", type=int, default=2); parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0); parser.add_argument("--biases", action="store_true")
    parser.add_argument("--case", choices=("full_rank", "rank_deficient", "near_cutoff"), default="full_rank")
    parser.add_argument("--output", type=Path, required=True); args = vars(parser.parse_args(argv)); output = args.pop("output")
    try:
        result = run_benchmark(**args)
    except ValueError as exc:
        parser.error(str(exc))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(output))))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
