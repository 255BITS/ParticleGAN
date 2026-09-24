"""Synthetic matched-law intervention at the frozen mode-hold warm state.

At update 1000, replace the target by the *current live* 12-generator-output
Gaussian mixture with the same .029 output standard deviation, and zero the
critic's final affine layer. Fork four identical states: ordinary constant
Adam and the existing functional-metric G step, each with an independent
matched no-update control after a +.35 target translation at update 1200.
This changes the target and critic state deliberately; no score here is a
production mode-hold, shared-gate or cold-acquisition result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.continuous_probe import (
    _run_extended, prepared_config, run_probe,
)
from benchmarks.toy100.warm_equilibrium_probe import (
    constant_rate_context, training_state_sha256,
)
from benchmarks.transfer_suite.protocol import required_tasks
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100.functional_metric_scratch import functional_metric


class _Forked(Exception):
    pass


def gaussian_mixture_mmd2(target: torch.Tensor, generated: torch.Tensor,
                          *, sigma: float = .029, bandwidth: float = .21) -> float:
    """Exact RBF MMD² between equally weighted isotropic Gaussian mixtures.

    The same covariance is used on both sides. For d=2, convolution of the
    RBF with X-Y ~ N(center_delta, 2 sigma² I) multiplies each center kernel
    by h²/(h²+2 sigma²) and changes its squared-distance denominator to
    2(h²+2 sigma²).
    """
    if (target.ndim != 2 or target.shape != generated.shape
            or target.shape[1] != 2 or len(target) < 2):
        raise ValueError("matching two-dimensional center tables required")
    if sigma <= 0 or bandwidth <= 0:
        raise ValueError("positive Gaussian scales required")
    x, y = target.double(), generated.double()
    scale = bandwidth**2 + 2 * sigma**2
    prefactor = bandwidth**2 / scale
    kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square() / (2 * scale)).mean()
    value = prefactor * (kernel(x, x) + kernel(y, y) - 2 * kernel(x, y))
    return max(0.0, float(value))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def _write_gzip(path: Path, value) -> None:
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True,
                                              allow_nan=False).encode(), mtime=0))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _moment_steps(state: dict) -> dict:
    result = {}
    for role in ("g", "d"):
        optimizer = state[f"opt_{role}"]
        counts = [int(row["step"].item() if isinstance(row["step"], torch.Tensor)
                      else row["step"])
                  for row in optimizer.state.values() if "step" in row]
        if not counts or min(counts) != max(counts):
            raise RuntimeError(f"{role} Adam moment counters disagree")
        result[role] = counts[0]
    return result


def _diagnostic(state: dict, target: torch.Tensor, step: int, label: str) -> dict:
    with torch.no_grad():
        clean = state["generator"].model(state["prior"].z).detach()
        mmd2 = gaussian_mixture_mmd2(target, clean)
        indexed_rms = float((target - clean).square().sum(1).mean().sqrt())
        centroid_error = float((target.mean(0) - clean.mean(0)).norm())
    return dict(step=step, label=label, mixture_mmd2=mmd2,
                indexed_center_rms=indexed_rms,
                centroid_error=centroid_error)


def _check_rates(context: dict, *, start: int, end: int) -> dict:
    expected = {"g": .00425, "d": .00425, "prior": .0085}
    rows = context["post_checkpoint_rate_ranges"]
    if set(rows) != set(expected):
        raise RuntimeError("missing an applied constant-rate group")
    for name, rate in expected.items():
        if (rows[name]["min"] != rate or rows[name]["max"] != rate
                or rows[name]["observations"] != end - start):
            raise RuntimeError(f"{name} actual post-prefix rate/call mismatch")
    return rows


def _run_forks(config: dict, output: Path, prefix_reference: str) -> None:
    spec = next(row for row in required_tasks() if row["name"] == "mode_hold")
    spec = {**spec, "steps": 1600}
    recipe, noise, _ = declared_recipe(config)
    if (noise["output_noise_std"] != .029 or noise["input_noise_anneal_end"] != .1):
        raise RuntimeError("expected original fixed-noise law")
    variants = ("constant", "constant_frozen", "metric", "metric_frozen")
    children = {}
    child_name = None
    adapter_stack = ExitStack()
    target_origin = None
    start_hash = None
    intervention = None
    diagnostics = []
    checkpoint_hashes = {}
    checkpoint_moments = {}
    metric_receipt = None

    def at_1000(state):
        nonlocal child_name, target_origin, start_hash, intervention, metric_receipt
        if state["completed_steps"] != 1000:
            raise RuntimeError("matched intervention did not start at 1000")
        start_hash = training_state_sha256(state)
        if start_hash != prefix_reference:
            raise RuntimeError("extended prefix differs from original 1200-step host")
        policy = state["noise_policy"]
        if (policy.output_scale is not None or
                not math.isclose(policy.output_sigma, .029, rel_tol=0, abs_tol=1e-12)
                or policy.input_sigma != 0):
            raise RuntimeError("target and generator observation-noise laws differ")
        with torch.no_grad():
            centers = state["generator"].model(state["prior"].z).detach().clone()
            if centers.shape != (12, 2):
                raise RuntimeError("unexpected learned support")
            target_origin = centers.clone()
            means = state["means"]
            means.resize_(centers.shape).copy_(centers)
            critic = state["critic"].model
            affine = critic.net[-1]
            if not isinstance(affine, nn.Linear) or affine.out_features != 1:
                raise RuntimeError("unexpected critic final affine")
            old_final_norm = float((affine.weight.square().sum() +
                                    affine.bias.square().sum()).sqrt())
            affine.weight.zero_()
            affine.bias.zero_()
            if torch.count_nonzero(affine.weight) or torch.count_nonzero(affine.bias):
                raise RuntimeError("critic final affine did not reset")
        intervention = dict(original_prefix_sha256=start_hash,
                            shared_post_intervention_sha256=training_state_sha256(state),
                            target_centers=target_origin.tolist(), target_sigma=.029,
                            critic_last_affine_before_norm=old_final_norm,
                            critic_last_affine_zero=True,
                            Adam_second_moments_preserved=True,
                            EMA_preserved=True,
                            training_rng_preserved=True)
        old_sample_ring = mode_hold.sample_ring
        old_diversity = mode_hold.diversity
        old_checkpoint = mode_hold.checkpoint
        for name in variants:
            pid = os.fork()
            if pid == 0:
                child_name = name
                # Only the target law changes. Its widths match the frozen
                # generator output noise; all training RNG streams remain live.
                adapter_stack.enter_context(patch.object(
                    mode_hold, "sample_ring",
                    lambda means, n, sigma, generator: old_sample_ring(
                        means, n, .029, generator)))
                adapter_stack.enter_context(patch.object(
                    mode_hold, "diversity",
                    lambda samples, means, *, detailed=False: old_diversity(
                        samples, means, sigma=.029, detailed=detailed)))

                def measure_step(step, measure):
                    if step == 1200:
                        diagnostics.append(_diagnostic(state, state["means"].clone(),
                                                       step, "pre_shift"))
                    old_checkpoint(step, measure)
                    if step % 10 == 0:
                        label = "post_shift" if step == 1200 else "after"
                        diagnostics.append(_diagnostic(state, state["means"].clone(),
                                                       step, label))
                    if step in (1200, 1600):
                        checkpoint_hashes[step] = training_state_sha256(state)
                        checkpoint_moments[step] = _moment_steps(state)

                adapter_stack.enter_context(patch.object(
                    mode_hold, "checkpoint", measure_step))
                adapter_stack.enter_context(constant_rate_context(state, lr=.00425))
                delegate = state["base_adam_step"]
                if name.startswith("metric"):
                    setter = state["set_step_delegate"]
                    holder = {}
                    def capture_setter(fn):
                        holder["fn"] = fn
                        setter(fn)
                    state["set_step_delegate"] = capture_setter
                    metric_receipt = adapter_stack.enter_context(
                        functional_metric(output_step=.029, state=state))
                    delegate = holder["fn"]
                if name.endswith("frozen"):
                    def frozen_delegate(optimizer, *args, **kwargs):
                        if policy._step_calls > 1200:
                            return None
                        return delegate(optimizer, *args, **kwargs)
                    state["set_step_delegate"](frozen_delegate)
                return
            children[name] = pid
        raise _Forked()

    try:
        result, context = _run_extended(
            spec, recipe, noise, config, noise_horizon=1200,
            diagnostic_every=10, dense_after=1000, dense_until=1600,
            shift_step=1200, shift=(.35, 0.0), freeze_after_shift=False,
            log=None, checkpoint_hook_step=1000, checkpoint_hook=at_1000)
    except _Forked:
        errors = {}
        for name, pid in children.items():
            _, status = os.waitpid(pid, 0)
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                errors[name] = status
        if errors:
            raise RuntimeError(f"matched-law child failure: {errors}")
    except BaseException:
        if child_name is not None:
            _write_json(output / f"{child_name}.error.json",
                        dict(error=traceback.format_exc()))
            traceback.print_exc()
            os._exit(1)
        raise
    else:
        if child_name is None or target_origin is None or intervention is None:
            raise RuntimeError("fork returned without an intervention")
        if len(diagnostics) != 61:  # 1010..1600 by tens, plus pre-shift.
            raise RuntimeError("synthetic-target diagnostic cadence incomplete")
        _check_rates(context, start=1000, end=1600)
        expected_moments = 1200 if child_name.endswith("frozen") else 1600
        if checkpoint_moments[1600] != {"d": expected_moments,
                                         "g": expected_moments}:
            raise RuntimeError("actual Adam moment update count differs")
        row = dict(variant=child_name,
                   scope="synthetic_matched_population_diagnostic_only",
                   shared_gate_eligible=False, result=result,
                   intervention=intervention,
                   diagnostics=diagnostics,
                   shift_pair=context["shift_pair"],
                   optimizer_final=context["optimizer_final"],
                   checkpoint_hashes=checkpoint_hashes,
                   checkpoint_moments=checkpoint_moments,
                   applied_rate_ranges=context["post_checkpoint_rate_ranges"],
                   noise=context["noise_receipt"],
                   functional_metric_receipt=metric_receipt)
        _write_gzip(output / f"{child_name}.json.gz", row)
        print(json.dumps(dict(event="CHILD_DONE", variant=child_name,
                              seconds=result["seconds"],
                              hold_last=next(p for p in diagnostics
                                             if p["step"] == 1200 and p["label"] == "pre_shift"),
                              shift_last=diagnostics[-1],
                              optimizer_updates=checkpoint_moments[1600])), flush=True)
        os._exit(0)
    finally:
        adapter_stack.close()


def _read_gzip(path: Path):
    return json.loads(gzip.decompress(path.read_bytes()))


def _assess(row: dict, *, hold_limit: float, response_limit: float) -> dict:
    points = row["diagnostics"]
    hold = [p for p in points if 1000 < p["step"] <= 1200
            and p["label"] != "post_shift"]
    response = [p for p in points if 1200 < p["step"] <= 1600]
    if len(hold) != 20 or len(response) != 40:
        raise RuntimeError("hold or response observations missing")
    return dict(hold_checks=len(hold), hold_below_bound=sum(
        p["mixture_mmd2"] <= hold_limit for p in hold),
        hold_max_mmd2=max(p["mixture_mmd2"] for p in hold),
        hold_last_mmd2=hold[-1]["mixture_mmd2"],
        immediate_shift_mmd2=next(p["mixture_mmd2"] for p in points
                                  if p["step"] == 1200 and p["label"] == "post_shift"),
        response_checks=len(response), response_below_bound=sum(
            p["mixture_mmd2"] <= response_limit for p in response),
        response_last_mmd2=response[-1]["mixture_mmd2"],
        response_last_indexed_rms=response[-1]["indexed_center_rms"],
        response_last_centroid_error=response[-1]["centroid_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    sources = ["reports/toy100/matched_population_diagnostic.py",
               "reports/toy100/functional_metric_scratch.py",
               "benchmarks/toy100/continuous_probe.py",
               "benchmarks/toy100/warm_equilibrium_probe.py",
               "benchmarks/locked_shared/mode_hold.py",
               "benchmarks/transfer_suite/legacy_noise_adapters.py"]
    source_sha = {}
    for name in sources:
        original = ROOT / name
        archived = output / "source_archive" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, archived)
        source_sha[name] = _sha(original)
    bounds = dict(hold_mmd2_max=.01, response_mmd2_max=.01,
                  response_deadline=1600, observations_every=10,
                  analytic_mmd_bandwidth=.21, target_sigma=.029)
    declaration = dict(scope="synthetic_matched_population_diagnostic_only",
                       shared_gate_eligible=False, seed=0,
                       scheduled_prefix_step=1000, hold_steps=200,
                       shift_step=1200, shift=[.35, 0], recovery_steps=400,
                       noise_horizon=1200,
                       variants=["constant", "constant_frozen", "metric", "metric_frozen"],
                       intervention="12 current live clean G(prior.z) centers + N(0,.029^2I) target; zero D final affine; preserve Adam/EMA/RNG",
                       bounds=bounds, source_sha256=source_sha)
    _write_json(output / "declaration.json", declaration)
    print(json.dumps(dict(event="PREDECLARED", sha256=_sha(output / "declaration.json"))), flush=True)
    prefix = {}
    def record_prefix(state):
        prefix["sha256"] = training_state_sha256(state)
    baseline = run_probe(config, mode="scheduled", steps=1200,
                         checkpoint_hook_step=1000, checkpoint_hook=record_prefix)
    if not baseline["stationary"]["pass_all"]:
        raise RuntimeError("scheduled calibration did not pass")
    _write_gzip(output / "scheduled_reference.json.gz",
                dict(evidence=baseline, step1000_sha256=prefix["sha256"]))
    _run_forks(config, output, prefix["sha256"])
    rows = {name: _read_gzip(output / f"{name}.json.gz") for name in declaration["variants"]}
    common = {row["intervention"]["shared_post_intervention_sha256"] for row in rows.values()}
    if len(common) != 1 or any(row["intervention"]["original_prefix_sha256"] != prefix["sha256"]
                               for row in rows.values()):
        raise RuntimeError("four children did not share the exact intervention")
    for active, frozen in (("constant", "constant_frozen"), ("metric", "metric_frozen")):
        a, b = rows[active], rows[frozen]
        if (a["checkpoint_hashes"]["1200"] != b["checkpoint_hashes"]["1200"]
                or a["diagnostics"][:21] != b["diagnostics"][:21]):
            raise RuntimeError(f"{active} and frozen sibling differ before shift")
    assessed = {name: _assess(row, hold_limit=bounds["hold_mmd2_max"],
                               response_limit=bounds["response_mmd2_max"])
                for name, row in rows.items()}
    summary = dict(scope=declaration["scope"], shared_gate_eligible=False,
                   exact_original_prefix=True, exact_intervention_fork=True,
                   matched_pre_shift_frozen_siblings=True,
                   initial_population_mmd2=gaussian_mixture_mmd2(
                       torch.tensor(rows["constant"]["intervention"]["target_centers"]),
                       torch.tensor(rows["constant"]["intervention"]["target_centers"])),
                   bounds=bounds, results=assessed,
                   no_production_gate_credit=True)
    _write_json(output / "summary.json", summary)
    print(json.dumps(dict(event="DONE", **summary)), flush=True)


if __name__ == "__main__":
    main()
