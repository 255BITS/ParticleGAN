"""Read-only warm replay around three failed sampled-real coverage updates.

Only clean support forward passes and detached copies occur inside training.
Cell centroids and ring-center diagnostics are computed after the final state
has been checked against the frozen original warm run. The candidate is not
modified, and its warm failure remains a failure.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from reports.toy100 import coverage_smoothed_candidate as coverage
from reports.toy100.coverage_pullback import prior_adam_metric


CHECK_STEPS = (1132, 1133, 1134, 1147, 1148, 1149, 1185, 1186, 1187)
FAILED_STEPS = (1133, 1148, 1186)


def _rng_states(local):
    policy = local.get("noise_policy")
    stream = local.get("stream")
    values = [torch.get_rng_state().clone()]
    if stream is not None:
        values.append(stream.get_state().clone())
    if policy is not None:
        values.append(policy.input_stream.get_state().clone())
        if policy.output_stream is not None:
            values.append(policy.output_stream.get_state().clone())
    return values


class DiagnosticRecorder(coverage.CoverageSmoothedRecorder):
    """Observe three clean support states without changing the update rule."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostic_enabled = False
        self.diagnostic_rows = []
        self.counterfactual_states = {}
        self._diagnostic_step = None
        self._pre_gan = None

    def _clean_support(self, local):
        before = _rng_states(local)
        with torch.no_grad():
            generator = local["generator"]
            prior = local["prior"]
            clean = getattr(generator, "model", generator)
            points = clean(prior.z).detach().clone()
        after = _rng_states(local)
        if len(before) != len(after) or any(not torch.equal(a, b)
                                            for a, b in zip(before, after)):
            raise RuntimeError("clean diagnostic forward consumed training RNG")
        return points

    def phases(self, step, opt_d, opt_g, local):
        self._diagnostic_step = step + 1
        if self.diagnostic_enabled and self._diagnostic_step in CHECK_STEPS:
            self._pre_gan = self._clean_support(local)
        yield from super().phases(step, opt_d, opt_g, local)

    def _correct(self, real, opt_g):
        capture = self.diagnostic_enabled and self._diagnostic_step in CHECK_STEPS
        if capture:
            local = self._local
            post_gan = self._clean_support(local)
            real_copy = real.detach().clone()
            if self._diagnostic_step in FAILED_STEPS:
                clean = getattr(local["generator"], "model", local["generator"])
                prior_z = local["prior"].z
                with torch.no_grad():
                    self.counterfactual_states[self._diagnostic_step] = dict(
                        generator_state={name: tensor.detach().clone()
                                         for name, tensor in clean.state_dict().items()},
                        prior_z=prior_z.detach().clone(),
                        prior_adam_metric=prior_adam_metric(opt_g, prior_z).clone(),
                        d_real_batch=real_copy.clone(),
                        post_gan_clean_support=post_gan.clone(),
                    )
        super()._correct(real, opt_g)
        if capture:
            post_projection = self._clean_support(local)
            if self._pre_gan is None:
                raise RuntimeError("missing pre-GAN clean support")
            self.diagnostic_rows.append(dict(
                step=self._diagnostic_step,
                pre_gan=self._pre_gan.tolist(),
                post_gan=post_gan.tolist(),
                post_projection=post_projection.tolist(),
                d_real_batch=real_copy.tolist(),
                correction=self.coverage_records[-1],
            ))


def _without_seconds(value):
    if isinstance(value, dict):
        return {key: _without_seconds(item) for key, item in value.items()
                if "seconds" not in key}
    if isinstance(value, list):
        return [_without_seconds(item) for item in value]
    return value


def _support_grade(points, means):
    distances = torch.cdist(points, means)
    nearest, which = distances.min(dim=1)
    hit = nearest <= .21
    counts = torch.bincount(which[hit], minlength=len(means))
    return dict(clean_hq_fraction=float(hit.float().mean()),
                clean_modes=int((counts > 0).sum()),
                clean_mode_counts=counts.tolist(),
                nearest_mode=which.tolist(),
                nearest_distance=nearest.tolist(),
                clean_hq=hit.tolist())


def _analyze(raw, observed):
    # These known ring centers are used only after the replay and final-hash
    # comparison, never by the candidate or the in-training observer.
    means = mode_hold.ring_means()
    results = []
    for row in raw:
        before = torch.tensor(row["pre_gan"])
        after_gan = torch.tensor(row["post_gan"])
        after_projection = torch.tensor(row["post_projection"])
        real = torch.tensor(row["d_real_batch"])
        assignment = torch.cdist(real, after_gan).argmin(dim=1)
        counts = torch.bincount(assignment, minlength=len(after_gan))
        targets = torch.stack([real[assignment == i].mean(dim=0)
                               if counts[i] else after_gan[i]
                               for i in range(len(after_gan))])
        real_modes = torch.cdist(real, means).argmin(dim=1)
        cell_real_modes = [sorted(set(real_modes[assignment == i].tolist()))
                           for i in range(len(after_gan))]
        before_distance = (after_gan - targets).norm(dim=1)
        after_distance = (after_projection - targets).norm(dim=1)
        actual = observed[row["step"]]
        results.append(dict(
            step=row["step"], failed_warm_check=not (actual["modes"] >= 8
                                                          and actual["hq"] >= .9),
            observed_live=dict(modes=actual["modes"], hq=actual["hq"]),
            pre_gan=dict(points=row["pre_gan"], **_support_grade(before, means)),
            post_gan=dict(points=row["post_gan"], **_support_grade(after_gan, means)),
            ideal_target_cloud=dict(points=targets.tolist(), **_support_grade(targets, means)),
            post_projection=dict(points=row["post_projection"],
                                 **_support_grade(after_projection, means)),
            d_real_batch=row["d_real_batch"],
            nearest_real_cell=dict(assignment=assignment.tolist(),
                                   assigned_counts=counts.tolist(),
                                   centroids=targets.tolist(),
                                   distinct_ring_modes_per_cell=cell_real_modes),
            per_particle=dict(
                gan_clean_displacement=(after_gan-before).norm(dim=1).tolist(),
                projection_clean_displacement=(after_projection-after_gan).norm(dim=1).tolist(),
                centroid_distance_before=before_distance.tolist(),
                centroid_distance_after=after_distance.tolist(),
                centroid_distance_gain=(before_distance-after_distance).tolist()),
            correction=row["correction"],
        ))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "coverage"
        recorder.correction = method == "coverage"
        recorder.diagnostic_enabled = method == "coverage"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + (target - completed - outer),
            moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False,
                       scratch_optimizer_policy=coverage.METHOD if recorder.enabled else "control")
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            receipt.update(recorder.receipt())
            (args.output / "captured-raw.json").write_text(
                json.dumps(recorder.diagnostic_rows, allow_nan=False) + "\n")
            torch.save(recorder.counterfactual_states,
                       args.output / "post-gan-counterfactual-states.pt")

    variants = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                for name in ("identity", "coverage")}
    with patch.object(coverage, "CoverageSmoothedRecorder", DiagnosticRecorder):
        summary = run_warm_variants(
            config, variants, output_dir=args.output / "forks",
            prefix_context=lambda: coverage.coverage_smoothed_candidate(start_step=1000))
    reference = json.loads(args.reference.read_text())
    replay = json.loads((args.output / "forks/coverage.json").read_text())
    if (reference["warm_state_sha256"] != replay["warm_state_sha256"]
            or reference["final_state_sha256"] != replay["final_state_sha256"]):
        raise RuntimeError("observer changed the frozen warm initial or final training state")
    for key in ("observations", "diagnostic", "noise", "dynamics_receipt", "optimizer_final"):
        if _without_seconds(reference[key]) != _without_seconds(replay[key]):
            raise RuntimeError(f"observer changed frozen warm {key}")
    raw = json.loads((args.output / "captured-raw.json").read_text())
    if [row["step"] for row in raw] != list(CHECK_STEPS):
        raise RuntimeError("requested neighbor/failure clean states were not all captured")
    observed = {row["step"]: row for row in replay["diagnostic"]}
    analyzed = _analyze(raw, observed)
    states = torch.load(args.output / "post-gan-counterfactual-states.pt",
                        weights_only=True)
    if sorted(states) != list(FAILED_STEPS):
        raise RuntimeError("failed-step counterfactual states were not all captured")
    verdict = dict(status="EXACT_REFERENCE_PARITY",
                   reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                   observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   candidate_source_sha256=hashlib.sha256(
                       (ROOT / "reports/toy100/coverage_smoothed_candidate.py").read_bytes()).hexdigest(),
                   warm_state_sha256=replay["warm_state_sha256"],
                   final_state_sha256=replay["final_state_sha256"],
                   post_gan_counterfactual_sha256=hashlib.sha256(
                       (args.output / "post-gan-counterfactual-states.pt").read_bytes()).hexdigest(),
                   local_stability=replay["local_stability"],
                   rows=analyzed)
    (args.output / "diagnosis.json").write_text(json.dumps(verdict, indent=2,
                                                           allow_nan=False) + "\n")
    print(json.dumps(dict(event="PARITY_AND_DIAGNOSIS_DONE",
                          status=verdict["status"],
                          failing_steps=verdict["local_stability"]["failing_steps"],
                          warm_state_sha256=verdict["warm_state_sha256"],
                          final_state_sha256=verdict["final_state_sha256"])), flush=True)


if __name__ == "__main__":
    main()
