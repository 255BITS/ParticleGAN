"""Read-only stage capture of the exact PR84 passing-state continuation.

This observer runs the selected, unchanged scratch update from the scheduled
step-1000 state.  Clean generator forwards and tensor copies are the only
operations inserted in training.  Ring centers and fixed evaluation noise are
used after full-state parity is established, never by the candidate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from particlegan.particle_prior import ParticlePrior
from reports.toy100 import pr84_smoothed_candidate as pr84
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


DENSE_START = 1300
DENSE_END = 1600
STATE_STEPS = (1324, 1325, 1326, 1380, 1389, 1390, 1391, 1400,
               1530, 1539, 1540, 1541, 1570)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return deepcopy(value)


def _rng(local):
    policy = local["noise_policy"]
    return dict(torch=torch.get_rng_state().clone(),
                data=local["stream"].get_state().clone(),
                input=policy.input_stream.get_state().clone(),
                output=(None if policy.output_stream is None else
                        policy.output_stream.get_state().clone()))


def _same_rng(before, after):
    return all((a is None and b is None) or
               (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and torch.equal(a, b)) for a, b in
               ((before[key], after[key]) for key in before))


class CaptureRecorder(pr84.SmoothedBothBoundRecorder):
    """Capture support and full selected states without touching host updates."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.capture_enabled = False
        self.current_step = None
        self.stage_rows = []
        self.saved_states = {}
        self.sampled_batches = {}

    @staticmethod
    def _clean(local, *, model_state=None, prior_z=None):
        generator = local["generator"]
        clean = getattr(generator, "model", generator)
        z = local["prior"].z if prior_z is None else prior_z
        with torch.no_grad():
            value = (clean(z) if model_state is None else
                     functional_call(clean, model_state, (z,)))
        return value.detach().clone()

    @staticmethod
    def _state(local, opt_d, opt_g):
        policy = local["noise_policy"]
        return _clone(dict(
            generator=local["generator"].state_dict(),
            critic=local["critic"].state_dict(),
            prior=local["prior"].state_dict(),
            optimizer_d=opt_d.state_dict(), optimizer_g=opt_g.state_dict(),
            ema_g=local["ema_g"], ema_z=local["ema_z"],
            rng=_rng(local), noise=dict(step_calls=policy.receipt()["step_calls"],
                                        input_sigma=policy.input_sigma,
                                        output_sigma=policy.output_sigma),
        ))

    def phases(self, step, opt_d, opt_g, local):
        number = step + 1
        self.current_step = number
        capture = self.capture_enabled and DENSE_START <= number <= DENSE_END
        save = self.capture_enabled and number in STATE_STEPS
        if capture or save:
            rng_before = _rng(local)
            base_z = local["prior"].z.detach().clone()
            clean = getattr(local["generator"], "model", local["generator"])
            base_model = {name: tensor.detach().clone()
                          for name, tensor in clean.state_dict().items()}
            base = self._clean(local)
            if not _same_rng(rng_before, _rng(local)):
                raise RuntimeError("pre-step clean support consumed training RNG")
            if save:
                self.saved_states[number] = dict(pre_step=self._state(local, opt_d, opt_g))
            stage = dict(step=number, pre_step=base.tolist())
        for phase in super().phases(step, opt_d, opt_g, local):
            yield phase
            if capture or save:
                if phase == 0:
                    stage["post_d"] = self._clean(local).tolist()
                    if save:
                        self.saved_states[number]["post_d"] = self._state(local, opt_d, opt_g)
                elif phase == 1:
                    stage["unbounded_joint"] = self._clean(local).tolist()
                    stage["unbounded_network_only"] = self._clean(
                        local, prior_z=base_z).tolist()
                    stage["unbounded_prior_only"] = self._clean(
                        local, model_state=base_model).tolist()
                    if save:
                        self.saved_states[number]["post_unbounded_g"] = self._state(
                            local, opt_d, opt_g)
                elif phase == 2:
                    stage["bounded_joint"] = self._clean(local).tolist()
                    stage["bounded_network_only"] = self._clean(
                        local, prior_z=base_z).tolist()
                    stage["bounded_prior_only"] = self._clean(
                        local, model_state=base_model).tolist()
                    if save:
                        self.saved_states[number]["post_bounded_g"] = self._state(
                            local, opt_d, opt_g)
                else:
                    raise RuntimeError("unexpected PR84 replay phase")
        if capture:
            stage["record"] = _clone(self.records[-1])
            stage["batches"] = self.sampled_batches.get(number, [])
            self.stage_rows.append(stage)

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        result = super().step(optimizer, ordinary_step, closure)
        if (self.capture_enabled and self.current_step in STATE_STEPS
                and self.phase == 1 and optimizer is self.optimizers[0]):
            # D* has materialized; G and prior remain at their base point.
            self.saved_states[self.current_step]["post_accepted_d"] = self._state(
                self._local, *self.optimizers)
        return result


def _untimed(value):
    if isinstance(value, dict):
        return {key: _untimed(item) for key, item in value.items()
                if "seconds" not in key}
    if isinstance(value, list):
        return [_untimed(item) for item in value]
    return value


def _analyze(rows, result):
    means = mode_hold.ring_means()
    observed = {row["step"]: row for row in result["diagnostic"]}
    analyzed = []
    for row in rows:
        number = row["step"]
        supports = {name: torch.tensor(row[name], dtype=torch.float32)
                    for name in ("pre_step", "post_d", "unbounded_joint",
                                 "unbounded_network_only", "unbounded_prior_only",
                                 "bounded_joint", "bounded_network_only",
                                 "bounded_prior_only")}
        index, noise = fixed_draw(number, supports["bounded_joint"])
        grades = {name: score_support(value, index, noise, means)
                  for name, value in supports.items()}
        if number in observed and (grades["bounded_joint"]["modes"] != observed[number]["modes"]
                                   or grades["bounded_joint"]["hq"] != observed[number]["hq"]):
            raise RuntimeError(f"fixed evaluation parity failed at step {number}")
        base = supports["pre_step"]
        def movement(name):
            delta = supports[name] - base
            return dict(rms=float(delta.square().sum(dim=1).mean().sqrt()),
                        maximum=float(delta.norm(dim=1).max()),
                        per_particle=delta.norm(dim=1).tolist())
        analyzed.append(dict(step=number, stages=row, grades=grades,
                             movement={name: movement(name) for name in supports
                                       if name != "pre_step"},
                             observed=(None if number not in observed else
                                       dict(modes=observed[number]["modes"],
                                            hq=observed[number]["hq"]))))
    return analyzed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())

    @contextmanager
    def prefix():
        with patch.object(pr84, "SmoothedBothBoundRecorder", CaptureRecorder):
            with pr84.pr84_smoothed_candidate(start_step=1000) as pair:
                recorder, _ = pair
                original_real = mode_hold.sample_ring
                original_prior_sample = ParticlePrior.sample

                def sample_ring(*values, **kwargs):
                    drawn = original_real(*values, **kwargs)
                    if (recorder.capture_enabled and recorder.current_step in STATE_STEPS
                            and recorder.phase in (0, 1, 2)):
                        recorder.sampled_batches.setdefault(recorder.current_step, []).append(
                            dict(kind="real", phase=recorder.phase,
                                 values=drawn.detach().clone().tolist()))
                    return drawn

                def sample_prior(prior, *values, **kwargs):
                    drawn = original_prior_sample(prior, *values, **kwargs)
                    if (recorder.capture_enabled and recorder.current_step in STATE_STEPS
                            and recorder.phase in (0, 1, 2)):
                        latent, indices = drawn
                        recorder.sampled_batches.setdefault(recorder.current_step, []).append(
                            dict(kind="prior", phase=recorder.phase,
                                 latent=latent.detach().clone().tolist(),
                                 indices=indices.detach().clone().tolist()))
                    return drawn

                with patch.object(mode_hold, "sample_ring", sample_ring), \
                     patch.object(ParticlePrior, "sample", sample_prior):
                    yield pair

    @contextmanager
    def activate(name, state, prefix_result):
        recorder, _ = prefix_result
        recorder.enabled = name == "original"
        recorder.capture_enabled = name == "original"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer,
            moment_updates=target)
        receipt = dict(method=name, shared_gate_eligible=False,
                       scratch_optimizer_policy=pr84.METHOD if recorder.enabled else "control")
        if name == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            receipt.update(recorder.receipt())
            (args.output / "captured-raw.json").write_text(
                json.dumps(recorder.stage_rows, allow_nan=False) + "\n")
            torch.save(recorder.saved_states, args.output / "selected-states.pt")

    variants = {name: (lambda state, prefix_result, name=name:
                       activate(name, state, prefix_result))
                for name in ("identity", "original")}
    summary = run_warm_variants(config, variants, output_dir=args.output / "forks",
                                steps=2400, prefix_context=prefix)
    with gzip.open(args.reference, "rt") if args.reference.suffix == ".gz" else \
            args.reference.open() as file:
        reference = json.load(file)
    observed = json.loads((args.output / "forks/original.json").read_text())
    if (reference["warm_state_sha256"] != observed["warm_state_sha256"] or
            reference["final_state_sha256"] != observed["final_state_sha256"]):
        raise RuntimeError("capture changed the original warm or final state")
    ref_diag = {row["step"]: row for row in reference["diagnostic"]}
    new_diag = {row["step"]: row for row in observed["diagnostic"]}
    if set(ref_diag) != set(new_diag) or any(ref_diag[step] != new_diag[step]
                                             for step in ref_diag):
        raise RuntimeError("capture changed an original diagnostic")
    if reference["dynamics_receipt"]["records"] != \
            observed["dynamics_receipt"]["records"]:
        raise RuntimeError("capture changed the curvature/width records")
    for key in ("noise", "post_checkpoint_rate_ranges", "optimizer_final", "final", "ema"):
        if _untimed(reference[key]) != _untimed(observed[key]):
            raise RuntimeError(f"capture changed {key}")
    raw = json.loads((args.output / "captured-raw.json").read_text())
    if [row["step"] for row in raw] != list(range(DENSE_START, DENSE_END + 1)):
        raise RuntimeError("dense capture has a missing step")
    analyzed = _analyze(raw, observed)
    states = torch.load(args.output / "selected-states.pt", weights_only=True)
    if sorted(states) != list(STATE_STEPS):
        raise RuntimeError("selected full states have a missing step")
    diagnosis = dict(status="EXACT_REFERENCE_PARITY", scope="diagnostic_read_only",
                     reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                     observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     candidate_source_sha256=hashlib.sha256((ROOT / "reports/toy100/pr84_smoothed_candidate.py").read_bytes()).hexdigest(),
                     warm_state_sha256=observed["warm_state_sha256"],
                     final_state_sha256=observed["final_state_sha256"],
                     selected_states_sha256=hashlib.sha256((args.output / "selected-states.pt").read_bytes()).hexdigest(),
                     local_stability=summary["variants"]["original"]["local_stability"],
                     long_hold=summary["variants"]["original"]["long_hold"],
                     rows=analyzed)
    (args.output / "diagnosis.json").write_text(json.dumps(diagnosis, allow_nan=False) + "\n")
    print(json.dumps(dict(event="DONE", status=diagnosis["status"],
                          captured=len(analyzed),
                          failed_hold=diagnosis["long_hold"]["failing_steps"],
                          selected_states=str(args.output / "selected-states.pt"))), flush=True)


if __name__ == "__main__":
    main()
