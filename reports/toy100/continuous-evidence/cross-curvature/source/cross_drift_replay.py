"""Exact same-sample replay of cross-only competitive updates.

After each accepted cross-only update, the recorder returns to the update's
base point and replays the same host data/noise at three points: the accepted
joint step, only the D part of it, and only the G+prior part. That separates
each player's field change into own-player curvature (the block the
cross-only solve ignores), cross response and their non-additive
interaction. Clean outputs of the twelve particles, their nearest ring modes
and the base critic's input gradient describe functional motion and target
mismatch. Accepted parameters, buffers and every RNG stream are restored
bit-for-bit, so the trajectory equals the undiagnosed method; the warm run
checks this against a plain cross-only sibling's final state hash.

Diagnostic only: not a candidate, and not eligible for any gate.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from reports.toy100 import implicit_extra_scratch as implicit_module
from reports.toy100.cross_competitive_scratch import CrossCompetitiveRecorder
from reports.toy100.implicit_extra_scratch import ImplicitExtraRecorder

HQ_RADIUS = .21


def _norm(value):
    return float(torch.linalg.vector_norm(value))


class ReplayCrossRecorder(CrossCompetitiveRecorder):
    def __init__(self, record=lambda step: True, **options):
        super().__init__(**options)
        self.record = record
        self.replays = []
        self.replay_restores_verified = 0

    @torch.no_grad()
    def _functional(self, local):
        model = local["generator"]; clean = getattr(model, "model", model)
        critic = local["critic"]; clean_d = getattr(critic, "model", critic)
        means = local["means"].double(); z = local["prior"].z.detach()
        outputs = clean(z).double()
        distance = torch.cdist(outputs, means)
        nearest, mode = distance.min(1)
        with torch.enable_grad():
            x = outputs.detach().float().requires_grad_(True)
            gradient, = torch.autograd.grad(clean_d(x).sum(), x)
        center_values = clean_d(means.float()).double().flatten()
        return dict(outputs=outputs, nearest=nearest, mode=mode,
                    critic_gradient=gradient.double(), center_values=center_values)

    def _describe(self, before, after):
        displacement = after["outputs"] - before["outputs"]
        local_unit = before["outputs"].new_zeros(displacement.shape)
        centers = self._means[before["mode"]]
        offset = centers - before["outputs"]
        far = before["nearest"] > 1e-3
        local_unit[far] = offset[far] / before["nearest"][far, None]
        approach = (displacement * local_unit).sum(1)
        grad = before["critic_gradient"]
        grad_unit = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-30)
        radial_pull = (grad_unit * local_unit).sum(1)
        occupancy = torch.bincount(before["mode"], minlength=len(self._means))
        centered = before["center_values"] - before["center_values"].mean()
        occ = occupancy.double() - occupancy.double().mean()
        corr = (float((centered * occ).sum() / (centered.norm() * occ.norm()))
                if centered.norm() > 0 and occ.norm() > 0 else None)
        moved = displacement.norm(dim=1)
        worst = int(moved.argmax())
        return dict(
            particle_motion_max=float(moved.max()), particle_motion_rms=float(moved.square().mean().sqrt()),
            worst_particle=dict(index=worst, before_distance=float(before["nearest"][worst]),
                                after_distance=float(after["nearest"][worst]),
                                mode_before=int(before["mode"][worst]), mode_after=int(after["mode"][worst]),
                                occupancy_of_mode_before=int(occupancy[before["mode"][worst]]),
                                critic_radial_pull=float(radial_pull[worst])),
            outside_hq_before=int((before["nearest"] > HQ_RADIUS).sum()),
            outside_hq_after=int((after["nearest"] > HQ_RADIUS).sum()),
            nearest_distance_max_before=float(before["nearest"].max()),
            nearest_distance_max_after=float(after["nearest"].max()),
            mode_switches=int((before["mode"] != after["mode"]).sum()),
            occupancy_before=occupancy.tolist(),
            approach_mean=float(approach.mean()), approach_min=float(approach.min()),
            critic_radial_pull_mean=float(radial_pull[far].mean()) if far.any() else None,
            critic_radial_pull_min=float(radial_pull[far].min()) if far.any() else None,
            critic_gradient_norm_max=float(grad.norm(dim=1).max()),
            center_value_occupancy_correlation=corr)

    def phases(self, step, opt_d, opt_g, local):
        active = self.enabled and step >= self.start_step
        generator = super().phases(step, opt_d, opt_g, local)
        if not active:
            yield from generator
            return
        base = rng_before = before = None
        for value in generator:
            if base is None:
                base = {p: v.clone() for p, v in self.base.items()}
                rng_before = [state.clone() for state in self.rng_before]
                self._means = local["means"].double()
                before = self._functional(local) if self.record(step) else None
            yield value
        if before is None:
            return
        accepted = self._copy_parameters()
        buffers = [(buffer, buffer.detach().clone()) for buffer, _ in self.buffers]
        after = self._functional(local)
        streams_after = self._rng(self.streams)
        self.base = base; self.base_flat = self._flat(base); self.rng_before = rng_before
        self.rng_after = streams_after
        q0 = self.solve_q0; alpha = self.last_scale
        u = (self._flat(accepted) - self.base_flat) / self.root_metric
        split = sum(p.numel() for group in opt_d.param_groups for p in group["params"])
        u_d = torch.zeros_like(u); u_d[:split] = u[:split]
        u_g = torch.zeros_like(u); u_g[split:] = u[split:]
        q_full, _ = yield from ImplicitExtraRecorder._evaluate(self, u, "replay_joint")
        q_d, _ = yield from ImplicitExtraRecorder._evaluate(self, u_d, "replay_d_only")
        q_g, _ = yield from ImplicitExtraRecorder._evaluate(self, u_g, "replay_g_only")
        with torch.no_grad():
            for p, value in accepted.items():
                p.copy_(value)
            for buffer, value in buffers:
                buffer.copy_(value)
        if (any(not torch.equal(p, value) for p, value in accepted.items())
                or not all(torch.equal(a, b) for a, b in zip(streams_after, self._rng(self.streams)))):
            raise RuntimeError("replay changed the accepted update")
        self.replay_restores_verified += 1
        self.phase = None; self.base, self.point = {}, {}
        if self.accounting is not None:
            self.accounting(self.rows[opt_d]["calls"], self.outer_steps)
        scale = alpha * _norm(q0)
        players = {}
        for role, block in (("d", slice(0, split)), ("g", slice(split, len(u)))):
            own_moved = q_d if role == "d" else q_g
            other_moved = q_g if role == "d" else q_d
            players[role] = dict(
                explicit=alpha * _norm(q0[block]), step=_norm(u[block]),
                own_curvature=alpha * _norm((own_moved - q0)[block]),
                cross_response=alpha * _norm((other_moved - q0)[block]),
                joint_response=alpha * _norm((q_full - q0)[block]),
                interaction=alpha * _norm((q_full - q_d - q_g + q0)[block]),
                cross_residual=_norm(u[block] + alpha * torch.cat((q_g[:split], q_d[split:]))[block]),
                joint_residual=_norm(u[block] + alpha * q_full[block]))
        self.replays.append(dict(
            step=step + 1, alpha=alpha, explicit_norm=scale, players=players,
            cross_relative_residual=_norm(u + alpha * torch.cat((q_g[:split], q_d[split:]))) / scale,
            joint_relative_residual=_norm(u + alpha * q_full) / scale,
            functional=self._describe(before, after)))


@contextmanager
def replay_cross(task="mode_hold", record=lambda step: True, **options):
    class Bound(ReplayCrossRecorder):
        def __init__(self, **kwargs):
            super().__init__(record=record, **kwargs)
    with patch.object(implicit_module, "ImplicitExtraRecorder", Bound):
        with implicit_module.implicit_extra(task=task, **options) as value:
            yield value


def warm_variants():
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context

    def factory(method):
        @contextmanager
        def activate(state, prefix):
            recorders = prefix
            recorder = recorders.get(method)
            for name, other in recorders.items():
                other.enabled = name == method
            completed, target = state["completed_steps"], state["target_steps"]
            receipt = dict(method=method, shared_gate_eligible=False, scope="diagnostic_replay_only")
            if recorder is not None:
                def accounting(calls, outer):
                    state["declare_optimizer_accounting"](
                        calls=completed + calls + (target - completed - outer), moment_updates=target)
                recorder.accounting = accounting
            if method == "identity":
                yield receipt
                return
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            receipt.update(recorder.summary() if hasattr(recorder, "summary") else {},
                           replays=getattr(recorder, "replays", None),
                           replay_restores_verified=getattr(recorder, "replay_restores_verified", None),
                           rng_replay_verified=recorder.rng_replay_verified,
                           joint_residual_diagnostics=recorder.joint_residuals,
                           accepted_scales=[row["scale"] for row in recorder.solves if row["accepted"]])
        return activate
    return {name: factory(name) for name in ("identity", "cross_only", "cross_replay")}


def _stack_prefix():
    """Install both a plain and a replaying recorder; each child enables one."""
    from contextlib import ExitStack
    from reports.toy100.extra_adam_scratch import HOSTS, transformed_function
    from benchmarks.locked_shared import mode_hold
    import ast

    @contextmanager
    def context():
        tree, _, _ = transformed_function(mode_hold, "mode_hold")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "phases"]
        calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))
        ast.fix_missing_locations(tree)
        recorders = dict(cross_only=CrossCompetitiveRecorder(start_step=1000),
                         cross_replay=ReplayCrossRecorder(start_step=1000))
        for recorder in recorders.values():
            recorder.enabled = False

        class Dispatch:
            def phases(self, *args):
                active = [r for r in recorders.values() if r.enabled]
                return (active[0] if active else recorders["cross_only"]).phases(*args)

            def step(self, optimizer, ordinary, closure=None):
                active = [r for r in recorders.values() if r.enabled]
                return (active[0] if active else recorders["cross_only"]).step(optimizer, ordinary, closure)

        dispatch = Dispatch(); ordinary = torch.optim.Adam.step
        with ExitStack() as stack:
            stack.enter_context(patch.dict(mode_hold.__dict__, {"_extra_state": dispatch}))
            namespace = {}
            exec(compile(tree, "<cross-replay-mode_hold>", "exec"), mode_hold.__dict__, namespace)
            stack.enter_context(patch.object(mode_hold, HOSTS["mode_hold"], namespace[HOSTS["mode_hold"]]))
            stack.enter_context(patch.object(torch.optim.Adam, "step",
                                             lambda opt, closure=None: dispatch.step(opt, ordinary, closure)))
            yield recorders
    return context


def run_warm(output):
    from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    summary = run_warm_variants(config, warm_variants(), output_dir=output, prefix_context=_stack_prefix())
    plain = summary["variants"]["cross_only"]; replay = summary["variants"]["cross_replay"]
    summary["replay_matches_plain_cross_only"] = (
        plain["final_state_sha256"] == replay["final_state_sha256"]
        and plain["local_stability"] == replay["local_stability"])
    if not summary["replay_matches_plain_cross_only"]:
        raise RuntimeError("replay diagnostics perturbed the cross-only trajectory")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def run_cold(output):
    """Cold constant-rate cross-only mode-hold acquisition, diagnostic only."""
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="cross_competitive_response", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap"); config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    output.mkdir(parents=True, exist_ok=False)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")
    with replay_cross(task="mode_hold") as (recorder, _):
        result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    verdict = test_verdict(spec, result)
    data = dict(scope="diagnostic_replay_only", shared_gate_eligible=False, verdict=verdict,
                live=result["live"], observations=result.get("observations"),
                summary=recorder.summary() if hasattr(recorder, "summary") else None,
                replays=recorder.replays, replay_restores_verified=recorder.replay_restores_verified,
                rng_replay_verified=recorder.rng_replay_verified)
    (output / "cold_mode_hold.json").write_text(json.dumps(data, allow_nan=False) + "\n")
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("warm", "cold"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__), ROOT / "reports/toy100/cross_competitive_scratch.py",
        ROOT / "reports/toy100/implicit_extra_scratch.py", ROOT / "reports/toy100/extra_adam_scratch.py",
        ROOT / "benchmarks/toy100/warm_equilibrium_probe.py", ROOT / "benchmarks/toy100/continuous_probe.py")}
    result = run_warm(args.output) if args.stage == "warm" else run_cold(args.output)
    (args.output / "declaration.json").write_text(json.dumps(dict(
        stage=args.stage, scope="diagnostic_replay_only", seed=0, shared_gate_eligible=False,
        torch=torch.__version__, source=source), indent=2) + "\n")
    print(json.dumps(dict(event="DONE", stage=args.stage,
                          result={k: v for k, v in result.items() if k != "variants"}
                          if isinstance(result, dict) else result), default=str), flush=True)


if __name__ == "__main__":
    main()
