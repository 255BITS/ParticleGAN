"""One G step from the AVX512 reach stuck state, curvature bound on and off.

Trains the unchanged reach-.5 cold ring to ``--step`` completed updates, then
replays that next generator step twice at stencil width .5. The first replay
keeps the PR #82 own-curvature bound (.25). The second restores the same
weights, Adam moments and RNG and takes the step with the bound off. Mode
centers grade the two displacements afterward. They are not a training term.

Decision, fixed before looking at the step:
  build                 missing mode moves closer only with the bound off
  kill_shared_network   it moves closer in neither replay
  kill_premise          the bound-on step already moves closer, or only it does
  not_stuck             the captured cloud already covers every mode
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator
from reports.toy100.alternating_curvature_scratch import AlternatingCurvatureRecorder
from reports.toy100.pr84_reach_candidate import ReachRecorder
from reports.toy100.pr84_smoothed_candidate import SmoothedBothBoundRecorder
from reports.toy100.pr84_stuck_reach_diagnostic import FACTORIES

HQ_RADIUS = 0.21
CLOSURE = 1e-8


class _Stop(Exception):
    pass


def approaches(dist_before, dist_after, projection):
    """True when the pre-step nearest particles move toward the missing center."""
    if not dist_before or len(dist_before) != len(dist_after) or len(dist_before) != len(projection):
        raise ValueError("approach vectors must be the same non-empty length")
    before = sum(dist_before) / len(dist_before)
    after = sum(dist_after) / len(dist_after)
    toward = sum(projection) / len(projection)
    return before - after > CLOSURE and toward > 0.


def verdict(per_mode_on, per_mode_off, n_missing):
    if n_missing == 0:
        return "not_stuck"
    if len(per_mode_on) != n_missing or len(per_mode_off) != n_missing:
        raise ValueError("one approach flag per missing mode")
    on = all(per_mode_on)
    off = all(per_mode_off)
    if off and not on:
        return "build"
    if not off and not on:
        return "kill_shared_network"
    return "kill_premise"


def _streams(local):
    streams = [v for v in local.values() if isinstance(v, torch.Generator)]
    policy = local.get("noise_policy")
    if policy is not None:
        streams.extend(v for name in ("input_stream", "output_stream")
                       if isinstance((v := getattr(policy, name, None)), torch.Generator))
    return list({id(s): s for s in streams}.values())


def _opt_state(opt):
    return {p: {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
            for p, st in opt.state.items()}


def _put_opt_state(opt, saved):
    opt.state.clear()
    for p, st in saved.items():
        opt.state[p] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()}


def support(local):
    gen = local["generator"]
    clean = getattr(gen, "model", gen)
    with torch.no_grad():
        return clean(local["prior"].z).detach()


def _round(values):
    return [round(float(v), 6) for v in values]


def grade_mode(y0, y1, center):
    dist0 = torch.cdist(y0, center[None]).flatten()
    order = torch.argsort(dist0)[:3]
    d0 = torch.cdist(y0[order], center[None]).flatten()
    d1 = torch.cdist(y1[order], center[None]).flatten()
    direction = center - y0[order]
    unit = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    proj = ((y1[order] - y0[order]) * unit).sum(1)
    before, after, projection = _round(d0), _round(d1), _round(proj)
    return dict(nearest=[int(i) for i in order], dist_before=before, dist_after=after,
                projection=projection, closure=round(sum(before) - sum(after), 6),
                approaches=approaches(before, after, projection))


def summarise(y, means):
    dist = torch.cdist(y, means)
    missing = [int(k) for k in torch.nonzero(dist.min(0).values >= HQ_RADIUS).flatten()]
    occupancy = [int((dist.argmin(1) == k).sum()) for k in range(len(means))]
    return missing, occupancy


def replay(step, output):
    """Run reach to ``step`` and write one JSON object describing both G steps."""
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")
    original_phases = SmoothedBothBoundRecorder.phases
    original_arm = ReachRecorder._arm_smoothed_critic
    found = {}

    def emit(**row):
        print(json.dumps(row, default=float), flush=True)

    def arm(self):
        original_arm(self)
        forced = getattr(self, "_force_width", None)
        if forced is not None and self._smooth_on:
            self.row["unforced_width"] = self._smooth_width
            self._smooth_width = forced
            self.row["critic_width"] = forced

    def snapshot(recorder, opt_d, opt_g, local):
        params = [(p, p.detach().clone()) for opt in (opt_d, opt_g) for p in recorder._params(opt)]
        buffers = [(b, b.detach().clone()) for name in ("generator", "critic", "prior")
                   if isinstance((m := local.get(name)), torch.nn.Module) for b in m.buffers()]
        return dict(params=params, buffers=buffers, opt_d=_opt_state(opt_d), opt_g=_opt_state(opt_g),
                    rng=AlternatingCurvatureRecorder._rng(_streams(local)), streams=_streams(local),
                    outer=recorder.outer_steps, records=list(recorder.records),
                    calls=[(opt, recorder.rows[opt]["calls"]) for opt in recorder.rows],
                    verified=recorder.rng_replay_verified, bound=recorder.curvature_bound)

    def restore(recorder, opt_d, opt_g, snap):
        with torch.no_grad():
            for p, saved in snap["params"]:
                p.copy_(saved)
            for b, saved in snap["buffers"]:
                b.copy_(saved)
        _put_opt_state(opt_d, snap["opt_d"])
        _put_opt_state(opt_g, snap["opt_g"])
        AlternatingCurvatureRecorder._set_rng(snap["streams"], snap["rng"])
        recorder.outer_steps = snap["outer"]
        recorder.records = list(snap["records"])
        for opt, calls in snap["calls"]:
            recorder.rows[opt]["calls"] = calls
        recorder.rng_replay_verified = snap["verified"]
        recorder.curvature_bound = snap["bound"]
        recorder._force_width = None
        recorder.phase = None

    def dynamics(recorder):
        row = recorder.records[-1]
        return dict(rho=row["g"]["rho"], factor=row["g"]["factor"],
                    sharpness=row.get("critic_sharpness"), width=row.get("critic_width"),
                    unforced_width=row.get("unforced_width"), advantage=row.get("critic_advantage"))

    def phases(self, s, opt_d, opt_g, local):
        if self.outer_steps == step and "replay" not in found:
            snap = snapshot(self, opt_d, opt_g, local)
            y0 = support(local)
            means = local["means"]
            missing, occupancy = summarise(y0, means)
            emit(event="CAPTURED", step=step, missing=missing, occupancy=occupancy)
            critic = local["critic"]
            while not isinstance(critic, SimpleMLPDiscriminator):
                critic = critic.model
            gen = getattr(local["generator"], "model", local["generator"])
            torch.save(dict(critic=critic.state_dict(), generator=gen.state_dict(),
                            z=local["prior"].z.detach().clone(), means=means.detach().clone(),
                            opt_d=opt_d.state_dict(), opt_g=opt_g.state_dict(),
                            missing=missing, occupancy=occupancy), output / "stuck-state.pt")
            sides = {}
            for name, bound in (("on", 0.25), ("off", 1e6)):
                if name == "off":
                    restore(self, opt_d, opt_g, snap)
                    drift = float((support(local) - y0).abs().max())
                    if drift > 1e-6:
                        raise RuntimeError(f"state restore drifted by {drift}")
                self._force_width = 0.5
                self.curvature_bound = bound
                yield from original_phases(self, s, opt_d, opt_g, local)
                y1 = support(local)
                modes = [dict(mode=k, **grade_mode(y0, y1, means[k])) for k in missing]
                sides[name] = dict(bound=bound, dynamics=dynamics(self), modes=modes,
                                   approaches=[m["approaches"] for m in modes],
                                   points_after=[[round(float(v), 6) for v in row] for row in y1])
                emit(event="REPLAY", bound=name, **{k: sides[name][k] for k in ("dynamics", "approaches")})
            decision = verdict(sides["on"]["approaches"], sides["off"]["approaches"], len(missing))
            found["replay"] = dict(step=step, missing=missing, occupancy=occupancy,
                                   on=sides["on"], off=sides["off"], verdict=decision,
                                   points_before=[[round(float(v), 6) for v in row] for row in y0])
            emit(event="VERDICT", verdict=decision, missing=missing)
            raise _Stop
        if self.outer_steps % 50 == 0:
            y = support(local)
            missing, occupancy = summarise(y, local["means"])
            last = self.records[-1] if self.records else {}
            emit(event="STEP", step=self.outer_steps, missing=missing, occupancy=occupancy,
                 sharpness=last.get("critic_sharpness"), width=last.get("critic_width"),
                 g_factor=(last.get("g") or {}).get("factor"))
        yield from original_phases(self, s, opt_d, opt_g, local)

    ReachRecorder._arm_smoothed_critic = arm
    SmoothedBothBoundRecorder.phases = phases
    try:
        with FACTORIES["reach"](task="mode_hold"):
            run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    except _Stop:
        pass
    finally:
        SmoothedBothBoundRecorder.phases = original_phases
        ReachRecorder._arm_smoothed_critic = original_arm
    if "replay" not in found:
        raise RuntimeError("replay did not run")
    report = found["replay"]
    report.update(cpu=torch.backends.cpu.get_cpu_capability(), torch=torch.__version__,
                  seed=0, threads=1, host="neural", width=0.5,
                  reach_source=hashlib.sha256((ROOT / "reports/toy100/pr84_reach_candidate.py").read_bytes()).hexdigest(),
                  rule="build only if every missing mode approaches with the .25 bound off and not with it on")
    (output / "replay.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.step, args.output)


if __name__ == "__main__":
    main()
