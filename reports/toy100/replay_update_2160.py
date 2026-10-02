"""Read-only replay of fence-restore update 2160.

Same host as the hold probe: scheduled prefix through update 1000, then
constant rates and the fence-restore clip. No rule change. Prints one JSON
line per watched update and the graded checkpoints around 2160.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from benchmarks.locked_shared.mode_hold import SIGMA, ring_means
from benchmarks.toy100.continuous_probe import run_probe
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context
from reports.toy100.exit_aware_step_clip import (
    ExitAwareRecorder, exit_scales, support_fence,
)

HQ_RADIUS = 3.0 * SIGMA
WATCH = {2159, 2160, 2161}
_ORIGINAL = ExitAwareRecorder._clip_exits
_RATE = None


def _summary(before, proposed, applied, reals, scales):
    centers = ring_means().to(before.device)
    fence = float(support_fence(reals))
    d_real = lambda x: torch.cdist(x, reals).min(dim=1).values
    d_center = lambda x: torch.cdist(x, centers).min(dim=1).values
    r0, r1, ra = d_real(before), d_real(proposed), d_real(applied)
    c0, c1, ca = d_center(before), d_center(proposed), d_center(applied)
    step_p = (proposed - before).norm(dim=1)
    step_a = (applied - before).norm(dim=1)
    outside = ca > HQ_RADIUS
    inside_fence = ra <= fence + 1e-8
    hole = outside & inside_fence

    def pack(dist):
        return dict(median=float(dist.median()), max=float(dist.max()),
                    frac_over_fence=float((dist > fence + 1e-8).float().mean()),
                    frac_over_hq=float((dist > HQ_RADIUS).float().mean()))

    worst = torch.topk(ca, k=min(8, ca.numel())).indices.tolist()
    rows = []
    for i in worst:
        rows.append(dict(
            i=i, real_before=float(r0[i]), real_proposed=float(r1[i]),
            real_applied=float(ra[i]), center_before=float(c0[i]),
            center_proposed=float(c1[i]), center_applied=float(ca[i]),
            step_proposed=float(step_p[i]), step_applied=float(step_a[i]),
            scale=float(scales[i]),
        ))
    return dict(
        n=int(before.shape[0]), fence=fence, hq_radius=HQ_RADIUS,
        real_before=pack(r0), real_proposed=pack(r1), real_applied=pack(ra),
        center_before=pack(c0), center_proposed=pack(c1), center_applied=pack(ca),
        hq_frac_before=float((c0 <= HQ_RADIUS).float().mean()),
        hq_frac_proposed=float((c1 <= HQ_RADIUS).float().mean()),
        hq_frac_applied=float((ca <= HQ_RADIUS).float().mean()),
        hole_applied=int(hole.sum()),
        hole_and_center_rose=int((hole & (ca > c0 + 1e-8)).sum()),
        clipped=int((scales < 1 - 1e-6).sum()),
        min_scale=float(scales.min()),
        median_step_proposed=float(step_p.median()),
        median_step_applied=float(step_a.median()),
        max_step_proposed=float(step_p.max()),
        max_step_applied=float(step_a.max()),
        worst=rows,
    )


def _observing_clip(self):
    host = self.start_step + int(self.row.get("outer_step", 0))
    watch = host in WATCH
    before = proposed = reals = None
    if watch:
        local = self._local
        generator, prior = local["generator"], local["prior"]
        opt_g = self.optimizers[1]
        proposed_w = [p.detach().clone() for p in self._params(opt_g)]
        before = self._clean_positions(self.g_base)
        proposed = self._clean_positions(proposed_w)
        reals = self._reals
    _ORIGINAL(self)
    if not watch:
        return
    local = self._local
    generator, prior = local["generator"], local["prior"]
    clean = getattr(generator, "model", generator)
    with torch.no_grad():
        applied = clean(prior.z).detach()
        scales, _ = exit_scales(before, proposed, reals)
    payload = _summary(before, proposed, applied, reals, scales)
    g = self.row.get("g") or {}
    payload.update(update=host, rho=g.get("rho"), factor=g.get("factor"),
                   exit_clipped=self.row.get("exit_clipped"),
                   exit_restored=self.row.get("exit_restored"))
    print("event=REPLAY " + json.dumps(payload), flush=True)


def main():
    global _RATE
    ExitAwareRecorder._clip_exits = _observing_clip
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    holder = {}

    def hook(state):
        global _RATE
        recorder = holder["recorder"]
        recorder.clip_enabled = True
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        _RATE = constant_rate_context(state)
        receipt = _RATE.__enter__()
        print(json.dumps(dict(event="RATES", **receipt)), flush=True)

    def log(row):
        step = row.get("step", 0)
        if row.get("event") == "checkpoint" and step in (1000, 1200, 2140, 2150, 2160, 2170):
            print(json.dumps(row), flush=True)

    with __import__("reports.toy100.exit_aware_step_clip", fromlist=["exit_aware_candidate"]).exit_aware_candidate(start_step=1000) as (recorder, _):
        holder["recorder"] = recorder
        run_probe(
            config, mode="scheduled", steps=2170, diagnostic_every=10,
            checkpoint_hook_step=1000, checkpoint_hook=hook, log=log,
        )
    if _RATE is not None:
        _RATE.__exit__(None, None, None)


if __name__ == "__main__":
    main()
