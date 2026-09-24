"""Read-only critic-support trace on stall reach. Training is unchanged.

Logs, after each update, a critic-landscape occupancy of the clean particles:
how many spatially distinct local maxima of the critic the particles occupy.
Mode centers are graded only for the offline log. G curvature is recorded and
not used as a detector.
"""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator  # noqa: E402
from benchmarks.locked_shared.mode_hold import SIGMA, sample_ring  # noqa: E402
from reports.toy100 import pr84_reach_candidate as reach  # noqa: E402

PROBE = .25
SEPARATION = 1.0
HQ_RADIUS = .21


def support_count(module, points, probe=PROBE, separation=SEPARATION):
    center = module(points)
    on_peak = torch.ones(points.shape[0], dtype=torch.bool)
    for dim in range(points.shape[-1]):
        shift = torch.zeros_like(points)
        shift[:, dim] = probe
        on_peak &= center >= module(points + shift)
        on_peak &= center >= module(points - shift)
    chosen = points[on_peak]
    keep = []
    for point in chosen:
        if all(float((point - kept).norm()) >= separation for kept in keep):
            keep.append(point)
    return int(on_peak.sum()), len(keep), float(center.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", type=int, default=1900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from benchmarks.toy100.continuous_probe import run_probe

    rows = []
    original = reach.ReachRecorder._arm_smoothed_critic

    def arm(self):
        original(self)
        local = self._local or {}
        critic, generator, prior = local.get("critic"), local.get("generator"), local.get("prior")
        if critic is None or generator is None or prior is None or not self._smooth_on:
            return
        module = critic
        while not isinstance(module, SimpleMLPDiscriminator) and hasattr(module, "model"):
            module = module.model
        if not isinstance(module, SimpleMLPDiscriminator):
            return
        points = getattr(generator, "model", generator)(prior.z).detach()
        means = local.get("means")
        armed = self._smooth_on
        self._smooth_on = False
        try:
            n_peak, basins, logit = support_count(module, points)
            if means is not None:
                probe = torch.Generator()
                probe.manual_seed(0)
                real = sample_ring(means, 128, SIGMA, probe)
                real_values = module(real)
                fake_values = module(points)
                self.row["support_gap"] = float(fake_values.mean() - real_values.mean())
                level = real_values.median()
                self.row["support_frac"] = float((fake_values >= level).float().mean())
        finally:
            self._smooth_on = armed
        self.row["support_peaks"] = n_peak
        self.row["support_basins"] = basins
        self.row["support_logit"] = logit

    reach.ReachRecorder._arm_smoothed_critic = arm

    original_phases = reach.ReachRecorder.phases

    def phases(self, step, opt_d, opt_g, local):
        yield from original_phases(self, step, opt_d, opt_g, local)
        rec = self.records[-1] if self.records else {}
        with torch.no_grad():
            gen = local["generator"]
            y = getattr(gen, "model", gen)(local["prior"].z).detach()
            dist = torch.cdist(y, local["means"])
            modes = int((dist.min(0).values < HQ_RADIUS).sum())
            hq = float((dist.min(1).values < HQ_RADIUS).float().mean())
        row = dict(step=self.outer_steps, modes=modes, hq=round(hq, 3),
                   basins=rec.get("support_basins"), peaks=rec.get("support_peaks"),
                   logit=None if rec.get("support_logit") is None else round(rec["support_logit"], 4),
                   gap=None if rec.get("support_gap") is None else round(rec["support_gap"], 4),
                   frac=None if rec.get("support_frac") is None else round(rec["support_frac"], 3),
                   sharp=None if rec.get("critic_sharpness") is None else round(rec["critic_sharpness"], 3),
                   adv=None if rec.get("critic_advantage") is None else round(rec["critic_advantage"], 4),
                   g_factor=round(rec.get("g", {}).get("factor", 0.), 3))
        rows.append(row)
        if self.outer_steps % 25 == 0 or (1700 <= self.outer_steps <= args.until):
            print(json.dumps(row), flush=True)

    reach.ReachRecorder.phases = phases
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    with reach.pr84_reach_candidate(task="mode_hold", ramp="stall") as (recorder, _source):
        def hook(state):
            declare = state["declare_optimizer_accounting"]
            recorder.accounting = lambda calls, outer: declare(
                calls=calls + (args.until - outer), moment_updates=args.until)
            recorder.accounting(recorder.rows[recorder.optimizers[0]]["calls"], recorder.outer_steps)
        run_probe(config, mode="constant", steps=args.until, diagnostic_every=50,
                  checkpoint_hook_step=1, checkpoint_hook=hook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows) + "\n")
    print(json.dumps(dict(event="TRACE_DONE", n=len(rows), cpu=torch.backends.cpu.get_cpu_capability())), flush=True)


if __name__ == "__main__":
    main()
