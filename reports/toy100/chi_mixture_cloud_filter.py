"""Free-particle filter for one reverse-KL + chi-squared KDE drift.

Cao & Wei (arXiv:2603.10592) Table 1: the Wasserstein velocity of each
f-divergence shares the KDE score gap, with weight p/q for reverse KL and
q/p for chi-squared. The practical field drops the shared 1/h^2 factor, as
in the h^2-scaled drifting identity, leaving

    v = (p/q + q/p) * (m_p - m_q)

where m_p, m_q are Gaussian-kernel mean shifts. Forward KL (weight 1) is the
drifting field already flagged for a separate test; this filter does not run it.

Bandwidth is one declared state rule, not a sweep: the maximum
real-to-nearest-particle distance, clamped to [0.07, 3]. A quantile at or
below 7/8 stays inside the covered mass and cannot see one missing mode.
Euler displacement is capped at 0.1 per particle. Clean HQ is a proxy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch


WIDTH_FLOOR = 0.07
WIDTH_CAP = 3.0
STEP_CAP = 0.1
# Above 1 - 1/8 so one fully missing mode, not its boundary, sets the width.
QUANTILE = 1.0


def bandwidth(real, particles):
    gap = torch.cdist(real, particles).min(dim=1).values
    return torch.quantile(gap, QUANTILE).clamp(WIDTH_FLOOR, WIDTH_CAP)


def field(particles, real, h):
    """Return v and the density ratio at each particle. float64 in, float64 out."""
    def stats(samples):
        log_k = -0.5 * torch.cdist(particles, samples).square() / (h * h)
        log_k = log_k - log_k.max(dim=1, keepdim=True).values
        weight = log_k.exp()
        mass = weight.sum(dim=1)
        mean = (weight @ samples) / mass[:, None]
        return mass / len(samples), mean

    p, m_p = stats(real)
    q, m_q = stats(particles)
    ratio = (p / q.clamp_min(1e-30)).clamp(1e-6, 1e6)
    weight = ratio + 1.0 / ratio
    return weight[:, None] * (m_p - m_q), ratio


def grade(x, centers):
    dist = torch.cdist(x, centers)
    return dict(modes=int((dist.min(0).values <= .21).sum()),
                clean_hq=float((dist.min(1).values <= .21).double().mean()))


def run():
    torch.set_num_threads(1)
    theta = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    centers = 3 * torch.stack((theta.cos(), theta.sin()), 1)
    assignments = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7])
    angle = torch.arange(12, dtype=torch.float64) * (2 * math.pi / 12)
    offset = .029 * torch.stack((angle.cos(), angle.sin()), 1)
    # Deterministic within-mode cloud: 16 angles at the data width, not a seed.
    sample_angle = torch.arange(16, dtype=torch.float64) * (2 * math.pi / 16)
    sample_offset = .07 * torch.stack((sample_angle.cos(), sample_angle.sin()), 1)
    reals = (centers[:, None, :] + sample_offset[None]).reshape(-1, 2)
    assert reals.shape == (128, 2)

    same = centers[:4].repeat(3, 1)
    rest, _ = field(same, same, torch.tensor(1.0, dtype=torch.float64))
    if float(rest.abs().max()) != 0.:
        raise RuntimeError("identical clouds must rest")

    cases = {}
    passing = centers[assignments] + offset
    missing = centers[torch.where(assignments == 6, torch.tensor(5), assignments)] + offset
    specs = {
        "distinct_vs_centers": (passing.clone(), centers),
        "missing_vs_centers": (missing.clone(), centers),
        "distinct_vs_width_cloud": (passing.clone(), reals),
        "missing_vs_width_cloud": (missing.clone(), reals),
    }
    for name, (x0, real) in specs.items():
        initial = grade(x0, centers)
        graded = []
        x = x0.clone()
        for step in range(1, 201):
            h = bandwidth(real, x)
            drift, ratio = field(x, real, h)
            move = drift.norm(dim=1, keepdim=True).clamp_min(1e-12)
            x = x + drift * (STEP_CAP / move).clamp(max=1.0)
            row = dict(step=step, **grade(x, centers), h=float(h),
                       drift_max=float(drift.norm(dim=1).max()),
                       ratio_max=float(ratio.max()))
            graded.append(row)
        final_grade = grade(x, centers)
        cases[name] = dict(
            initial=initial, final=final_grade,
            first_failed_quality=next((r["step"] for r in graded
                                       if name.startswith("distinct") and (r["modes"] < 8 or r["clean_hq"] < .9)), None),
            first_full_support=next((r["step"] for r in graded if r["modes"] == 8 and r["clean_hq"] >= .9), None),
            min_clean_hq=min(r["clean_hq"] for r in graded),
            max_h=max(r["h"] for r in graded),
            trace=graded)
    return dict(
        method="reverse_kl_plus_chi2_kde_drift",
        source="https://arxiv.org/abs/2603.10592",
        weights=dict(reverse_kl=1.0, chi2=1.0),
        quantile=QUANTILE, width_floor=WIDTH_FLOOR, width_cap=WIDTH_CAP, step_cap=STEP_CAP,
        identical_cloud_max_drift=0.0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="clean free-particle diagnostic, not a host gate",
        cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    began = time.perf_counter()
    result = run()
    result["seconds"] = time.perf_counter() - began
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    brief = {k: {n: v for n, v in c.items() if n != "trace"} for k, c in result["cases"].items()}
    print(json.dumps(dict(seconds=result["seconds"], cases=brief), indent=2), flush=True)
