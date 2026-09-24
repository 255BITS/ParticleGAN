"""Offline replay of the frozen mode-hold 4096-draw noisy HQ diagnostic.

This scores saved clean support clouds after training; it never enters the
candidate policy. It reproduces the exact evaluation indices and output noise
for the pinned seed-0 mode-hold host with no isolated output-noise stream.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold


STAGES = ("pre_gan", "post_gan", "ideal_target_cloud", "post_projection")


def fixed_draw(step, support):
    """Return the host's fixed particle indices and Gaussian output noise."""
    if support.shape != (12, 2) or step < 240:
        raise ValueError("expected 12 two-dimensional particles after noise warmup")
    with torch.random.fork_rng(devices=[]):
        # NoisePolicy.evaluation(step), seed=0, output_noise_rng=None.
        torch.random.default_generator.manual_seed(402 + step)
        # mode_hold.measure calls prior.sample with a fresh seed+9 stream.
        index = torch.randint(0, len(support), (mode_hold.EVAL_N,),
                              generator=torch.Generator().manual_seed(9))
        noise = .029 * torch.randn_like(support[index])
    return index, noise


def score_support(support, index, noise, means):
    draws = support[index] + noise
    grade = mode_hold.diversity(draws, means)
    nearest = torch.cdist(draws, means).min(dim=1).values
    hit = nearest <= .21
    counts = torch.bincount(index, minlength=len(support))
    good = torch.bincount(index[hit], minlength=len(support))
    return dict(modes=grade["modes"], hq=grade["hq"],
                particle_draws=counts.tolist(),
                particle_hq_rate=(good / counts).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    source = json.loads(args.diagnosis.read_text())
    if source["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("clean support capture did not prove reference parity")
    means = mode_hold.ring_means()
    rows = []
    for row in source["rows"]:
        points = {name: torch.tensor(row[name]["points"], dtype=torch.float32)
                  for name in STAGES}
        index, noise = fixed_draw(row["step"], points["post_projection"])
        grades = {name: score_support(points[name], index, noise, means)
                  for name in STAGES}
        if (grades["post_projection"]["modes"] != row["observed_live"]["modes"]
                or grades["post_projection"]["hq"] != row["observed_live"]["hq"]):
            raise RuntimeError(f"step {row['step']}: exact host diagnostic did not replay")
        rows.append(dict(step=row["step"], grades=grades,
                         original_live=row["observed_live"],
                         exact_original_live_parity=True))
    result = dict(scope="post-training diagnostic only", seed=0,
                  eval_n=mode_hold.EVAL_N, particle_index_seed=9,
                  output_noise_seed="402 + step", output_sigma=.029,
                  output_noise_rng=None, hq_radius=.21,
                  clean_capture_sha256=hashlib.sha256(args.diagnosis.read_bytes()).hexdigest(),
                  scorer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="FIXED_EVAL_PARITY", steps=len(rows),
                          failed=[row["step"] for row in rows
                                  if row["grades"]["post_projection"]["hq"] < .9])), flush=True)


if __name__ == "__main__":
    main()
