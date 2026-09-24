"""Fixed-cloud diagnosis of energy improvement versus frozen mode-hold HQ."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch

from benchmarks.locked_shared import mode_hold


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "reports/toy100/continuous-evidence/energy-signal/backtracking_warm_fail.tar.gz"


def score(real, fake):
    # The real-real term is unchanged between proposals.
    return float(2 * torch.cdist(real, fake).mean() - torch.cdist(fake, fake).mean())


def run():
    torch.set_num_threads(1)
    means = mode_hold.ring_means()
    # Stratification removes finite-bin imbalance while preserving the
    # frozen equal-weight Gaussian ring target.
    real = means.repeat_interleave(1024, dim=0) + mode_hold.SIGMA * torch.randn(
        8192, 2, generator=torch.Generator().manual_seed(0))
    # Twelve equal-weight particles cannot each occupy a distinct one of
    # eight target modes. Four centers therefore have a duplicate.
    occupancy = [2, 2, 2, 2, 1, 1, 1, 1]
    base = torch.cat([means[index].repeat(count, 1)
                      for index, count in enumerate(occupancy)])
    moved = base.clone()
    for index, angle in ((4, 4.0), (6, 4.8)):
        moved[index] = mode_hold.RADIUS * torch.tensor(
            [math.cos(angle), math.sin(angle)])

    score_rng = torch.Generator().manual_seed(11)
    score_indices = torch.randint(len(base), (768,), generator=score_rng)
    score_noise = .029 * torch.randn(768, 2, generator=score_rng)
    eval_rng = torch.Generator().manual_seed(9)
    eval_indices = torch.randint(len(base), (mode_hold.EVAL_N,), generator=eval_rng)
    eval_noise = .029 * torch.randn(mode_hold.EVAL_N, 2, generator=eval_rng)
    rows = {}
    for name, support in (("base", base), ("moved", moved)):
        generated = support[score_indices] + score_noise
        quality = mode_hold.diversity(support[eval_indices] + eval_noise, means,
                                      detailed=True)
        rows[name] = dict(energy_score=score(real, generated),
                          modes=quality["modes"], hq=quality["hq"],
                          hq_counts=quality["hq_counts"])
    if not (rows["moved"]["energy_score"] < rows["base"]["energy_score"]
            and rows["base"]["hq"] >= mode_hold.PASS_HQ
            and rows["moved"]["hq"] < mode_hold.PASS_HQ
            and rows["base"]["modes"] == rows["moved"]["modes"] == 8):
        raise RuntimeError("declared energy/HQ conflict did not reproduce")

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        branch = json.load(archive.extractfile("forks/energy_backtrack.json"))
    checkpoints = {point["step"]: point for point in branch["diagnostic"]}
    trust = branch["dynamics_receipt"]["rows"]
    warm = {str(step): dict(hq=checkpoints[step]["hq"],
                            modes=checkpoints[step]["modes"],
                            accepted_scale=trust[step - 1001]["accepted_scale"],
                            before=trust[step - 1001]["before"],
                            after=trust[step - 1001]["after"])
            for step in (1173, 1174, 1175)}
    if not (warm["1174"]["accepted_scale"] == 1.
            and all(after < before for before, after in
                    zip(warm["1174"]["before"], warm["1174"]["after"]))
            and warm["1173"]["hq"] >= mode_hold.PASS_HQ
            and warm["1174"]["hq"] < mode_hold.PASS_HQ):
        raise RuntimeError("archived warm failure trace changed")
    source = Path(__file__)
    return dict(scope="objective_conflict_diagnostic_only",
                fixed_streams=dict(real=0, score=11, evaluation=9),
                real_samples=8192, real_per_mode=1024, score_samples=768,
                evaluation_samples=mode_hold.EVAL_N,
                real_sigma=mode_hold.SIGMA, output_noise_std=.029,
                radius=mode_hold.RADIUS, occupancy=occupancy,
                moved_particles=[dict(index=4, angle=4.0),
                                 dict(index=6, angle=4.8)],
                constructed=rows,
                energy_change=rows["moved"]["energy_score"] - rows["base"]["energy_score"],
                warm_update_trace=warm,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                host_source_sha256=hashlib.sha256(
                    Path(mode_hold.__file__).read_bytes()).hexdigest(),
                warm_archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                torch=str(torch.__version__),
                cpu_capability=torch.backends.cpu.get_cpu_capability())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(json.dumps(dict(energy_change=result["energy_change"],
                          constructed=result["constructed"],
                          warm_1174=result["warm_update_trace"]["1174"])),
          flush=True)


if __name__ == "__main__":
    main()
