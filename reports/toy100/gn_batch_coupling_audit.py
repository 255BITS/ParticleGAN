"""Read-only, fixed-seed audit of the frozen batch-dependent vector critic."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from particlegan import BatchDistanceDiscriminator


def main() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    critic = BatchDistanceDiscriminator()
    batch = torch.randn(128, 2, requires_grad=True)
    scores = critic(batch)
    # Warm the autograd path before timing either derivative strategy.
    torch.autograd.grad(scores.sum(), batch, create_graph=True, retain_graph=True)
    start = time.perf_counter()
    summed = torch.autograd.grad(scores.sum(), batch, create_graph=True,
                                 retain_graph=True)[0]
    summed_seconds = time.perf_counter() - start
    start = time.perf_counter()
    diagonal = torch.stack([
        torch.autograd.grad(scores[index], batch, create_graph=True,
                            retain_graph=True)[0][index]
        for index in range(len(batch))
    ])
    exact_seconds = time.perf_counter() - start
    root = Path(__file__).resolve().parents[2]
    return dict(
        seed=0, batch_shape=list(batch.shape), cpu_threads=1,
        summed_gradient_seconds=summed_seconds,
        exact_diagonal_seconds=exact_seconds,
        max_abs_difference=float((diagonal - summed).detach().abs().max()),
        mean_abs_difference=float((diagonal - summed).detach().abs().mean()),
        critic_source_sha256=hashlib.sha256(
            (root / "particlegan/discriminators.py").read_bytes(),
        ).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=root, text=True).strip(),
        implication="sum of batch logits has a different input gradient from each row's own logit",
    )


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
