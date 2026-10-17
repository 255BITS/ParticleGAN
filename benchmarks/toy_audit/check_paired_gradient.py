"""A zero-update counterexample to coincident-particle gradient symmetry.

The original two-pole host pairs each fake logit with a different real logit.
Equal fake coordinates therefore need not have equal generator gradients.
This software probe establishes no full-budget convergence or reachability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import torch

from benchmarks.legacy.locked_shared import LOCKED_SHARED, make_gan_loss
from benchmarks.locked_shared.two_pole import HostCritic, real_batch


def counterexample():
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        critic, gan = HostCritic(), make_gan_loss()
        n = LOCKED_SHARED.n_particles
        real = critic(real_batch(n)).detach()

        def gradient(real_logits):
            particles = torch.zeros(n, 1, requires_grad=True)
            fake = critic(particles)
            loss = gan.g_loss(fake, real_logits)
            loss = loss + LOCKED_SHARED.particle_l2 * particles.square().mean()
            return torch.autograd.grad(loss, particles)[0], fake.detach()

        paired, fake = gradient(real)
        constant, _ = gradient(real.mean().expand_as(real))
        permuted, _ = gradient(real.flip(0))
        result = dict(particles=n, fake_logits_all_equal=bool((fake == fake[0]).all()),
                      real_logits_min=float(real.min()), real_logits_max=float(real.max()),
                      paired_gradient_min=float(paired.min()), paired_gradient_max=float(paired.max()),
                      paired_gradient_range=float(paired.max() - paired.min()),
                      paired_gradients_all_equal=bool((paired == paired[0]).all()),
                      constant_real_gradient_range=float(constant.max() - constant.min()),
                      reversed_pair_gradient_max_error=float((permuted - paired.flip(0)).abs().max()))
    result["caller_global_rng_preserved"] = bool(torch.equal(rng_before, torch.get_rng_state()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = counterexample()
    root = Path(__file__).resolve().parents[2]
    receipt = dict(format="paired_gradient_counterexample_v1", software_updates=0,
                   training_campaigns=0, fixture="original HostCritic before any optimizer update",
                   conclusion="Coincident fake coordinates do not imply coincident gradients under the original paired RpGAN loss.",
                   limitation="This does not prove the unchanged host can meet any new density gate within its budget.",
                   result=result, runtime=dict(python=platform.python_version(), torch=torch.__version__, threads=1),
                   source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                  ("benchmarks/legacy/locked_shared.py", "benchmarks/legacy/gan_loss.py",
                                   "benchmarks/locked_shared/two_pole.py", "benchmarks/toy_audit/check_paired_gradient.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
