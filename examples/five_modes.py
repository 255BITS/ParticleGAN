#!/usr/bin/env python
"""Five-mode text toy: an autoencoder with a joint critic keeps five words apart.

Problem only: an encoder E, a generator G, one learnable particle per word
(apple / grape / lemon / melon / berry) in a 2-D latent, and a BiGAN-style
joint critic D(x, z) that scores (text, E(text)) against (G(z), z). Everything
else -- the recipe-built optimizers and their LR schedule, the RpGAN loss, the
K3P critic penalty, critic input / generator output noise, EMA, logging --
comes from the shipped recipe through ``benchmarks.toy_runner``::

    python -m examples.five_modes --log runs/toy-refactor/example_five_modes.log

The critic is presented to the runner as one module over the flat joint
vector ``[x, z]``, so the penalty (and input noise) act on the whole pair, the
BiGAN analogue of a penalty on grad_x D(x). PASS means the EMA autoencoder
reconstructs all five words and the five particles decode to five distinct
words (the 1-to-1 mapping the toy is about).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python examples/five_modes.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from particlegan import get_recipe, init  # noqa: E402
from benchmarks.toy_runner import Networks, Sample, ToyProblem, View, main, run  # noqa: E402

WORDS = ['apple', 'grape', 'lemon', 'melon', 'berry']
TYPO = 'aple'
CHARS = "abcdefghijklmnopqrstuvwxyz_ "
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
SEQ_LEN = 6
X_DIM = len(CHARS) * SEQ_LEN
Z_DIM = 2  # five modes need no overcomplete latent
BATCH = 256
STEPS = 20_000
EVAL_N = 2048


def str_to_tensor(text_list):
    indices = [[CHAR_IDX.get(c, 26) for c in text.ljust(SEQ_LEN, '_')[:SEQ_LEN]] for text in text_list]
    return F.one_hot(torch.tensor(indices), num_classes=len(CHARS)).permute(0, 2, 1).float()


def tensor_to_str(tensor_logits):
    rows = torch.argmax(tensor_logits, dim=1).cpu().tolist()
    return ["".join(CHARS[i] for i in row).replace('_', '').strip() for row in rows]


def _mlp(*widths):
    layers = []
    for i, o in zip(widths[:-1], widths[1:]):
        layers += [nn.Linear(i, o), nn.LeakyReLU(0.2)]
    return nn.Sequential(*layers[:-1])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), _mlp(X_DIM, 128, 64, Z_DIM))

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _mlp(Z_DIM, 64, 128, X_DIM)

    def forward(self, z):
        return self.net(z).view(-1, len(CHARS), SEQ_LEN)


class JointCritic(nn.Module):
    """D over the flat joint vector ``[x_flat, z]`` (see ``join_pair``)."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(X_DIM + Z_DIM, 256, 128, 1)

    def forward(self, joint):
        return self.net(joint)


def join_pair(x, z):
    return torch.cat([x.flatten(1), z], dim=1)


class FiveModes(ToyProblem):
    name = "example_five_modes"

    def __init__(self):
        self.vocab = str_to_tensor(WORDS)

    def recipe(self):
        return get_recipe(z_dim=Z_DIM, num_particles=len(WORDS), batch_size=BATCH, total_steps=STEPS)

    def networks(self, recipe, seed):
        return Networks(
            generator=init.deterministic_orthogonal_(Generator(), seed=seed),
            critics=init.deterministic_orthogonal_(JointCritic(), seed=seed + 1),
            encoder=init.deterministic_orthogonal_(Encoder(), seed=seed + 2),
            prior=init.deterministic_orthogonal_(recipe.make_prior()),
        )

    def real(self, n, stream):
        idx = torch.randint(len(WORDS), (n,), generator=stream, device=stream.device)
        return self.vocab.to(stream.device)[idx]

    def fake(self, nets, n, stream, real):
        z, idx = nets.prior.sample(n, generator=stream)
        return Sample(F.softmax(nets.generator(z), dim=1), condition=(z,), indices=idx)

    def views(self, nets, real, fake):
        # The pairing: (text, E(text)) vs (G(z), z) under one joint critic.
        return [View("critic", join_pair(real.x, nets.encoder(real.x)), join_pair(fake.x, fake.condition[0]))]

    def metrics(self, model):
        nets = model.nets
        device = next(nets.generator.parameters()).device
        vocab = self.vocab.to(device)
        recon = tensor_to_str(nets.generator(nets.encoder(vocab)))
        typo = tensor_to_str(nets.generator(nets.encoder(str_to_tensor([TYPO]).to(device))))[0]
        particle_words = tensor_to_str(nets.generator(nets.prior.z))
        samples = tensor_to_str(model.sample(EVAL_N).x)
        return {
            "recon_acc": sum(w == r for w, r in zip(WORDS, recon)),
            "particle_words": len(set(particle_words) & set(WORDS)),
            "sample_valid": sum(s in WORDS for s in samples) / len(samples),
            "sample_modes": len(set(samples) & set(WORDS)),
            "typo_to_apple": int(typo == 'apple'),
        }

    def verdict(self, metrics):
        n = len(WORDS)
        return "PASS" if metrics["recon_acc"] == n and metrics["particle_words"] == n else "FAIL"


def train(**run_kwargs) -> dict:
    """Train on the shared runner; ``run_kwargs`` go to ``benchmarks.toy_runner.run``."""
    return run(FiveModes(), **run_kwargs)


if __name__ == "__main__":
    raise SystemExit(main(FiveModes()))
