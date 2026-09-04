"""
Generator / discriminator for the sparse conditional mixed-output toy.

The surrounding recipe (RpGAN logistic + one-sided cap penalty, Fourier
features on D, EMA read-out, particle prior) is the 100-Gaussians champion,
unchanged. Only the heads differ, and each head is a config switch so that a
study can move exactly one thing at a time:

Generator  G(z, c) -> (x in R^d, y in simplex^K, logits l in R^K)
  * class enters as a learned embedding concatenated with z,
  * `real_head`:
      - 'linear': x = W h                       (v0; expected to smear)
      - 'gated':  x = m * v, m = STE(sigmoid(g) > 0.5)   exact zeros from the
                  forward threshold; a sparsity penalty on mean(sigmoid(g))
                  is available to the trainer via `gate_prob`
      - 'topk':   x = m * v, m = STE(top-k of g)          exact k-sparsity by
                  construction (uses the true k: a structural oracle, kept
                  as the ceiling for the gated head)
  * `cat_mode` picks how the categorical symbol reaches D at train time:
      - 'gumbel_st':   hard one-hot forward, Gumbel-softmax backward
      - 'gumbel_soft': soft Gumbel-softmax sample (D can see it is not one-hot)
      - 'st_argmax':   hard argmax forward, softmax backward (no Gumbel noise)
      - 'soft':        plain softmax probabilities
    At eval time (`G.eval()`) every mode emits the hard argmax one-hot.

Discriminator  D(x, y[, c]) — the symbol y is part of the *sample*, so D
always sees it; whether D sees the *condition* c is the UCD question:
  * 'scalar': one logit, c never enters D. The floor: G gets no signal that
              could tie x to c.
  * 'concat': one logit, one-hot(c) concatenated to the input (the classic
              condition injection UCD argues against).
  * 'proj':   projection discriminator, logit = psi(h) + <h, E[c]>.
  * 'ucd':    R^C output, no c in the input (arXiv:2510.00624). The
              adversarial logit for a sample of class c is out[:, c]; the
              trainer adds lambda_1 * CE(out, c) on reals and fakes.

`JointCritic` adapts D to `lib.grad_regularizers.GradRegularizer`, which
expects a module mapping one tensor to one scalar per sample: it takes the
concatenation [x | y] and a fixed class vector.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, n_hidden: int) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(dim, hidden), nn.LeakyReLU(0.2, inplace=True)]
        dim = hidden
    return nn.Sequential(*layers)


def init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class FourierFeatures(nn.Module):
    """[x, sin(pi 2^i x), cos(pi 2^i x)] for i < n_freq, per input dim (as in the 100-Gaussians D)."""

    def __init__(self, in_dim: int, n_freq: int) -> None:
        super().__init__()
        self.n_freq = int(n_freq)
        # Multiplier on the sin/cos terms (the raw x always passes through).
        # The trainer can ramp it 0 -> 1 for a coarse-to-fine D schedule.
        self.scale = 1.0
        self.out_dim = in_dim * (1 + 2 * self.n_freq) if self.n_freq > 0 else in_dim
        if self.n_freq > 0:
            self.register_buffer("freqs", torch.pi * (2.0 ** torch.arange(self.n_freq, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq == 0:
            return x
        xf = x.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], dim=1)
        if self.scale != 1.0:
            feats = feats * self.scale
        return torch.cat([x, feats], dim=1)


class SparseCondGenerator(nn.Module):
    REAL_HEADS = ("linear", "gated", "topk")
    CAT_MODES = ("gumbel_st", "gumbel_soft", "st_argmax", "soft")

    def __init__(
        self,
        z_dim: int,
        n_classes: int,
        d: int,
        n_symbols: int,
        k: int,
        hidden: int = 128,
        n_hidden: int = 3,
        emb_dim: int = 8,
        real_head: str = "linear",
        cat_mode: str = "gumbel_st",
    ) -> None:
        super().__init__()
        if real_head not in self.REAL_HEADS:
            raise ValueError(f"real_head must be one of {self.REAL_HEADS}, got {real_head!r}")
        if cat_mode not in self.CAT_MODES:
            raise ValueError(f"cat_mode must be one of {self.CAT_MODES}, got {cat_mode!r}")
        self.d, self.k, self.n_symbols = int(d), int(k), int(n_symbols)
        self.real_head, self.cat_mode = real_head, cat_mode
        # emb_dim = 0 makes G unconditional (c is ignored): the control for
        # "can this recipe cover the sparse modes at all", conditioning aside.
        self.emb = nn.Embedding(n_classes, emb_dim) if emb_dim > 0 else None
        self.trunk = mlp(z_dim + emb_dim, hidden, n_hidden)
        self.real = nn.Linear(hidden, d if real_head == "linear" else 2 * d)
        # Trainer-controlled switch for a gate warm-up: while False, the gated
        # / topk heads emit the ungated values (mask == 1, no STE), so the run
        # can find the modes first and only then start pruning dims.
        self.gate_on = True
        self.cat = nn.Linear(hidden, n_symbols)
        init_linear(self)

    def forward(self, z: torch.Tensor, c: torch.Tensor, tau: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.trunk(z if self.emb is None else torch.cat([z, self.emb(c)], dim=1))

        # ---- real head ----
        r = self.real(h)
        gate_prob: Optional[torch.Tensor] = None
        if self.real_head == "linear":
            x = r
        elif not self.gate_on:
            x = r[:, : self.d]
            gate_prob = torch.sigmoid(r[:, self.d :])
        else:
            v, g = r[:, : self.d], r[:, self.d :]
            p = torch.sigmoid(g)
            if self.real_head == "gated":
                hard = (p > 0.5).to(p.dtype)
            else:  # topk
                hard = torch.zeros_like(p).scatter_(1, g.topk(self.k, dim=1).indices, 1.0)
            m = hard + p - p.detach()  # straight-through: hard forward, sigmoid backward
            x = m * v
            gate_prob = p

        # ---- categorical head ----
        logits = self.cat(h)
        if not self.training:
            y = F.one_hot(logits.argmax(dim=1), self.n_symbols).to(logits.dtype)
        elif self.cat_mode == "gumbel_st":
            y = F.gumbel_softmax(logits, tau=tau, hard=True)
        elif self.cat_mode == "gumbel_soft":
            y = F.gumbel_softmax(logits, tau=tau, hard=False)
        elif self.cat_mode == "st_argmax":
            p_soft = F.softmax(logits / tau, dim=1)
            hard = F.one_hot(logits.argmax(dim=1), self.n_symbols).to(logits.dtype)
            y = hard + p_soft - p_soft.detach()
        else:  # soft
            y = F.softmax(logits / tau, dim=1)

        out = {"x": x, "y": y, "logits": logits}
        if gate_prob is not None:
            out["gate_prob"] = gate_prob
        return out


class SparseJointDiscriminator(nn.Module):
    D_MODES = ("scalar", "concat", "proj", "ucd")

    def __init__(
        self,
        d: int,
        n_symbols: int,
        n_classes: int,
        hidden: int = 128,
        n_hidden: int = 3,
        fourier: int = 2,
        d_mode: str = "ucd",
    ) -> None:
        super().__init__()
        if d_mode not in self.D_MODES:
            raise ValueError(f"d_mode must be one of {self.D_MODES}, got {d_mode!r}")
        self.d, self.n_symbols, self.n_classes, self.d_mode = int(d), int(n_symbols), int(n_classes), d_mode
        self.fourier = FourierFeatures(d, fourier)
        in_dim = self.fourier.out_dim + n_symbols + (n_classes if d_mode == "concat" else 0)
        self.trunk = mlp(in_dim, hidden, n_hidden)
        self.head = nn.Linear(hidden, n_classes if d_mode == "ucd" else 1)
        if d_mode == "proj":
            self.proj = nn.Embedding(n_classes, hidden)
            nn.init.normal_(self.proj.weight, std=0.02)
        init_linear(self.trunk)
        init_linear(self.head)

    def features(self, x: torch.Tensor, y: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        parts = [self.fourier(x), y]
        if self.d_mode == "concat":
            if c is None:
                raise ValueError("d_mode='concat' needs c")
            parts.append(F.one_hot(c, self.n_classes).to(x.dtype))
        return self.trunk(torch.cat(parts, dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Returns {'adv': (B,) adversarial logit for the given class c,
                 'class_logits': (B, C) (ucd only)}.
        For 'scalar' the class is ignored entirely.
        """
        h = self.features(x, y, c)
        out = self.head(h)
        if self.d_mode == "ucd":
            if c is None:
                raise ValueError("d_mode='ucd' needs c to select the adversarial component")
            return {"adv": out.gather(1, c.unsqueeze(1)).squeeze(1), "class_logits": out}
        if self.d_mode == "proj":
            if c is None:
                raise ValueError("d_mode='proj' needs c")
            return {"adv": out.squeeze(1) + (h * self.proj(c)).sum(dim=1)}
        return {"adv": out.squeeze(1)}

    @torch.no_grad()
    def class_logits(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        """The R^C output (ucd only) -- the UCD Nash-equilibrium probe."""
        if self.d_mode != "ucd":
            return None
        return self.head(self.features(x, y, None))


class JointCritic(nn.Module):
    """D restricted to one class vector, on the joint input [x | y] -- what GradRegularizer penalizes."""

    def __init__(self, D: SparseJointDiscriminator, c: torch.Tensor) -> None:
        super().__init__()
        self.D = D
        self.c = c

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, : self.D.d], xy[:, self.D.d :]
        return self.D(x, y, self.c)["adv"]


class XOnlyCritic(nn.Module):
    """Same, but the penalty only sees the gradient w.r.t. x (y is held fixed)."""

    def __init__(self, D: SparseJointDiscriminator, y: torch.Tensor, c: torch.Tensor) -> None:
        super().__init__()
        self.D = D
        self.y = y.detach()
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.D(x, self.y, self.c)["adv"]
