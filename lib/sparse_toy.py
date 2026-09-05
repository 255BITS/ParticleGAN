"""
Sparse conditional toy with mixed real + categorical output.

Implements the generative process of
`reports/sparse-conditional-with-categorical-and-real-output.md` §1:

  * `n_modes` modes in R^d, each supported on exactly `k` of the `d` dims.
    Active dims take a value from a small alphabet ({-2, -1, 1, 2}) plus
    N(0, sigma^2) noise; inactive dims are **exactly 0**. That exact zero is
    the whole point of the toy: a generator that smears 0.05 of mass onto the
    inactive dims looks fine to the eye and to W1, but is wrong.
  * every mode belongs to one of `n_classes` conditioning classes
    (`c(m) = m mod C`, balanced by construction),
  * every mode emits one of `n_symbols` categorical symbols. `symbol_map`
    picks the map: 'identity' (s(m) = c(m), K = C, the v1 deterministic
    case) or 'split' (s(m) = 2 c(m) + parity of m // C, K = 2C, so each
    class has a 50/50 distribution over two symbols and "right distribution
    over symbols" is distinguishable from "right single symbol").

All randomness goes through explicit torch.Generators. The mode table is built
on the CPU from `seed` alone, so the *same* problem instance is trained on
every device and by every run that shares `seed`, regardless of the training
seed.
"""

from typing import Optional, Tuple

import torch


class SparseMixedToy:
    """The problem instance: mode table + samplers + nearest-mode assignment."""

    SYMBOL_MAPS = ("identity", "split")

    def __init__(
        self,
        d: int = 24,
        k: int = 3,
        n_modes: int = 64,
        n_classes: int = 8,
        n_symbols: int = 8,
        sigma: float = 0.05,
        alphabet: Tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0),
        symbol_map: str = "identity",
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if k > d:
            raise ValueError(f"k={k} must be <= d={d}")
        if n_modes % n_classes != 0:
            raise ValueError(f"n_modes={n_modes} must be a multiple of n_classes={n_classes}")
        if symbol_map not in self.SYMBOL_MAPS:
            raise ValueError(f"symbol_map must be one of {self.SYMBOL_MAPS}, got {symbol_map!r}")
        if symbol_map == "identity" and n_symbols != n_classes:
            raise ValueError("symbol_map='identity' needs n_symbols == n_classes")
        if symbol_map == "split" and n_symbols != 2 * n_classes:
            raise ValueError("symbol_map='split' needs n_symbols == 2 * n_classes")

        self.d, self.k = int(d), int(k)
        self.n_modes, self.n_classes, self.n_symbols = int(n_modes), int(n_classes), int(n_symbols)
        self.sigma = float(sigma)
        self.alphabet = tuple(float(a) for a in alphabet)
        self.symbol_map = symbol_map
        self.seed = int(seed)
        self.device = device if device is not None else torch.device("cpu")

        gen = torch.Generator().manual_seed(self.seed)
        alpha = torch.tensor(self.alphabet, dtype=torch.float32)
        # Reseed until every dim is used by at least one mode and all centers
        # are distinct (the report's construction check). Both failures are
        # rare at the defaults; the loop is cheap insurance.
        for _attempt in range(1000):
            supports = torch.stack([torch.randperm(self.d, generator=gen)[: self.k] for _ in range(self.n_modes)])
            values = alpha[torch.randint(0, len(self.alphabet), (self.n_modes, self.k), generator=gen)]
            centers = torch.zeros(self.n_modes, self.d)
            centers.scatter_(1, supports, values)
            active = torch.zeros(self.n_modes, self.d, dtype=torch.bool)
            active.scatter_(1, supports, True)
            dims_used = active.any(dim=0).all()
            pd = torch.cdist(centers, centers)
            pd.fill_diagonal_(float("inf"))
            distinct = bool((pd > 0.5).all())
            if dims_used and distinct:
                break
        else:  # pragma: no cover - would need an absurd (d, k, N)
            raise RuntimeError("could not build a valid mode table")

        modes = torch.arange(self.n_modes)
        class_of_mode = modes % self.n_classes
        if symbol_map == "identity":
            symbol_of_mode = class_of_mode.clone()
        else:
            symbol_of_mode = 2 * class_of_mode + (modes // self.n_classes) % 2

        dev = self.device
        self.centers = centers.to(dev)                       # (N, d)
        self.active = active.to(dev)                         # (N, d) bool
        self.class_of_mode = class_of_mode.to(dev)           # (N,)
        self.symbol_of_mode = symbol_of_mode.to(dev)         # (N,)
        # Modes of each class, (C, N/C): class c owns modes c, c+C, c+2C, ...
        self.class_modes = torch.stack(
            [torch.nonzero(class_of_mode == c).flatten() for c in range(self.n_classes)]
        ).to(dev)
        self.min_center_dist = float(pd.min())

        # True conditional p(s | c) as a (C, K) table, from the mode table.
        p_sc = torch.zeros(self.n_classes, self.n_symbols)
        p_sc.index_put_((class_of_mode, symbol_of_mode), torch.ones(self.n_modes), accumulate=True)
        self.p_symbol_given_class = (p_sc / p_sc.sum(dim=1, keepdim=True)).to(dev)

    # -------------------------
    #  Sampling
    # -------------------------

    def _draw(self, m: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        noise = torch.randn(m.shape[0], self.d, device=self.device, generator=generator) * self.sigma
        return self.centers[m] + noise * self.active[m]

    def sample(
        self, n: int, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniform over modes. Returns (x, c, s, m)."""
        m = torch.randint(0, self.n_modes, (n,), device=self.device, generator=generator)
        return self._draw(m, generator), self.class_of_mode[m], self.symbol_of_mode[m], m

    def sample_given_class(
        self, c: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Real samples conditioned on the given classes. Returns (x, s, m)."""
        per = self.class_modes.shape[1]
        j = torch.randint(0, per, (c.shape[0],), device=self.device, generator=generator)
        m = self.class_modes[c, j]
        return self._draw(m, generator), self.symbol_of_mode[m], m

    def symbol_target(self, c: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """A symbol drawn from the true p(s | c) -- the supervised anchor target."""
        if self.symbol_map == "identity":
            return c
        probs = self.p_symbol_given_class[c]
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)

    # -------------------------
    #  Assignment
    # -------------------------

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest mode over all d dims (so smear on inactive dims counts). Returns (m_hat, dist)."""
        dist = torch.cdist(x.to(self.centers.dtype), self.centers)
        d_min, m_hat = dist.min(dim=1)
        return m_hat, d_min

    def describe(self) -> str:
        overlap = self.active.float().sum(dim=0)
        return (
            f"SparseMixedToy(d={self.d}, k={self.k}, modes={self.n_modes}, classes={self.n_classes}, "
            f"symbols={self.n_symbols}, sigma={self.sigma}, symbol_map={self.symbol_map}, seed={self.seed}) "
            f"min_center_dist={self.min_center_dist:.2f} dim_usage[min/max]={int(overlap.min())}/{int(overlap.max())}"
        )
