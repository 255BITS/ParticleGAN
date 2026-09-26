"""MisGAN toy: the 100-Gaussian grid lifted to 8D, missingness mechanisms,
closed-form imputation baselines (mean, kNN, Bayes-optimal) and metrics.

Data: x2 ~ 100-Gaussian grid (``sample_100gaussians``, sigma 0.03), lifted by a
fixed random orthonormal A (8x2): x_raw = A x2 + eta * e, eta = 0.01 * sigma.
Training and evaluation work in standardized coordinates (per-coordinate mean
and std of the *observed* training entries), where the fill value tau = 0 is the
observed mean. ``to2d`` maps back with A^T for the grid metrics.

Masks use 1 = observed. Every mechanism is independent of x (MCAR), as MisGAN
assumes. Nothing here consumes the global RNG.
"""
import math

import torch

from lib.mog_metrics import sample_metrics
from lib.toy_metrics import sliced_w1
from lib.toy_models import sample_100gaussians

DIM = 8
MODE_STD = 0.03
LIFT_NOISE = 0.01 * MODE_STD
BITS = 2 ** torch.arange(DIM)
# Four sensor-group dropout patterns (observed coordinates) with unequal
# probabilities. P0-P2 jointly observe every coordinate pair; P3 keeps one
# sensor, so 10% of block rows are ambiguous (a single linear view of x2).
BLOCK_PATTERNS = (((0, 1, 2, 3, 4), 0.4), ((3, 4, 5, 6, 7), 0.3),
                  ((0, 1, 2, 5, 6, 7), 0.2), ((7,), 0.1))
MECHANISMS = ("mcar_p20", "mcar_p50", "mcar_p80", "block")


def grid_centers(device=None, dtype=torch.float32):
    coords = torch.arange(10, device=device, dtype=dtype) - 4.5
    return torch.cartesian_prod(coords, coords)  # same order as lib.toy_metrics


def _block_masks():
    masks = torch.zeros(len(BLOCK_PATTERNS), DIM)
    for row, (observed, _) in enumerate(BLOCK_PATTERNS):
        masks[row, list(observed)] = 1
    return masks, torch.tensor([p for _, p in BLOCK_PATTERNS])


def pattern_probs(mechanism):
    """Exact probability of each of the 2^8 masks (index = sum m_c 2^c)."""
    all_masks = ((torch.arange(2 ** DIM)[:, None] // BITS) % 2).double()
    if mechanism.startswith("mcar_p"):
        p = int(mechanism[6:]) / 100
        return (all_masks * (1 - p) + (1 - all_masks) * p).prod(1)
    if mechanism == "block":
        masks, probs = _block_masks()
        out = torch.zeros(2 ** DIM, dtype=torch.float64)
        out[(masks.long() * BITS).sum(1)] = probs.double()
        return out
    raise ValueError(f"unknown mechanism {mechanism!r}; choose {MECHANISMS}")


def sample_masks(mechanism, n, generator, device="cpu"):
    if mechanism.startswith("mcar_p"):
        p = int(mechanism[6:]) / 100
        return (torch.rand(n, DIM, generator=generator) >= p).float().to(device)
    if mechanism == "block":
        masks, probs = _block_masks()
        return masks[torch.multinomial(probs, n, replacement=True, generator=generator)].to(device)
    raise ValueError(f"unknown mechanism {mechanism!r}; choose {MECHANISMS}")


def pattern_index(masks):
    return ((masks > 0.5).long() * BITS.to(masks.device)).sum(-1)


class Problem:
    """Fixed incomplete training pool plus a clean, masked test set (CPU generation)."""

    def __init__(self, mechanism, n_train=20_000, n_test=10_000, seed=0, device="cpu"):
        self.mechanism, self.device = mechanism, torch.device(device)
        rng = torch.Generator().manual_seed(seed)
        self.A = torch.linalg.qr(torch.randn(DIM, 2, generator=rng, dtype=torch.float64))[0]

        def draw(n):
            x2 = sample_100gaussians(n, "cpu", generator=rng, std=MODE_STD).double()
            raw = x2 @ self.A.T + LIFT_NOISE * torch.randn(n, DIM, generator=rng, dtype=torch.float64)
            return raw, torch.cdist(x2, grid_centers(dtype=torch.float64)).argmin(1)

        raw_train, _ = draw(n_train)
        m_train = sample_masks(mechanism, n_train, rng).double()
        raw_test, self.y_test = draw(n_test)
        m_test = sample_masks(mechanism, n_test, rng).double()
        raw_ref, _ = draw(n_test)  # an independent clean sample: the sliced-W1 floor
        count = m_train.sum(0).clamp_min(1)
        self.mean = (raw_train * m_train).sum(0) / count
        self.std = (((raw_train - self.mean) ** 2 * m_train).sum(0) / count).sqrt()
        f32 = dict(device=self.device, dtype=torch.float32)
        self.x_train_full = self.standardize(raw_train).to(**f32)  # oracle only
        self.m_train = m_train.to(**f32)
        self.x_train = self.x_train_full * self.m_train  # f_0: tau = 0
        self.x_test = self.standardize(raw_test).to(**f32)
        self.m_test = m_test.to(**f32)
        self.x_ref = self.standardize(raw_ref).to(**f32)
        self.y_test = self.y_test.to(self.device)
        self.probs = pattern_probs(mechanism)
        self.missing_rate = (self.probs[:, None] * (1 - (torch.arange(2 ** DIM)[:, None] // BITS) % 2)).sum(0)

    def standardize(self, raw):
        return (raw - self.mean.to(raw)) / self.std.to(raw)

    def raw(self, x):
        return x.double() * self.std.to(x.device) + self.mean.to(x.device)

    def to2d(self, x):
        return (self.raw(x) @ self.A.to(x.device)).float()

    def off_plane(self, x):
        raw, A = self.raw(x), self.A.to(x.device)  # the data plane passes through 0
        return float((raw - raw @ A @ A.T).norm(dim=1).mean())

    def train_batch(self, n, generator):
        idx = torch.randint(len(self.m_train), (n,), device=self.device, generator=generator)
        return self.x_train[idx], self.m_train[idx], self.x_train_full[idx]


# ---------------------------------------------------------------- baselines

@torch.no_grad()
def bayes_posterior(problem, x, m, draws=0, generator=None):
    """Exact posterior over the 100 modes given the observed coordinates.

    Returns ``(probs (n, 100), samples (draws, n, 8) or None)``; samples are
    exact draws of the missing coordinates from p(x_M | x_O), in standardized
    units, with observed coordinates kept.
    """
    dev = x.device
    A = problem.A.to(dev)
    mu = grid_centers(dev, torch.float64)
    raw = problem.raw(x)
    s2, e2 = MODE_STD ** 2, LIFT_NOISE ** 2
    probs = torch.empty(len(x), 100, dtype=torch.float64, device=dev)
    out = None if not draws else raw.expand(draws, -1, -1).clone()
    keys = pattern_index(m)
    for key in keys.unique().tolist():
        rows = (keys == key).nonzero().squeeze(1)
        obs = [c for c in range(DIM) if key >> c & 1]
        miss = [c for c in range(DIM) if not key >> c & 1]
        if obs:
            Ao, y = A[obs], raw[rows][:, obs]
            S = s2 * Ao @ Ao.T + e2 * torch.eye(len(obs), dtype=torch.float64, device=dev)
            L = torch.linalg.cholesky(S)
            r = y[:, None, :] - (mu @ Ao.T)[None]  # (n, 100, |O|)
            z = torch.linalg.solve_triangular(L, r.reshape(-1, len(obs)).T, upper=False).T
            logw = -0.5 * z.pow(2).sum(1).reshape(len(rows), 100)
            cov = torch.linalg.inv(torch.eye(2, dtype=torch.float64, device=dev) / s2 + Ao.T @ Ao / e2)
        else:
            logw = torch.zeros(len(rows), 100, dtype=torch.float64, device=dev)
            cov = s2 * torch.eye(2, dtype=torch.float64, device=dev)
        p = logw.softmax(1)
        probs[rows] = p
        if draws and miss:
            k = torch.multinomial(p.float(), draws, replacement=True, generator=generator).T  # (draws, n)
            mean = mu[k] / s2  # (draws, n, 2)
            if obs:
                mean = mean + (y @ Ao / e2)[None]
            mean = mean @ cov.T
            chol = torch.linalg.cholesky(cov)
            eps = torch.randn(draws, len(rows), 2, generator=generator, device=dev, dtype=torch.float64)
            x2 = mean + eps @ chol.T
            noise = torch.randn(draws, len(rows), len(miss), generator=generator, device=dev, dtype=torch.float64)
            vals = x2 @ A[miss].T + LIFT_NOISE * noise
            block = out[:, rows]
            block[:, :, miss] = vals
            out[:, rows] = block
    if draws:
        out = problem.standardize(out).float()
        out = m * x + (1 - m) * out
    return probs.float(), out


@torch.no_grad()
def knn_impute(problem, x, m, k=5, chunk=1024):
    """kNN on the observed coordinates against the incomplete training pool.

    Distance is the RMS difference over coordinates observed in both rows;
    each missing coordinate is the mean of the k nearest rows that observe it.
    Rows with no usable neighbour fall back to the mean (0).
    """
    xt, mt = problem.x_train, problem.m_train
    out = []
    for xb, mb in zip(x.split(chunk), m.split(chunk)):
        a = xb * mb
        overlap = mb @ mt.T
        sq = (a * xb) @ mt.T + mb @ (xt * xt * mt).T - 2 * a @ (xt * mt).T
        dist = sq.clamp_min(0) / overlap.clamp_min(1)
        dist[overlap == 0] = float("inf")
        fill = torch.zeros_like(xb)
        for c in range(DIM):
            d = dist.masked_fill(mt[:, c][None] == 0, float("inf"))
            val, idx = d.topk(k, dim=1, largest=False)
            ok = torch.isfinite(val)
            num = (xt[idx, c] * ok).sum(1)
            fill[:, c] = num / ok.sum(1).clamp_min(1)
        out.append(mb * xb + (1 - mb) * fill)
    return torch.cat(out)[None]


def mean_impute(problem, x, m):
    return (x * m)[None]  # standardized: the observed mean is 0


# ------------------------------------------------------------------ metrics

@torch.no_grad()
def generation_metrics(problem, x):
    """Modes /100 and % within 3 sigma (2D, via A^T), sliced W1 to clean test data (8D)."""
    stats, _, _ = sample_metrics(problem.to2d(x))
    ref = problem.x_test[: len(x)]
    return dict(modes=stats["modes"], hq=100 * stats["hq"],
                swd=sliced_w1(x[: len(ref)], ref, n_proj=256, seed=7),
                off=problem.off_plane(x))


@torch.no_grad()
def mask_metrics(problem, m):
    """Per-coordinate missing-rate MAE, TV over the 256 patterns and over the
    observed-count histogram (vs the exact mechanism), and mask softness."""
    hard = (m > 0.5).float()
    rate = 1 - hard.mean(0).double().cpu()
    emp = torch.bincount(pattern_index(hard).cpu(), minlength=2 ** DIM).double() / len(m)
    counts = (torch.arange(2 ** DIM)[:, None] // BITS % 2).sum(1)
    hist = lambda p: torch.zeros(DIM + 1, dtype=torch.float64).index_add_(0, counts, p)  # noqa: E731
    return dict(m_mae=float((rate - problem.missing_rate).abs().mean()),
                m_tv=float(0.5 * (emp - problem.probs).abs().sum()),
                m_tvk=float(0.5 * (hist(emp) - hist(problem.probs)).abs().sum()),
                m_soft=float(torch.minimum(m, 1 - m).mean()))


@torch.no_grad()
def imputation_metrics(problem, draws, post, m=None):
    """``draws``: (K, n, 8) imputations of the test rows; ``post``: Bayes mode
    posterior (n, 100). Mode accuracy (averaged over draws), accuracy on
    ambiguous rows (<= 1 observed coordinate), per-row TV between the modes the
    K draws land in and the posterior, per-row std over draws and RMSE on the
    missing coordinates, and modes / 3-sigma % of the first imputed set."""
    m = problem.m_test if m is None else m
    K, n, _ = draws.shape
    x2 = problem.to2d(draws.reshape(K * n, DIM))
    nearest = torch.cdist(x2, grid_centers(x2.device)).argmin(1).reshape(K, n)
    hit = (nearest == problem.y_test[None]).float()
    lo = m.sum(1) <= 1
    hist = torch.zeros(n, 100, device=draws.device)
    hist.scatter_add_(1, nearest.T, torch.full((n, K), 1.0 / K, device=draws.device))
    miss = 1 - m
    nmiss = miss.sum().clamp_min(1)
    std = draws.std(0, correction=0) if K > 1 else torch.zeros_like(draws[0])
    err = (draws - problem.x_test[None]) ** 2
    stats, _, _ = sample_metrics(x2[:n])
    return dict(acc=float(hit.mean()), acc_lo=float(hit[:, lo].mean()) if lo.any() else float("nan"),
                itv=float(0.5 * (hist - post).abs().sum(1).mean()),
                istd=float((std * miss).sum() / nmiss),
                rmse=math.sqrt(float((err * miss[None]).sum() / (nmiss * K))),
                imodes=stats["modes"], ihq=100 * stats["hq"])


def baselines(problem, draws=16, seed=11):
    """Imputation metrics of the untrained baselines on the fixed test rows."""
    gen = torch.Generator(device=problem.device).manual_seed(seed)
    post, bayes = bayes_posterior(problem, problem.x_test, problem.m_test, draws, gen)
    out = {"bayes": imputation_metrics(problem, bayes, post),
           "knn": imputation_metrics(problem, knn_impute(problem, problem.x_test, problem.m_test), post),
           "mean": imputation_metrics(problem, mean_impute(problem, problem.x_test, problem.m_test), post)}
    # MAP mode (deterministic ceiling on accuracy) and the sliced-W1 floor.
    out["bayes"]["acc_map"] = float((post.argmax(1) == problem.y_test).float().mean())
    out["floor"] = generation_metrics(problem, problem.x_ref)
    return out
