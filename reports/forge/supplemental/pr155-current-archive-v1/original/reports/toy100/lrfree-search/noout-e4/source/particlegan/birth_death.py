"""Fisher-Rao birth-death moves for a particle table (recipe ``particle_birth_death``).

Spec: reports/design-birthdeath.md (rev 2). Once per full turnover of a FIFO
reservoir R of the last N D-real rows (N = table size), every particle i is
scored at its clean centre q_i = G(z_i) with the kNN log density ratio

    x_i = d log(r_k^R(q_i) / r_k^F(q_i))

against a fake pool F (an iid N-sample of the exact sampling law; see
maybe_apply for the two deviations from the spec). Under local Poisson sampling x_i = log(rho/pi)(q_i) +
logit(Beta(k, k)), sd s_k = sqrt(2 psi'(k)). Evidence S_i accumulates over a
sign-consistent, local excursion; p_i = exp(-z_i^2 / 2) (Rayleigh tail of the
meander endpoint), Benjamini-Hochberg at q over particles with n_i >= 2,
soft-thresholded exponentiated step beta_i, Bernoulli deaths (1 - e^-beta) and
single-clone births (e^-beta - 1), paired by rank; unmatched deaths clone
normalisation parents drawn with weight e^-beta. Row moves copy the parent's
latent (+ the package's latent jitter), EMA latent, Adam/AMSGrad rows and A2
history. All randomness comes from a private stream; no training stream,
parameter or optimizer state is touched except by an executed move.
"""
import math

import torch


def _knn(query, points, k, exclude=None, chunk=2048):
    """Exact k smallest distances (ascending) and indices from each query row
    to ``points``; ``exclude[i]`` (optional) is a point index skipped for query i.
    Shortlist 2k+2 candidates with the matmul cdist, then recompute exactly."""
    m = min(len(points) - (exclude is not None), 2 * k + 2)
    ds, ids = [], []
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        dist = torch.cdist(q, points, compute_mode="use_mm_for_euclid_dist")
        if exclude is not None:
            dist.scatter_(1, exclude[start:start + chunk, None], float("inf"))
        idx = dist.topk(m, dim=1, largest=False).indices
        exact = (q[:, None, :] - points[idx]).norm(dim=2)
        exact, order = exact.sort(1)
        ds.append(exact[:, :k])
        ids.append(idx.gather(1, order)[:, :k])
    return torch.cat(ds), torch.cat(ids)


def _levina_bickel(dist):
    """MacKay-Ghahramani pooled Levina-Bickel MLE from ascending kNN distances [n, k]."""
    k = dist.shape[1]
    ok = dist[:, 0] > 0
    if not bool(ok.any()):
        return float("nan"), 0
    t = dist[ok].double()
    inverse = (t[:, -1:] / t[:, :-1]).log().sum(1) / (k - 1)
    mean = float(inverse.mean())
    return (1. / mean if mean > 0 else float("nan")), int(ok.sum())


def _normal_score(x, k):
    """Exact null probability transform of a kNN log volume ratio: under the null
    sigmoid(x) ~ Beta(k, k), so w = Phi^-1(I_sigmoid(x)(k, k)) ~ N(0, 1) exactly.
    I_u(k, k) = P(Binomial(2k-1, u) >= k), summed in log space on the smaller tail."""
    a = -x.abs()                                   # use the lower tail, restore the sign
    log_u, log_1u = torch.nn.functional.logsigmoid(a), torch.nn.functional.logsigmoid(-a)
    j = torch.arange(k, 2 * k, dtype=torch.float64, device=x.device)
    log_binom = (torch.lgamma(torch.tensor(2. * k, dtype=torch.float64, device=x.device))
                 - torch.lgamma(j + 1) - torch.lgamma(2 * k - j))
    log_cdf = torch.logsumexp(log_binom + j * log_u[:, None] + (2 * k - 1 - j) * log_1u[:, None], dim=1)
    w = torch.special.ndtri(log_cdf.exp().clamp(1e-300, .5))
    return torch.where(x > 0, -w, w)


class ParticleBirthDeath:
    Q = .05          # BH false-discovery rate per evaluation; dimension-test level

    def __init__(self, trainer, seed):
        prior = trainer.prior
        if not prior.z.requires_grad:
            raise ValueError("particle_birth_death needs a trainable particle table")
        self.N = n = prior.z.shape[0]
        self.k = min(math.ceil((math.log2(n / self.Q) + 1) / 2), n - 2)
        if self.k < 4:
            raise ValueError("particle_birth_death needs at least 6 particles")
        self.s_k = math.sqrt(2. * float(torch.polygamma(1, torch.tensor(float(self.k), dtype=torch.float64))))
        self.dry_run = False   # test hook: evaluate, never move
        device, dtype = trainer.device, trainer.dtype
        self.reservoir = None  # [N, D_out], allocated on the first batch
        self.fill = 0
        self.cursor = 0
        self.rows_since_eval = 0
        self.S = torch.zeros(n, device=device, dtype=torch.float64)   # evidence in nats (step size)
        self.W = torch.zeros(n, device=device, dtype=torch.float64)   # evidence in normal scores (test)
        self.n = torch.zeros(n, device=device, dtype=torch.long)
        self.anchor = None     # [N, D_out]
        self.radius = torch.zeros(n, device=device, dtype=dtype)
        self.pending = torch.zeros(n, device=device, dtype=torch.bool)
        self.stream = torch.Generator(device=device).manual_seed(seed)
        self.counters = {k: 0 for k in ("evals", "dim_skips", "discoveries", "realised_deaths",
                                        "realised_births", "stays", "moves", "matched", "normalised",
                                        "waited_deaths", "waited_births", "stale_resets")}
        self.last = {}
        self.moved_rows = None  # transient (not checkpointed): rows moved by the latest maybe_apply

    # --------------------------------------------------------------- reservoir
    @torch.no_grad()
    def observe_real(self, real):
        x = real.detach().flatten(1)
        if self.reservoir is None:
            self.reservoir = torch.zeros(self.N, x.shape[1], device=x.device, dtype=x.dtype)
            self.anchor = torch.zeros_like(self.reservoir)
        x = x[-self.N:]
        end = self.cursor + len(x)
        if end <= self.N:
            self.reservoir[self.cursor:end] = x
        else:
            split = self.N - self.cursor
            self.reservoir[self.cursor:] = x[:split]
            self.reservoir[:end - self.N] = x[split:]
        self.cursor = end % self.N
        self.fill = min(self.N, self.fill + len(x))
        self.rows_since_eval += len(real)

    def ready(self):
        return self.fill == self.N and self.rows_since_eval >= self.N

    # --------------------------------------------------------------- sampling law
    @torch.no_grad()
    def _nearest_other(self, z, points):
        """Half-NN radius as DV12's perturb_latent (zero distances masked), matmul shortlist."""
        d, _ = _knn(z, points, min(4, len(points) - 1) + 1)
        d = d.masked_fill(d == 0, float("inf"))
        nearest = d.min(1).values
        return torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))

    @torch.no_grad()
    def _jitter(self, trainer, latent, prior, noise):
        c = trainer.controller
        if c is None or c.variant not in ("dv10", "dv11", "dv12"):
            return torch.zeros_like(latent)
        trust = c.support_trust if c.variant == "dv11" else 1.
        displacement = c.latent_bandwidth * trust * noise
        if c.variant == "dv12":
            radius = self._nearest_other(latent, prior.z.detach())
            norm = displacement.norm(dim=1)
            displacement = displacement * (radius / norm.clamp_min(1e-20)).clamp_max(1.).unsqueeze(1)
        return displacement

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def maybe_apply(self, trainer, sigma_out):
        """Run one evaluation (and its moves) when the reservoir has turned over."""
        self.moved_rows = None
        if not self.ready():
            return None
        self.rows_since_eval = 0
        self.counters["evals"] += 1
        G, prior = trainer.G, trainer.prior
        modes = [(m, m.training) for m in G.modules()]
        try:
            G.eval()
            z = prior.z.detach()
            # [dev from spec rev 2] F is an iid N-sample of the exact sampling law (table rows drawn
            # with replacement, as prior.sample does), not one draw per particle: R is iid from pi,
            # so under rho = pi the two pools are identically distributed and both x and the
            # dimension test are symmetric (a stratified F failed the null test: d_R 1.89 vs d_F 2.10).
            pick = torch.randint(self.N, (self.N,), device=z.device, generator=self.stream)
            latent = z[pick]
            noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=self.stream)
            y = G(latent + self._jitter(trainer, latent, prior, noise))
            eps = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=self.stream)
            u = torch.rand(self.N, device=z.device, dtype=torch.float64, generator=self.stream)
            if sigma_out:
                y = y + float(sigma_out) * eps
            q = G(z).flatten(1)
        finally:
            for m, flag in modes:
                m.training = flag
        F, R, n, k = y.flatten(1), self.reservoir, self.N, self.k
        own = torch.arange(n, device=q.device)
        rR, _ = _knn(q, R, k)
        # [dev from spec rev 2] no own-draw exclusion and no log(N/(N-1)): q_i is a fixed point
        # given the table and F is an iid N-sample of rho, so E[#F in a ball] = N * int rho (own
        # kernel included), exactly as R's under rho = pi. Excluding i's draw biased x toward deficit
        # by the own-kernel share of rho(q_i) (unit test (a): 84% of null evaluations had discoveries).
        rF, _ = _knn(q, F, k)
        dR, nR = _levina_bickel(_knn(R, R, k, exclude=own)[0])
        dF, nF = _levina_bickel(_knn(F, F, k, exclude=own)[0])
        D_out = q.shape[1]
        last = dict(step=trainer.completed_steps + 1, k=k, d_R=dR, d_F=dF)
        self.last = last
        # BD-GUARD FIX (one mechanism): the kNN-ratio null already absorbs small
        # dimension mismatches, and at N=20000 the asymptotic-variance
        # significance test fires on every evaluation (spec rev 2 R2), so it is
        # not a statistically bounded guard here. Use the data dimension d_R
        # (the reference law) directly, clamped to [1, D_out]. Skip only when
        # d_R is undefined (no bounded dimension can be formed); record d_F as
        # a diagnostic and keep counting such skips in dim_skips.
        if not math.isfinite(dR):
            self.counters["dim_skips"] += 1
            last["skip"] = "dimension undefined"
            return last
        d = min(max(dR, 1.), D_out)   # data dimension only; never skip on significance
        last["d"] = d
        rk_R, rk_F = rR[:, -1].double(), rF[:, -1].double()
        x = d * (rk_R / rk_F).log()
        finite = torch.isfinite(x)
        x = torch.where(finite, x, torch.zeros_like(x))
        # ---- sequential evidence with sign-flip / locality restarts
        # [dev from spec rev 2] the test runs on exact normal scores w of each term (x's null law
        # logit Beta(k,k) has logistic tails: Gaussian/Rayleigh on raw nats gave P(p<=.01) = .017
        # in the null unit test); S (nats) runs in lock-step and only sets the step size beta.
        w = torch.where(finite, _normal_score(x, k), torch.zeros_like(x))
        S, W, cnt = self.S, self.W, self.n
        moved_out = (q - self.anchor).norm(dim=1) > self.radius
        # [dev from spec rev 2, R3 reverted] a sign flip ends the excursion and the flipping term is
        # discarded (n = 0): keeping it starts the next excursion with a term selected for
        # |w| > |W_prev|, which made the Rayleigh p anti-conservative (null test: P(p<=.01) = .017).
        # A walk restarted from 0 and conditioned on its sign is the meander the p-value assumes.
        flip = finite & (cnt > 0) & ~moved_out & (torch.sign(W + w) != torch.sign(W))
        restart = finite & ((cnt == 0) | moved_out) & ~flip
        extend = finite & ~restart & ~flip
        S.copy_(torch.where(restart, x, torch.where(extend, S + x, torch.where(flip, torch.zeros_like(S), S))))
        W.copy_(torch.where(restart, w, torch.where(extend, W + w, torch.where(flip, torch.zeros_like(W), W))))
        cnt.copy_(torch.where(restart, torch.ones_like(cnt), torch.where(extend, cnt + 1,
                                                                         torch.where(flip, torch.zeros_like(cnt), cnt))))
        self.anchor[restart] = q[restart].to(self.anchor.dtype)
        self.radius[restart] = rF[restart, -1].to(self.radius.dtype)
        # ---- BH over eligible particles
        eligible = finite & (cnt >= 2)
        m = int(eligible.sum())
        zscore = W.abs() / cnt.clamp_min(1).double().sqrt()
        p = torch.exp(-.5 * zscore.square())
        discover = torch.zeros_like(eligible)
        zstar = None
        if m:
            pe = p[eligible].sort().values
            ranks = torch.arange(1, m + 1, device=pe.device, dtype=pe.dtype)
            passed = (pe <= ranks * self.Q / m).nonzero()
            if len(passed):
                j = int(passed[-1]) + 1
                cutoff = j * self.Q / m
                zstar = math.sqrt(-2. * math.log(cutoff))
                discover = eligible & (p <= cutoff)
        beta = torch.zeros_like(S)
        if zstar is not None:
            # soft threshold in nats (z* s_k sqrt(n)); direction from the test statistic W
            size = (S.abs() - zstar * self.s_k * cnt.double().sqrt()).clamp_min(0.)
            size = torch.where(torch.sign(S) == torch.sign(W), size, torch.zeros_like(size))
            beta = torch.where(discover, torch.sign(W) * size / cnt.clamp_min(1).double(), beta).clamp_min(-50.)
        prob = torch.where(beta > 0, 1 - torch.exp(-beta), (torch.exp(-beta) - 1).clamp_max(1.))
        discover = discover & (beta != 0)
        realised = discover & (u < prob)
        stay = discover & ~realised
        deaths = (realised & (beta > 0)).nonzero().flatten()
        births = (realised & (beta < 0)).nonzero().flatten()
        deaths = deaths[beta[deaths].argsort(descending=True, stable=True)]
        births = births[beta[births].argsort(stable=True)]
        pairs = min(len(deaths), len(births))
        child, parent = [deaths[:pairs]], [births[:pairs]]
        rest = deaths[pairs:]
        normalised = 0
        # BD-PAIR FIX: evidence-certified transport only (st3 paired design).
        # The normalisation-parent rule clones neutral rows into dead sites;
        # native evidence shows neutral parents are ~85% of all moves and
        # precision/coverage fall with them. Unmatched deaths wait for a
        # certified deficit birth instead of taking a neutral parent.
        if len(rest):
            normalised = 0
        child, parent = torch.cat(child), torch.cat(parent)
        waited_deaths = len(deaths) - len(child)
        waited_births = len(births) - pairs
        self.pending.zero_()
        self.pending[deaths[len(child):]] = True
        self.pending[births[pairs:]] = True
        spent = stay.clone()
        spent[child] = True
        spent[births[:pairs]] = True
        c = self.counters
        c["discoveries"] += int(discover.sum())
        c["realised_deaths"] += len(deaths)
        c["realised_births"] += len(births)
        c["stays"] += int(stay.sum())
        c["waited_deaths"] += waited_deaths
        c["waited_births"] += waited_births
        last.update(eligible=m, discoveries=int(discover.sum()), zstar=zstar,
                    excess=int((discover & (beta > 0)).sum()), deficit=int((discover & (beta < 0)).sum()),
                    realised_deaths=len(deaths), realised_births=len(births), matched=pairs,
                    normalised=normalised, waited_deaths=waited_deaths, waited_births=waited_births,
                    moves=0 if self.dry_run else len(child), mean_abs_x=float(x[finite].abs().mean()) if bool(finite.any()) else None)
        if self.dry_run:
            return last
        self.S[spent] = 0.
        self.W[spent] = 0.
        self.n[spent] = 0
        if len(child):
            self._move(trainer, child, parent)
            self.moved_rows = child
            sites = torch.cat((q[child], q[parent]))
            dist = torch.cdist(self.anchor, sites, compute_mode="use_mm_for_euclid_dist")
            stale = (self.n > 0) & (dist <= self.radius[:, None]).any(1)
            c["stale_resets"] += int(stale.sum())
            self.S[stale] = 0.
            self.W[stale] = 0.
            self.n[stale] = 0
            self.S[child] = 0.
            self.W[child] = 0.
            self.n[child] = 0
            c["moves"] += len(child)
            c["matched"] += pairs
            c["normalised"] += normalised
        return last

    @torch.no_grad()
    def _move(self, trainer, child, parent):
        prior, ema = trainer.prior, trainer.ema_prior
        zp = prior.z.detach()[parent]
        noise = torch.randn(zp.shape, device=zp.device, dtype=zp.dtype, generator=self.stream)
        delta = self._jitter(trainer, zp, prior, noise)
        prior.z[child] = zp + delta
        ema.z[child] = ema.z[parent] + delta
        state = trainer.opt_g.state.get(prior.z, {})
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.shape == prior.z.shape:
                value[child] = value[parent]
        history = trainer.opt_g.latent_history
        if history is not None:
            history[child] = history[parent]

    # --------------------------------------------------------------- state
    _TENSORS = ("reservoir", "S", "W", "n", "anchor", "radius", "pending")

    def diagnostics(self):
        return dict(k=self.k, s_k=self.s_k, fill=self.fill, rows_since_eval=self.rows_since_eval,
                    counters=dict(self.counters), last=self.last,
                    in_excursion=int((self.n > 0).sum()), eligible_now=int((self.n >= 2).sum()))

    def state_dict(self):
        return dict(**{k: getattr(self, k) for k in self._TENSORS},
                    fill=self.fill, cursor=self.cursor, rows_since_eval=self.rows_since_eval,
                    counters=dict(self.counters), last=dict(self.last), stream=self.stream.get_state())

    def load_state_dict(self, state):
        self.check_state(state)
        for key in self._TENSORS:
            value = state[key]
            setattr(self, key, None if value is None else value.clone().to(self.S.device))
        self.fill, self.cursor, self.rows_since_eval = int(state["fill"]), int(state["cursor"]), int(state["rows_since_eval"])
        self.counters, self.last = dict(state["counters"]), dict(state["last"])
        self.stream.set_state(state["stream"].cpu())

    def check_state(self, state):
        keys = set(self._TENSORS) | {"fill", "cursor", "rows_since_eval", "counters", "last", "stream"}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError("invalid birth-death state")
        for key in ("S", "W", "n", "radius", "pending"):
            if not isinstance(state[key], torch.Tensor) or state[key].shape != getattr(self, key).shape:
                raise ValueError(f"birth-death {key} does not match the table")
        for key in ("reservoir", "anchor"):
            if state[key] is not None and (not isinstance(state[key], torch.Tensor) or len(state[key]) != self.N):
                raise ValueError(f"birth-death {key} does not match the table")
        try:
            torch.Generator(device=self.stream.device).set_state(state["stream"].cpu())
        except (TypeError, RuntimeError, AttributeError) as error:
            raise ValueError("invalid birth-death stream state") from error
