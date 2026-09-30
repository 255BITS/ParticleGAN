"""Experimental horizon-free data innovation and game stability controller.

Real statistics control scalar mobility only. They never enter a loss, sample,
parameter assignment, or target fit. Fixed random nonlinear features see more
than changes of the mean. Generator gradient alignment is a distinct signal.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import torch


class DataDriftController:
    def __init__(self, variant="dv1"):
        self.variant = variant
        if variant in ("dv10", "dv11", "dv12"):
            self.latent_bandwidth = None
        if variant == "dv11":
            self.support_mean = self.support_variance = None
            self.support_score = 0.0
            self.support_trust = 1.0
        if variant == "dv12":
            self.latent_applications = []
        self.mobility = 1.0
        self.game_trust = 1.0
        self.game_ratio = 1.0
        self.payoff_error = 0.0
        if variant in ("dv8", "dv9"):
            self.pair_mean = self.pair_variance = None
            self.pair_score = self.pair_drive = 0.0
        if variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            self.data_memory = 0.0
        if variant in ("dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            self.fast_reference = None
            self.fast_mean_variance = self.slow_mean_variance = self.mean_covariance = None
        self.alignment = 0.0
        self.data_score = 0.0
        self.data_drive = 0.0
        self.last_cosine = 0.0
        self.previous_gradient = None
        self.location = self.scale = self.projection = None
        self.reference = self.variance = None
        self.updates = 0
        self.reopens = 0
        self.closed = False

    @staticmethod
    @torch.no_grad()
    def routed_geometry(table, log_mass):
        """Support and cell width of the routed bank's represented mass.

        Exact duplicate positions form one atom whose mass is the sum of its
        rows. Coordinate spread is mass weighted and the cell count is the
        effective number of those atoms, 1 / sum(p**2). Inactive rows never
        enter either calculation. Routing usage and real observations do not
        enter this prior geometry.
        """
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or not all(table.shape)
                or not table.is_floating_point() or not isinstance(log_mass, torch.Tensor)
                or log_mass.shape != (len(table),) or log_mass.device != table.device
                or log_mass.dtype != table.dtype or bool(torch.isnan(log_mass).any())
                or bool(torch.isposinf(log_mass).any())):
            raise ValueError("routed geometry requires a floating table and matching finite or inactive log masses")
        active = torch.isfinite(log_mass)
        if not bool(active.any()):
            raise ValueError("routed geometry requires positive represented mass")
        table, log_mass = table.detach()[active], log_mass.detach()[active]
        if not bool(torch.isfinite(table).all()):
            raise ValueError("active routed geometry centers must be finite")
        centers, inverse, counts = torch.unique(table, dim=0, return_inverse=True, return_counts=True)
        # Group log masses before normalization: duplicating an atom must not
        # change its effective cell count. Sorted segments avoid atomic sums.
        ordered = log_mass[inverse.argsort(stable=True)]
        maxima = torch.segment_reduce(ordered, "max", lengths=counts)
        relative = (ordered - maxima.repeat_interleave(counts)).exp()
        masses = maxima + torch.segment_reduce(relative, "sum", lengths=counts).log()
        weights = masses.softmax(0)
        positive = weights > 0
        centers, weights = centers[positive], weights[positive]
        location = (weights.unsqueeze(1) * centers).sum(0)
        variance = (weights.unsqueeze(1) * (centers - location).square()).sum(0)
        effective_atoms = weights.square().sum().reciprocal()
        width = variance.sqrt() * effective_atoms.pow(-1. / table.shape[1])
        return centers, width

    @staticmethod
    def routed_prior(table, log_mass):
        """Bind immutable geometry to one complete fast/average candidate.

        Build a fresh view after owner updates. Reusing it within one complete
        multi-site forward avoids recomputing geometry at each routing site.
        """
        support, width = DataDriftController.routed_geometry(table, log_mass)
        return SimpleNamespace(z=table, log_mass=log_mass, _mass_support=support, _mass_width=width)

    @torch.no_grad()
    def observe_prior(self, prior):
        """Kernel width from learned latent geometry, never real-data statistics.

        N**(-1/d) is the linear size of one of N equal-volume latent cells.
        Running coordinate spread makes the width follow ordinary prior learning
        without an acquisition clock or an evaluator horizon.
        """
        if self.variant not in ("dv10", "dv11", "dv12"):
            return
        if hasattr(prior, "log_mass"):
            width = getattr(prior, "_mass_width", None)
            if width is None:
                _, width = self.routed_geometry(prior.z, prior.log_mass)
        else:
            z = prior.z.detach()
            width = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
        if self.latent_bandwidth is None:
            self.latent_bandwidth = width.clone()
        else:
            self.latent_bandwidth.lerp_(width, .01)

    def perturb_latent(self, latent, stream, prior=None, record=False):
        if self.variant not in ("dv10", "dv11", "dv12"):
            return latent
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
        trust = self.support_trust if self.variant == "dv11" else 1.
        displacement = self.latent_bandwidth * trust * noise
        if self.variant == "dv12":
            if prior is None:
                raise ValueError("local support requires the corresponding prior")
            with torch.no_grad():
                nearest = torch.full((len(latent),), float("inf"), device=latent.device, dtype=latent.dtype)
                # Squared distances by broadcasting (~25x faster than cdist's no-mm path); sqrt is monotone and
                # correctly rounded, so min-then-sqrt equals the old sqrt-then-min bitwise (d == 0 iff d**2 == 0).
                # 2048 // z_dim centers per chunk keeps peak memory below the old cdist path's.
                query = latent.detach()[:, None, :]
                support = prior.z.detach()
                if hasattr(prior, "log_mass"):
                    support = getattr(prior, "_mass_support", None)
                    if support is None:
                        support, _ = self.routed_geometry(prior.z, prior.log_mass)
                for centers in support.split(max(1, 2048 // latent.shape[1])):
                    distance = query - centers[None, :, :]
                    distance = distance.mul_(distance).sum(-1)
                    distance.masked_fill_(distance == 0, float("inf"))
                    nearest = torch.minimum(nearest, distance.min(1).values)
                nearest = nearest.sqrt()
                radius = torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))
                norm = displacement.norm(dim=1)
                fraction = (radius / norm.clamp_min(1e-20)).clamp_max(1.)
            displacement = displacement * fraction.unsqueeze(1)
            if record:
                self.latent_applications.append({"radius_min": float(radius.min()),
                    "radius_mean": float(radius.mean()), "radius_max": float(radius.max()),
                    "perturbation_rms": float(displacement.detach().square().mean().sqrt()),
                    "clipped_fraction": float((fraction < 1.).float().mean())})
                self.latent_applications = self.latent_applications[-2:]
        return latent + displacement

    @torch.no_grad()
    def observe_support(self, generator, critic, latent, sigma_out, stream):
        """Probe full latent width even while its applied width is suppressed."""
        if self.variant != "dv11":
            return
        modes = [(m, m.training) for root in (generator, critic) for m in root.modules()]
        try:
            generator.eval()
            critic.eval()
            noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
            clean = generator(latent)
            probe = generator(latent + self.latent_bandwidth * noise)
            if sigma_out:
                epsilon = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=stream)
                clean = clean + sigma_out * epsilon
                probe = probe + sigma_out * epsilon
            difference = (critic(clean) - critic(probe)).flatten()
            mean = difference.mean()
            variance = (difference.var(unbiased=False) / len(difference)).clamp_min(1e-10)
            if self.support_mean is None:
                self.support_mean = torch.zeros_like(mean)
                self.support_variance = variance.clone()
            self.support_mean.lerp_(mean, .02)
            self.support_variance.mul_(.98 ** 2).add_(variance, alpha=.02 ** 2)
            self.support_score = float(self.support_mean / self.support_variance.clamp_min(1e-10).sqrt())
            unexplained = max(0., (self.support_score - 3.) / 3.)
            self.support_trust = 1. / (1. + unexplained * unexplained)
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def observe_real(self, real):
        x = real.flatten(1)
        if self.location is None:
            self.location = x.mean(0)
            self.scale = x.std(0, unbiased=False).clamp_min(1e-6)
            # Local stream: feature initialization never advances training RNG.
            stream = torch.Generator(device="cpu").manual_seed(1729)
            self.projection = (torch.randn(x.shape[1], 32, generator=stream, device="cpu")
                               / math.sqrt(x.shape[1])).to(x)
        u = ((x - self.location) / self.scale) @ self.projection
        features = torch.cat((u, torch.cos(u), torch.sin(u), torch.cos(2*u), torch.sin(2*u)), 1)
        mean = features.mean(0)
        var = features.var(0, unbiased=False).clamp_min(1e-6)
        if self.reference is None:
            self.reference, self.variance = mean.clone(), var.clone()
        if self.variant in ("dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            batch_variance = var / len(x)
            if self.fast_reference is None:
                self.fast_reference = mean.clone()
                self.fast_mean_variance = batch_variance.clone()
                self.slow_mean_variance = batch_variance.clone()
                self.mean_covariance = batch_variance.clone()
            self.fast_reference.lerp_(mean, .1)
            self.reference.lerp_(mean, .01)
            self.fast_mean_variance.mul_(.9**2).add_(batch_variance, alpha=.1**2)
            self.slow_mean_variance.mul_(.99**2).add_(batch_variance, alpha=.01**2)
            self.mean_covariance.mul_(.9*.99).add_(batch_variance, alpha=.1*.01)
            variance_of_difference = (self.fast_mean_variance + self.slow_mean_variance - 2*self.mean_covariance).clamp_min(1e-10)
            self.data_score = float(((self.fast_reference - self.reference).square() / variance_of_difference).mean().sqrt())
        else:
            variance_of_difference = (var + self.variance * (.01 / 1.99)) / len(x)
            self.data_score = float(((mean - self.reference).square() / variance_of_difference.clamp_min(1e-10)).mean().sqrt())
            self.reference.lerp_(mean, .01)
        self.variance.lerp_(var, .01)
        self.data_drive = min(1., max(0., (self.data_score - 3.) / 3.))
        return self._advance_mobility()

    @torch.no_grad()
    def observe_blind(self):
        """``observe_real`` without any statistic of the real batch: the data-drift term is 0 and the mobility follows the
        game terms (payoff error, generator-gradient alignment) only."""
        self.data_score = 0.0
        self.data_drive = 0.0
        return self._advance_mobility()

    def _advance_mobility(self):
        if self.variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            memory_speed = .05 if self.data_drive > self.data_memory else .005
            self.data_memory += memory_speed * (self.data_drive - self.data_memory)
        target = max(self.data_drive, min(1., max(0., self.alignment / .2)))
        if self.variant in ("dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            target = max(self.data_drive, min(1., self.payoff_error ** 2))
        if self.variant == "dv8":
            target = max(target, self.pair_drive)
        speed = .05 if target > self.mobility else .005
        self.mobility += speed * (target - self.mobility)
        if self.mobility < .1:
            self.closed = True
        elif self.mobility > .3 and self.closed:
            self.reopens += 1
            self.closed = False
        self.updates += 1
        return self.current_scales()

    def current_scales(self):
        prior_mobility = max(self.mobility, self.pair_drive) if self.variant == "dv9" else self.mobility
        return ((.01 + .99 * self.mobility) * self.game_trust,
                (.05 + .95 * prior_mobility) * self.game_trust)

    @torch.no_grad()
    def observe_pair(self, real, fake):
        if self.variant not in ("dv8", "dv9"):
            return
        def features(x):
            u = ((x.flatten(1) - self.location) / self.scale) @ self.projection
            return torch.cat((u, torch.cos(u), torch.sin(u), torch.cos(2*u), torch.sin(2*u)), 1)
        real_features, fake_features = features(real), features(fake)
        difference = real_features.mean(0) - fake_features.mean(0)
        variance = (real_features.var(0, unbiased=False) / len(real) +
                    fake_features.var(0, unbiased=False) / len(fake)).clamp_min(1e-10)
        if self.pair_mean is None:
            self.pair_mean = torch.zeros_like(difference)
            self.pair_variance = variance.clone()
        self.pair_mean.lerp_(difference, .02)
        self.pair_variance.mul_(.98 ** 2).add_(variance, alpha=.02 ** 2)
        self.pair_score = float((self.pair_mean.square() / self.pair_variance.clamp_min(1e-10)).mean().sqrt())
        self.pair_drive = min(1., max(0., (self.pair_score - 3.) / 3.))

    def critic_scale(self):
        # Separate game balance acts even before the KA2 reference exists.
        return 1. / (1. + self.payoff_error ** 2) if self.variant in ("dv7", "dv8", "dv9", "dv10", "dv11", "dv12") else 1.

    def observe_game(self, record):
        if self.variant not in ("dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            return
        ratio = 1.0
        if record.sur_base and record.last_sur is not None:
            ratio = record.last_sur / record.sur_base
        self.game_ratio = ratio
        evidence = self.data_drive if self.variant in ("dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12") else self.data_memory
        unexplained = max(0., ratio - 1.) * (1. - evidence)
        self.game_trust = 1. / (1. + unexplained * unexplained)

    @torch.no_grad()
    def observe_generator(self, generator, loss_gan=None, loss_discriminator=None):
        if self.variant in ("dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"):
            error = max(0., float(loss_gan - loss_discriminator) / math.log(2.))
            self.payoff_error += .02 * (error - self.payoff_error)
        values = [p.grad.detach().flatten() for p in generator.parameters() if p.grad is not None]
        if not values:
            return
        gradient = torch.cat(values)
        if self.previous_gradient is not None:
            self.last_cosine = float(torch.dot(gradient, self.previous_gradient) /
                (gradient.norm() * self.previous_gradient.norm()).clamp_min(1e-20))
            self.alignment += .02 * (self.last_cosine - self.alignment)
        self.previous_gradient = gradient.clone()

    def diagnostics(self):
        return {**({"latent_applications": self.latent_applications} if self.variant == "dv12" else {}), **({"support_score": self.support_score, "support_trust": self.support_trust, "applied_latent_bandwidth": None if self.latent_bandwidth is None else (self.latent_bandwidth * self.support_trust).tolist()} if self.variant == "dv11" else {}), **({"latent_bandwidth": self.latent_bandwidth.tolist()} if self.variant in ("dv10", "dv11", "dv12") and self.latent_bandwidth is not None else {}), **({"pair_score": self.pair_score, "pair_drive": self.pair_drive} if self.variant in ("dv8", "dv9") else {}), **({"data_memory": self.data_memory} if self.variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12") else {}), **{k: getattr(self, k) for k in ("payoff_error", "game_trust", "game_ratio", "variant", "mobility", "alignment", "data_score", "data_drive", "last_cosine", "updates", "reopens", "closed")}}

    def state_dict(self):
        return deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != set(self.__dict__) or state["variant"] != self.variant:
            raise ValueError("incompatible continuous controller state")
        if self.variant in ("dv10", "dv11", "dv12"):
            bandwidth = state["latent_bandwidth"]
            current = self.latent_bandwidth
            if bandwidth is not None and (not isinstance(bandwidth, torch.Tensor)
                    or bandwidth.ndim != 1 or not bandwidth.is_floating_point()
                    or not torch.isfinite(bandwidth).all() or bool((bandwidth < 0).any())
                    or (current is not None and (bandwidth.shape != current.shape
                                                or bandwidth.dtype != current.dtype
                                                or bandwidth.device != current.device))):
                raise ValueError("continuous controller latent bandwidth does not match the table")
            if current is not None and bandwidth is None:
                raise ValueError("continuous controller latent bandwidth is missing")
        self.__dict__.update(deepcopy(state))


# One-sided alpha = .05 Student-t quantiles t_{.95}(dof), dof = 1..11 (the
# window uses K = 12 pairs at scale b -> <= 11 dof, K/2 = 6 at 2b -> <= 5).
_T95 = (6.313752, 2.919986, 2.353363, 2.131847, 2.015048, 1.943180,
        1.894579, 1.859548, 1.833113, 1.812461, 1.795885)
# Table rows can use either tested scale for a decision. Four one-sided tests
# (two directions x two scales) spend .05 per window by Bonferroni, so each
# uses alpha=.0125. These are nominal per-window levels, not anytime bounds.
_T9875 = (25.451700, 6.205347, 4.176535, 3.495406, 3.163381, 2.968687,
          2.841244, 2.751524, 2.685011, 2.633767, 2.593093)


class SettleTest:
    """Stationarity (settling) test of one optimizer param group (lr_control="stationarity").

    Intrinsic time ``tau`` advances by applied/base LR each update; a block
    ends when ``tau`` reaches ``b`` (remainder carried). Block displacement
    D_k = theta(end) - theta(start); consecutive pairs give r = cos(D1, D2)
    at scale b and, every 4 blocks, cos(D1+D2, D3+D4) at scale 2b. A window
    of 2K blocks is decided by one-sided t tests (alpha .05 each way) at both
    scales with one host sync:
      STATIONARY (v_b=-1, v_2b!=+1): s /= 2, b = max(b0, b/2)
      DRIFT      (v_b=+1, v_2b!=-1): s = min(1, 2s), b = max(b0, b/2)
      MIXED (significant, opposite) or INCONCLUSIVE (v_b=0): b *= 2
      REVERSAL (decisive, opposite to the previous decisive): apply, b *= 2.
    Pairs where either block did not move are dropped (fewer dof). A zero-
    spread window is decided by the sign of its mean. A window without any
    valid pair is frozen (no decision). Holds no parameter references.
    """

    K = 12
    B0 = 1.0
    GAMMA = .5

    def __init__(self):
        self.s, self.b, self.tau = 1.0, self.B0, 0.0
        self.anchor = None
        self.blocks = []            # displacements of the current 4-block quad
        self.r_b, self.r_2b = [], []  # device scalars; NaN marks a dropped pair
        self.blocks_in_window = 0
        self.last_decisive = 0
        self.last_decisive_scale = None
        self.last = {}
        self.counts = {k: 0 for k in ("drift", "stationary", "mixed", "inconclusive", "reversal",
                                      "frozen", "dropped_pairs", "reopens", "restarts", "rebases")}
        self.windows = 0
        self.log = []
        self.last_block = None
        self.rows = None
        self.last_row_overlap = None
        self.invalid_block_rows = None

    @staticmethod
    def flat(params):
        return torch.cat([p.detach().reshape(-1) for p in params])

    @torch.no_grad()
    def begin(self, params):
        """Take the first anchor (at the first update, after any caller setup)."""
        if self.anchor is None:
            self.anchor = self.flat(params).clone()

    @torch.no_grad()
    def restart(self, params, reopen=False):
        """Discard the partial window's current quad and re-anchor (after a
        non-gradient jump). ``reopen``: also s = 1, b = b0, clear the window."""
        self.anchor = self.flat(params).clone()
        self.blocks = []
        self.tau = 0.0
        self.invalid_block_rows = None
        if reopen:
            self.s, self.b = 1.0, self.B0
            self.r_b, self.r_2b = [], []
            self.blocks_in_window = 0
            self.last_decisive = 0      # no REVERSAL judged against pre-change evidence
            self.last_decisive_scale = None
            self.counts["reopens"] += 1
        else:
            self.counts["restarts"] += 1

    @torch.no_grad()
    def rebase(self, params, rows):
        """Remove a non-gradient jump of table rows ``rows`` (birth-death move) from
        the current block by re-anchoring only those rows; the block, the quad and
        tau are kept, so frequent moves cannot starve the tester of blocks.
        Falls back to restart() when the group is not a row table."""
        if self.anchor is None or self.rows is None:
            return self.restart(params)
        theta = self.flat(params).view(self.rows, -1)
        self.anchor.view(self.rows, -1)[rows] = theta[rows]
        # A cloned row begins a new particle lineage.  Remove its evidence
        # from every pair in the current window and from the unfinished block.
        # Otherwise old and new displacements can form a spurious cosine.
        if self.invalid_block_rows is None:
            self.invalid_block_rows = torch.zeros(self.rows, device=theta.device, dtype=torch.bool)
        self.invalid_block_rows[rows] = True
        for block in self.blocks:
            block.view(self.rows, -1)[rows] = float("nan")
        for pair in self.r_b + self.r_2b:
            pair[rows] = float("nan")
        self.counts["rebases"] += 1

    def _cos(self, a, b):
        if self.rows is not None:
            # A table row is one independently sampled particle.  The global
            # cosine weights a row by its displacement energy, so a small set
            # of migrating particles can declare DRIFT for the whole prior.
            # Average bounded row cosines instead; rows absent from either
            # block have no direction and contribute no evidence.
            aa, bb = a.view(self.rows, -1), b.view(self.rows, -1)
            na, nb = aa.norm(dim=1), bb.norm(dim=1)
            valid = (na > torch.finfo(a.dtype).tiny) & (nb > torch.finfo(b.dtype).tiny)
            self.last_row_overlap = valid.float().mean()
            cosine = (aa * bb).sum(dim=1) / (na * nb).clamp_min(torch.finfo(a.dtype).tiny)
            return torch.where(valid, cosine.clamp(-1., 1.), torch.full_like(cosine, float("nan")))
        denom = a.norm() * b.norm()
        tiny = torch.finfo(a.dtype).tiny
        return torch.where(denom > tiny, (a @ b) / denom.clamp_min(tiny), torch.full_like(denom, float("nan")))

    @torch.no_grad()
    def observe(self, params, ratio, step=None):
        """Call after the group's optimizer step with applied LR / base LR."""
        if self.anchor is None:
            raise RuntimeError("SettleTest.observe before the anchor was taken")
        self.tau += float(ratio)
        if self.tau < self.b:
            return None
        self.tau -= self.b
        theta = self.flat(params)
        d = theta - self.anchor
        if self.invalid_block_rows is not None and self.rows is not None:
            d.view(self.rows, -1)[self.invalid_block_rows] = float("nan")
            self.invalid_block_rows.zero_()
        self.anchor = theta.clone()
        self.last_block = d
        self.blocks.append(d)
        self.blocks_in_window += 1
        if len(self.blocks) % 2 == 0:
            self.r_b.append(self._cos(self.blocks[-2], self.blocks[-1]))
        if len(self.blocks) == 4:
            self.r_2b.append(self._cos(self.blocks[0] + self.blocks[1], self.blocks[2] + self.blocks[3]))
            self.blocks = []
        if self.blocks_in_window >= 2 * self.K:
            return self._decide(step)
        return None

    @staticmethod
    def _verdict(values, strict=False):
        r = values[~torch.isnan(values)].clamp(-1., 1.).double()
        n = len(r)
        if n == 0:
            return None, 0, float("nan"), float("nan")
        mean = float(r.mean())
        if n == 1:
            return 0, 1, mean, float("nan")
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return (0 if mean == 0. else (1 if mean > 0 else -1)), n, mean, math.copysign(float("inf"), mean)
        t = mean / (sd / math.sqrt(n))
        table = _T9875 if strict else _T95
        crit = table[min(n - 1, len(table)) - 1]
        return (1 if t > crit else -1 if t < -crit else 0), n, mean, t

    def _decide(self, step):
        tested_b = self.b
        values = torch.stack(self.r_b + self.r_2b)
        if self.rows is not None:
            values = torch.nanmean(values, dim=1)
        values = values.cpu()   # the window's one host sync
        nb = len(self.r_b)
        strict = self.rows is not None
        vb, nb_ok, mb, tb = self._verdict(values[:nb], strict)
        v2, n2_ok, m2, t2 = self._verdict(values[nb:], strict)
        self.counts["dropped_pairs"] += (nb - nb_ok) + (len(values) - nb - n2_ok)
        self.windows += 1
        v2 = 0 if v2 is None else v2
        if vb is None:
            decision = "frozen"
        elif strict and vb * v2 == -1:
            decision = "mixed"
        elif strict and (vb == -1 or v2 == -1):
            decision = "stationary"
        elif strict and (vb == 1 or v2 == 1):
            decision = "drift"
        elif vb == -1 and v2 != 1:
            decision = "stationary"
        elif vb == 1 and v2 != -1:
            decision = "drift"
        elif vb != 0 and v2 == -vb:
            decision = "mixed"
        else:
            decision = "inconclusive"
        self.counts[decision] += 1
        if decision in ("stationary", "drift"):
            sign = -1 if decision == "stationary" else 1
            evidence_scale = 2 * tested_b if strict and vb != sign and v2 == sign else tested_b
            at_ceiling = self.s == 1.0
            self.s = self.s * self.GAMMA if sign < 0 else min(1.0, self.s / self.GAMMA)
            if decision == "drift" and at_ceiling and self.rows is not None:
                # A DRIFT verdict cannot raise a table LR already at its cap.
                # Probe a longer scale instead of repeating the same verdict.
                # The window tested b and 2b; if both show drift, 4b is the
                # next untested dyadic scale.  This lets a table with local
                # coherent steps but long-run reversal reach its settling
                # scale without using a clock or a magnitude threshold.
                self.b *= 4 if v2 == 1 else 2
            elif self.last_decisive == -sign and (not strict or self.last_decisive_scale == evidence_scale):
                self.counts["reversal"] += 1
                self.b = 2 * evidence_scale if strict else self.b * 2
            elif strict and vb == 0 and v2 != 0:
                # The first decisive evidence was at 2b. Halve that
                # evidence scale, retaining the current b for the next test.
                self.b = max(self.B0, self.b)
            else:
                self.b = max(self.B0, self.b / 2)
            self.last_decisive = sign
            self.last_decisive_scale = evidence_scale
        elif decision in ("mixed", "inconclusive"):
            self.b *= 2
        self.last = dict(decision=decision, step=step, tested_b=tested_b, verdict_b=vb, verdict_2b=v2,
                         evidence_scale=(2 * tested_b if strict and vb == 0 and v2 != 0 else tested_b),
                         t_b=tb, t_2b=t2, mean_r_b=mb, mean_r_2b=m2,
                         n_b=nb_ok, n_2b=n2_ok)
        self.log = (self.log + [[step, decision, self.s, self.b]])[-8:]
        self.r_b, self.r_2b, self.blocks = [], [], []
        self.blocks_in_window = 0
        return decision

    def diagnostics(self):
        out = dict(s=self.s, b=self.b, tau=self.tau, windows=self.windows, last=self.last,
                   counts=dict(self.counts), log=self.log, pairs_in_window=len(self.r_b))
        out["blocks_in_window"] = self.blocks_in_window
        out["intrinsic_steps_to_next_decision"] = max(0., (2 * self.K - self.blocks_in_window) * self.b - self.tau)
        if self.rows and self.last_block is not None:
            energy = torch.nan_to_num(self.last_block.view(self.rows, -1)).square().sum(1)
            top = max(1, math.ceil(.01 * self.rows))
            total = float(energy.sum())
            out["top1pct_row_energy_share"] = float(energy.topk(top).values.sum()) / total if total > 0 else None
            out["last_row_pair_overlap"] = (float(self.last_row_overlap)
                                            if self.last_row_overlap is not None else None)
        return out

    def state_dict(self):
        return deepcopy({k: v for k, v in self.__dict__.items()})

    def load_state_dict(self, state, numel):
        if set(state) != set(self.__dict__):
            raise ValueError("incompatible settle-test state")
        anchor = state["anchor"]
        if anchor is not None and (not isinstance(anchor, torch.Tensor) or anchor.numel() != numel):
            raise ValueError("settle-test anchor does not match its parameter group")
        if any(not isinstance(d, torch.Tensor) or d.numel() != numel for d in state["blocks"]):
            raise ValueError("settle-test blocks do not match its parameter group")
        pair_numel = state["rows"] if state["rows"] is not None else 1
        if any(not isinstance(r, torch.Tensor) or r.numel() != pair_numel
               for r in state["r_b"] + state["r_2b"]):
            raise ValueError("settle-test pair evidence does not match its parameter group")
        mask = state["invalid_block_rows"]
        if mask is not None and (state["rows"] is None or not isinstance(mask, torch.Tensor)
                                 or mask.shape != (state["rows"],) or mask.dtype != torch.bool):
            raise ValueError("settle-test row lineage mask does not match its parameter group")
        self.__dict__.update(deepcopy(state))


# --- Sequential (early-stopping) table test ---------------------------------------------------
# Error accounting per decision, identical to SettleTest's nominal .05: every one of the two
# scales (b, 2b) spends .0125 on an anytime-valid early-stopping e-process (Ville's inequality,
# any number of looks) and .0125 on the window-end fixed-n test (two one-sided tests at .00625,
# the quantiles below, dof 1..11) -> .025 per scale, .05 per decision.
_T99375 = (50.923037, 8.8602, 5.391949, 4.314656, 3.810005, 3.521223,
           3.335295, 3.205955, 3.110935, 3.038243, 2.980872)
_EARLY_ALPHA = .0125
_LOG_EARLY = math.log(1. / _EARLY_ALPHA)
# Effect-size prior variances of the mixture (equal weights, fixed in advance): the mixture of
# test martingales is a test martingale, and it needs no assumed effect size.
_PRIOR_G = tuple(4. ** k for k in range(5))


def _log_t_bayes_factor(t, n):
    """log mixture Bayes factor of H1: mu/sigma ~ N(0, g) against H0: mu = 0 for a one-sample t
    statistic on n >= 2 observations, sigma unknown (right-Haar prior). Under H0 this is a test
    martingale for every sigma (Lai 1976; Perez-Ortiz et al. 2022), so P(sup_n BF_n >= 1/a) <= a."""
    nu = n - 1
    t2 = t * t
    terms = []
    for g in _PRIOR_G:
        ng = n * g
        terms.append(-.5 * math.log1p(ng)
                     + .5 * n * (math.log1p(t2 / nu) - math.log1p(t2 / (nu * (1. + ng)))))
    top = max(terms)
    return top + math.log(sum(math.exp(x - top) for x in terms) / len(terms))


class SequentialSettleTest(SettleTest):
    """SettleTest that may decide after any completed pair instead of only at the window end.

    Same statistic (row-averaged cosine of consecutive block displacements at scales b and 2b),
    same scale search, same actions and the same nominal error per decision (.05) as SettleTest.
    What changes is *when* a decision may be taken:

    * after every completed pair the evidence collected so far in the window is tested with an
      anytime-valid e-process per scale (mixture t-test Bayes factor >= 1/.0125); a crossing
      decides immediately, so strong evidence no longer waits for the 2K-block window to close;
    * if nothing has crossed when the window fills (2K blocks) the fixed-n one-sided t tests are
      applied at alpha .00625 per direction per scale (the remaining half of the .05 budget), and
      the usual STATIONARY / DRIFT / MIXED / INCONCLUSIVE actions follow.

    ``early=False`` with ``final_table=_T9875`` reproduces SettleTest's decisions exactly.
    Row-lineage handling (birth-death rebase) is inherited: the moved rows are removed from every
    pair of the running window, and the scale means are recomputed from the row vectors at each look.
    """

    def __init__(self, early=True, final_table=None, early_stationary_only=False):
        super().__init__()
        self.early = bool(early)
        self.early_stationary_only = bool(early_stationary_only)
        self.final_table = tuple(_T99375 if final_table is None else final_table)
        self.counts["early"] = 0
        self.counts["held"] = 0
        self.looks = 0
        self.last_look = {}
        # Off-support gate (set by GANTrainer each step from the birth-death support test):
        # ``exclude``: rows that do not vote (off-support rows); ``hold_descent``: a STATIONARY verdict
        # is deferred (early: evidence kept; window end: window discarded) while too much of the table
        # is off-support.
        self.exclude = None
        self.hold_descent = False
        # Table release rule (a DRIFT verdict below the ceiling raises s): "any" = the SettleTest table rule (either scale
        # positive and not opposed); "both" = both scales significantly positive; "never" = no release from this tester.
        self.release_rule = "any"
        # ANCHOR: evidence scale (in intrinsic time) of the last accepted STATIONARY decision = the scale at which the bulk
        # mean-reverts. Under a stationary bulk the block cosine is positive below it and negative near it, so a DRIFT verdict
        # whose evidence scale is below twice this anchor is what a settled table produces and is not evidence of drift.
        self.b_anchor = None
        self.counts["drift_blocked"] = 0

    def restart(self, params, reopen=False):
        super().restart(params, reopen=reopen)
        if reopen:
            self.b_anchor = None

    @torch.no_grad()
    def _scale_values(self):
        """Per-pair scalars (mean over the rows that have a direction) for the b and 2b scales;
        one host sync."""
        nb = len(self.r_b)
        if not (self.r_b or self.r_2b):
            return torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
        values = torch.stack(self.r_b + self.r_2b)
        if self.rows is not None:
            if self.exclude is not None:
                values = values.masked_fill(self.exclude.to(values.device)[None, :], float("nan"))
            values = torch.nanmean(values, dim=1)
        values = values.cpu().double()
        return values[:nb], values[nb:]

    @staticmethod
    def _early_stat(values):
        """Anytime-valid look: dict(verdict, n, mean, t, log_bf); verdict 0 until the e-process crosses."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        out = dict(verdict=0, n=n, mean=float(r.mean()) if n else float("nan"), t=float("nan"),
                   log_bf=float("-inf"))
        if n < 2:
            return out
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return out          # degenerate spread: left to the window-end rule
        t = out["mean"] / (sd / math.sqrt(n))
        out["t"] = t
        out["log_bf"] = _log_t_bayes_factor(t, n)
        if out["log_bf"] >= _LOG_EARLY:
            out["verdict"] = 1 if t > 0 else -1
        return out

    def _final_verdict(self, values):
        """Window-end fixed-n one-sided t tests (table quantiles of ``final_table``)."""
        r = values[~torch.isnan(values)].clamp(-1., 1.)
        n = len(r)
        if n == 0:
            return dict(verdict=None, n=0, mean=float("nan"), t=float("nan"), log_bf=float("nan"))
        mean = float(r.mean())
        if n == 1:
            return dict(verdict=0, n=1, mean=mean, t=float("nan"), log_bf=float("nan"))
        sd = float(r.std(unbiased=True))
        if sd == 0.:
            return dict(verdict=(0 if mean == 0. else (1 if mean > 0 else -1)), n=n, mean=mean,
                        t=math.copysign(float("inf"), mean), log_bf=float("nan"))
        t = mean / (sd / math.sqrt(n))
        crit = self.final_table[min(n - 1, len(self.final_table)) - 1]
        return dict(verdict=(1 if t > crit else -1 if t < -crit else 0), n=n, mean=mean, t=t,
                    log_bf=float("nan"))

    def _conclude(self, sb, s2, tested_b, step, early):
        """Map the two scale verdicts to a decision (table rule of SettleTest) and act on it."""
        vb, v2 = sb["verdict"], s2["verdict"]
        v2 = 0 if v2 is None else v2
        if vb is None:
            decision = "frozen"
        elif vb * v2 == -1:
            decision = "mixed"
        elif vb == -1 or v2 == -1:
            decision = "stationary"
        elif vb == 1 or v2 == 1:
            decision = "drift"
            if self.s < 1.0:
                drift_scale = 2 * tested_b if vb != 1 and v2 == 1 else tested_b
                if self.release_rule == "never" or (self.release_rule == "both" and not (vb == 1 and v2 == 1)):
                    decision = "inconclusive"   # a release needs the stated evidence; otherwise search a longer scale
                elif (self.release_rule == "anchor" and self.b_anchor is not None
                      and drift_scale < 2 * self.b_anchor):
                    decision = "inconclusive"
                    self.counts["drift_blocked"] += 1
        else:
            decision = "inconclusive"
        if decision == "stationary" and self.hold_descent:
            decision = "held"           # off-support mass above the FDR budget: descent is deferred
        self.counts[decision] += 1
        if early:
            self.counts["early"] += 1
        if decision in ("stationary", "drift"):
            sign = -1 if decision == "stationary" else 1
            evidence_scale = 2 * tested_b if vb != sign and v2 == sign else tested_b
            at_ceiling = self.s == 1.0
            self.s = self.s * self.GAMMA if sign < 0 else min(1.0, self.s / self.GAMMA)
            if decision == "drift" and at_ceiling and self.rows is not None:
                self.b *= 4 if v2 == 1 else 2
            elif self.last_decisive == -sign and self.last_decisive_scale == evidence_scale:
                self.counts["reversal"] += 1
                self.b = 2 * evidence_scale
            elif vb == 0 and v2 != 0:
                self.b = max(self.B0, self.b)
            else:
                self.b = max(self.B0, self.b / 2)
            self.last_decisive = sign
            self.last_decisive_scale = evidence_scale
            if sign < 0:
                self.b_anchor = evidence_scale
        elif decision in ("mixed", "inconclusive"):
            self.b *= 2
        self.last = dict(decision=decision, step=step, tested_b=tested_b, verdict_b=vb, verdict_2b=v2,
                         evidence_scale=(2 * tested_b if vb == 0 and v2 != 0 else tested_b),
                         t_b=sb["t"], t_2b=s2["t"], mean_r_b=sb["mean"], mean_r_2b=s2["mean"],
                         n_b=sb["n"], n_2b=s2["n"], early=bool(early),
                         log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        self.log = (self.log + [[step, decision, self.s, self.b]])[-8:]
        self.r_b, self.r_2b, self.blocks = [], [], []
        self.blocks_in_window = 0
        return decision

    def _look(self, step):
        """Early-stopping look after a completed pair; None while no e-process has crossed."""
        if len(self.r_b) < 2 and len(self.r_2b) < 2:
            return None
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._early_stat(vb_values), self._early_stat(v2_values)
        self.looks += 1
        self.last_look = dict(step=step, n_b=sb["n"], n_2b=s2["n"], t_b=sb["t"], t_2b=s2["t"],
                              log_bf_b=sb["log_bf"], log_bf_2b=s2["log_bf"])
        if sb["verdict"] == 0 and s2["verdict"] == 0:
            return None
        if self.hold_descent and (sb["verdict"] == -1 or s2["verdict"] == -1):
            return None                 # a descent is due but held: keep the evidence, decide when the hold lifts
        if self.early_stationary_only:
            # Only a descent (STATIONARY: cheap to undo) may be decided early, and only when the other
            # scale can speak (>= 2 values) and does not oppose it. DRIFT / release and MIXED are decided
            # at the window end with the full two-scale table, so an oscillation (r_b > 0, r_2b < 0)
            # keeps its MIXED outcome instead of becoming an early DRIFT.
            calm_b = sb["verdict"] == -1 and s2["verdict"] != 1 and s2["n"] >= 2 and s2["mean"] <= 0.
            calm_2b = s2["verdict"] == -1 and sb["verdict"] != 1 and sb["n"] >= 2 and sb["mean"] <= 0.
            if not (calm_b or calm_2b):
                return None
        self.windows += 1
        return self._conclude(sb, s2, self.b, step, early=True)

    def _decide(self, step):
        """Window end (2K blocks, nothing crossed early): fixed-n tests, then the usual actions."""
        tested_b = self.b
        vb_values, v2_values = self._scale_values()
        sb, s2 = self._final_verdict(vb_values), self._final_verdict(v2_values)
        self.counts["dropped_pairs"] += (len(vb_values) - sb["n"]) + (len(v2_values) - s2["n"])
        self.windows += 1
        return self._conclude(sb, s2, tested_b, step, early=False)

    @torch.no_grad()
    def observe(self, params, ratio, step=None):
        """Call after the group's optimizer step with applied LR / base LR."""
        if self.anchor is None:
            raise RuntimeError("SettleTest.observe before the anchor was taken")
        self.tau += float(ratio)
        if self.tau < self.b:
            return None
        self.tau -= self.b
        theta = self.flat(params)
        d = theta - self.anchor
        if self.invalid_block_rows is not None and self.rows is not None:
            d.view(self.rows, -1)[self.invalid_block_rows] = float("nan")
            self.invalid_block_rows.zero_()
        self.anchor = theta.clone()
        self.last_block = d
        self.blocks.append(d)
        self.blocks_in_window += 1
        new_pair = False
        if len(self.blocks) % 2 == 0:
            self.r_b.append(self._cos(self.blocks[-2], self.blocks[-1]))
            new_pair = True
        if len(self.blocks) == 4:
            self.r_2b.append(self._cos(self.blocks[0] + self.blocks[1], self.blocks[2] + self.blocks[3]))
            self.blocks = []
        if self.blocks_in_window >= 2 * self.K:
            return self._decide(step)
        if new_pair and self.early:
            return self._look(step)
        return None

    def diagnostics(self):
        out = super().diagnostics()
        out["sequential"] = True
        out["hold_descent"] = self.hold_descent
        out["excluded_rows"] = None if self.exclude is None else int(self.exclude.sum())
        out["b_anchor"] = self.b_anchor
        out["early_decisions"] = self.counts.get("early", 0)
        out["looks"] = self.looks
        out["last_look"] = self.last_look
        return out


class StationarityLR:
    """Per-group settling tests for explicitly owned optimizers.

    ``optimizers`` starts with generator and critic owners; a separate table
    owner may follow.  The isolated ``prior_param`` group gets the sequential
    row-lineage-aware tester, independent of any training-loop class.
    """

    def __init__(self, optimizers, prior_param=None, release_rule="any"):
        self.testers = []
        for optimizer in optimizers:
            row = []
            for group in optimizer.param_groups:
                trainable = [p for p in group["params"] if p.requires_grad]
                tester = SettleTest() if trainable else None
                if tester is not None and prior_param is not None and len(group["params"]) == 1 \
                        and group["params"][0] is prior_param and prior_param.dim() == 2:
                    tester = SequentialSettleTest(early_stationary_only=True, final_table=_T9875)     # descent may be decided early
                    tester.rows = prior_param.shape[0]
                    tester.release_rule = release_rule
                row.append(tester)
            self.testers.append(row)

    def pairs(self, optimizers):
        for optimizer, row in zip(optimizers, self.testers):
            for group, tester in zip(optimizer.param_groups, row):
                if tester is not None:
                    yield group, tester

    def diagnostics(self):
        names = ("g", "d", "table")
        return {f"{names[i] if i < len(names) else 'optimizer' + str(i)}{j}": t.diagnostics()
                for i, row in enumerate(self.testers)
                for j, t in enumerate(row) if t is not None}

    def state_dict(self):
        return [[None if t is None else t.state_dict() for t in row] for row in self.testers]

    def load_state_dict(self, state, optimizers):
        if (not isinstance(state, list) or len(state) != len(self.testers)
                or any(not isinstance(a, list) or len(a) != len(b) for a, b in zip(state, self.testers))):
            raise ValueError("settle state group count does not match the optimizers")
        for optimizer, row, saved in zip(optimizers, self.testers, state):
            for group, tester, value in zip(optimizer.param_groups, row, saved):
                if (tester is None) != (value is None):
                    raise ValueError("settle state does not match the trainable groups")
                if tester is not None:
                    deepcopy(tester).load_state_dict(value, sum(p.numel() for p in group["params"]))
        for optimizer, row, saved in zip(optimizers, self.testers, state):
            for group, tester, value in zip(optimizer.param_groups, row, saved):
                if tester is not None:
                    tester.load_state_dict(value, sum(p.numel() for p in group["params"]))
