"""Experimental horizon-free data innovation and game stability controller.

Real statistics control scalar mobility only. They never enter a loss, sample,
parameter assignment, or target fit. Fixed random nonlinear features see more
than changes of the mean. Generator gradient alignment is a distinct signal.
"""
from copy import deepcopy
import math
import torch


class DataDriftController:
    def __init__(self, variant="dv1"):
        self.variant = variant
        if variant in ("dv14", "dv15", "dv16"):
            self.width_pair = {}
        if variant in ("dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            self.latent_bandwidth = None
        if variant == "dv11":
            self.support_mean = self.support_variance = None
            self.support_score = 0.0
            self.support_trust = 1.0
        if variant in ("dv12", "dv13", "dv14", "dv15", "dv16"):
            self.latent_applications = []
        self.mobility = 1.0
        self.game_trust = 1.0
        self.game_ratio = 1.0
        self.payoff_error = 0.0
        if variant in ("dv8", "dv9"):
            self.pair_mean = self.pair_variance = None
            self.pair_score = self.pair_drive = 0.0
        if variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            self.data_memory = 0.0
        if variant in ("dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
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

    @torch.no_grad()
    def observe_prior(self, prior):
        """Kernel width from learned latent geometry, never real-data statistics.

        N**(-1/d) is the linear size of one of N equal-volume latent cells.
        Running coordinate spread makes the width follow ordinary prior learning
        without an acquisition clock or an evaluator horizon.
        """
        if self.variant not in ("dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            return
        z = prior.z.detach()
        width = z.std(0, unbiased=False) * len(z) ** (-1. / z.shape[1])
        if self.latent_bandwidth is None:
            self.latent_bandwidth = width.clone()
        else:
            self.latent_bandwidth.lerp_(width, .01)

    def perturb_latent(self, latent, stream, prior=None, record=False, indices=None, sign=1.):
        if self.variant not in ("dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            return latent
        if self.variant in ("dv13", "dv14", "dv15", "dv16") and (prior is None or indices is None or not hasattr(prior, "log_width")):
            raise ValueError("learned support requires matching prior and particle indices")
        noise = (2. * torch.rand(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream) - 1.
                 if self.variant in ("dv15", "dv16") else
                 torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream))
        if self.variant == "dv16":
            # A bounded rank-one shear learns conditional correlation while each
            # coordinate retains variance 1/3. No real statistics enter samples.
            raw = prior.shape_shear[indices]
            shear = raw / (1. + raw.square().sum(1, keepdim=True)).sqrt()
            direction = 1. / math.sqrt(latent.shape[1])
            projection = noise.sum(1, keepdim=True) * direction
            norm = (1. + 2. * shear * direction + shear.square()).sqrt()
            noise = (noise + shear * projection) / norm
        trust = self.support_trust if self.variant == "dv11" else 1.
        displacement = self.latent_bandwidth * trust * noise
        if self.variant in ("dv13", "dv14", "dv15", "dv16"):
            width = self.latent_bandwidth * torch.nn.functional.softplus(prior.log_width[indices]) / math.log(2.)
            displacement = sign * width * noise
            if record:
                self.latent_applications.append({"width_min": float(width.detach().min()), "width_mean": float(width.detach().mean()), "width_max": float(width.detach().max()), "perturbation_rms": float(displacement.detach().square().mean().sqrt())})
                if self.variant == "dv16":
                    self.latent_applications[-1]["shear_norm_max"] = float(shear.detach().norm(dim=1).max())
                self.latent_applications = self.latent_applications[-2:]
        if self.variant == "dv12":
            if prior is None:
                raise ValueError("local support requires the corresponding prior")
            with torch.no_grad():
                nearest = torch.full((len(latent),), float("inf"), device=latent.device, dtype=latent.dtype)
                for centers in prior.z.detach().split(4096):
                    distance = torch.cdist(latent.detach(), centers, compute_mode="donot_use_mm_for_euclid_dist")
                    distance.masked_fill_(distance == 0, float("inf"))
                    nearest = torch.minimum(nearest, distance.min(1).values)
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
        if self.variant in ("dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
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
        if self.variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            memory_speed = .05 if self.data_drive > self.data_memory else .005
            self.data_memory += memory_speed * (self.data_drive - self.data_memory)
        target = max(self.data_drive, min(1., max(0., self.alignment / .2)))
        if self.variant in ("dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
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
        return 1. / (1. + self.payoff_error ** 2) if self.variant in ("dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16") else 1.

    def observe_game(self, record):
        if self.variant not in ("dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
            return
        ratio = 1.0
        if record.sur_base and record.last_sur is not None:
            ratio = record.last_sur / record.sur_base
        self.game_ratio = ratio
        evidence = self.data_drive if self.variant in ("dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16") else self.data_memory
        unexplained = max(0., ratio - 1.) * (1. - evidence)
        self.game_trust = 1. / (1. + unexplained * unexplained)

    @torch.no_grad()
    def observe_generator(self, generator, loss_gan=None, loss_discriminator=None):
        if self.variant in ("dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16"):
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
        return {**({"width_pair": self.width_pair} if self.variant in ("dv14", "dv15", "dv16") else {}), **({"latent_applications": self.latent_applications} if self.variant in ("dv12", "dv13", "dv14", "dv15", "dv16") else {}), **({"support_score": self.support_score, "support_trust": self.support_trust, "applied_latent_bandwidth": None if self.latent_bandwidth is None else (self.latent_bandwidth * self.support_trust).tolist()} if self.variant == "dv11" else {}), **({"latent_bandwidth": self.latent_bandwidth.tolist()} if self.variant in ("dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16") and self.latent_bandwidth is not None else {}), **({"pair_score": self.pair_score, "pair_drive": self.pair_drive} if self.variant in ("dv8", "dv9") else {}), **({"data_memory": self.data_memory} if self.variant in ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12", "dv13", "dv14", "dv15", "dv16") else {}), **{k: getattr(self, k) for k in ("payoff_error", "game_trust", "game_ratio", "variant", "mobility", "alignment", "data_score", "data_drive", "last_cosine", "updates", "reopens", "closed")}}

    def state_dict(self):
        return deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if set(state) != set(self.__dict__) or state["variant"] != self.variant:
            raise ValueError("incompatible continuous controller state")
        self.__dict__.update(deepcopy(state))
