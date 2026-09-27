"""Shared-state encoder transition GAN, declared as a problem only.

G1 -> st, G2 -> at, G3 -> st+1 from one MoG draw plus observed geometry, time
and class; E(st, at) -> particle code -> G3 adds real prediction and synthetic
composition. Critics: joint D(st, at, st+1), D_action(at) and one D_state
shared by st (at t) and st+1 (at t+dt). See docs/transition-gan.md.

This module declares the data sampler, the networks, the critic views, the
encoder's supervised terms, the metrics and the verdict. The optimizers (and
their LR schedule), loss, critic penalties, critic input noise, generator
output noise, EMA and logging come from the recipe through
``benchmarks.toy_runner``::

    python -u examples/transition_gan.py --seed 24002 --log runs/toy-refactor/example_transition_gan.log

The leaderboard artifact run (provenance, saved samples, viewer, checkpoint)
is ``experiments/train_transition.py``, which trains this same problem.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.toy_runner import Networks, Sample, ToyProblem, View, main
from lib.transition import (Transitions, TransitionScaler, TransitionGenerator, TransitionCritics,
                            TransitionEncoder, encoded_transition, composed_transition, metrics)
from particlegan import get_recipe, init

# The winning leaderboard configuration (reports/transition/leaderboard, encoder_shared_state).
PROBLEM_DEFAULTS = dict(encoder=True, shared_state_critic=True, encoder_width=128,
                        real_encoding_weight=1., synthetic_reconstruction_weight=1.,
                        architecture="branches", width=128, d_width=256, z_dim=32,
                        d_conditioning="concat", g_class_scale=8.0, g_context_scale=1.0,
                        num_particles=1024, critic_mode="joint_marginals", marginal_width=128, marginal_weight=1.0,
                        length=64, geometry_mode="discrete", steps=28_000, batch_size=256,
                        eval_per_context=512, normalization_samples=32768)

# Verdict: beat the best adversarial-only leaderboard entry (concat_class8_marginals) on
# held-out joint SW1 and transition residual, i.e. the encoder must earn its place.
PASS_JOINT_SW1 = 0.10027
PASS_RESIDUAL = 0.02393
EVAL_TICKS = (0, .25, .5, .75, 1)


def transition_recipe(cfg):
    """The shipped MoG recipe (sigma_rel 0.025) at this task's shape; 2 observed classes."""
    return get_recipe(prior_kind='mog', sigma_rel=0.025, z_dim=cfg["z_dim"], num_particles=cfg["num_particles"],
                      num_classes=2, conditioning="ucd" if cfg["d_conditioning"] == "ucd" else "conditional",
                      total_steps=cfg["steps"], batch_size=cfg["batch_size"])


class Score(nn.Module):
    """A transition critic's scalar score; the class-head logits are unused under concat."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, x, c, context):
        return self.critic(x, c, context)[0]


class TransitionGAN(ToyProblem):
    name = "transition_gan"

    def __init__(self, device="cpu", **overrides):
        unknown = set(overrides) - set(PROBLEM_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown transition options: {sorted(unknown)}")
        self.cfg = cfg = {**PROBLEM_DEFAULTS, **overrides}
        if cfg["d_conditioning"] != "concat":
            # UCD critics need a classification loss on the class-head logits, which the
            # recipe has no generic factory for; the historical UCD arms are not runnable here.
            raise NotImplementedError("the shared runner declares concat critics only (UCD arms are flagged)")
        # The normalization is fit on the run device, as on the leaderboard: fit() draws from a
        # device-specific generator (seed 91001), so a CPU fit would differ from a CUDA fit.
        self.device = torch.device(device)
        self.toy = Transitions(cfg["length"], self.device, cfg["geometry_mode"])
        self.scaler = TransitionScaler.fit(self.toy, cfg["normalization_samples"])
        self.layout = None
        self._composed = None

    # -- problem declaration --------------------------------------------------
    def recipe(self):
        return transition_recipe(self.cfg)

    def networks(self, recipe, seed):
        # Fixed init seeds (G=0, D=1, E=2), independent of the run seed, as on the leaderboard.
        cfg = self.cfg
        g = init.deterministic_orthogonal_(TransitionGenerator(
            cfg["z_dim"], cfg["architecture"], cfg["width"],
            class_scale=cfg["g_class_scale"], context_scale=cfg["g_context_scale"]), seed=0)
        self.layout = init.deterministic_orthogonal_(TransitionCritics(
            cfg["d_width"], cfg["critic_mode"], cfg["marginal_width"], cfg["d_conditioning"],
            shared_state=cfg["shared_state_critic"], scaler=self.scaler, length=cfg["length"]), seed=1)
        e = init.deterministic_orthogonal_(TransitionEncoder(
            cfg["z_dim"], cfg["encoder_width"], cfg["g_class_scale"], cfg["g_context_scale"]),
            seed=2) if cfg["encoder"] else None
        prior = init.deterministic_orthogonal_(recipe.make_prior())
        self.spread = recipe.make_prior_regularizer()
        critics = {name: Score(critic) for name, critic in self.layout.critics.items()}
        return Networks(generator=g, critics=critics, prior=prior, encoder=e)

    def data(self, device=None):
        """The toy and its fitted scaler; both live on the device given at construction."""
        if device is not None and torch.device(device) != self.device:
            raise ValueError(f"TransitionGAN was fit on {self.device}; construct TransitionGAN(device={str(device)!r})")
        return self.toy, self.scaler

    def real(self, n, stream):
        toy, scaler = self.data(stream.device)
        c, geom, tick, x = toy.batch(n, stream)
        return Sample(scaler(x), condition=(c, toy.condition(geom, tick)))

    def fake(self, nets, n, stream, real):
        if real is None:
            real = self.real(n, stream)
        c, context = real.condition
        z, ids = nets.prior.sample(len(c), stream)
        return Sample(nets.generator(z, c, context), condition=(c, context), indices=ids)

    def composed(self, nets, fake):
        """G1/G2 -> E -> G3 for this fake batch (cached for the encoder's losses)."""
        if self._composed is None or self._composed[0] is not fake.x:
            c, context = fake.condition
            with torch.set_grad_enabled(fake.x.requires_grad):
                composed, decoded, _ = composed_transition(nets.encoder, nets.generator, nets.prior,
                                                           fake.x, c, context)
            self._composed = (fake.x, composed, decoded)
        return self._composed[1:]

    def views(self, nets, real, fake):
        """Joint + marginal_weight * mean(marginals); with E, the mean of the prior and composed paths."""
        layout, cfg = self.layout.to(real.x.device), self.cfg
        c, context = real.condition
        paths = [(fake.x, 1.)]
        if nets.encoder is not None:
            paths = [(fake.x, .5), (self.composed(nets, fake)[0], .5)]
        views = []
        for x, share in paths:
            for role in layout.roles():
                xr, role_context = layout.inputs(role, real.x, context)
                xf, _ = layout.inputs(role, x, context)
                weight = 1. if role == "joint" else cfg["marginal_weight"]/3
                critic = "state" if cfg["shared_state_critic"] and role == "next_state" else role
                views.append(View(critic, xr, xf, (c, role_context), share*weight))
        return views

    def losses(self, role, nets, real, fake):
        if role != "generator":
            return {}
        cfg, prior = self.cfg, nets.prior
        # MoG draws add noise and standardize: regularize raw centers (whole table at <= 1024).
        raw = prior.z if prior.num_particles <= 1024 else prior.z[fake.indices.unique()]
        terms = {"prior_spread": self.spread(raw)}
        if nets.encoder is not None:
            c, context = real.condition
            decoded_real, _ = encoded_transition(nets.encoder, nets.generator, prior, real.x[:, :4], c, context)
            decoded_fake = self.composed(nets, fake)[1]
            # Reconstruct generated observations; never regress to latent IDs or analytic physics.
            # fake.x already carries the runner's generator output noise, so the composed path
            # encodes, and synthetic_mse targets, the noisy G1/G2 sample (the leaderboard used clean).
            terms["real_mse"] = cfg["real_encoding_weight"]*(decoded_real-real.x).square().mean()
            terms["synthetic_mse"] = cfg["synthetic_reconstruction_weight"]*(
                decoded_fake[:, :4]-fake.x[:, :4].detach()).square().mean()
        return terms

    def metrics(self, model):
        """Held-out geometries on the leaderboard's fixed reference draws (evaluate() in memory)."""
        nets = model.nets
        device = next(nets.generator.parameters()).device
        toy, scaler = self.data(device)
        ticks = sorted(set(round(f*(toy.length-2)) for f in EVAL_TICKS))
        cc, gg = toy.contexts("test")
        c0, geom0 = cc.repeat_interleave(len(ticks)), gg.repeat_interleave(len(ticks), 0)
        t0 = torch.tensor(ticks, device=device).repeat(len(cc))
        groups = torch.arange(len(c0), device=device).repeat_interleave(self.cfg["eval_per_context"])
        c, geom, tick = c0[groups], geom0[groups], t0[groups]
        context = toy.condition(geom, tick)
        rngs = [torch.Generator(device=device).manual_seed(99000+i) for i in range(3)]
        chunks = []
        for j in range(0, len(c), 256):
            z, _ = nets.prior.sample(len(c[j:j+256]), rngs[0])
            chunks.append(scaler.inverse(nets.generator(z, c[j:j+256], context[j:j+256])))
        x = torch.cat(chunks)
        if not torch.isfinite(x).all():
            return {"joint_sw1": float("inf"), "consistency_mean": float("inf"), "finite": False}
        real, real2 = toy.sample(c, geom, tick, rngs[1]), toy.sample(c, geom, tick, rngs[2])
        keys = ("joint_sw1", "state_sw1", "action_sw1", "next_state_sw1", "coverage", "precision",
                "spread_ratio", "consistency_mean", "consistency_p95")
        test = metrics(x, real, scaler, groups)
        row = {k: test[k] for k in keys}
        row["floor_joint_sw1"] = metrics(real2, real, scaler, groups)["joint_sw1"]
        interpolation = groups < 6*len(ticks)
        for name, mask in (("interp", interpolation), ("extrap", ~interpolation)):
            row[f"{name}_joint_sw1"] = metrics(x[mask], real[mask], scaler, groups[mask])["joint_sw1"]
        if nets.encoder is not None:
            errors = []
            for j in range(0, len(c), 256):
                sl = slice(j, j+256)
                decoded, _ = encoded_transition(nets.encoder, nets.generator, nets.prior,
                                                scaler(real[sl])[:, :4], c[sl], context[sl])
                errors.append((scaler.inverse(decoded)[:, 4:]-real[sl, 4:]).norm(dim=1))
            row["next_l2"] = float(torch.cat(errors).mean())
        return row

    def verdict(self, metrics):
        if metrics["joint_sw1"] < PASS_JOINT_SW1 and metrics["consistency_mean"] < PASS_RESIDUAL:
            return "PASS"
        return "FAIL"


if __name__ == "__main__":
    import argparse
    # The shared CLI owns --device; the problem needs it up front to fit its normalization there.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--device", default="cpu")
    known, _ = peek.parse_known_args()
    raise SystemExit(main(TransitionGAN(device=known.device)))
