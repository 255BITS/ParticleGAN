import unittest

import torch

from experiments.train_trajectory import DEFAULTS, validate
from lib.denoising_toy import DiffusionSchedule, DrawSource
from particlegan.grad_regularizers import GradientPenalty
from lib.trajectory import Routes, TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryCritic, generate, metrics


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.rng = torch.Generator().manual_seed(12)
        self.toy = Routes()

    def test_real_support_and_probabilities(self):
        for split in ("train", "test"):
            c, geom = self.toy.contexts(split)
            c, geom = c.repeat_interleave(2048), geom.repeat_interleave(2048, 0)
            x, route = self.toy.sample(c, geom, self.rng)
            d = self.toy.diagnose(x, geom)
            self.assertTrue(bool(d["valid"].all()))
            self.assertTrue(torch.equal(d["route"], route))
            self.assertLess(float(d["tube"].max()), 1e-6)
            for label, p in enumerate(self.toy.upper_probability):
                self.assertLess(abs(float(route[c == label].float().mean())-p), .025)

    def test_collision_between_frames(self):
        x = torch.zeros(1, 2, 64)
        x[:, 0, :32], x[:, 0, 32:] = -1, 1
        geom = torch.tensor([[0., 0., .27]])
        self.assertLess(float(self.toy.diagnose(x, geom)["clearance"]), 0)

    def test_continuous_geometry_support_and_holdout(self):
        toy = Routes(geometry_mode="continuous")
        c, geom, x = toy.batch(8192, self.rng)
        self.assertTrue(bool(toy.diagnose(x, geom)["valid"].all()))
        self.assertTrue(bool((geom >= torch.tensor([-.2, -.12, .22])).all()))
        self.assertTrue(bool((geom <= torch.tensor([.2, .12, .32])).all()))
        self.assertEqual(len(geom.unique(dim=0)), len(geom))
        cc, gg = toy.contexts("test")
        dc, dg = self.toy.contexts("test")
        torch.testing.assert_close(cc, dc)
        torch.testing.assert_close(gg, dg)
        self.assertTrue(bool((gg[-1] > geom.max(0).values).all()))
        self.assertFalse(bool((geom[:, None] == gg[None]).all(2).any()))
        self.assertEqual(set(c.tolist()), {0, 1})

    def test_temporal_discriminator_ucd_and_double_backward(self):
        self.check_feature_discriminator({**DEFAULTS, "d_architecture": "temporal"})

    def test_hybrid_discriminator_ucd_and_double_backward(self):
        self.check_feature_discriminator({**DEFAULTS, "d_architecture": "hybrid",
                                          "d_width": 192, "d_temporal_width": 32})

    def check_feature_discriminator(self, cfg):
        d = TrajectoryDiscriminator(cfg)
        baseline = TrajectoryDiscriminator(DEFAULTS)
        self.assertLess(abs(sum(p.numel() for p in d.parameters()) /
                            sum(p.numel() for p in baseline.parameters()) - 1), .05)
        c, geom, x = self.toy.batch(8, self.rng)
        context = self.toy.condition(geom)
        t = torch.arange(8) % 4 + 1
        real, xt = DiffusionSchedule(cfg["alpha_bar"]).forward_pair(x, t, self.rng)
        torch.testing.assert_close(d(real, c, context, xt, t)[1],
                                   d(real, 1-c, context, xt, 5-t)[1])
        candidate = real.detach().requires_grad_()
        grad = torch.autograd.grad(d(candidate, c, context, xt, t)[0].sum(), candidate)[0]
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.norm()), 0)
        reg = GradientPenalty(kappa=0., lazy_k=4)
        penalty, _ = reg.penalty(TrajectoryCritic(d, c, context, xt, t), real, real+.1,
                                 4, collect_stats=False)
        penalty.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in d.parameters()))
        self.assertGreater(float(d.stages[0][0].weight.grad.norm()), 0)
        if cfg["d_architecture"] == "hybrid":
            self.assertGreater(float(d.global_net[0].weight.grad.norm()), 0)

    def test_metrics_detect_collapse_and_invalid_paths(self):
        c = torch.zeros(512, dtype=torch.long)
        geom = torch.tensor([[0., 0., .27]]).expand(512, -1)
        real, _ = self.toy.sample(c, geom, self.rng)
        x = self.toy.templates(geom)[:, 1]
        m = metrics(self.toy, x, real, c, geom, torch.zeros_like(c))
        self.assertEqual(m["routes_covered"], 1)
        self.assertAlmostEqual(m["route_tv"], .2)
        self.assertEqual(m["contexts"][0]["coefficient_variance_ratio"][1], 0.)
        m = metrics(self.toy, x+5, real, c, geom, torch.zeros_like(c))
        self.assertEqual(m["valid"], 0)
        self.assertEqual(m["routes_covered"], 0)

    def test_shared_posterior_and_gradients(self):
        cfg = {**DEFAULTS, "width": 8, "d_width": 32}
        g, d = TrajectoryGenerator(cfg), TrajectoryDiscriminator(cfg)
        schedule = DiffusionSchedule(cfg["alpha_bar"])
        c, geom, x = self.toy.batch(8, self.rng)
        ctx = self.toy.condition(geom)
        t = torch.arange(8) % 4 + 1
        real, xt = schedule.forward_pair(x, t, self.rng)
        xt.requires_grad_()
        z = torch.randn(8, cfg["z_dim"], requires_grad=True)
        clean = g(z, c, ctx, xt, t)
        fake = schedule.reverse(clean, xt, t, torch.randn_like(x))
        self.assertTrue(torch.equal(fake[t == 1], clean[t == 1]))
        d(fake, c, ctx, xt, t)[0].mean().backward()
        for grad in (z.grad, xt.grad):
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()), 0)
        self.assertEqual(d.ucd_labels(c, t).tolist(), ((t-1)*2+c).tolist())
        # Joint UCD logits must not receive class or time through the backbone.
        l0 = d(real, c, ctx, xt.detach(), t)[1]
        l1 = d(real, 1-c, ctx, xt.detach(), 5-t)[1]
        torch.testing.assert_close(l0, l1)
        d.zero_grad()
        reg = GradientPenalty(kappa=0., lazy_k=4)
        penalty, _ = reg.penalty(TrajectoryCritic(d, c, ctx, xt.detach(), t), real, fake.detach(), 4, collect_stats=False)
        penalty.backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in d.parameters()))

    def test_sampler_determinism_under_intervention(self):
        cfg = {**DEFAULTS, "width": 8}
        g = TrajectoryGenerator(cfg)
        p = DrawSource("learned", 8, cfg["z_dim"], 1, "cpu")
        noise = DrawSource("gaussian", 8, 128, 2, "cpu")
        schedule = DiffusionSchedule(cfg["alpha_bar"])
        c = torch.zeros(8, dtype=torch.long)
        ctx = self.toy.condition(torch.tensor([[0., 0., .27]]).expand(8, -1))
        rngs = [torch.Generator().manual_seed(i) for i in range(3)]
        x = generate(g, p, noise, schedule, c, ctx, rngs, fixed_ids=torch.zeros(4, 8, dtype=torch.long), fixed_random=True)
        torch.testing.assert_close(x, x[:1].expand_as(x))

    def test_validation(self):
        validate(DEFAULTS)
        for override in ({"length": 63}, {"model": "diffusion"}, {"prior": "bad"}, {"steps": 0},
                         {"geometry_mode": "bad"}, {"d_architecture": "bad"}, {"d_temporal_width": 0}):
            with self.assertRaises(ValueError):
                validate({**DEFAULTS, **override})

    def test_alternative_arms_and_noise_gradients(self):
        c, geom, x = self.toy.batch(8, self.rng)
        ctx = self.toy.condition(geom)
        for override in ({"model": "gan"}, {"d_mode": "concat"}, {"noise": "learned"}, {"noise": "fixed"},
                         {"d_architecture": "hybrid", "d_mode": "concat", "d_temporal_width": 8},
                         {"d_architecture": "hybrid", "model": "gan", "d_temporal_width": 8}):
            cfg = {**DEFAULTS, "width": 8, "d_width": 32, **override}
            validate(cfg)
            g, d = TrajectoryGenerator(cfg), TrajectoryDiscriminator(cfg)
            noise = DrawSource(cfg["noise"], 16, 128, 4, "cpu")
            t = torch.full_like(c, 2) if g.diffusion else None
            xt = x if g.diffusion else None
            clean = g(torch.randn(8, cfg["z_dim"]), c, ctx, xt, t)
            fake = DiffusionSchedule(cfg["alpha_bar"]).reverse(clean, xt, t, noise.sample(8, self.rng)[0].reshape_as(x)) if g.diffusion else clean
            score, logits = d(fake, c, ctx, xt, t)
            score.mean().backward()
            self.assertTrue(bool(torch.isfinite(score).all()))
            self.assertGreater(float(g.out.weight.grad.norm()), 0)
            if cfg["noise"] == "learned":
                self.assertGreater(float(noise.table.grad.norm()), 0)
            if cfg["model"] == "gan":
                self.assertEqual(logits.shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
