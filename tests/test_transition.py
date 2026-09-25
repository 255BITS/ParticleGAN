import tempfile
import unittest
from pathlib import Path

import torch

from particlegan import K3PCritic
import yaml

from experiments.train_transition import (DEFAULTS, training_recipe, validate, train,
                                           discriminator_loss, generator_loss, prior_regularization)
from lib.trajectory import Routes
from lib.transition import (Transitions, TransitionScaler, TransitionGenerator,
                            TransitionDiscriminator, TransitionCritics, shuffle_blocks, residual, metrics)
from particlegan import get_recipe


class TransitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.toy = Transitions()
        self.rng = torch.Generator().manual_seed(12)

    def test_adjacent_sampler_matches_routes_and_scaling_roundtrip(self):
        c, geom = self.toy.contexts("train")
        c, geom = c.repeat(63), geom.repeat(63, 1)
        tick = torch.arange(63).repeat_interleave(8)
        paths, _ = Routes().sample(c, geom, torch.Generator().manual_seed(20))
        triple = self.toy.sample(c, geom, tick, torch.Generator().manual_seed(20))
        ids = torch.arange(len(c))
        torch.testing.assert_close(triple[:, :2], paths[ids, :, tick], atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(triple[:, 4:], paths[ids, :, tick+1], atol=2e-7, rtol=1e-6)
        self.assertLess(float(residual(triple).max()), 1e-7)
        scaler = TransitionScaler.fit(self.toy, 4096)
        torch.testing.assert_close(scaler.inverse(scaler(triple)), triple)
        self.assertLess(float(scaler.scale[2]), float(scaler.scale[0])/20)
        self.assertEqual(list(scaler.parameters()), [])

    def test_shared_latent_joint_gradients_and_ucd(self):
        recipe = training_recipe(DEFAULTS)
        prior = recipe.make_prior(num_particles=32)
        g, d = TransitionGenerator(), TransitionDiscriminator(32)
        c, geom, tick, real = self.toy.batch(16, self.rng)
        context = self.toy.condition(geom, tick)
        z, ids = prior.sample(16, self.rng)
        inputs = []
        handles = [b.register_forward_pre_hook(lambda m, args: inputs.append(args[0])) for b in g.branches]
        fake = g(z, c, context)
        for h in handles:
            h.remove()
        self.assertTrue(all(x is inputs[0] for x in inputs))
        torch.testing.assert_close(inputs[0][:, :32], z)
        d(fake, c, context)[0].mean().backward()
        for branch in g.branches:
            for p in branch.parameters():
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.norm()), 0)
        self.assertGreater(float(prior.z.grad[ids].norm()), 0)
        a = d(real, c, context)[1]
        b = d(real, 1-c, context)[1]
        torch.testing.assert_close(a, b)
        d.zero_grad()
        penalty, _ = recipe.make_gradient_penalty(kappa=0).penalty(
            lambda x: d(x, c, context)[0], real, fake.detach(), 1, self.rng, collect_stats=False)
        penalty.backward()
        self.assertGreater(float(d.net[0].weight.grad.norm()), 0)

    def test_controls_preserve_marginals_but_break_joint(self):
        c, geom = self.toy.contexts("test")
        c, geom = c[:2].repeat_interleave(512), geom[:2].repeat_interleave(512, 0)
        tick = torch.full_like(c, 20)
        groups = c.clone()
        real = self.toy.sample(c, geom, tick, self.rng)
        other = self.toy.sample(c, geom, tick, self.rng)
        shuffled = shuffle_blocks(other, groups, self.rng)
        for group in (0, 1):
            torch.testing.assert_close(other[groups == group].sort(0).values, shuffled[groups == group].sort(0).values)
        scaler = TransitionScaler.fit(self.toy, 4096)
        floor = metrics(other, real, scaler, groups)
        bad = metrics(shuffled, real, scaler, groups)
        self.assertGreater(bad["joint_sw1"], floor["joint_sw1"]*2)
        self.assertGreater(bad["consistency_mean"], .1)
        for name in ("state", "action", "next_state"):
            self.assertAlmostEqual(bad[name+"_sw1"], floor[name+"_sw1"], places=6)
        collapsed = metrics(real[:1].expand_as(real), real, scaler, groups)
        self.assertEqual(collapsed["spread_ratio"], 0.)
        self.assertLess(collapsed["coverage"], floor["coverage"])

    def test_default_recipe_and_parameter_budgets(self):
        r = training_recipe(DEFAULTS).to_dict()
        expected = get_recipe(prior_kind='mog', sigma_rel=0.025).to_dict()
        for key in ("z_dim", "num_particles", "num_classes", "conditioning", "total_steps", "batch_size"):
            expected[key] = r[key]
        self.assertEqual(r, expected)
        self.assertEqual(r["num_particles"], 1024)
        self.assertEqual(r["reg_arm"], "k3p")
        sizes = [sum(p.numel() for p in g.parameters()) for g in
                 (TransitionGenerator(), TransitionGenerator(32, "monolithic", 234))]
        self.assertLess(abs(sizes[0]/sizes[1]-1), .01)
        validate(DEFAULTS)
        for bad in (dict(architecture="bad"), dict(steps=0), dict(eval_per_context=1),
                    dict(num_particles=1), dict(critic_mode="bad"), dict(d_conditioning="bad"), dict(marginal_weight=float("nan"))):
            with self.assertRaises(ValueError):
                validate({**DEFAULTS, **bad})

    def test_default_config_matches_winner_and_historical_configs_keep_their_settings(self):
        root = Path(__file__).resolve().parents[1]
        def resolved(name):
            return {**DEFAULTS, **yaml.safe_load((root/'configs/transition'/name).read_text())}
        winner = yaml.safe_load((root/'reports/transition/leaderboard/encoder_shared_state_config.yaml').read_text())
        for candidate in (DEFAULTS, resolved('default.yaml')):
            self.assertEqual({k: v for k, v in candidate.items() if k != 'out_dir'},
                             {k: v for k, v in winner.items() if k != 'out_dir'})
            self.assertEqual(candidate['out_dir'], 'results/transition/default')
        for name in ('ucd_joint.yaml', 'marginals.yaml', 'monolithic.yaml'):
            cfg = resolved(name)
            validate(cfg)
            self.assertFalse(cfg['encoder'])
            self.assertFalse(cfg['shared_state_critic'])
            self.assertEqual(cfg['d_conditioning'], 'ucd')
            self.assertEqual(cfg['g_class_scale'], 1.)
        cfg = resolved('encoder_separate.yaml')
        self.assertTrue(cfg['encoder'])
        self.assertFalse(cfg['shared_state_critic'])

    def test_generator_input_gains_keep_shared_latent_and_parameter_budget(self):
        torch.manual_seed(50)
        base = TransitionGenerator()
        torch.manual_seed(50)
        scaled = TransitionGenerator(class_scale=8., context_scale=4.)
        self.assertEqual(sum(p.numel() for p in base.parameters()),
                         sum(p.numel() for p in scaled.parameters()))
        for a, b in zip(base.parameters(), scaled.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        z = torch.randn(8, 32)
        c, context = torch.arange(8)%2, torch.randn(8, 4)
        seen = []
        handles = [branch.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
                   for branch in scaled.branches]
        scaled(z, c, context)
        for handle in handles:
            handle.remove()
        self.assertTrue(all(x is seen[0] for x in seen))
        torch.testing.assert_close(seen[0][:, :32], z)
        torch.testing.assert_close(seen[0][:, 32:36], 4*context)
        torch.testing.assert_close(seen[0][:, 36:], 8*torch.nn.functional.one_hot(c, 2).float())
        for key in ("class_scale", "context_scale"):
            for value in (0., -1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    validate({**DEFAULTS, "g_"+key: value})
                with self.assertRaises(ValueError):
                    TransitionGenerator(**{key: value})
        # Old state dictionaries need no migration; omitted scales retain unit gain.
        restored = TransitionGenerator()
        restored.load_state_dict(base.state_dict())
        torch.testing.assert_close(restored(z, c, context), base(z, c, context), rtol=0, atol=0)

    def test_concat_class_input_and_penalty_without_ucd(self):
        recipe = training_recipe({**DEFAULTS, "d_conditioning": "concat"})
        self.assertEqual(recipe.conditioning, "conditional")
        d = TransitionCritics(32, "joint_marginals", 16, conditioning="concat")
        c, geom, tick, real = self.toy.batch(16, self.rng)
        context = self.toy.condition(geom, tick)
        fake = torch.randn_like(real).requires_grad_()
        for name, critic in d.critics.items():
            x = d.observation(name, real)
            first = critic(x, c, context)[0]
            changed = critic(x, 1-c, context)[0]
            self.assertGreater(float((first-changed).detach().abs().max()), 1e-6)
            self.assertEqual(critic(x, c, context)[1].shape, (16, 1))
        rngs = {name: torch.Generator().manual_seed(30+i) for i, name in enumerate(d.critics)}
        # Positive UCD weight must not invoke CE on a one-logit concat critic.
        loss, _ = discriminator_loss(d, real, fake.detach(), c, context, recipe.make_loss(),
                                      K3PCritic(recipe, d, None, kappa=0), 1, rngs, recipe.ucd_weight)
        loss.backward()
        for critic in d.critics.values():
            self.assertGreater(float(critic.net[0].weight.grad[:, -2:].norm()), 0)
        d.requires_grad_(False)
        lg, _ = generator_loss(d, real, fake, c, context, recipe.make_loss(), 1.)
        lg.backward()
        self.assertTrue(torch.isfinite(fake.grad).all())
        self.assertGreater(float(fake.grad.norm()), 0)

    def test_marginal_critics_isolation_gradients_and_bcap(self):
        recipe = training_recipe(DEFAULTS)
        d = TransitionCritics(32, "joint_marginals", 16)
        c, geom, tick, real = self.toy.batch(16, self.rng)
        context = self.toy.condition(geom, tick)
        fake = torch.randn_like(real).requires_grad_()
        d.requires_grad_(False)
        total, terms = generator_loss(d, real, fake, c, context, recipe.make_loss(), 1.)
        torch.testing.assert_close(total, terms["joint"] + (terms["state"]+terms["action"]+terms["next_state"])/3)
        for name, start in (("state", 0), ("action", 2), ("next_state", 4)):
            gradient = torch.autograd.grad(terms[name], fake, retain_graph=True)[0]
            self.assertGreater(float(gradient[:, start:start+2].norm()), 0)
            gradient[:, start:start+2] = 0
            self.assertEqual(float(gradient.norm()), 0)
        d.requires_grad_(True)
        rngs = {name: torch.Generator().manual_seed(30+i) for i, name in enumerate(d.critics)}
        loss, terms = discriminator_loss(d, real, fake.detach(), c, context, recipe.make_loss(),
                                         K3PCritic(recipe, d, None, kappa=0), 1, rngs, recipe.ucd_weight)
        loss.backward()
        for critic in d.critics.values():
            self.assertTrue(torch.isfinite(critic.net[0].weight.grad).all())
            self.assertGreater(float(critic.net[0].weight.grad.norm()), 0)
        torch.manual_seed(12)
        joint = TransitionCritics(32)
        torch.manual_seed(12)
        multiple = TransitionCritics(32, "joint_marginals", 16)
        for a, b in zip(joint.critics["joint"].parameters(), multiple.critics["joint"].parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mog_noise_raw_regularization_and_checkpoint(self):
        recipe = training_recipe(DEFAULTS)
        prior = recipe.make_prior()
        self.assertGreater(float(prior.sigma), 0)
        ids = torch.zeros(32, dtype=torch.long)
        z = prior(ids, generator=self.rng)
        self.assertGreater(float(z.detach().std(0).mean()), 0)
        seen = []
        def spread(raw):
            seen.append(raw)
            return recipe.make_prior_regularizer()(raw)
        prior_regularization(prior, ids, spread).backward()
        self.assertIs(seen[0], prior.z)
        self.assertEqual(seen[0].shape[0], 1024)
        restored = recipe.make_prior()
        restored.load_state_dict(prior.state_dict())
        eps = torch.randn(32, 32)
        torch.testing.assert_close(prior(ids, eps=eps), restored(ids, eps=eps))

    def test_cpu_training_checkpoint_and_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = {**DEFAULTS, "device": "cpu", "steps": 2, "batch_size": 8,
                   "encoder": False, "shared_state_critic": False, "d_conditioning": "ucd",
                   "g_class_scale": 8., "g_context_scale": 4.,
                   "critic_mode": "joint_marginals", "marginal_width": 8,
                   "width": 8, "d_width": 16, "eval_per_context": 8,
                   "normalization_samples": 128, "out_dir": folder,
                   "live_log": str(Path(folder)/"live.log")}
            summary = train(cfg)
            self.assertEqual(summary["real_draws"], 32)
            self.assertEqual(set(summary["critic_parameters"]), {"joint", "state", "action", "next_state"})
            self.assertGreater(float(summary["recipe"]["sigma_rel"]), 0)
            for name in ("source.zip", "viewer.html", "transitions.png", "summary.json", "log.txt"):
                self.assertTrue((Path(folder)/name).exists())
            saved = torch.load(Path(folder)/"final.pt", weights_only=True)
            g = TransitionGenerator(32, "branches", 8,
                                    class_scale=saved["config"].get("g_class_scale", 1.),
                                    context_scale=saved["config"].get("g_context_scale", 1.))
            g.load_state_dict(saved["G"])
            self.assertEqual(g.context_scale, 4.)
            self.assertEqual(saved["scaler"]["mean"].shape, (6,))
            with self.assertRaises(FileExistsError):
                train(cfg)


if __name__ == "__main__":
    unittest.main()
