import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


from experiments.train_transition import DEFAULTS, train, training_recipe, generator_loss, discriminator_loss
from lib.transition import (TransitionEncoder, TransitionGenerator, TransitionCritics, TransitionScaler,
                            encoded_transition, composed_transition, Transitions)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.recipe = training_recipe(DEFAULTS)
        self.prior = self.recipe.make_prior(num_particles=32)
        self.g = TransitionGenerator(width=16)
        self.e = TransitionEncoder(width=16)
        self.c = torch.arange(8) % 2
        self.context = torch.randn(8, 4)

    def test_observation_only_encoding_and_composed_gradients(self):
        obs = torch.randn(8, 4, requires_grad=True)
        decoded, encoding = encoded_transition(self.e, self.g, self.prior, obs, self.c, self.context)
        decoded.square().mean().backward()
        for module in (self.e, self.g, self.prior):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0)
        self.assertGreater(float(obs.grad.norm()), 0)
        self.assertTrue(torch.isfinite(obs.grad).all())
        self.assertEqual(float(encoding.kl.sum()), 0)
        centers = self.prior.means()[encoding.indices[:, 0]]
        self.assertLessEqual(float((encoding.codes[:, 0]-centers).detach().abs().max()), 3*float(self.prior.sigma)+1e-6)
        with self.assertRaises(ValueError):
            self.e(torch.randn(8, 6), self.c, self.context, self.prior)
        fake = self.g(self.prior.sample(8)[0], self.c, self.context)
        composed = composed_transition(self.e, self.g, self.prior, fake, self.c, self.context)[0]
        torch.testing.assert_close(composed[:, :4], fake[:, :4])
        grads = torch.autograd.grad(composed[:, 4:].square().mean(), [b[0].weight for b in self.g.branches])
        self.assertTrue(all(float(grad.norm()) > 0 for grad in grads))

    def test_shared_physical_coordinates_time_weighting_and_gradients(self):
        scaler = TransitionScaler(torch.tensor([1., 2., 0., 0., 3., 4.]), torch.tensor([2., 3., 1., 1., 4., 5.]))
        d = TransitionCritics(16, 'joint_marginals', 8, 'concat', shared_state=True, scaler=scaler)
        self.assertIs(d.critic_for('state'), d.critic_for('next_state'))
        self.assertEqual(len(d.critics), 3)
        physical = torch.randn(8, 6)
        physical[:, 4:] = physical[:, :2]
        x = scaler(physical)
        st, ct = d.inputs('state', x, self.context)
        sn, cn = d.inputs('next_state', x, self.context)
        torch.testing.assert_close(st, sn)
        torch.testing.assert_close(cn[:, -1], ct[:, -1]+1/63)
        torch.testing.assert_close(cn[:, :3], ct[:, :3])
        fake = torch.randn(8, 6, requires_grad=True)
        loss, terms = generator_loss(d, x, fake, self.c, self.context, self.recipe.make_loss(), 1.)
        torch.testing.assert_close(loss, terms['joint']+(terms['state']+terms['action']+terms['next_state'])/3)
        loss.backward()
        self.assertGreater(float(fake.grad[:, 4:].norm()), 0)
        rngs = {name: torch.Generator().manual_seed(20+i) for i, name in enumerate(d.roles())}
        opt_d = self.recipe.make_critic_optimizer(d, ema_critic=copy.deepcopy(d))
        loss, _ = discriminator_loss(d, x, fake.detach(), self.c, self.context, self.recipe.make_loss(),
                                     {name: self.recipe.make_critic_penalty(opt_d, kappa=0)
                                      for name, rng in rngs.items()}, 1.)
        d.zero_grad(); loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters()))

    def test_encoder_training_checkpoint_replays_both_paths(self):
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as folder:
                cfg = {**DEFAULTS, 'encoder': True, 'shared_state_critic': shared, 'device': 'cpu',
                       'steps': 2, 'batch_size': 8, 'g_class_scale': 1., 'width': 8, 'd_width': 16, 'marginal_width': 8,
                       'encoder_width': 8, 'critic_mode': 'joint_marginals', 'd_conditioning': 'concat',
                       'eval_per_context': 8, 'normalization_samples': 128, 'out_dir': folder,
                       'live_log': str(Path(folder)/'live.log')}
                summary = train(cfg)
                self.assertEqual(summary['real_draws'], 32)
                saved = torch.load(Path(folder)/'final.pt', weights_only=True)
                g = TransitionGenerator(width=8); g.load_state_dict(saved['G'])
                e = TransitionEncoder(width=8, class_scale=1.); e.load_state_dict(saved['E'])
                prior = self.recipe.make_prior(); prior.load_state_dict(saved['prior'])
                scaler = TransitionScaler(**saved['scaler'])
                data = np.load(Path(folder)/'test_samples.npz')
                inference = np.load(Path(folder)/'test_inference.npz')
                c = torch.from_numpy(data['c'][:8])
                context = Transitions().condition(torch.from_numpy(data['geom'][:8]), torch.from_numpy(data['tick'][:8]))
                real = scaler(torch.from_numpy(data['real'][:8]))
                decoded, _ = encoded_transition(e, g, prior, real[:, :4], c, context)
                torch.testing.assert_close(scaler.inverse(decoded), torch.from_numpy(inference['reconstruction'][:8]), atol=1e-6, rtol=1e-6)
                d = TransitionCritics(16, 'joint_marginals', 8, 'concat', shared_state=shared, scaler=scaler)
                d.load_state_dict(saved['D'])
                fake = scaler(torch.from_numpy(data['x'][:8]))
                synthetic = composed_transition(e, g, prior, fake, c, context)[0]
                torch.testing.assert_close(scaler.inverse(synthetic), torch.from_numpy(inference['synthetic'][:8]), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
