"""Prepared CPU software regressions for ROOT; agents do not execute these."""
import copy
import unittest

import torch
from torch import nn

from particlegan import get_recipe, prior_capabilities, prior_mechanisms
from particlegan.noisy_particle_prior import NoisyParticlePrior
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior
from particlegan.training import GANTrainer


class NoisyParticlePriorTests(unittest.TestCase):
    def make_pair(self, sigma):
        old = MoGParticlePrior(12, 2, sigma=sigma, standardize=False,
                               generator=torch.Generator().manual_seed(1701))
        new = NoisyParticlePrior(12, 2, sigma=sigma,
                                generator=torch.Generator().manual_seed(1701))
        return old, new

    def test_original_mog_initialization_and_draw_law_are_exact(self):
        for sigma in (0., .025):
            with self.subTest(sigma=sigma):
                old, new = self.make_pair(sigma)
                self.assertTrue(torch.equal(old.z, new.z))
                old_indices = torch.Generator().manual_seed(11)
                new_indices = torch.Generator().manual_seed(11)
                old_noise = torch.Generator().manual_seed(12)
                new_noise = torch.Generator().manual_seed(12)
                old_value, old_rows = old.sample(25, old_indices, noise_generator=old_noise)
                new_value, new_rows = new.sample(25, new_indices, noise_generator=new_noise)
                self.assertTrue(torch.equal(old_rows, new_rows))
                self.assertTrue(torch.equal(old_value, new_value))
                self.assertTrue(torch.equal(old_indices.get_state(), new_indices.get_state()))
                self.assertTrue(torch.equal(old_noise.get_state(), new_noise.get_state()))

    def test_kernel_is_pre_generator_and_explicit(self):
        table = nn.Parameter(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
        prior = NoisyParticlePrior.from_table(table, sigma=.025)
        rows = torch.tensor([2, 0, 2])
        eps = torch.tensor([[1., -1.], [0., 2.], [-2., 0.]])
        result = prior(rows, eps=eps)
        self.assertTrue(torch.equal(result, table[rows] + prior.sigma * eps))
        result.sum().backward()
        self.assertTrue(torch.equal(table.grad, torch.tensor([[1., 1.], [0., 0.], [2., 2.]])))
        self.assertEqual(list(prior.named_parameters()), [('z', table)])
        self.assertFalse(prior.sigma.requires_grad)

    def test_wrapping_keeps_parameter_optimizer_alias_and_no_rng_draw(self):
        table = nn.Parameter(torch.zeros(12, 1))
        optimizer = torch.optim.Adam([table], lr=.00425)
        before = torch.get_rng_state().clone()
        prior = NoisyParticlePrior.from_table(table, sigma=0.)
        self.assertIs(prior.z, table)
        self.assertIs(optimizer.param_groups[0]['params'][0], prior.z)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(dict(optimizer.state), {})
        with torch.no_grad():
            table[3].fill_(.75)
        self.assertTrue(torch.equal(prior(torch.tensor([3])), table[3:4]))

    def test_zero_kernel_consumes_no_noise_rng_with_fixed_rows(self):
        prior = NoisyParticlePrior.from_table(nn.Parameter(torch.zeros(12, 1)), sigma=0.)
        indices = torch.Generator().manual_seed(21)
        noise = torch.Generator().manual_seed(22)
        before_indices, before_noise = indices.get_state().clone(), noise.get_state().clone()
        value, rows = prior.sample(12, indices, fixed_first_n=True, noise_generator=noise)
        self.assertTrue(torch.equal(rows, torch.arange(12)))
        self.assertTrue(torch.equal(value, prior.z))
        self.assertTrue(torch.equal(before_indices, indices.get_state()))
        self.assertTrue(torch.equal(before_noise, noise.get_state()))

    def test_fixed_first_n_preserves_kernel_and_explicit_eps(self):
        prior = NoisyParticlePrior.from_table(nn.Parameter(torch.zeros(4, 2)), sigma=.025)
        stream = torch.Generator().manual_seed(44)
        before = stream.get_state().clone()
        eps = torch.ones(4, 2)
        value, rows = prior.sample(4, stream, fixed_first_n=True, eps=eps)
        self.assertTrue(torch.equal(rows, torch.arange(4)))
        self.assertTrue(torch.equal(value, prior.sigma * eps))
        self.assertTrue(torch.equal(before, stream.get_state()))

    def test_same_table_row_replacement_retains_prior_kernel_and_lineage(self):
        table = nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(12, 2))
        prior = NoisyParticlePrior.from_table(table, sigma=.025)
        identity = id(table)
        sigma = prior.sigma.clone()
        # A control mutates the existing row; it does not create another prior.
        with torch.no_grad():
            table[5].copy_(table[1])
        self.assertEqual(id(prior.z), identity)
        self.assertTrue(torch.equal(prior.sigma, sigma))
        self.assertTrue(torch.equal(prior(torch.tensor([5]), eps=torch.zeros(1, 2)), table[1:2]))

    def test_checkpoint_retains_fixed_kernel_without_read_standardization(self):
        _, prior = self.make_pair(.025)
        restored = NoisyParticlePrior(12, 2, sigma=0.)
        restored.load_state_dict(copy.deepcopy(prior.state_dict()))
        self.assertTrue(torch.equal(restored.z, prior.z))
        self.assertEqual(restored.kernel_contract(), prior.kernel_contract())
        with self.assertRaises(ValueError):
            restored.set_extra_state({'sigma_rel': 0., 'standardize': True})
        with self.assertRaises(ValueError):
            restored.set_extra_state({'sigma_rel': .025, 'standardize': False})

    def test_explicit_row_local_capability_and_a2_match_the_original_mog(self):
        old, new = self.make_pair(.025)
        old_cap, new_cap = prior_capabilities(old), prior_capabilities(new)
        self.assertEqual(new_cap['kind'], 'noisy_particle_cloud')
        for key, value in old_cap.items():
            if key != 'kind':
                self.assertEqual(new_cap[key], value)
        self.assertTrue(prior_mechanisms(new, latent_damping_max_rate=1., prior_beta1=0.)['a2']['enabled'])
        self.assertFalse(prior_mechanisms(new, latent_damping_max_rate=1., prior_beta1=.9)['a2']['enabled'])
        class Fake:
            z = new.z
        self.assertFalse(prior_capabilities(Fake())['a2_eligible'])

    def test_factory_requires_declared_absolute_sigma_and_preserves_other_presets(self):
        before = get_recipe('bcap').to_dict()
        recipe = get_recipe('atlas', prior_kind='noisy_particles', standardize=False,
                            num_particles=12, z_dim=2, batch_size=12)
        self.assertTrue(recipe.particle_birth_death)
        self.assertIsNone(recipe.total_steps)
        with self.assertRaises(ValueError):
            recipe.make_prior()
        self.assertIs(type(recipe.make_prior(sigma=.025)), NoisyParticlePrior)
        self.assertEqual(get_recipe('bcap').to_dict(), before)
        self.assertIs(type(get_recipe('bcap', num_particles=12).make_prior()), ParticlePrior)
        with self.assertRaises(ValueError):
            get_recipe('atlas', prior_kind='mog', standardize=False)
        with self.assertRaises(ValueError):
            get_recipe('atlas', prior_kind='noisy_particles', standardize=True)

    def test_public_trainer_keeps_the_actual_noisy_prior_and_named_noise_stream(self):
        recipe = get_recipe('atlas', prior_kind='noisy_particles', standardize=False,
                            num_particles=12, z_dim=2, batch_size=12)
        prior = recipe.make_prior(sigma=.025)
        indices = torch.Generator().manual_seed(31)
        noise = torch.Generator().manual_seed(32)
        trainer = GANTrainer(recipe, nn.Linear(2, 1), nn.Linear(1, 1), prior=prior,
                             latent_generator=indices, prior_noise_generator=noise,
                             max_steps=80)
        self.assertIs(trainer.prior, prior)
        self.assertIs(trainer.policy.prior, prior)
        self.assertIs(trainer.policy.table, prior.z)
        self.assertIs(trainer.prior_noise_generator, noise)
        self.assertIn('prior_noise_generator', trainer._STREAMS)
        self.assertEqual(trainer.completed_steps, 0)
        self.assertEqual(trainer.policy.completed_steps, 0)
        self.assertEqual(trainer.policy._phase, 'ready')
        self.assertIsNone(trainer.recipe.total_steps)
        self.assertEqual(trainer.max_steps, 80)


if __name__ == '__main__':
    unittest.main()
