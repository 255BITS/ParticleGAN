"""CPU software regressions for the owned MoG reaction sampler.

Source authored only. ROOT may execute these in an isolated candidate under
paid metadata. These fixtures are not benchmark trials or numerical grades.
"""
from copy import deepcopy
import unittest

import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.birth_death import ParticleBirthDeath, ParticleRows
from particlegan.noisy_particle_prior import NoisyParticlePrior
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior


def fixture_prior(*, sigma=.025, cls=MoGParticlePrior, n=12):
    generator = torch.Generator().manual_seed(27)
    if cls is ParticlePrior:
        prior = cls(n, 2, generator=generator)
    else:
        prior = cls(n, 2, sigma=sigma, standardize=False, generator=generator)
    with torch.no_grad():
        prior.z.copy_(torch.arange(n * 2, dtype=prior.z.dtype).reshape(n, 2) / 17.)
    return prior


def fixture_rows(prior, *, bind=True, generate=None, features=None):
    optimizer = torch.optim.Adam([prior.z], lr=.00425, amsgrad=True)
    return ParticleRows(table=prior.z, optimizer=optimizer,
                        averaged_table=prior.z.detach().clone(),
                        generate=(lambda codes: codes) if generate is None else generate,
                        critic_features=features,
                        mog_prior=prior if bind else None)


class MoGReactionKernelTests(unittest.TestCase):
    def test_positive_width_matches_public_sampler_and_private_rng(self):
        prior = fixture_prior()
        rows = fixture_rows(prior)
        birth = ParticleBirthDeath(rows, seed=309)
        expected_stream = torch.Generator().manual_seed(309)
        table_before = prior.z.detach().clone()
        average_before = rows.averaged_table.clone()
        global_before = torch.get_rng_state().clone()
        expected_codes, expected_ids = prior.sample(12, generator=expected_stream)
        codes, ids = birth._sample_fake_latents()
        self.assertTrue(torch.equal(codes, expected_codes))
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(birth.stream.get_state(), expected_stream.get_state()))
        self.assertFalse(torch.equal(codes, prior.z.detach()[ids]))
        self.assertTrue(torch.equal(prior.z, table_before))
        self.assertTrue(torch.equal(rows.averaged_table, average_before))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_before))
        self.assertFalse(rows.optimizer.state)
        self.assertIs(rows.mog_prior.z, rows.table)
        self.assertAlmostEqual(birth.diagnostics()['fake_pool_prior']['sigma'], .025, places=7)

    def test_zero_width_matches_legacy_rows_and_consumes_no_kernel_rng(self):
        prior = fixture_prior(sigma=0.)
        legacy = fixture_prior(cls=ParticlePrior)
        birth = ParticleBirthDeath(fixture_rows(prior), seed=31)
        old = ParticleBirthDeath(fixture_rows(legacy, bind=False), seed=31)
        codes, ids = birth._sample_fake_latents()
        old_codes, old_ids = old._sample_fake_latents()
        self.assertTrue(torch.equal(codes, old_codes))
        self.assertTrue(torch.equal(ids, old_ids))
        self.assertTrue(torch.equal(birth.stream.get_state(), old.stream.get_state()))

    def test_unbound_particle_contract_preserves_exact_draw_and_config(self):
        prior = fixture_prior(cls=ParticlePrior)
        birth = ParticleBirthDeath(fixture_rows(prior, bind=False), seed=53)
        expected = torch.Generator().manual_seed(53)
        expected_ids = torch.randint(12, (12,), generator=expected)
        codes, ids = birth._sample_fake_latents()
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(codes, prior.z.detach()[expected_ids]))
        self.assertTrue(torch.equal(birth.stream.get_state(), expected.get_state()))
        self.assertEqual(birth._config(), {'table_shape': (12, 2), 'space': 'data',
                                         'isolation': False, 'feature_scale': 'none'})
        self.assertNotIn('fake_pool_prior', birth.diagnostics())

    def test_whole_reaction_pool_includes_kernel_before_existing_noise(self):
        prior = fixture_prior()
        captured = []
        rows = fixture_rows(prior, features=lambda value: captured.append(value.clone()) or value)
        birth = ParticleBirthDeath(rows, seed=87, space='critic')
        birth.dry_run = True
        real = torch.randn(12, 2, generator=torch.Generator().manual_seed(19))
        birth.observe_real(real)
        expected_stream = torch.Generator().manual_seed(87)
        expected_latent, _ = prior.sample(12, generator=expected_stream)
        # Existing reference path consumes the DV12 jitter draw even when
        # controller=None; the displacement is exactly zero in this fixture.
        torch.randn(prior.z.shape, generator=expected_stream)
        output_eps = torch.randn(expected_latent.shape, generator=expected_stream)
        torch.rand(12, dtype=torch.float64, generator=expected_stream)
        expected_fake = expected_latent + .029 * output_eps
        table_before = prior.z.detach().clone()
        average_before = rows.averaged_table.clone()
        global_before = torch.get_rng_state().clone()
        birth.maybe_apply(.029)
        self.assertEqual(len(captured), 3)
        self.assertTrue(torch.equal(captured[0], table_before))
        self.assertTrue(torch.equal(captured[1], expected_fake))
        self.assertTrue(torch.equal(captured[2], real))
        self.assertTrue(torch.equal(birth.stream.get_state(), expected_stream.get_state()))
        self.assertTrue(torch.equal(prior.z, table_before))
        self.assertTrue(torch.equal(rows.averaged_table, average_before))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_before))
        self.assertEqual(birth.counters['evals'], 1)
        self.assertFalse(rows.optimizer.state)

    def test_public_trainer_binds_exact_raw_mog_and_keeps_training_streams(self):
        prior = fixture_prior(n=256)
        recipe = get_recipe('atlas', num_particles=256, z_dim=2,
                            batch_size=128, prior_kind='mog', standardize=False)
        streams = {name: torch.Generator().manual_seed(seed)
                   for name, seed in [('latent_generator', 40), ('noise_generator', 41),
                                      ('prior_noise_generator', 42), ('eval_generator', 43)]}
        trainer = GANTrainer(recipe, nn.Linear(2, 1),
                             nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)),
                             prior=prior, seed=0, **streams)
        trainer.policy._feature_selection.observe_shape(torch.zeros(128, 1))
        self.assertEqual(trainer.policy._feature_selection.state['actual_backend'], 'knn')
        birth = trainer.policy.birth_death
        self.assertIs(birth.rows.mog_prior, prior)
        self.assertIs(birth.rows.table, prior.z)
        self.assertIs(birth.rows.optimizer, trainer.opt_g)
        before = {name: stream.get_state().clone() for name, stream in streams.items()}
        birth._sample_fake_latents()
        for name, stream in streams.items():
            self.assertTrue(torch.equal(stream.get_state(), before[name]), name)
        self.assertEqual(trainer.completed_steps, 0)
        self.assertEqual(trainer.policy._phase, 'ready')
        self.assertFalse(trainer.opt_g.state)
        self.assertFalse(trainer.opt_d.state)
        self.assertTrue(trainer.prior_mechanisms['a2']['enabled'])

    def test_other_prior_types_are_not_silently_bound(self):
        for cls in (ParticlePrior, NoisyParticlePrior):
            with self.subTest(cls=cls.__name__):
                prior = fixture_prior(cls=cls)
                with self.assertRaisesRegex(ValueError, 'exact raw MoG'):
                    fixture_rows(prior)
                # The unchanged optional/default contract remains available.
                self.assertIsNone(fixture_rows(prior, bind=False).mog_prior)

    def test_wrong_table_standardized_or_mutable_width_is_rejected(self):
        prior = fixture_prior()
        foreign = fixture_prior()
        with self.assertRaisesRegex(ValueError, 'exact raw MoG'):
            ParticleRows(table=foreign.z, optimizer=torch.optim.Adam([foreign.z]),
                         averaged_table=foreign.z.detach().clone(), generate=lambda x: x,
                         mog_prior=prior)
        prior.standardize = True
        with self.assertRaisesRegex(ValueError, 'exact raw MoG'):
            fixture_rows(prior)
        prior.standardize = False
        prior.sigma.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'fixed finite'):
            fixture_rows(prior)

    def test_legacy_base_key_state_refused_only_for_corrected_mog(self):
        base_keys = (*ParticleBirthDeath._TENSORS, 'fill', 'cursor', 'rows_since_eval',
                     'counters', 'last', 'stream')
        for cls, bind in ((MoGParticlePrior, True), (ParticlePrior, False)):
            with self.subTest(cls=cls.__name__):
                prior = fixture_prior(cls=cls)
                birth = ParticleBirthDeath(fixture_rows(prior, bind=bind), seed=62)
                full = birth.state_dict()
                legacy = {key: deepcopy(full[key]) for key in base_keys}
                if bind:
                    with self.assertRaisesRegex(ValueError, 'exact kernel configuration'):
                        birth.check_state(legacy)
                else:
                    birth.check_state(legacy)

    def test_checkpoint_kernel_identity_cannot_be_omitted_or_changed(self):
        prior = fixture_prior()
        birth = ParticleBirthDeath(fixture_rows(prior), seed=61)
        state = birth.state_dict()
        birth.check_state(state)
        for mutation in ('omitted', 'changed_width', 'foreign_table'):
            with self.subTest(mutation=mutation):
                bad = deepcopy(state)
                if mutation == 'omitted':
                    del bad['config']['fake_pool_prior']
                elif mutation == 'changed_width':
                    bad['config']['fake_pool_prior']['sigma'] = .5
                else:
                    bad['config']['fake_pool_prior']['same_table_parameter'] = False
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    birth.check_state(bad)


if __name__ == '__main__':
    unittest.main()
