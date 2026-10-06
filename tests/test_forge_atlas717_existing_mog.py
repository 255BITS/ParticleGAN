"""ROOT-paid software regression source; not a scientific run or grade."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest

import torch
from particlegan import GANTrainer, get_recipe
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior
from experiments.forge import atlas_existing_mog as bridge
from experiments.forge.api import CapabilityError, FormulationContext, task_formulation_context
from experiments.forge.adapters import adapter_preflight
from experiments.forge.contracts import stable_hash, validate_idea
from experiments.forge.planning import candidate_revision_for
from experiments.forge.views import validate_view

ROOT = Path(__file__).resolve().parents[1]


def task(name):
    return json.loads((ROOT / 'configs/forge/tasks' / (name + '.json')).read_text())


def candidate():
    return json.loads((ROOT / 'configs/forge/ideas' / (bridge.CANDIDATE_ID + '.json')).read_text())


class Atlas717ExistingMoG(unittest.TestCase):
    def test_original_six_payloads_and_fixed_view(self):
        declaration = candidate()
        validate_idea(declaration)
        view = json.loads((ROOT / 'configs/forge/views' / (bridge.VIEW_ID + '.json')).read_text())
        tasks = {name: task(name) for name in bridge.PARENTS}
        validate_view(view, tasks)
        self.assertEqual([row['task'] for row in view['assignments']], list(bridge.PARENTS))
        self.assertEqual(sum(bridge.ALLOWANCES.values()), 2220)
        for name, original in tasks.items():
            with self.subTest(name=name):
                self.assertEqual(bridge.validate(original, root=ROOT)['task_id'], name)
                self.assertEqual(hashlib.sha256((ROOT / 'configs/forge/tasks' / (name + '.json')).read_bytes()).hexdigest(), bridge.PARENTS[name]['raw_sha256'])

    def test_scientific_declaration_changes_are_rejected(self):
        original = task('gaussian1d_acquisition')
        mutations = (('sigma', lambda x: x['execution']['prior'].update(sigma=.03)),
                     ('particles', lambda x: x['execution']['host_definition'].update(particles=128)),
                     ('horizon', lambda x: x['execution'].update(steps=999)),
                     ('threshold', lambda x: x['evaluation']['thresholds'][0].__setitem__(2, 4095)),
                     ('type', lambda x: x['execution']['prior'].update(kind='noisy_particle_cloud')))
        for label, mutate in mutations:
            changed = deepcopy(original); mutate(changed)
            with self.subTest(label=label), self.assertRaises(ValueError):
                bridge.validate(changed)

    def test_scope_requires_distinct_hashed_track_and_unchanged_controls(self):
        fixed = candidate()
        self.assertTrue(bridge.supports_candidate(fixed))
        for name, value in [('id', 'atlas'), ('recipe_preset', 'ka2'), ('recipe_overrides', {'lr': .0085}), ('extensions', {'x': 1})]:
            changed = deepcopy(fixed); changed[name] = value
            self.assertFalse(bridge.supports_candidate(changed))
        changed = deepcopy(fixed); changed['claim_contract']['experimental_track'] = 'atlas717_noisy025'
        self.assertFalse(bridge.supports_candidate(changed))
        fixed['resolved_recipe'] = bridge.EXPECTED_RECIPES['two_pole']
        fixed['prior'] = task('two_pole')['execution']['prior']
        changed = deepcopy(fixed); changed['claim_contract']['experimental_track'] = 'atlas717_noisy025'
        self.assertNotEqual(candidate_revision_for('same-frozen-source', fixed), candidate_revision_for('same-frozen-source', changed))

    def test_current_full79_task_metadata_admits_only_supported_owners(self):
        fixed = candidate()
        protocol = json.loads((ROOT / bridge.PROTOCOL_PATH).read_text())
        for name in bridge.SUPPORTED:
            with self.subTest(name=name):
                context = task_formulation_context(fixed, task(name), protocol, root=ROOT)
                self.assertEqual(stable_hash(asdict(context.recipe)), stable_hash(bridge.EXPECTED_RECIPES[name]))
                self.assertIsNone(context.recipe.total_steps)
                self.assertEqual(adapter_preflight(task(name), fixed, root=ROOT), [])
        for name, reason in bridge.BLOCKED.items():
            with self.subTest(name=name):
                self.assertIn(reason, '; '.join(bridge.blockers(task(name), fixed, root=ROOT)))
                with self.assertRaises(CapabilityError):
                    task_formulation_context(fixed, task(name), protocol, root=ROOT)

    def test_unrelated_context_does_not_get_mog_policy_admission(self):
        with self.assertRaises(CapabilityError):
            FormulationContext(recipe_preset='atlas', prior=dict(kind='mog', sigma=.025, standardize=False, learnable=True))
        metadata = FormulationContext(recipe_preset='atlas', candidate_id=bridge.CANDIDATE_ID,
            prior=dict(kind='mog', sigma=.025, standardize=False, learnable=True))
        with self.assertRaises(CapabilityError):
            metadata.build_trainer(None, None)

    def test_recipe_raw_mog_eligibility_preserves_other_presets(self):
        recipe = get_recipe('atlas', prior_kind='mog', sigma_rel=0., standardize=False,
                            num_particles=12, z_dim=2, batch_size=4)
        self.assertTrue(recipe.particle_birth_death)
        self.assertTrue(recipe.row_evidence_gate)
        prior = recipe.make_prior(sigma=.025)
        self.assertIs(type(prior), MoGParticlePrior)
        with self.assertRaises(ValueError):
            get_recipe('atlas', prior_kind='mog', sigma_rel=0., standardize=True)
        bcap = get_recipe('bcap', prior_kind='mog', sigma_rel=0., standardize=True)
        self.assertFalse(bcap.particle_birth_death)
        self.assertEqual(bcap.reg_arm, 'b_cap')
        self.assertEqual(bcap.reg_coeff, 1.)
        self.assertIs(type(bcap.make_prior(sigma=.025)), MoGParticlePrior)
        self.assertIs(type(get_recipe('atlas').make_prior()), ParticlePrior)

    def test_public_trainer_keeps_exact_mog_parameter_and_kernel(self):
        # Tiny CPU owner is a software constructor check; no update/draw/grade.
        recipe = get_recipe('atlas', prior_kind='mog', sigma_rel=0., standardize=False,
                            num_particles=12, z_dim=2, batch_size=4)
        prior = recipe.make_prior(sigma=.025)
        location = prior.z
        sigma = prior.sigma.detach().clone()
        generator = torch.nn.Linear(2, 2)
        critic = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
        trainer = GANTrainer(recipe, generator, critic, prior=prior, max_steps=2)
        self.assertIs(trainer.prior, prior)
        self.assertIs(type(trainer.prior), MoGParticlePrior)
        self.assertIs(trainer.policy.table, location)
        self.assertIs(trainer.policy.prior.z, location)
        self.assertIs(trainer.policy.table_optimizer, trainer.opt_g)
        self.assertTrue(torch.equal(prior.sigma, sigma))
        self.assertEqual(trainer.completed_steps, 0)
        self.assertEqual(trainer.policy.completed_steps, 0)
        self.assertFalse(trainer.opt_g.state)
        self.assertFalse(trainer.opt_d.state)
        self.assertIsNotNone(trainer.policy.birth_death)
        self.assertIsNotNone(trainer.policy.row_evidence)
        contract = bridge.prior_contract(prior)
        self.assertEqual(contract['code_path'], 'particlegan.particle_prior.MoGParticlePrior')
        self.assertEqual(contract['sigma_dtype'], 'torch.float32')
        self.assertAlmostEqual(contract['sigma'], .025, places=7)

    def test_existing_mog_sampling_is_the_original_public_sampler(self):
        recipe = get_recipe('atlas', prior_kind='mog', sigma_rel=0., standardize=False,
                            num_particles=12, z_dim=2, batch_size=4)
        prior = recipe.make_prior(sigma=.025)
        same = deepcopy(prior)
        one = torch.Generator().manual_seed(137)
        two = torch.Generator().manual_seed(137)
        noise_one = torch.Generator().manual_seed(211)
        noise_two = torch.Generator().manual_seed(211)
        actual = prior.sample(32, generator=one, noise_generator=noise_one)
        reference = same.sample(32, generator=two, noise_generator=noise_two)
        # Public MoG.sample returns (latent sample tensor, component row IDs).
        self.assertIs(type(actual), tuple)
        self.assertIs(type(reference), tuple)
        self.assertEqual(len(actual), 2)
        self.assertEqual(len(reference), 2)
        self.assertTrue(torch.equal(actual[0], reference[0]))
        self.assertTrue(torch.equal(actual[1], reference[1]))
        self.assertTrue(torch.equal(one.get_state(), two.get_state()))
        self.assertTrue(torch.equal(noise_one.get_state(), noise_two.get_state()))

    def test_maintained_runtime_envelope_and_physical_logical_devices(self):
        original = task('gaussian1d_acquisition')
        protocol = json.loads((ROOT / bridge.PROTOCOL_PATH).read_text())
        science = dict(candidate_revision='inert-revision', protocol=protocol,
                       compute=dict(backend='cuda', threads=1, model='INERT_GPU_COHORT'))
        # Exact maintained planning normalization; TaskSpec resources remain unchanged.
        resources = original['resources']
        planned = dict(memory_mb=resources['gpu_memory_mb'], gpus=resources['gpus'],
            **({'host_memory_mb': resources['host_memory_mb']} if 'host_memory_mb' in resources else {}),
            cpu_threads=resources['cpu_threads'], allow_cpu=False, backend='cuda',
            gpu_model=science['compute'].get('model'))
        job = dict(task_id=original['id'], task_ids=[original['id']],
            resources=planned, budget_seconds=resources['timeout_seconds'],
            science=science, compatibility_key=stable_hash(science))
        request = dict(candidate=candidate(), candidate_revision='inert-revision',
            source=dict(digest='inert-source'), protocol=protocol,
            tasks={original['id']: original}, jobs=[job])
        output = ROOT / 'inert-private-attempt-that-is-not-created'
        envelope = dict(schema_version=1, request=deepcopy(request), job=deepcopy(job),
            worker=dict(device='1', directory=str(output), token='inert-token', attempt='inert-attempt'))
        # Matches runtime.py: ordinary request has no _worker; physical1 maps to logicalcuda0.
        self.assertNotIn('_worker', request)
        self.assertEqual(bridge.vector_admission_contract(request, original, output, 'cuda:0',
            envelope, cuda_visible_devices='1')['physical_device'], '1')
        for physical, logical, visible in [('1', 'cuda:1', '1'), ('1', 'cuda:0', '0'),
                                          ('cpu', 'cpu', ''), ('1', 'cuda:0', '0,1')]:
            changed = deepcopy(envelope); changed['worker']['device'] = physical
            with self.subTest(physical=physical, logical=logical, visible=visible), self.assertRaises(ValueError):
                bridge.vector_admission_contract(request, original, output, logical, changed,
                    cuda_visible_devices=visible)
        self.assertNotEqual(job['resources'], original['resources'])
        for changes in ({'memory_mb': 2049}, {'gpu_model': 'foreign'}, {'allow_cpu': True},
                        {'gpus': 0}, {'cpu_threads': 2}, {'backend': 'cpu'}):
            changed = deepcopy(envelope)
            changed['job']['resources'].update(changes)
            changed['request']['jobs'][0] = deepcopy(changed['job'])
            with self.subTest(normalized_resource=changes), self.assertRaises(ValueError):
                bridge.vector_admission_contract(changed['request'], original, output, 'cuda:0', changed,
                    cuda_visible_devices='1')
        for field in ('request', 'job'):
            changed = deepcopy(envelope)
            if field == 'request':changed['request']['candidate_revision'] = 'foreign'
            else:changed['job']['budget_seconds'] = 119
            with self.subTest(field=field), self.assertRaises(ValueError):
                bridge.vector_admission_contract(request, original, output, 'cuda:0', changed,
                    cuda_visible_devices='1')

    def test_missing_or_fabricated_owner_evidence_is_invalid(self):
        self.assertEqual(bridge.validate_evidence(task('gaussian1d_acquisition'), {})['status'], 'INVALID')
        self.assertEqual(bridge.validate_evidence(task('ae_gan_hold'), {})['status'], 'INVALID')
        self.assertEqual(bridge.validate_evidence(task('two_pole'), {})['status'], 'INVALID')


if __name__ == '__main__':
    unittest.main()
