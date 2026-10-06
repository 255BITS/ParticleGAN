"""Three genuine 932 CPU software controls. AUTHORED_NOT_RUN.

No original target training/evaluation loop or previous control suite is run.
"""
from copy import deepcopy
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OWNER_PIN = ('ced28b3fe73d1fa74a000b1f10552ada7bc0f703cd8c94c6fff6ce8d04841a07', 76162)
CONTRACT_PIN = ('2d324291e85d40564e3fd11ed3d5e6130dace9cb14f1080bc6529a427caf36fb', 14313)
OPTIMIZER_TEST_PIN = ('64e562da2a8fe3a5412f78c8fcf23eaeaa86ddf4b5b7b693ca508bc6d1ded2e5', 19154)
HELPER_PIN = ('52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 29610)
LR_TEST_PIN = ('3da79006816bcb88488333c200f55077af5d6e3c97f2dafbbedd863fc7279a40', 13934)
RECORDER_PIN = ('0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 23311)
CHECKS = ('declared_objective_and_actual_owner_coefficient',
    'pinned_loss_coefficient_gradient_delta', 'preserved_optimizer_clock_and_recorder_checkpoint')
LAST_OBJECTIVE = None
LAST_OPTIMIZER = None
LAST_RESOURCES = None
LAST_OPERATIONS = None


def _load(relative, name, pin):
    path = ROOT / relative
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != pin:
        raise ValueError('focused932 Source differs: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ParticleL2Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.adapter = _load('experiments/forge/atlas932_two_pole_owner.py',
            'experiments.forge.atlas932_two_pole_owner', OWNER_PIN)
        cls.optimizer_tests = _load('tests/test_atlas927_particle_amsgrad.py', '_atlas932_optimizer_checks', OPTIMIZER_TEST_PIN)
        cls.helper = _load('tests/test_atlas889_two_pole_observer.py', '_atlas932_fixture', HELPER_PIN)
        cls.lr_tests = _load('tests/test_atlas921_completed_lr_clock.py', '_atlas932_lr_checks', LR_TEST_PIN)
        cls.recorder = _load('experiments/forge/atlas889_two_pole_observer.py',
            'experiments.forge.atlas889_two_pole_observer', RECORDER_PIN)
        cls.candidate = json.loads((ROOT / ('configs/forge/ideas/' + cls.adapter.CANDIDATE_ID + '.json')).read_bytes())
        cls.task = json.loads(cls.adapter._source(ROOT, cls.adapter.TASK_PATH))
        cls.protocol = json.loads(cls.adapter._source(ROOT, cls.adapter.PROTOCOL_PATH))
        cls.binding = cls.adapter.resolve_binding(ROOT, cls.candidate, cls.task, cls.protocol)

    def fresh(self):
        def guard():
            for relative in self.adapter.SOURCE_PINS:
                self.adapter._source(ROOT, relative)
            raw = (ROOT / 'experiments/forge/atlas932_two_pole_owner.py').read_bytes()
            self.assertEqual((hashlib.sha256(raw).hexdigest(), len(raw)), OWNER_PIN)
        return self.adapter.construct_owner(ROOT, self.binding, source_guard=guard)

    def owner(self, **kwargs):
        return self.optimizer_tests.ParticleAMSGradControls.owner(self, **kwargs)

    def _event(self, recorder, name):
        return self.lr_tests.CompletedLRClockControls._event(self, recorder, name)

    def test_declared_objective_and_actual_owner_coefficient(self):
        global LAST_OBJECTIVE, LAST_OPTIMIZER, LAST_RESOURCES
        from experiments.forge.contracts import validate_idea
        from experiments.forge.api import task_formulation_context
        validate_idea(self.candidate)
        context = task_formulation_context(self.candidate, self.task, self.protocol, root=ROOT)
        self.assertEqual(context.ordinary_two_pole_binding, self.binding)
        self.assertEqual(self.adapter.public_parent_task(self.task, ROOT), self.adapter.ORIGINAL_TASK_DECLARATION)
        self.assertNotEqual(self.task['id'], 'two_pole')
        self.assertEqual(self.task['evaluation'], self.adapter.ORIGINAL_TASK_DECLARATION['evaluation'])
        self.assertEqual(self.task['resources'], self.adapter.ORIGINAL_TASK_DECLARATION['resources'])
        self.assertEqual(self.task['execution']['steps'], 80)
        self.assertEqual(self.binding['effective_objective_variant'], self.adapter.OBJECTIVE_VARIANT)
        self.assertEqual(self.binding['effective_optimizer_variant'], self.adapter.OPTIMIZER_VARIANT)
        owner = self.fresh()
        initial = self.adapter.owner_initial_receipt(owner)
        self.assertIs(type(owner.particle_l2), float)
        self.assertEqual(owner.particle_l2, 0.)
        LAST_OBJECTIVE = self.adapter.objective_receipt(owner)
        LAST_OPTIMIZER = self.adapter.optimizer_variant_receipt(owner)
        LAST_RESOURCES = {'backend': str(owner.table.device), 'cpu_threads': owner.torch.get_num_threads(),
            'gpus': self.task['resources']['gpus'], 'budget_seconds': self.task['resources']['timeout_seconds']}
        self.assertEqual(initial['objective_variant'], LAST_OBJECTIVE)
        self.assertIs(owner.opt_g.param_groups[0]['amsgrad'], False)
        self.assertIs(owner.opt_g.param_groups[1]['amsgrad'], True)
        self.assertIs(owner.opt_d.param_groups[0]['amsgrad'], True)
        with self.assertRaises(ValueError):
            self.adapter.resolve_binding(ROOT, self.candidate, self.adapter.ORIGINAL_TASK_DECLARATION, self.protocol)
        bad = deepcopy(self.candidate)
        bad['claim_contract']['objective_variant']['particle_l2'] = .02
        self.assertFalse(self.adapter.supports_candidate(bad))
        with self.assertRaises(ValueError):
            self.adapter.resolve_binding(ROOT, bad, self.task, self.protocol)
        owner.particle_l2 = .02
        with self.assertRaisesRegex(ValueError, 'exactly zero'):
            self.adapter.objective_receipt(owner)

    def test_pinned_loss_coefficient_gradient_delta(self):
        zero, reference = self.fresh(), self.fresh()
        torch = zero.torch
        from benchmarks.locked_shared import two_pole as host
        raw = self.adapter._source(ROOT, self.adapter.HOST_PATH)
        train = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        assignments = [n for n in ast.walk(train) if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'g_loss'
            and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Add)]
        self.assertEqual(len(assignments), 1)
        expression = assignments[0].value
        self.assertEqual(ast.unparse(expression), 'g_loss + particle_l2 * particles.square().mean()')
        compiled = compile(ast.Expression(expression), '<pinned-original-L2-expression>', 'eval')
        geometry = torch.linspace(-.12, .15, 12).reshape(12, 1)
        values, generated_banks = [], []
        for owner, coefficient in ((zero, zero.particle_l2),
                (reference, self.adapter.OBJECTIVE_VARIANT['reference_particle_l2'])):
            with torch.no_grad():
                owner.table.copy_(geometry)
            owner.bind_real(host.real_batch(12))
            owner.noise.set_step(0)  # original begin_step connects StepNoise.output_sigma
            owner.opt_g.zero_grad(set_to_none=True)
            owner.opt_d.zero_grad(set_to_none=True)
            # Actual declared public GAN/noise path; only the pinned L2 scalar differs.
            generated = owner.policy.generate(owner.table, rows=owner.row_indices)
            gan_loss = owner.loss.g_loss(owner.critic(generated), owner.critic(owner.real).detach())
            loss = eval(compiled, {'__builtins__': {}},
                {'g_loss': gan_loss, 'particle_l2': coefficient, 'particles': owner.table})
            loss.backward()
            values.append((gan_loss.detach(), loss.detach()))
            generated_banks.append(generated.detach().clone())
        self.assertTrue(torch.equal(generated_banks[0], generated_banks[1]))
        self.assertTrue(torch.equal(values[0][0], values[1][0]))
        torch.testing.assert_close(values[1][1] - values[0][1], .02 * geometry.square().mean(), rtol=1e-5, atol=1e-7)
        expected = 2. * .02 * geometry / geometry.numel()
        self.assertGreater(float(expected.abs().max()), 0.)
        torch.testing.assert_close(reference.table.grad - zero.table.grad, expected, rtol=1e-5, atol=1e-7)
        self.assertIsNotNone(zero.policy.log_output_sigma.grad)
        self.assertIsNotNone(reference.policy.log_output_sigma.grad)
        self.assertTrue(torch.equal(zero.policy.log_output_sigma.grad, reference.policy.log_output_sigma.grad))
        for a, b in zip(zero.critic.parameters(), reference.critic.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))
        self.assertEqual(zero.streams.audit(), reference.streams.audit())
        # Pure software reference, not a constructed .02 scientific candidate.
        self.assertEqual(zero.particle_l2, 0.)
        self.assertEqual(reference.particle_l2, 0.)

    def test_preserved_optimizer_clock_and_recorder_checkpoint(self):
        global LAST_OPERATIONS
        # Fresh coefficient-zero owners; fixture helpers are reused, test methods are not.
        arms = []
        saved = None
        loaded_owner = None
        purity_failures = None
        for recorded in (False, True):
            owner = self.owner(boost=True)
            self.assertEqual(owner.particle_l2, 0.)
            # Install on the fresh owner before spies, so serializer-hook
            # prior values remain the original instance attributes.
            recorder = self.recorder.PassiveTwoPoleRecorder(owner).install() if recorded else None
            counts, remove = self.helper._spy(owner)
            try:
                try:
                    # One bounded software update using genuine lifecycle hooks.
                    # The held table gradient preserves the populated direct-gain
                    # fixture; this is not the scientific target/evaluation loop.
                    held_gradient = owner.table.grad.detach().clone()
                    owner.noise.set_step(owner.policy.completed_steps)
                    owner.opt_d.zero_grad(set_to_none=True)
                    fake = owner.noise.output(owner.table, generator_step=False)
                    loss_d = owner.critic(owner.real).square().mean() + owner.critic(fake).square().mean()
                    owner.policy.before_critic_backward()
                    counts['backwards'] += 1
                    loss_d.backward()
                    owner.opt_d.step()
                    owner.after_critic_step()
                    owner.opt_g.zero_grad(set_to_none=True)
                    loss_g = (owner.table * held_gradient).sum() + .03 * owner.policy.log_output_sigma
                    owner.policy.before_generator_backward()
                    counts['backwards'] += 1
                    loss_g.backward()
                    owner.after_generator_backward(loss_g, loss_d)
                    sentinel = object()
                    self.assertIs(owner.opt_g.step(lambda: sentinel), sentinel)
                    owner.after_generator_step()
                    if recorder is not None:
                        actual = self._event(recorder, 'actual_adam_inputs')['point']
                        restored = self._event(recorder, 'after_direct_group_restore')['group']
                        ratio = self._event(recorder, 'table_tester_actual_input')['ratio']
                        self.assertEqual(ratio, actual['group']['lr'] / actual['group']['base_lr'])
                        self.assertGreater(ratio, restored['lr'] / restored['base_lr'])
                        q = self._event(recorder, 'table_group_q_actual_return')
                        self.assertEqual(q['consumer_group']['betas'], [0., .999])
                        self.assertEqual(q['consumer_group']['lr'], restored['lr'])
                        point = q['point']
                        t = point['optimizer_step']['values']
                        expected_q = sum(abs(g[0]) / (math.sqrt(v[0] / (1 - .999 ** t))
                            + q['consumer_group']['eps']) for g, v in zip(
                            point['training_gradient']['values'], point['exp_avg_sq']['values'])) / 12
                        self.assertAlmostEqual(q['q']['values'], expected_q, places=5)
                        self.assertEqual(recorder.calls['direct_end'], 1)
                        self.assertEqual(recorder.calls['table_tester_observe'], 1)
                        self.assertEqual(recorder.calls['table_group_q'], 1)
                        self.assertEqual(recorder.unknown, [])
                        self.assertEqual(recorder.purity_failures, 0)
                        purity_failures = recorder.purity_failures
                    owner.finish_step()
                    self.assertEqual(owner.policy.completed_steps, 1)
                    self.assertEqual(owner.policy._phase, 'ready')
                    self.assertEqual(owner.calls, dict.fromkeys(owner.calls, 1))
                    if recorder is not None:
                        self.assertEqual(recorder.unknown, [])
                        self.assertEqual(recorder.purity_failures, 0)
                        purity_failures = recorder.purity_failures
                finally:
                    # Spies were attached AFTER the recorder; their own remove
                    # restores the recorder's exact wrappers and identities.
                    remove()
                checkpoint_recorder = recorder  # same fresh-installed instance
                checkpoint = deepcopy(owner.policy.state_dict())
                self.assertIs(checkpoint['optimizers'][0]['param_groups'][0]['amsgrad'], False)
                self.assertIs(checkpoint['optimizers'][0]['param_groups'][1]['amsgrad'], True)
                self.assertIs(checkpoint['optimizers'][1]['param_groups'][0]['amsgrad'], True)
                self.assertNotIn('observe', checkpoint['lr_settle'][0][0])
                if checkpoint_recorder is not None:
                    self.assertEqual(checkpoint_recorder.unknown, [])
                    self.assertEqual(checkpoint_recorder.purity_failures, 0)
                    purity_failures += checkpoint_recorder.purity_failures
                    saved, loaded_owner = checkpoint, owner
            finally:
                if recorder is not None:
                    recorder.uninstall()
            arms.append((deepcopy(counts), self.helper._snapshot(owner)))
        self.assertEqual(arms[0], arms[1])
        categories = {'extra_prior_samples': 'prior', 'extra_rng_draws': 'rng',
            'extra_model_forwards': 'forwards', 'extra_backward_calls': 'backwards',
            'extra_optimizer_steps': 'optimizer', 'extra_decision_evaluations': 'decisions',
            'state_getter_calls': 'getter', 'foreign_device_initializations': 'foreign'}
        LAST_OPERATIONS = {name: abs(arms[1][0][key] - arms[0][0][key])
            for name, key in categories.items()}
        LAST_OPERATIONS['state_mutations'] = purity_failures
        self.assertTrue(all(type(v) is int and v == 0 for v in LAST_OPERATIONS.values()))
        previous = loaded_owner.opt_g.param_groups[0]
        loaded_owner.policy.load_state_dict(saved)
        self.assertIsNot(loaded_owner.opt_g.param_groups[0], previous)
        self.assertIs(loaded_owner.opt_g.param_groups[0]['params'][0], loaded_owner.table)
        self.adapter.objective_receipt(loaded_owner)
        self.assertEqual(self.adapter.optimizer_variant_receipt(loaded_owner)[
            'effective_optimizer_variant'], self.adapter.OPTIMIZER_VARIANT)


def control_source_pins():
    expected = {'experiments/forge/atlas932_two_pole_owner.py': OWNER_PIN,
        'experiments/forge/atlas932_contract.py': CONTRACT_PIN,
        'tests/test_atlas927_particle_amsgrad.py': OPTIMIZER_TEST_PIN,
        'tests/test_atlas889_two_pole_observer.py': HELPER_PIN,
        'tests/test_atlas921_completed_lr_clock.py': LR_TEST_PIN,
        'experiments/forge/atlas889_two_pole_observer.py': RECORDER_PIN}
    result = {}
    for relative, pin in expected.items():
        raw = (ROOT / relative).read_bytes()
        if (hashlib.sha256(raw).hexdigest(), len(raw)) != pin:
            raise ValueError('actual controlled932 Source changed: ' + relative)
        result[relative] = {'sha256': pin[0], 'bytes': pin[1]}
    own = Path(__file__).resolve()
    if own != ROOT / 'tests/test_atlas932_particle_l2.py':
        raise ValueError('focused932 controls must use the installed owned Source')
    raw = own.read_bytes()
    result['tests/test_atlas932_particle_l2.py'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.TestSuite(ParticleL2Controls('test_' + name) for name in CHECKS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    good = result.wasSuccessful() and result.testsRun == len(CHECKS) and all(
        x is not None for x in (LAST_OBJECTIVE, LAST_OPTIMIZER, LAST_RESOURCES, LAST_OPERATIONS))
    record = {'schema': 'forge_atlas932_particle_l2_control_result_v1',
        'status': 'PASS' if good else 'FAIL', 'checks': dict.fromkeys(CHECKS, 'PASS') if good else {},
        'source_pins': control_source_pins(), 'resources': LAST_RESOURCES, 'added_operations': LAST_OPERATIONS,
        'base_recipe_sha256': None if LAST_OPTIMIZER is None else LAST_OPTIMIZER['base_recipe_sha256'],
        'effective_optimizer_variant': None if LAST_OPTIMIZER is None else LAST_OPTIMIZER['effective_optimizer_variant'],
        'effective_optimizer_variant_sha256': None if LAST_OPTIMIZER is None else LAST_OPTIMIZER['effective_optimizer_variant_sha256'],
        'effective_objective_variant': None if LAST_OBJECTIVE is None else LAST_OBJECTIVE['effective_objective_variant'],
        'effective_objective_variant_sha256': None if LAST_OBJECTIVE is None else LAST_OBJECTIVE['effective_objective_variant_sha256']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise ValueError('preserve the earlier focused932 result')
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
