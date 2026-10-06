"""927 focused ROOT-paid CPU controls. AUTHORED_NOT_RUN.

The real isolated constructor is used in every fixture. Synthetic gradients and
moments exercise optimizer branches; they are not original scientific grades.
"""
from copy import deepcopy
from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OWNER_PIN = ('0fdd73021ab38f660b8c1ee4b292e32ef9fa39077751ba85d7679a64d283a587', 68998)
CONTRACT_PIN = ('7a2be1b94d723776a4a938868cefc8cc6962fd37946d5b41e1fe0c49ea15e01d', 10860)
HELPER_PIN = ('52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 29610)
LR_TEST_PIN = ('3da79006816bcb88488333c200f55077af5d6e3c97f2dafbbedd863fc7279a40', 13934)
RECORDER_PIN = ('0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 23311)
CHECKS = ('registry_base_recipe_and_task_bound', 'fresh_owned_table_only_group_override',
    'populated_adam_denominator_branch', 'completed_lr_clock_and_q_preserved',
    'checkpoint_recorder_and_roundtrip_flags', 'closure_and_failure_forwarding',
    'wrong_variant_or_role_rejected', 'original_task_gate_and_core_scope')
LAST_OPERATIONS = None
LAST_VARIANT = None
LAST_RESOURCES = None


def _load(relative, name, pin):
    path = ROOT / relative
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != pin:
        raise ValueError('focused927 Source differs: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ParticleAMSGradControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.helper = _load('tests/test_atlas889_two_pole_observer.py', '_atlas927_fixture', HELPER_PIN)
        cls.lr_tests = _load('tests/test_atlas921_completed_lr_clock.py', '_atlas927_lr_checks', LR_TEST_PIN)
        cls.adapter = _load('experiments/forge/atlas927_two_pole_owner.py',
            'experiments.forge.atlas927_two_pole_owner', OWNER_PIN)
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
            raw = (ROOT / 'experiments/forge/atlas927_two_pole_owner.py').read_bytes()
            self.assertEqual((hashlib.sha256(raw).hexdigest(), len(raw)), OWNER_PIN)
        return self.adapter.construct_owner(ROOT, self.binding, source_guard=guard)

    def owner(self, *, corrected=True, boost=True):
        if corrected is not True:
            raise ValueError('927 controls never silently choose an old owner factory')
        owner = self.fresh()
        torch = owner.torch
        owner.real = torch.linspace(-.4, .7, 12).reshape(12, 1)
        with torch.no_grad():
            owner.table.copy_(torch.linspace(-.12, .15, 12).reshape(12, 1))
        gradient = torch.linspace(-.2, .2, 12).reshape(12, 1)
        owner.table.grad = gradient.clone()
        state = owner.opt_g.state[owner.table]
        # Deliberately unequal raw-v/running-max, both retained before the step.
        state.update(step=torch.tensor(3.), exp_avg=torch.zeros_like(owner.table),
            exp_avg_sq=torch.full_like(owner.table, .0004),
            max_exp_avg_sq=torch.full_like(owner.table, .02))
        owner.opt_g.direct_response.started = True
        centered = (gradient - gradient.mean(dim=0, keepdim=True)).flatten()
        owner.opt_g.direct_history.copy_(centered if boost else -centered)
        owner.policy.log_output_sigma.grad = torch.full_like(owner.policy.log_output_sigma, .03)
        for group, tester in owner.policy.lr_settle.pairs(owner.policy.optimizers):
            tester.begin(group['params'])
        return owner

    def _event(self, recorder, name):
        return self.lr_tests.CompletedLRClockControls._event(self, recorder, name)

    def test_registry_base_recipe_and_task_bound(self):
        from experiments.forge.contracts import validate_idea
        from experiments.forge.api import task_formulation_context
        validate_idea(self.candidate)
        context = task_formulation_context(self.candidate, self.task, self.protocol, root=ROOT)
        self.assertEqual(context.ordinary_two_pole_binding, self.binding)
        self.assertEqual(self.binding['base_recipe_sha256'], self.adapter.BASE_RECIPE_SHA256)
        self.assertEqual(self.binding['effective_optimizer_variant'], self.adapter.OPTIMIZER_VARIANT)
        self.assertEqual(self.binding['effective_optimizer_variant_sha256'],
            self.adapter.digest(self.adapter.OPTIMIZER_VARIANT))
        self.assertEqual(self.adapter.digest(self.binding['recipe']), self.adapter.BASE_RECIPE_SHA256)
        self.assertEqual(len(self.binding['recipe']), 79)
        self.assertEqual(self.task, self.adapter.TASK_DECLARATION)
        wrong = deepcopy(self.candidate)
        wrong['claim_contract']['optimizer_variant']['groups'][0]['amsgrad'] = True
        self.assertFalse(self.adapter.supports_candidate(wrong))
        self.assertTrue(self.adapter.blockers(self.task, wrong, ROOT))

    def test_fresh_owned_table_only_group_override(self):
        global LAST_VARIANT, LAST_RESOURCES
        owner = self.fresh()
        initial = self.adapter.owner_initial_receipt(owner)
        measured = self.adapter.optimizer_variant_receipt(owner)
        self.assertEqual(initial['optimizer_variant'], measured)
        self.assertEqual(measured['base_recipe_sha256'], self.adapter.BASE_RECIPE_SHA256)
        self.assertEqual(measured['effective_optimizer_variant'], self.adapter.OPTIMIZER_VARIANT)
        self.assertEqual(measured['effective_optimizer_variant_sha256'], self.adapter.digest(self.adapter.OPTIMIZER_VARIANT))
        self.assertIs(owner.opt_g.param_groups[0]['amsgrad'], False)
        self.assertIs(owner.opt_g.param_groups[1]['amsgrad'], True)
        self.assertIs(owner.opt_d.param_groups[0]['amsgrad'], True)
        self.assertIs(owner.opt_g.defaults['amsgrad'], True)
        self.assertIs(owner.opt_d.defaults['amsgrad'], True)
        self.assertIs(owner.recipe.amsgrad, True)
        self.assertEqual(owner.opt_g.state, {})
        self.assertEqual(owner.opt_d.state, {})
        self.assertIs(owner.opt_g.direct_response.params[0], owner.table)
        self.assertEqual(owner.policy.roles, [['table', 'noise'], ['critic']])
        LAST_VARIANT = deepcopy(measured)
        LAST_RESOURCES = {'backend': str(owner.table.device), 'cpu_threads': owner.torch.get_num_threads(),
            'gpus': self.task['resources']['gpus'], 'budget_seconds': self.task['resources']['timeout_seconds']}

    def test_populated_adam_denominator_branch(self):
        owner = self.owner(boost=True)
        torch = owner.torch
        group, noise = owner.opt_g.param_groups
        before = owner.table.detach().clone()
        gradient = owner.table.grad.detach().clone()
        critic_before = [(p.detach().clone(), deepcopy(g)) for g in owner.opt_d.param_groups
                         for p in g['params']]
        reference_table = torch.nn.Parameter(before.clone())
        reference_table.grad = gradient.clone()
        reference_noise = torch.nn.Parameter(owner.policy.log_output_sigma.detach().clone())
        reference_noise.grad = owner.policy.log_output_sigma.grad.detach().clone()
        # A controlled original-AMSGrad optimizer branch, not a baseline case run.
        reference = owner.recipe.make_generator_optimizer(
            [{'params': [reference_table], 'forge_role': 'prior'}], direct_particles=[reference_table])
        reference.add_param_group({**{k: deepcopy(v) for k, v in noise.items() if k != 'params'},
                                   'params': [reference_noise]})
        reference.state[reference_table] = deepcopy(owner.opt_g.state[owner.table])
        reference.direct_response.started = owner.opt_g.direct_response.started
        reference.direct_history.copy_(owner.opt_g.direct_history)
        ref_max_before = reference.state[reference_table]['max_exp_avg_sq'].clone()
        marker = object()
        self.assertIs(owner.opt_g.step(lambda: marker), marker)
        self.assertIs(reference.step(lambda: marker), marker)
        self.assertGreater(owner.opt_g.direct_response.last_gain, 1.)
        self.assertEqual(owner.opt_g.direct_response.last_gain, reference.direct_response.last_gain)
        actual = owner.opt_g.state[owner.table]
        v = actual['exp_avg_sq']
        t = float(actual['step'])
        lr = group['lr'] * owner.opt_g.direct_response.last_gain
        raw_expected = before - lr * gradient / ((v / (1. - .9 ** t)).sqrt() + group['eps'])
        max_expected = before - lr * gradient / ((ref_max_before / (1. - .9 ** t)).sqrt() + group['eps'])
        torch.testing.assert_close(owner.table, raw_expected, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(reference_table, max_expected, rtol=1e-5, atol=1e-7)
        self.assertFalse(torch.equal(owner.table, reference_table))
        self.assertTrue(torch.equal(actual['exp_avg_sq'], reference.state[reference_table]['exp_avg_sq']))
        self.assertTrue(torch.equal(actual['exp_avg'], reference.state[reference_table]['exp_avg']))
        self.assertTrue(torch.equal(actual['max_exp_avg_sq'], ref_max_before))
        self.assertTrue(torch.equal(owner.policy.log_output_sigma, reference_noise))
        self.assertEqual(group['betas'], (0., .999))
        self.assertIs(group['amsgrad'], False)
        self.assertIs(reference.param_groups[0]['amsgrad'], True)
        self.assertIs(noise['amsgrad'], True)
        self.assertEqual(group['lr'], owner.policy.initial_lrs[0][0])
        for (saved, settings), p in zip(critic_before, owner.critic.parameters()):
            self.assertTrue(torch.equal(saved, p))
            self.assertIs(settings['amsgrad'], True)
        # Corrected clock is consumed once; restored beta/q law is unchanged.
        owner.policy._phase = 'generator_step'
        owner.after_generator_step()

    def test_completed_lr_clock_and_q_preserved(self):
        global LAST_OPERATIONS
        self.lr_tests.CompletedLRClockControls.test_boosted_and_unboosted_ratio_only_q_and_other_groups_preserved(self)
        # Compare recorder OFF/ON around the identical new optimizer law.
        arms = []
        purity_failures = 0
        for recorded in (False, True):
            owner = self.owner(boost=True)
            counts, remove = self.helper._spy(owner)
            recorder = self.recorder.PassiveTwoPoleRecorder(owner).install() if recorded else None
            if recorder is not None:
                recorder.current = {'step': 1, 'events': []}
                recorder.records.append(recorder.current)
            try:
                owner.opt_g.step()
                owner.policy._phase = 'generator_step'
                owner.after_generator_step()
                if recorder is not None:
                    self.assertEqual(recorder.unknown, [])
                    self.assertEqual(recorder.purity_failures, 0)
                    purity_failures += recorder.purity_failures
            finally:
                if recorder is not None:
                    recorder.uninstall()
                remove()
            arms.append((deepcopy(counts), self.helper._snapshot(owner)))
        off, on = arms
        self.assertEqual(off, on)
        category = {'extra_prior_samples': 'prior', 'extra_rng_draws': 'rng',
            'extra_model_forwards': 'forwards', 'extra_backward_calls': 'backwards',
            'extra_optimizer_steps': 'optimizer', 'extra_decision_evaluations': 'decisions',
            'state_getter_calls': 'getter', 'foreign_device_initializations': 'foreign'}
        measured = {name: abs(on[0][key] - off[0][key]) for name, key in category.items()}
        measured['state_mutations'] = purity_failures
        self.assertTrue(all(type(v) is int and v == 0 for v in measured.values()))
        LAST_OPERATIONS = measured

    def test_checkpoint_recorder_and_roundtrip_flags(self):
        self.lr_tests.CompletedLRClockControls.test_real_owner_checkpoint_recorder_and_load_refresh_live_group(self)
        owner = self.owner()
        recorder = self.recorder.PassiveTwoPoleRecorder(owner).install()
        try:
            for _ in range(4):
                self.helper._synthetic_update(owner, {'backwards': 0})
            saved = deepcopy(owner.policy.state_dict())
            self.assertIs(saved['optimizers'][0]['param_groups'][0]['amsgrad'], False)
            self.assertIs(saved['optimizers'][0]['param_groups'][1]['amsgrad'], True)
            self.assertIs(saved['optimizers'][1]['param_groups'][0]['amsgrad'], True)
            self.assertNotIn('observe', saved['lr_settle'][0][0])
            self.assertEqual(recorder.unknown, [])
            self.assertEqual(recorder.purity_failures, 0)
        finally:
            recorder.uninstall()
        before_group = owner.opt_g.param_groups[0]
        owner.policy.load_state_dict(saved)
        self.assertIsNot(owner.opt_g.param_groups[0], before_group)
        self.assertIs(owner.opt_g.param_groups[0]['params'][0], owner.table)
        self.assertEqual(self.adapter.optimizer_variant_receipt(owner)['effective_optimizer_variant'], self.adapter.OPTIMIZER_VARIANT)
        self.helper._synthetic_update(owner, {'backwards': 0})
        self.assertEqual(owner.policy.completed_steps, 5)
        self.assertIs(owner.opt_g.param_groups[0]['amsgrad'], False)
        self.assertIs(owner.opt_g.param_groups[1]['amsgrad'], True)
        self.assertIs(owner.opt_d.param_groups[0]['amsgrad'], True)

    def test_closure_and_failure_forwarding(self):
        self.lr_tests.CompletedLRClockControls.test_closure_exception_is_same_object_and_clears_capture(self)
        self.lr_tests.CompletedLRClockControls.test_adam_exception_restores_group_and_cannot_publish_completed_ratio(self)

    def test_wrong_variant_or_role_rejected(self):
        owner = self.fresh()
        g = owner.opt_g.param_groups
        for group in (g[0], g[1], owner.opt_d.param_groups[0]):
            prior = group['amsgrad']
            group['amsgrad'] = not prior
            try:
                with self.assertRaisesRegex(ValueError, 'AMSGrad-off927'):
                    self.adapter.optimizer_variant_receipt(owner)
            finally:
                group['amsgrad'] = prior
        p = g[0]['params'][0]
        g[0]['params'][0] = owner.policy.log_output_sigma
        try:
            with self.assertRaisesRegex(ValueError, 'AMSGrad-off927'):
                self.adapter.optimizer_variant_receipt(owner)
        finally:
            g[0]['params'][0] = p
        self.adapter.optimizer_variant_receipt(owner)
        forged = deepcopy(self.candidate)
        forged['claim_contract']['optimizer_variant']['groups'][0]['amsgrad'] = 0
        self.assertFalse(self.adapter.supports_candidate(forged))

    def test_original_task_gate_and_core_scope(self):
        self.assertEqual(self.task['execution']['steps'], 80)
        self.assertEqual(self.task['evaluation']['observations'], 24)
        self.assertEqual(self.task['evaluation']['minimum_stable_checks'], 5)
        self.assertEqual(self.task['evaluation']['thresholds'], [['mean_abs', '>=', .3], ['grad_med', '<=', 1.]])
        self.assertEqual(self.task['resources']['gpus'], 0)
        self.assertEqual(self.task['resources']['cpu_threads'], 1)
        self.assertEqual(self.task['resources']['timeout_seconds'], 300)
        self.assertEqual(self.protocol['seed'], 0)
        self.assertEqual(self.binding['recipe'], self.helper.RECIPE)
        self.assertIs(self.binding['recipe']['amsgrad'], True)
        for relative, pin in self.helper.CORE_PINS.items():
            self.assertEqual(self.adapter.SOURCE_PINS[relative], {'sha256': pin[0], 'bytes': pin[1]})
        owner = self.fresh()
        self.assertEqual(owner.audit.recipe, owner.recipe)
        self.assertEqual(owner.audit.rows['direct_particle_gain']['calls'], 0)


def control_source_pins():
    expected = {'experiments/forge/atlas927_two_pole_owner.py': OWNER_PIN,
        'experiments/forge/atlas927_contract.py': CONTRACT_PIN,
        'tests/test_atlas889_two_pole_observer.py': HELPER_PIN,
        'tests/test_atlas921_completed_lr_clock.py': LR_TEST_PIN,
        'experiments/forge/atlas889_two_pole_observer.py': RECORDER_PIN}
    result = {}
    for relative, pin in expected.items():
        raw = (ROOT / relative).read_bytes()
        if (hashlib.sha256(raw).hexdigest(), len(raw)) != pin:
            raise ValueError('actual controlled Source changed: ' + relative)
        result[relative] = {'sha256': pin[0], 'bytes': pin[1]}
    own = Path(__file__).resolve()
    if own != ROOT / 'tests/test_atlas927_particle_amsgrad.py':
        raise ValueError('focused927 controls must execute from the installed owned Source')
    raw = own.read_bytes()
    result['tests/test_atlas927_particle_amsgrad.py'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.TestSuite(ParticleAMSGradControls('test_' + name) for name in CHECKS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    good = result.wasSuccessful() and result.testsRun == len(CHECKS) and LAST_VARIANT is not None and LAST_OPERATIONS is not None and LAST_RESOURCES is not None
    record = {'schema': 'forge_atlas927_particle_amsgrad_control_result_v1',
        'status': 'PASS' if good else 'FAIL',
        'checks': dict.fromkeys(CHECKS, 'PASS') if good else {},
        'added_operations': LAST_OPERATIONS,
        'resources': LAST_RESOURCES, 'source_pins': control_source_pins(),
        'base_recipe_sha256': None if LAST_VARIANT is None else LAST_VARIANT['base_recipe_sha256'],
        'effective_optimizer_variant': None if LAST_VARIANT is None else LAST_VARIANT['effective_optimizer_variant'],
        'effective_optimizer_variant_sha256': None if LAST_VARIANT is None else LAST_VARIANT['effective_optimizer_variant_sha256']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('a previous focused927 result must be preserved')
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
