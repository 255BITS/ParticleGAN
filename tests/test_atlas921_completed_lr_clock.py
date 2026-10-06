"""Focused ROOT-paid CPU software controls; AUTHORED_NOT_RUN.

Reuse the pinned original synthetic fixture, not the original target/evaluator.
No original eleven-method suite is rerun here. Project imports are late.
"""
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
OWNER_PIN = ('c0eb7e20fb6c2caca5c759b54c00bd742a4f2f3b1e7e4a42545ac8233af3aa76', 64151)
HELPER_PIN = ('52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 29610)
RECORDER_PIN = ('0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 23311)


def _load(relative, name, pin):
    path = ROOT / relative
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != pin:
        raise ValueError('focused control Source differs: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CompletedLRClockControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.helper = _load('tests/test_atlas889_two_pole_observer.py', '_atlas921_fixture', HELPER_PIN)
        cls.adapter = _load('experiments/forge/atlas889_two_pole_owner.py', '_atlas921_owner', OWNER_PIN)
        cls.recorder = _load('experiments/forge/atlas889_two_pole_observer.py', '_atlas921_recorder', RECORDER_PIN)

    def owner(self, *, corrected=True, boost=True):
        fixture = self.helper._fixture()
        owner = self.adapter.TwoPoleOwner()
        owner.__dict__.update(vars(fixture))
        owner.calls = {'begin_step': 0, 'after_critic_step': 0,
            'after_generator_backward': 0, 'after_generator_step': 0, 'finish_step': 0}
        owner.observations, owner.purity, owner.retained, owner._reading_step = [], [], [], None
        owner.binding = {'software_fixture': True}
        owner.noisy_observations = []
        torch = owner.torch
        gradient = torch.linspace(-.2, .2, 12).reshape(12, 1)
        owner.table.grad = gradient.clone()
        centered = (gradient - gradient.mean(dim=0, keepdim=True)).flatten()
        owner.opt_g.direct_history.copy_(centered if boost else -centered)
        owner.policy.log_output_sigma.grad = torch.full_like(owner.policy.log_output_sigma, .03)
        for group, tester in owner.policy.lr_settle.pairs(owner.policy.optimizers):
            tester.begin(group['params'])
        if corrected:
            self.adapter.install_completed_table_lr_clock(owner)
        return owner

    def _event(self, recorder, name):
        matches = [e['payload'] for row in recorder.records for e in row['events'] if e['name'] == name]
        self.assertEqual(len(matches), 1, name)
        return matches[0]

    def test_current_adam_values_versions_moments_aliases_rng_and_return_match(self):
        for boost in (False, True):
            with self.subTest(boost=boost):
                original = self.owner(corrected=False, boost=boost)
                corrected = self.owner(boost=boost)
                sentinel = object()
                calls = []
                def closure():
                    calls.append('called')
                    return sentinel
                self.assertIs(original.opt_g.step(closure), sentinel)
                self.assertIs(corrected.opt_g.step(closure), sentinel)
                self.assertEqual(calls, ['called', 'called'])
                # Full reachable numerical leaves, tensor versions, gradients,
                # semantic aliases, modes and global/private/named RNG states.
                self.assertEqual(self.helper._snapshot(original), self.helper._snapshot(corrected))
                self.assertIs(corrected.policy.table_optimizer, corrected.opt_g)
                self.assertIs(corrected.opt_g.param_groups[0]['params'][0], corrected.table)
                self.assertEqual(corrected.opt_g.param_groups[0]['betas'], (0., .999))
                self.assertTrue(corrected.opt_g.param_groups[0]['amsgrad'])

    def test_boosted_and_unboosted_ratio_only_q_and_other_groups_preserved(self):
        for boost in (False, True):
            with self.subTest(boost=boost):
                owner = self.owner(boost=boost)
                recorder = self.recorder.PassiveTwoPoleRecorder(owner).install()
                recorder.current = {'step': 1, 'events': []}
                recorder.records.append(recorder.current)
                other = owner.policy.lr_settle.testers[0][1]
                other_tau = other.tau
                seen = []
                original_observe = owner.policy.surprise.observe
                def observe(key, optimizer, group):
                    seen.append((key, optimizer, group, group['lr'], group['betas']))
                    return original_observe(key, optimizer, group)
                owner.policy.surprise.observe = observe
                try:
                    owner.opt_g.step()
                    owner.policy._phase = 'generator_step'  # synthetic hook boundary
                    owner.after_generator_step()
                    actual = self._event(recorder, 'actual_adam_inputs')['point']
                    restored = self._event(recorder, 'after_direct_group_restore')['group']
                    ratio = self._event(recorder, 'table_tester_actual_input')['ratio']
                    self.assertEqual(ratio, actual['group']['lr'] / actual['group']['base_lr'])
                    self.assertAlmostEqual(ratio, restored['lr'] / restored['base_lr'] * actual['direct_response']['gain'])
                    if boost:
                        self.assertGreater(ratio, restored['lr'] / restored['base_lr'])
                    else:
                        self.assertEqual(ratio, restored['lr'] / restored['base_lr'])
                    self.assertEqual([row[0] for row in seen], ['0.0', '0.1'])
                    for _, optimizer, group, lr, betas in seen:
                        self.assertIs(optimizer, owner.opt_g)
                        self.assertEqual((lr, betas), (group['lr'], group['betas']))
                    q = self._event(recorder, 'table_group_q_actual_return')
                    self.assertEqual(q['consumer_group']['betas'], [0., .999])
                    self.assertEqual(q['consumer_group']['lr'], restored['lr'])
                    point = q['point']
                    t = point['optimizer_step']['values']
                    expected_q = sum(abs(g[0]) / (math.sqrt(v[0] / (1 - .999 ** t)) + q['consumer_group']['eps'])
                        for g, v in zip(point['training_gradient']['values'], point['exp_avg_sq']['values'])) / 12
                    self.assertAlmostEqual(q['q']['values'], expected_q, places=5)
                    self.assertEqual(other.tau, other_tau + owner.opt_g.param_groups[1]['lr'] /
                        owner.policy.initial_lrs[0][1] - other.b)
                    self.assertEqual(recorder.calls['direct_end'], 1)
                    self.assertEqual(recorder.calls['table_tester_observe'], 1)
                    self.assertEqual(recorder.calls['table_group_q'], 1)
                    self.assertEqual(recorder.purity_failures, 0)
                    self.assertEqual(recorder.unknown, [])
                finally:
                    owner.policy.surprise.observe = original_observe
                    recorder.uninstall()

    def test_no_gradient_token_preserves_original_consumer_without_stale_gain(self):
        original = self.owner(corrected=False)
        corrected = self.owner()
        for owner in (original, corrected):
            owner.table.grad = None
            owner.opt_g.step()
            owner.policy._phase = 'generator_step'
            owner.after_generator_step()
        self.assertEqual(self.helper._snapshot(original), self.helper._snapshot(corrected))
        with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
            corrected.policy._settle_observe(0)

    def test_closure_exception_is_same_object_and_clears_capture(self):
        owner = self.owner()
        before = self.helper._snapshot(owner)
        marker = RuntimeError('synthetic closure error')
        def closure():
            raise marker
        with self.assertRaises(RuntimeError) as caught:
            owner.opt_g.step(closure)
        self.assertIs(caught.exception, marker)
        self.assertEqual(self.helper._snapshot(owner), before)
        with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
            owner.policy._settle_observe(0)
        owner.opt_g.step()  # a fresh successful attempt remains usable
        owner.policy._phase = 'generator_step'
        owner.after_generator_step()

    def test_adam_exception_restores_group_and_cannot_publish_completed_ratio(self):
        owner = self.owner()
        module = sys.modules[type(owner.opt_g).__module__]
        original_adam = module._adam_step
        marker = RuntimeError('synthetic Adam error')
        group = owner.opt_g.param_groups[0]
        settings = (group['lr'], group['betas'])
        def fail(optimizer):
            raise marker
        module._adam_step = fail
        try:
            with self.assertRaises(RuntimeError) as caught:
                owner.opt_g.step()
            self.assertIs(caught.exception, marker)
            self.assertEqual((group['lr'], group['betas']), settings)
            with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
                owner.policy._settle_observe(0)
        finally:
            module._adam_step = original_adam
        owner.opt_g.step()
        owner.policy._phase = 'generator_step'
        owner.after_generator_step()

    def test_current_clock_one_shot_and_missing_restoration_fail_closed(self):
        owner = self.owner()
        with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
            owner.policy._settle_observe(0)
        owner.opt_g.step()
        owner.policy.completed_steps = 1
        with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
            owner.policy._settle_observe(0)
        owner.policy.completed_steps = 0
        owner.opt_g.step()
        owner.policy._phase = 'generator_step'
        owner.after_generator_step()
        with self.assertRaisesRegex(RuntimeError, 'completed direct Adam LR'):
            owner.policy._settle_observe(0)
        end = owner.opt_g.direct_response.end
        owner.opt_g.direct_response.end = lambda token: None
        try:
            with self.assertRaisesRegex(RuntimeError, 'restoration witness'):
                owner.opt_g.step()
        finally:
            owner.opt_g.direct_response.end = end

    def test_real_owner_checkpoint_recorder_and_load_refresh_live_group(self):
        import numpy as np
        import random
        owner = self.owner()
        def tensor_bytes(value):
            if isinstance(value, owner.torch.Tensor):
                return {'shape': list(value.shape), 'dtype': str(value.dtype)}, value.detach().contiguous().numpy().tobytes()
            if isinstance(value, np.ndarray):
                return {'shape': list(value.shape), 'dtype': str(value.dtype)}, value.tobytes()
            return None
        owner.tensor_bytes = tensor_bytes
        owner.global_sha256 = lambda: self.adapter.typed_digest(
            (owner.torch.get_rng_state(), random.getstate(), np.random.get_state()), tensor_bytes)
        recorder = self.recorder.PassiveTwoPoleRecorder(owner).install()
        tester = owner.policy.lr_settle.testers[0][0]
        observe_hook = vars(tester)['observe']
        try:
            for _ in range(4):
                self.helper._synthetic_update(owner, {'backwards': 0})
            owner.checkpoint(4, lambda: {'synthetic_measurement': 1.})
            self.assertTrue(owner.purity[0]['pure'])
            saved = deepcopy(owner.policy.state_dict())
            self.assertEqual(len(self.adapter.typed_digest(saved, tensor_bytes)), 64)
            self.assertNotIn('observe', saved['lr_settle'][0][0])
            self.assertIs(vars(tester)['observe'], observe_hook)
            # The unchanged recorder supports serialization while installed;
            # original strict controller loading requires its hooks detached.
            # This control does not invent an active-recorder resume contract.
            self.assertEqual(recorder.purity_failures, 0)
            self.assertEqual(recorder.unknown, [])
            recorder.uninstall()
            self.assertNotIn('observe', vars(tester))
            old_group = owner.opt_g.param_groups[0]
            owner.policy.load_state_dict(saved)
            self.assertIsNot(owner.opt_g.param_groups[0], old_group)
            self.assertIs(owner.opt_g.param_groups[0]['params'][0], owner.table)
            self.assertNotIn('observe', vars(tester))
            self.helper._synthetic_update(owner, {'backwards': 0})
            self.assertEqual(owner.policy.completed_steps, 5)
            self.assertEqual(recorder.purity_failures, 0)
            self.assertEqual(recorder.unknown, [])
        finally:
            recorder.uninstall()
        self.assertNotIn('observe', vars(tester))

    def test_wrong_owner_alias_rejected_before_installation(self):
        owner = self.owner(corrected=False)
        step, end = owner.opt_g.step, owner.opt_g.direct_response.end
        owner.policy.table_optimizer = owner.opt_d
        with self.assertRaisesRegex(ValueError, 'original direct CPU12x1'):
            self.adapter.install_completed_table_lr_clock(owner)
        self.assertEqual(owner.opt_g.step, step)
        self.assertEqual(owner.opt_g.direct_response.end, end)
        self.assertNotIn('_settle_observe', vars(owner.policy))


if __name__ == '__main__':
    unittest.main(verbosity=2)
