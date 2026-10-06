"""CPU SOFTWARE fixtures only; no learner, scorer or empirical trial.

Authored NOT_RUN. ROOT runs this exact Source under its paid metadata phase.
The real public MoG and birth/death arithmetic are exercised with deterministic
callback fixtures. Populated fast storage and live gradients are read directly;
no checkpoint getter or served swap is used by the observer.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from particlegan.birth_death import ParticleRows, ParticleBirthDeath
from particlegan.particle_prior import MoGParticlePrior
from experiments.forge.atlas844_radius_observer import (
    RadiusObserver, LiveReader, ObserverPurityError, CLOCKS, SOFTWARE_CHECKS,
    EXTRA_OPERATIONS, _row_bytes_differ, _same_bytes, legacy_radius)


ADDED_OPERATIONS = dict.fromkeys(EXTRA_OPERATIONS, 0)


class NeverForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('held', torch.ones(1))
        self.weight.grad = torch.ones_like(self.weight)

    def forward(self, *args, **kwargs):
        raise AssertionError('fixture has no model forward')


class GenerateFixture:
    def __init__(self):
        self.calls = 0

    def __call__(self, latent):
        self.calls += 1
        return latent[:, :1]


@contextmanager
def recorder_fences():
    """Count/deny forbidden operations while the actual recorder work runs.

    Original delegated core execution occurs outside this fence. Deliberate
    tampering negatives are separately tested, never counted as valid paths.
    """
    from contextlib import ExitStack
    def deny(name):
        def refused(*args, **kwargs):
            ADDED_OPERATIONS[name] += 1
            raise AssertionError('observer attempted ' + name)
        return refused
    with ExitStack() as stack:
        for name in ('rand', 'randn', 'randint', 'rand_like', 'randn_like', 'randperm',
                     'multinomial', 'normal', 'bernoulli'):
            stack.enter_context(patch.object(torch, name, deny('extra_rng_draws')))
        stack.enter_context(patch.object(MoGParticlePrior, 'sample', deny('extra_prior_samples')))
        stack.enter_context(patch.object(torch.nn.Module, '_call_impl', deny('extra_model_forwards')))
        stack.enter_context(patch.object(torch.Tensor, 'backward', deny('extra_backward_calls')))
        stack.enter_context(patch.object(torch.autograd, 'backward', deny('extra_backward_calls')))
        stack.enter_context(patch.object(torch.autograd, 'grad', deny('extra_backward_calls')))
        for cls in (torch.optim.Adam, torch.optim.SGD):
            stack.enter_context(patch.object(cls, 'step', deny('extra_optimizer_steps')))
        stack.enter_context(patch.object(torch.nn.Module, 'state_dict', deny('state_getter_calls')))
        stack.enter_context(patch.object(torch.optim.Optimizer, 'state_dict', deny('state_getter_calls')))
        stack.enter_context(patch.object(torch.cuda, '_lazy_init', deny('foreign_device_initializations')))
        stack.enter_context(patch.object(ParticleBirthDeath, 'maybe_apply', deny('extra_decision_evaluations')))
        stack.enter_context(patch.object(ParticleBirthDeath, '_isolation_pick', deny('extra_decision_evaluations')))
        yield


def fixture(output, *, enabled=True, supports='duplicate', bandwidth=.25):
    init = torch.Generator().manual_seed(13)
    prior = MoGParticlePrior(256, 2, init_std=0., sigma=.025,
                             standardize=False, generator=init)
    with torch.no_grad():
        if supports == 'duplicate':
            prior.z[5:, 0] = 2.
        elif supports == 'distinct':
            prior.z[:, 0] = torch.arange(256) * 2.
        elif supports == 'rounded':
            # Exact shortlist distances: five coincident (1,0) rows, then (3,0).
            prior.z[:, 0] = 1.
            prior.z[5:, 0] = 3.
        elif supports != 'zero':
            raise ValueError(supports)
    prior.z.grad = torch.ones_like(prior.z)
    g, d = NeverForward(), NeverForward()
    d.eval()
    opt_g = torch.optim.Adam([prior.z, g.weight], amsgrad=True)
    opt_d = torch.optim.Adam([d.weight])
    opt_g.state[prior.z] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(prior.z),
        exp_avg_sq=torch.ones_like(prior.z), max_exp_avg_sq=torch.ones_like(prior.z))
    opt_g.latent_history = prior.z.detach().clone()
    average = prior.z.detach().clone()
    controller = SimpleNamespace(variant='dv12', latent_bandwidth=bandwidth)
    policy = SimpleNamespace(G=g, D=d, prior=prior, table=prior.z,
        table_optimizer=opt_g, completed_steps=0, _phase='ready', _feature_selection=None,
        _fast={'generator': {'weight': g.weight.detach().clone() + 2.},
               'table': prior.z.detach().clone() + 3.}, after_generator_backward=lambda **kwargs: None)
    generate = GenerateFixture()
    rows = ParticleRows(table=prior.z, optimizer=opt_g, averaged_table=average,
        generate=generate, controller=controller, completed_steps=lambda: policy.completed_steps,
        mog_prior=prior)
    birth = ParticleBirthDeath(rows, seed=85)
    policy.birth_death = birth
    policy._feature_selection = SimpleNamespace(reference_birth_death=birth,
        state={'actual_backend': 'knn'})
    trainer = SimpleNamespace(G=g, D=d, prior=prior, opt_g=opt_g, opt_d=opt_d,
        policy=policy, device=torch.device('cpu'))
    streams = SimpleNamespace(streams={'named': torch.Generator().manual_seed(18)})
    context = SimpleNamespace(streams=streams)
    observer = RadiusObserver(context, trainer, Path(output).resolve(), enabled=enabled).install()
    original_observe = observer._observe
    def measured_observe(callback):
        with recorder_fences():
            return original_observe(callback)
    observer._observe = measured_observe
    return observer, trainer, generate


class Atlas844SoftwareControls(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ordinal = 0

    def make(self, **kwargs):
        self.ordinal += 1
        out = Path(self.tmp.name)/str(self.ordinal)
        out.mkdir()
        return fixture(out, **kwargs)

    def jitter(self, observer, noise):
        latent = observer.birth.rows.table.detach()[[0, 1]].clone()
        original = observer.originals['_jitter']
        seen = []
        def track(a, b):
            self.assertIs(a, latent); self.assertIs(b, noise)
            value = original(a, b); seen.append(value); return value
        observer.originals['_jitter'] = track
        try:
            result = observer.birth._jitter(latent, noise)
        finally:
            observer.originals['_jitter'] = original
        self.assertEqual(len(seen), 1)
        self.assertIs(result, seen[0])
        return result

    def test_duplicate_positive_radius(self):
        observer, _, _ = self.make()
        query = observer.birth.rows.table.detach()[[0]].clone()
        support = observer.birth.rows.table.detach().clone()
        self.assertEqual(float(legacy_radius(query, support)[0]), 0.)
        self.assertEqual(float(observer.birth._nearest_other(query, support)[0]), 1.)
        self.jitter(observer, torch.ones(2, 2))
        self.assertIsNotNone(observer.first_positive)
        self.assertTrue(observer.witness['metadata']['full_query_noise_batch'])
        self.assertEqual(tuple(observer.witness['tensors']['query'].shape), (2, 2))
        self.assertEqual(tuple(observer.witness['tensors']['support'].shape), (256, 2))
        self.assertTrue(observer.witness['metadata']['source_marker_original_noise_draw'])

    def test_distinct_support(self):
        observer, _, _ = self.make(supports='distinct')
        query = observer.birth.rows.table.detach()[[0, 1]].clone()
        support = observer.birth.rows.table.detach().clone()
        self.assertTrue(_same_bytes(legacy_radius(query, support),
                                   observer.birth._nearest_other(query, support)))
        self.jitter(observer, torch.ones(2, 2)*.01)
        self.assertIsNone(observer.first_positive)
        self.assertIsNone(observer.witness)

    def test_all_zero_support(self):
        observer, _, _ = self.make(supports='zero')
        result = self.jitter(observer, torch.ones(2, 2))
        self.assertTrue(_same_bytes(result, torch.zeros_like(result)))
        self.assertIsNone(observer.first_positive)
        self.assertIsNone(observer.witness)

    def test_radius_without_effect(self):
        observer, _, _ = self.make()
        self.jitter(observer, torch.zeros(2, 2))
        first = deepcopy(observer.first_positive)
        self.assertIsNotNone(first)
        self.assertIsNone(observer.witness)
        self.jitter(observer, torch.ones(2, 2))
        self.assertEqual(observer.first_positive, first)
        self.assertEqual(observer.first_affected['event']['call_ordinal'], 2)
        self.assertEqual(observer.witness['metadata']['event']['call_ordinal'], 2)

    def test_clipped_and_rounded_effects(self):
        observer, _, _ = self.make()
        self.jitter(observer, torch.ones(2, 2)*100.)
        self.assertEqual(observer.counts['fake_pool']['new_cap_active'], 2)
        self.assertTrue(observer.witness['metadata']['delta_changed'])
        rounded, _, _ = self.make(supports='rounded')
        # Positive 2^-26 x displacement rounds away at float32 x=1; y stays zero.
        noise = torch.zeros(2, 2)
        noise[:, 0] = 2.**-24
        result = self.jitter(rounded, noise)
        self.assertIsNotNone(rounded.witness)
        captured = rounded.witness['tensors']
        self.assertEqual(captured['query'].dtype, torch.float32)
        self.assertTrue(_same_bytes(captured['old_radius'], torch.zeros(2)))
        self.assertTrue(_same_bytes(captured['new_radius'], torch.ones(2)))
        self.assertTrue(_same_bytes(captured['old_delta'], torch.zeros_like(noise)))
        self.assertTrue(_same_bytes(captured['new_delta'], noise*.25))
        self.assertTrue(_same_bytes(result, noise*.25))
        self.assertTrue(rounded.witness['metadata']['old_zero_new_positive'])
        self.assertTrue(rounded.witness['metadata']['delta_changed'])
        self.assertFalse(rounded.witness['metadata']['rounded_input_changed'])
        self.assertTrue(_same_bytes(captured['old_input'], captured['new_input']))
        self.assertTrue(bool(_row_bytes_differ(torch.zeros(1), -torch.zeros(1))[0]))

    def test_fake_pool_coverage(self):
        observer, _, generate = self.make(supports='distinct')
        # The actual original core executes once; recorder fences cover only observation.
        observer.birth.observe_real(torch.arange(256, dtype=torch.float32).reshape(256, 1))
        result = observer.birth.maybe_apply(0.)
        self.assertIsNotNone(result)
        self.assertEqual(observer.counts['gates']['ready'], 1)
        self.assertEqual(observer.counts['gates']['evaluations'], 1)
        self.assertEqual(observer.counts['fake_pool']['jitter_calls'], 1)
        self.assertEqual(observer.counts['fake_pool']['radius_calls'], 1)
        self.assertEqual(observer.counts['fake_pool']['query_rows'], 256)
        self.assertEqual(generate.calls, 2)

    def test_primary_and_isolation_coverage(self):
        observer, _, _ = self.make()
        child, parent = torch.tensor([6]), torch.tensor([0])
        self.assertIsNone(observer.birth._move(child, parent))
        self.assertEqual(observer.counts['primary_move']['jitter_calls'], 1)
        # The wrapper delegates one supplied original callback; no second isolation decision.
        seen = []
        expected = (torch.tensor([7]), torch.tensor([0]))
        def original_pick(a, b):
            seen.append((a, b)); return expected
        observer.originals['_isolation_pick'] = original_pick
        flagged = torch.zeros(256, dtype=torch.bool)
        result = observer.birth._isolation_pick(flagged, child)
        self.assertIs(result, expected)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], flagged); self.assertIs(seen[0][1], child)
        observer.birth._move(*result)
        self.assertEqual(observer.counts['isolation_move']['jitter_calls'], 1)
        self.assertEqual(observer.counts['gates']['isolation_pick_calls'], 1)

    def test_same_pair_commit(self):
        observer, _, _ = self.make()
        observer.birth._move(torch.tensor([6]), torch.tensor([0]))
        self.assertEqual(observer.counts['primary_move']['commit_joins'], 1)
        self.assertIs(observer.witness['metadata']['actual_commit_matches'], True)
        self.assertEqual(observer.witness['tensors']['parent'].tolist(), [0])
        self.assertEqual(observer.witness['tensors']['child'].tolist(), [6])
        self.assertTrue(_same_bytes(observer.birth.rows.table.detach()[[6]],
                                   observer.witness['tensors']['new_input']))

    def test_on_off_original_calls_and_rng(self):
        off, off_trainer, off_generate = self.make(enabled=False, supports='distinct')
        on, on_trainer, on_generate = self.make(enabled=True, supports='distinct')
        global_before = torch.get_rng_state().clone()
        records = []
        for observer in (off, on):
            counts = dict(randn=0, randint=0, rand=0, sample=0)
            from contextlib import ExitStack
            originals = {name: getattr(torch, name) for name in ('randn','randint','rand')}
            sample = MoGParticlePrior.sample
            def sampling(prior, *args, **kwargs):
                counts['sample'] += 1; return sample(prior, *args, **kwargs)
            with ExitStack() as stack:
                for name, fn in originals.items():
                    def counted(*args, _name=name, _fn=fn, **kwargs):
                        counts[_name] += 1; return _fn(*args, **kwargs)
                    stack.enter_context(patch.object(torch, name, counted))
                stack.enter_context(patch.object(MoGParticlePrior, 'sample', sampling))
                observer.birth.observe_real(torch.arange(256, dtype=torch.float32).reshape(256,1))
                result = observer.birth.maybe_apply(0.)
                observer.birth._move(torch.tensor([6]),torch.tensor([0]))
            records.append((result, counts))
        self.assertEqual(records[0], records[1])
        self.assertEqual(off_generate.calls, on_generate.calls)
        for a, b in ((off_trainer.prior.z, on_trainer.prior.z),
                     (off.birth.rows.averaged_table, on.birth.rows.averaged_table),
                     (off.birth.stream.get_state(),on.birth.stream.get_state()),
                     (off.context.streams.streams['named'].get_state(),on.context.streams.streams['named'].get_state())):
            self.assertTrue(_same_bytes(a,b))
        for key in off_trainer.opt_g.state[off_trainer.prior.z]:
            self.assertTrue(_same_bytes(off_trainer.opt_g.state[off_trainer.prior.z][key],
                                       on_trainer.opt_g.state[on_trainer.prior.z][key]))
        self.assertTrue(_same_bytes(global_before,torch.get_rng_state()))
        self.assertEqual(on.purity_failures,0)

    def test_populated_fast_purity(self):
        observer, trainer, _ = self.make()
        before = observer.reader.capture()
        self.jitter(observer,torch.ones(2,2))
        after = observer.reader.capture()
        self.assertEqual(before['fingerprint'],after['fingerprint'])
        self.assertTrue(trainer.G.training); self.assertFalse(trainer.D.training)
        self.assertIsNotNone(trainer.policy._fast)
        self.assertTrue(any('.grad' in path for path in before['records']))
        self.assertEqual(set(before['partitions']),
                         {'learned_values','gradients','optimizer','control','rng','identity'})
        # Arbitrary callback closures have not been silently certified as full coverage.
        self.assertFalse(before['complete'])

    def test_mutation_is_detected(self):
        observer, trainer, _ = self.make()
        # Deliberately adversarial callback: detection, with no restoration.
        original = observer._observe
        with self.assertRaises(ObserverPurityError):
            original(lambda: trainer.prior.z.grad.add_(1.))
        self.assertEqual(observer.purity_failures,1)
        self.assertEqual(float(trainer.prior.z.grad[0,0]),2.)

    def test_forbidden_getters_and_models(self):
        observer, _, _ = self.make()
        self.jitter(observer,torch.ones(2,2))
        observer.scheduled_update = 833
        observer.policy.after_generator_backward()
        # Virtual device only: a foreign generator must be UNKNOWN before a read.
        class ProbeGenerator:
            def __init__(self, device):
                self.device=torch.device(device);self.calls=0
            def get_state(self):
                self.calls+=1
                if self.device.type=='cuda':raise AssertionError('foreign state read')
                return torch.zeros(4,dtype=torch.uint8)
        foreign, cpu = ProbeGenerator('cuda:7'), ProbeGenerator('cpu')
        observer.reader.capture()
        with patch.object(torch,'Generator',ProbeGenerator):
            observer.reader._walk(foreign,'virtual.foreign_generator',False)
            observer.reader._walk(cpu,'virtual.cpu_generator',False)
        self.assertEqual(foreign.calls,0)
        self.assertEqual(cpu.calls,1)
        self.assertEqual(observer.reader.records['virtual.foreign_generator']['kind'],'unknown')
        self.assertTrue(any('foreign generator device' in reason for reason in observer.reader.unknown))
        self.assertEqual(ADDED_OPERATIONS,dict.fromkeys(EXTRA_OPERATIONS,0))

    def test_witness_overflow(self):
        observer, _, _ = self.make()
        with patch('experiments.forge.atlas844_radius_observer.WITNESS_BYTES',1):
            result = self.jitter(observer,torch.ones(2,2))
        first = deepcopy(observer.first_affected)
        self.assertIsNotNone(first)
        self.assertIsNone(observer.witness)
        self.assertTrue(bool(result.abs().sum()>0))
        self.jitter(observer,torch.ones(2,2))
        self.assertEqual(observer.first_affected,first)
        self.assertIsNone(observer.witness)
        self.assertTrue(any('limit' in reason for reason in observer.unknown))

    def test_missing_coverage_is_unknown(self):
        observer, _, _ = self.make()
        observer.policy.completed_steps = 1000
        observer.counts['gates'].update(maybe_apply_entries=1000,ready=1,evaluations=1,not_ready=999)
        observer.birth.counters['evals']=1
        result = observer.finish()
        self.assertFalse(result['complete_call_count_joins'])
        self.assertEqual(result['coverage_status'],'UNKNOWN')
        observer.birth._jitter = observer.originals['_jitter']
        self.assertFalse(observer.verify_live_owner())
        self.assertTrue(any('owner/backend' in reason for reason in observer.unknown))
        from experiments.forge import atlas844_radius_owner as owner
        candidate = dict(id=owner.CANDIDATE_ID,recipe_preset='atlas',recipe_overrides={},
                         claim_contract={'experimental_track':owner.TRACK_ID})
        self.assertTrue(owner.supports_candidate(candidate))
        bad=deepcopy(candidate);bad['recipe_overrides']={'lr':.1}
        self.assertFalse(owner.supports_candidate(bad))
        root=Path(__file__).resolve().parents[1]
        task=__import__('json').loads((root/'configs/forge/tasks/gaussian1d_acquisition.json').read_text())
        self.assertEqual(owner.validate(task)['task_id'],'gaussian1d_acquisition')
        wrong=deepcopy(task);wrong['execution']['prior']['sigma']=.03
        with self.assertRaises(ValueError):owner.validate(wrong)

    def test_scheduled_snapshot_bounds(self):
        observer, _, _ = self.make()
        for step in CLOCKS:
            observer.policy.completed_steps=step
            observer.scheduled_observation(step)
        self.assertEqual([row['completed_steps'] for row in observer.snapshots],list(CLOCKS))
        with self.assertRaises(ValueError):observer.scheduled_observation(834)
        for step in (833,834,835):
            observer.scheduled_update=step
            observer.policy.completed_steps=step-1
            observer.update_boundary('pre_update',step)
            observer.policy.after_generator_backward()
            observer.policy.completed_steps=step
            observer.update_boundary('post_update',step)
        self.assertEqual(len(observer.neighborhood),9)
        self.assertEqual([row['scheduled_update'] for row in observer.neighborhood],
                         [833]*3+[834]*3+[835]*3)
        self.assertEqual([row['completed_steps'] for row in observer.neighborhood],
                         [832,832,833,833,833,834,834,834,835])
        self.assertEqual(len(observer.snapshots),24)
        self.assertEqual(ADDED_OPERATIONS,dict.fromkeys(EXTRA_OPERATIONS,0))


if __name__ == '__main__':
    unittest.main()
