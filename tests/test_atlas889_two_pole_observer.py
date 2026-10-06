"""ROOT-paid synthetic controls. Author status: AUTHORED_NOT_RUN.

Small synthetic owners use the original CPU classes and populated moments.
They do not execute the original target problem or its scored evaluator.
Run from the installed/copied Source, with CPU1, before any physical claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE_PINS = {
    'particlegan/k3p.py': ('200ead2c27ab0b2aa6068f1b1b1b5c826b8125aa8602bfec08b31aaa2894401c', 33012),
    'particlegan/policy.py': ('7a6ac1410260d720f58b6903310443f3f8dc3a283c1f3b5abd921965835a4967', 81173),
    'particlegan/continuous.py': ('4ceab49c7d51d1769ae91b7f8bc892eaf380ca6fbd77a948a086ade6627e9a41', 60875),
    'particlegan/recipes.py': ('79905bb948cb967f80a087063960ea72e72c04d3b572b4b90ff4fa1dda5f299f', 56022),
    'particlegan/birth_death.py': ('7482b29f0065be446a6b5e087a4b55d0961c8a6f2602cf9919ca502edca78c91', 48741),
}
RECIPE = {'alpha_bar': [1.0, 0.9, 0.5, 0.05, 0.0001], 'amsgrad': True, 'batch_size': 12, 'beta2_anneal_end': 0.2, 'beta2_end': None, 'betas': [0.0, 0.999], 'birth_death_backend': 'auto', 'birth_death_cells': 128, 'birth_death_chunk': 256, 'birth_death_feature_scale': 'std', 'birth_death_isolation': True, 'birth_death_metric_rank': 8, 'birth_death_parent_policy': 'real_anchor', 'birth_death_space': 'critic', 'conditioning': 'scalar', 'continuous_policy': 'dv12', 'critic_formulation': 'ka2', 'critic_payoff_damping': True, 'critic_r1_real': True, 'd_guard_min_steps': 200, 'd_guard_ratio': 5.0, 'd_lr_mult': 1.0, 'direct_particle_betas': [0.0, 0.9], 'direct_particle_gain': True, 'distance_reduction': 'sum', 'ema_decay': 0.995, 'encoder_mode': 'none', 'eps': 1e-08, 'input_noise_anneal_end': 0.1, 'input_noise_std': 0.0, 'latent_damping_max_rate': 0.5, 'loss': 'relativistic', 'lr': 0.00425, 'lr_anneal_start': 0.6, 'lr_control': 'stationarity', 'lr_floor': 0.05, 'model': 'gan', 'name': 'atlas', 'network_lr_floor': 0.01, 'network_lr_horizon_cap': 1600, 'num_classes': None, 'num_particles': 12, 'observation_sigma': 0.03, 'optimizer_family': 'formulation', 'output_noise_mode': 'learnable', 'output_noise_std': 0.029, 'output_noise_warmup': 0.0, 'particle_birth_death': True, 'prior_betas': None, 'prior_kind': 'particles', 'prior_lr_mult': 2.0, 'prior_reg': 0.0, 'reconstruction_weight': 1.0, 'reg_anchor_min_decay': 0.9, 'reg_anchor_weight': 1.0, 'reg_arm': None, 'reg_coeff': 3.0, 'reg_coeff_anneal_end': 0.2, 'reg_coeff_end': None, 'reg_every': 1, 'reg_kappa': 1.0, 'reopen_anchor': 'release', 'reopen_guard': 'settled', 'reopen_signal': 'optimizer', 'routing_temperature': 0.25, 'row_evidence_exclude': True, 'row_evidence_gate': True, 'row_evidence_hold': True, 'row_evidence_hot': True, 'row_evidence_null': 'scaled', 'row_policy': 'independent', 'serve_average': 4.0, 'sigma_rel': 0.0, 'standardize': False, 'table_release_rule': 'anchor', 'total_steps': None, 'ucd_target': 'class', 'ucd_weight': 0.02, 'z_dim': 1}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pinned_core():
    # An independent private module namespace; no shared-tree edits or884 patch.
    for relative, expected in CORE_PINS.items():
        raw = (ROOT / relative).read_bytes()
        if (hashlib.sha256(raw).hexdigest(), len(raw)) != expected:
            raise ValueError('synthetic control requires original core Source: ' + relative)
    name = '_atlas889_original_cpu_core'
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / 'particlegan')]
        sys.modules[name] = package
    import importlib
    recipes = importlib.import_module(name + '.recipes')
    policy = importlib.import_module(name + '.policy')
    k3p = importlib.import_module(name + '.k3p')
    continuous = importlib.import_module(name + '.continuous')
    for module in (recipes, policy, k3p, continuous):
        actual = Path(module.__file__).resolve()
        relative = actual.relative_to(ROOT).as_posix()
        if (actual != ROOT / relative or relative not in CORE_PINS
                or (hashlib.sha256(actual.read_bytes()).hexdigest(), actual.stat().st_size) != CORE_PINS[relative]):
            raise ValueError('foreign original control module')
    return recipes.Recipe, policy.UpdatePolicy


def _fixture():
    import torch
    from experiments.forge.rng import NamedStreams
    Recipe, UpdatePolicy = _pinned_core()
    # CPU generator only: no CUDA seeding/device initialization.
    torch.random.default_generator.manual_seed(909)
    random.seed(909)
    import numpy as np
    np.random.seed(909)
    recipe = Recipe(**RECIPE)
    table = torch.nn.Parameter(torch.linspace(-.12, .15, 12).reshape(12, 1))
    generator = torch.nn.Identity()
    # The actual fixed Atlas recipe needs a genuine hidden critic feature
    # space. Reuse the pinned original stored HostCritic; no callback or
    # birth/death branch is mocked or disabled by this synthetic fixture.
    from benchmarks.locked_shared import two_pole as original_host
    host_path = ROOT / 'benchmarks/locked_shared/two_pole.py'
    host_raw = host_path.read_bytes()
    if (Path(original_host.__file__).resolve() != host_path
            or hashlib.sha256(host_raw).hexdigest() !=
                'eee207af6d12475f7dbb106087f9d9e269d284274e69e4d199f6d8a38cb54f5e'
            or len(host_raw) != 6753):
        raise ValueError('synthetic critic must come from the exact original host Source')
    critic = original_host.HostCritic()
    opt_g = recipe.make_generator_optimizer([{'params': [table], 'forge_role': 'prior'}], direct_particles=[table])
    opt_d = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic))
    streams = NamedStreams(0, device='cpu')
    named = {'latent_generator': streams.generator('prior', component='latent', purpose='indices'),
        'penalty_generator': streams.generator('noise', component='penalty', purpose='training'),
        'noise_generator': streams.generator('noise', component='generator', purpose='output'),
        'eval_generator': streams.generator('eval', component='sampler', purpose='samples')}
    policy = UpdatePolicy(recipe, generator, critic, prior=None, table=table,
        generator_optimizer=opt_g, critic_optimizer=opt_d, table_optimizer=opt_g,
        roles=[['table'], ['critic']], streams=named, seed=0)
    # A meaningful pre-existing AMSGrad/variable-beta denominator fixture.
    state = opt_g.state[table]
    state.update(step=torch.tensor(3.), exp_avg=torch.zeros_like(table),
        exp_avg_sq=torch.full_like(table, .0004), max_exp_avg_sq=torch.full_like(table, .001))
    opt_g.direct_response.started = True
    opt_g.direct_history.copy_(torch.linspace(-.5, .5, 12))
    owner = SimpleNamespace(torch=torch, recipe=recipe, table=table, generator=generator, critic=critic,
        opt_g=opt_g, opt_d=opt_d, policy=policy, streams=streams, prior=None,
        real=torch.linspace(-.4, .7, 12).reshape(12, 1), row_indices=torch.arange(12), calls={})
    return owner


def _snapshot(owner):
    """Compare all reachable synthetic numerical leaves, aliases and versions.

    The two fresh owners have different addresses; aliases use stable root paths.
    No state_dict/serving getter is invoked, including by the controls.
    """
    from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
    torch = owner.torch
    names = {id(owner.table): 'table', id(owner.policy.log_output_sigma): 'output_sigma'}
    names.update({id(p): 'critic.' + name for name, p in owner.critic.named_parameters()})
    # CriticPenalty._names is intentionally keyed by live Module addresses.
    # Normalize only that exact cache's keys, retaining every value and the
    # critic's complete named-module alias paths. Other integers are untouched.
    penalty = owner.policy.penalty
    critic_name_cache = vars(penalty)['_names']
    if penalty.critic is not owner.critic or type(critic_name_cache) is not dict:
        raise ValueError('snapshot requires the actual owned CriticPenalty identity cache')
    critic_modules = {id(module): name for name, module in owner.critic.named_modules()}
    critic_aliases = {}
    for name, module in owner.critic.named_modules(remove_duplicate=False):
        critic_aliases.setdefault(id(module), []).append(name)
    leaves, seen = {}, {}
    def walk(v, path):
        if v is None or type(v) in (bool, int, str):
            leaves[path] = v
        elif type(v) is float:
            leaves[path] = v.hex()
        elif isinstance(v, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)):
            pass
        elif id(v) in seen:
            leaves[path] = ('alias', seen[id(v)])
        elif isinstance(v, torch.Tensor):
            seen[id(v)] = path
            leaves[path] = ('tensor', str(v.dtype), tuple(v.shape), v._version,
                bool(v.requires_grad), v.detach().numpy().tobytes())
            if (v.is_leaf or v.retains_grad) and v.grad is not None:
                walk(v.grad, path + '.grad')
        elif isinstance(v, torch.Generator):
            leaves[path] = ('generator', str(v.device), v.get_state().numpy().tobytes())
        elif isinstance(v, (torch.dtype, torch.device)):
            leaves[path] = str(v)
        elif isinstance(v, dict):
            seen[id(v)] = path
            def key(k):
                if v is critic_name_cache:
                    if type(k) is not int or k not in critic_modules:
                        raise ValueError('stale or foreign CriticPenalty module identity')
                    return 'critic_module:' + json.dumps(critic_aliases[k], separators=(',', ':'))
                return names.get(id(k), 'tensor_key') if isinstance(k, torch.Tensor) else str(k)
            for k in sorted(v, key=key):
                walk(v[k], path + '/' + key(k))
        elif isinstance(v, (list, tuple)):
            seen[id(v)] = path
            for i, x in enumerate(v):
                walk(x, path + '/' + str(i))
        elif isinstance(v, (set, frozenset)):
            leaves[path] = tuple(sorted(str(x) for x in v))
        elif hasattr(v, '__dict__'):
            seen[id(v)] = path
            walk(vars(v), path + '.__dict__')
        else:
            leaves[path] = type(v).__module__ + '.' + type(v).__qualname__
    for name in ('table', 'generator', 'critic', 'opt_g', 'opt_d', 'policy', 'streams', 'real', 'row_indices', 'calls'):
        walk(getattr(owner, name), name)
    leaves['global.cpu_rng'] = torch.get_rng_state().numpy().tobytes()
    leaves['global.python_rng'] = random.getstate()
    import numpy as np
    st = np.random.get_state()
    leaves['global.numpy_rng'] = (st[0], st[1].tobytes(), st[2], st[3], st[4])
    return leaves


def _spy(owner):
    """Count actual original calls, including controller calls, in each arm."""
    counts = dict(prior=0, rng=0, forwards=0, backwards=0, optimizer=0, decisions=0, getter=0, foreign=0)
    hooks = []
    def hook(obj, name, category):
        old = getattr(obj, name)
        previous = vars(obj).get(name, None)
        had = name in vars(obj)
        def call(*args, **kwargs):
            counts[category] += 1
            return old(*args, **kwargs)
        setattr(obj, name, call)
        hooks.append((obj, name, had, previous))
    for model in (owner.generator, owner.critic):
        hook(model, 'forward', 'forwards')
    for optimizer in (owner.opt_g, owner.opt_d):
        hook(optimizer, 'step', 'optimizer')
        hook(optimizer, 'state_dict', 'getter')
    hook(owner.policy, 'state_dict', 'getter')
    hook(owner.policy.surprise, 'group_q', 'decisions')
    hook(owner.policy.surprise, 'decide', 'decisions')
    tester = owner.policy.lr_settle.testers[0][0]
    hook(tester, 'observe', 'decisions')
    hook(owner.policy.birth_death, 'maybe_apply', 'decisions')
    hook(owner.torch.cuda, '_lazy_init', 'foreign')
    for name in ('rand', 'randn', 'randint', 'normal', 'rand_like', 'randn_like'):
        hook(owner.torch, name, 'rng')
    def remove():
        for obj, name, had, old in reversed(hooks):
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
    return counts, remove


def _synthetic_update(owner, counts):
    # A tiny explicitly synthetic loss, not the original two_pole target/evaluator.
    policy = owner.policy
    policy.begin_step(owner.real, execution_limit=80)
    owner.opt_d.zero_grad()
    fake_d = policy.generate(owner.table, rows=owner.row_indices).detach()
    r, f = owner.critic(owner.real), owner.critic(fake_d)
    policy.observe_critic_pair(r, f)
    loss_d = (r.square().mean() + f.square().mean())
    counts['backwards'] += 1
    loss_d.backward()
    owner.opt_d.step()
    policy.after_critic_step()
    owner.opt_g.zero_grad()
    fake_g = policy.generate(owner.table, rows=owner.row_indices)
    loss_g = -.1 * owner.critic(fake_g).mean() + .03 * owner.table.square().mean()
    counts['backwards'] += 1
    loss_g.backward()
    policy.after_generator_backward(loss_gan=loss_g, loss_critic=loss_d)
    owner.opt_g.step()
    policy.after_generator_step()
    return policy.finish_step()


LAST_OPERATIONS = None


class ObserverControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.recorder_module = _load(ROOT / 'experiments/forge/atlas889_two_pole_observer.py', '_atlas889_passive_recorder')
        cls.contract = _load(ROOT / 'experiments/forge/atlas889_contract.py', '_atlas889_control_contract')

    def _arm(self, enabled, steps=3):
        owner = _fixture()
        counts, remove = _spy(owner)
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install() if enabled else None
        try:
            for _ in range(steps):
                _synthetic_update(owner, counts)
        finally:
            if recorder is not None:
                recorder.uninstall()
            remove()
        return owner, counts, recorder

    def test_original_adam_off_on_populated_moments(self):
        off, count_off, _ = self._arm(False)
        on, count_on, recorder = self._arm(True)
        self.assertEqual(_snapshot(off), _snapshot(on))
        self.assertEqual(count_off, count_on)
        self.assertEqual(recorder.purity_failures, 0)
        self.assertEqual(recorder.unknown, [])
        self.assertEqual(recorder.scope_unknown, [])
        global LAST_OPERATIONS
        keys = ('extra_prior_samples', 'extra_rng_draws', 'extra_model_forwards', 'extra_backward_calls',
            'extra_optimizer_steps', 'extra_decision_evaluations', 'state_getter_calls', 'foreign_device_initializations')
        categories = ('prior', 'rng', 'forwards', 'backwards', 'optimizer', 'decisions', 'getter', 'foreign')
        LAST_OPERATIONS = {key: count_on[category] - count_off[category] for key, category in zip(keys, categories)}
        LAST_OPERATIONS['state_mutations'] = recorder.purity_failures
        self.assertEqual(LAST_OPERATIONS, dict.fromkeys(self.contract.ZERO_OPERATIONS, 0))
        # Address normalization must not hide an actual cache-binding change.
        cache = vars(on.policy.penalty)['_names']
        saved_cache = dict(cache)
        before_cache_change = _snapshot(on)
        first, second = id(on.critic.fc1), id(on.critic.fc2)
        try:
            cache[first], cache[second] = cache[second], cache[first]
            self.assertNotEqual(_snapshot(on), before_cache_change)
        finally:
            cache.clear()
            cache.update(saved_cache)
        self.assertEqual(_snapshot(on), before_cache_change)
        try:
            cache[-1] = cache.pop(first)
            with self.assertRaisesRegex(ValueError, 'stale or foreign CriticPenalty module identity'):
                _snapshot(on)
        finally:
            cache.clear()
            cache.update(saved_cache)
        self.assertEqual(_snapshot(on), before_cache_change)

    def test_original_policy_hooks_off_on(self):
        off, _, _ = self._arm(False, steps=4)
        on, _, recorder = self._arm(True, steps=4)
        self.assertEqual(_snapshot(off), _snapshot(on))
        self.assertEqual([r['step'] for r in recorder.records], [1, 2, 3, 4])
        self.assertEqual(recorder.calls, dict.fromkeys(recorder.calls, 4))
        self.assertTrue(all(event['pure'] for row in recorder.records for event in row['events']))

    def test_actual_beta_lr_q_inputs_and_return(self):
        _, _, recorder = self._arm(True, steps=1)
        events = {e['name']: e['payload'] for e in recorder.records[0]['events']}
        adam = events['actual_adam_inputs']['point']
        restored = events['after_direct_group_restore']['group']
        q = events['table_group_q_actual_return']
        self.assertEqual(adam['group']['betas'], [0., .9])
        self.assertEqual(restored['betas'], [0., .999])
        self.assertAlmostEqual(adam['group']['lr'], restored['lr'] * adam['direct_response']['gain'])
        self.assertEqual(q['consumer_group']['betas'], [0., .999])
        self.assertEqual(q['point']['training_gradient']['shape'], [12, 1])
        st = q['point']
        t = st['optimizer_step']['values']
        gradients = st['training_gradient']['values']
        v = st['exp_avg_sq']['values']
        expected = sum(abs(g[0]) / (math.sqrt(vv[0] / (1 - .999 ** t)) + q['consumer_group']['eps'])
            for g, vv in zip(gradients, v)) / 12
        self.assertAlmostEqual(q['q']['values'], expected, places=5)
        self.assertIn('max_exp_avg_sq', st)

    def test_original_finish_event_identity(self):
        owner = _fixture()
        counts, remove = _spy(owner)
        original = owner.policy.birth_death.maybe_apply
        sentinel = {'moves': [], 'synthetic_control_event': 31}
        def event(*args, **kwargs):
            original(*args, **kwargs)
            return sentinel
        owner.policy.birth_death.maybe_apply = event
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install()
        try:
            returned = _synthetic_update(owner, counts)
            self.assertIs(returned, sentinel)
            self.assertEqual(recorder.calls['finish_step'], 1)
        finally:
            recorder.uninstall()
            remove()

    def test_closure_exception_passthrough(self):
        owner = _fixture()
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install()
        marker = RuntimeError('inert closure fault')
        def closure():
            raise marker
        before = _snapshot(owner)
        try:
            with self.assertRaises(RuntimeError) as caught:
                owner.opt_g.step(closure)
            self.assertIs(caught.exception, marker)
            self.assertEqual(_snapshot(owner), before)
            self.assertEqual(recorder.calls['direct_begin'], 0)
        finally:
            recorder.uninstall()

    def test_adam_exception_and_restore_passthrough(self):
        owner = _fixture()
        owner.table.grad = owner.torch.ones_like(owner.table)
        module = sys.modules[type(owner.opt_g).__module__]
        original = module._adam_step
        marker = RuntimeError('inert Adam fault')
        def failed(optimizer):
            raise marker
        module._adam_step = failed
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install()
        recorder.current = {'step': 1, 'events': []}
        group = owner.opt_g.param_groups[0]
        lr, betas = group['lr'], group['betas']
        try:
            with self.assertRaises(RuntimeError) as caught:
                owner.opt_g.step()
            self.assertIs(caught.exception, marker)
            self.assertEqual((group['lr'], group['betas']), (lr, betas))
            self.assertEqual(recorder.calls['direct_end'], 1)
            names = [e['name'] for e in recorder.current['events']]
            self.assertIn('after_adam_attempt_before_restore', names)
            self.assertIn('optimizer_raised', names)
            self.assertNotIn('optimizer_returned', names)
        finally:
            module._adam_step = original
            recorder.uninstall()

    def test_observer_fault_does_not_replace_q(self):
        owner = _fixture()
        sentinel = owner.torch.tensor(.25)
        calls = []
        owner.policy.surprise.group_q = lambda *args, **kwargs: (calls.append(args), sentinel)[1]
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install()
        def failed():
            raise RuntimeError('inert observer lookup fault')
        recorder._group = failed
        try:
            returned = owner.policy.surprise.group_q(owner.opt_g, owner.opt_g.param_groups[0])
            self.assertIs(returned, sentinel)
            self.assertEqual(len(calls), 1)
            self.assertTrue(any('group_q postprocessing' in r for r in recorder.unknown))
        finally:
            recorder.uninstall()

    def test_capture_fault_marks_unknown(self):
        owner = _fixture()
        counts, remove = _spy(owner)
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner).install()
        recorder._point = lambda: (_ for _ in ()).throw(RuntimeError('inert capture fault'))
        try:
            _synthetic_update(owner, counts)
            self.assertEqual(owner.policy.completed_steps, 1)
            self.assertTrue(recorder.unknown)
            self.assertFalse(recorder.receipt(completed_steps=1, scored_clocks=[])['complete'])
        finally:
            recorder.uninstall()
            remove()

    def test_bounded_records_and_declared_scope_unknown(self):
        owner = _fixture()
        recorder = self.recorder_module.PassiveTwoPoleRecorder(owner)
        recorder.calls = dict.fromkeys(recorder.calls, 80)
        recorder.records = [{'step': i, 'events': []} for i in range(1, 81)]
        clocks = [math.ceil(i * 80 / 24) for i in range(1, 25)]
        recorder.scope_unknown = ['inert_declared_live_leaf:UNKNOWN']
        self.assertFalse(recorder.receipt(completed_steps=80, scored_clocks=clocks)['complete'])
        recorder.scope_unknown = []
        recorder.current = recorder.records[-1]
        recorder.bytes = self.recorder_module.MAX_TRACE_BYTES
        recorder._capture('inert_overflow', lambda: 1)
        self.assertTrue(recorder.disabled)
        self.assertEqual(len(recorder.records), 80)
        self.assertFalse(recorder.receipt(completed_steps=80, scored_clocks=clocks)['complete'])

    def test_private_source_origin_no884_patch(self):
        _pinned_core()
        forbidden = {'988b95709ca2fd1664a51b0fd14cfd4e0764e09d15c29085797237a4fb54c310',
            '5c76c3ffca76697dcb8ff56a4648e79792c4d95cfb87d830230e9884befc8e74'}
        self.assertTrue(all(sha not in forbidden for sha, _ in CORE_PINS.values()))

    def test_original_recipe_task_registry_binding(self):
        from experiments.forge import api, atlas889_two_pole_owner as owner
        from experiments.forge.contracts import validate_idea
        from experiments.forge.views import validate_view
        original = json.loads((ROOT / 'configs/forge/tasks/two_pole.json').read_bytes())
        candidate = json.loads((ROOT / ('configs/forge/ideas/' + self.contract.CANDIDATE_ID + '.json')).read_bytes())
        control = json.loads((ROOT / 'configs/forge/ideas/atlas-existing-mog-nearest-positive791-v1.json').read_bytes())
        view = json.loads((ROOT / ('configs/forge/views/' + self.contract.VIEW_ID + '.json')).read_bytes())
        protocol = json.loads((ROOT / 'configs/forge/protocols/screening.json').read_bytes())
        validate_idea(candidate)
        validate_view(view, {'two_pole': original})
        from experiments.forge.views import DIAGNOSTIC_SCOPES
        self.assertIn(view['evidence_scope'], DIAGNOSTIC_SCOPES)
        self.assertTrue(all(a['importance'] == 'diagnostic' for a in view['assignments']))
        # Exercise the planner's task-free metadata constructor as well as the
        # task binding below. No model, sampler or scientific case is constructed.
        defaults = json.loads((ROOT / 'configs/forge/defaults.json').read_bytes())
        def planner_metadata(declaration):
            return api.FormulationContext(recipe_preset=declaration.get('recipe_preset'),
                recipe_overrides=declaration.get('recipe_overrides', {}),
                prior={**defaults['prior'], **declaration.get('prior', {})},
                seed=protocol['seed'],
                requires_capabilities=declaration.get('requires_capabilities', []),
                extensions=declaration.get('extensions', {}),
                initializer=declaration.get('initializer', 'deterministic_orthogonal'),
                execution_path=declaration.get('execution_path', 'public_trainer'),
                candidate_id=declaration['id'])
        metadata = planner_metadata(candidate)
        metadata_control = planner_metadata(control)
        self.assertIsNone(metadata.policy_task)
        self.assertIsNone(metadata._trainer)
        self.assertIsNone(metadata._policy)
        self.assertTrue(metadata._existing_mog)
        self.assertEqual(asdict(metadata.recipe), asdict(metadata_control.recipe))
        self.assertEqual(metadata.prior_config, metadata_control.prior_config)
        self.assertEqual(metadata.streams.manifest(), metadata_control.streams.manifest())
        with self.assertRaises(api.CapabilityError):
            planner_metadata(dict(candidate, id='inert-unrelated-candidate'))
        # Maintained planner-shaped source.files is SHA strings; proof preserves
        # independently captured typed byte pins. These are inert metadata only.
        inert_pin = {'inert_source.py': {'sha256': '4' * 64, 'bytes': 7}}
        inert_request = dict(request_id='inert-no-attempt', candidate=candidate, view=view,
            tasks={'two_pole': original}, protocol={'seed': 0}, candidate_revision='2' * 64,
            source={'digest': '1' * 64, 'files': {'inert_source.py': '4' * 64}},
            jobs=[{'task_id': 'two_pole', 'budget_seconds': 300,
                'resources': {'backend': 'cpu', 'cpu_threads': 1, 'gpus': 0}}])
        inert_controls = dict(schema=self.contract.CONTROL_SCHEMA, status='PASS',
            checks=dict.fromkeys(self.contract.CHECKS, 'PASS'),
            added_operations=dict.fromkeys(self.contract.ZERO_OPERATIONS, 0),
            effective_recipe_sha256=self.contract.EFFECTIVE_RECIPE_SHA256)
        proof = self.contract.build_software_proof(inert_request, inert_controls, inert_pin)
        self.assertEqual(proof['source_pins'], inert_pin)
        malformed = deepcopy(inert_request)
        malformed['source']['files'] = inert_pin
        with self.assertRaises(ValueError):
            self.contract.build_software_proof(malformed, inert_controls, inert_pin)
        self.assertEqual(self.contract.task_digest(original), self.contract.TASK_DIGEST)
        self.assertEqual(self.contract.digest(RECIPE), self.contract.EFFECTIVE_RECIPE_SHA256)
        current = api.task_formulation_context(candidate, original, protocol, root=ROOT)
        before = api.task_formulation_context(control, original, protocol, root=ROOT)
        self.assertEqual(asdict(current.recipe), asdict(before.recipe))
        self.assertEqual(owner.digest(asdict(current.recipe)), self.contract.EFFECTIVE_RECIPE_SHA256)
        self.assertEqual(current.receipt()['prior'], before.receipt()['prior'])
        host = current.receipt()['field_ownership']['task_contract']['host']['value']
        ref = before.receipt()['field_ownership']['task_contract']['host']['value']
        self.assertIn('passive_observation', host)
        self.assertNotIn('passive_observation', ref)
        unrelated = dict(candidate, id='inert-unrelated-candidate')
        self.assertFalse(owner.supports_candidate(unrelated))
        self.assertTrue(owner.blockers(original, unrelated, root=ROOT))
        bad_task = deepcopy(original)
        bad_task['execution']['steps'] = 81
        self.assertTrue(owner.blockers(bad_task, candidate, root=ROOT))
        self.assertEqual(owner.supporting_source_paths({'id': 'inert_unrelated_task'}, candidate, ROOT), ())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls-json', required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObserverControls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    from experiments.forge.atlas889_contract import CHECKS, CONTROL_SCHEMA, EFFECTIVE_RECIPE_SHA256
    passed = result.wasSuccessful() and result.testsRun == len(CHECKS) and LAST_OPERATIONS is not None
    output = dict(schema=CONTROL_SCHEMA, status='PASS' if passed else 'FAIL',
        checks=dict.fromkeys(CHECKS, 'PASS') if passed else {},
        added_operations=LAST_OPERATIONS, effective_recipe_sha256=EFFECTIVE_RECIPE_SHA256,
        software_only=True, original_target_run=False)
    # ROOT runs this writer inside its actual bounded paid software phase.
    path = Path(args.controls_json)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
