"""Opt-in feature integration for the current caller-owned UpdatePolicy API.

Auto scope is a conservative capability contract, not a quality oracle.
The generator/noise factor is an empirical backend calibration. The current
reference law, custom callback ownership and routed contracts remain intact.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import torch

from .continuous import StationarityLR, _T9875
from .feature_cells import FeatureCellBirthDeath, BoundedLatentGeometry, LatentLineage, population_policy
from .output_moments import MAX_RANK
from .population_continuity import PopulationSequentialSettleTest


class _TableOwner:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @property
    def latent_history(self):
        return getattr(self.optimizer, 'latent_history', None)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


class FeatureFacade:
    """Only the owners used by immutable RA11 feature/count/kernel helpers."""
    def __init__(self, policy):
        self.policy = policy
        self.prior = SimpleNamespace(z=policy.table)
        self.ema_prior = SimpleNamespace(z=policy.averaged_table)
        self.opt_g = _TableOwner(policy.table_optimizer)

    @property
    def mean_partial_recovery(self):
        """Partial occupied-group evidence follows an actual R1 fire only."""
        detector = self.policy.surprise
        return detector is not None and type(detector.fires) is int and detector.fires > 0

    def __getattr__(self, name):
        if name in ('G', 'D', 'ema_G', 'recipe', 'device', 'dtype', 'controller',
                    'completed_steps', '_STREAMS', 'latent_generator', 'penalty_generator',
                    'noise_generator', 'eval_generator'):
            return getattr(self.policy, name)
        raise AttributeError(name)


class _FeatureBackend(FeatureCellBirthDeath):
    """Keep the current API's mixed-module evaluation flags around legacy work."""
    def _features(self, trainer, *args, **kwargs):
        flags = [(module,module.training) for module in trainer.D.modules()]
        try:
            return super()._features(trainer,*args,**kwargs)
        finally:
            for module,flag in flags:
                module.training = flag

    def maybe_apply(self, trainer, sigma):
        flags = [(module,module.training) for root in (trainer.G,trainer.D) for module in root.modules()]
        try:
            return super().maybe_apply(trainer,sigma)
        finally:
            for module,flag in flags:
                module.training = flag


class FrozenFeatureSampler:
    """Independent frozen lineage plus derived geometry for served samples."""
    def __init__(self, state):
        self.state = deepcopy(state)
        lineage = LatentLineage(state['rows'], state['degree'], state['neighbors'].device)
        lineage.neighbors = state['neighbors'].clone()
        self.geometry = BoundedLatentGeometry(rank=state['rank'], neighbors=state['candidate_neighbors'],
                                             chunk=state['chunk'], lineage=lineage)

    def perturb_latent(self, latent, stream, controller, prior, *, rows=None):
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=stream)
        return latent + self.geometry.displacement(latent, prior, controller.latent_bandwidth, noise, rows=rows)


class FeatureSelection:
    SCHEMA = 1
    GENERATOR_NOISE_FACTOR = .25

    def __init__(self, policy, seed):
        self.policy, self.seed = policy, seed
        self.facade = FeatureFacade(policy)
        self.reference_birth_death = policy.birth_death
        self.base_lrs = deepcopy(policy.initial_lrs)
        self.requested = policy.recipe.birth_death_backend
        self.state = self._selection(None, self.base_lrs)
        if self.state['actual_backend'] == 'feature_cells':
            self._install_controls(self.state, self._controls(self.state))

    def _scope_reason(self):
        policy = self.policy
        if policy.routed_control is not None:
            return 'routed_rows_owns_controls'
        if (policy.generation is not None or policy.critic_features is not None
                or policy.encoder is not None or policy.router is not None):
            return 'caller_callbacks_own_representation'
        return None

    def _selection(self, shape, bases):
        population = population_policy(len(self.policy.table))
        width = None if shape is None else math.prod(shape)
        reason = self._scope_reason()
        if shape is None and self.requested == 'auto':
            actual, reason = 'pending', 'awaiting_first_real_shape'
        elif reason is not None:
            actual = 'routed' if self.policy.routed_control is not None else 'knn'
        elif not population['finite_resolution_feasible']:
            actual, reason = 'knn', 'finite_resolution_infeasible'
        elif self.requested == 'feature_cells':
            actual, reason = 'feature_cells', 'explicit_feature_cells'
        elif width <= MAX_RANK:
            actual, reason = 'feature_cells', 'finite_resolution_and_complete_raw_moment_frame'
        else:
            actual, reason = 'knn', 'raw_output_exceeds_complete_moment_frame'
        factor = self.GENERATOR_NOISE_FACTOR if actual == 'feature_cells' and self.requested == 'auto' else 1.
        mapping = [[dict(role=role, base_rate=float(base),
                         factor=factor if role in ('generator', 'noise') else 1.,
                         rate=float(base) * (factor if role in ('generator', 'noise') else 1.))
                    for role,base in zip(roles,row)] for roles,row in zip(self.policy.roles,bases)]
        return dict(schema=self.SCHEMA, requested_backend=self.requested, actual_backend=actual,
                    selection_reason=reason, output_shape=None if shape is None else list(shape),
                    raw_output_width=width, moment_rank_bound=MAX_RANK, population_policy=population,
                    generator_noise_factor=factor, rate_mapping=mapping,
                    sampling_backend='feature_cells' if actual == 'feature_cells' else
                                     'routed' if actual == 'routed' else 'controller_reference')

    def _controls(self, selection):
        policy = self.policy
        actual = selection['actual_backend']
        birth = (_FeatureBackend(self.facade, self.seed) if actual == 'feature_cells'
                 else self.reference_birth_death)
        settle = (None if policy.lr_settle is None else StationarityLR(
            policy.optimizers, prior_param=policy.table, release_rule=policy.recipe.table_release_rule))
        if actual == 'feature_cells' and settle is not None:
            for optimizer,row in zip(policy.optimizers, settle.testers):
                for j,group in enumerate(optimizer.param_groups):
                    if len(group['params']) == 1 and group['params'][0] is policy.table and row[j] is not None:
                        tester = PopulationSequentialSettleTest(early_stationary_only=True, final_table=_T9875)
                        tester.rows = len(policy.table)
                        tester.release_rule = policy.recipe.table_release_rule
                        row[j] = tester
        return birth, settle

    def _install_controls(self, selection, controls):
        birth, settle = controls
        self.policy.birth_death, self.policy.lr_settle = birth, settle
        self.policy.initial_lrs = [[item['rate'] for item in row] for row in selection['rate_mapping']]
        self.state = deepcopy(selection)

    def observe_shape(self, real):
        shape = tuple(real.shape[1:])
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError('feature backend needs a nonempty real output shape')
        saved_shape = self.state['output_shape']
        if saved_shape is not None:
            if shape != tuple(saved_shape):
                raise ValueError('real output shape differs from the frozen backend selection')
            if self._selection(shape,self.base_lrs) != self.state:
                raise ValueError('representation ownership differs from the frozen backend selection')
            return
        selection = self._selection(shape, self.base_lrs)
        if self.state['actual_backend'] == 'pending' or selection['actual_backend'] != self.state['actual_backend']:
            if self.policy.completed_steps != 0:
                raise ValueError('pending feature selection cannot have completed updates')
            if selection['actual_backend'] == 'feature_cells' or self.state['actual_backend'] == 'feature_cells':
                self._install_controls(selection, self._controls(selection))
            else:
                self.state = selection
        else:
            self.state = selection

    def state_dict(self):
        return deepcopy(self.state)

    def check_generated_shape(self, generated):
        shape = self.state['output_shape']
        if shape is not None and self._scope_reason() is None and tuple(generated.shape[1:]) != tuple(shape):
            raise ValueError('generator output shape differs from the frozen real-output shape')

    def prepare_restore(self, saved, state):
        """Validate route/rates and make private controls, without model mutation."""
        if not isinstance(saved, dict) or set(saved) != set(self.state):
            raise ValueError('invalid backend-selection checkpoint schema')
        if (type(saved['schema']) is not int or saved['schema'] != self.SCHEMA
                or type(saved['moment_rank_bound']) is not int
                or type(saved['generator_noise_factor']) is not float):
            raise ValueError('invalid backend-selection scalar types')
        population=population_policy(len(self.policy.table))
        saved_population=saved['population_policy']
        if (not isinstance(saved_population,dict) or set(saved_population)!=set(population)
                or any(type(saved_population[key]) is not type(population[key]) for key in population)):
            raise ValueError('invalid backend population certificate types')
        if saved['raw_output_width'] is not None and type(saved['raw_output_width']) is not int:
            raise ValueError('invalid backend raw-output width type')
        shape = saved['output_shape']
        if shape is not None and (not isinstance(shape, list) or not shape
                                 or any(type(d) is not int or d <= 0 for d in shape)):
            raise ValueError('invalid frozen backend output shape')
        mapping = saved['rate_mapping']
        if (not isinstance(mapping, list) or len(mapping) != len(self.policy.roles)
                or any(not isinstance(row, list) or len(row) != len(roles)
                       for row,roles in zip(mapping,self.policy.roles))):
            raise ValueError('invalid backend rate mapping')
        bases = []
        for row,roles in zip(mapping,self.policy.roles):
            values = []
            for item,role in zip(row,roles):
                if (not isinstance(item,dict) or set(item) != {'role','base_rate','factor','rate'}
                        or item['role'] != role or type(item['base_rate']) is not float
                        or type(item['factor']) is not float or type(item['rate']) is not float
                        or not math.isfinite(item['base_rate']) or item['base_rate'] <= 0):
                    raise ValueError('invalid backend base-rate owner')
                values.append(item['base_rate'])
            bases.append(values)
        expected = self._selection(None if shape is None else tuple(shape), bases)
        if saved != expected:
            raise ValueError('backend-selection law does not match its declared scope and rates')
        steps = state['completed_steps']
        if shape is None and steps != 0:
            raise ValueError('unresolved backend output shape after completed updates')
        rates = [[item['rate'] for item in row] for row in mapping]
        if state['initial_lrs'] != rates:
            raise ValueError('checkpoint rates disagree with the frozen backend mapping')
        current_shape = self.state['output_shape']
        if current_shape is not None and shape != current_shape:
            raise ValueError('checkpoint output shape differs from the resolved policy')
        controls = self._controls(expected)
        birth, settle = controls
        if birth is not None and expected['actual_backend'] == 'feature_cells':
            birth.check_state(state['birth_death'])
            birth.check_paired_average_step(state['birth_death'], steps)
        return dict(selection=expected, controls=controls, bases=bases)

    def commit_restore(self, prepared):
        self.base_lrs = deepcopy(prepared['bases'])
        self._install_controls(prepared['selection'], prepared['controls'])

    def sampler_snapshot(self):
        birth = self.policy.birth_death
        if self.state['actual_backend'] != 'feature_cells':
            return None
        return dict(rows=birth.N, degree=birth.lineage.degree, neighbors=birth.lineage.neighbors.clone(),
                    rank=birth.latent_geometry.rank, chunk=birth.latent_geometry.chunk,
                    candidate_neighbors=birth.latent_geometry.neighbors,
                    paired_average=deepcopy(birth.paired_average),
                    rows_since_eval=birth.rows_since_eval, fill=birth.fill,
                    backend_selection=self.state_dict())

    def saved_served_source(self, state):
        if state['backend_selection']['actual_backend'] != 'feature_cells':
            return None
        birth = FeatureCellBirthDeath(self.facade, self.seed)
        birth.load_state_dict(state['birth_death'])
        eligible = birth.paired_average_eligible(state['completed_steps'])
        return 'averaged' if self.policy.recipe.serve_average > 0 and eligible else 'fast'
