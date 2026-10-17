"""Read-only fixture/source/selection validators, separate from training laws."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE = ORIGINAL / 'validation-cb64-ra11/learned'
SHAPES = {'toy': [2], 'mnist': [1, 28, 28]}
ROLES = [['generator', 'table', 'noise'], ['critic']]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_members(path):
    module = ast.parse(Path(path).read_text())
    return {node.name: ast.dump(node, include_attributes=False)
            for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def verify_fixture_sources():
    """Preserve scorer ASTs and pin the exact approved runner transformations."""
    original, current = ast_members(LANE / 'run_training.py'), ast_members(ROOT / 'run_training.py')
    unchanged = ('frechet', 'embedding_score', 'ImageEvaluation')
    require(all(current[name] == original[name] for name in unchanged), 'original image scorer AST changed')
    reference_gate = ast_members(ORIGINAL / 'quality/leaderboard.py')['toy_gate']
    require(ast_members(__file__)['toy_gate'] == reference_gate, 'original Toy25 gate AST changed')
    transformation = json.loads((ROOT / 'source-transform-receipt.json').read_text())
    require(sha(LANE / 'run_training.py') == transformation['original_runner_sha256'], 'original runner changed')
    require(sha(ROOT / 'run_training.py') == transformation['adapted_runner_sha256'], 'adapter differs from approved source transformation')
    return dict(original_image_scorer_ast_identical=True, original_toy_gate_ast_identical=True,
                original_models_metrics_byte_pinned=True, original_checkpoint_budget_preserved=True,
                original_primary_output_noise_explicit=True, extra_begin_step_calls=0,
                transformed_runner_sha256=sha(ROOT / 'run_training.py'))


def toy_gate(metrics):
    return (metrics['precision'] >= .90 and metrics['coverage'] == 25
        and metrics['mass_tv'] <= .10 and len(metrics['supported_mass']) == 25
        and min(metrics['supported_mass']) >= .01)


def finite_population_certificate(n, q=.05):
    """Independent exact reconstruction of the declared conformal/BH rule."""
    probability = Fraction(str(q))
    calibration = n // 2
    denominator = probability.numerator * (calibration + 1)
    minimum = (n * probability.denominator + denominator - 1) // denominator
    maximum = n * probability.numerator // probability.denominator
    feasible = minimum <= maximum
    return dict(schema=1, requested_backend='feature_cells', population=n, q=q,
                calibration_rows=calibration, minimum_bh_flags=minimum,
                maximum_guard_flags=maximum, finite_resolution_feasible=feasible,
                actual_backend='feature_cells' if feasible else 'knn',
                matching_sampler='feature_cells' if feasible else 'controller_reference',
                rule='ceil(N/[Q*(floor(N/2)+1)]) <= floor(Q*N)')


def validate_checkpoint_state(state, roles, problem, recipe):
    """Check source-declared scope/rates at every original checkpoint boundary."""
    require(roles == ROLES, 'learned fixture optimizer parameter ownership changed')
    require(recipe.birth_death_backend == 'auto', 'learned fixture must run actual auto candidate')
    require(recipe.num_particles == 1024 and recipe.z_dim == 128 and recipe.batch_size == 128,
            'original learned fixture dimensions changed')
    require((recipe.lr, recipe.prior_lr_mult, recipe.d_lr_mult) == (.00425, 2., 1.),
            'shared candidate input base rates changed')
    require((recipe.reopen_signal, recipe.reopen_anchor) == ('optimizer', 'release'), 'shared R1 mechanism changed')
    validate_guard_state(state, roles, recipe)
    step = state['completed_steps']
    shape = None if step == 0 else SHAPES[problem]
    width = None if shape is None else math.prod(shape)
    population = finite_population_certificate(1024)
    if shape is None:
        actual, reason = 'pending', 'awaiting_first_real_shape'
    elif not population['finite_resolution_feasible']:
        actual, reason = 'knn', 'finite_resolution_infeasible'
    elif width <= 8:
        actual, reason = 'feature_cells', 'finite_resolution_and_complete_raw_moment_frame'
    else:
        actual, reason = 'knn', 'raw_output_exceeds_complete_moment_frame'
    factor = .25 if actual == 'feature_cells' else 1.
    bases = [[.00425, .0085, .00425], [.00425]]
    mapping = [[dict(role=role, base_rate=base, factor=factor if role in ('generator', 'noise') else 1.,
                     rate=base * (factor if role in ('generator', 'noise') else 1.))
                for role, base in zip(row_roles, row_bases)] for row_roles, row_bases in zip(roles, bases)]
    expected = dict(schema=1, requested_backend='auto', actual_backend=actual,
                    selection_reason=reason, output_shape=shape, raw_output_width=width,
                    moment_rank_bound=8, population_policy=population,
                    generator_noise_factor=factor, rate_mapping=mapping,
                    sampling_backend='feature_cells' if actual == 'feature_cells' else 'controller_reference')
    require(state.get('backend_selection') == expected, 'checkpoint auto metadata disagrees with fixture scope/rate certificate')
    require(state['initial_lrs'] == [[item['rate'] for item in row] for row in mapping],
            'saved optimizer bases disagree with selection role rates')
    feature = state.get('birth_death', {}).get('backend') == 'feature_cells'
    require(feature == (actual == 'feature_cells'), 'selected backend and saved reaction state disagree')
    if feature:
        require(state['birth_death']['population_policy'] == population, 'reaction population certificate differs')
        require('paired_average' in state['birth_death'] and 'lineage_neighbors' in state['birth_death'],
                'feature checkpoint lacks original lease/lineage state')
    return deepcopy(expected)


def validate_guard_state(state, roles, recipe):
    """Read the source-declared scalar guard certificate without changing it."""
    require(recipe.reopen_guard == 'settled', 'actual shared settled reopen guard must be enabled')
    guard = state.get('reopen_guard')
    require(isinstance(guard, dict) and set(guard) ==
            {'schema', 'ka2_anchor_started', 'calm_network', 'excursion', 'epoch_rebases'},
            'checkpoint settled guard schema differs')
    require(type(guard['schema']) is int and guard['schema'] == 1, 'guard schema must be integer1')
    require(guard['ka2_anchor_started'] is None or type(guard['ka2_anchor_started']) is bool,
            'guard objective phase must be bool or None')
    require(type(guard['epoch_rebases']) is int and guard['epoch_rebases'] >= 0,
            'guard epoch count must be a nonnegative integer')
    def witnesses(values):
        require(isinstance(values, dict), 'guard network witnesses must be a map')
        for key, witness in values.items():
            require(isinstance(key, str) and isinstance(witness, dict)
                    and set(witness) == {'role', 'scale'}, 'guard network witness schema differs')
            try:
                parts = key.split('.')
                require(len(parts) == 2, 'guard owner must be optimizer.group')
                i, j = map(int, parts)
                require(i >= 0 and j >= 0, 'guard owner index must be nonnegative')
                role = roles[i][j]
            except (IndexError, ValueError):
                raise RuntimeError('guard network witness points outside explicit optimizer owners')
            require(witness['role'] == role and role in ('generator', 'critic', 'encoder', 'router'),
                    'guard witness must belong to the explicit network owner')
            scale = witness['scale']
            require(type(scale) in (int, float) and math.isfinite(scale) and 0 < scale < 1,
                    'guard witness must certify a contracted finite positive ladder scale')
    witnesses(guard['calm_network'])
    excursion = guard['excursion']
    if excursion is not None:
        require(isinstance(excursion, dict) and set(excursion) == {'step', 'network'},
                'guard excursion schema differs')
        require(type(excursion['step']) is int and 0 <= excursion['step'] <= state['completed_steps'],
                'guard excursion cursor differs from training state')
        witnesses(excursion['network'])
    return deepcopy(guard)


def validate_initial_streams(torch, trainer):
    receipts = {}
    for name, offset in zip(trainer._STREAMS, (2, 3, 4, 5)):
        actual = getattr(trainer, name).get_state()
        expected = torch.Generator(device=trainer.device).manual_seed(314159 + offset).get_state()
        require(torch.equal(actual, expected), f'original initial stream changed: {name}')
        receipts[name] = dict(device=str(actual.device), dtype=str(actual.dtype),
                              numel=actual.numel(), sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
    birth = trainer.birth_death.stream.get_state()
    expected_birth = torch.Generator(device=trainer.device).manual_seed(314165).get_state()
    require(torch.equal(birth, expected_birth), 'original initial birth stream changed')
    receipts['birth_death.stream'] = dict(device=str(birth.device), dtype=str(birth.dtype), numel=birth.numel(),
                                        sha256=hashlib.sha256(birth.cpu().numpy().tobytes()).hexdigest())
    return receipts
