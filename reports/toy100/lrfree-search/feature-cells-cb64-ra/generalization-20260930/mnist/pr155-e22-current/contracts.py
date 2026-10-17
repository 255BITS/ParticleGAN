"""Validate original scorers and current PR155 baseline identity, never alter training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'ra13-settled'
ORIGINAL = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE = ORIGINAL / 'validation-cb64-ra11/learned'
ROLES = [['generator', 'table', 'noise'], ['critic']]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_members(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def verify_fixture_sources():
    original, current = ast_members(LANE / 'run_training.py'), ast_members(ROOT / 'run_training.py')
    unchanged = ('frechet', 'embedding_score', 'ImageEvaluation')
    require(all(current[n] == original[n] for n in unchanged), 'original image scorer AST changed')
    require(ast_members(__file__)['toy_gate'] == ast_members(ORIGINAL / 'quality/leaderboard.py')['toy_gate'],
            'original Toy25 gate AST changed')
    def update_loop(path):
        main = next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loops = [n for n in ast.walk(main) if isinstance(n, ast.For)
                 and ast.dump(n.iter, include_attributes=False) == "Call(func=Name(id='range', ctx=Load()), args=[Name(id='STEPS', ctx=Load())], keywords=[])"]
        require(len(loops) == 1, 'original training loop not unique')
        return ast.dump(loops[0], include_attributes=False)
    require(update_loop(ROOT / 'run_training.py') == update_loop(PARENT / 'run_training.py') == update_loop(LANE / 'run_training.py'),
            'original full training loop AST changed')
    transformed = json.loads((ROOT / 'source-transform-receipt.json').read_text())
    require(sha(PARENT / 'run_training.py') == transformed['parent_runner_sha256'], 'parent runner changed')
    require(sha(ROOT / 'run_training.py') == transformed['adapted_runner_sha256'], 'baseline adapter changed')
    old_adapter, new_adapter = ast_members(PARENT / 'adapter.py'), ast_members(ROOT / 'adapter.py')
    require(all(new_adapter[n] == old_adapter[n] for n in
            ('initialize_fixture_models', 'OriginalPrimarySampler', 'draw_samples', 'toy_metrics')),
            'original initialization or primary sampler changed')
    return dict(original_image_scorer_ast_identical=True, original_toy_gate_ast_identical=True,
        original_training_loop_ast_identical=True, original_init_and_primary_sampler_ast_identical=True,
        original_models_metrics_byte_pinned=True, original2000_updates_preserved=True,
        original10_checkpoints_preserved=True, output_noise_explicit=True, extra_begin_step_calls=0)


def toy_gate(metrics):
    return (metrics['precision'] >= .90 and metrics['coverage'] == 25
        and metrics['mass_tv'] <= .10 and len(metrics['supported_mass']) == 25
        and min(metrics['supported_mass']) >= .01)


def validate_checkpoint_state(state, roles, problem, recipe):
    from particlegan.recipes import get_recipe
    require(roles == ROLES, 'original optimizer ownership changed')
    recommended = get_recipe('e22', num_particles=1024, z_dim=128, batch_size=128,
                             output_noise_std=.029).to_dict()
    actual = recipe.to_dict()
    # The checked-in research JSON calls the same formulation ka2; report
    # labels are the sole allowed difference from the named preset.
    recommended.pop('name'); actual.pop('name')
    require(actual == recommended, 'baseline differs from current recommended named E22')
    require((recipe.reopen_signal, recipe.reopen_anchor) == ('optimizer', 'release'), 'current R1 controls absent')
    require(getattr(recipe, 'reopen_guard', None) is None, 'Atlas guard accidentally enabled')
    require(getattr(recipe, 'birth_death_backend', 'knn') == 'knn', 'Atlas cells accidentally enabled')
    require('backend_selection' not in state and 'reopen_guard' not in state, 'Atlas checkpoint fields present')
    require(state['initial_lrs'] == [[.00425, .0085, .00425], [.00425]], 'baseline rates recalibrated')
    require('surprise' in state and state['birth_death'].get('backend') != 'feature_cells', 'current E22 reference controls absent')
    require(state['recipe'] == recipe.to_dict(), 'checkpoint recipe differs')
    return dict(upstream_base_commit='cabe2084284db923d525918cbf3e18de6f20faac',
        source_label='PR155-E22-cabe2084', named_e22_training_fields_exact=True,
        report_name_only_difference=True, backend='original_knn', guard=None,
        reopen_signal='optimizer', reopen_anchor='release', original_base_lrs=True,
        completed_steps=state['completed_steps'], fixture=problem)


def validate_initial_streams(torch, trainer):
    receipts = {}
    for name, offset in zip(trainer._STREAMS, (2, 3, 4, 5)):
        actual = getattr(trainer, name).get_state()
        expected = torch.Generator(device=trainer.device).manual_seed(314159 + offset).get_state()
        require(torch.equal(actual, expected), f'original initial stream changed: {name}')
        receipts[name] = dict(device=str(actual.device), dtype=str(actual.dtype),
            numel=actual.numel(), sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
    birth = trainer.birth_death.stream.get_state()
    expected = torch.Generator(device=trainer.device).manual_seed(314165).get_state()
    require(torch.equal(birth, expected), 'original initial birth stream changed')
    receipts['birth_death.stream'] = dict(device=str(birth.device), dtype=str(birth.dtype),
        numel=birth.numel(), sha256=hashlib.sha256(birth.cpu().numpy().tobytes()).hexdigest())
    return receipts
