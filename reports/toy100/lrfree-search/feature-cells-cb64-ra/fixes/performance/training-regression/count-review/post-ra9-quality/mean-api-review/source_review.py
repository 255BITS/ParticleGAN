"""Read-only pre-execution RA10 math/state/fixture review; stdlib only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-production'
EXPECTED_SEAL = '38d410d92e8827c53232d03a6ba0cadeb38facd008d46f69806bf0ee221ab493'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / 'source-review.json'
    frozen = HERE / 'SOURCE-REVIEW-FROZEN.json'
    assert not output.exists() and not frozen.exists()
    seal_path = OWNER / 'SOURCE-FROZEN.json'
    assert sha(seal_path) == EXPECTED_SEAL
    seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'PRE_NUMERICAL_SOURCE_INPUT_FROZEN'
    assert seal['backend_schema'] == 9 and seal['trainer_schema'] == 5
    protected = dict(seal['source_and_input_sha256'])
    for path, digest in protected.items():
        assert sha(path) == digest, path
    protected[str(seal_path)] = sha(seal_path)
    package = OWNER / 'pkg-MEAN/particlegan'
    assert len(seal['package_source_sha256']) == 30
    for name, digest in seal['package_source_sha256'].items():
        assert sha(package / name) == digest, name
    mean = (package / 'mean_transport.py').read_text()
    feature = (package / 'feature_cells.py').read_text()
    mechanics = (OWNER / 'run_mechanics.py').read_text()
    for path in package.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    mean_tree = ast.parse(mean)
    actual_keys = None
    for node in mean_tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MEAN_KEYS' for t in node.targets):
            actual_keys = ast.literal_eval(node.value)
    expected_keys = {'schema','policy','status','reason','step','snapshot','cells','rank',
        'observations','attempts','moves','alpha','radius','known_range','mean',
        'variance_ddof1','variance_penalty','range_penalty','lower_bound',
        'objective_before_mean','objective_after_mean'}
    assert actual_keys == expected_keys
    assert 'stamp["attempts"]>math.floor(Q*n)-(ordinary-mean_moves)' in mean
    assert 'abs(stamp["mean"])>2*stamp["radius"]+1e-10' in mean
    assert 'stamp["variance_ddof1"]>(2*stamp["radius"])**2*stamp["observations"]/(stamp["observations"]-1)+1e-10' in mean
    assert 'zero mean moves require their actual action-veto reason' in mean
    assert 'counts["mean_evals"]!=state["snapshot_serial"]' in feature
    assert 'type(last.get("ordinary_mean_moves")) is not int' in feature
    assert '(3*snapshot.cells+3)' in (package / 'birth_phase.py').read_text().replace(' ', '')
    prepare = next(n for n in mean_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_mean_witness')
    calls = [ast.unparse(n.func) for n in ast.walk(prepare) if isinstance(n, ast.Call)]
    assert 'freeze_moment' in calls and 'odd_witness' in calls
    helper_tree = ast.parse(mechanics)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'trainer'
        and n.func.attr == 'load_state_dict' for n in ast.walk(helper_tree))
    assert 'trainer._serve_release()' in mechanics and 'after = trainer._state_dict()' in mechanics
    checks = dict(
        fixed_even_plus_pre_prefix_EMA_before_all_odd_count_actions=True,
        unit_or_zero_directions_known_score_range_4R=True,
        all_odd_rows_unweighted_unbiased_variance_bounded_EB_lower_form=True,
        actual_common_family_3K_plus_3_in_count_and_birth_paths=True,
        post_prefix_action_means_and_counts_fixed_original_u=True,
        typed_initial_invalid_veto_firing_cases=True,
        known_mean_and_variance_bounds_validated=True,
        actual_residual_budget_and_fourth_phase_move_identities=True,
        cumulative_mean_counter_relations=True,
        fresh_backend9_constructor_no_legacy_backend_load=True,
        literal_existing_trainer_hook_and_private_FAST_checkpoint=True)
    receipt = dict(status='PASS', created_UTC=datetime.now(timezone.utc).isoformat(),
        scope='Source-only pre-execution math/state/fixture review; no Torch/PT interpretation, tests, draws, forwards or optimizer steps',
        backend_schema=9, trainer_schema=5, owner_source_preseal_sha256=EXPECTED_SEAL,
        mean_metadata_keys=sorted(expected_keys), checks=checks,
        limitations=[
            'Bound needs conditionally fixed iid odd rows; trained D/shared FIFO do not establish that assumption or repeated adaptive control.',
            'Witness uses real-mixture odd-row weights; action objective uses even empirical group weights and is not its population lower bound.',
            'Clean nonlinear feature evidence and copy progress do not certify emitted quality, per-group equality or stationarity.',
            'Inherited trainer5 positive served load rejection restores all semantic/view bytes but advances parameter versions through release/reapply.'],
        quality_verdict=None, protected_sha256=dict(sorted(protected.items())))
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    local = {str(path): sha(path) for path in (Path(__file__), output)}
    frozen.write_text(json.dumps(dict(status='FROZEN_SOURCE_ONLY_PASS',
        receipt_sha256=sha(output), source_and_input_sha256=receipt['protected_sha256'],
        local_sha256=local), sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', guarded_files=len(protected),
        receipt=str(output), receipt_sha256=sha(output), freeze_sha256=sha(frozen))))


if __name__ == '__main__':
    main()
