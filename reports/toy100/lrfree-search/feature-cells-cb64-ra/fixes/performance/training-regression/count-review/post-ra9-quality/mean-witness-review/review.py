"""Source-only conditional witness review. Never import Torch or load a state."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-transport'
PKG = ROOT / 'pkg-CB64-RA9'
EXPECTED_PACKAGE = '2d00c77ae0e7545ac253ff81aa729158d69016be82edc117e5597bdd664b86ce'
EXPECTED_WITNESS = '5c6cdfb31a1023c746cd25514bdc7225e75d94387df0edb1920eb6b5ebd6bcb7'
EXPECTED_DESIGN = 'ebc240caa2888b5977743fb782568285ea89407e7708df078d53446319a4a7c3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expression(tree, function, target):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    for value in ast.walk(node):
        if isinstance(value, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in value.targets):
            return ast.unparse(value.value)
    raise AssertionError((function, target))


def main():
    package_files = sorted(PKG.rglob('*.py'))
    assert len(package_files) == 29
    identity = hashlib.sha256()
    for path in package_files:
        identity.update(path.relative_to(PKG).as_posix().encode() + b'\0')
        identity.update(path.read_bytes() + b'\0')
    assert identity.hexdigest() == EXPECTED_PACKAGE
    witness = OWNER / 'witness.py'
    design = OWNER / 'DESIGN.md'
    transport = OWNER / 'transport.py'
    assert digest(witness) == EXPECTED_WITNESS
    assert digest(design) == EXPECTED_DESIGN
    tree = ast.parse(witness.read_text())
    frozen = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'freeze_moment')
    assert [a.arg for a in frozen.args.args] == ['snapshot', 'even_features', 'ema_features']
    assert not any(isinstance(n, ast.Name) and 'odd' in n.id for n in ast.walk(frozen))
    assert expression(tree, 'freeze_moment', 'radius') == 'math.sqrt(snapshot.rank / Q)'
    assert expression(tree, 'freeze_moment', 'residual') == 'even_means - ema_means'
    assert expression(tree, 'odd_witness', 'alpha') == 'Q / (3 * snapshot.cells + 3)'
    assert expression(tree, 'odd_witness', 'variance') == 'values.var(unbiased=True)'
    assert expression(tree, 'odd_witness', 'mean') == 'values.mean()'
    assert expression(tree, 'odd_witness', 'known_range') == '4 * fixed.radius'
    assert expression(tree, 'odd_witness', 't') == 'math.log(2 / alpha)'
    assert expression(tree, 'odd_witness', 'variance_penalty') == '(2 * variance * t / len(values)).sqrt()'
    assert expression(tree, 'odd_witness', 'range_penalty') == '7 / 3 * known_range * t / (len(values) - 1)'
    assert expression(tree, 'odd_witness', 'lcb') == 'mean - variance_penalty - range_penalty'
    assert expression(tree, 'common_count_family', 'cutoff') == 'Q / (3 * snapshot.cells + 3)'
    assert expression(tree, 'common_count_family', 'significant') == "data['pvalues'] <= cutoff"
    code = ast.unparse(tree)
    assert 'len(odd_features) <= 1' in code and 'zero_direction_observations' in code
    assert 'values.abs() <= 2 * fixed.radius + 1e-10' in code
    transport_tree = ast.parse(transport.read_text())
    eligibility = next(n for n in transport_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observe_view')
    assert 'categories % 2 == 0' in ast.unparse(eligibility)
    assert 'torch.isfinite(coordinates).all(1)' in ast.unparse(eligibility)
    preview = next(n for n in transport_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'preview_pairs')
    assert 'vf.eligible & ve.eligible' in ast.unparse(preview)
    protocol = ' '.join(design.read_text().split())
    assert 'Root selected inside-only before measurement' in protocol
    assert 'There are no odd-count importance weights.' in protocol
    assert 'Clean-objective progress does not prove emitted' in protocol
    assert 'shared' in protocol and 'no repeated adaptive or population-stationarity guarantee' in protocol
    config = ROOT / 'configs/overrides-CB64-RA9.json'
    assert digest(config) == 'b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
    inputs = {str(p): digest(p) for p in [witness, design, config, *package_files]}
    receipt = dict(
        status='PASS', scope='pre-execution source-only conditional mean-witness mathematics',
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        numerical_execution=False, torch_import=False, checkpoint_interpretation=False,
        model_forward=False, RNG_draws=0, CUDA_context=False, inferential_authority=False,
        package_sha256=identity.hexdigest(), input_sha256=inputs,
        inspected_transport_at_time_sha256=digest(transport),
        inspected_transport_fragment_ast_sha256={
            'observe_view': hashlib.sha256(ast.dump(eligibility, include_attributes=False).encode()).hexdigest(),
            'preview_eligibility_check': 'vf.eligible & ve.eligible',
        },
        checks={
            'even_only_centers_scales_topology_and_unit_or_zero_residual': 'PASS',
            'invalid_missing_zero_scale_groups_whole_witness_veto': 'PASS',
            'all_odd_rows_no_postselection_or_importance_weighting': 'PASS',
            'n_at_least_two_before_ddof1_and_denominator': 'PASS',
            'norm_R_score_bounds_and_known_range_4R': 'PASS',
            'lower_tail_empirical_Bernstein_formula_and_strict_trigger': 'PASS',
            'actual_K_common_3K_plus_3_raw_count_rethresholding': 'PASS',
            'even_objective_vs_true_mixture_witness_weights_distinguished': 'PASS',
            'premeasurement_inside_only_restriction': 'PASS',
            'shared_training_and_clean_vs_emitted_limits': 'PASS',
        },
        report_sha256=digest(HERE / 'REPORT.md'),
        outside_scope=['prototype input freeze and driver', 'preview/packet ownership qualification',
                       'numerical reachability or power', 'production integration/checkpoint law',
                       'training, emitted quality, and candidate qualification'],
        requirements_before_execution=[
            'Owner must freeze all helpers/protocol/source/input maps.',
            'Lineage reviewer must independently qualify the completed preview/driver.',
            'witness.py and DESIGN.md hashes must remain those in this receipt; any change needs a supplemental source review.',
            'One fixed grid-then-toy CPU invocation only; retain any zero lower bound or capacity as negative evidence.',
        ])
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], reviewed_inputs=len(inputs),
                         witness_sha256=digest(witness), design_sha256=digest(design),
                         receipt_sha256=digest(HERE / 'receipt.json')), sort_keys=True))


if __name__ == '__main__':
    main()
