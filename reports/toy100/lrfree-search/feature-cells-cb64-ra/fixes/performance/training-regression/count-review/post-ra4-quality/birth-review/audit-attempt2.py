"""Independent immutable-source and saved-receipt review; no numerical rerun."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER = ROOT / 'integration/review/training-regression/post-ra4-quality'
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checked = {}


def check_map(values, directory=None):
    for name, expected in values.items():
        path = Path(name) if Path(name).is_absolute() else directory / name
        assert sha(path) == expected, str(path)
        checked[str(path)] = expected


ready_path = OWNER / 'READY.json'
assert sha(ready_path) == '69840cc4ddcb6a25b00b3a386eebd4c35671f24a96a191a0d4a8d43db69bd60a'
checked[str(ready_path)] = sha(ready_path)
ready = json.loads(ready_path.read_text())
package = Path(ready['package_root'])
base = Path(ready['base_package'])
check_map(ready['package_source_sha256'], package)
for key in ('local_source_sha256', 'numerical_source_sha256', 'evidence_sha256'):
    check_map(ready[key], OWNER)
check_map(ready['external_evidence_sha256'])
assert hashlib.sha256(json.dumps(ready['package_source_sha256'], sort_keys=True,
    separators=(',', ':')).encode()).hexdigest() == ready['full_package_source_sha256']
actual_package = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}
assert actual_package == ready['package_source_sha256']
base_map = {str(p.relative_to(base)): sha(p) for p in sorted(base.rglob('*.py'))}
check_map(base_map, base)
changed = [name for name in actual_package if actual_package[name] != base_map.get(name)]
assert changed == ready['changed_package_files'] == [
    'particlegan/anchor_birth.py', 'particlegan/birth_phase.py', 'particlegan/feature_cells.py']
for name in ('anchor_birth.py', 'birth_phase.py'):
    assert (OWNER / name).read_bytes() == (package / 'particlegan' / name).read_bytes()

old_tree = ast.parse((base / 'particlegan/feature_cells.py').read_text())
new_tree = ast.parse((package / 'particlegan/feature_cells.py').read_text())
get_class = lambda tree: next(n for n in tree.body if isinstance(n, ast.ClassDef)
    and n.name == 'FeatureCellSnapshot')
get_method = lambda cls: next(n for n in cls.body if isinstance(n, ast.FunctionDef)
    and n.name == 'select_parents')
old_cls, new_cls = get_class(old_tree), get_class(new_tree)
old_method, new_method = get_method(old_cls), get_method(new_cls)
assert [a.arg for a in new_method.args.kwonlyargs][-2:] == ['supported_counts', 'reserved_rows']
isolation_source = ast.unparse(new_method)
assert 'eligible[reserved_rows] = False' in isolation_source
assert 'allowed[reserved_rows] = False' in isolation_source
assert 'kept = supported_counts.clone()' in isolation_source
assert 'guard_passed=0 < len(dead) <= Q * len(features)' in isolation_source
new_cls.body[new_cls.body.index(new_method)] = deepcopy(old_method)
assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)

trees = {name: ast.parse((OWNER / name).read_text()) for name in ('anchor_birth.py', 'birth_phase.py')}
functions = {node.name: node for tree in trees.values() for node in tree.body
    if isinstance(node, ast.FunctionDef)}
source = {name: ast.unparse(node) for name, node in functions.items()}
for tree in trees.values():
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
    modules += [n.module for n in imports if isinstance(n, ast.ImportFrom)]
    assert all(m in ('math', 'time', 'torch', 'anchor_birth') for m in modules), modules
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any('rand' in name or name.endswith('.step') or name.endswith('.backward')
        or name.startswith('torch.cuda') for name in calls), calls
assert "multiplicity') != 3 * snapshot.cells + 2" in source['global_certificate_residual']
assert "cutoff') != 0.05 / (3 * snapshot.cells + 2)" in source['global_certificate_residual']
assert 'spent_birth = (destinations.remainder(2) == 0).sum()' in source['global_certificate_residual']
assert 'spent_death = (categories[children].remainder(2) == 1).sum()' in source['global_certificate_residual']
assert '(raw_birth - spent_birth).clamp_min(0)' in source['global_certificate_residual']
assert 'capacity = min(4, remaining' in source['allocate_anchor_births']
assert "attempt['current']" in source['allocate_anchor_births'] and "attempt['average']" in source['allocate_anchor_births']
assert 'pool_counts[cell] != 0' in source['allocate_anchor_births']
assert 'group_vacancies[groups[cell]] <= 0' in source['allocate_anchor_births']
assert 'donor_rows = (flags & ~inside & ~protected_sources)' in source['allocate_anchor_births']
assert 'remaining - len(children)' in source['allocate_anchor_births']
assert 'pvalues[0] > 0.05' in source['_accepted'] and 'category[0] == 2 * cell' in source['_accepted']
assert 'max_moves=budget' in source['plan_residual_global_copies']
assert "supported_counts=birth_plan['planned_supported_counts']" in source['plan_residual_global_copies']
assert "birth_plan['source_seed_rows']" in source['plan_residual_global_copies']
assert "attempt[model].pop('seconds', None)" in source['plan_real_anchor_births']
assert not any(word in source['novel_birth_diagnostics'] for word in ('seconds', 'time.', 'perf_counter'))
assert "lineage.neighbors[children] = -1" in source['invalidate_birth_rows']
assert 'for value in state.values()' in source['apply_anchor_births']
assert 'value.ndim and value.shape[0] == len(prior.z)' in source['apply_anchor_births']
assert 'history[children] = 0' in source['apply_anchor_births']
assert 'register_copies' not in source['apply_anchor_births']

receipts = {}
for name in ('cpu-contract-final/result.json', 'cpu-contract-final-edges/result.json',
             'saved-production-contract.json', 'source-contract.json'):
    obj = json.loads((OWNER / name).read_text())
    assert obj['status'] == 'PASS', name
    receipts[name] = obj
    for key, values in obj.items():
        if 'sha256' in key and isinstance(values, dict):
            check_map(values, OWNER)
    for record in obj.get('records', []) + obj.get('edge_records', []):
        assert record['checks'] and all(value is True for value in record['checks'].values()), record

mechanical = receipts['cpu-contract-final-edges/result.json']
assert mechanical['source_unchanged'] and mechanical['source_sha256_before'] == mechanical['source_sha256_after']
assert len(mechanical['records']) == 15 and len(mechanical['edge_records']) == 9
assert mechanical['cuda_initialized'] is False
by_name = {r['name']: r for r in mechanical['records']}
for r in mechanical['records']:
    assert r['mass'] + r['local'] + r['birth'] + r['global_copy'] <= r['budget']
    assert 0 <= r['birth'] <= 4
assert by_name['birth_budget_zero']['birth'] == 0
assert by_name['birth_budget_two']['birth'] == 2
assert by_name['absent_copy_parents']['birth'] == 4
assert by_name['global_certificate_exhaustion']['mass'] == 32
assert by_name['global_certificate_exhaustion']['birth'] == by_name['global_certificate_exhaustion']['global_copy'] == 0
assert by_name['small_guard_51']['isolation'] > 0 and by_name['small_guard_52']['isolation'] == 0
assert by_name['supported_mass_small_holes']['mass'] == 51 and by_name['supported_mass_small_holes']['isolation'] == 46

saved = receipts['saved-production-contract.json']
assert saved['global_rng_unchanged'] and saved['cuda_initialized'] is False
assert saved['new_training_steps'] == saved['committed_births'] == saved['generated_emissions'] == 0
assert saved['quality_verdict'] is None and len(saved['records']) == 2
saved_totals = []
for r in saved['records']:
    plan = r['novel_birth']
    assert plan['moves'] == r['birth'] == 4
    assert plan['paired_ema_births'] == 4 and plan['supported_copy_parent_rows'] == []
    assert all(category % 2 == 0 for category in plan['new_latent_destination_categories'])
    assert all(category % 2 == 1 for category in plan['death_category_ids'])
    assert len(set(plan['source_seed_rows'] + plan['child_rows'])) == 8
    assert r['mass'] + r['local'] + r['birth'] + r['global_copy'] == plan['budget'] == 51
    before, after = plan['certificates_before'], plan['certificates_after']
    for kind in ('death', 'birth'):
        assert after['spent_' + kind] == before['spent_' + kind] + 4
        assert after['residual_' + kind] == max(0, before['residual_' + kind] - 4)
        assert after['raw_' + kind] == before['raw_' + kind]
    for a in plan['acceptance']:
        assert a['current_p'] > .05 and a['paired_ema_p'] > .05
        assert 0 <= a['current_linearizations'] <= 4 and 0 <= a['paired_ema_linearizations'] <= 4
    saved_totals.append({k: r[k] for k in ('step', 'mass', 'local', 'birth', 'global_copy')})

assert all(sha(path) == expected for path, expected in checked.items())
receipt = dict(status='PASS', scope='Independent source/freeze/AST and saved CPU receipt audit; no duplicate numerical suite',
    checks=dict(owner_maps_exact=True, package_modules=29, changed_package_files=changed,
        only_existing_method_change='FeatureCellSnapshot.select_parents',
        unchanged_even_fitted_geometry_and_3K_plus_2_count_family=True,
        four_birth_work_bound=True, paired_original_support_and_requested_inside_cell=True,
        shared_ordinary_budget=True, gross_actual_destination_spending=True,
        no_certificate_replenishment=True, supported_ledger_and_group_caps=True,
        copy_parents_and_solver_seeds_kept_distinct=True, later_row_reservations=True,
        rare_survivors_preserved=True, inherited_isolation_guard=True,
        independent_live_ema_commit_and_row_state_reset=True,
        novel_incarnation_has_no_seed_lineage_link=True,
        no_random_draw_or_optimizer_step_in_helpers=True, saved_global_rng_unchanged=True,
        no_timing_in_semantic_diagnostics=True, mechanical_variants=15, rejection_cases=9,
        saved_model_states=2, saved_paired_births=8),
    saved_production_phase_totals=saved_totals, reviewed_hashes=checked,
    numerical_reruns=0, new_training_steps=0, new_seeds=0, cuda_contexts=0, quality_verdict=None,
    limits=['This qualifies the frozen helpers/API and retained CPU evidence, not learned/native quality.',
        'Allocation-only fixtures contain constructed proposed coordinates; actual saved model callbacks are separate evidence.',
        'Legacy isolation retains its separate small-flag guard; total isolation plus ordinary moves can exceed the ordinary 5% cap.',
        'Root integration/checkpoint/source scope is covered by separate independent and owner receipts.'])
assert not (OUT / 'receipt.json').exists() and not (OUT / 'FROZEN.json').exists()
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
frozen = dict(status='PASS', files={str(OUT / n): sha(OUT / n) for n in
    ('audit.py', 'REPORT.md', 'receipt.json', 'audit-attempt1.py', 'audit-attempt1.log')},
    reviewed_hashes=checked)
(OUT / 'FROZEN.json').write_text(json.dumps(frozen, indent=2) + '\n')
print(json.dumps(dict(status='PASS', receipt=str(OUT / 'receipt.json'),
    receipt_sha256=sha(OUT / 'receipt.json'), freeze_sha256=sha(OUT / 'FROZEN.json'),
    reviewed_files=len(checked)), indent=2))
