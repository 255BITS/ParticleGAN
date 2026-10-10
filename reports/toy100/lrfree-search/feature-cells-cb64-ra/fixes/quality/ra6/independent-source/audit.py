"""Independent stdlib proof of the RA6 diagnostic serialization correction."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'pkg-CB64-RA5'
PACKAGE = ROOT / 'pkg-CB64-RA6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(package):
    return {str(p.relative_to(package / 'particlegan')): sha(p)
        for p in sorted((package / 'particlegan').rglob('*.py'))}


def once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)


def main():
    old_ready_path = ROOT / 'quality/ra5/READY.json'
    assert sha(old_ready_path) == '13b8210f19765796b08dc7c189c7d86f20008601cd717511080122a9f6f4c35c'
    old_ready = json.loads(old_ready_path.read_text())
    for p, expected in old_ready['numerical_source_sha256'].items():
        assert sha(p) == expected, p
    old = sources(BASE)
    new = sources(PACKAGE)
    assert old == old_ready['package_source_sha256']
    assert set(old) == set(new) and len(new) == 29
    assert [n for n in old if old[n] != new[n]] == ['feature_cells.py']
    composition_path = ROOT / 'quality/ra6/COMPOSITION.json'
    composition = json.loads(composition_path.read_text())
    assert new == composition['source_sha256']
    assert composition['base_ready_sha256'] == sha(old_ready_path)
    value = (PACKAGE / 'particlegan/feature_cells.py').read_text()
    base = (BASE / 'particlegan/feature_cells.py').read_text()
    # Four explicit independent inverse edits, confined to one diagnostics block.
    restored = once(value,
        '            novel_diagnostics = novel_birth_diagnostics(birth_plan)\n            last.update(ordinary_copy_moves=',
        '            last.update(ordinary_copy_moves=')
    restored = once(restored, 'novel_birth_target_cells=novel_diagnostics["new_latent_target_cells"]',
        'novel_birth_target_cells=birth_plan["target_cell_ids"].clone()')
    restored = once(restored,
        'novel_birth_children=novel_diagnostics["child_rows"],novel_birth_seed_rows=novel_diagnostics["source_seed_rows"]',
        'novel_birth_children=newborn.clone(),novel_birth_seed_rows=birth_plan["source_seed_rows"].clone()')
    restored = once(restored, 'novel_birth=novel_diagnostics', 'novel_birth=novel_birth_diagnostics(birth_plan)')
    assert restored == base
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(base), include_attributes=False)
    tree = ast.parse(value)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureCellBirthDeath')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'maybe_apply')
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == 'novel_birth_diagnostics']
    assert len(calls) == 1
    helper_tree = ast.parse((PACKAGE / 'particlegan/birth_phase.py').read_text())
    diagnostics = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'novel_birth_diagnostics')
    helper_source = ast.unparse(diagnostics)
    assert 'tensor.detach().cpu().tolist()' in helper_source
    assert 'new_latent_target_cells=values(' in helper_source
    assert 'child_rows=values(' in helper_source and 'source_seed_rows=values(' in helper_source
    config = ROOT / 'configs/overrides-CB64-RA6.json'
    assert config.read_bytes() == (ROOT / 'configs/overrides-CB64-RA5.json').read_bytes()
    assert sha(config) == composition['config_sha256'] == old_ready['config_sha256']
    digest = hashlib.sha256()
    for p in sorted((PACKAGE / 'particlegan').rglob('*.py')):
        compile(p.read_text(), str(p), 'exec')
        digest.update(str(p.relative_to(PACKAGE / 'particlegan')).encode() + b'\0' + p.read_bytes() + b'\0')
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
        package_root=str(PACKAGE), package_sha256=digest.hexdigest(), package_source_sha256=new,
        composition_sha256=sha(composition_path), base_ready_sha256=sha(old_ready_path), config_sha256=sha(config),
        checks=dict(all_29_files_scoped=True, other_28_modules_byte_exact=True,
            four_independent_inverse_edits_restore_complete_RA5_bytes_and_AST=True,
            changed_block_only_diagnostics=True, existing_diagnostics_helper_called_once=True,
            cached_helper_lists_match_all_three_new_JSON_fields=True, config_bytes_exact=True,
            all_proposal_acceptance_moves_training_math_RNG_code_exact=True,
            inherited_API_schemas_policies_and_quality_gates_unchanged=True,
            original_RA5_freeze_and_failed_evidence_preserved=True),
        numerical_tests_repeated=0, cuda_initialized=False, defects=[],
        qualification='Source equivalence only; count owner performs saved-input JSON serialization contract. No quality verdict.')
    assert not (HERE / 'receipt.json').exists()
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', package_sha256=receipt['package_sha256'], receipt_sha256=sha(HERE / 'receipt.json'))))


if __name__ == '__main__':
    main()
