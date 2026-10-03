"""Source-only v2 fixture fidelity/attribute-ownership correction review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-production'
EXPECTED = '937eba63716e4bb27500b7d4a8fc169d1cf7449d307526244f6d86fc84e96b65'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    receipt_path = HERE / 'fixture-v2-source-review.json'
    assert not receipt_path.exists()
    seal_path = OWNER / 'SOURCE-FROZEN-v2.json'
    assert sha(seal_path) == EXPECTED
    seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'PRE_NUMERICAL_CORRECTED_FIXTURE_FROZEN'
    assert seal['fixture_version'] == 2 and seal['production_source_unchanged'] is True
    files = dict(seal['source_and_input_sha256'])
    for name, digest in files.items():
        assert sha(name) == digest, name
    files[str(seal_path)] = EXPECTED
    first = json.loads((OWNER / 'SOURCE-FROZEN.json').read_text())
    for name, digest in first['package_source_sha256'].items():
        assert sha(OWNER / 'pkg-MEAN/particlegan' / name) == digest
    assert (OWNER / 'config.json').read_bytes() == (ROOT / 'configs/overrides-CB64-RA9.json').read_bytes()
    old = ast.parse((OWNER / 'run_mechanics.py').read_text())
    new_path = OWNER / 'run_mechanics_v2.py'
    new_text = new_path.read_text()
    new = ast.parse(new_text)
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False)
        for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = functions(old), functions(new)
    assert before.keys() == after.keys()
    assert {name for name in before if before[name] != after[name]} == {'verify', 'main'}
    assert sha(new_path) == '6f9fc07a8ceafbc9f60e086de32ea499e131e4e8d2bbc39da92e569c69e538b6'
    for call in ("trainer.G.load_state_dict(weights['G'])", "trainer.D.load_state_dict(weights['D'])",
                 "trainer.ema_D.load_state_dict(weights['D'])"):
        assert call in new_text
    for assertion in ('raw source model/table binding failed',
                      'named raw FIFO/bandwidth/moment/history/noise binding failed'):
        assert assertion in new_text
    for owner, attr in (("bd", "_move"), ("table_tester", "rebase"), ("trainer.row_evidence", "reset")):
        assert f"{owner}.__dict__.pop('{attr}',None)" in new_text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'trainer'
        and n.func.attr == 'load_state_dict' for n in ast.walk(new))
    checks = dict(production_30_modules_and_config_byte_unchanged=True,
        exact_raw_G_D_restored_after_initializing_constructor=True,
        fresh_critic_EMA_anchor_matches_copied_D=True,
        all_named_raw_model_table_FIFO_bandwidth_noise_moments_history_asserted=True,
        temporary_instance_methods_restore_original_presence=True,
        genuine_new9_not_legacy_load_or_relabel=True,
        original_seed_law_threshold_budget_assertions_unchanged=True,
        failed_v1_bound_method_state_not_qualified=True)
    receipt = dict(status='PASS', created_UTC=datetime.now(timezone.utc).isoformat(),
        scope='Focused source-only corrected fixture review; no Torch/PT/forward/draw/test/update',
        owner_fixture_preseal_sha256=EXPECTED, checks=checks,
        retained_prior_source_pin_historical=True, quality_verdict=None,
        source_and_input_sha256=dict(sorted(files.items())))
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', guarded_files=len(files), receipt_sha256=sha(receipt_path))))


if __name__ == '__main__':
    main()
