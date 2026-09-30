"""Raw-byte guard and source order only; never execute prototype or parse PT."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER = ROOT / 'integration/review/training-regression/post-ra9-quality/mean-category-transport'
EXPECTED_PRESEAL = '9818d0439d36b6801ee9c5c21554c5e9b504bf1f1494b1443ebcc89d1ee520ab'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    preseal_path = OWNER / 'SOURCE-FROZEN.json'
    assert digest(preseal_path) == EXPECTED_PRESEAL
    preseal = json.loads(preseal_path.read_text())
    assert preseal['status'] == 'PRE_NUMERICAL_SOURCE_INPUT_FROZEN'
    assert preseal['numerical_reads'] == preseal['torch_imports'] == preseal['PT_interpretations'] == 0
    assert preseal['intended_runs'] == 1
    inputs = preseal['source_and_input_sha256']
    assert len(inputs) == 47
    for path, expected in inputs.items():
        assert digest(path) == expected, path
    mathematical = json.loads((HERE / 'receipt.json').read_text())
    assert mathematical['status'] == 'PASS' and not mathematical['inferential_authority']
    for path, expected in mathematical['input_sha256'].items():
        assert inputs[path] == expected
    driver = OWNER / 'run_prototype.py'
    tree = ast.parse(driver.read_text())
    calls = {name: [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == name]
             for name in ('freeze_moment', 'odd_witness', 'common_count_family')}
    assert all(len(value) == 1 for value in calls.values())
    frozen, odd = calls['freeze_moment'][0], calls['odd_witness'][0]
    assert ast.unparse(frozen) == 'freeze_moment(snapshot, real_features[0::2], ema_features)'
    assert ast.unparse(odd) == 'odd_witness(snapshot, fixed, real_features[1::2])'
    assert frozen.lineno < odd.lineno
    assert ast.unparse(calls['common_count_family'][0]) == 'common_count_family(snapshot, fast_features)'
    result = dict(
        status='PASS', scope='final pre-execution source and input-byte binding for the conditional witness',
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_preseal_sha256=EXPECTED_PRESEAL,
        mathematical_receipt_sha256=digest(HERE / 'receipt.json'),
        input_sha256={**inputs, str(preseal_path): digest(preseal_path)},
        checked_inputs=len(inputs),
        driver_even_EMA_direction_before_all_odd='PASS',
        fresh_common_family_driver_dispatch='PASS',
        raw_bytes_only=True, checkpoint_interpretations=0, Torch_imports=0,
        model_forwards=0, RNG_draws=0, numerical_prototype_runs=0,
        inferential_authority=False,
        limits=[
            'Fixed-score iid premises are not established by shared trained D/FIFO/EMA data.',
            'The ordinary odd scalar mean and the even-mass squared copy objective use different group weights.',
            'Clean learned-feature progress does not certify emitted mean or original quality.',
            'Full preview/packet ownership is independently reviewed by lineage before execution.',
            'No measured power, legal-capacity, stationarity, or candidate-quality claim.',
        ])
    output = HERE / 'BUNDLE-RECEIPT.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], verified_maps=len(inputs),
                         bundle_receipt_sha256=digest(output)), sort_keys=True))


if __name__ == '__main__':
    main()
