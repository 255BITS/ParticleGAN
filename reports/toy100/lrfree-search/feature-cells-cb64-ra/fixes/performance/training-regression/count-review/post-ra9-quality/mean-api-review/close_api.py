"""Close the independent API gate after both CPU invocations exited."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'quality/ra10/integration-contract'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = HERE / 'continuation-attempt1/receipt.json'
    assert sha(source) == '5e0a95b7e362bf798b0744aea14b74eadbd09d46db50f8071118a16226518aa7'
    receipt = json.loads(source.read_text())
    assert receipt['status'] == 'PASS' and receipt['CPU_updates_total'] == 2
    assert receipt['API_sample_calls'] == 6 and receipt['API_rows_per_sample'] == 17
    assert receipt['earlier_passing_cases_repeated'] is False
    assert receipt['earlier_API_samples_repeated'] is False
    assert sum(len(r['rejected_controls']) for r in receipt['records']) == 17
    assert all(r['status'] == 'PASS' and r['serving']['status'] == 'PASS' for r in receipt['records'])
    assert receipt['cuda_initialized'] is False and receipt['quality_verdict'] is None
    for name, digest in receipt['source_and_input_sha256'].items():
        assert sha(name) == digest, name
    assert not OUTPUT.exists(), 'preserve existing gate artifacts'
    OUTPUT.mkdir(parents=True)
    target = OUTPUT / 'receipt.json'
    target.write_bytes(source.read_bytes())
    report = OUTPUT / 'REPORT.md'
    report.write_text('''# Independent RA10 CPU API/state qualification

PASS against composed package c8f8b343 and the unchanged RA9 config.
Trainer schema5 and genuine fresh backend9 initial/reacted states cold-load
with complete state equality; derived charts, geometry caches and moved-row
references are discarded. Returned public checkpoints are independent.

Grid has an actual firing mean witness with935 mean copies; toy has a real
veto with51 legacy actions and no mean preview. Both measured positive
serving leases pass warm/cold sample continuation: forwarded row IDs, outputs,
evaluation cursor and full state match. Six API calls total,17 rows each,
without scoring. Actual phase totals, common3K+3 metadata and persisted moved
row evidence/participation agree with the separate24-control hook proof.

All17 malformed metadata controls reject atomically for semantic state,
served values, RNG and existing cache contents. The inherited trainer5 served
release/reapply increases served parameter versions by two on rejection;
this declared cache invalidation is preserved.

Exactly two CPU updates total (one next update per branch) match every loss,
state leaf, gradient/mode and cursor. The original recipe is horizon-free
(total_steps=None); the mechanical clock1->2 remains below the original
2000-update toy schedule. No extra reaction, planner or passing API case was
repeated. The initial private None-horizon guard failure and its already
passed records remain frozen; continuation changes only that helper guard.

All protected bytes remain unchanged. No CUDA, quality score, quality cloud,
new seed, serving/noise override or production change occurred. These are
CPU fresh-law mechanical fixtures, not historical CUDA replay. The witness's
trained-D/shared-FIFO conditional-independence caveat and both strict toy plus
Grid100 quality gates remain separate and unqualified by this receipt.
''')
    locals = {str(p): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
    locals.update({str(p): sha(p) for p in (target, report)})
    frozen = OUTPUT / 'FROZEN.json'
    frozen.write_text(json.dumps(dict(status='FROZEN_POST_EXIT_PASS',
        receipt_sha256=sha(target), package_sha256=receipt['package_sha256'],
        source_and_input_sha256=receipt['source_and_input_sha256'], local_sha256=locals,
        retained_failed_preparations_and_helper_attempts=True,
        CPU_updates_total=2, API_sample_calls=6, quality_verdict=None), sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', receipt=str(target), receipt_sha256=sha(target),
        authoritative_freeze=str(frozen), freeze_sha256=sha(frozen),
        protected_inputs=len(receipt['source_and_input_sha256']), closed_local_files=len(locals))))


if __name__ == '__main__':
    main()
