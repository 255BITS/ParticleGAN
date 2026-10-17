"""Stdlib-only seal of the completed manual source review; no model imports."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT/'integration/review/training-regression/post-ra9-quality/mean-category-transport'
EXPECTED_SEAL = '9818d0439d36b6801ee9c5c21554c5e9b504bf1f1494b1443ebcc89d1ee520ab'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not (HERE/'receipt.json').exists() and not (HERE/'FROZEN.json').exists()
    source_seal = OWNER/'SOURCE-FROZEN.json'
    assert sha(source_seal) == EXPECTED_SEAL
    protected = json.loads(source_seal.read_text())['source_and_input_sha256']
    assert len(protected) == 47
    for path, digest in protected.items():
        assert sha(path) == digest, path
    package = sorted((ROOT/'pkg-CB64-RA9').rglob('*.py'))
    assert len(package) == 29
    assert {str(p) for p in package} == {p for p in protected if '/pkg-CB64-RA9/' in p}
    helper_map = {}
    for path in sorted(OWNER.glob('*.py')):
        ast.parse(path.read_text())
        helper_map[str(path)] = sha(path)
    math = ROOT/'performance/training-regression/count-review/post-ra9-quality/mean-witness-review'
    extra = [source_seal, math/'BUNDLE-RECEIPT.json', math/'FROZEN.json']
    for path in extra:
        protected[str(path)] = sha(path)
    receipt = dict(status='PASS', scope='source-only preexecution integration review for one fixed scratch grid/toy prototype',
        finished_UTC=datetime.now(timezone.utc).isoformat(), source_preseal_sha256=EXPECTED_SEAL,
        owner_guard_count=47, unchanged_RA9_package_files=29, helper_source_sha256=helper_map,
        source_and_input_sha256=protected,
        findings_corrected=['inside-only eligibility', 'whole-packet prewrite schema checks',
            'paired table/cache/lineage layout', 'postdraw stream and bandwidth epoch',
            'parameter versions separated from exact buffer bytes and bindings'],
        reviewed=['bounded globally disjoint pairs and shared ordinary budget',
            'raw FAST/EMA inputs and scratch clone ownership', 'one shared draw and own-prior geometry',
            'actual offspring cell/category/group/support retention', 'virtual EMA objective',
            'exact stored-coordinate commit and copy optimizer/history inheritance',
            'single lineage/cache commit and consumed-packet rejection',
            'driver source guards before Torch/PT interpretation'],
        limitations=['pure functional fixture callbacks; no custom-module numerical neutrality test',
            'production reset/rebase/counters/serving/checkpoint and fourth-phase audit integration absent',
            'no statistical, emitted-distribution or quality qualification'],
        reviewer_torch_imports=0, reviewer_PT_interpretations=0, reviewer_forwards=0,
        reviewer_random_draws=0, reviewer_numerical_tests=0, reviewer_CUDA_contexts=0,
        owner_or_package_edits=0)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for path, digest in protected.items():
        assert sha(path) == digest, path
    locals_ = [HERE/name for name in ('PRECOMMIT.md','DRAFT-FINDINGS.md','REPORT.md','freeze_review.py','receipt.json')]
    frozen = dict(status='FROZEN_SOURCE_ONLY_REVIEW_PASS', created_UTC=datetime.now(timezone.utc).isoformat(),
        local_file_sha256={str(p):sha(p) for p in locals_}, protected_file_sha256=protected)
    (HERE/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({'receipt':str(HERE/'receipt.json'),'receipt_sha256':sha(HERE/'receipt.json'),
        'frozen':str(HERE/'FROZEN.json'),'frozen_sha256':sha(HERE/'FROZEN.json'),
        'local_files':len(locals_),'protected_files':len(protected)}))


if __name__ == '__main__':
    main()
