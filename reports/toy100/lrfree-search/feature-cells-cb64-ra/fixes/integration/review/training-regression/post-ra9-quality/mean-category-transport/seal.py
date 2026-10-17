"""Stdlib-only pre-numerical source/input seal; never reads PT values."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = HERE/'SOURCE-FROZEN.json'
    if target.exists():
        raise SystemExit('preseal already exists; preserve it and use a separate attempt')
    local = sorted(HERE.glob('*.py')) + [HERE/'DESIGN.md', HERE/'PROTOCOL.json']
    package = sorted((ROOT/'pkg-CB64-RA9').rglob('*.py'))
    inputs = [
        ROOT/'validation-cb64-ra9/screens/runs/grid100/final-state.pt',
        ROOT/'validation-cb64-ra9/learned/training/toy/CB64-RA9/checkpoint-2000.pt',
        ROOT/'quality/ra9/READY.json',
        ROOT/'quality/ra9/COMPOSITION.json',
        ROOT/'configs/overrides-CB64-RA9.json',
        ROOT/'validation-cb64-ra9/source-freeze.json',
        ROOT/'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py',
        ROOT/'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py',
        ROOT/'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py',
        ROOT/'integration/review/training-regression/post-ra9-quality/local-moment-authority/attempt1/result.json',
        ROOT/'integration/review/training-regression/post-ra9-quality/local-moment-authority/FROZEN.json',
        ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/mean-preview-review/PRECOMMIT.md',
    ]
    for source in local:
        if source.suffix == '.py':
            ast.parse(source.read_text())
    paths = sorted(set(local+package+inputs))
    assert all(p.is_file() for p in paths)
    receipt = dict(status='PRE_NUMERICAL_SOURCE_INPUT_FROZEN',
        created_UTC=datetime.now(timezone.utc).isoformat(), numerical_reads=0,
        torch_imports=0, PT_interpretations=0, intended_runs=1,
        cases=['saved final RA9 grid', 'saved final RA9 toy'],
        source_and_input_sha256={str(p):sha(p) for p in paths})
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(preseal_path=str(target), sha256=sha(target), files=len(paths))))


if __name__ == '__main__':
    main()
