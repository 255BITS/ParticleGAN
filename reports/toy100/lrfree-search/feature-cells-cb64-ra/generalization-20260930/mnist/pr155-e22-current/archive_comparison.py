"""Add the current E22 comparative evidence as small artifacts only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT.parents[1]
ALLOWED = {'.py', '.md', '.json', '.jsonl', '.log'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = ROOT / 'ARCHIVE-APPEND.json'
    assert not target.exists(), 'preserve sealed inventories'
    closure = json.loads((ROOT / 'comparison-closure/FROZEN.json').read_text())
    assert closure['status'] == 'COMPLETE_CURRENT_PR155_E22_VS_ATLAS_COMPARISON' and closure['validity'] == 'VALID'
    for name, value in closure['artifact_sha256'].items():
        assert sha(ROOT / 'comparison-closure' / name) == value
    baseline = GENERAL / 'visualization-pr223-convergence'
    selected = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ALLOWED and p != target]
    selected += [p for p in sorted((baseline / 'pkg-pr155-e22-cabe2084').rglob('*.py'))]
    selected += [baseline / 'BASELINE-SOURCE-FREEZE.json', baseline / 'E22-RECIPE-IDENTITY.json',
                 baseline / 'configs/pr155-e22.json']
    selected += [p for p in sorted((baseline / 'source-reference').rglob('*')) if p.is_file() and p.suffix in ALLOWED]
    selected = sorted(set(selected))
    files = [dict(path=str(p.relative_to(GENERAL)), sha256=sha(p), bytes=p.stat().st_size) for p in selected]
    assert len(files) == len({f['path'] for f in files})
    value = dict(status='SEALED_ADDITIVE_CURRENT_PR155_E22_LEARNED_COMPARISON',
        created_utc=datetime.now(timezone.utc).isoformat(), root=str(GENERAL), files=files,
        file_count=len(files), total_bytes=sum(p['bytes'] for p in files),
        copy_policy='Add absent relative paths; accept existing byte-identical targets; reject all conflicts.',
        excludes=['weights', 'checkpoints', 'datasets', 'sample clouds', 'images', 'lockfiles'],
        atlas_original_v4_qualification_unchanged=True, archived_e22_none_not_relabelled=True,
        visualization_pin='User-authorized GIF/SVG/PNG and rendering receipts are separate additive artifacts.')
    with target.open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=value['status'], file_count=value['file_count'],
                         total_bytes=value['total_bytes'], inventory_sha256=sha(target), path=str(target))))


if __name__ == '__main__':
    main()
