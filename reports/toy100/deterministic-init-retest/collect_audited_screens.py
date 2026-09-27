"""Archive newly completed API screens only after independent runtime audit."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BATCH = Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/batch.json')


def previous_score(name, new_source):
    old = ROOT.parent / 'continuous-api-search/evidence' / (name + '-mode_hold')
    if not (old / 'result.json').exists():
        return 'NOT_MEASURED_ON_THIS_EXACT_SCREEN'
    before = json.loads((old / 'declaration.json').read_text())
    manifest = json.loads((ROOT / 'port-source' / name / 'port-manifest.json').read_text())
    expected = {key: value['candidate_sha256'] for key, value in manifest['files'].items()
                if value['candidate_sha256'] is not None}
    declared = {key: value for key, value in before['source_sha256'].items()
                if key.startswith('particlegan/') and key.endswith('.py')}
    assert declared == expected, 'Old score source does not match this candidate'
    result = json.loads((old / 'result.json').read_text())
    convergence = result['metrics']['verdict']['convergence']
    assert convergence['complete'] and convergence['observations'] == 24
    old_recipe = dict(before['recipe'])
    new_recipe = dict(json.loads((new_source / 'runtime.json').read_text())['recipe'])
    old_recipe.pop('initialization', None)
    new_recipe.pop('initialization', None)
    caveat = '' if old_recipe == new_recipe else ' (recipe metadata differs; inspect sources)'
    return (f"{result['status']} {convergence['passing_observations']}/24; "
            f"final streak {convergence['passing_suffix']}" + caveat)


def main():
    added = []
    for record in json.loads(BATCH.read_text()):
        for path in Path(record['directory']).glob('*/repo/reports/fixed-init-mode-hold/*/result.json'):
            name = path.parent.name
            audit_path = ROOT / (name + '-runtime-audit.json')
            if (ROOT / 'evidence' / (name + '-new-init')).exists() or not audit_path.exists():
                continue
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == 'PASS' and Path(audit['source']) == path.parent
            subprocess.run([sys.executable, str(ROOT / 'archive_screen.py'), '--source', str(path.parent),
                '--id', name + '-new-init', '--old-comparison', previous_score(name, path.parent)], check=True)
            added.append(name)
    print(json.dumps(dict(archived=added)))


if __name__ == '__main__':
    main()
