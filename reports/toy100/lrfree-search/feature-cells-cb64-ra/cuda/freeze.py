"""Freeze restored-CUDA execution sources and unchanged external evidence."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'feature-cells-config-20260929'
DATA = ROOT.parent / 'scaling-portability-20260929/validation'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (ROOT / 'source-freeze.json').exists():
        raise SystemExit('An execution freeze already exists; do not replace it.')
    previous = json.loads((PREVIOUS / 'review/source-freeze.json').read_text())
    externals = {}
    for name, expected in previous['candidate_files'].items():
        p = PREVIOUS / name
        assert sha(p) == expected, p
        externals[str(p)] = expected
    baseline = json.loads((PREVIOUS / 'validation/candidate-freeze.json').read_text())
    for variant in baseline['variants'].values():
        for name, expected in variant['source_sha256'].items():
            p = Path(variant['package_root']) / 'particlegan' / name
            assert sha(p) == expected, p
            externals[str(p)] = expected
        p = Path(variant['config_path'])
        assert sha(p) == variant['config_sha256'], p
        externals[str(p)] = variant['config_sha256']
    for name, expected in baseline['frozen_harness_sources'].items():
        p = HARNESS / name
        assert sha(p) == expected, p
        externals[str(p)] = expected
    fixture = json.loads((HARNESS / 'tasks/native100_fixture.json').read_text())
    for name, expected in fixture['host_source_sha256'].items():
        p = Path(fixture['frozen_repo']) / name
        assert sha(p) == expected, p
        externals[str(p)] = expected
    for name in ('models_metrics.py', 'data-receipt.json', 'evaluator.pt',
                 'EVALUATOR-DIAGNOSTIC-PROTOCOL.md', 'data/toy-stream.pt',
                 'data/image-stream.pt', 'runs/toy/E22/config.json', 'runs/mnist/E22/config.json'):
        p = DATA / name
        externals[str(p)] = sha(p)
    for p in (DATA / 'data/MNIST').rglob('*'):
        if p.is_file():
            externals[str(p)] = sha(p)
    for p in HARNESS.rglob('*'):
        if p.is_file() and p.suffix in ('.gz', '.pt', '.npy', '.npz'):
            externals[str(p)] = sha(p)
    externals[str(PREVIOUS / 'implementation/READY.json')] = sha(PREVIOUS / 'implementation/READY.json')
    externals[str(PREVIOUS / 'review/source-freeze.json')] = sha(PREVIOUS / 'review/source-freeze.json')
    ready = {}
    local_files = {}
    for lane in ('learned', 'screens'):
        p = ROOT / lane / 'READY.json'
        receipt = json.loads(p.read_text())
        assert receipt['status'] == 'READY', lane
        assert not receipt.get('quality_runs_started', False), lane
        assert not receipt.get('numerical_execution_started', False), lane
        ready[lane] = dict(path=str(p), sha256=sha(p), receipt=receipt)
        local_files[str(p.relative_to(ROOT))] = sha(p)
        # Freeze declared inputs; reports and leaderboards are generated outputs.
        declared = receipt.get('local_source_sha256', receipt.get('lane_source_sha256'))
        assert isinstance(declared, dict) and declared, lane
        for name, expected in declared.items():
            source = ROOT / lane / name
            assert sha(source) == expected, source
            if source.suffix == '.py':
                ast.parse(source.read_text())
            local_files[str(source.relative_to(ROOT))] = expected
        for name in ('SOURCE-FREEZE.json', 'source-freeze.json', 'preparation-receipt.json'):
            source = ROOT / lane / name
            if source.exists():
                local_files[str(source.relative_to(ROOT))] = sha(source)
    for name in ('PROTOCOL.md', 'launch.py', 'freeze.py'):
        p = ROOT / name
        if p.suffix == '.py':
            ast.parse(p.read_text())
        local_files[name] = sha(p)
    cpu_manifest = PREVIOUS / 'artifact_manifest.json'
    cpu_files = json.loads(cpu_manifest.read_text())['files']
    for name, record in cpu_files.items():
        assert sha(PREVIOUS / name) == record['sha256'], name
    result = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  candidate=previous, variants=baseline['variants'],
                  lanes=ready, local_sources=local_files, external_sources=externals,
                  original_cpu_archive=dict(path=str(cpu_manifest), sha256=sha(cpu_manifest),
                                            files={name: r['sha256'] for name, r in cpu_files.items()}),
                  purpose='Canonical CUDA retest; no candidate, fixture or gate changes.')
    (ROOT / 'source-freeze.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(event='cuda_execution_frozen', local_files=len(local_files),
                          external_files=len(externals), candidate_package=previous['package_sha256'])))


if __name__ == '__main__':
    main()
