"""Seal source-only native artifact helpers before any checkpoint reads."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    assert not (HERE / 'CHECKER-FROZEN.json').exists()
    ready_path = ROOT / 'quality/ra10/READY.json'
    assert sha(ready_path) == '9f29f98761b136055ef5bfce3697de0cd45cb8d12c2d65fee01e5f5281edf6fd'
    ready = read(ready_path)
    assert ready['package_sha256'] == 'c8f8b343bded25ce8153bdd74ab9f722b369c0e087aee8a44d3c6e816baef2e4'
    assert ready['backend_schema'] == 9 and ready['trainer_schema'] == 5
    paths = {Path(p) for p in ready['numerical_source_sha256']}
    frozen_path = ROOT / 'validation-cb64-ra10/source-freeze.json'
    assert sha(frozen_path) == '4cd1be039f2d6fb9bd12fe3d93bae60ca331bb68d77d365c9eadb4019bd8b856'
    frozen = read(frozen_path)
    paths.update(Path(p) for p in frozen['external_sources'])
    paths.update(ROOT / 'validation-cb64-ra10' / p for p in frozen['local_sources'])
    original = ROOT / 'integration/review/audit_learned.py'
    assert sha(original) == '683acf472985084a911edd6763d63d411772563d7fcb71df4a547d2a502347ba'
    baseline_audit = ROOT.parent / 'feature-cells-cuda-retest-20260929/audit'
    baseline_ready = read(baseline_audit / 'READY.json')
    assert sha(baseline_audit/'check_artifacts.py') == baseline_ready['files']['check_artifacts.py']
    assert baseline_ready['files']['check_artifacts.py'] == '4b405136829b2275218db1406ba346dada3c62dd1b931cdd71fdb7069f739aae'
    paths.update((ready_path, frozen_path, original,
        ROOT.parent / 'feature-cells-cuda-retest-20260929/audit/check_artifacts.py',
        ROOT / 'integration/review/ra10-toy-artifact-audit/audit_completed.py',
        ROOT / 'integration/review/ra10-toy-artifact-audit/CHECKER-FROZEN.json',
        ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-lane-review/FROZEN.json',
        ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-lane-review/MONITOR-START-FROZEN.json'))
    paths.add(baseline_audit/'READY.json')
    files = {}
    for name in ('PROTOCOL.md', 'audit_completed.py', 'prepare_checker.py', 'seal_inputs.py', 'close_audit.py'):
        p = HERE / name
        files[str(p)] = sha(p)
        if p.suffix == '.py': compile(p.read_text(), str(p), 'exec')
    for p, h in ready['numerical_source_sha256'].items(): assert sha(p) == h, p
    for p, h in frozen['external_sources'].items(): assert sha(p) == h, p
    for p, h in frozen['local_sources'].items(): assert sha(ROOT/'validation-cb64-ra10'/p) == h, p
    value = dict(status='FROZEN_CPU_ONLY_NATIVE_ARTIFACT_CHECKER', utc=datetime.now(timezone.utc).isoformat(),
        source_and_input_sha256={str(p):sha(p) for p in sorted(paths)}, helper_sha256=files,
        package_sha256=ready['package_sha256'], config_sha256=ready['config_sha256'],
        ready_sha256=sha(ready_path), source_freeze_sha256=sha(frozen_path),
        backend_schema=9, trainer_schema=5, Torch_imported=False, PT_objects_loaded=0,
        model_forwards=0, training_updates=0, quality_verdict=None)
    (HERE / 'CHECKER-FROZEN.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'], guards=len(paths), helper_sha256=sha(HERE/'audit_completed.py'),
        freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'))))


if __name__ == '__main__': main()
