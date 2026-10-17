"""Archive this experiment's compact evidence without copying raw tensors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/lrfree-search/feature-cells-cb64-ra/fixes')
TEXT_SUFFIXES = {'.py', '.json', '.md', '.patch', '.csv', '.jsonl'}
SKIP_COMPONENTS = {'__pycache__', 'logs', 'monitor-baseline-check',
                   'learned-audit-baseline-check', 'ra3-artifact-audit-state-review'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_complete():
    """Require completed evidence, including valid quality failures."""
    lane = ROOT / 'validation-ra4'
    plan = json.loads((lane / 'jobs.json').read_text())
    jobs = json.loads((lane / 'execution-results.json').read_text())
    assert len(plan) == len(jobs) == 19, 'original job queue is incomplete'
    for planned, actual in zip(plan, jobs):
        assert planned['name'] == actual['name']
        assert actual['returncode'] == 0 and actual['status'] != 'ERROR'
        assert actual['result_sha256'] == sha(Path(actual['result']))
    monitor = ROOT / 'integration/review/ra4-indexed-api-monitor'
    accepted = json.loads((monitor / 'summary.json').read_text())
    assert accepted['completed'] == accepted['total'] == 16
    assert accepted['status'] in ('PASS', 'FAIL')
    assert accepted['source_integrity']['status'] == 'VALID'
    for record in accepted['records']:
        assert record['canonical_fixture_validity'] == 'VALID'
        assert record['acceptance_status'] == record['primary_status'] in ('PASS', 'FAIL')
        assert record['indexed_api_expectation_adapter']['original_strict_status'] == 'ERROR'
        for base, name in ((monitor, 'acceptance-receipt.json'),
                           (monitor, 'strict-plain-acceptance-receipt.json'),
                           (ROOT / 'integration/review/ra4-validation-monitor', 'acceptance-receipt.json')):
            assert (base / 'canonical-receipts/screens/runs' / record['task'] / name).exists()
    assert (monitor / 'READ-ONLY-ARTIFACT-MANIFEST.json').exists()
    final = ROOT / 'performance/training-regression/count-review/final-ra4-audit'
    assert json.loads((final / 'receipt.json').read_text())['status'] in ('VALID', 'PASS')
    assert (final / 'FROZEN.json').exists()
    assert json.loads((ROOT / 'leaderboard.json').read_text())['status'] == 'COMPLETE'


def verify_ra11_study_complete():
    """Close the RA11 study, retaining broader quality failures explicitly."""
    lane = ROOT / 'validation-cb64-ra11'
    plan = json.loads((lane / 'jobs.json').read_text())
    jobs = json.loads((lane / 'execution-results.json').read_text())
    assert len(plan) == len(jobs) == 19, 'RA11 original job queue is incomplete'
    for planned, actual in zip(plan, jobs):
        assert planned['name'] == actual['name']
        assert actual['returncode'] == 0 and actual['status'] != 'ERROR'
        assert actual['result_sha256'] == sha(Path(actual['result']))
    quality = json.loads((ROOT / 'quality/results/CB64-RA11.json').read_text())
    assert quality['toy_gate'] == quality['grid_gate'] == 'PASS'
    assert quality['canonical_fixture_validity'] == quality['artifact_validity'] == 'VALID'
    replay = json.loads((lane / 'learned/replay-CB64-RA11.json').read_text())
    assert set(replay) == {'toy', 'mnist'}
    assert all(record['status'] == 'PASS' for record in replay.values())
    monitor = ROOT / 'integration/review/validation-cb64-ra11-monitor'
    accepted = json.loads((monitor / 'summary.json').read_text())
    assert accepted['completed'] == accepted['total'] == 16
    assert accepted['source_integrity']['status'] == 'VALID'
    for record in accepted['records']:
        assert record['canonical_fixture_validity'] == 'VALID'
        assert record['acceptance_status'] == record['primary_status'] in ('PASS', 'FAIL')
    assert (monitor / 'READ-ONLY-ARTIFACT-MANIFEST.json').exists()
    final = json.loads((ROOT / 'quality/results/CB64-RA11-regressions.json').read_text())
    assert final['validation_complete'] is True and final['evidence_validity'] == 'VALID'
    assert final['package_sha256'] == quality['package_sha256']
    assert final['config_sha256'] == quality['config_sha256']
    assert final['general_base_package_recommended'] is False
    audit = ROOT / 'quality/ra11/final-regression-review'
    assert final['final_audit_receipt_sha256'] == sha(audit / 'receipt.json')
    assert final['final_audit_proof_sha256'] == sha(audit / 'FINAL-FROZEN.json')
    assert json.loads((audit / 'receipt.json').read_text())['status'] in ('VALID', 'PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', required=True, choices=('QUALITY_IN_PROGRESS', 'COMPLETE', 'RA11_STUDY_COMPLETE'))
    args = parser.parse_args()
    if args.status == 'COMPLETE':
        verify_complete()
    elif args.status == 'RA11_STUDY_COMPLETE':
        verify_ra11_study_complete()
    copied, omitted = [], []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(ROOT)
        parts = relative.parts
        core_package = parts[0].startswith('pkg-CB64-RA')
        private_package = any(part.startswith('pkg-') for part in parts) and not core_package
        private_source = private_package and path.name in (
            'feature_cells.py', 'training.py', 'continuous.py', 'row_evidence.py',
            'anchor_birth.py', 'birth_phase.py', 'particle_prior.py', 'mean_transport.py',
            'output_moments.py')
        failed_log = path.suffix == '.log' and (
            'failed' in path.name.lower() or any(part.startswith('failed-') for part in parts))
        compact = path.suffix in TEXT_SUFFIXES or failed_log or path.name in ('cpu-lineage', 'gpu-lineage')
        allowed = (compact and path.stat().st_size <= 1_000_000
                   and not any(part in SKIP_COMPONENTS for part in parts)
                   and not (private_package and not private_source)
                   and 'trace' not in path.name.lower()
                   and path.suffix != '.jsonl')
        # Preserve the audit summarized by the primary RA3 report.
        if parts[:2] == ('integration', 'review') and 'ra3-artifact-audit-state-review' in parts:
            allowed = path.name in ('summary.json', 'AUDITOR-IDENTITY.json', 'REPORT.md')
        if not allowed:
            if not private_package and path.stat().st_size:
                omitted.append(dict(path=str(path), bytes=path.stat().st_size,
                                    reason='raw, large, redundant or execution log; retained locally'))
            continue
        target = ARCHIVE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == sha(path)
        copied.append(dict(source=str(path), archive=str(relative),
                           sha256=sha(path), bytes=path.stat().st_size))
    manifest = dict(status=args.status, copied_utc=datetime.now(timezone.utc).isoformat(),
                    completion_scope=('RA11 original toy/grid, 19-job queue and final validity audit; broader quality failures retained, no general package promotion.'
                        if args.status == 'RA11_STUDY_COMPLETE' else 'Original RA4 completion gate or intermediate experiment archive.'),
                    source_root=str(ROOT), files=copied,
                    omitted_files=omitted,
                    policy='Frozen sources and compact evidence, including strict/adapted canonical receipts and failed diagnostics; datasets, checkpoints and raw traces remain at recorded local paths.')
    (ARCHIVE/'SOURCE-ARCHIVE.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status=args.status, files=len(copied),
                         bytes=sum(row['bytes'] for row in copied), archive=str(ARCHIVE))))


if __name__ == '__main__':
    main()
