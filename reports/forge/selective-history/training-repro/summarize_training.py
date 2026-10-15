"""Package already-certified receipts; no trainer import or model evaluation."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('PARTICLEGAN_COMBINATION_ARTIFACT_ROOT', Path(__file__).resolve().parent))
SOURCE = '39fad3b03e5178bcdc68cc17ba012ac4083d5887147bef91f0bbeee1205baea6'
CAMPAIGNS = [('', 'initial'), ('repairs', 'repairs'), ('final-critic-screen', 'final-critic')]
ARMS = {
    'recipe-prior-refactor-bcap-pure-v1': 'B',
    'bcap-selective-history-transport-v1': 'T',
    'bcap-selective-history-whole-v1': 'W',
    'bcap-selective-history-adversarial-v1': 'S',
    'bcap-transport-penalty2-v1': 'TP',
    'bcap-selective-history-halfjoint-v1': 'SH',
    'bcap-transport-halfcritic-v1': 'TD',
    'bcap-selective-history-halfjoint-halfcritic-v1': 'SHD',
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaigns, attempts = [], []
    curves = {r['attempt']: r for r in read(ROOT / 'curve-analysis.json')['attempts']}
    for external, packaged in CAMPAIGNS:
        directory = ROOT / external
        complete, audit = read(directory / 'completion.json'), read(directory / 'independent-result-audit.json')
        assert complete['status'] == 'COMPLETE' and complete['source_digest'] == SOURCE
        assert complete['scientific_retries'] == 0 and complete['reserved_seconds'] == 0
        assert audit['independently_recomputed_grades'] and audit['all_full_original_updates']
        assert complete['spent_seconds'] == audit['actual_execution_seconds']
        campaigns.append({
            'campaign': complete['campaign'], 'status': 'COMPLETE',
            'declared_ceiling_seconds': complete['campaign_cap_seconds'],
            'charged_execution_seconds': complete['spent_seconds'],
            'scientific_attempts': complete['scientific_attempts'], 'scientific_retries': 0,
            'outstanding_reservations_seconds': 0,
            'completion': f'training-evidence/{packaged}/completion.json',
            'completion_sha256': digest(directory / 'completion.json'),
            'independent_audit': f'training-evidence/{packaged}/independent-result-audit.json',
            'independent_audit_sha256': digest(directory / 'independent-result-audit.json'),
        })
        for job in complete['jobs']:
            curve = curves[job['attempt_id']]
            assert curve['completed_full_schedule']
            assert curve['diagnostic_gate'] == job['gate_status']
            attempts.append({
                'arm': ARMS[job['candidate']], 'candidate': job['candidate'],
                'campaign': complete['campaign'], 'task': job['task'],
                'attempt': job['attempt_id'], 'certified_grade': job['gate_status'],
                'scientific_scope': 'research_diagnostic', 'qualification_input': False,
                'completed_full_schedule': True, 'observations': curve['observations'],
                'passing_observations': curve['passing_observations'],
                'terminal_suffix': curve['terminal_suffix'],
                'full_gate_thresholds': curve['declared_thresholds'],
                'final_metrics': job['metrics'], 'receipt_hashes': job['file_sha256'],
                'external_artifact_directory': job['external_artifact_directory'],
            })
    assert len(attempts) == 16
    surviving = [arm for arm in ARMS.values()
                 if sum(r['arm'] == arm and r['certified_grade'] == 'PASS' for r in attempts) == 2]
    assert surviving == ['B']
    ordinary_dir = ROOT / 'ordinary-tier1'
    ordinary_audit = read(ordinary_dir / 'independent-result-audit.json')
    ordinary_manifest = read(ordinary_dir / 'launch-manifest.json')
    assert ordinary_manifest['source_digest'] == SOURCE
    ordinary_rows = []
    for row in ordinary_audit['rows']:
        ordinary_rows.append({k: row.get(k) for k in (
            'task', 'attempt', 'status', 'importance', 'metrics', 'evaluator_result',
            'retry_of', 'terminal_status', 'cost', 'reasons', 'note',
            'provenance_completed_steps', 'executed_updates', 'completed_steps', 'receipt_hashes',
            'independent_frozen_regrade_exact', 'diagnostic_used_for_qualification')})
    ordinary = {
        'campaign': ordinary_audit['campaign'], 'candidate': ordinary_audit['candidate'],
        'status': ordinary_audit['status'],
        'fresh_tier1_qualified': ordinary_audit['fresh_tier1_qualified'],
        'required_passes': ordinary_audit['required_passes'],
        'required_total': ordinary_audit['required_total'],
        'declared_ceiling_seconds': ordinary_audit['declared_campaign_budget_seconds'],
        'certified_execution_seconds_so_far': ordinary_audit['actual_execution_seconds'],
        'diagnostic_clock_separate': True, 'rows': ordinary_rows,
        'attempt_count': ordinary_audit.get('attempt_count', len(ordinary_rows)),
        'task_count': ordinary_audit.get('task_count', 7),
        'superseded_attempts_retained': ordinary_audit.get('superseded_attempts_retained', []),
        'selected_current_attempts': ordinary_audit.get('selected_current_attempts', []),
        'independent_audit': 'training-evidence/ordinary-tier1/independent-result-audit.json',
        'independent_audit_sha256': digest(ordinary_dir / 'independent-result-audit.json'),
        'actual_enqueue_command': ordinary_manifest['commands'][0],
    }
    if (ordinary_dir / 'completion.json').exists():
        completion = read(ordinary_dir / 'completion.json')
        ordinary['completion'] = 'training-evidence/ordinary-tier1/completion.json'
        ordinary['completion_sha256'] = digest(ordinary_dir / 'completion.json')
        ordinary['completion_status'] = completion['status']
        ordinary['charged_execution_seconds'] = completion['spent_seconds']
        ordinary['outstanding_reservations_seconds'] = completion['reserved_seconds']
        ordinary['scientific_retries'] = completion['scientific_retries']
        ordinary['six_required_verdict'] = completion['six_required_verdict']
        ordinary['operational_retries'] = completion['operational_retries']
        ordinary['operational_retry_status'] = completion['operational_retry_status']
    out = {
        'schema': 'selective_history_certified_training_summary_v1',
        'kind': 'read_only_certified_receipt_summary', 'script_sha256': digest(Path(__file__)),
        'new_training_updates': 0, 'new_model_evaluations': 0,
        'scientific_source_digest': SOURCE, 'runtime': ordinary_manifest['runtime'],
        'seed': 0, 'reliability_scope': 'Single prescribed deterministic initialization; no multi-seed reliability claim.',
        'historical_v1_preflight_status': 'BLOCKED',
        'historical_v1_preflight_sha256': 'c90de15ba0f5056f62a5e3a810c5ff3887cc3901431d457c1c3773f92fc2fb45',
        'fresh_training_authority': 'Separately user-authorized v2 screen and bounded registered repair/fallback studies.',
        'diagnostic_campaigns': campaigns, 'diagnostic_attempts': attempts,
        'diagnostic_declared_ceilings_seconds': sum(r['declared_ceiling_seconds'] for r in campaigns),
        'diagnostic_charged_execution_seconds': sum(r['charged_execution_seconds'] for r in campaigns),
        'diagnostic_outstanding_reservations_seconds': 0,
        'transport_history_survivors': [], 'recipe_family_tuning_stopped': True,
        'ordinary_baseline': ordinary,
    }
    if 'charged_execution_seconds' in ordinary:
        out['all_campaign_charged_execution_seconds'] = (
            out['diagnostic_charged_execution_seconds'] + ordinary['charged_execution_seconds'])
        out['all_campaign_declared_ceilings_seconds'] = (
            out['diagnostic_declared_ceilings_seconds'] + ordinary['declared_ceiling_seconds'])
        out['all_attempts_including_diagnostic_clock'] = len(attempts) + ordinary['attempt_count']
        out['all_outstanding_reservations_seconds'] = ordinary['outstanding_reservations_seconds']
    (ROOT / 'training-summary.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({
        'diagnostic_attempts': len(attempts), 'transport_history_survivors': [],
        'diagnostic_execution_seconds': out['diagnostic_charged_execution_seconds'],
        'ordinary_status': ordinary['status'], 'ordinary_required_passes': ordinary['required_passes'],
        'ordinary_fresh_qualified': ordinary['fresh_tier1_qualified'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
