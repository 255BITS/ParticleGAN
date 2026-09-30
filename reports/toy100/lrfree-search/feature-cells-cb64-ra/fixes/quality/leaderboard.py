"""Summarize final saved CUDA gates; never select an earlier checkpoint."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = ROOT.parent/'feature-cells-cuda-retest-20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def toy_gate(metrics):
    return (metrics['precision'] >= .90 and metrics['coverage'] == 25
        and metrics['mass_tv'] <= .10 and len(metrics['supported_mass']) == 25
        and min(metrics['supported_mass']) >= .01)


def main():
    paths = [*BASE.glob('learned/training/toy/*/result.json'),
        *ROOT.glob('validation*/learned/training/toy/*/result.json')]
    rows = []
    for path in sorted(paths):
        result = read(path)
        if result.get('status') != 'COMPLETE':
            continue
        metrics = result['final']['metrics']
        lane = path.parents[4]
        grid_path = lane/'screens/runs/grid100/result.json'
        grid = read(grid_path) if grid_path.exists() else None
        variant = result['variant']
        if grid is not None and grid.get('cand') != variant:
            grid = None
        if lane == ROOT/'validation-ra4':
            receipt = ROOT/'integration/review/ra4-indexed-api-monitor/canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
        else:
            receipt = ROOT/'integration/review'/f'{lane.name}-monitor/canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
        acceptance = read(receipt) if receipt.exists() else None
        passed = toy_gate(metrics)
        grid_pass = bool(grid is not None and grid.get('status') == 'PASS'
            and grid.get('native',{}).get('status') == 'PASS')
        grid_gate = 'PENDING' if grid is None else ('PASS' if grid_pass else 'FAIL')
        if grid is None and not passed and lane.name.startswith('validation-cb64-'):
            grid_gate = 'NOT_RUN_TOY_FAILED'
        row = dict(variant=variant,lane=str(lane),device=result['device'],steps=result['steps'],
            toy_precision=metrics['precision'],toy_modes=metrics['coverage'],toy_mass_tv=metrics['mass_tv'],
            toy_min_supported_mode_mass=min(metrics['supported_mass']),toy_gate='PASS' if passed else 'FAIL',
            grid_gate=grid_gate,
            grid_fixture_validity=None if acceptance is None else acceptance.get('canonical_fixture_validity'),
            both_raw_quality_gates_pass=passed and grid_pass,
            toy_training_seconds=result['training_seconds'],toy_result_sha256=sha(path),
            toy_result=str(path),grid_result=str(grid_path) if grid is not None else None,
            grid_result_sha256=sha(grid_path) if grid is not None else None,
            grid_holdout=None if grid is None else grid.get('native',{}).get('holdout'),
            grid_holdout_pass=None if grid is None else grid.get('native',{}).get('holdout_pass'),
            grid_final_center_sigma=None if grid is None else grid.get('final',{}).get('acc_center_rms_sigma'),
            grid_final_max_cov_eig=None if grid is None else grid.get('final',{}).get('max_cov_eig_ratio'),
            status_scope='final saved CUDA metrics; source/fixture acceptance recorded separately')
        mnist_path = lane/'learned/training/mnist'/variant/'result.json'
        mnist = read(mnist_path) if mnist_path.exists() else None
        embedding = {} if mnist is None else mnist.get('final',{}).get('metrics',{}).get('active_embedding',{})
        row['mnist_active_frechet'] = embedding.get('embedding_frechet')
        row['mnist_recall'] = embedding.get('embedding_recall')
        row['mnist_result'] = str(mnist_path) if mnist is not None else None
        row['mnist_result_sha256'] = sha(mnist_path) if mnist is not None else None
        replay_path = lane/'learned'/f'replay-{variant}.json'
        replay = read(replay_path) if replay_path.exists() else None
        row['cuda_replay'] = ('PENDING' if replay is None else
            'PASS' if all(r.get('status')=='PASS' for r in replay.values()) else 'FAIL')
        rows.append(row)
    for path in sorted(ROOT.glob('validation*/learned/training/toy/*/error.json')):
        error=read(path)
        rows.append(dict(variant=error['variant'],lane=str(path.parents[4]),device=error['device'],
            steps=error['completed_steps'],toy_precision=None,toy_modes=None,toy_mass_tv=None,
            toy_gate='ERROR',grid_gate='NOT_RUN_RUNTIME_ERROR',grid_fixture_validity=None,both_raw_quality_gates_pass=False,
            error=error['error'],error_phase=error['phase'],error_receipt=str(path),error_receipt_sha256=sha(path)))
    recorded = {(row['lane'], row['variant']) for row in rows}
    for directory in sorted(ROOT.glob('validation-cb64-*/learned/training/toy/*')):
        if not directory.is_dir():
            continue
        lane, variant = directory.parents[3], directory.name
        if (str(lane), variant) in recorded:
            continue
        rows.append(dict(variant=variant, lane=str(lane), device=None, steps=None,
            toy_precision=None, toy_modes=None, toy_mass_tv=None, toy_gate='PENDING',
            grid_gate='PENDING', grid_fixture_validity=None, both_raw_quality_gates_pass=False,
            status_scope='prospective run in progress; no final metrics or quality verdict'))
    rows.sort(key=lambda a:(a['both_raw_quality_gates_pass'],a['toy_gate']=='PASS',
        -1. if a['toy_precision'] is None else a['toy_precision'],
        a.get('grid_holdout_pass') is True),reverse=True)
    leaders = [row['variant'] for row in rows if row['both_raw_quality_gates_pass']]
    recommendation = ('No recommendation for this target until one candidate passes both unchanged gates and required validity/replay checks.'
        if not leaders else f"{', '.join(leaders)} leads the joint quality target. A package recommendation awaits required replay and portability acceptance.")
    qualification_path = ROOT/'quality/results/CB64-RA11-regressions.json'
    qualification = read(qualification_path) if qualification_path.exists() else None
    if qualification is not None and qualification['validation_complete']:
        recommendation = ('CB64-RA11 is the validated toy/grid winner and passes all three native tests plus exact CUDA replay. '
            'MNIST and five portability regressions prevent a general base-package recommendation.')
    qualification_summary = None if qualification is None else {
        key:qualification[key] for key in ('status','validation_complete','evidence_validity',
            'native_gates','canonical_counts','cuda_replay','mnist_active_embedding',
            'general_base_package_recommended','original_jobs_completed','canonical_screens_completed')}
    value = dict(updated_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        recommendation=recommendation,
        completed_target_validation=qualification_summary,
        completed_target_validation_receipt=None if qualification is None else str(qualification_path),
        completed_target_validation_receipt_sha256=None if qualification is None else sha(qualification_path),
        no_earlier_checkpoint_selection=True,no_new_seed_runs=True)
    (HERE/'leaderboard.json').write_text(json.dumps(value,indent=2)+'\n')
    lines = ['# Toy and grid quality target','',value['recommendation'],'',
        '| Candidate | Toy P | Modes | TV | Toy | Grid | Grid validity | Final center σ | Final max covariance | Holdout |',
        '|---|---:|---:|---:|---|---|---|---:|---:|---|']
    for row in rows:
        precision='—' if row['toy_precision'] is None else f"{row['toy_precision']:.6f}"
        coverage='—' if row['toy_modes'] is None else f"{row['toy_modes']}/25"
        tv='—' if row['toy_mass_tv'] is None else f"{row['toy_mass_tv']:.6f}"
        center='—' if row.get('grid_final_center_sigma') is None else f"{row['grid_final_center_sigma']:.6f}"
        covariance='—' if row.get('grid_final_max_cov_eig') is None else f"{row['grid_final_max_cov_eig']:.6f}"
        holdout='—' if row.get('grid_holdout_pass') is None else ('PASS' if row['grid_holdout_pass'] else 'FAIL')
        lines.append(f"| {row['variant']} | {precision} | {coverage} | {tv} | {row['toy_gate']} | {row['grid_gate']} | {row['grid_fixture_validity'] or 'pending'} | {center} | {covariance} | {holdout} |")
    lines += ['', 'Complete toy entries use the final update 2000; runtime errors have no quality verdict. Grid requires all original terminal observations and the independent holdout.',
        'Equal toy results are ordered by the original independent holdout pass. Final grid metrics are diagnostic: Grid still requires all five terminal checks and the holdout. Center limit is0.20σ; maximum covariance ratio limit is1.7.',
        'A completed training process is separate from passing the quality gate. CPU mechanism tests do not establish GPU quality.', '',
        '## Broader learned-model results', '',
        '| Candidate | MNIST active feature distance ↓ | MNIST recall | CUDA replay |',
        '|---|---:|---:|---|']
    for row in rows:
        distance='—' if row.get('mnist_active_frechet') is None else f"{row['mnist_active_frechet']:.6f}"
        recall='—' if row.get('mnist_recall') is None else f"{row['mnist_recall']:.2%}"
        lines.append(f"| {row['variant']} | {distance} | {recall} | {row.get('cuda_replay','PENDING')} |")
    lines += ['', 'MNIST uses the same frozen 2000-update comparison. Replay PASS measures reproducible continuation, separately from sample quality.',
        'RA11 is the joint toy/grid leader but its MNIST result regresses severely. It is not a general base-package recommendation.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(rows=len(rows),both_gate_candidates=[r['variant'] for r in rows if r['both_raw_quality_gates_pass']]),indent=2))


if __name__ == '__main__':
    main()
