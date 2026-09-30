"""Read compact live progress without importing torch or touching a run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from leaderboard import toy_gate

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='CB64-RA9')
    args = parser.parse_args()
    variant = args.variant
    lane = ROOT / f'validation-{variant.lower()}'
    if variant == 'CB64-RA4':
        lane = ROOT / 'validation-ra4'
    checkpoints, progress = [], None
    log = lane / 'logs' / f'learned-toy-{variant}.log'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A writer may not have finished its last line yet.
            if row.get('event') == 'training_progress':
                progress = {key: row[key] for key in (
                    'step', 'training_seconds', 'updates_per_second')}
            elif row.get('event') == 'training_checkpoint':
                metrics = row['metrics']
                checkpoint=dict(step=row['step'],
                    precision=metrics['precision'], modes=metrics['coverage'],
                    mass_tv=metrics['mass_tv'],
                    min_supported_mode_mass=min(metrics['supported_mass']))
                reaction=row.get('diagnostics',{}).get('birth_death',{})
                average=reaction.get('paired_average')
                if average:
                    checkpoint['paired_average']=dict(
                        coherent_rows=average['coherent_rows'],required=average['required'],
                        eligible=average['eligible'],
                        age_real_rows=reaction.get('paired_average_age_real_rows'))
                checkpoints.append(checkpoint)
    directory = lane / 'learned' / 'training' / 'toy' / variant
    result, error = read(directory / 'result.json'), read(directory / 'error.json')
    final = bool(result and result.get('status') == 'COMPLETE'
        and result.get('steps') == 2000 and result['final']['step'] == 2000)
    gate = ('PASS' if toy_gate(result['final']['metrics']) else 'FAIL') if final else 'PENDING'
    grid = read(lane / 'screens' / 'runs' / 'grid100' / 'result.json')
    if grid and grid.get('cand') != variant:
        grid = None
    grid_gate = ('PASS' if grid.get('status') == 'PASS'
        and grid.get('native', {}).get('status') == 'PASS' else 'FAIL') if grid else 'PENDING'
    if grid is None and final and gate == 'FAIL' and lane.name.startswith('validation-cb64-'):
        grid_gate = 'NOT_RUN_TOY_FAILED'
    elif grid is None and error is not None:
        grid_gate = 'NOT_RUN_RUNTIME_ERROR'
    grid_progress=None
    metrics_path=lane/'screens/runs/grid100/metrics.jsonl'
    if metrics_path.exists():
        observations=[]
        for line in metrics_path.read_text().splitlines():
            try:
                observations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if observations:
            last=observations[-1]
            grid_progress={key:last.get(key) for key in (
                'step','seconds','precision','modes','mass_tv',
                'acc_center_rms_sigma','acc_abs_cov_trace_bias','acc_radial_ks','acc_passed')}
            grid_progress['observations']=len(observations)
            grid_progress['terminal_observations']=sum(
                row['step'] in (6000,6250,6500,6750,7000) for row in observations)
    print(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        variant=variant, lane=str(lane), progress=progress, checkpoints=checkpoints,
        final_toy_gate=gate, grid_gate=grid_gate,grid_progress=grid_progress,
        runtime_error=None if error is None else error.get('error'),
        both_raw_quality_gates_pass=gate == grid_gate == 'PASS',
        scope='Read-only progress; intermediate checkpoints are not final verdicts. Source, fixture and replay acceptance are separate.'), indent=2))


if __name__ == '__main__':
    main()
