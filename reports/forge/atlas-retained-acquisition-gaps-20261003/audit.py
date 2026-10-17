"""Saved-numeric-JSON chronology only; no package, tensor, or scorer imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_ROOT = Path('/ml2/hypergan/forge-atlas-current-gpu-diagnostics-20261003-native-v2')
FORENSIC = Path('/ml2/hypergan/pg-atlas-native-v2-completion-review-20261003/FULL_COMPLETION.json')
COMMIT = '9563dea57bb150f2a0275bbe8d785bf76210fca3'
DIGEST = 'db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037'
PINS = {
    RAW_ROOT / 'study.json': '7594771d9fa8abff4ba66cd4945fb72bff6b1d0faba64e98235ed4f5531f59da',
    FORENSIC: '33f5a05c379f0b92ebe9d3fc995e2ccfc9f532511ac68a5f44e852f5b75f0c54',
}
CASE_PINS = {
    'vector_overlap_policy_selected_cloud_v1': {
        'raw-result.json': '353b58b3dd339a3d85b1df0817ad8d9c587e5349684c9355c569ac542e41c585',
        'resolved.json': 'edc0ecb3aa4da3df0a480dd0a0403ac66bf22128a92d7aa2c8e600aa0305af97',
        'graded-result.json': '82fd068f7f496920820ead7230685d6828f10553a4e789d22e3cac551e7d4fda',
    },
    'img_intensity2_policy_selected_cloud_v1': {
        'raw-result.json': '10de02ea5563b8731483f51f773df5b8782efb1ddad9a40066614b17ea2146bd',
        'resolved.json': '71cc1c99c4bfe8f1a6284a71ca91e51c1ab831daa1b6cb57c60ae2bf015bd40e',
        'graded-result.json': 'f2a765a339a063e67ff78d4644ea23228c8c169a410f578c5717358051c142ae',
    },
}
for case_id, files in CASE_PINS.items():
    PINS.update({RAW_ROOT / 'attempts' / case_id / name: value for name, value in files.items()})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    for path, expected in PINS.items():
        assert path.is_file() and not path.is_symlink(), path
        assert sha(path) == expected, path


def load(path: Path):
    return json.loads(path.read_text())


def finite_number(value):
    assert type(value) in (int, float) and math.isfinite(value), value
    return value


def failed_bounds(observation, thresholds):
    failures = []
    for metric, op, bound in thresholds:
        value = finite_number(observation[metric])
        finite_number(bound)
        assert op in ('>=', '<='), op
        passed = value >= bound if op == '>=' else value <= bound
        if not passed:
            failures.append({'metric': metric, 'op': op, 'bound': bound, 'value': value,
                             'violation': bound - value if op == '>=' else value - bound})
    return failures


def case_record(case_id, study):
    directory = RAW_ROOT / 'attempts' / case_id
    raw = load(directory / 'raw-result.json')
    resolved = load(directory / 'resolved.json')
    grade = load(directory / 'graded-result.json')['grades'][case_id]
    task = resolved['request']['tasks'][case_id]
    source = resolved['request']['source']
    assert source['origin_commit'] == COMMIT and source['digest'] == DIGEST
    assert len(source['files']) == 1366
    evaluation = task['evaluation']
    observations = raw['evidence']['observations']
    horizon = task['execution']['steps']
    count = evaluation['observations']
    required = evaluation['minimum_stable_checks']
    assert count == 24 and required == 5 and horizon % count == 0
    assert [row['step'] for row in observations] == list(range(horizon // count, horizon + 1, horizon // count))
    assert raw['cost']['completed_steps'] == horizon
    assert all(value == horizon for value in raw['cost']['optimizer_updates'].values())
    thresholds = evaluation['thresholds']
    failures = [failed_bounds(row, thresholds) for row in observations]
    passing = [not row for row in failures]
    first = next(row['step'] for row, passed in zip(observations, passing) if passed)
    runs, run = [], []
    for row, passed in zip(observations, passing):
        if passed:
            run.append(row['step'])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    acquisition = next(values[:required] for values in runs if len(values) >= required)
    suffix = runs[-1] if passing[-1] else []
    terminal = observations[-required:]
    final_five_passed = all(passing[-required:])
    convergence = grade['evaluator_result']['convergence']
    assert convergence['first_pass_step'] == first
    assert convergence['confirmed_step'] == acquisition[-1]
    assert convergence['stable_from_step'] == suffix[0]
    assert convergence['passing_observations'] == sum(passing)
    assert convergence['passing_suffix'] == len(suffix)
    assert convergence['observations'] == count and convergence['complete'] is True
    assert grade['status'] == grade['gate_status'] == 'PASS' and final_five_passed
    job = next(job for job in study['jobs'] if case_id in job['task_ids'])
    assert job['status'] == 'COMPLETE'
    policy = raw['evidence']['policy_observation']
    backend = policy['backend_selection']
    source_subset = {name: source['files'][name] for name in (
        'benchmarks/transfer_suite/protocol.py',
        'benchmarks/transfer_suite/image_tasks.py',
        'benchmarks/transfer_suite/vector_tasks.py',
        'experiments/forge/adapters.py',
        'experiments/forge/sampling.py',
        'experiments/forge/views.py',
        'particlegan/policy.py',
        'particlegan/training.py',
    )}
    for name, value in source_subset.items():
        PINS[Path(source['snapshot_path']) / name] = value
    return {
        'task_id': case_id,
        'source': {'commit': COMMIT, 'digest': DIGEST, 'files': len(source['files']),
                   'scientific_source_sha256': source_subset},
        'original_status': grade['status'], 'original_gate_status': grade['gate_status'],
        'completed_steps': horizon, 'observation_count': count,
        'required_terminal_checks': required, 'thresholds': thresholds,
        'sampling_law': evaluation['sampling_law'],
        'output_noise': policy['output_noise'],
        'latent_policy': policy['latent_policy'], 'controller': policy['controller'],
        'sampler': policy['sampler'], 'final_selected_source': policy['selected_source'],
        'final_snapshot_sha256': policy['snapshot_sha256'],
        'backend': backend['actual_backend'], 'sampling_backend': backend['sampling_backend'],
        'prior_population': backend['population_policy']['population'],
        'observation_measurement': evaluation.get('measurement'),
        'first_pass_step': first,
        'first_five_read_window': acquisition,
        'confirmed_step': acquisition[-1],
        'passing_runs': [{'first': values[0], 'last': values[-1], 'checks': len(values)} for values in runs],
        'failures': [{'step': row['step'], 'failed_bounds': bounds}
                     for row, bounds in zip(observations, failures) if bounds],
        'losses_after_first_pass': [row['step'] for row, bounds in zip(observations, failures)
                                   if bounds and row['step'] > first],
        'losses_after_five_read_confirmation': [row['step'] for row, bounds in zip(observations, failures)
                                               if bounds and row['step'] > acquisition[-1]],
        'final_passing_suffix_start': suffix[0], 'final_passing_suffix_checks': len(suffix),
        'passing_observations': sum(passing),
        'final_five_steps': [row['step'] for row in terminal], 'final_five_passed': final_five_passed,
        'final_original_metrics': observations[-1],
        'original_independent_grade_convergence': convergence,
        'supervised_paid_seconds': job['paid_wall_seconds'],
        'ordinary_qualification': False, 'default_adoption': False,
        'transient_classification_rows': [row for row in observations if case_id.startswith('img_') and row['step'] in (350, 375, 400, 425)],
    }


def main():
    verify_inputs()
    study = load(RAW_ROOT / 'study.json')
    assert study['status'] == 'COMPLETE_DIAGNOSTIC'
    records = [case_record(case_id, study) for case_id in CASE_PINS]
    forensic = load(FORENSIC)
    ring_record = next(row for row in forensic['records'] if row['task_id'] == 'ring_hold_policy_selected_cloud_v1')
    assert ring_record['scientific']['first_failure'] == {
        'cover': 1.0, 'effective_modes': 7.345964431762695,
        'hq': 0.89453125, 'modes': 8, 'n_modes': 8, 'step': 1407}
    for role in ('raw', 'resolved', 'grading'):
        pin = ring_record[role]
        PINS[Path(pin['path'])] = pin['sha256']
    assert ring_record['scientific']['confirmation_completed_at'] == 1400
    assert ring_record['scientific']['confirmation_checks'] == 200
    assert ring_record['scientific']['passing_hold_checks'] == 6
    verify_inputs()
    result = {
        'schema': 'pg_atlas_retained_acquisition_gaps_v1',
        'source': {'commit': COMMIT, 'digest': DIGEST},
        'status': 'SAVED_NUMERIC_CHRONOLOGY_MATCHES_ORIGINAL_GRADES',
        'scope': 'Two original transfer curves plus immutable ring precision-failure clarification',
        'records': records,
        'ring': {'task_ids': ['ring_hold_policy_selected_cloud_v1', 'ring_extension_policy_selected_cloud_v1'],
                 'source': ring_record['source'], 'shared_physical_job': True,
                 'source_bound_forensic_report': str(FORENSIC),
                 'confirmation_steps': {'first': 1201, 'last': 1400, 'checks': 200},
                 'passing_disjoint_hold_steps': {'first': 1401, 'last': 1406, 'checks': 6},
                 'first_failure': ring_record['scientific']['first_failure'],
                 'original_min_hq': 0.9, 'failed_bound': 'hq',
                 'interpretation': 'Precision loss with all eight modes retained; full hold not completed; no reacquisition.',
                 'original_status': 'POST_CONVERGENCE_FAIL',
                 'observations': 207, 'ordinary_qualification': False},
        'original_whole_batch': forensic['whole_batch'],
        'limits': ['Passing numeric observations do not prove behavior between the declared checkpoints.',
                   'An early instantaneous PASS is not a five-read acquisition.',
                   'Original task thresholds remain unchanged; no stronger CDF/component/image-TV gate is imported.',
                   'No per-update causal attribution is established by these observations.'],
        'verification': {'consumed_files': len(PINS), 'input_bytes_unchanged': True,
                         'new_models': 0, 'checkpoint_deserializations': 0,
                         'new_draws': 0, 'official_scorer_calls': 0,
                         'training_updates': 0, 'GPU_calls': 0},
        'inputs': [{'path': str(path), 'sha256': value, 'bytes': path.stat().st_size}
                   for path, value in sorted(PINS.items(), key=lambda item: str(item[0]))],
        'reproducer_sha256': sha(Path(__file__)),
    }
    output = HERE / 'gaps.json'
    assert not output.exists(), 'Do not overwrite an earlier receipt'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'], 'records': len(records),
                      'consumed_files': len(PINS), 'output': str(output), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
