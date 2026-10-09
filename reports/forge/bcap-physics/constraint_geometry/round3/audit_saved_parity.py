"""Compare inactive trained arms from certified saved bytes; no forwards/draws."""
from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.artifacts import verify_artifacts
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import _scored_outputs
from publish import QUEUE, REQUESTS, OUT


def without_delta(value):
    if isinstance(value, dict):
        return {k: without_delta(v) for k, v in value.items()
                if k not in {'constraint_geometry', 'constraint_geometry_mode', 'strict_progress'}}
    if isinstance(value, (list, tuple)):
        return type(value)(without_delta(v) for v in value)
    return value


def numeric_outputs(value):
    """Whole-checkpoint hashes include recipe/optimizer metadata by design."""
    if isinstance(value, dict):
        return {k: numeric_outputs(v) for k, v in value.items()
                if k not in {'training_state_sha256', 'primary_state_sha256', 'confirmed_state_sha256'}}
    if isinstance(value, (tuple, list)):
        return type(value)(numeric_outputs(v) for v in value)
    return value


def compare(a, b, path='$'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return [path]
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [path + '.keys']
        return [p for k in a for p in compare(a[k], b[k], path + '.' + str(k))]
    elif isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return [path + '.length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in compare(x, y, path + f'[{i}]')]
    elif a != b:
        return [path]
    return []


def load(job):
    aid = job['result']['attempt_id']
    durable = ROOT / 'reports/forge/attempts' / aid
    result = read_json(durable / 'result.json')
    row = result['task_results'][0]
    request = read_json(durable / 'request.json')['request']
    certificate = read_json(durable / 'evidence.json')
    assert certificate['result_hash'] == stable_hash(result)
    assert certificate['source'] == request['source']
    descriptor = row['evidence']['provenance_checkpoint']
    local = Path(descriptor['artifact_root'])
    verify_artifacts(local, descriptor['artifact_manifest'])
    path = local / descriptor['path']
    assert file_hash(path) == descriptor['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    samples, _ = _scored_outputs(request['tasks'][row['task_id']], row['evidence'], Path(certificate['local_artifact_root']))
    return {**row, 'attempt_id': aid}, saved, samples


def main():
    torch.set_num_threads(1)
    state = read_json(QUEUE / 'queue/state.json')
    outputs = []
    for task in ('two_pole', 'gaussian1d_smoke', 'gaussian1d_stability'):
        pair = {}
        for role, rid in REQUESTS.items():
            job = next(j for j in state['jobs'].values()
                       if rid in j['subscribers'] and j['definition']['task_id'] == task)
            pair[role] = load(job)
        candidate, control = pair['candidate'], pair['control']
        saved = candidate[1]
        ambient = None
        if 'trainer' in saved:
            optimizers = saved['trainer']['optimizers']
            cohort = {k: saved[k] for k in ('trainer', 'streams', 'initialization', 'prior')}
            base = {k: control[1][k] for k in cohort}
            ambient = dict(cross_arm_mismatches=compare(
                {k: saved['trainer'][k] for k in ('cpu_rng', 'cuda_rng')},
                {k: control[1]['trainer'][k] for k in ('cpu_rng', 'cuda_rng')}),
                reason='Per-worker ambient global states; public trainer uses isolated named model RNG inside fork_rng, then restores ambient state.')
            for role, (row, final, _) in pair.items():
                root = Path(row['evidence']['artifact_root'])
                verify_artifacts(root, row['evidence']['artifact_manifest'])
                initial = torch.load(root / 'initial-state.pt', map_location='cpu', weights_only=False)
                differences = compare({k: initial['trainer'][k] for k in ('cpu_rng', 'cuda_rng')},
                                      {k: final['trainer'][k] for k in ('cpu_rng', 'cuda_rng')})
                ambient[role + '_unchanged_within_run'] = not differences
                assert not differences, (task, role, differences)
            cohort = {**cohort, 'trainer': {k:v for k,v in cohort['trainer'].items() if k not in {'cpu_rng','cuda_rng'}}}
            base = {**base, 'trainer': {k:v for k,v in base['trainer'].items() if k not in {'cpu_rng','cuda_rng'}}}
        else:
            optimizers = saved['optimizers']['generator']
            cohort = {k: saved[k] for k in ('models', 'role_parameters', 'optimizers', 'streams')}
            base = {k: control[1][k] for k in cohort}

        def stats(value):
            if isinstance(value, dict):
                return ([value['constraint_geometry']['stats']] if 'constraint_geometry' in value else []) + [x for v in value.values() for x in stats(v)]
            if isinstance(value, (tuple, list)):
                return [x for v in value for x in stats(v)]
            return []

        counters = stats(optimizers)
        assert counters and all(x['projected_steps'] == 0 for x in counters), (task, counters)
        state_mismatches = compare(without_delta(cohort), without_delta(base))
        observation_mismatches = compare(candidate[0]['evidence']['observations'], control[0]['evidence']['observations'])
        output_mismatches = compare(numeric_outputs(candidate[2]), numeric_outputs(control[2]))
        outputs.append(dict(task_id=task, candidate_attempt_id=candidate[0]['attempt_id'],
                            control_attempt_id=control[0]['attempt_id'],
                            candidate_constraint_stats=counters,
                            state_mismatches=state_mismatches, observation_mismatches=observation_mismatches,
                            scored_output_mismatches=output_mismatches,
                            ambient_global_rng=ambient,
                            bitwise_equal=not (state_mismatches or observation_mismatches or output_mismatches)))
    result=dict(schema_version=1, qualification_input=False, checks=outputs,
                excluded_declared_delta=['constraint_geometry', 'constraint_geometry_mode', 'strict_progress'],
                separated_unused_ambient_state=['trainer.cpu_rng', 'trainer.cuda_rng'],
                excluded_cross_arm_full_checkpoint_hashes=['training_state_sha256', 'primary_state_sha256', 'confirmed_state_sha256'],
                note='Original checkpoint bytes and same-state confirmation hashes remain intact; every consumed named stream and numeric output is compared.',
                optimizer_updates_added=0, sampling_draws_added=0)
    atomic_json(OUT / 'inactive-trained-parity.json', result)
    print(json.dumps(result))
    assert all(x['bitwise_equal'] for x in outputs), 'inactive trained parity failed; inspect saved mismatch paths'


if __name__ == '__main__':
    main()
