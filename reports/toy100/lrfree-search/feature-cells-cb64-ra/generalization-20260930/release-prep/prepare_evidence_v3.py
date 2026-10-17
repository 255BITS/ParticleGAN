"""Bind completed source preparations without inferring pending numerical results."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def files(lane, names):
    return {name: sha(lane / name) for name in names}


def main():
    assert not (HERE / 'FINALIZER-V3-PREPARATION.json').exists()
    lanes = {}
    for name in ('validation-ra14', 'validation-ra14-r2', 'validation-ra14-moving-r2',
                 'validation-ra15', 'validation-ra16'):
        lane = ROOT / name
        frozen = read(lane / 'SOURCE-FREEZE.json')
        package = Path(frozen['package']) / 'particlegan'
        mapping = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}
        manifest = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        lanes[str(lane)] = dict(source_freeze_sha256=sha(lane / 'SOURCE-FREEZE.json'),
            package_sha256=frozen['package_sha256'], package_manifest_sha256=manifest,
            config_sha256=frozen['config_sha256'], frozen_file_count=len(frozen['hashes']))
    candidate = 'RA16-portability'
    package = ROOT / ('pkg-' + candidate)
    mapping = {str(path.relative_to(package / 'particlegan')): sha(path)
               for path in sorted((package / 'particlegan').rglob('*.py'))}
    source = read(ROOT / 'portability/ra16-portability/SOURCE-BRIDGE.json')
    assert source['package_sha256'] == 'ba20f8e6353b2aca4725258835a68c1dfaa056a0484b2d683a5bb74a1eca0aa0'
    replay = ROOT / 'mnist/ra16-replay'
    replay_freeze = read(replay / 'SOURCE-FREEZE.json')
    replay_names = list(replay_freeze['local_source_sha256']) + ['SOURCE-FREEZE.json', 'CPU-CLOSED.json']
    GPU = ROOT / 'diagnostics/ra16-portable-gpu'
    GPU_freeze = read(GPU / 'SOURCE-FREEZE.json')
    old_replay = ROOT / 'mnist/ra15-replay'
    old_freeze = read(old_replay / 'SOURCE-FREEZE.json')
    preparation = dict(status='SEALED_SOURCE_ONLY_NUMERICAL_COMPLETION_REQUIRED', latest_candidate=candidate,
        qualification_base=dict(reference_PR=155, candidate_draft_PR=223,
            original_reference_commit='f459cb6d6aaaabeb1af076ec53ad7a963618de90',
            observed_upstream_commit='cabe2084284db923d525918cbf3e18de6f20faac',
            observed_upstream_commit_source='Root fetched and independently reviewed PR155 advancement',
            observed_upstream_source_qualification='PENDING_SEPARATE_BRIDGE_OR_FRESH_EXECUTIONS_AND_SUITE',
            repository_HEAD_before_RA16_source_commit='c05ea9f8fe48d5d0e499bc4e6801170b452886f8',
            tested_package_source_identity='Byte-pinned RA16 source based on the original f459 reference'),
        latest_package=dict(path=str(package), package_sha256=source['package_sha256'], source_sha256=mapping),
        affected_quality_lane=str(ROOT / 'validation-ra15'), latest_suite_lane=str(ROOT / 'validation-ra16'),
        lanes=lanes, bridges={
            'no_fire': dict(path=str(ROOT / 'validation-ra15/NO-FIRE-EXECUTION-BRIDGE.json'),
                sha256=sha(ROOT / 'validation-ra15/NO-FIRE-EXECUTION-BRIDGE.json'),
                required_fields=dict(status='PASS_SOURCE_EQUIVALENCE_ONLY_FRESH_AFFECTED_GATES_PENDING',
                    candidate='RA15-partial-recovery', bridged_quality_task_count=15,
                    original_quality_task_count=19, original_fresh_training_labels_preserved=True)),
            'portability': dict(path=str(ROOT / 'validation-ra16/EXECUTION-BRIDGE.json'),
                sha256=sha(ROOT / 'validation-ra16/EXECUTION-BRIDGE.json'),
                required_fields=dict(status='PASS_DEFAULT_CPU_VALID_CHECKPOINT_SOURCE_EQUIVALENCE_ONLY',
                    candidate=candidate, package_sha256=source['package_sha256'],
                    default_CPU_AST_source_equivalence=True, default_CPU_full_state_trace_equivalence=True,
                    valid_checkpoint_schema_and_math_unchanged=True, bridged_RA14_quality_task_count=15,
                    required_full_quality_task_count=19, no_quality_claims_from_source_proof=True))},
        latest_replay=dict(lane=str(replay), prepared_file_sha256=files(replay, replay_names),
            required_closed_status='PASS_LATEST_RA16_ORIGINAL_CUDA40_REPLAY'),
        portable_GPU_regression=dict(lane=str(GPU), prepared_file_sha256=files(GPU,
            list(GPU_freeze['local_source_sha256']) + ['SOURCE-FREEZE.json'])),
        retained_RA15_unlaunched_replay_file_sha256=files(old_replay,
            list(old_freeze['local_source_sha256']) + ['SOURCE-FREEZE.json', 'CPU-CLOSED.json']),
        helper_file_sha256=files(HERE, ['finalize_evidence_v3.py', 'FINALIZER-V3-PROTOCOL.md', 'prepare_evidence_v3.py',
            'FINALIZER-V3-CPU-PREFLIGHT.json']),
        quality_gates_or_criteria_changed=False, numerical_labels_preserved=True,
        latest_suite_counts='Read actual pytest summary only after completion',
        latest_replay_status='Read actual closed40 update receipt only after completion',
        GPU_operations=0, tensor_loads=0, model_calls=0, scorer_calls=0, repository_mutations=0)
    with (HERE / 'FINALIZER-V3-PREPARATION.json').open('x') as stream:
        stream.write(json.dumps(preparation, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=preparation['status'], preparation_sha256=sha(HERE / 'FINALIZER-V3-PREPARATION.json'),
        helper_sha256=sha(HERE / 'finalize_evidence_v3.py'), latest_candidate=candidate), sort_keys=True))


if __name__ == '__main__':
    main()
