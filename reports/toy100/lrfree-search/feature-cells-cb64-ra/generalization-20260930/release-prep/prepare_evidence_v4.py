"""Seal current-source closure inputs after source preparation; no numerical work."""
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


def specification(path):
    return dict(path=str(path), sha256=sha(path))


def main():
    target = HERE / 'FINALIZER-V4-PREPARATION.json'
    assert not target.exists()
    old = read(HERE / 'FINALIZER-V3-R3-PREPARATION.json')
    lane = ROOT / 'validation-ra17'
    source = read(ROOT / 'portability/ra17-current-pr155/SOURCE-BRIDGE.json')
    assert source['package_sha256'] == '500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61'
    assert source['current_PR155_base'] == 'cabe2084284db923d525918cbf3e18de6f20faac'
    frozen = read(lane / 'SOURCE-FREEZE.json')
    package = Path(source['package'])
    mapping = {str(path.relative_to(package / 'particlegan')): sha(path)
               for path in sorted((package / 'particlegan').rglob('*.py'))}
    assert mapping == source['source_sha256']
    manifest = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    lanes = dict(old['lanes'])
    lanes[str(lane)] = dict(source_freeze_sha256=sha(lane / 'SOURCE-FREEZE.json'),
        package_sha256=source['package_sha256'], package_manifest_sha256=manifest,
        config_sha256=source['config_sha256'], frozen_file_count=len(frozen['hashes']))
    replay = ROOT / 'mnist/ra17-replay'
    replay_freeze = read(replay / 'SOURCE-FREEZE.json')
    replay_names = list(replay_freeze['local_source_sha256']) + ['SOURCE-FREEZE.json', 'CPU-CLOSED.json']
    historical = HERE / 'final-v3-r3-attempt1'
    qualification = read(historical / 'QUALIFICATION.json')
    assert qualification['evidence_validity'] == 'VALID' and qualification['quality_qualification'] == 'PASS'
    nodes = [
        ['tests.test_e22_routed_validation', 'test_71_site_mix_has_no_scalar_readbacks_and_finish_has_exactly_one'],
        ['tests.test_e22_routed_readbacks', 'test_cuda_scalar_reads_do_not_grow_with_routing_site_count'],
        ['tests.test_e22_routed_readbacks', 'test_71_site_lazy_and_eager_observation_match_gradients_resume_and_serving[cuda]'],
        ['tests.test_feature_portability', 'test_cuda_default_preserves_feature_reactions_and_checkpoint_replay']]
    names = ['finalize_evidence_v4.py', 'prepare_evidence_v4.py', 'FINALIZER-V4-PROTOCOL.md',
        'FINALIZER-V3-R3-PREPARATION.json']
    prepared = dict(status='SEALED_CURRENT_SOURCE_ACTUAL_GATES_REQUIRED', latest_candidate='RA17-current-pr155',
        qualification_base=dict(current_PR155_commit=source['current_PR155_base'],
            merged_repository_head_at_source_freeze=source['merged_repository_head'],
            original_reference_commit='f459cb6d6aaaabeb1af076ec53ad7a963618de90',
            historical_RA16_source_commit='b25de08cfbeef9fe8aa06522324e132c3e71ae0f',
            candidate_draft_PR=223, current_source_qualification_requires_actual_latest_gates=True),
        repository='/ml2/hypergan/ParticleGAN-ra11-pr155', lanes=lanes,
        latest_suite_lane=str(lane), latest_package=dict(path=str(package),
            package_sha256=source['package_sha256'], source_sha256=mapping),
        current_source_bridge=specification(ROOT / 'portability/ra17-current-pr155/SOURCE-BRIDGE.json'),
        execution_bridge=specification(lane / 'EXECUTION-BRIDGE.json'),
        noise_floor_proof=specification(ROOT / 'diagnostics/upstream-noise-floor-applicability/receipt.json'),
        historical_qualification=dict(lane=str(historical), closed=specification(historical / 'FROZEN.json')),
        latest_replay=dict(lane=str(replay), prepared_file_sha256={name: sha(replay / name) for name in replay_names},
            required_closed_status='PASS_LATEST_RA17_ORIGINAL_CUDA40_REPLAY'),
        required_CUDA_nodes=nodes, CI_smoke=dict(receipt=specification(lane / 'CI-SMOKE.json'),
            report=specification(lane / 'ci-smoke-cpu.json')),
        helper_file_sha256={name: sha(HERE / name) for name in names},
        historical_labels_preserved=True, original_task_set_or_quality_criteria_changed=False,
        no_universal_noise_floor_training_parity_claim=True,
        tensor_loads=0, model_calls=0, GPU_operations=0, repository_mutations=0)
    with target.open('x') as stream:
        stream.write(json.dumps(prepared, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=prepared['status'], preparation_sha256=sha(target),
        helper_sha256=sha(HERE / 'finalize_evidence_v4.py'), current_PR155_commit=source['current_PR155_base']), sort_keys=True))


if __name__ == '__main__':
    main()
