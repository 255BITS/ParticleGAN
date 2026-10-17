"""Bind the original completed quality records to closed current-PR155 source proof."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
pins = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    pins[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(pin(path).read_text())


def main():
    assert not (ROOT / 'EXECUTION-BRIDGE.json').exists()
    source = STUDY / 'portability/ra17-current-pr155'
    bridge = read(source / 'SOURCE-BRIDGE.json')
    frozen = read(source / 'SOURCE-FREEZE.json')
    assert sha(source / 'SOURCE-BRIDGE.json') == 'bd692c3d83353cfdcd7fb6c7ef08c4efa9b5467b0cd5d2f7887bbdd728ed8d8f'
    assert sha(source / 'SOURCE-FREEZE.json') == '4fd0ff0ea5ffa0c9f60a98f44f1ad8737e8a6a1d957f216db5142aff750c507d'
    assert bridge['status'] == 'CPU_PASS_LATEST_BASE_GPU_GATES_PENDING'
    assert bridge['package_sha256'] == '500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61'
    assert bridge['CPU_contracts_passed'] == 104 and not bridge['CUDA_initialized']
    for field in ('original_19_quality_and_2_learned_noise_math_preserved',
                  'lazy_diagnostics_training_outputs_gradients_RNG_math_preserved',
                  'valid_many_site_routing_codes_usage_gradient_math_preserved',
                  'public_DV12_diagnostics_and_checkpoint_key_preserved', 'original_40_replay_ranges_covered'):
        assert bridge[field], field
    assert bridge['fresh_training_required_for_original_noise_floor_fixtures'] == []
    assert bridge['minimum_original_all_step_table_s_bound'] == .25
    for path, value in frozen['hashes'].items():
        assert sha(pin(path)) == value, path
    for path, value in bridge['original_noise_floor_proof_checkpoint_hashes'].items():
        assert sha(pin(path)) == value, path
    historical = STUDY / 'release-prep/final-v3-r3-attempt1'
    closed = read(historical / 'FROZEN.json')
    for name, value in closed['file_sha256'].items():
        assert sha(pin(historical / name)) == value, name
    qualification = read(historical / 'QUALIFICATION.json')
    assert qualification['evidence_validity'] == 'VALID' and qualification['quality_qualification'] == 'PASS'
    assert len(qualification['gates']) == 19 and all(row['quality_status'] == 'PASS' for row in qualification['gates'])
    older = read(STUDY / 'validation-ra16/EXECUTION-BRIDGE.json')
    for path, value in older['read_only_file_sha256'].items():
        assert sha(pin(path)) == value, path
    for name in ('SOURCE-FREEZE.json', 'FULL-TESTS.json', 'full-pytest.log'):
        pin(STUDY / 'validation-ra16' / name)
    for name in ('SOURCE-FREEZE.json', 'INPUTS.json', 'CPU-CLOSED.json', 'CLOSED.json'):
        pin(STUDY / 'mnist/ra16-replay' / name)
    for relative in ('tests/test_e22_routed_validation.py', 'tests/test_e22_routed_readbacks.py',
                     'tests/test_feature_portability.py'):
        pin(Path('/ml2/hypergan/ParticleGAN-ra11-pr155') / relative)
    receipt = dict(status='PASS_ORIGINAL_FINITE_HORIZON_CURRENT_PR155_SOURCE_EQUIVALENCE_ONLY',
        candidate='RA17-current-pr155', package_sha256=bridge['package_sha256'], config_sha256=bridge['config_sha256'],
        current_PR155_reference=bridge['current_PR155_base'], merged_repository_head=bridge['merged_repository_head'],
        previous_closed_candidate='RA16-portability', previous_closed_qualification=str(historical),
        previous_closed_qualification_sha256=sha(historical / 'FROZEN.json'),
        original_19_quality_math_preserved=True, original_2_learned_training_math_preserved=True,
        original_40_checkpoint_continuation_ranges_covered=True, minimum_all_step_table_s_bound=.25,
        universal_training_parity_claimed=False, all_actual_execution_labels_preserved=True,
        bridged_quality_gates=qualification['gates'], bridged_quality_task_count=19,
        fresh_RA17_original_quality_training_required=False, fresh_RA17_learned_training_required=False,
        actual_latest_RA17_original_CUDA40_replay='PENDING_REQUIRED', actual_latest_RA17_full_suite='PENDING_REQUIRED',
        required_actual_CUDA_node_capture=True, required_exact_upstream_CI_CPU_CLI_smoke=True,
        CUDA_accelerator_predicate_tests='Actual four required nodes captured by full-suite JUnit; no extra duplicate GPU job',
        read_only_file_sha256=pins, GPU_operations=0, model_calls=0, scorer_calls=0, tensor_loads=0,
        original_gate_thresholds_or_budgets_changed=False, source_only_proof_not_new_training=True)
    with (ROOT / 'EXECUTION-BRIDGE.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], bridge_sha256=sha(ROOT / 'EXECUTION-BRIDGE.json'),
        read_only_files=len(pins)), sort_keys=True))


if __name__ == '__main__':
    main()
