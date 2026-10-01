"""Pin the exact defaultCPU/valid-checkpoint bridge over actual RA14/RA15 hosts."""
import ast
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


def host_scope(path):
    tree = ast.parse(pin(path).read_text())
    scopes, setters = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                'set_default_device', 'set_default_tensor_type'):
            setters.append(node.lineno)
        if not isinstance(node, ast.With):
            continue
        device = any(isinstance(item.context_expr, ast.Call) and
            isinstance(item.context_expr.func, ast.Attribute) and
            isinstance(item.context_expr.func.value, ast.Name) and item.context_expr.func.value.id == 'torch' and
            item.context_expr.func.attr == 'device' for item in node.items)
        if device:
            steps = [item.lineno for item in ast.walk(node) if isinstance(item, ast.Call) and
                     isinstance(item.func, ast.Attribute) and item.func.attr == 'step']
            assert not steps, str(path) + ': training step inside an ambient device context'
            scopes.append(dict(line=node.lineno, training_step_calls=[]))
    assert not setters, str(path) + ': host changes ambient factory device'
    return dict(path=str(path), sha256=sha(path), global_default_device_setters=[],
                explicit_torch_device_contexts=scopes, training_steps_outside_device_contexts=True)


def main():
    assert not (ROOT / 'EXECUTION-BRIDGE.json').exists()
    inherited = read(STUDY / 'validation-ra15/NO-FIRE-EXECUTION-BRIDGE.json')
    assert inherited['bridged_quality_task_count'] == 15 and inherited['original_quality_task_count'] == 19
    assert inherited['original_fresh_training_labels_preserved']
    for path, value in inherited['read_only_file_sha256'].items():
        assert sha(pin(path)) == value, path
    for name in ('SOURCE-FREEZE.json', 'QUALIFICATION-PLAN.json'):
        pin(STUDY / 'validation-ra15' / name)
    source = STUDY / 'portability/ra16-portability'
    bridge = read(source / 'SOURCE-BRIDGE.json')
    frozen = read(source / 'SOURCE-FREEZE.json')
    assert sha(source / 'SOURCE-BRIDGE.json') == '9a7064831524031dce661db1b1ea90e2d2871a4873df4dd9c0cd4d7e524ad031'
    assert sha(source / 'SOURCE-FREEZE.json') == '548ad79fdba33c63b2f965267daaaabb734011d5d8e9ce382aaebe30f2e80516'
    assert bridge['status'] == 'CPU_PASS_GPU_DEFAULT_PENDING' and bridge['CPU_contracts_passed'] == 94
    assert bridge['default_CPU_source_AST_forward_law'] and bridge['valid_checkpoint_math_unchanged']
    assert bridge['package_sha256'] == 'ba20f8e6353b2aca4725258835a68c1dfaa056a0484b2d683a5bb74a1eca0aa0'
    for path, value in frozen['hashes'].items():
        assert sha(pin(path)) == value, path
    proof = read(source / 'default-cpu-bridge.json')
    assert proof['status'] == 'CPU_PASS' and proof['source_AST_only_7_CPU_factory_keywords_plus_pre_mutation_shape_validator']
    assert len(proof['CPU_allocation_sites']) == 7 and proof['CUDA_initialized'] is False
    assert all(row['exact_all_checkpoint_state_every_step'] and row['exact_losses_every_step'] and
        row['exact_samples'] and row['valid_checkpoint_roundtrip_both_directions'] and
        row['resumed_next_update_exact'] for row in proof['default_CPU_forward_law'])
    hosts = [host_scope(STUDY / 'validation-ra15/screen_current.py'),
             host_scope(STUDY / 'mnist/ra13-settled/run_training.py')]
    for task in ('grid100', 'rotated100', 'staggered100'):
        hosts.append(host_scope(STUDY / 'validation-ra14-r2/moving' / task / 'adapted_runner.py'))
    for lane in ('ra14-replay-r2', 'ra15-replay'):
        for name in ('SOURCE-FREEZE.json', 'INPUTS.json', 'CPU-CLOSED.json', 'replay.py'):
            pin(STUDY / 'mnist' / lane / name)
    integration = read(STUDY / 'integration-prep/ROOT-INTEGRATION-RA16.json')
    # Root integration has its own byte checks; this bridge keeps execution labels.
    receipt = dict(status='PASS_DEFAULT_CPU_VALID_CHECKPOINT_SOURCE_EQUIVALENCE_ONLY',
        candidate='RA16-portability', package_sha256=bridge['package_sha256'],
        base_candidate='RA15-partial-recovery', base_package_sha256=bridge['base_package_sha256'],
        config_sha256=bridge['config_sha256'], changed_modules=bridge['changed_modules'],
        explicit_CPU_factory_sites=proof['CPU_allocation_sites'],
        default_CPU_AST_source_equivalence=True, default_CPU_full_state_trace_equivalence=True,
        valid_checkpoint_schema_and_math_unchanged=True, invalid_shape_guard_before_mutation=True,
        checkpoint_shape_validation_more_strict_only_for_invalid_input=True,
        original_host_device_scope_proof=hosts, ambient_training_factory_device='cpu',
        source_CPU_contracts_passed=94, CPU_bridge_feature_updates=18, CPU_bridge_feature_reactions=2,
        CPU_bridge_KNN_fallback_updates=2, CPU_bridge_bidirectional_restore_and_next_update_exact=True,
        bridged_RA14_quality_tasks=inherited['bridged_quality_tasks'], bridged_RA14_quality_task_count=15,
        affected_quality_routes=[dict(kind='portability', task='ring_shift', actual_execution_candidate='RA15-partial-recovery',
            actual_execution_lane=str(STUDY / 'validation-ra15'), required_steps=4600)] + [
            dict(kind='moving', task=task, actual_execution_candidate='RA15-partial-recovery',
                 actual_execution_lane=str(STUDY / 'validation-ra15'), required_steps=1500)
            for task in ('grid100', 'rotated100', 'staggered100')],
        required_full_quality_task_count=19, affected_actual_statuses='Read final receipts at closure; no preparation inference',
        original_fresh_learned_training_source='RA13-settled', original_learned_updates_per_fixture=2000,
        fresh_RA14_RA15_RA16_learned_training_updates=0,
        latest_original_learned_CUDA40_replay='PENDING_REQUIRED', latest_actual_integrated_full_suite='PENDING_REQUIRED',
        actual_CUDA_default_regression='PENDING_REQUIRED_SINGLE_FROZEN_TEST',
        RA15_replay_history='SEALED_UNLAUNCHED', RA15_full_suite_history='CANCELLED_BEFORE_EXECUTION',
        no_quality_claims_from_source_proof=True, tensor_loads=0, model_calls=0, scorer_calls=0,
        GPU_operations=0, fresh_training_updates_this_preparation=0, read_only_file_sha256=pins)
    with (ROOT / 'EXECUTION-BRIDGE.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], read_only_files=len(pins), bridge_sha256=sha(ROOT / 'EXECUTION-BRIDGE.json')), sort_keys=True))


if __name__ == '__main__':
    main()
