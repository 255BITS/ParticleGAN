"""Prepare the original two-fixture2x10 CUDA replay without launching it."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1]
BASE = ROOT.parent / 'ra14-replay-r2'
NAME = 'RA16-portability'
PACKAGE = STUDY / 'pkg-RA16-portability'
CONFIG = STUDY / 'configs/RA16-portability.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    assert not (ROOT / 'INPUTS.json').exists()
    old = json.loads((BASE / 'INPUTS.json').read_text())
    quality_bridge_path = STUDY / 'validation-ra16/EXECUTION-BRIDGE.json'
    quality_bridge = json.loads(quality_bridge_path.read_text())
    assert quality_bridge['bridged_RA14_quality_task_count'] == 15
    closure = STUDY / 'portability/ra16-portability/SOURCE-BRIDGE.json'
    source = json.loads(closure.read_text())
    assert source['status'] == 'CPU_PASS_GPU_DEFAULT_PENDING' and source['default_CPU_source_AST_forward_law'] and source['valid_checkpoint_math_unchanged']
    from common import package_digest, package_sources
    assert package_digest(PACKAGE) == source['package_sha256'] == 'ba20f8e6353b2aca4725258835a68c1dfaa056a0484b2d683a5bb74a1eca0aa0'
    assert CONFIG.read_bytes() == (STUDY / 'configs/RA14-replay.json').read_bytes()
    GPU_lane = STUDY / 'diagnostics/ra16-portable-gpu/attempt-1'
    GPU_completion = json.loads((GPU_lane / 'COMPLETION.json').read_text())
    GPU_result = json.loads((GPU_lane / 'TEST-RESULT.json').read_text())
    assert GPU_completion['status'] == GPU_result['status'] == 'PASS'
    assert GPU_result['package_sha256'] == source['package_sha256']
    assert GPU_result['portable_diagnostic_actual_step_calls'] == 40
    selected = dict(package_root=str(PACKAGE), package_sha256=package_digest(PACKAGE),
        source_sha256=package_sources(PACKAGE), config_path=str(CONFIG), config_sha256=sha(CONFIG),
        config=json.loads(CONFIG.read_text()), source_variant=NAME,
        intervention='Seven explicit CPU factory devices preserve the original defaultCPU law; valid checkpoint shape validation preserves saved math. Chained learned path fires0.')
    bridge = dict(old['restoration_bridge'])
    bridge.update(candidate_variant=NAME, additional_training_updates=0,
        fresh_numerical_scope='LatestRA16 original native andCPU-map CUDA continuation only',
        numerical_results_inherited_by_source_proof=False,
        proof=dict(status='PASS_CHAINED_NO_FIRE_RESTORATION_COMPATIBILITY',
            candidate_package_sha256=selected['package_sha256'],
            origin_package_sha256=old['restoration_bridge']['proof']['origin_package_sha256'],
            config_bytes_identical=True, config_sha256=sha(CONFIG), checkpoint_schema_unchanged=True,
            RA14_restoration_helper_bytes_unchanged=True, typed_no_fire_learned_endpoints=True,
            RA15_partial_recovery_no_fire_path_explicitly_bridged=True,
            RA16_default_CPU_factory_math_and_valid_checkpoint_shape_explicitly_bridged=True,
            actual_RA16_CUDA_default_single_portability_regression_passed=True,
            explicit_no_fire_execution_bridge=str(quality_bridge_path),
            explicit_no_fire_execution_bridge_sha256=sha(quality_bridge_path),
            latest_original_native_CPU_map_replay_required=True,
            original_training_label='RA13 fresh2000updates per fixture; no fresh RA14/RA15/RA16 learned training'))
    read_only = dict(old['read_only_file_sha256'])
    read_only.update(quality_bridge['read_only_file_sha256'])
    extras = [BASE / name for name in ('INPUTS.json', 'SOURCE-FREEZE.json', 'CLOSED.json', 'replay.py',
        'adapter.py', 'contracts.py', 'common.py', 'bridge.py', 'RESTORATION-BRIDGE.json', 'CPU-CLOSED.json')]
    extras += [closure, quality_bridge_path, STUDY / 'validation-ra16/SOURCE-FREEZE.json', CONFIG,
        STUDY / 'portability/ra16-portability/SOURCE-FREEZE.json']
    extras += [STUDY / 'mnist/ra15-replay' / name for name in ('INPUTS.json', 'SOURCE-FREEZE.json', 'CPU-CLOSED.json')]
    extras += [STUDY / 'diagnostics/ra16-portable-gpu/attempt-1' / name for name in ('COMPLETION.json', 'TEST-RESULT.json', 'run.log')]
    for path in extras:
        read_only[str(path)] = sha(path)
    for path, expected in read_only.items():
        assert sha(path) == expected, path
    write(ROOT / 'RESTORATION-BRIDGE.json', bridge)
    read_only[str(ROOT / 'RESTORATION-BRIDGE.json')] = sha(ROOT / 'RESTORATION-BRIDGE.json')
    inputs = dict(old, variants={NAME: selected}, read_only_file_sha256=read_only, restoration_bridge=bridge,
        candidate_review_receipt=dict(path=str(closure), sha256=sha(closure)),
        execution_policy='Rootlaunch only; original40updates under existing shared serialGPUlock and strict parkedPIDidentity checks.',
        fresh_training_planned=False, fresh_replay_updates_required=40)
    write(ROOT / 'INPUTS.json', inputs)
    write(ROOT / 'preparation-receipt.json', dict(status='SOURCE_PREPARED_UNLAUNCHED_CPU_CHECK_PENDING',
        package_sha256=selected['package_sha256'], config_sha256=selected['config_sha256'],
        replay_runner_byte_identical_to_closed_RA14=True, update_scorer_stream_seed_changed=False,
        fresh_training_planned=False, original_learned_training_source='RA13-settled',
        required_fresh_replay_updates=40, gpu_operations=0, training_updates=0,
        command=['/tmp/pr38-default-env/bin/python', '-u', '-B', str(ROOT / 'launch_replay.py')]))
    files = sorted([path for path in ROOT.iterdir() if path.suffix in ('.py', '.md', '.json')])
    write(ROOT / 'SOURCE-FREEZE.json', dict(status='FROZEN_REPLAY_ONLY_NOT_LAUNCHED',
        package_sha256=selected['package_sha256'], config_sha256=selected['config_sha256'],
        local_source_sha256={path.name: sha(path) for path in files}))
    print(json.dumps(dict(status='FROZEN_NOT_LAUNCHED', source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'),
        inputs_sha256=sha(ROOT / 'INPUTS.json'), read_only_files=len(read_only)), sort_keys=True))


if __name__ == '__main__':
    main()
