"""Seal the immutable source/test recipe for the single actual CUDA-default test."""
import json
from pathlib import Path
from launch import ROOT, sha, verify, write

STUDY = ROOT.parents[1]
SOURCE = STUDY / 'portability/ra16-portability'


def main():
    assert not (ROOT / 'SOURCE-FREEZE.json').exists()
    bridge = json.loads((SOURCE / 'SOURCE-BRIDGE.json').read_text())
    frozen = json.loads((SOURCE / 'SOURCE-FREEZE.json').read_text())
    assert bridge['status'] == 'CPU_PASS_GPU_DEFAULT_PENDING'
    assert bridge['CPU_contracts_passed'] == 94 and bridge['CUDA_initialized'] is False
    assert bridge['default_CPU_source_AST_forward_law'] and bridge['valid_checkpoint_math_unchanged']
    pins = dict(frozen['hashes'])
    pins[str(SOURCE / 'SOURCE-BRIDGE.json')] = sha(SOURCE / 'SOURCE-BRIDGE.json')
    pins[str(SOURCE / 'SOURCE-FREEZE.json')] = sha(SOURCE / 'SOURCE-FREEZE.json')
    for path, value in pins.items():
        assert sha(path) == value, path
    write(ROOT / 'INPUTS.json', dict(status='FROZEN_UNLAUNCHED_ORIGINAL_SINGLE_PORTABILITY_REGRESSION',
        package_root=bridge['package'], package_sha256=bridge['package_sha256'],
        test_node=bridge['prepared_GPU_test_node'], read_only_file_sha256=pins,
        original_seed=1234, selected_test_count=1, actual_CUDA_required=True,
        quality_thresholds_changed=False, scorer_calls_planned=0, root_launch_only=True))
    write(ROOT / 'PREPARATION.json', dict(status='SOURCE_READY_UNLAUNCHED',
        package_sha256=bridge['package_sha256'], guarded_files=len(pins), GPU_operations=0,
        training_updates=0, model_calls=0, command=['/tmp/pr38-default-env/bin/python', '-u', '-B', str(ROOT / 'launch.py')],
        log=str(ROOT / 'attempt-1/run.log')))
    local = {path.name: sha(path) for path in sorted(ROOT.iterdir()) if path.is_file() and
             path.suffix in ('.py', '.json', '.md')}
    write(ROOT / 'SOURCE-FREEZE.json', dict(status='FROZEN_UNLAUNCHED', local_source_sha256=local,
        package_sha256=bridge['package_sha256'], read_only_file_count=len(pins)))
    _, result = verify()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
