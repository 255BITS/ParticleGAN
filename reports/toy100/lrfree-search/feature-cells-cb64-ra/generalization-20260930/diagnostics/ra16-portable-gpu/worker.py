"""Owned worker for exactly one immutable portability test; no quality scoring."""
import json
import os
from pathlib import Path
import sys
import time

from launch import GPU_UUID, LOCK, parked_owner, verify, write

attempt, descriptor = Path(sys.argv[1]).resolve(), int(sys.argv[2])
assert attempt.parent == Path(__file__).resolve().parent
actual, expected = os.fstat(descriptor), LOCK.stat()
assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'missing inherited shared lock'
inputs, before = verify()
parked_owner()
sys.path.insert(0, inputs['package_root'])
import torch
import pytest
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(.2, 0)
props = torch.cuda.get_device_properties(0)
assert 'GPU-' + str(props.uuid).removeprefix('GPU-').lower() == GPU_UUID
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.reset_peak_memory_stats(0)
import particlegan
assert Path(particlegan.__file__).resolve().is_relative_to(Path(inputs['package_root']).resolve())


class CountResults:
    def __init__(self):
        self.tests = []
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.tests.append(dict(node=report.nodeid, outcome=report.outcome))


count = CountResults()
start = time.time()
code = pytest.main(['-q', '-ra', inputs['test_node']], plugins=[count])
_, after = verify()
parked_owner()
passed = code == 0 and len(count.tests) == 1 and count.tests[0]['outcome'] == 'passed'
write(attempt / 'TEST-RESULT.json', dict(status='PASS' if passed else 'FAIL', returncode=int(code),
    actual_tests=count.tests, test_node=inputs['test_node'], source_integrity_before=before,
    source_integrity_after=after, numerical_wall_seconds=time.time()-start,
    package_sha256=inputs['package_sha256'], resources=dict(device='cuda:0', physical_gpu=0,
        gpu_uuid=GPU_UUID, memory_fraction=.2, GPU_name=props.name),
    original_seed=1234, compared_default_devices=['cpu', 'cuda'],
    initial_updates_per_default_context=18, next_update_calls_per_default_context=2,
    portable_diagnostic_actual_step_calls=40, required_feature_reactions_minimum_per_context=2,
    exactly_selected_single_regression_test=True, quality_gate_or_threshold_changed=False,
    peak_allocated_GPU_MiB=torch.cuda.max_memory_allocated(0)/2**20,
    peak_reserved_GPU_MiB=torch.cuda.max_memory_reserved(0)/2**20,
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
    signaled_processes=[]))
raise SystemExit(0 if passed else (int(code) or 1))
