"""Standalone proof: CPU computation with a visible CUDA card stays on CPU."""
import argparse
import json
from pathlib import Path
import runpy
import traceback

import pytest
import torch

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("trace", "reject", "native"))
args = parser.parse_args()
root = Path(__file__).resolve().parent
contracts = root.parent / "test_ra12_contracts.py"
attempts = []
print("torch", torch.__version__, flush=True)
print("initial_cuda_initialized", torch.cuda.is_initialized(), flush=True)
assert not torch.cuda.is_initialized()

if args.mode != "native":
    def forbidden_init(*unused_args, **unused_kwargs):
        attempts.append(traceback.format_stack())
        print("FORBIDDEN_CUDA_LAZY_INIT_STACK", "".join(attempts[-1]), flush=True)
        raise RuntimeError("CPU attempted CUDA initialization; blocked before driver init")
    torch.cuda._lazy_init = forbidden_init

if args.mode == "trace":
    torch.set_num_threads(1)
    api = runpy.run_path(str(contracts))
    try:
        api["test_auto_matches_explicit_feature_through_original_reaction"]()
    except RuntimeError:
        assert len(attempts) == 1
    else:
        raise AssertionError("Frozen source did not reproduce the reported CPU CUDA query")
    status = 0
else:
    status = pytest.main(["-q", "-p", "no:cacheprovider", str(contracts),
                          str(root / "test_cpu_optimizer_scope.py")])
    assert not attempts
    api = runpy.run_path(str(contracts))
    trainer = api["make"]()
    streams = trainer.state_dict()["streams"]
    assert all(state.dtype == torch.uint8 and state.device.type == "cpu"
               for state in streams.values())
    print("typed_rng_buffers", json.dumps({name: {"dtype": str(state.dtype),
                                                   "device": str(state.device)}
                                           for name, state in streams.items()}), flush=True)

print("forbidden_cuda_initialization_attempts", len(attempts), flush=True)
print("final_cuda_initialized", torch.cuda.is_initialized(), flush=True)
assert not torch.cuda.is_initialized()
raise SystemExit(status)
