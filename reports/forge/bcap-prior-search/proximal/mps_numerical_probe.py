"""Fixed-tensor CUDA software probe; no model, optimizer, data RNG or training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_multithreading_enabled(False)

def fixture(rows, cols, modulus, scale):
    values = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    values = ((values * 37 + values // cols * 19) % modulus).to(torch.float32)
    return ((values - modulus / 2) / scale).cuda()

def digest(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()

def compute():
    x = fixture(1024, 128, 257, 128).requires_grad_()
    w = fixture(128, 64, 263, 160).requires_grad_()
    y = torch.tanh(x @ w)
    gx, gw = torch.autograd.grad(y.square().mean(), (x, w), create_graph=True)
    second = torch.autograd.grad(gx.square().sum(), w)[0]
    matrix = fixture(128, 64, 269, 128)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    polar = u @ vh
    return dict(gemm_tanh=digest(y), input_gradient=digest(gx),
                parameter_gradient=digest(gw), second_derivative=digest(second),
                singular_values=digest(s), svd_u=digest(u), svd_vh=digest(vh),
                polar_factor=digest(polar))

label = sys.argv[1]
torch.cuda.synchronize()
started = time.perf_counter()
first, second = compute(), compute()
torch.cuda.synchronize()
device = torch.cuda.get_device_properties(0)
result = dict(schema_version=1, label=label, scope="fixed_tensor_software_probe",
              numerical_digests=first, repeated_identically=first == second,
              model_calls=0, optimizer_updates=0, sampling_draws=0,
              elapsed_seconds=time.perf_counter()-started,
              device=device.name, sm_count=device.multi_processor_count,
              compute_capability=f"{device.major}.{device.minor}",
              torch_version=torch.__version__, cuda_version=torch.version.cuda,
              deterministic=torch.are_deterministic_algorithms_enabled(),
              tf32=torch.backends.cuda.matmul.allow_tf32,
              cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              pipe_directory=os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
Path(__file__).with_name(label + ".json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
