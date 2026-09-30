"""Isolate bounded sampler cost on frozen saved tensors, without training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--package-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / 'result.json').exists():
    raise SystemExit('Result exists; refusing overwrite.')
sys.path.insert(0, str(args.package_root))
import torch
from particlegan.feature_cells import BoundedLatentGeometry, bounded_jitter

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'geometry/gpu-inputs.pt'
EXPECTED = '07bfa15a806659459bd4f1bb634cf58969bff7a9f25ec69924e1fe8ba22fb263'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(INPUT) == EXPECTED
sources = {str(p): sha(p) for p in sorted((args.package_root/'particlegan').rglob('*.py'))}
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(.2, 0)
uuid = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower()
assert uuid == 'GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
inputs = torch.load(INPUT, map_location='cpu', weights_only=False)['cases']
rows = []

for name in ('native_cpu_saved_density', 'mnist_E22_0'):
    case = next(c for c in inputs if c['name'] == name)
    points, query, width, noise = [case[k].to('cuda:0') for k in ('prior','query','bandwidth','noise')]
    prior = SimpleNamespace(z=points)
    sizes = (2048, 20000) if points.shape[1] == 2 else (128, 1024)
    for count in sizes:
        indices = torch.arange(count, device=points.device) % len(query)
        q, draw = query[indices], noise[indices]
        kernel = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
        methods = {'fixed_historical': lambda: bounded_jitter(q, draw),
                   'bounded_warm': lambda: kernel.displacement(q, prior, width, draw)}

        def cold():
            # Increment the table version without changing any numerical input.
            points.copy_(points)
            return kernel.displacement(q, prior, width, draw)

        methods['bounded_cold'] = cold
        record = dict(name=name, n=len(points), dimension=points.shape[1], query_rows=count,
                      query_policy='tile existing fixed256 rows/noise; no new RNG draws', timings={})
        for label, method in methods.items():
            method()
            elapsed = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                delta = method()
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
            record['timings'][label] = dict(milliseconds=elapsed,
                median_ms=sorted(elapsed)[1], displacement_rms=float(delta.square().mean().sqrt()))
        kernel.displacement(q, prior, width, draw)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            kernel.displacement(q, prior, width, draw)
            torch.cuda.synchronize()
        events = []
        for event in profile.key_averages():
            events.append(dict(name=event.key, calls=event.count,
                               cpu_us=event.self_cpu_time_total,
                               device_us=event.self_device_time_total))
        record['profile'] = sorted(events, key=lambda r:r['cpu_us'], reverse=True)
        record['work'] = dict(kernel.work)
        rows.append(record)
        print(json.dumps({k:record[k] for k in ('name','n','dimension','query_rows','timings')}), flush=True)

assert sources == {p:sha(p) for p in sources}
result = dict(status='PASS', scope='fixed-input sampler microprofile; no training/quality/scaling-law verdict',
              input_sha256=EXPECTED, package_sources=sources, script_sha256=sha(__file__),
              gpu_uuid=uuid, deterministic=True, tf32=False, cases=rows,
              peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
(args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(status='PASS', output=str(args.output))), flush=True)
