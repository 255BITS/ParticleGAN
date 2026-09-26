"""Native toy100 gate (grid100/rotated100/staggered100) for a declared candidate.
Setup follows gpu-known-winner-control/worker.py (resolve_problem_config + train, canonical coverage AND
accuracy suites, native CUDA init). Hooks follow probe.py: mechanism import, CUDA Adam state, latent.begin and
response.begin around exactly one native noncapturable CUDA Adam step."""
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
from unittest.mock import patch
p = argparse.ArgumentParser()
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--task', required=True, choices=['grid100', 'rotated100', 'staggered100'])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sample-stream', choices=('rng', 'sobol', 'r2'), default=None)
a = p.parse_args()
sys.path[:0] = [str(a.repo.resolve()), str(a.candidate.resolve())]
a.output.mkdir(parents=True, exist_ok=False)
import torch
from particlegan.sample_stream import apply as apply_sample_stream
apply_sample_stream(a.sample_stream)
if not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('CUDA with CUBLAS_WORKSPACE_CONFIG=:4096:8 required')
torch.cuda.set_device(0); torch.set_default_device('cuda:0'); torch.set_num_threads(1); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
import mechanism, response, latent
proof = {'adam_calls': 0, 'optimizers': {}}
original_init = torch.optim.Adam.__init__
original_step = torch.optim.Adam.step


def init(opt, *args, **kwargs):
    kwargs.setdefault('foreach', False); kwargs.setdefault('fused', False)
    original_init(opt, *args, **kwargs)


def step(opt, *args, **kwargs):
    values = [v for g in opt.param_groups for v in g['params']]
    assert all(v.device.type == 'cuda' for v in values) and all(v.grad is None or v.grad.device == v.device for v in values)
    item = proof['optimizers'].setdefault(str(id(opt)), dict(calls=0, parameters=sum(v.numel() for v in values)))
    item['calls'] += 1; proof['adam_calls'] += 1
    for group in opt.param_groups:
        for v in group['params']:
            if v.grad is not None and not opt.state[v]:
                opt.state[v]['step'] = torch.zeros((), dtype=torch.float32, device=v.device)
                opt.state[v]['exp_avg'] = torch.zeros_like(v, memory_format=torch.preserve_format)
                opt.state[v]['exp_avg_sq'] = torch.zeros_like(v, memory_format=torch.preserve_format)
    latent_saved = latent.begin(opt)
    response_saved = response.begin(opt)
    result = original_step(opt, *args, **kwargs)
    response.end(response_saved)
    latent.end(latent_saved)
    if item['calls'] == 1 or item['calls'] % 1000 == 0:
        item['state_devices'] = sorted({str(t.device) for s in opt.state.values() for t in s.values() if isinstance(t, torch.Tensor)})
        item['state_dtypes'] = sorted({str(t.dtype) for s in opt.state.values() for t in s.values() if isinstance(t, torch.Tensor)})
    if proof['adam_calls'] % 2000 == 0:
        print(json.dumps(dict(event='UPDATES', task=a.task, adam_calls=proof['adam_calls'], latent_scoped=latent.receipt['scoped_calls'])), flush=True)
    return result


started = time.perf_counter()
record = dict(candidate=str(a.candidate), task=a.task, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              candidate_hashes={n: hashlib.sha256((a.candidate / n).read_bytes()).hexdigest() for n in ('config.json', 'mechanism.py', 'response.py', 'latent.py')},
              environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS')},
              torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
try:
    from benchmarks.toy100.train import train
    from benchmarks.toy100.config import resolve_problem_config
    from benchmarks.toy100.gate import evaluate_suite
    from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
    config = json.loads((a.candidate / 'config.json').read_text())
    cfg = resolve_problem_config(config, a.task, device='cuda:0')
    with patch.object(torch.optim.Adam, '__init__', init), patch.object(torch.optim.Adam, 'step', step):
        summary = train(cfg, a.output / 'native' / a.task)
    coverage = evaluate_suite(a.output / 'native', problem=a.task)
    accuracy = accuracy_suite(a.output / 'native', problem=a.task)
    record.update(status='PASS' if coverage['status'] == 'PASS' and accuracy['status'] == 'PASS' else 'FAIL',
                  summary=summary, coverage=coverage, accuracy=accuracy, steps=cfg['steps'])
except Exception:
    record.update(status='ERROR', error=traceback.format_exc())
record.update(seconds=time.perf_counter() - started, proof=proof, latent_receipt=dict(calls=latent.receipt['calls'], scoped_calls=latent.receipt['scoped_calls'], tables=latent.receipt['tables'], rows=latent.receipt['rows'][:40]),
              response_calls=response.receipt['calls'], regularizer_calls=mechanism.receipt['calls'])
(a.output / 'result.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
print(json.dumps({k: record.get(k) for k in ('task', 'status', 'seconds', 'error')}), flush=True)
