"""Necessary particle-update invariants; no new benchmark or seed experiments."""
import importlib.util
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import torch

started = time.perf_counter()
root = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = 'cuda:0'


def run_case(module, rotation=None):
    positions = torch.tensor([[-1., .2], [.4, -.5], [1.2, .8], [2., -1.]], device=device)
    directions = torch.tensor([[.1, -.3], [.4, .2], [-.2, .8], [.6, -.1]], device=device)
    if rotation is not None:
        positions = positions @ rotation
        directions = directions @ rotation
    table = torch.nn.Parameter(positions.clone())
    network = torch.nn.Parameter(torch.tensor([.3, -.4], device=device))
    reference = torch.nn.Parameter(network.detach().clone())
    prior = SimpleNamespace(z=table, num_particles=len(table))
    update = module.ParticleUpdate()
    update.register(prior)
    optimizer = torch.optim.Adam([{'params': [network]}, {'params': [table], 'lr': .0085}],
                                 lr=.00425, betas=(0., .999), foreach=False, fused=False)
    reference_optimizer = torch.optim.Adam([reference], lr=.00425, betas=(0., .999), foreach=False, fused=False)
    indices = torch.tensor([0, 1, 1, 2], device=device)
    for iteration in range(3):
        optimizer.zero_grad()
        latent = table[indices]
        if hasattr(update, 'sampled'):
            update.sampled(prior, latent, indices)
        (latent * directions * (iteration + 1)).sum().backward()
        network.grad = torch.tensor([.12, -.07], device=device) * (iteration + 1)
        reference.grad = network.grad.clone()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        update.step(optimizer, torch.optim.Adam.step)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        reference_optimizer.step()
        assert torch.equal(network, reference), 'network Adam was changed'
        assert torch.equal(table[3], positions[3]), 'unused row moved without a gradient'
        assert torch.isfinite(table).all()
        for state in optimizer.state.values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    assert state[key].device.type == 'cuda'
    assert update.receipt()['updates'] == [3]
    return table.detach()


checks = []
error = None
try:
    for code in sorted((root / 'candidates').iterdir()):
        spec = importlib.util.spec_from_file_location(code.name, code / 'particle_update.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = run_case(module)
        checks.append(dict(candidate=code.name, check='network_identity_rng_unused_rows_cuda_finite_counts', status='PASS'))
        if code.name in ('shared_geometry', 'visit_isotropic'):
            rotation = torch.tensor([[1., -1.], [1., 1.]], device=device) / 2 ** .5
            rotated = run_case(module, rotation)
            torch.testing.assert_close(rotated, output @ rotation, rtol=1e-5, atol=1e-6)
            checks.append(dict(candidate=code.name, check='rotation_equivariance', status='PASS'))
except Exception:
    error = traceback.format_exc()
status = 'FAIL' if error else 'PASS'
result = dict(status=status, cases=len(checks), checks=checks,
              seconds=time.perf_counter()-started, error=error)
(root / 'mechanism-checks.json').write_text(json.dumps(result, indent=2) + '\n')
with (root.parents[3] / 'tests.jsonl').open('a') as stream:
    stream.write(json.dumps(dict(candidate='regression', gate='particle_mechanism_invariants',
        status=status, seconds=result['seconds'], metrics=dict(cases=len(checks)),
        artifact=str(root / 'mechanism-checks.json'), error=error)) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'checks'}), flush=True)
raise SystemExit(1 if error else 0)
