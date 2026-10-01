"""Close public-init parity on the actual frozen RA13 candidate; no policy, forwards or updates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CURRENT = ROOT.parents[1] / 'pkg-RA14-replay'
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
ORIGINAL_INPUTS = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/learned/INPUTS.json')
sys.path.insert(0, str(CURRENT))
sys.path.insert(0, str(PREV))
import torch
from models_metrics import networks, model_hash, tensor_hash
from particlegan.init import deterministic_orthogonal_
from particlegan.particle_prior import ParticlePrior


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
expected = json.loads(ORIGINAL_INPUTS.read_text())['expected_initial_hashes']
fixtures = {}
for problem in ('toy', 'mnist'):
    torch.manual_seed(314159)
    G, D = networks(problem)
    prior = ParticlePrior(1024, 128, generator=torch.Generator().manual_seed(314160))
    rng_before = torch.get_rng_state().clone()
    deterministic_orthogonal_(G, seed=0)
    deterministic_orthogonal_(D, seed=1)
    initial = dict(initial_generator_sha256=model_hash(G), initial_critic_sha256=model_hash(D),
                   initial_prior_sha256=tensor_hash(prior.z))
    assert initial == expected[problem] and torch.equal(torch.get_rng_state(), rng_before)
    fixtures[problem] = dict(actual_hashes=initial, expected_hashes=expected[problem],
                             exact_initial_hash_parity=True, public_init_rng_consumed=False)
assert not torch.cuda.is_initialized()
paths = [ORIGINAL_INPUTS, PREV / 'models_metrics.py', CURRENT / 'particlegan/init.py',
         CURRENT / 'particlegan/_qr.py', CURRENT / 'particlegan/particle_prior.py', Path(__file__).resolve()]
receipt = dict(status='PASS_PUBLIC_INIT_EXACT_ORIGINAL_PARITY', current_api_root=str(CURRENT),
               fixtures=fixtures, training_updates=0, policy_constructions=0, model_forwards=0,
               sampling_calls=0, evaluator_calls=0, cuda_context_initialized=False,
               source_file_sha256={str(path): sha(path) for path in paths})
target = ROOT / 'init-parity-candidate.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(status=receipt['status'], fixture_count=len(fixtures), receipt_sha256=sha(target))), flush=True)
