"""Verify both original learned init/stream/pending contracts without execution."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
import sys
import common
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json

inputs = common.verify_inputs()
name = common.VARIANTS[0]
common.select_package(inputs, name)
sys.path.insert(0, str(common.PREV))
import torch
from models_metrics import networks, model_hash, tensor_hash
from particlegan.recipes import Recipe
from particlegan.particle_prior import ParticlePrior
from particlegan.training import GANTrainer
from adapter import initialize_fixture_models
from contracts import verify_fixture_sources, validate_checkpoint_state, validate_initial_streams

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_initialized()
sources = verify_fixture_sources()
rows = {}
for problem in common.PROBLEMS:
    torch.manual_seed(common.SEED)
    config = dict(inputs['variants'][name]['config'], z_dim=128, num_particles=1024, batch_size=128)
    assert 'initialization' not in config
    recipe = Recipe(**config)
    G, D = networks(problem)
    prior = ParticlePrior(1024, 128, generator=torch.Generator().manual_seed(common.SEED + 1))
    before = torch.get_rng_state().clone()
    initialize_fixture_models(G, D)
    assert torch.equal(torch.get_rng_state(), before)
    initial = dict(initial_generator_sha256=model_hash(G), initial_critic_sha256=model_hash(D),
                   initial_prior_sha256=tensor_hash(prior.z))
    assert initial == inputs['expected_initial_hashes'][problem]
    trainer = GANTrainer(recipe, G, D, prior=prior, seed=common.SEED, serial_backward=True)
    assert initial == dict(initial_generator_sha256=model_hash(G), initial_critic_sha256=model_hash(D),
                           initial_prior_sha256=tensor_hash(prior.z))
    streams = validate_initial_streams(torch, trainer)
    state = trainer.state_dict()
    selection = validate_checkpoint_state(state, trainer.policy.roles, problem, recipe)
    assert trainer.completed_steps == 0 and selection['actual_backend'] == 'pending'
    assert not torch.cuda.is_initialized()
    rows[problem] = dict(initial_hashes=initial, initial_lrs=state['initial_lrs'], roles=trainer.policy.roles,
                         streams=streams, backend_selection=selection, public_init_rng_consumed=False)
receipt = dict(status='PASS', device='cpu', model_forwards=0, training_updates=0,
               sampling_calls=0, evaluator_calls=0, begin_step_calls=0,
               cuda_context_initialized=False, original_checkpoint_zero_selection='pending',
               source_contract=sources, fixtures=rows,
               source_freeze_sha256=common.sha(common.ROOT / 'SOURCE-FREEZE.json'),
               inputs_sha256=common.sha(common.ROOT / 'INPUTS.json'))
common.write_json(common.ROOT / 'preflight-cpu.json', receipt)
print(json.dumps(receipt), flush=True)
