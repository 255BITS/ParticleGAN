"""Saved original checkpoint/schema/alias audit; no models, sampling or GPU."""
import common
from copy import deepcopy
import json
import os
from pathlib import Path

import torch
from contracts import validate_checkpoint_state

assert not torch.cuda.is_initialized()
def reject_cuda(*args, **kwargs):
    raise AssertionError('NoCUDA during learned saved-state preflight')
torch.cuda._lazy_init = reject_cuda
torch.set_num_threads(1)
before = torch.get_rng_state().clone()
inputs = common.verify_inputs()
common.select_package(inputs, 'RA15-partial-recovery')
from particlegan.recipes import Recipe
rows = {}
for problem in common.PROBLEMS:
    origin = inputs['restoration_bridge']['fixtures'][problem]
    path = Path(origin['checkpoint_path'])
    assert common.sha(path) == origin['checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['trainer']
    assert state['completed_steps'] == 1000 and state['device'] == 'cuda:0'
    assert state['surprise']['fires'] == 0 and type(state['surprise']['fires']) is int
    assert checkpoint['data_position'] == 2 * 1000 * 128
    assert checkpoint['receipt_sha256'] == origin['run_config_sha256']
    assert state['serial_backward'] is True
    table = state['lr_settle'][0][1]
    assert table['last_block'] is table['blocks'][-1]
    assert table['last_block'].untyped_storage().data_ptr() == table['blocks'][-1].untyped_storage().data_ptr()
    recipe = Recipe(**dict(inputs['variants']['RA15-partial-recovery']['config'],
        num_particles=1024, z_dim=128, batch_size=128))
    assert state['recipe'] == recipe.to_dict()
    selected = validate_checkpoint_state(state, state['policy']['roles'], problem, recipe)
    rng = common.rng_cpu_buffers(torch, state)
    rows[problem] = dict(status='PASS', checkpoint_sha256=common.sha(path), completed_steps=1000,
        endpoint_source='RA13-settled', data_position=checkpoint['data_position'], R1_fires=0,
        selected_backend=selected['actual_backend'], source_recipe_schema_matches=True,
        table_lastblock_blocks_alias_identical=True, rng_buffers=rng)
assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
receipt = dict(status='PASS_SAVED_CHECKPOINT_SOURCE_ALIAS_PREFLIGHT', fixtures=rows,
    checkpoint_loads=2, model_calls=0, sampling_calls=0, training_updates=0, begin_step_calls=0,
    cuda_context_initialized=False, CPU_rng_unchanged=True,
    source_freeze_sha256=common.sha(common.ROOT / 'SOURCE-FREEZE.json'),
    inputs_sha256=common.sha(common.ROOT / 'INPUTS.json'),
    limitations=['No model construction or live CUDA restoration occurred; original40update CUDA replay remains required.'])
with (common.ROOT / 'CPU-CLOSED.json').open('x') as out:
    out.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(status=receipt['status'], fixtures=2, checkpoint_loads=2,
    model_calls=0, cuda_initialized=False)), flush=True)
