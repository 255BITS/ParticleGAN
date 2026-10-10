"""Read saved checkpoints on CPU; never construct a model or run a scorer."""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'validation-ra14-r2/moving/rotated100'
inputs = {}


def read_bytes(path):
    raw = Path(path).read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def stats(tensor):
    value = tensor.detach().double().reshape(-1)
    finite = value[torch.isfinite(value)]
    result = dict(shape=list(tensor.shape), elements=tensor.numel(), finite=len(finite))
    if len(finite):
        result.update(mean=float(finite.mean()), rms=float(finite.square().mean().sqrt()),
                      minimum=float(finite.min()), maximum=float(finite.max()),
                      quantiles=torch.quantile(finite, torch.tensor([.05, .5, .95], dtype=torch.float64)).tolist())
    return result


def small(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else stats(value)
    if isinstance(value, dict):
        return {str(k): small(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [small(v) for v in value]
    return value


assert not torch.cuda.is_initialized()
cuda_initialization_attempts = []
def forbid_cuda(*args, **kwargs):
    cuda_initialization_attempts.append('attempt')
    raise RuntimeError('This diagnostic permits CPU checkpoint reads only')
torch.cuda._lazy_init = forbid_cuda
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
rows = []
for step in (500, 1000, 1500):
    path = RUN / f'checkpoint-{step:06d}.pt'
    read_bytes(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['completed_steps'] == step
    groups = []
    for i, optimizer in enumerate(state['optimizers']):
        for j, group in enumerate(optimizer['param_groups']):
            tester = state['lr_settle'][i][j]
            moments = []
            for pid in group['params']:
                memory = optimizer['state'].get(pid, {})
                record = dict(parameter_id=pid, step=small(memory.get('step')),
                              state_keys=list(memory))
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key in memory:
                        record[key] = stats(memory[key])
                if 'exp_avg_sq' in memory and 'max_exp_avg_sq' in memory:
                    v, maximum = memory['exp_avg_sq'].double(), memory['max_exp_avg_sq'].double()
                    positive = v > 0
                    record['amsgrad_denominator_inflation_over_current_v'] = stats(
                        (maximum[positive] / v[positive]).sqrt())
                    if 'exp_avg' in memory:
                        time = float(memory['step'])
                        bias = 1. - group['betas'][1] ** time
                        m = memory['exp_avg'].double()
                        record['raw_first_moment_over_amsgrad_denominator'] = stats(
                            m / ((maximum / bias).sqrt() + group['eps']))
                moments.append(record)
            kept = ('s', 'b', 'tau', 'last_decisive', 'last_decisive_scale', 'last', 'counts',
                    'windows', 'log', 'blocks_in_window', 'b_anchor')
            groups.append(dict(optimizer=i, group=j, role=state['policy']['roles'][i][j],
                               calibrated_base_lr=state['initial_lrs'][i][j],
                               stored_applied_lr=group['lr'], betas=list(group['betas']),
                               tester={k: small(tester[k]) for k in kept if k in tester},
                               moment_memory=moments))
    critic = dict(state['optimizers'][1]['regularizer']['record'])
    history = critic.pop('sur_hist')
    critic.update(surprise_history_length=len(history),
                  surprise_history=history,
                  effective_ema_decay=1. - critic['alpha'] * (1. - critic['anchor_min_decay']))
    generator_controls = state['optimizers'][0]['regularizer']
    birth = state['birth_death']
    rows.append(dict(step=step, groups=groups, surprise=small(state['surprise']),
                     reopen_guard=small(state['reopen_guard']), ka2=critic,
                     generator_regularizers=small(generator_controls),
                     controller=small(state['controller']), backend_selection=state['backend_selection'],
                     serving=dict(state['policy'], paired_average=birth['paired_average'],
                                  averaging_rate=min(1., state['lr_settle'][0][1]['s'] /
                                                     (state['recipe']['serve_average'] * state['lr_settle'][0][1]['b']))),
                     feature_lifecycle=dict(snapshot_serial=birth['snapshot_serial'], fill=birth['fill'],
                                            cursor=birth['cursor'], rows_since_eval=birth['rows_since_eval'],
                                            last_reaction_step=birth['last']['step'],
                                            counters=birth['counters'],
                                            mean_transport=birth['last']['mean_transport'],
                                            support_memory={k: stats(birth[k]) for k in ('S', 'W', 'n', 'pending')}),
                     row_evidence=small({k: v for k, v in state['row_evidence'].items()
                                         if k in ('fraction', 'valid', 'counters', 'scale_c')})))
del state
assert not torch.cuda.is_initialized() and not cuda_initialization_attempts
receipt = json.loads(read_bytes(RUN / 'COMPLETION.json'))
for relative in ('validation-ra14-r2/SOURCE-FREEZE.json',
                 'release-prep/FINALIZER-V2-PREPARATION.json',
                 'release-prep/finalize_evidence_v2.py',
                 'pkg-RA14-replay/particlegan/policy.py',
                 'pkg-RA14-replay/particlegan/continuous.py',
                 'pkg-RA14-replay/particlegan/feature_cells.py',
                 'pkg-RA14-replay/particlegan/feature_reference.py',
                 'pkg-RA14-replay/particlegan/feature_policy.py',
                 'pkg-RA14-replay/particlegan/ka2.py',
                 'pkg-RA14-replay/particlegan/k3p.py',
                 'pkg-RA14-replay/particlegan/mean_transport.py',
                 'pkg-RA14-replay/particlegan/output_moments.py'):
    read_bytes(ROOT / relative)
read_bytes(RUN / 'adapted_runner.py')
report = dict(status='CPU_SAVED_STATE_INSPECTION_COMPLETE', checkpoints=rows,
              original_quality_status=receipt['quality_status'], original_verdict=receipt['verdict'],
              checkpoint_loads_this_script=3, model_calls=0, scorer_calls=0,
              training_updates=0, gpu_operations=0, cuda_initialization_attempts=0,
              input_sha256=inputs,
              limitations=['Checkpoint histories retain only the last8 settling decisions and last8 R1 fires.',
                           'Stored first moments are raw gradients after latent damping restores the parent beta1=0 state; normalized-memory diagnostics are not a replay of the applied table update.',
                           'The external CUDA real-data cursor is not stored in these trainer checkpoints. Any continuation must advance the original seed1234 CUDA generator by two real batches per completed update.'])
with (HERE / 'CONTROLLER-STATE.json').open('x') as handle:
    handle.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(dict(status=report['status'], output=str(HERE / 'CONTROLLER-STATE.json'),
                      checkpoint_loads=3, model_calls=0, gpu_operations=0), sort_keys=True))
