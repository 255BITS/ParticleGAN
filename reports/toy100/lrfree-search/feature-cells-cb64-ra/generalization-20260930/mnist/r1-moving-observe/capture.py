from copy import deepcopy
import hashlib
import struct
import math
import torch

def digest(value):
    """Typed, delimited state fingerprint; tensor shape/dtype/device and bits count."""
    h=hashlib.sha256()
    def token(x):
        b=x if isinstance(x,bytes) else str(x).encode()
        h.update(str(len(b)).encode()+b':'+b)
    def add(x):
        if isinstance(x,torch.Tensor):
            token('tensor');token(tuple(x.shape));token(x.dtype);token(x.device)
            token(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            token('dict');token(len(x))
            for k in sorted(x,key=lambda v:(type(v).__name__,repr(v))):add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            token(type(x).__name__);token(len(x))
            for v in x:add(v)
        elif isinstance(x,float):token('float64');token(struct.pack('!d',x))
        else:token(type(x).__name__);token(repr(x))
    add(value)
    return h.hexdigest()

def semantic_state(state):
    """Only this birth/death observational duration is removed."""
    state=deepcopy(state)
    state.get('birth_death',{}).get('last',{}).pop('eval_seconds',None)
    return state

def plain(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        flat = value.reshape(-1)
        return {'shape': list(value.shape), 'numel': value.numel(), 'first_64': flat[:64].tolist()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return {'length': len(value), 'first_64': [plain(v) for v in value[:64]]}
        return [plain(v) for v in value]
    return value

def rng_digest(trainer):
    values = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(0)}
    for name in trainer._STREAMS:
        values[name] = getattr(trainer, name).get_state()
    if trainer.birth_death is not None:
        values['birth'] = trainer.birth_death.stream.get_state()
    return digest(values)

def context(trainer):
    policy = trainer
    record = trainer.opt_d.record
    sigma = policy.log_output_sigma
    scales = [t.s for i, row in enumerate(policy.lr_settle.testers) if i != 1 for t in row]
    settle = policy.controller.mobility
    if scales and all(s is not None for s in scales):
        settle = 1.0 if any(s > 1. / 64. for s in scales) else settle
    from particlegan.training import output_noise_std
    floor = output_noise_std(trainer.recipe, trainer.completed_steps) * settle
    testers = {}
    for i, row in enumerate(policy.lr_settle.testers):
        for j, tester in enumerate(row):
            if tester is not None:
                testers[f'{i}.{j}'] = plain({key: getattr(tester, key) for key in
                    ('s', 'b', 'tau', 'windows', 'last', 'counts', 'log', 'blocks_in_window')})
    birth = policy.birth_death
    return dict(completed_steps=trainer.completed_steps,
        ka2=record.state_dict(), ka2_pure_a_next_call=record.calls + 1 < 800,
        clipped_tensors=trainer.opt_d.guard.clipped_tensors,
        controller=plain({key: getattr(policy.controller, key) for key in
                         ('mobility', 'data_drive')}),
        roles=[["generator", "table", "noise"], ["critic"]], legacy_roles=policy.roles, lrs=[[group['lr'] for group in opt.param_groups] for opt in (policy.opt_g, policy.opt_d)],
        noise=dict(log_sigma=float(sigma.detach()), unconstrained_sigma=float(sigma.detach().exp()),
                   last_output_sigma=policy.last_output_sigma, floor=floor, settle=settle,
                   grad=None if sigma.grad is None else float(sigma.grad.detach()),
                   adam_state=plain(policy.opt_g.state.get(sigma, {}))),
        testers=testers,
        birth=dict(counters=dict(birth.counters), last=plain(birth.last),
                   moved_rows=plain(birth.moved_rows)))

