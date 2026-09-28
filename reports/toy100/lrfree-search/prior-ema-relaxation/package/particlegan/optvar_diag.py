"""Matched, opt-in Adam denominator audit for the native100 replay."""
import json, os
from pathlib import Path
import torch

def capture(trainer):
    root = os.environ.get('PARTICLEGAN_ADAM_DIAG_DIR')
    if not root:
        return
    step = trainer.completed_steps
    if step not in (5000, 5250, 5500, 5750):
        return
    p = trainer.prior.z
    opt = trainer.opt_g
    group = next(g for g in opt.param_groups if any(q is p for q in g['params']))
    state = opt.state[p]
    v = state.get('max_exp_avg_sq', state['exp_avg_sq'])
    beta2 = group['betas'][1]
    denom = (v / (1. - beta2 ** float(state['step']))).sqrt().add(group['eps'])
    g = state['exp_avg']
    update = group['lr'] * g / denom
    row = dict(step=step, lr=group['lr'], beta2=beta2,
               exp_avg_sq_mean=float(state['exp_avg_sq'].mean()),
               exp_avg_sq_rms=float(state['exp_avg_sq'].mean().sqrt()),
               max_exp_avg_sq_mean=float(v.mean()),
               max_exp_avg_sq_rms=float(v.mean().sqrt()),
               denom_mean=float(denom.mean()), denom_rms=float(denom.square().mean().sqrt()),
               normalized_grad_rms=float((g / denom).square().mean().sqrt()),
               lr_normalized_update_rms=float(update.square().mean().sqrt()),
               exp_avg_rms=float(g.square().mean().sqrt()),
               table_rms=float(p.detach().square().mean().sqrt()))
    path = Path(root); path.mkdir(parents=True, exist_ok=True)
    with (path / 'prior_adam.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    if step == 5750:
        raise RuntimeError('intentional stop after matched Adam denominator capture at step 5750')
