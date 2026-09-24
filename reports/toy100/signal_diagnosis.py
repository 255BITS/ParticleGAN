"""Estimate repeated-batch G-gradient coherence at one passing fixed state.

This is an attribution diagnostic, not a candidate or a quality gate. A nonzero
mean gradient is not proof of a learnable discrepancy: the finite host can have
irreducible distribution mismatch. Shifted-target rows before D adapts are also
not a test of whether the critic can learn the new target.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from benchmarks.locked_shared.mode_hold import sample_ring, BATCH, SIGMA
from benchmarks.toy100.continuous_probe import run_probe
from benchmarks.toy100.warm_equilibrium_probe import training_state_sha256
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


class DiagnosticComplete(Exception):
    pass


def summarize(values):
    gradients = torch.stack(values)
    mean_square = float(gradients.mean(0).square().sum())
    noise_trace = float(gradients.var(0, unbiased=True).sum())
    signal = mean_square - noise_trace / len(values)
    energy = float(gradients.square().sum(1).mean())
    return dict(batches=len(values), coordinates=gradients.shape[1],
        mean_square=mean_square, variance_trace=noise_trace,
        mean_square_bias_corrected=signal,
        coherent_energy_fraction=max(0., signal) / energy if energy else 0.)


def diagnose(state, config, batches=32):
    recipe, _, _ = declared_recipe(config)
    gan, cap = recipe.make_loss(), recipe.make_gradient_penalty()
    if recipe.prior_reg != 0:
        raise ValueError('diagnostic requires the declared zero-prior-penalty core')
    g, d, prior = state['generator'], state['critic'], state['prior']
    policy = state['noise_policy']
    before = training_state_sha256(state)
    d_weights, d_opt = deepcopy(d.state_dict()), deepcopy(state['opt_d'].state_dict())
    noise_counts = dict(policy._counts)
    streams = [policy.input_stream] + ([] if policy.output_stream is None else [policy.output_stream])
    stream_states = [stream.get_state().clone() for stream in streams]
    groups = state['opt_g'].param_groups
    result = []
    try:
        for shift, d_updates in ((0., 0), (.35, 0), (0., 32), (.35, 32)):
            d.load_state_dict(d_weights)
            state['opt_d'].load_state_dict(deepcopy(d_opt))
            for stream, saved in zip(streams, stream_states):
                stream.set_state(saved)
            data = torch.Generator().set_state(state['stream'].get_state())
            means = state['means'] + torch.tensor([shift, 0.])
            with torch.random.fork_rng(devices=[]):
                for step in range(d_updates):
                    real = sample_ring(means, BATCH, SIGMA, data)
                    latent, _ = prior.sample(BATCH, generator=data)
                    with policy.discriminator():
                        fake = g(latent).detach()
                    loss = gan.d_loss(d(real), d(fake)) + cap(d, real, fake, step=1001 + step)
                    opt = state['opt_d']
                    opt.zero_grad()
                    loss.backward()
                    for group in opt.param_groups:
                        group['lr'] = .00425
                    state['base_adam_step'](opt)
                values = {role: [] for role in ('g', 'prior')}
                advantages = []
                parameters = [p for group in groups for p in group['params']]
                for _ in range(batches):
                    latent, _ = prior.sample(BATCH, generator=data)
                    fake_logits = d(g(latent))
                    real_logits = d(sample_ring(means, BATCH, SIGMA, data))
                    gradients = iter(torch.autograd.grad(gan.g_loss(fake_logits, real_logits), parameters))
                    advantages.append(float((2 * (real_logits - fake_logits).sigmoid().mean() - 1).detach()))
                    for group in groups:
                        role = 'prior' if group.get('_comparison_prior') else 'g'
                        scaled = []
                        for p in group['params']:
                            moment = state['opt_g'].state[p]
                            denom = (moment['exp_avg_sq'] / (1 - group['betas'][1] ** float(moment['step']))).sqrt() + group['eps']
                            scaled.append((next(gradients).detach().double() / denom.double()).flatten())
                        values[role].append(torch.cat(scaled))
                result.append(dict(shift_x=shift, critic_only_updates=d_updates,
                    g=summarize(values['g']), prior=summarize(values['prior']),
                    mean_advantage=sum(advantages) / len(advantages)))
    finally:
        d.load_state_dict(d_weights)
        state['opt_d'].load_state_dict(deepcopy(d_opt))
        policy._counts = noise_counts
        for stream, saved in zip(streams, stream_states):
            stream.set_state(saved)
    after = training_state_sha256(state)
    if before != after:
        raise RuntimeError('diagnostic failed to restore the captured training state')
    return dict(scope='gradient_signal_diagnostic_only', shared_gate_eligible=False,
                warm_state_sha256=before, restored_training_state=True, rows=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/toy100/constraints_simple_regularization.json').read_text())
    result = {}
    def hook(state):
        result.update(diagnose(state, config))
        raise DiagnosticComplete()
    try:
        run_probe(config, mode='scheduled', checkpoint_hook_step=1000, checkpoint_hook=hook)
    except DiagnosticComplete:
        pass
    result.update(config=config, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
