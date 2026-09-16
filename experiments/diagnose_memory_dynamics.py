"""Completed-only local memory/clock/adapter probes; no training or ranking changes."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import memory_handoff_scout as handoff
from experiments import memory_core_scout as core
from experiments.memory_scout import rollout, diagnostics


@contextmanager
def bypass_adapter(generator):
    adapter = generator.memory_adapter
    generator.memory_adapter = None
    try:
        yield
    finally:
        generator.memory_adapter = adapter


def gain_metrics(difference, perturbation):
    gain = (difference.square().flatten(1).sum(1)/
            perturbation.square().flatten(1).sum(1).clamp_min(1e-12)).sqrt()
    return {'rms': float(gain.square().mean().sqrt()), 'median': float(gain.median()),
            'p95': float(torch.quantile(gain, .95)), 'max': float(gain.max()),
            'fraction_above_one': float((gain > 1).float().mean())}


def feedback_jacobian_metrics(g, writer, z, memory, time_index):
    """Worst infinitesimal one-step gain, not a long-run stability guarantee."""
    def transition(flat, latent):
        state = flat.reshape(1, *memory.shape[1:])
        point, _ = handoff.local_point(g, latent[None], state, time_index)
        return writer.write(state, point).flatten()
    with torch.enable_grad():
        jacobian = torch.func.vmap(torch.func.jacrev(transition, argnums=0))(
            memory.detach().flatten(1), z.detach()).detach()
    largest = torch.linalg.svdvals(jacobian)[:, 0]
    return {'largest_singular_value_mean': float(largest.mean()),
            'largest_singular_value_p95': float(torch.quantile(largest, .95)),
            'fraction_above_one': float((largest > 1).float().mean())}


@torch.no_grad()
def diagnose(path, device, bypass_rollouts=False, jacobians=False):
    # A checkpoint is written before evaluation; summary marks evaluation done.
    json.loads((path/'summary.json').read_text())
    saved = torch.load(path/'model.pt', map_location=device, weights_only=False)
    cfg = handoff.Config(**saved['config'])
    g, d, prior, _ = handoff.build(cfg, device)
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval()
    z = prior(torch.arange(cfg.eval_batch, device=device))
    with np.load(path/'trajectories.npz') as arrays:
        observed = torch.as_tensor(arrays['observed_prefix32'].copy(), device=device)
        clean_np = arrays['continuation_reference'].copy()
        saved_first = {n: arrays[f'prefix{n}'][:, 0].copy() for n in cfg.eval_prefixes}
    clean = torch.as_tensor(clean_np, device=device)
    rows = {}
    for n in cfg.eval_prefixes:
        memory = core.context(d.writer, observed[:, :n])
        point, _ = handoff.local_point(g, z, memory, n)
        np.testing.assert_allclose(point.cpu().numpy(), saved_first[n], atol=1e-5, rtol=1e-5)
        translated = g.readable_memory(z, memory, time_index=n)
        row = {'normal_target_mse': float((point-clean[:, n]).square().mean()),
               'raw_memory_rms': float(memory.square().mean().sqrt()),
               'adapter_correction_rms': float((translated-memory).square().mean().sqrt()),
               'raw_across_episode_std': float(memory.std(0).mean()),
               'translated_across_episode_std': float(translated.std(0).mean())}
        for key, read, time in [('memory_zero', torch.zeros_like(memory), n),
                                ('memory_shuffle', memory.roll(1, 0), n), ('clock_zero', memory, 0)]:
            altered, _ = handoff.local_point(g, z, read, time)
            row[key] = {'target_mse': float((altered-clean[:, n]).square().mean()),
                        'output_change_mse': float((altered-point).square().mean())}
        with bypass_adapter(g):
            bypass, _ = handoff.local_point(g, z, memory, n)
        row['adapter_bypass_target_mse'] = float((bypass-clean[:, n]).square().mean())
        noise = torch.randn(memory.shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(81800+n))
        direction = noise/noise.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        next_memory = d.writer.write(memory, point)
        next_real = d.writer.write(memory, clean[:, n])
        after_generated, _ = handoff.local_point(g, z, next_memory, n+1)
        after_real, _ = handoff.local_point(g, z, next_real, n+1)
        row['next_prediction_after_generated_write_mse'] = float((after_generated-clean[:, n+1]).square().mean())
        row['next_prediction_after_clean_real_write_mse'] = float((after_real-clean[:, n+1]).square().mean())
        row['generated_vs_clean_real_write_mse'] = float((next_memory-next_real).square().mean())
        if jacobians:
            row['feedback_jacobian'] = feedback_jacobian_metrics(g, d.writer, z, memory, n)
        perturbations = {}
        for amount in (.03, .05, .15):
            delta = amount*direction
            corrupted = memory+delta
            altered, _ = handoff.local_point(g, z, corrupted, n)
            altered_next = d.writer.write(corrupted, altered)
            repaired = g.readable_memory(z, corrupted, time_index=n)
            with bypass_adapter(g):
                bypass, _ = handoff.local_point(g, z, corrupted, n)
            perturbations[str(amount)] = {
                'target_mse': float((altered-clean[:, n]).square().mean()),
                'output_change_mse': float((altered-point).square().mean()),
                'adapter_bypass_target_mse': float((bypass-clean[:, n]).square().mean()),
                'translated_consistency_mse': float((repaired-translated).square().mean()),
                'raw_repair_mse': float((repaired-memory).square().mean()),
                'feedback_gain': gain_metrics(altered_next-next_memory, delta),
                'same_real_input_writer_gain': gain_metrics(d.writer.write(corrupted, clean[:, n])-next_real, delta)}
        row['perturbations'] = perturbations
        rows[f'prefix{n}'] = row
    results = {'name': cfg.name, 'steps': cfg.steps, 'source': str(path), 'evaluation_only': True,
               'note': 'Saved evaluation panel, local probes on real-prefix memories. Dependence and local robustness do not imply stable autonomous continuation.',
               'local': rows}
    if bypass_rollouts and g.memory_adapter is not None:
        with bypass_adapter(g):
            generated, _ = rollout(g, d.writer, z, cfg.eval_steps)
            results['adapter_bypass_cold_long'] = diagnostics(generated.cpu().numpy())
            results['adapter_bypass_warm_long'] = {}
            for n in cfg.eval_prefixes:
                generated, _ = core.continuation(g, d.writer, z, observed[:, :n], cfg.eval_steps)
                results['adapter_bypass_warm_long'][f'prefix{n}'] = core.fidelity(generated.cpu().numpy(), clean_np, n)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bypass-rollouts', action='store_true')
    parser.add_argument('--jacobians', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = [diagnose(path, args.device, args.bypass_rollouts, args.jacobians) for path in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    for result in results:
        print(result['name'], {k: v['normal_target_mse'] for k, v in result['local'].items()})
