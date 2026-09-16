"""Read-only checkpoint audit; no training or optimizer steps."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments import memory_handoff_scout as h
from experiments import memory_core_scout as core
from experiments import memory_scout as base
from experiments.memory_orbit_metrics import fit_circle, angles


@torch.no_grad()
def audit(run):
    saved = torch.load(run/'model.pt', map_location='cuda:0', weights_only=False)
    cfg = h.Config(**saved['config'])
    g, d, prior, recipe = h.build(cfg, 'cuda:0')
    for name, module in [('generator', g), ('critic', d), ('prior', prior)]:
        module.load_state_dict(saved[name])
        module.eval()
    recorded = json.loads((run/'config.json').read_text())
    assert json.loads(json.dumps(recipe.to_dict())) == recorded['resolved_recipe']
    assert cfg.adversarial_only and cfg.mismatch_context == 'clean'
    assert not any((cfg.predict_weight, cfg.temporal_weight, cfg.repair_weight,
                    cfg.stability_g_weight, cfg.stability_d_weight))
    assert recipe.reg_arm == 'b_cap' and recipe.reg_method == 'autograd'
    assert recipe.reg_every == 1 and recipe.reg_coeff == recipe.reg_kappa == 1
    arrays = dict(np.load(run/'trajectories.npz'))
    z = prior(torch.arange(cfg.eval_batch, device='cuda:0'))
    result = {'name': cfg.name, 'step': saved['step'], 'device': 'cuda:0',
              'recipe_matches_saved': True, 'replay_max_abs_error': {},
              'warm_prefix32': {}}
    cold, _ = base.rollout(g, d.writer, z, cfg.eval_steps)
    for name, generated in [('generated', cold)]:
        expected = torch.as_tensor(arrays[name], device='cuda:0')
        torch.testing.assert_close(generated, expected, rtol=0, atol=1e-6)
        result['replay_max_abs_error'][name] = float((generated-expected).abs().max())
    for n in (8, 32):
        prefix = torch.as_tensor(arrays[f'observed_prefix{n}'], device='cuda:0')
        generated, _ = core.continuation(g, d.writer, z, prefix, cfg.eval_steps)
        expected = torch.as_tensor(arrays[f'prefix{n}'], device='cuda:0')
        torch.testing.assert_close(generated, expected, rtol=0, atol=1e-6)
        result['replay_max_abs_error'][f'prefix{n}'] = float((generated-expected).abs().max())

    # Training's full-write pair must reproduce the first two runtime outputs.
    observed = torch.as_tensor(arrays['continuation_reference'][:, :64].copy(), device='cuda:0')
    observed[:, :32] = torch.as_tensor(arrays['observed_prefix32'], device='cuda:0')
    positions = torch.full((len(z), 1), 33, device='cuda:0')
    view, actual, fake = h.local_pair_examples(cfg, g, d, observed, observed,
                                             positions, z, positions.flatten())
    expected = torch.as_tensor(arrays['prefix32'][:, :2], device='cuda:0')
    torch.testing.assert_close(fake, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual, observed[:, 32:34], rtol=0, atol=0)
    result['pair_matches_runtime_max_abs_error'] = float((fake-expected).abs().max())

    # The zero-strength point path must give the same first runtime output too.
    positions.fill_(32)
    memory = h.selected_memories(d.writer, observed, positions, cfg.max_prefix)
    generated, _ = h.local_point(g, z, memory, positions.flatten())
    torch.testing.assert_close(generated, expected[:, 0], rtol=0, atol=1e-6)
    result['point_matches_runtime_max_abs_error'] = float((generated-expected[:, 0]).abs().max())

    # B-cap differentiates candidate coordinates, not the recurrent transition.
    # For this concatenation/LeakyReLU head the input slope is locally constant
    # in M, so even an active exact penalty has no writer gradient here.
    with torch.enable_grad():
        memory = h.selected_memories(d.writer, observed, positions, cfg.max_prefix)
        penalty = recipe.make_gradient_penalty()(h.CandidateView(d, memory, positions.flatten()),
                                                observed[:, 32], generated, step=cfg.steps)
        gradients = torch.autograd.grad(penalty, tuple(d.writer.parameters()), allow_unused=True)
    result['clean_point_bcap'] = {
        'value': float(penalty.detach()),
        'writer_gradient_max_abs': max(float(v.abs().max()) for v in gradients if v is not None),
        'note': 'Clean prefix32, clean next target, generated fake. GAN losses still train writer.',
    }

    reference = arrays['continuation_reference'].astype(np.float64)
    center, radius = fit_circle(reference[:, :32])
    ref = reference-center[:, None]
    omega = angles(ref[:, 0], ref[:, 1])
    generated = arrays['prefix32'].astype(np.float64)
    offsets = generated-center[:, None]
    preceding = np.concatenate((ref[:, 31:32], offsets[:, :-1]), axis=1)
    angular = angles(preceding, offsets)
    radial = np.linalg.norm(offsets, axis=-1)/radius[:, None]-1
    position = np.linalg.norm(generated-reference[:, 32:32+cfg.eval_steps], axis=-1)/radius[:, None]
    for length in (1, 8, 32, 64, 256, 1024):
        rmse = np.sqrt(np.mean(radial[:, :length]**2, axis=1))
        direction = (angular[:, :length]*np.sign(omega[:, None]) > 0).mean(1)
        speed = np.abs(angular[:, :length].mean(1)-omega)
        startup = position[:, 0]
        result['warm_prefix32'][str(length)] = {
            'mean_radial_rmse': float(rmse.mean()),
            'mean_position_error_relative': float(position[:, :length].mean()),
            'radial_pass_count': int((rmse < .1).sum()),
            'direction_pass_count': int((direction > .95).sum()),
            'speed_pass_count': int((speed < .03).sum()),
            'startup_pass_count': int((startup < .2).sum()),
            'joint_threshold_count': int(((rmse < .1) & (direction > .95)
                                         & (speed < .03) & (startup < .2)).sum()),
        }
    result['short_window_note'] = 'Threshold diagnostics only; short windows are not full-circle success.'
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    paths = [ROOT/'runs/memory_path/principles_round12/runs/match_shuffle25',
             ROOT/'runs/memory_path/principles_round12_followup/runs/match_shuffle25_5k']
    results = [audit(path) for path in paths]
    (Path(__file__).parent/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
