"""Disposable current-public-optimizer maps on frozen original gradient panels.

No model trajectory is continued. Every map restores the same original model
and normalized-optimizer state. Only the designated diagnostic signal history
advances between panels. Source gradients are replayed by linear autograd
surrogates; actual nonlinear objectives are independently checked/evaluated.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from unittest.mock import patch

START = time.monotonic()
HERE = Path(__file__).resolve().parent
CURRENT = Path('/ml2/hypergan/particlegan-selective-history-20261015')
sys.path.insert(0, str(CURRENT))
signal.alarm(440)

import numpy as np
import scipy
import torch

from particlegan import Recipe
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.dualnorm import NormalizedOptimizer
from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return deepcopy(value)


def flat(values):
    return torch.cat([value.detach().flatten() for value in values])


INACTIVE_REMOVED = dict(cell_gaussian_objective=None, cell_reference_policy='cumulative',
    cell_quota_policy='largest_remainder', cell_shape_weighting='mass', cell_assignment_refinement='none',
    cell_transport_policy='integer', cell_distribution_shape='none', cell_covariance_memory='unchanged',
    cell_gradient_projection=False, cell_extragradient_map=False)


def construct(payload, mode):
    original = payload['original_state']
    recipe = deepcopy(payload['recipe'])
    assert list(recipe.pop('optimizer_signal_roles')) == ['generator']
    removed = {}
    for key, expected in INACTIVE_REMOVED.items():
        if key in recipe:
            actual = recipe.pop(key)
            assert actual == expected, f'active unsupported archived feature {key}: {actual}'
            removed[key] = actual
    assert not set(recipe) - {field.name for field in fields(Recipe)}
    assert recipe.get('constraint_geometry_mode', 'none') == 'none'
    assert recipe['optimizer_family'] == 'dualnorm' and recipe.get('optimizer_momentum', 0.) == 0.
    assert recipe['kinetic_transport_weight'] == recipe['kinetic_transport_local_weight'] == 1.
    recipe.update(constraint_geometry_mode='direction_blend', generator_history_mode=mode)
    prior = deepcopy(original['prior'])
    prior.pop('learnable', None)
    context = FormulationContext(recipe_preset=None, recipe_overrides=recipe, prior=prior,
        prior_contract=original['prior_contract'], seed=0, device='cuda:0',
        initializer=original['initializer'], execution_path='public_trainer')
    generator, critic = build_vector_models(context, payload['target_spec'])
    trainer = context.build_trainer(generator, critic, max_steps=original['trainer'].get('max_steps'))
    trainer.G.train(); trainer.D.eval(); trainer.D.requires_grad_(False)
    load_models(trainer, payload)
    named = {'G.' + n: p for n, p in trainer.G.named_parameters()}
    named.update({'prior.' + n: p for n, p in trainer.prior.named_parameters()})
    params = [p for group in trainer.opt_g.param_groups for p in group['params']]
    assert len(params) == len(payload['parameter_map'])
    for parameter, row in zip(params, payload['parameter_map']):
        assert parameter is named[row['name']] and list(parameter.shape) == row['shape']
        assert str(parameter.dtype) == row['dtype'] and parameter.requires_grad
        assert trainer.opt_g.param_groups[row['group']]['params'][row['slot']] is parameter
        assert trainer.opt_g.param_groups[row['group']]['role'] == row['role']
    assert trainer.recipe.input_noise_std == trainer.recipe.output_noise_std == 0.
    assert trainer.recipe.prior_reg == trainer.recipe.prior_l2 == 0.
    trainer._diagnostic_empty_optimizer = deepcopy(trainer.opt_g.state_dict())
    return trainer, params, removed


def load_models(trainer, payload):
    for name, saved in payload['original_state']['trainer']['models'].items():
        module = getattr(trainer, name)
        module.load_state_dict(saved, strict=True)
        assert state_digest(cpu(module.state_dict())) == state_digest(saved), f'{name} model roundtrip'


def mapped_state(trainer, payload, history, panel_index):
    saved = deepcopy(payload['original_state']['trainer']['optimizers'][0])
    assert set(saved) == {'state', 'param_groups', 'dualnorm'}
    dropped = saved['dualnorm'].pop('gradient_signal')
    assert dropped['roles'] == ['generator']
    state = deepcopy(trainer._diagnostic_empty_optimizer)
    for key in ('state', 'param_groups', 'dualnorm'):
        state[key] = saved[key]
    if history is not None:
        state['component_history']['signal'] = deepcopy(history)
        # Synthetic bookkeeping: previous independent diagnostic observations,
        # never original training updates or accumulated model trajectories.
        state['constraint_geometry']['stats']['steps'] = panel_index
    trainer.opt_g.load_state_dict(state)
    assert state_digest(cpu(trainer.opt_g.state_dict())) == state_digest(cpu(state))
    return state


def objective(trainer, payload, panel):
    rows = panel['g_rows'].to('cuda:0')
    old_prior = payload['original_state']['trainer']['models']['prior']['z'].to('cuda:0')
    # Exactly preserve original baseline latent bits; width subtraction/re-add
    # could alter them. The difference remains differentiable in updated rows.
    latent = panel['g_latent'].to('cuda:0') + (trainer.prior.z[rows] - old_prior[rows])
    fake = trainer.G(latent)
    real = panel['real_g'].to('cuda:0')
    real_logits, fake_logits = trainer.D(real), trainer.D(fake)
    adversarial = trainer.loss.g_loss(fake_logits, real_logits)
    global_transport = trainer.recipe.kinetic_transport_loss(fake, real)
    local_transport = trainer.recipe.kinetic_transport_local_loss(fake, real)
    raw = trainer.prior.z if trainer.recipe.num_particles <= 1024 else trainer.prior.z[torch.unique(rows)]
    other = trainer.recipe.prior_reg * trainer.prior_regularizer(raw)
    total = adversarial + other
    total = total + global_transport
    total = total + local_transport
    channels = dict(adversarial=adversarial, global_transport=global_transport,
        local_transport=local_transport, transport=global_transport + local_transport, other=other, total=total)
    return channels, fake, real_logits, fake_logits


def derivatives(channels, params):
    result = {}
    for key, value in channels.items():
        gradients = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        if key == 'total': assert all(g is not None for g in gradients), 'source replay cannot invent absent gradient slots'
        result[key] = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gradients)]
    return result


def finite_readout(channels, fake):
    x = fake.detach().double()
    centered = x - x.mean(0)
    values = {name: float(value.detach()) for name, value in channels.items()}
    assert all(math.isfinite(value) for value in values.values())
    assert bool(torch.isfinite(x).all())
    return dict(losses=values, output_mean=cpu(x.mean(0)).tolist(),
                output_covariance=cpu(centered.T @ centered / len(x)).tolist())


def baseline_parity(trainer, params, payload, panel):
    channels, fake, real_logits, fake_logits = objective(trainer, payload, panel)
    assert torch.equal(cpu(fake), panel['generated']), 'current G baseline differs from source output'
    assert torch.equal(cpu(real_logits), panel['real_logits'])
    assert torch.equal(cpu(fake_logits), panel['fake_logits'])
    gradients = derivatives(channels, params)
    errors = {}
    for name in channels:
        assert float(channels[name].detach()) == panel['losses'][name], f'{name} baseline scalar differs'
        errors[name] = max(float((cpu(g) - old).abs().max())
                           for g, old in zip(gradients[name], panel['gradients'][name]))
        assert errors[name] == 0., f'{name} gradient parity {errors[name]}'
    return finite_readout(channels, fake), errors


def make_surrogate(params, values):
    return sum((parameter * value.to(parameter)).sum() for parameter, value in zip(params, values))


def one_map(trainer, params, payload, panel, history, index):
    load_models(trainer, payload)
    mapped = mapped_state(trainer, payload, history, index)
    optimizer = trainer.opt_g
    optimizer.zero_grad(set_to_none=True)
    optimizer.set_sampled_rows(trainer.prior.z, panel['g_rows'].to('cuda:0'))
    originals = [p.detach().clone() for p in params]
    gradients = panel['gradients']
    adversarial = make_surrogate(params, gradients['adversarial'])
    total = make_surrogate(params, gradients['total'])
    before_rng = torch.cuda.get_rng_state().clone()
    constraint_geometry_backward(total, optimizer, (adversarial,), adversarial_loss=adversarial)
    pre_map = flat([p.grad for p in params])
    raw_normals = optimizer._protected.detach().clone()
    expected_normal = [g.to(p).clone() for g, p in zip(gradients['adversarial'], params)]
    for i, row in enumerate(payload['parameter_map']):
        if row['role'] == 'prior':
            masked = torch.zeros_like(expected_normal[i])
            masked[panel['g_rows']] = expected_normal[i][panel['g_rows']]
            expected_normal[i] = masked
    assert torch.equal(raw_normals[0], flat(expected_normal)), 'raw current normal changed'
    for i, row in enumerate(payload['parameter_map']):
        if row['role'] != 'generator':
            assert torch.equal(params[i].grad, gradients['total'][i].to(params[i])), 'non-G gradient transformed'
    normalized = {}
    original_step = NormalizedOptimizer.step
    def observe_step(instance, closure=None):
        assert instance is optimizer and not normalized
        result = original_step(instance, closure)
        normalized['parameters'] = [p.detach().clone() for p in params]
        return result
    with patch.object(NormalizedOptimizer, 'step', observe_step):
        optimizer.step()
    assert torch.equal(before_rng, torch.cuda.get_rng_state()), 'optimizer replay consumed RNG'
    after = [p.detach().clone() for p in params]
    normal_delta = flat([p - old for p, old in zip(normalized['parameters'], originals)])
    final_delta = flat([p - old for p, old in zip(after, originals)])
    assert bool(torch.isfinite(pre_map).all()) and bool(torch.isfinite(normal_delta).all()) and bool(torch.isfinite(final_delta).all())
    after_state = optimizer.state_dict()
    for identifier, old in mapped['state'].items():
        assert after_state['state'][identifier]['step'] == old['step'] + 1
    with torch.no_grad():
        channels, fake, _, _ = objective(trainer, payload, panel)
        readout = finite_readout(channels, fake)
    assert state_digest(cpu(trainer.D.state_dict())) == state_digest(payload['original_state']['trainer']['models']['D'])
    gen_mask = torch.cat([torch.full((p.numel(),), row['role'] == 'generator', device=p.device, dtype=torch.bool)
                          for p, row in zip(params, payload['parameter_map'])])
    row = dict(pre_map_gradient=cpu(pre_map), normalized_proposal=cpu(normal_delta),
               blended_displacement=cpu(final_delta), actual_parameters=[cpu(p) for p in after],
               raw_protected_normals=cpu(raw_normals), finite_after=readout,
               protected_derivative=float(raw_normals[0].double() @ final_delta.double()),
               geometry=deepcopy(optimizer.constraint_geometry_stats),
               direction_blend=deepcopy(optimizer.direction_blend_stats),
               generator_displacement_norm=float(final_delta[gen_mask].double().norm()),
               total_displacement_norm=float(final_delta.double().norm()),
               history=None if not hasattr(optimizer, 'gradient_signal') else optimizer.gradient_signal.state_dict())
    if row['history'] is not None:
        row['signal_receipt'] = deepcopy(row['history']['packets']['generator']['last'])
    return row, cpu(gen_mask)


def synthesize(payload):
    trainer, params, _ = construct(payload, 'none')
    panel = deepcopy(payload['panels'][0])
    panel['real_g'] = panel['real_g'][:1].repeat(len(panel['real_g']), 1)
    panel['real_d'] = panel['real_g'].clone()
    channels, fake, real_logits, fake_logits = objective(trainer, payload, panel)
    panel['gradients'] = cpu(derivatives(channels, params))
    panel['losses'] = {key: float(value.detach()) for key, value in channels.items()}
    panel.update(generated=cpu(fake), real_logits=cpu(real_logits), fake_logits=cpu(fake_logits))
    result = deepcopy(payload)
    result['name'] = 'synthetic-repeated-real'
    result['panels'] = []
    for index in range(32):
        copied = deepcopy(panel); copied['index'] = index
        result['panels'].append(copied)
    result['synthetic'] = dict(source='gaussian1000 panel0', target='first real row repeated to original batch size',
        latent='same first-panel latent rows/noise repeated', rng_advances=False, original_window_credit=False)
    return result


def run_case(payload):
    assert len(payload['panels']) == 32 and payload['optimizer_updates'] == 0
    models = {arm: construct(payload, mode) for arm, mode in [('T', 'none'), ('W', 'whole'), ('S', 'adversarial')]}
    histories = {'T': None, 'W': None, 'S': None}
    raw_rows, compact = [], []
    for index, panel in enumerate(payload['panels']):
        assert panel['index'] == index
        load_models(models['T'][0], payload)
        baseline, errors = baseline_parity(models['T'][0], models['T'][1], payload, panel)
        arms = {}
        for arm, (trainer, params, _) in models.items():
            arms[arm], gen_mask = one_map(trainer, params, payload, panel, histories[arm], index)
            histories[arm] = arms[arm]['history']
        channel_vectors = {name: flat(values).double() for name, values in panel['gradients'].items()}
        inner = {f'{a}:{b}': float(channel_vectors[a] @ channel_vectors[b])
                 for a, b in [('adversarial', 'transport'), ('adversarial', 'other'), ('transport', 'other')]}
        derivative = {}
        for arm in arms:
            delta = arms[arm]['blended_displacement'].double()
            derivative[arm] = {name: float(vector @ delta) for name, vector in channel_vectors.items()}
        sd = arms['S']['blended_displacement'].double()
        tolerance = 64 * torch.finfo(arms['S']['blended_displacement'].dtype).eps * float(channel_vectors['transport'].norm()) * float(sd.norm())
        reversal = derivative['T']['transport'] < 0 and derivative['S']['transport'] > tolerance
        tnorm = arms['T']['generator_displacement_norm']
        summary = dict(index=index, baseline=baseline, baseline_gradient_max_absolute_error=errors,
            channel_norms={key: float(value.norm()) for key, value in channel_vectors.items()}, channel_inner_products=inner,
            derivative=derivative, fresh_transport_reversal=bool(reversal), reversal_tolerance=tolerance,
            identity_correction='not applicable: these recovered unconditional hosts have no identity objective',
            generator_T_S_identical=torch.equal(arms['T']['blended_displacement'][gen_mask], arms['S']['blended_displacement'][gen_mask]),
            generator_S_over_T_displacement_ratio=arms['S']['generator_displacement_norm'] / tnorm if tnorm else None,
            arms={arm: {key: value for key, value in item.items() if key in
                ('finite_after', 'protected_derivative', 'geometry', 'direction_blend', 'signal_receipt',
                 'generator_displacement_norm', 'total_displacement_norm')} for arm, item in arms.items()})
        raw_rows.append(dict(index=index, arms=arms, summary=summary))
        compact.append(summary)
        print(json.dumps(dict(event='panel', name=payload['name'], index=index,
            rho_S=summary['arms']['S']['signal_receipt']['rho'], reversal=bool(reversal),
            identical=summary['generator_T_S_identical'])), flush=True)
    output = HERE / (payload['name'] + '-current-maps.pt')
    torch.save(dict(schema_version=1, source_name=payload['name'], parameter_map=payload['parameter_map'], panels=raw_rows), output)
    return dict(name=payload['name'], status='PASS' if not any(row['fresh_transport_reversal'] for row in compact) else 'FAIL',
        panel_count=32, gradient_replay_parity='exact current-vs-source scalars/gradients before every panel',
        removed_inactive_recipe_fields=models['T'][2],
        phase_target_means=payload['target_spec']['means'], synthetic=payload.get('synthetic'),
        identical_T_S_generator_panels=sum(row['generator_T_S_identical'] for row in compact),
        noninitial_effect_panels=sum(not row['generator_T_S_identical'] for row in compact[1:]),
        fresh_transport_reversal_panels=[row['index'] for row in compact if row['fresh_transport_reversal']],
        output_path=str(output), output_sha256=sha(output), panels=compact)


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.autograd.set_multithreading_enabled(False)
    export = json.loads((HERE / 'original-gradient-panels.json').read_text())
    roster = json.loads((CURRENT / 'reports/forge/selective-history/stage-roster.json').read_text())
    report = dict(schema_version=1, kind='independent_frozen_state_counterfactual_current_optimizer_maps',
        script_sha256=sha(__file__), original_exports_sha256=sha(HERE / 'original-gradient-panels.json'),
        current_roster_sha256=sha(CURRENT / 'reports/forge/selective-history/stage-roster.json'),
        current_core_sha256={name: sha(CURRENT/name) for name in
            ('particlegan/optim/component_history.py', 'particlegan/optim/gradient_signal.py',
             'particlegan/optim/dualnorm.py', 'particlegan/optim/direction_blend.py',
             'particlegan/optim/constraint_geometry.py', 'particlegan/gan_loss.py',
             'particlegan/kinetic_transport.py', 'lib/toy_models.py')},
        current_source_roster=roster,
        preflight_sha256=sha(CURRENT / 'reports/forge/selective-history/PREFLIGHT.md'),
        runtime=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                     device=torch.cuda.get_device_name(0), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
        mapping=dict(original_models='strict state_dict roundtrip before every panel',
            original_normalized_optimizer='state/param_groups preserved; dualnorm.gradient_signal deliberately removed',
            geometry='new current direction wrapper; all motion statistics reset per independent proposal; steps=prior observation count for W/S only',
            history='empty initially; advances only designated original frozen-state channel, never original archived whole-history packet',
            gradients='linear autograd surrogates replay exact exported original gradients after exact nonlinear baseline parity check',
            finite_latent='original_latent+(updated_prior[rows]-original_prior[rows])',
            operation_order='g+(rho-1)*a for S; rho*g for W; then current public normalized optimizer and raw-normal direction blend',
            no_updates=['critic', 'schedule', 'original artifacts', 'original RNG streams'],
            model_trajectory_continued=False),
        campaign_training=False, available_state_diagnostic_only=True, required_windows_missing=True,
        overall_preflight_status='BLOCKED', cases=[])
    def save():
        report['process_wall_seconds'] = time.monotonic() - START
        (HERE / 'current-counterfactual-maps.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    first = None
    for record in export['states']:
        name = record['name']
        print(json.dumps(dict(event='start', name=name)), flush=True)
        try:
            assert record['status'] == 'PASS' and sha(record['path']) == record['sha256']
            payload = torch.load(record['path'], map_location='cpu', weights_only=True)
            assert sha(payload['original_checkpoint_path']) == payload['original_checkpoint_sha256']
            if first is None: first = payload
            row = run_case(payload)
            row.update(input_path=record['path'], input_sha256=record['sha256'])
        except Exception as error:
            row = dict(name=name, status='BLOCKED', reason=repr(error), traceback=traceback.format_exc())
        report['cases'].append(row); save()
        print(json.dumps({key: value for key, value in row.items() if key != 'panels'}), flush=True)
    if first is not None:
        try:
            row = run_case(synthesize(first))
        except Exception as error:
            row = dict(name='synthetic-repeated-real', status='BLOCKED', reason=repr(error), traceback=traceback.format_exc())
        report['cases'].append(row); save()
    report['available_reversal_observed'] = any(row.get('fresh_transport_reversal_panels') for row in report['cases'])
    report['available_noninitial_effect_observed'] = any(row.get('noninitial_effect_panels', 0) > 0
        for row in report['cases'] if row['name'] != 'synthetic-repeated-real')
    save()
    print(json.dumps(dict(event='finished', process_wall_seconds=report['process_wall_seconds'],
        statuses={row['name']: row['status'] for row in report['cases']}, overall_preflight_status='BLOCKED')), flush=True)


if __name__ == '__main__': main()
