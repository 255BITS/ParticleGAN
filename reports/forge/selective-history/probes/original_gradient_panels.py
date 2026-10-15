"""Observe 32 frozen-state gradient panels from each recovered original.

Only the disposable context's named sampling streams advance. No optimizer,
policy, original estimator, model parameter or original artifact is updated.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import original_restore_parity as original
from original_restore_parity import torch, np, scipy, state_digest, sha

START = time.monotonic()
signal.alarm(450)
HERE = Path(__file__).resolve().parent
CURRENT = Path('/ml2/hypergan/particlegan-selective-history-20261015')


def core_state(context):
    state = deepcopy(context.state_dict())
    del state['streams']
    del state['trainer']['streams']
    return state


def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return deepcopy(value)


def gradients(loss, parameters, *, retain_graph=True):
    values = (torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
              if loss.requires_grad else [None] * len(parameters))
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)]


def describe_parameters(trainer):
    names = {id(p): 'G.' + n for n, p in trainer.G.named_parameters()}
    names.update({id(p): 'prior.' + n for n, p in trainer.prior.named_parameters()})
    parameters, rows = [], []
    for gi, group in enumerate(trainer.opt_g.param_groups):
        for pi, parameter in enumerate(group['params']):
            assert parameter.requires_grad and id(parameter) in names
            rows.append({'name': names[id(parameter)], 'shape': list(parameter.shape),
                         'dtype': str(parameter.dtype), 'role': group['role'], 'group': gi, 'slot': pi})
            parameters.append(parameter)
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {r['role'] for r in rows} == {'generator', 'prior'}
    return parameters, rows


def observe(context, trainer, spec, task_id, panel_index, parameters):
    recipe = trainer.recipe
    assert recipe.input_noise_std == 0 and recipe.output_noise_std == 0
    assert recipe.prior_reg == 0 and recipe.prior_l2 == 0 and trainer.cell_objective is None
    assert recipe.optimizer_signal_roles == ('generator',) or list(recipe.optimizer_signal_roles) == ['generator']
    assert trainer.G.__class__.__name__ == 'SimpleMLPGenerator'
    assert trainer.D.__class__.__name__ == 'SimpleMLPDiscriminator'
    assert all(type(m).__name__ in {'SimpleMLPGenerator', 'SimpleMLPDiscriminator', 'Sequential', 'Linear', 'LeakyReLU'}
               for model in [trainer.G, trainer.D] for m in model.modules())
    fixed_before = state_digest(core_state(context))
    stream_before = cpu(context.streams.state_dict())
    data = context.streams.generator('data', component='target', purpose='training', device='cpu')
    sampler = original.gaussian_target if task_id.startswith('gaussian') else original.vector_target
    real_cpu = sampler(spec, recipe.batch_size, data, trainer.completed_steps + panel_index)
    real = real_cpu.to('cuda:0')
    flags = [p.requires_grad for p in trainer.D.parameters()]
    modes = [(m, m.training) for model in [trainer.G, trainer.D] for m in model.modules()]
    # These four original hosts use deterministic MLPs, no input/output noise,
    # and BCAP/R1 has no sampling operation. Preserve the exact D-prior then
    # G-prior stream order; no critic step or penalty state mutation occurs.
    try:
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.set_rng_state(trainer.model_generator.get_state(), 'cuda:0')
            try:
                trainer.D.train(); trainer.G.eval()
                with torch.no_grad():
                    d_latent, d_rows = trainer._sample_training_prior(len(real))
                    d_offset = d_latent - trainer.prior.z[d_rows]
                    d_fake = trainer._generate(trainer.G, d_latent, 0., trainer.noise_generator, rows=d_rows)
                    trainer._noisy_D(real); trainer._noisy_D(d_fake)
                trainer.D.eval(); trainer.G.train(); trainer.D.requires_grad_(False)
                latent, rows = trainer._sample_training_prior(len(real))
                offset = latent.detach() - trainer.prior.z[rows].detach()
                fake = trainer._generate(trainer.G, latent, 0., trainer.noise_generator, rows=rows)
                fake_logits = trainer._noisy_D(fake)
                real_logits = trainer._noisy_D(real)
                adversarial = trainer.loss.g_loss(fake_logits, real_logits)
                global_transport = recipe.kinetic_transport_loss(fake, real)
                local_transport = recipe.kinetic_transport_local_loss(fake, real)
                prior_reg = adversarial.new_zeros(())
                if trainer.prior.z.requires_grad and (recipe.prior_regularizer == 'vicreg' or recipe.prior_reg > 0):
                    raw = trainer.prior.z if recipe.num_particles <= 1024 else trainer.prior.z[torch.unique(rows)]
                    prior_reg = trainer.prior_regularizer(raw)
                other = recipe.prior_reg * prior_reg
                # Match the original scalar construction order exactly.
                total = adversarial + other
                total = total + global_transport
                total = total + local_transport
                channels = {'adversarial': adversarial, 'global_transport': global_transport,
                            'local_transport': local_transport, 'transport': global_transport + local_transport,
                            'other': other, 'total': total}
                grads = {name: cpu(gradients(value, parameters)) for name, value in channels.items()}
                losses = {name: float(value.detach()) for name, value in channels.items()}
                assert all(torch.isfinite(g).all().item() for values in grads.values() for g in values)
                assert all(np.isfinite(v) for v in losses.values())
                assert torch.isfinite(fake).all()
                panel = {'index': panel_index, 'observation_sequence_index': trainer.completed_steps + panel_index,
                         'gradients': grads, 'protected_gradient_channel': 'adversarial', 'losses': losses,
                         'real_d': cpu(real), 'real_g': cpu(real), 'same_real_tensor_for_d_and_g': True,
                         'd_latent': cpu(d_latent), 'd_rows': cpu(d_rows), 'd_width_offset': cpu(d_offset),
                         'g_latent': cpu(latent), 'g_rows': cpu(rows), 'g_width_offset': cpu(offset),
                         'prior_sigma': cpu(trainer.prior.sigma), 'generated': cpu(fake), 'd_generated': cpu(d_fake),
                         'fake_logits': cpu(fake_logits), 'real_logits': cpu(real_logits),
                         'stream_state_before': stream_before,
                         'output_mean': cpu(fake.mean(0)),
                         'output_covariance': cpu((fake-fake.mean(0)).T @ (fake-fake.mean(0)) / len(fake))}
            finally:
                trainer.model_generator.set_state(torch.cuda.get_rng_state('cuda:0'))
    finally:
        for parameter, flag in zip(trainer.D.parameters(), flags): parameter.requires_grad_(flag)
        for module, training in modes: module.training = training
    panel['stream_state_after'] = cpu(context.streams.state_dict())
    panel['streams_before_sha256'] = state_digest(panel['stream_state_before'])
    panel['streams_after_sha256'] = state_digest(panel['stream_state_after'])
    panel['frozen_state_sha256'] = state_digest(core_state(context))
    assert fixed_before == panel['frozen_state_sha256'], 'frozen model/optimizer/policy state changed'
    panel['real_sha256'] = state_digest(panel['real_g'])
    return panel


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.autograd.set_multithreading_enabled(False)
    original.verify_snapshot(original.SOURCE)
    transport_hash = sha(original.SOURCE/'particlegan/kinetic_transport.py')
    assert transport_hash == sha(CURRENT/'particlegan/kinetic_transport.py')
    runtime = {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
               'scipy': scipy.__version__, 'device': torch.cuda.get_device_name(0)}
    assert (torch.__version__, np.__version__, scipy.__version__) == ('2.11.0+cu128','2.4.4','1.17.1')
    report = {'schema_version': 1, 'kind': 'fixed_state_observational_gradients', 'optimizer_updates': 0,
              'candidate_outputs': False, 'history_updates': 0, 'panel_count_per_state': 32,
              'source_digest': original.inventory['source_digest'], 'runtime': runtime,
              'preflight_sha256': sha(CURRENT/'reports/forge/selective-history/PREFLIGHT.md'),
              'script_sha256': sha(__file__), 'loader_sha256': sha(original.__file__),
              'original_and_current_transport_sha256': transport_hash, 'states': []}
    cases = [(0,'evaluator/state.pt','gaussian1000'), (1,'evaluator/pre-shift-state.pt','gaussian4000'),
             (1,'evaluator/state.pt','gaussian6000'), (2,'provenance/provenance-state.pt','unequal1200')]
    for ai, relative, name in cases:
        print(json.dumps({'event':'start','name':name}), flush=True)
        row = {'name': name}
        try:
            path, packet, request, task_id = original.load_original(original.inventory['attempts'][ai],relative)
            context, trainer, spec = original.construct_and_restore(request,task_id,packet)
            # Original next-batch phase law: source switches before update4001.
            if task_id.startswith('gaussian') and trainer.completed_steps >= 4000: spec['means'] = [[3.]]
            parameters, roles = describe_parameters(trainer)
            initial_core = state_digest(core_state(context))
            panels = [observe(context,trainer,spec,task_id,i,parameters) for i in range(32)]
            assert state_digest(core_state(context)) == initial_core
            payload = {'schema_version':1, 'kind':report['kind'], 'name':name, 'source_digest':report['source_digest'],
                       'original_checkpoint_path':str(path), 'original_checkpoint_sha256':sha(path),
                       'original_state':packet, 'original_request':request, 'task_id':task_id,
                       'task':request['tasks'][task_id], 'recipe':trainer.recipe.to_dict(),
                       'target_spec':spec, 'parameter_map':roles, 'frozen_core_sha256':initial_core, 'panels':panels,
                       'temporal_training_history':False, 'optimizer_updates':0,
                       'original_and_current_transport_sha256':transport_hash}
            target = HERE/(name+'-frozen-gradients.pt'); torch.save(payload,target)
            row.update(status='PASS',path=str(target),sha256=sha(target),bytes=target.stat().st_size,
                       panels=32, original_checkpoint_path=str(path),original_checkpoint_sha256=sha(path),
                       frozen_core_sha256=initial_core, parameter_map=roles,
                       phase_target_means=spec['means'], original_completed_steps=trainer.completed_steps,
                       losses=[p['losses'] for p in panels],
                       real_batch_sha256=[p['real_sha256'] for p in panels],
                       first_streams_sha256=panels[0]['streams_before_sha256'],last_streams_sha256=panels[-1]['streams_after_sha256'])
        except Exception as exc:
            row.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        report['states'].append(row)
        report['process_wall_seconds']=time.monotonic()-START
        (HERE/'original-gradient-panels.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in {'losses','real_batch_sha256','parameter_map'}}),flush=True)
    print(json.dumps({'event':'finished','process_wall_seconds':report['process_wall_seconds'],
                      'statuses':[r['status'] for r in report['states']]}),flush=True)


if __name__ == '__main__': main()
