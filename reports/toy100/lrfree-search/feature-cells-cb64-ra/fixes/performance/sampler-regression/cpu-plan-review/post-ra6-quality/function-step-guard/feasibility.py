"""CPU fixed-state feasibility; never imports an oracle or starts training."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch import nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PACKAGE = ROOT / 'pkg-CB64-RA7'
sys.path.insert(0, str(PACKAGE))
from particlegan.feature_cells import FeatureCellSnapshot, Q, SCALE_PRIOR
from particlegan.birth_phase import learned_latent_features
from particlegan.k3p import K3PGeneratorAdam
from guard_design import LocalMotionChart, bounded_fifo_rows, interpolate_parameters

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(f'tensor:{v.dtype}:{tuple(v.shape)}:'.encode())
            h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in v:
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for child in v: visit(child)
        else:
            h.update(f'{type(v).__name__}:{v!r}'.encode())
    visit(value)
    return h.hexdigest()


def linear(weight, bias):
    module = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(module)
    module.in_features, module.out_features = weight.shape[1], weight.shape[0]
    module.weight = nn.Parameter(weight.clone())
    module.bias = nn.Parameter(bias.clone())
    return module


def mlp(weights):
    return nn.Sequential(linear(weights['0.weight'], weights['0.bias']), nn.LeakyReLU(.2),
        linear(weights['2.weight'], weights['2.bias']), nn.LeakyReLU(.2),
        linear(weights['4.weight'], weights['4.bias']))


def safe_capture(fn, modules):
    """Refuse a callback with hidden eval-time stochastic/stateful effects."""
    rng = torch.get_rng_state().clone()
    modes = [(module, module.training) for root in modules for module in root.modules()]
    buffers = [(value, value.clone()) for root in modules for value in root.buffers()]
    try:
        with torch.no_grad(): result = fn()
        mutated = not torch.equal(rng, torch.get_rng_state()) or any(
            not torch.equal(value, old) for value, old in buffers)
        if mutated:
            raise RuntimeError('unsupported stochastic or stateful eval callback')
        return result
    finally:
        torch.set_rng_state(rng)
        for value, old in buffers: value.copy_(old)
        for module, mode in modes: module.training = mode


def real_capture(critic, head, real):
    fired = []
    handle = head.register_forward_pre_hook(lambda module, inputs: fired.append(inputs[0]))
    modes = [(module, module.training) for module in critic.modules()]
    try:
        critic.eval(); critic(real)
        assert len(fired) == 1
        return fired[0].double()
    finally:
        handle.remove()
        for module, mode in modes: module.training = mode


class FunctionalG(nn.Module):
    def __init__(self, base, parameters):
        super().__init__()
        self.base, self.candidate = base, parameters
        self.candidate_buffers = {name: value.clone() for name, value in base.named_buffers()}
    def forward(self, latents):
        return torch.func.functional_call(self.base, (self.candidate, self.candidate_buffers),
            (latents,), strict=True)


def joint_proposal(state):
    saved = state['optimizers'][0]
    weights = state['models']['G']
    parameter_groups = []
    parameters = [nn.Parameter(value.clone()) for value in weights.values()]
    parameters += [nn.Parameter(state['models']['prior']['z'].clone()),
        nn.Parameter(state['output_noise']['log_sigma'].clone())]
    assert [group['params'] for group in saved['param_groups']] == [list(range(6)), [6], [7]]
    for group in saved['param_groups']:
        assert group['betas'][0] == 0. and group['weight_decay'] == 0.
        parameter_groups.append(dict(group, params=[parameters[pid] for pid in group['params']]))
    optimizer = K3PGeneratorAdam(parameter_groups, latent_table=parameters[6],
        latent_max_rate=state['recipe']['latent_damping_max_rate'])
    optimizer.load_state_dict(deepcopy(saved))
    for group in optimizer.param_groups:
        for parameter in group['params']:
            parameter.grad = optimizer.state[parameter]['exp_avg'].clone()
    optimizer.step()
    return parameters, optimizer


def mode_buffer_controls():
    g = nn.Sequential(linear(torch.eye(2), torch.zeros(2)), nn.BatchNorm1d(2), nn.Dropout(.5))
    d = nn.Sequential(linear(torch.eye(2), torch.zeros(2)), nn.LeakyReLU(.2),
        linear(torch.ones(1, 2), torch.zeros(1)))
    g.train(); d.train(); g[2].eval(); d[1].eval()
    bd = SimpleNamespace(_heads=[d[-1]], sample_shape=(2,))
    trainer = SimpleNamespace(D=d)
    callback = learned_latent_features(bd, trainer, g)
    modes = [m.training for model in (g, d) for m in model.modules()]
    before = tree_hash([g.state_dict(), d.state_dict()])
    rng = torch.get_rng_state().clone()
    x = torch.arange(16).reshape(8, 2).float() / 16
    first = safe_capture(lambda: callback(x), (g, d))
    second = safe_capture(lambda: callback(x), (g, d))
    assert torch.equal(first, second) and before == tree_hash([g.state_dict(), d.state_dict()])
    assert modes == [m.training for model in (g, d) for m in model.modules()]
    assert torch.equal(rng, torch.get_rng_state()) and not d[-1]._forward_pre_hooks
    counter = torch.zeros(())
    def malicious():
        counter.add_(1.)
        return torch.rand(4)
    dummy = nn.Module(); dummy.register_buffer('counter', counter)
    try:
        safe_capture(malicious, (dummy,))
    except RuntimeError as error:
        assert 'unsupported' in str(error)
    else: raise AssertionError('hidden stochastic callback accepted')
    assert float(counter) == 0. and torch.equal(rng, torch.get_rng_state())
    return dict(mixed_modes_restored=True, batchnorm_buffers_unchanged=True, dropout_no_rng=True,
        hooks_removed=True, stochastic_stateful_callback_rejected_and_restored=True)


def chart(state, critic, real_features, baseline):
    return LocalMotionChart(FeatureCellSnapshot.fit(real_features,
        generator=torch.Generator().set_state(state['cpu_rng']),
        cells=state['birth_death']['settings']['cells'], rank=state['birth_death']['settings']['rank'],
        chunk=state['birth_death']['settings']['chunk']), real_features, baseline, q=Q, prior=SCALE_PRIOR)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    output = HERE / 'receipt.json'
    if output.exists(): raise SystemExit('Preserve existing receipt; use a new attempt area.')
    ready = json.loads((ROOT / 'quality/ra7/READY.json').read_text())
    package_map = {str(p.relative_to(PACKAGE / 'particlegan')): sha(p)
        for p in sorted((PACKAGE / 'particlegan').glob('*.py'))}
    assert package_map == ready['package_source_sha256'] and len(package_map) == 29
    paths = [ROOT / f'validation-cb64-ra7/learned/training/toy/CB64-RA7/checkpoint-{step:04d}.pt'
        for step in (100, 2000)]
    source_paths = [*sorted((PACKAGE / 'particlegan').glob('*.py')), Path(__file__),
        HERE / 'guard_design.py', HERE / 'PROTOCOL.md', ROOT / 'quality/ra7/READY.json',
        ROOT / 'configs/overrides-CB64-RA7.json']
    before_files = {str(p): sha(p) for p in source_paths + paths}
    rng = torch.get_rng_state().clone()
    records = []
    controls = mode_buffer_controls()
    for path in paths:
        started = time.perf_counter()
        state = torch.load(path, map_location='cpu', weights_only=False)['trainer']
        before_state = tree_hash(state)
        g, d = mlp(state['models']['G']), mlp(state['models']['D'])
        # Preserve deliberately mixed submodule modes as well as tensor state.
        g.train(); d.train(); g[1].eval(); d[1].eval()
        before_models = tree_hash([g.state_dict(), d.state_dict()])
        modes = [m.training for model in (g, d) for m in model.modules()]
        bd = SimpleNamespace(_heads=[d[-1]], sample_shape=(2,))
        trainer = SimpleNamespace(D=d)
        row_ids = bounded_fifo_rows(state['birth_death']['fill'], state['birth_death']['cursor'])
        real = state['birth_death']['reservoir'][row_ids]
        assert len(real) <= 384 and len(torch.unique(row_ids)) == len(row_ids)
        real_features = safe_capture(lambda: real_capture(d, d[-1], real), (d,))
        z = state['models']['prior']['z']
        probe_rows = torch.div(torch.arange(128) * len(z), 128, rounding_mode='floor')
        probes = z[probe_rows].detach().clone()
        baseline = safe_capture(lambda: learned_latent_features(bd, trainer, g)(probes), (g, d))
        geometry = chart(state, d, real_features, baseline)
        parameters, optimizer = joint_proposal(state)
        reference_parameters, reference_optimizer = joint_proposal(state)
        assert tree_hash(optimizer.state_dict()) == tree_hash(reference_optimizer.state_dict())
        proposed = dict(zip(state['models']['G'], [p.detach().clone() for p in parameters[:6]]))
        assert tree_hash(proposed) == tree_hash(dict(zip(state['models']['G'],
            [p.detach().clone() for p in reference_parameters[:6]])))
        trial_weights = {}
        def candidate(fraction):
            weights = interpolate_parameters(state['models']['G'], proposed, fraction)
            trial_weights[fraction] = weights
            model = FunctionalG(g, weights)
            return safe_capture(lambda: learned_latent_features(bd, trainer, model)(probes), (model, d))
        fraction, attempts = geometry.choose(candidate)
        accepted_weights = interpolate_parameters(state['models']['G'], proposed, fraction)
        with torch.no_grad():
            for parameter, value in zip(parameters[:6], accepted_weights.values()): parameter.copy_(value)
        # Parameter acceptance cannot mutate any optimizer moment/history or the
        # full joint prior/sigma update. Exactly one update per private optimizer.
        assert tree_hash(optimizer.state_dict()) == tree_hash(reference_optimizer.state_dict())
        assert all(torch.equal(p, q) for p, q in zip(parameters[6:], reference_parameters[6:]))
        assert all(float(optimizer.state[p]['step']) == state['completed_steps'] + 1 for p in parameters)
        assert torch.equal(optimizer.latent_history, reference_optimizer.latent_history)
        nominal = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] *= fraction
        clock = dict(nominal=nominal, applied=optimizer.param_groups[0]['lr'],
            original_base=state['initial_lrs'][0][0], nominal_intrinsic=nominal / state['initial_lrs'][0][0],
            applied_intrinsic=optimizer.param_groups[0]['lr'] / state['initial_lrs'][0][0],
            prior_intrinsic=optimizer.param_groups[1]['lr'] / state['initial_lrs'][0][1],
            sigma_intrinsic=optimizer.param_groups[2]['lr'] / state['initial_lrs'][0][2])
        # Reconstruct from a serialized copy of the inputs, never the ephemeral
        # historical backend snapshot. Rebuild and candidate decision must match.
        stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
        replay_state = torch.load(stream, map_location='cpu', weights_only=False)
        replay = chart(replay_state, d, real_features.clone(), baseline.clone())
        replay_fraction, replay_attempts = replay.choose(candidate)
        assert replay_fraction == fraction and replay_attempts == attempts
        assert tree_hash(replay_state) == before_state
        ema_rate = min(1., state['lr_settle'][0][1]['s'] /
            (state['recipe']['serve_average'] * state['lr_settle'][0][1]['b']))
        original_ema = state['models']['ema_G']
        accepted_ema = {name: old.clone().mul_(1. - ema_rate).add_(accepted_weights[name], alpha=ema_rate)
            for name, old in original_ema.items()}
        ema_probes = state['models']['ema_prior']['z'][probe_rows].detach().clone()
        ema_old_model = FunctionalG(g, original_ema)
        ema_new_model = FunctionalG(g, accepted_ema)
        ema_before = safe_capture(lambda: learned_latent_features(bd, trainer, ema_old_model)(ema_probes),
            (ema_old_model, d))
        ema_after = safe_capture(lambda: learned_latent_features(bd, trainer, ema_new_model)(ema_probes),
            (ema_new_model, d))
        ema_projected_before = geometry.snapshot.transform(ema_before)
        ema_projected_after = geometry.snapshot.transform(ema_after)
        assert before_models == tree_hash([g.state_dict(), d.state_dict()])
        assert modes == [m.training for model in (g, d) for m in model.modules()]
        assert not d[-1]._forward_pre_hooks and before_state == tree_hash(state)
        records.append(dict(checkpoint=state['completed_steps'], reference_rows=len(real),
            reference_row_ids=row_ids.tolist(), probe_rows=probe_rows.tolist(),
            cells=geometry.snapshot.cells, rank=geometry.snapshot.rank,
            topology=geometry.snapshot.mass_topology, real_only_covered_mass=geometry.represented_real_mass,
            activation=geometry.active, accepted_fraction=fraction, trials=attempts, clock=clock,
            minimum_even_cell_rows=int(geometry.snapshot.reference_counts.min()),
            median_even_cell_rows=float(geometry.snapshot.reference_counts.double().median()),
            ema=dict(rate=ema_rate, parameter_update_after_acceptance=True,
                projected_motion_rms=float((ema_projected_after - ema_projected_before).square().sum(1).mean().sqrt())),
            joint_optimizer_moments_history_steps_prior_sigma_exact=True,
            mode_buffer_hooks_grads_and_rng_preserved=True, bounded_chart_resume_decision_exact=True,
            cpu_seconds=time.perf_counter() - started,
            fixed_chart_work=dict(geometry.snapshot.work)))
        print(json.dumps({k: records[-1][k] for k in ['checkpoint', 'activation', 'real_only_covered_mass',
            'accepted_fraction', 'trials', 'clock']}), flush=True)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert before_files == {str(p): sha(p) for p in source_paths + paths}
    late = records[-1]
    json_write(output, dict(status='PASS_FIXED_INPUT_RESERVED_DESIGN_PROOF', utc=datetime.now(timezone.utc).isoformat(),
        design_status='RESERVED_NOT_INSTALLED_NOT_QUALITY_QUALIFIED', records=records, controls=controls,
        late_recorded_proposal_constrained=late['activation'] and late['accepted_fraction'] < 1.,
        early_original_motion_retained=not records[0]['activation'] and records[0]['accepted_fraction'] == 1.,
        package_sha256=ready['package_sha256'], source_and_input_sha256=before_files,
        sources_inputs_checkpoint_tensors_unchanged=True, cpu_rng_unchanged=True, cuda_initialized=False,
        new_seeds=0, new_training_steps=0, private_repeated_gradient_joint_optimizer_calls=4,
        whole_table_generator_forwards=0, oracle_imports_or_decisions=False, production_changed=False,
        limits=['Recorded saved gradients are repeated, not actual next gradients or historical actions.',
            'Current-D bounded CPU chart is a new measurement, not the saved GPU action partition.',
            'Small-cell covariance and coarse topology may fail to resolve rare/thin/native grid components.',
            'Q-budget and coverage activation are empirical engineering guards without a null or repeated-adaptive guarantee.',
            'Up to four trial G+D forwards plus one baseline and bounded real D forward add material image cost; O(P_G) parameter backup and delta storage are required.',
            'Original fitter/topology cold scalar synchronization cost has not been profiled; this is not a scalable production implementation.',
            'No production schema/policy/fallback integration, same-law training continuation or CUDA parity was tested.',
            'A one-step guard does not bound accumulated many-step motion or ensure emitted quality.']))
    print(json.dumps(dict(status='PASS', output=str(output), records=len(records))), flush=True)
