"""Deterministic first-order energy-distance prerequisite, never a training run.

Only one proposed statistic: signed Euclidean energy U, with no bandwidth,
clamp, normalization or selected coefficient. Squared distances occur solely
in the explicitly labeled degeneracy control. No models, recipes, optimizers,
samples, task evaluators or RNG draws are permitted.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'reports/forge/r1r2-weak-penalty-repair'
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def euclidean(x, y):
    """Native direct-distance kernel, with an explicit detached zero at ties."""
    # Avoid both n*m*d broadcasting and cancellation in the matrix-multiply
    # distance formula. The tie branch changes no value or positive distance.
    distance = torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    return torch.where(distance > 0, distance, distance.detach())


def energy_u(fake, real):
    """2 cross mean minus two ordered off-diagonal within-sample means."""
    if (fake.ndim < 2 or real.ndim < 2 or len(fake) < 2 or len(real) < 2
            or fake.shape[1:] != real.shape[1:] or not fake.is_floating_point()
            or fake.dtype != real.dtype or fake.device != real.device):
        raise ValueError('matching floating sample spaces, at least two rows each')
    x, y = fake.flatten(1), real.detach().flatten(1)
    cross, ff = euclidean(x, y), euclidean(x, x)
    with torch.no_grad():
        rr = euclidean(y, y)
    n, m = len(x), len(y)
    return (2 * cross.mean() - (ff.sum() - ff.diagonal().sum()) / (n * (n - 1))
            - (rr.sum() - rr.diagonal().sum()) / (m * (m - 1)))


@contextmanager
def prohibit_live_execution():
    def forbidden(*_args, **_kwargs):
        raise RuntimeError('Prerequisite forbids models, recipes, optimizers and draws')
    methods = [(torch.nn.Module, '__init__'), (torch.nn.Module, '__call__'),
               (torch.optim.Optimizer, '__init__'), (Recipe, '__init__')]
    methods += [(torch, name) for name in ('rand', 'randn', 'randint', 'randperm',
                'rand_like', 'randn_like', 'normal', 'multinomial')]
    methods += [(random, name) for name in ('random', 'randint', 'randrange', 'choice', 'gauss')]
    methods += [(np.random, name) for name in ('random', 'rand', 'randn', 'randint',
                'normal', 'choice', 'default_rng')]
    with ExitStack() as stack:
        for owner, name in methods:
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def finite_difference():
    x = torch.tensor([[-.73, .21], [.19, -.61], [.82, .48], [-.14, .92]], dtype=torch.float64,
                     requires_grad=True)
    y = torch.tensor([[-.42, -.83], [.64, -.19], [.08, .57]], dtype=torch.float64,
                     requires_grad=True)
    loss = energy_u(x, y)
    gradient, = torch.autograd.grad(loss, x)
    errors, epsilon = [], 1e-6
    for index in range(x.numel()):
        values = []
        for sign in (-1, 1):
            shifted = x.detach().clone()
            shifted.flatten()[index] += sign * epsilon
            values.append(float(energy_u(shifted, y)))
        errors.append(abs((values[1] - values[0]) / (2 * epsilon) - float(gradient.flatten()[index])))
    assert max(errors) < 2e-9 and y.grad is None
    # Check target detachment with an explicitly requested unused gradient.
    _, target_gradient = torch.autograd.grad(energy_u(x, y), (x, y), allow_unused=True)
    assert target_gradient is None
    permutation = torch.tensor([2, 0, 3, 1])
    reordered = x.detach()[permutation].requires_grad_(True)
    reordered_value = energy_u(reordered, y.detach()[[1, 2, 0]])
    reordered_gradient, = torch.autograd.grad(reordered_value, reordered)
    assert abs(float(reordered_value.detach()-loss.detach())) < 1e-15
    assert torch.allclose(reordered_gradient, gradient[permutation], atol=1e-15, rtol=0)
    rotation = x.new_tensor([[0., -1.], [1., 0.]])
    translation = x.new_tensor([3., -2.])
    assert abs(float((energy_u(x @ rotation+translation, y @ rotation+translation)-loss).detach())) < 1e-15
    assert abs(float((energy_u(3*x,3*y)-3*loss).detach())) < 1e-15
    return dict(coordinates=len(errors), epsilon=epsilon, max_absolute_error=max(errors),
                absolute_error_bound=2e-9, targets_detached=True, unequal_batch_sizes_supported=True,
                permutation_value_and_gradient_equivariance=True,rigid_motion_invariance=True,
                positive_linear_scale_homogeneity=True)


def fixtures():
    dtype = torch.float64
    identical = torch.tensor([[-1.], [1.]] * 16, dtype=dtype)
    signed = float(energy_u(identical, identical))
    assert signed < 0  # Preserve signed U; do not silently rectify it.
    ties = torch.zeros(8, 2, dtype=dtype, requires_grad=True)
    self_gradient, = torch.autograd.grad(euclidean(ties, ties).sum(), ties)
    assert torch.equal(self_gradient, torch.zeros_like(self_gradient))

    # The actual pole host has twelve zero coordinates and balanced fixed spread.
    offsets = torch.linspace(-.05, .05, 6, dtype=dtype)
    poles = torch.cat((-1 + offsets, 1 + offsets))[:, None]
    zero = torch.zeros(12, 1, dtype=dtype, requires_grad=True)
    zero_gradient, = torch.autograd.grad(energy_u(zero, poles), zero)
    assert float(zero_gradient.abs().max()) < 1e-15

    # Parameter tensors only: original shared + slot concept output repeated 8 times.
    shared = torch.zeros(2, dtype=dtype, requires_grad=True)
    slots = torch.zeros(2, 2, dtype=dtype, requires_grad=True)
    concept = (shared + slots[1]).expand(8, 2)
    real_concept = torch.tensor([0., 1.], dtype=dtype).expand(8, 2)
    gs, gt = torch.autograd.grad(energy_u(concept, real_concept), (shared, slots))
    assert torch.equal(gt[0], torch.zeros(2, dtype=dtype))
    assert torch.allclose(gs, torch.tensor([0., -2.], dtype=dtype))

    reference = torch.tensor([[-1.], [1.]] * 32, dtype=dtype)
    widths = []
    for width in (.3, 2.):
        scale = torch.tensor(width, dtype=dtype, requires_grad=True)
        x = reference * scale
        value = energy_u(x, reference)
        derivative, = torch.autograd.grad(value, scale)
        widths.append(dict(scale=width, signed_energy_u=float(value.detach()),
                           d_loss_d_scale=float(derivative)))
    assert widths[0]['d_loss_d_scale'] < 0 < widths[1]['d_loss_d_scale']
    wide = 2 * reference
    squared_v = (2 * (wide[:, None] - reference[None]).square().sum(-1).mean()
                 - (wide[:, None] - wide[None]).square().sum(-1).mean()
                 - (reference[:, None] - reference[None]).square().sum(-1).mean())
    assert float(squared_v) == 0 and float(energy_u(wide, reference)) > 0

    angles = torch.arange(8, dtype=dtype) * (2 * torch.pi / 8)
    circle = torch.stack((angles.cos(), angles.sin()), 1).repeat(8, 1)
    scales = torch.tensor([.5, 2.], dtype=dtype, requires_grad=True)
    value = energy_u(circle * scales, circle)
    directional, = torch.autograd.grad(value, scales)
    assert float(directional[0]) < 0 < float(directional[1])
    return dict(signed_equal_empirical_samples_u=signed, clamped=False,
        duplicate_rows=dict(zero_subgradient_max=float(self_gradient.abs().max())),
        symmetric_zero_pole=dict(particles=12, target='actual fixed balanced +/-1 with +/- .05 spread',
            energy_gradient_max=float(zero_gradient.abs().max()),
            limit='This term alone cannot break the exact zero/symmetric state.'),
        duplicate_token=dict(rows=8, shared_gradient=gs.tolist(), slot_gradients=gt.tolist(),
            limit='Unused slot has no direct gradient; shared parameters can still move unused output. Keep hold.'),
        width_fixtures=widths, squared_distance_control=dict(equal_means=True,
            different_widths=True, squared_population_V_on_empirical_laws=float(squared_v),
            nonsquared_signed_u=float(energy_u(wide, reference)),
            limit='Squared energy population discrepancy only identifies means; V is a labeled control, not selected U.'),
        anisotropic_fixture=dict(radial_scale=.5, tangent_scale=2.,
            d_loss_d_scales=directional.tolist(),
            limit='Deterministic circular atoms, not Gaussian modes or evidence of optimizer/training repair.'))


def word_limit():
    chars, words, length = 'abcdefghijklmnopqrstuvwxyz_ ', ('apple', 'grape', 'lemon', 'melon', 'berry'), 6
    bank = torch.zeros(5, len(chars), length, dtype=torch.float64)
    for row, word in enumerate(words):
        for position, char in enumerate(word.ljust(length, '_')):
            bank[row, chars.index(char), position] = 1
    target = bank.repeat(4, 1, 1)
    records = []
    for confidence in (2., 8., 20., 40., 80.):
        logits = (bank[:1].repeat(20, 1, 1) * confidence).requires_grad_(True)
        probabilities = logits.softmax(1)
        value = energy_u(probabilities, target)
        output_gradient, logit_gradient = torch.autograd.grad(value, (probabilities, logits))
        records.append(dict(logit_gap=confidence, energy_u=float(value.detach()),
            output_gradient_norm=float(output_gradient.norm()), logit_gradient_norm=float(logit_gradient.norm())))
    assert records[-1]['logit_gradient_norm'] < records[0]['logit_gradient_norm'] * 1e-12
    return dict(actual_host_output='per-character softmax, flattened 6x28 = 168; latent coordinates excluded',
        fixture='20 deterministic repeated canonical rows, collapsed apple logits; not actual sampled words',
        records=records, limit='Bounded output-space directions need not rescue saturated logits or inverse E reconstruction.')


def saved_ring():
    publication_path = OUT / 'publication/results.json'
    audit_path = OUT / 'density-audit.json'
    publication, audit = read(publication_path), read(audit_path)
    assert publication['complete'] and publication['source_digest'] == audit['source_digest'] == SOURCE
    assert audit['publication_sha256'] == sha(publication_path)
    records, files = [], None
    for role in ('non_saturating', 'weak_gamma'):
        row = next(r for r in publication['certified_results']
                   if r['role'] == role and r['task_id'] == 'ring16_acquisition')
        certified = audit['cached_parents'][role] if role == 'non_saturating' else audit['rings'][0]
        attempt = Path(row['checkpoint']['artifact_root']).parent
        request_path, panel_path, state_path = (attempt / name for name in
            ('request.json', 'observed-samples.pt', 'provenance/provenance-state.pt'))
        request = read(request_path)['request']
        assert all(sha(attempt / name) == digest for name, digest in certified['artifacts_sha256'].items())
        assert sha(state_path) == row['checkpoint']['sha256']
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state_digest(state) == row['checkpoint']['state_sha256']
        if files is None:
            files = request['source']['files']
        assert files == request['source']['files'] and len(files) == 1214
        assert all(sha(ROOT / path) == digest for path, digest in files.items())
        panels = torch.load(panel_path, map_location='cpu', weights_only=True)
        assert len(panels) == 96 and panels[-1]['step'] == 1600
        definition = request['tasks']['ring16_acquisition']['execution']['host_definition']
        centers = torch.tensor(definition['means'], dtype=torch.float64)
        angles = torch.arange(8, dtype=torch.float64) * (2 * torch.pi / 8)
        offsets = .1 * 2**.5 * torch.stack((angles.cos(), angles.sin()), 1)
        target = (centers[:, None] + offsets[None]).reshape(128, 2)
        # Existing evaluation outputs only; no resampling or forward computation.
        x = panels[-1]['samples'][:128].double().detach().requires_grad_(True)
        value = energy_u(x, target)
        gradient, = torch.autograd.grad(value, x)
        labels = torch.cdist(x.detach(), centers).argmin(1)
        radial = centers / centers.norm(dim=1, keepdim=True)
        tangent = torch.stack((-radial[:, 1], radial[:, 0]), 1)
        drift = []
        for component in range(16):
            selected = labels == component
            if int(selected.sum()) < 2:
                continue
            centered = x.detach()[selected] - x.detach()[selected].mean(0)
            force = -gradient[selected]
            derivatives = [float(2 * ((centered * axis[component]).sum(1)
                * (force * axis[component]).sum(1)).mean()) for axis in (radial, tangent)]
            drift.append(dict(component=component, samples=int(selected.sum()),
                radial_variance_derivative=derivatives[0], tangent_variance_derivative=derivatives[1]))
        records.append(dict(role=role, attempt_id=row['attempt_id'], candidate_revision=row['candidate_revision'],
            execution_origin=request['source']['origin_commit'], publication_gate_status=row['gate_status'],
            original_gate_metrics=row['metrics'], artifacts_sha256={
                'request':sha(request_path), 'panels':sha(panel_path), 'checkpoint':sha(state_path)},
            checkpoint_state_sha256=row['checkpoint']['state_sha256'], saved_panels=96,
            fixture='first128 ordered endpoint evaluation outputs vs128 deterministic circle atoms at declared centers',
            target_is_actual_gaussian_or_seen_training_batch=False, signed_energy_u=float(value.detach()),
            output_gradient_norm=float(gradient.norm()), available_drift_components=len(drift),
            radial_expanding_components=sum(r['radial_variance_derivative'] > 0 for r in drift),
            tangent_contracting_components=sum(r['tangent_variance_derivative'] < 0 for r in drift),
            mean_radial_variance_derivative=sum(r['radial_variance_derivative'] for r in drift)/len(drift),
            mean_tangent_variance_derivative=sum(r['tangent_variance_derivative'] for r in drift)/len(drift)))
    return dict(publication_sha256=sha(publication_path), density_audit_sha256=sha(audit_path),
        records=records, limit='Ordered prefixes and deterministic target atoms are diagnostics; output-space force is not G/prior/AMS update.'), files


def resource_check():
    assert torch.cuda.is_available()
    device, records = torch.device('cuda:0'), []
    duplicates = torch.zeros(8, 2, device=device, requires_grad=True)
    duplicate_gradient, = torch.autograd.grad(energy_u(duplicates,duplicates.detach()),duplicates)
    assert torch.equal(duplicate_gradient,torch.zeros_like(duplicate_gradient))
    offsets = torch.linspace(-.05,.05,6,device=device)
    target_poles = torch.cat((-1+offsets,1+offsets))[:,None]
    zero_poles = torch.zeros(12,1,device=device,requires_grad=True)
    pole_gradient, = torch.autograd.grad(energy_u(zero_poles,target_poles),zero_poles)
    assert float(pole_gradient.abs().max()) < 1e-7
    atomic = dict(exact_duplicate_gradient_max=float(duplicate_gradient.abs().max()),
        symmetric_zero_pole_gradient_max=float(pole_gradient.abs().max()),
        symmetric_zero_pole_rows_have_identical_gradient=bool(torch.equal(
            pole_gradient,pole_gradient[:1].expand_as(pole_gradient))),dtype='float32')
    del duplicates,duplicate_gradient,offsets,target_poles,zero_poles,pole_gradient
    shapes = [('pole',12,1), ('token',8,2), ('clock',8,2), ('ae',64,2),
              ('gaussian',128,1), ('ring',128,2), ('words',256,168)]
    for name, n, d in shapes:
        flat = torch.arange(n*d, dtype=torch.float32, device=device).reshape(n,d)
        x = torch.sin(flat * .017).requires_grad_(True)
        y = torch.cos(flat * .013 + .23)
        for _ in range(2):
            torch.autograd.grad(energy_u(x,y),x)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            value = energy_u(x,y)
            gradient, = torch.autograd.grad(value,x)
            torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter()-start)*1000)
            assert torch.isfinite(value) and torch.isfinite(gradient).all()
            del value,gradient
        peak = torch.cuda.max_memory_allocated(device)
        records.append(dict(host=name, fake_shape=[n,d], real_shape=[n,d], dtype='float32',
            warmups=2,repetitions=5,forward_and_fake_backward_ms_median=float(np.median(elapsed)),
            elapsed_ms_min=min(elapsed),elapsed_ms_max=max(elapsed),
            peak_allocated_bytes=peak,incremental_peak_allocated_bytes=peak-baseline))
        del flat,x,y
    return dict(device=str(device),gpu_name=torch.cuda.get_device_name(device),torch_version=torch.__version__,
        atomic_GPU_checks=atomic,records=records,
        scope='Pure deterministic synthetic array forward+input backward, includes real/real constant; no model training or performance guarantee.')


def main():
    torch.set_num_threads(1)
    # Initialize CUDA contexts before the snapshots; no seeds or draws are used.
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    before = random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    queue = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue/queue/state.json')
    queue_before = sha(queue)
    with prohibit_live_execution():
        checked = finite_difference()
        controls = fixtures()
        words = word_limit()
        ring,files = saved_ring()
        resources = resource_check()
    after = random.getstate(),np.random.get_state(),torch.get_rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0]
    assert np.array_equal(before[1][1],after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2],after[2])
    assert all(torch.equal(a,b) for a,b in zip(cuda_before,torch.cuda.get_rng_state_all(),strict=True))
    assert sha(queue) == queue_before and all(sha(ROOT/path)==digest for path,digest in files.items())
    print(json.dumps(dict(schema_version=1,evidence_type='deterministic_software_prerequisite',
        qualification_input=False, selected_candidate=None, selected_coefficient=None, selected_scale=None,
        statistic='2 mean Euclidean cross - ordered off-diagonal fake/fake mean - ordered off-diagonal real/real mean',
        signed_U_unclamped=True,real_targets_detached=True,zero_subgradient_at_duplicate_rows=True,
        finite_difference=checked,fixtures=controls,word_softmax_limit=words,saved_ring=ring,
        resources=resources,source_digest=SOURCE,source_files_verified=len(files),
        scientific_source_unchanged=True,queue_snapshot_sha256=queue_before,queue_unchanged=True,
        python_numpy_torch_cpu_and_all_cuda_rng_states_unchanged=True,
        model_recipe_optimizer_constructions=0,optimizer_updates_added=0,new_sampling_draws=0,
        probe_sha256=sha(Path(__file__)),
        limits=['Finite signed U can be negative and has minibatch variance; no rectification or bandwidth selection.',
            'Under independent iid real/fake rows U is unbiased for population energy, not for a gate or optimizer trajectory.',
            'Raw Euclidean energy has units of output length; global coefficient and coordinate scale remain unselected.',
            'Repeated fixture rows and saved evaluation prefixes are not independent training minibatches.',
            'Word marginal energy excludes latent coordinates and supplies no paired inverse-E reconstruction objective.',
            'Only first derivatives are checked; this probe establishes no public API binding or higher-order contract.']),
        indent=2,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()
