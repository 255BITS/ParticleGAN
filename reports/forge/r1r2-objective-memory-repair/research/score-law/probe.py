"""Read-only score-law diagnosis on exact saved AMSGrad/m110 Ring evidence.

No model construction, autograd, sampler, optimizer, training or gate evaluation.
Target centers are explicit diagnostic fixtures, never historical training pairs.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import torch

from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'reports/forge/r1r2-objective-memory-repair'
EXECUTION = '2b0ad9ec7c5b2c1e91ab0d70751da0b7e36a7b42'
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(value):
    return dict(min=float(value.min()), mean=float(value.mean()), max=float(value.max()),
                population_std=float(value.std(unbiased=False)))


def critic(points, saved):
    """Exact source Fourier/LeakyReLU equations and analytic input derivative."""
    frequencies = saved['freqs']
    assert len(frequencies) == 2
    xf = points.unsqueeze(-1) * frequencies
    hidden = torch.cat([points, torch.sin(xf).flatten(1), torch.cos(xf).flatten(1)], 1)
    jacobian = points.new_zeros(len(points), 10, 2)
    jacobian[:, :2] = torch.eye(2, dtype=points.dtype)
    for dimension in range(2):
        for frequency in range(2):
            jacobian[:, 2 + dimension * 2 + frequency, dimension] = (
                frequencies[frequency] * torch.cos(xf[:, dimension, frequency]))
            jacobian[:, 6 + dimension * 2 + frequency, dimension] = (
                -frequencies[frequency] * torch.sin(xf[:, dimension, frequency]))
    for layer in (0, 2, 4):
        weight, bias = saved[f'net.{layer}.weight'], saved[f'net.{layer}.bias']
        hidden = hidden @ weight.T + bias
        jacobian = torch.einsum('oi,nij->noj', weight, jacobian)
        if layer != 4:
            jacobian = jacobian * torch.where(hidden >= 0, 1., .2).unsqueeze(-1)
            hidden = torch.where(hidden >= 0, hidden, .2 * hidden)
    return hidden[:, 0], jacobian[:, 0]


def shape_proxy(points, labels, radial, tangent, gradients, weights):
    """Per-component centered variance drift under an input-space force only."""
    radial_drift, tangent_drift = [], []
    force = weights[:, None] * gradients
    for component in range(16):
        mask = labels == component
        centered = points[mask] - points[mask].mean(0)
        for axis, result in ((radial, radial_drift), (tangent, tangent_drift)):
            result.append(float(2 * ((centered * axis[component]).sum(1)
                                   * (force[mask] * axis[component]).sum(1)).mean()))
    return dict(radial_variance_drift=radial_drift, tangent_variance_drift=tangent_drift,
                mean_radial_variance_drift=sum(radial_drift) / 16,
                mean_tangent_variance_drift=sum(tangent_drift) / 16,
                radial_expanding_components=sum(v > 0 for v in radial_drift),
                tangent_contracting_components=sum(v < 0 for v in tangent_drift))


def main():
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    audit = read(OUT / 'density-audit.json')
    publication_path = OUT / 'publication/results.json'
    assert audit['execution_commit'] == EXECUTION and audit['source_digest'] == SOURCE
    assert audit['publication_sha256'] == sha(publication_path)
    published = next(row for row in read(publication_path)['certified_results']
                     if row['role'] == 'ams_m110' and row['task_id'] == 'ring16_acquisition')
    prior_audit = next(row for row in audit['rings'] if row['role'] == 'ams_m110')
    attempt = Path(published['checkpoint']['artifact_root']).parent
    request = read(attempt / 'request.json')['request']
    checkpoint_path = attempt / 'provenance/provenance-state.pt'
    panels_path = attempt / 'observed-samples.pt'
    assert sha(checkpoint_path) == published['checkpoint']['sha256']
    assert sha(panels_path) == prior_audit['artifacts_sha256']['observed-samples.pt']
    assert published['candidate_revision'] == request['candidate_revision'] == prior_audit['candidate_revision']
    assert request['source']['digest'] == SOURCE
    assert all(sha(ROOT / path) == expected for path, expected in request['source']['files'].items())
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert state_digest(state) == published['checkpoint']['state_sha256']
    assert all(state_digest(state['streams']['states'][name]) == digest
               for name, digest in prior_audit['named_stream_state_sha256'].items())
    trainer = state['trainer']
    recipe = asdict(Recipe(**trainer['recipe']))
    proposed = asdict(Recipe(**(trainer['recipe'] | {'loss': 'non_saturating'})))
    assert [name for name in recipe if recipe[name] != proposed[name]] == ['loss']
    assert recipe['loss'] == 'relativistic' and recipe['reg_arm'] == 'a_r1r2'
    assert recipe['reg_coeff'] == .1 and recipe['reg_every'] == 1 and recipe['amsgrad']
    assert recipe['betas'] == (.5, .99) and recipe['prior_betas'] == (0., .99)
    assert trainer['completed_steps'] == 1600
    panels = torch.load(panels_path, map_location='cpu', weights_only=True)
    assert len(panels) == 96 and panels[-1]['step'] == 1600
    points = panels[-1]['samples'].double()
    models = trainer['models']
    saved = {name: value.double() for name, value in models['D'].items()}
    definition = request['tasks']['ring16_acquisition']['execution']['host_definition']
    centers = points.new_tensor(definition['means'])
    fake, gradients = critic(points, saved)
    real, _ = critic(centers, saved)
    labels = torch.cdist(points, centers).argmin(1)
    radial = centers / centers.norm(dim=1, keepdim=True)
    tangent = torch.stack([-radial[:, 1], radial[:, 0]], 1)
    paired = torch.sigmoid(real[None, :] - fake[:, None])
    weights = {'paired_uniform_center_fixture': paired.mean(1),
               'non_saturating': torch.sigmoid(-fake), 'least_squares': 1 - fake}
    # Numerical derivative check uses deterministic +/-1e-6 offsets of 32
    # saved points; these are separate probes, never added to metric panels.
    epsilon, errors = 1e-6, []
    for coordinate in range(2):
        offset = points[:32].new_zeros(32, 2)
        offset[:, coordinate] = epsilon
        plus, _ = critic(points[:32] + offset, saved)
        minus, _ = critic(points[:32] - offset, saved)
        errors.append(float(((plus - minus) / (2 * epsilon) - gradients[:32, coordinate]).abs().max()))
    assert max(errors) < 1e-7
    ratios = weights['non_saturating'] / weights['paired_uniform_center_fixture']
    moment_norms = []
    for optimizer, roles in zip(trainer['optimizers'], (('G', 'prior'), ('D',)), strict=True):
        for group, role in zip(optimizer['param_groups'], roles, strict=True):
            moment_norms.append(dict(role=role, betas=list(group['betas']),
                norm=sum(float(optimizer['state'][key]['exp_avg'].double().square().sum())
                         for key in group['params']) ** .5,
                instantaneous_gradient_recoverable=group['betas'][0] == 0))
    critic_optimizer = trainer['optimizers'][1]
    bias_key = critic_optimizer['param_groups'][0]['params'][-1]
    bias_state = critic_optimizer['state'][bias_key]
    assert bias_state['exp_avg'].shape == saved['net.4.bias'].shape == (1,)
    assert float(bias_state['exp_avg'].abs().max()) == 0
    assert float(bias_state['exp_avg_sq'].abs().max()) == 0
    assert float(bias_state['max_exp_avg_sq'].abs().max()) == 0
    fake_probability, real_probability = torch.sigmoid(fake), torch.sigmoid(real)
    absolute_score_anchoring = dict(
        paired_common_shift_gradient=0., paired_common_shift_curvature=0.,
        saved_critic_output_bias_first_second_and_max_second_moments_all_zero=True,
        non_saturating_center_fixture_common_shift_gradient=float(
            fake_probability.mean() - (1 - real_probability).mean()),
        non_saturating_center_fixture_common_shift_curvature=float(
            (fake_probability * (1 - fake_probability)).mean()
            + (real_probability * (1 - real_probability)).mean()),
        interpretation='A distinct restoring curvature is present under non_saturating; '
            'this does not establish a harmful score-offset drift or a future density repair.')
    assert python_rng == random.getstate() and torch.equal(torch_rng, torch.get_rng_state())
    source_paths = ('particlegan/gan_loss.py', 'particlegan/training.py', 'particlegan/recipes.py',
                    'particlegan/grad_regularizers.py', 'lib/toy_models.py',
                    'experiments/forge/behavior_adapters.py', 'benchmarks/locked_shared/two_pole.py',
                    'benchmarks/locked_shared/hosts/unused_token_hold.py',
                    'benchmarks/locked_shared/hosts/ae_gan_hold.py',
                    'benchmarks/toy_audit/api_images.py', 'experiments/forge/clockfree.py')
    result = dict(schema_version=1, scope='saved_endpoint_score_law_diagnosis_not_training_or_qualification',
        decision='reject_endpoint_generator_starvation_repair; one_native_non_saturating_critic_anchoring_contrast_is_defensible_at_low_confidence',
        proposed_recipe_delta=dict(field='loss', before='relativistic', after='non_saturating',
            complete_recipe_changed_fields=['loss'], execution='prospective_shared_initialization_not_checkpoint_continuation'),
        execution_commit=EXECUTION, source_digest=SOURCE, candidate_id=published['candidate_id'],
        candidate_revision=published['candidate_revision'], attempt_id=published['attempt_id'],
        publication_sha256=sha(publication_path), density_audit_sha256=sha(OUT / 'density-audit.json'),
        checkpoint_sha256=sha(checkpoint_path), complete_state_sha256=state_digest(state),
        saved_panels_sha256=sha(panels_path), source_sha256={path: sha(ROOT / path) for path in source_paths},
        reproduction_source_sha256=sha(Path(__file__)), sample_count=len(points), observation_step=1600,
        fixture='All 16 declared target centers, uniformly averaged for paired weights; no sampled target data.',
        saved_fake_scores=summary(fake), target_center_fixture_scores=summary(real),
        saved_critic_output_bias=float(saved['net.4.bias'][0]),
        analytic_critic_input_gradient_norm=summary(gradients.norm(dim=1)),
        all_center_pair_generator_weights=summary(paired.flatten()),
        generator_weights={name: dict(**summary(weight), negative_count=int((weight < 0).sum()),
            input_force_shape_proxy=shape_proxy(points, labels, radial, tangent, gradients, weight))
            for name, weight in weights.items()},
        non_saturating_to_paired_fixture_weight_ratio=summary(ratios),
        non_saturating_paired_fixture_weight_cosine=float(torch.nn.functional.cosine_similarity(
            weights['non_saturating'], weights['paired_uniform_center_fixture'], dim=0)),
        saved_first_moments=moment_norms, analytic_derivative_finite_difference_max_errors=errors,
        absolute_score_anchoring=absolute_score_anchoring,
        numerical_diagnostic_dtype='float64 using exact saved float32 tensors converted without initialization',
        historical_proof_limit='No saved contemporaneous training scores/real-fake pairings or instantaneous G/D gradients. '
            'Endpoint EMA moments average history. Input forces exclude the generator Jacobian, Adam preconditioning '
            'and later critic refitting; they do not predict trained parameter updates or qualification.',
        new_draws=0, new_updates=0, new_grades=0, rng_states_unchanged=True)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
