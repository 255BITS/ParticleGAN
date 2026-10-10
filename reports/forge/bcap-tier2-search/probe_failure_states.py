"""Frozen BCAP checkpoint diagnostics: no optimizer steps or random sampling.

Saved public-API models are restored for derivatives. Deterministic latent
censuses and cubature are separate diagnostics, never qualification samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import ndtri
import torch
from torch.func import functional_call, jvp

from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import ParticleRegularizer
from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradientPenalty
from particlegan.optim.dualnorm import polar_factor
from benchmarks.locked_shared import trajectory
from benchmarks.locked_shared.hosts import residual_student
from benchmarks.toy100.problems import evaluation_geometry
from experiments.forge.state import state_digest


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Changed scientific input: {path}")
    return actual


def stats(values):
    x = torch.as_tensor(values).detach().double().flatten().numpy()
    if not len(x):
        return None
    return dict(mean=float(x.mean()), min=float(x.min()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), max=float(x.max()))


def cosine(a, b):
    a, b = a.detach().flatten(), b.detach().flatten()
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 0 else None


def grads(loss, parameters):
    return tuple(torch.zeros_like(p) if g is None else g.detach()
                 for p, g in zip(parameters, torch.autograd.grad(
                     loss, parameters, retain_graph=True, allow_unused=True)))


def flatten(values):
    return torch.cat([v.flatten() for v in values])


def direction(gradient, rate, *, prior=False):
    # Public FP32 polar cutoff, with FP64 diagnostic gradients rounded to FP32.
    if prior:
        norm = gradient.norm(dim=1, keepdim=True)
        return -rate * gradient / torch.hypot(norm, torch.full_like(norm, .001))
    if gradient.ndim == 2:
        if gradient.norm() < 1e-8:
            return torch.zeros_like(gradient)
        return -rate * math.sqrt(max(1., gradient.shape[0] / gradient.shape[1])) * \
            polar_factor(gradient.float(), smoothing=.001).double()
    norm = gradient.norm()
    return -rate * gradient / torch.hypot(norm, norm.new_tensor(.001))


def spectral_summary(module, gradients, rate):
    rows = []
    for (name, parameter), gradient in zip(module.named_parameters(), gradients):
        singular = torch.linalg.svdvals(gradient) if gradient.ndim == 2 else gradient.norm()[None]
        normalized = direction(gradient, rate)
        rows.append(dict(parameter=name, shape=list(parameter.shape),
                         gradient_norm=float(gradient.norm()), singular_values=stats(singular),
                         fraction_singular_above_smoothing=float((singular > .001).double().mean()),
                         fraction_singular_at_least_10x_smoothing=float((singular > .01).double().mean()),
                         predicted_step_norm=float(normalized.norm()),
                         predicted_step_over_parameter_norm=float(normalized.norm() / parameter.detach().norm())
                         if parameter.norm() > 0 else None))
    return rows


def restore(models, conditional=None):
    with torch.random.fork_rng(devices=[]):
        if conditional:
            g_state, d_state = models['generator'], models['discriminator']
            constructor = residual_student.ResidualHead if conditional == 'residual_student' else trajectory._Generator
            g = constructor(16, 4, 64) if conditional == 'residual_student' else constructor(16, 4, 16, 64)
            d = trajectory._Critic(32, 64)
        else:
            g_state, d_state = models['G'], models['D']
            width = g_state['net.0.weight'].shape[0]
            hidden = sum(k.endswith('.weight') for k in g_state) - 1
            g = SimpleMLPGenerator(g_state['net.0.weight'].shape[1], width, hidden,
                                   g_state[f'net.{2 * hidden}.weight'].shape[0])
            d = SimpleMLPDiscriminator(g_state[f'net.{2 * hidden}.weight'].shape[0],
                                       d_state['net.0.weight'].shape[0], hidden, len(d_state['freqs']))
    g.load_state_dict(g_state)
    d.load_state_dict(d_state)
    for module in (g, d):
        module.double().eval()
        for child in module.modules():
            if isinstance(child, torch.nn.LeakyReLU):
                child.inplace = False
    return g, d


def critic_geometry(critic, fake, real, correction, *, conditioning=None):
    x = fake.detach().clone().requires_grad_(True)
    score = critic(x) if conditioning is None else critic(conditioning, x)
    slope, = torch.autograd.grad(score.sum(), x, retain_graph=True)
    loss = GANLoss('non_saturating').g_loss(score)
    loss_slope, = torch.autograd.grad(loss, x)
    real_x = real.detach().clone().requires_grad_(True)
    real_score = critic(real_x) if conditioning is None else critic(conditioning, real_x)
    real_slope, = torch.autograd.grad(real_score.sum(), real_x)
    dot = (slope * correction).flatten(1).sum(1)
    dnorm, rnorm = slope.flatten(1).norm(dim=1), real_slope.flatten(1).norm(dim=1)
    nonzero = correction.flatten(1).norm(dim=1) > 1e-12
    derivative = float((loss_slope * correction).sum())
    with torch.no_grad():
        def value(points):
            logits = critic(points) if conditioning is None else critic(conditioning, points)
            return GANLoss('non_saturating').g_loss(logits)
        finite_difference = float((value(x + 1e-5 * correction) - value(x - 1e-5 * correction)) / 2e-5)
    return dict(fake_score=stats(score), real_score=stats(real_score),
                fake_input_gradient_norm=stats(dnorm), real_input_gradient_norm=stats(rnorm),
                fake_fraction_above_cap=float((dnorm > 1).double().mean()),
                real_fraction_above_cap=float((rnorm > 1).double().mean()),
                score_ascent_correction_cosine=cosine(slope, correction),
                fraction_score_ascent_toward_correction=float((dot[nonzero] > 0).double().mean()),
                correction_generator_loss_derivative=derivative,
                correction_finite_difference=finite_difference,
                finite_difference_absolute_error=abs(derivative - finite_difference))


def model_motion(generator, latent, gradients, rate, *, correction=None, conditioning=None,
                 prior_gradient=None, prior_rate=.03):
    parameters = dict(generator.named_parameters())
    tangent = {name: direction(gradient, rate) for name, gradient in zip(parameters, gradients)}
    latent_tangent = torch.zeros_like(latent) if prior_gradient is None else direction(
        prior_gradient, prior_rate, prior=True)
    def forward(p, z):
        args = (z,) if conditioning is None else (conditioning, z)
        return functional_call(generator, p, args)
    output, network_motion = jvp(forward, (parameters, latent), (tangent, torch.zeros_like(latent)))
    _, prior_motion = jvp(forward, (parameters, latent),
                         ({k: torch.zeros_like(v) for k, v in parameters.items()}, latent_tangent))
    motion = (network_motion + prior_motion).detach()
    result = dict(network_output_step_norm=stats(network_motion.flatten(1).norm(dim=1)),
                  prior_output_step_norm=stats(prior_motion.flatten(1).norm(dim=1)),
                  combined_output_step_norm=stats(motion.flatten(1).norm(dim=1)),
                  output_mean_motion=motion.mean(dim=0).detach().tolist(),
                  negative_loss_derivative=float(sum((g * tangent[k]).sum()
                                                     for k, g in zip(parameters, gradients))))
    if correction is not None:
        dot = (motion * correction).flatten(1).sum(1)
        result.update(correction_cosine=cosine(motion, correction),
                      fraction_moves_toward_correction=float((dot > 0).double().mean()),
                      mean_squared_error_derivative=float((-2 * dot.mean() / correction[0].numel()).detach()))
    # Verify the derivative with functional parameter probes. Saved weights and
    # optimizer state are unchanged; these are not applied optimizer updates.
    epsilon = 1e-5
    with torch.no_grad():
        plus = forward({k: p + epsilon * tangent[k] for k, p in parameters.items()},
                       latent + epsilon * latent_tangent)
        minus = forward({k: p - epsilon * tangent[k] for k, p in parameters.items()},
                        latent - epsilon * latent_tangent)
        finite_difference = (plus - minus) / (2 * epsilon)
    result['jvp_relative_finite_difference_error'] = float(((finite_difference - motion).norm() /
                                                          motion.norm().clamp_min(1e-12)).detach())
    return result, output.detach(), motion.detach()


def distribution_probe(state, saved_fake, saved_real, centers, sigmas, masses=None, covariance=None,
                       real_component_weights=None):
    before = state_digest(state)
    trainer = state['trainer']
    if state['recipe']['loss'] != 'non_saturating':
        raise ValueError('Wrong loss in trained checkpoint')
    for optimizer in trainer['optimizers']:
        if any(optimizer['dualnorm'][k] != v for k, v in
               [('family', 'dualnorm'), ('momentum', 0.), ('smoothing', .001)]):
            raise ValueError('Wrong checkpoint optimizer')
        for group in optimizer['param_groups']:
            expected = dict(generator=.012, critic=.018, prior=.03)[group['role']]
            if abs(group['lr'] - expected) > 1e-12:
                raise ValueError('Wrong checkpoint role step')
    g, d = restore(trainer['models'])
    table = trainer['models']['prior']['z'].double()
    with torch.no_grad():
        census = g(table)
    ids = torch.cdist(census, centers).argmin(1)
    census_counts = torch.bincount(ids, minlength=len(centers))
    result = dict(completed_steps=trainer['completed_steps'],
                  diagnostic_scope='CPU FP64 restored weights; center census omits latent MoG noise; no served-law substitution',
                  latent_center_count=len(table), latent_center_allocation=census_counts.tolist(),
                  latent_center_nearest_distance_target_sigmas=stats((census - centers[ids]).norm(dim=1) / sigmas[ids]),
                  latent_center_genuine_fraction=float(((census - centers[ids]).norm(dim=1) <= 3 * sigmas[ids]).double().mean()))
    if masses is not None:
        result['target_masses'] = masses
    # Saved served samples and target draws for input-gradient measurements.
    if saved_fake.shape[1] == 1:
        order = saved_fake[:, 0].argsort()
        quantiles = torch.from_numpy(ndtri((np.arange(len(order)) + .5) / len(order))).double()
        correction = torch.zeros_like(saved_fake)
        correction[order, 0] = centers[0, 0] + sigmas[0] * quantiles - saved_fake[order, 0]
    else:
        assigned = torch.cdist(saved_fake, centers).argmin(1)
        correction = centers[assigned] - saved_fake
    result['saved_sample_critic_geometry'] = critic_geometry(d, saved_fake, saved_real, correction)
    # Deterministic second-moment cubature around evenly spaced stored rows.
    indices = torch.linspace(0, len(table) - 1, min(len(table), 512)).round().long()
    z = table[indices]
    dimension = z.shape[1]
    sigma = float(trainer['models']['prior']['sigma'])
    offsets = torch.cat((torch.eye(dimension), -torch.eye(dimension))) * math.sqrt(dimension) * sigma
    latent = (z[:, None, :] + offsets).flatten(0, 1)
    fake = g(latent)
    loss = GANLoss('non_saturating').g_loss(d(fake))
    gradients = grads(loss, tuple(g.parameters()))
    assignments = torch.cdist(fake.detach(), centers).argmin(1)
    correction = centers[assignments] - fake.detach()
    if fake.shape[1] == 1:
        order = fake.detach()[:, 0].argsort()
        quantiles = torch.from_numpy(ndtri((np.arange(len(order)) + .5) / len(order))).double()
        correction[order, 0] = centers[0, 0] + sigmas[0] * quantiles - fake.detach()[order, 0]
    motion, output, delta = model_motion(g, latent, gradients, .012, correction=correction)
    motion['step_norm_target_sigmas'] = stats(delta.norm(dim=1) / sigmas[assignments])
    motion['coherent_mean_motion_energy_fraction'] = float(delta.mean(0).square().sum() /
                                                        delta.square().sum(1).mean())
    dot = (delta * correction).sum(1)
    motion['quadratic_correction_optimal_rate_multiplier'] = float(dot.mean() / delta.square().sum(1).mean())
    motion['fraction_full_linear_step_increases_fixed_correction_error'] = float(
        ((delta - correction).square().sum(1) > correction.square().sum(1)).double().mean())
    motion['linear_surrogate_rate_sensitivity'] = [dict(multiplier=factor,
        fixed_correction_mse_ratio=float((factor * delta - correction).square().sum() / correction.square().sum()),
        fraction_increases_fixed_correction_error=float(((factor * delta - correction).square().sum(1) >
                                                         correction.square().sum(1)).double().mean()))
        for factor in (.02, .05, .1, .25, .5, 1.)]
    motion['per_parameter_network_output_motion'] = []
    parameters = dict(g.named_parameters())
    tangent = {k: direction(gs, .012) for k, gs in zip(parameters, gradients)}
    for key in parameters:
        _, contribution = jvp(lambda p: functional_call(g, p, (latent,)), (parameters,),
                              ({k: tangent[k] if k == key else torch.zeros_like(v) for k, v in parameters.items()},))
        motion['per_parameter_network_output_motion'].append(dict(parameter=key,
            median_step_target_sigmas=float(torch.median(contribution.norm(dim=1) / sigmas[assignments]).detach()),
            mean_motion=contribution.mean(0).detach().tolist()))
    motion['network_only_first_order_step_not_actual_next_update'] = True
    result.update(generator_spectra=spectral_summary(g, gradients, .012),
                  generator_motion=motion,
                  cubature=dict(rows=len(indices), points=len(latent), sigma=sigma,
                                row_selection='all if <=512, otherwise evenly spaced deterministic rows',
                                output_mean=output.mean(0).tolist(), output_std=output.std(0, unbiased=False).tolist()))
    if fake.shape[1] == 1:
        centered = output - output.mean(0)
        result['generator_motion']['std_first_order_change'] = float((centered * delta).mean() / output.std(unbiased=False))
    # The saved evaluation pairing is a fixed diagnostic, not a training batch.
    if real_component_weights is None:
        d_game = GANLoss('non_saturating').d_loss(d(saved_real), d(saved_fake))
        penalty = GradientPenalty(arm='b_cap', coeff=1, kappa=1)(d, saved_real, saved_fake)
    else:
        # Cubature integrates each mode separately with its declared mass.
        real_groups = saved_real.reshape(len(centers), -1, saved_real.shape[-1])
        d_game = sum(weight * GANLoss('non_saturating').d_loss(d(real), d(saved_fake))
                     for weight, real in zip(real_component_weights, real_groups))
        penalty = sum(weight * GradientPenalty(arm='b_cap', coeff=1, kappa=1)(d, real, saved_fake)
                      for weight, real in zip(real_component_weights, real_groups))
    d_game_grad = grads(d_game, tuple(d.parameters()))
    d_penalty_grad = grads(penalty, tuple(d.parameters()))
    total = tuple(a + b for a, b in zip(d_game_grad, d_penalty_grad))
    result['critic_objective_balance'] = dict(game_value=float(d_game.detach()), penalty_value=float(penalty.detach()),
        game_gradient_norm=float(flatten(d_game_grad).norm()), penalty_gradient_norm=float(flatten(d_penalty_grad).norm()),
        game_penalty_cosine=cosine(flatten(d_game_grad), flatten(d_penalty_grad)),
        combined_spectra=spectral_summary(d, total, .018))
    # Local latent Jacobians at every stored center, not sampled output spread.
    jacobian = []
    for axis in range(dimension):
        basis = torch.zeros_like(table)
        basis[:, axis] = 1
        _, response = jvp(g, (table,), (basis,))
        jacobian.append(response.detach())
    jacobian = torch.stack(jacobian, dim=2)
    singular = torch.linalg.svdvals(jacobian)
    result['center_latent_jacobian'] = dict(singular_values=stats(singular),
        local_mog_total_variance_over_target_total_variance=stats(
            sigma ** 2 * jacobian.square().sum((1, 2)) / (census.shape[1] * sigmas[ids].square())))
    result['per_target_center'] = [dict(component=k, assigned_latent_centers=int(census_counts[k]),
        local_mog_total_variance_ratio=stats(sigma ** 2 * jacobian[ids == k].square().sum((1, 2)) /
                                            (census.shape[1] * sigmas[k] ** 2))) for k in range(len(centers))]
    cube = output.reshape(len(indices), 2 * dimension, -1)
    component_means = cube.mean(1)
    within = (cube - component_means[:, None]).square().sum(-1).mean()
    between = (component_means - component_means.mean(0)).square().sum(-1).mean()
    result['cubature_variance_decomposition'] = dict(within_latent_component=float(within),
        between_latent_components=float(between), within_fraction=float(within / (within + between)))
    if covariance is not None:
        for k, row in enumerate(result['per_target_center']):
            selected = jacobian[ids == k]
            if not len(selected):
                row['local_minimum_covariance_eigen_ratio'] = None
                continue
            chol = torch.linalg.cholesky(covariance[k])
            whitened = torch.linalg.solve_triangular(chol, selected, upper=False)
            eigen = torch.linalg.eigvalsh(sigma ** 2 * whitened @ whitened.transpose(1, 2))
            row['local_minimum_covariance_eigen_ratio'] = stats(eigen[:, 0])
    result['input_checkpoint_unchanged'] = before == state_digest(state)
    if not result['input_checkpoint_unchanged']:
        raise ValueError('Analysis mutated its checkpoint')
    return result


def conditional_probe(state, name):
    before = state_digest(state)
    recipe = state['applied']['recipe']
    for key, expected in [('loss', 'non_saturating'), ('reg_arm', 'b_cap'), ('reg_coeff', 1.),
                          ('reg_kappa', 1.), ('optimizer_family', 'dualnorm'), ('optimizer_smoothing', .001)]:
        if recipe[key] != expected:
            raise ValueError(f'Conditional checkpoint has a different recipe: {key}')
    optimizers = [*state['optimizers']['generator'], state['optimizers']['discriminator']]
    for optimizer in optimizers:
        if optimizer['dualnorm']['momentum'] != 0 or optimizer['dualnorm']['smoothing'] != .001:
            raise ValueError('Wrong conditional normalized optimizer')
        for group in optimizer['param_groups']:
            if abs(group['lr'] - dict(generator=.012, prior=.03, critic=.018)[group['role']]) > 1e-12:
                raise ValueError('Wrong conditional role step')
    g, d = restore(state['models'], name)
    latent = state['models']['prior0']['z'].double().detach().requires_grad_(True)
    slow, target = trajectory.trajectories()
    slow, target = slow.double(), target.double()
    fake = g(slow, latent)
    mse = (fake - target).square().mean()
    cover = trajectory.PROTOCOL['cover_weight'] * trajectory._cover(fake, target)
    adversarial = GANLoss('non_saturating').g_loss(d(slow, fake))
    prior_l2 = trajectory.PROTOCOL['particle_l2'] * latent.square().mean()
    spread = ParticleRegularizer(weight=trajectory.PROTOCOL['vicreg_weight'])(latent)
    losses = dict(adversarial=adversarial, coverage=cover, prior_l2=prior_l2, prior_spread=spread)
    if name == 'residual_student':
        losses['paired_mse'] = residual_student.RESIDUAL_WEIGHT * mse
    all_parameters = (*g.parameters(), latent)
    component_gradients = {key: grads(loss, all_parameters) for key, loss in losses.items()}
    total = tuple(sum(parts) for parts in zip(*component_gradients.values()))
    target_grad = grads(mse, all_parameters)
    result = dict(identity_mse=float(mse.detach()),
        nearest_trajectory_ids=torch.cdist(fake.detach(), target).argmin(1).tolist(),
        nearest_pad_ids=torch.cdist(fake.detach()[:, -2:], target[:, -2:]).argmin(1).tolist(),
        row_identity_mse=(fake.detach() - target).square().mean(1).tolist(),
        objective_values={key: float(loss.detach()) for key, loss in losses.items()},
        gradient_components={key: dict(network_norm=float(flatten(gs[:-1]).norm()), prior_norm=float(gs[-1].norm()),
            network_cosine_with_identity_gradient=cosine(flatten(gs[:-1]), flatten(target_grad[:-1])),
            prior_cosine_with_identity_gradient=cosine(gs[-1], target_grad[-1])) for key, gs in component_gradients.items()},
        component_network_cosines={f'{a}__{b}': cosine(flatten(component_gradients[a][:-1]), flatten(component_gradients[b][:-1]))
            for i, a in enumerate(losses) for b in list(losses)[i + 1:]},
        critic_geometry=critic_geometry(d, fake.detach(), target, target - fake.detach(), conditioning=slow),
        generator_spectra=spectral_summary(g, total[:-1], .012))
    result['objective_output_gradients'] = {}
    for key in ('adversarial', 'coverage', 'paired_mse'):
        if key not in losses:
            continue
        output_gradient, = torch.autograd.grad(losses[key], fake, retain_graph=True)
        correct = target - fake.detach()
        result['objective_output_gradients'][key] = dict(norm=float(output_gradient.norm()),
            fraction_rows_descent_improves_identity=float(((-output_gradient * correct).sum(1) > 0).double().mean()),
            row_norms=output_gradient.norm(dim=1).tolist())
    result['motion_by_objective'] = {}
    for key, gs in [*component_gradients.items(), ('combined', total)]:
        motion, _, delta = model_motion(g, latent.detach(), gs[:-1], .012,
            conditioning=slow, correction=target - fake.detach(), prior_gradient=gs[-1])
        motion['identity_mse_derivative'] = float(2 * ((fake.detach() - target) * delta).mean())
        result['motion_by_objective'][key] = motion
    result['row_prior_gradient_norms'] = total[-1].norm(dim=1).tolist()
    result['coverage_gradient_sensitivity'] = []
    for factor in (0., .25, .5, .75, 1., 1.25):
        changed = tuple(all_g + (factor - 1) * cover_g for all_g, cover_g in
                        zip(total, component_gradients['coverage']))
        motion, _, delta = model_motion(g, latent.detach(), changed[:-1], .012,
            conditioning=slow, correction=target - fake.detach(), prior_gradient=changed[-1])
        result['coverage_gradient_sensitivity'].append(dict(multiplier=factor,
            effective_host_cover_weight=factor * trajectory.PROTOCOL['cover_weight'],
            identity_mse_derivative=float(2 * ((fake.detach() - target) * delta).mean()),
            fraction_rows_moves_toward_identity=motion['fraction_moves_toward_correction']))
    result['public_host_weights'] = dict(cover_weight=trajectory.PROTOCOL['cover_weight'],
        particle_l2=trajectory.PROTOCOL['particle_l2'], vicreg_weight=trajectory.PROTOCOL['vicreg_weight'],
        residual_weight=residual_student.RESIDUAL_WEIGHT if name == 'residual_student' else 0)
    permutation = torch.cdist(fake.detach(), target).argmin(1)
    if permutation.unique().numel() == len(target):
        perfect_wrong_pairing = target[permutation]
        result['analytic_permutation_control'] = dict(scope='target-informed algebraic witness, no trained result',
            indices=permutation.tolist(), set_coverage_loss=float(trajectory._cover(perfect_wrong_pairing, target)),
            identity_mse=float((perfect_wrong_pairing - target).square().mean()))
    result['input_checkpoint_unchanged'] = before == state_digest(state)
    if not result['input_checkpoint_unchanged']:
        raise ValueError('Analysis mutated conditional checkpoint')
    return result


def image_probe(records, quality_rmse):
    endpoint = records[-1]
    images, targets = endpoint['samples'].double().flatten(1), endpoint['targets'].double().flatten(1)
    distance = (images[:, None] - targets).square().mean(2)
    ids = distance.argmin(1)
    assigned = targets[ids]
    background = assigned == 0
    squared = (images - assigned).square()
    background_error = (squared * background).sum(1)
    foreground_error = (squared * ~background).sum(1)
    brightness_error = []
    texture_error = []
    for image, target in zip(images, assigned):
        active = target != 0
        values = image[active]
        brightness_error.append(len(values) * (values.mean() - target[active].mean()).square())
        texture_error.append((values - values.mean()).square().sum())
    brightness_error = torch.stack(brightness_error)
    texture_error = torch.stack(texture_error)
    residual = (brightness_error + texture_error + background_error - squared.sum(1)).abs().max()
    if residual > 1e-12:
        raise ValueError('Image squared-error decomposition is not exact')
    rmse = distance.min(1).values.sqrt()
    invalid = rmse > quality_rmse
    rows = []
    for k, target in enumerate(targets):
        mask = ids == k
        foreground = target != 0
        rows.append(dict(component=k, latent_centers=int(mask.sum()),
            genuine_latent_centers=int((mask & ~invalid).sum()),
            foreground_prediction_mean=stats(images[mask][:, foreground]) if mask.any() else None,
            foreground_target_mean=float(target[foreground].mean()),
            background_prediction_mean=stats(images[mask][:, ~foreground]) if mask.any() else None,
            template_rmse=stats(rmse[mask])))
    original = endpoint['metrics']
    reproduced_hq = float((~invalid).double().mean())
    if abs(reproduced_hq - original['hq']) > 1e-12:
        raise ValueError('Saved image quality does not reproduce')
    history = []
    for row in records:
        values = row['samples'].double().flatten(1)
        labels = (values[:, None] - targets).square().mean(2).argmin(1)
        history.append(dict(step=row['step'], nearest_template_counts=torch.bincount(labels,
            minlength=len(targets)).tolist(), quality=row['metrics']['hq'], genuine_modes=row['metrics']['modes']))
    return dict(completed_steps=endpoint['step'], original_endpoint_metrics=original, quality_rmse=quality_rmse,
        nearest_template_counts=torch.bincount(ids, minlength=len(targets)).tolist(),
        quality_reproduced=reproduced_hq, invalid_images=int(invalid.sum()),
        background_fraction_of_total_squared_error=float(background_error.sum() / squared.sum()),
        invalid_background_fraction_of_squared_error=float(background_error[invalid].sum() /
                                                          squared[invalid].sum()) if invalid.any() else None,
        foreground_fraction_of_total_squared_error=float(foreground_error.sum() / squared.sum()),
        foreground_brightness_bias_fraction_of_squared_error=float(brightness_error.sum() / squared.sum()),
        foreground_nonuniformity_fraction_of_squared_error=float(texture_error.sum() / squared.sum()),
        pixel_mse_decomposition_max_residual=float(residual),
        per_template=rows, history_summary=dict(
            maximum_nearest_occupied_templates=max(sum(c > 0 for c in row['nearest_template_counts']) for row in history),
            maximum_genuine_modes=max(row['genuine_modes'] for row in history),
            first_full_nearest_template_coverage_step=next((row['step'] for row in history if all(row['nearest_template_counts'])), None),
            terminal_nearest_template_counts=history[-1]['nearest_template_counts']))


def plot_diagnostics(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = result['diagnostics']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    names = ['grid100', 'rotated100', 'staggered100']
    axes[0, 0].bar(names, [100 * data[n]['saved_sample_critic_geometry']['fraction_score_ascent_toward_correction'] for n in names],
                   color='#287d8e')
    axes[0, 0].set(ylabel='Saved samples pointing toward nearest center (%)', ylim=(0, 105),
                   title='Native critic direction is mostly useful')
    axes[0, 1].bar(names, [data[n]['generator_motion']['step_norm_target_sigmas']['median'] for n in names], color='#bb6235')
    axes[0, 1].axhline(3, color='black', linestyle='--', label='Quality radius: 3 sigma')
    axes[0, 1].set(ylabel='Median network motion / target sigma',
                   title='Linearized G motion exceeds the quality radius')
    axes[0, 1].legend(fontsize=8)
    for name, label in [('trajectory', 'Trajectory'), ('residual_student', 'Residual student')]:
        rows = data[name]['coverage_gradient_sensitivity']
        axes[1, 0].plot([r['multiplier'] for r in rows], [r['identity_mse_derivative'] for r in rows],
                        marker='o', label=label)
    axes[1, 0].axhline(0, color='black', linewidth=1)
    axes[1, 0].axvline(1, color='gray', linestyle='--', label='Recorded coverage weight')
    axes[1, 0].set(xlabel='Coverage gradient multiplier in frozen-state probe',
                   ylabel='First-order identity MSE change', title='Coverage competes with identity correction')
    axes[1, 0].legend(fontsize=8)
    images = ['img_bars4', 'img_blobs4', 'img_intensity2']
    bottom = np.zeros(3)
    for key, label, color in [('background_fraction_of_total_squared_error', 'Background activation', '#287d8e'),
                              ('foreground_brightness_bias_fraction_of_squared_error', 'Foreground mean bias', '#bb6235'),
                              ('foreground_nonuniformity_fraction_of_squared_error', 'Foreground nonuniformity', '#8a75a4')]:
        values = np.array([data[n][key] for n in images]) * 100
        axes[1, 1].bar(['Bars', 'Blobs', 'Intensity'], values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[1, 1].set(ylabel='Share of endpoint pixel squared error (%)',
                   title='Image fidelity failures affect different pixels')
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('Frozen BCAP failure diagnostics — zero training updates', fontsize=13)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-workspace', type=Path,
                        default=Path('/home/martyn/dev/ParticleGAN-bcap-tier2-search'))
    parser.add_argument('--output', type=Path, default=REPORT / 'failure-state-analysis.json')
    parser.add_argument('--figure', type=Path, default=REPORT / 'failure-mechanisms.png')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial_rng = torch.random.get_rng_state().clone()
    selected = read(REPORT / 'summary.json')['selected_candidate_id']
    winner = next(r for r in read(REPORT / 'readout.json')['trials'] if r['candidate_id'] == selected)
    recorded_tasks = {r['task']: r for r in winner['tasks']}
    result = dict(schema_version=1, candidate_id=selected, seed=0,
        source_digest=read(REPORT / 'readout.json')['source_digest'],
        scope='frozen_checkpoint_derivatives_and_deterministic_census', qualification_input=False,
        optimizer_updates_added=0, random_sampling_draws_added=0,
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                     device='cpu', derivative_dtype='float64', saved_weights_dtype='float32'),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in [REPORT / 'summary.json', REPORT / 'receipts.json',
                                                         REPORT / 'readout.json', Path(__file__)]},
        source_proofs={}, artifact_proofs=[], diagnostics={}, verified_receipt_files=0, task_bindings={})
    tasks, contracts = {}, {}
    for receipt in read(REPORT / 'receipts.json'):
        if receipt['candidate_id'] != selected:
            continue
        base = args.original_workspace / 'reports/forge/attempts' / receipt['attempt_id']
        for key, binding in receipt['provenance']['original_files'].items():
            verified(base / f'{key}.json', binding['sha256'])
            result['verified_receipt_files'] += 1
        original = read(base / 'result.json')['task_results'][0]
        source = read(base / 'evidence.json')
        if receipt['provenance']['source_digest'] != result['source_digest']:
            raise ValueError('Foreign source cohort')
        tasks[original['task_id']] = (original, source)
        contract = read(base / 'request.json')['request']['tasks'][original['task_id']]
        contracts[original['task_id']] = contract
        result['task_bindings'][original['task_id']] = dict(attempt_id=receipt['attempt_id'],
            candidate_revision=receipt['candidate_revision'], compatibility_key=original['compatibility_key'],
            qualification_tier=recorded_tasks[original['task_id']]['qualification_tier'],
            gate_status=original['gate_status'], prior=contract['execution']['prior'],
            initializer=contract['execution']['initializer'], sampling_law=contract['evaluation']['sampling_law'])
    result['tier2_status_counts'] = {status: sum(row['qualification_tier'] == 2 and row['gate_status'] == status
                                              for row in result['task_bindings'].values()) for status in ('PASS', 'FAIL')}
    if result['tier2_status_counts'] != {'PASS': 7, 'FAIL': 14}:
        raise ValueError('Original Tier 2 status counts changed')
    for filename in ('lib/toy_models.py', 'particlegan/optim/dualnorm.py', 'particlegan/gan_loss.py',
                     'particlegan/grad_regularizers.py', 'particlegan/vicreg_loss.py',
                     'benchmarks/locked_shared/trajectory.py', 'benchmarks/locked_shared/hosts/residual_student.py',
                     'benchmarks/toy100/problems.py', 'benchmarks/legacy/locked_shared.py', 'experiments/forge/state.py',
                     'experiments/forge/gaussian_tasks.py'):
        digest = verified(args.original_workspace / filename, source['source']['files'][filename])
        verified(ROOT / filename, digest)
        result['source_proofs'][filename] = digest

    def artifact(name, relative, *, provenance=False):
        original, evidence = tasks[name]
        binding = original['evidence']['provenance_checkpoint'] if provenance else original['evidence']
        root = Path(binding.get('artifact_root', evidence['local_artifact_root']))
        path = root / relative
        expected = (binding['artifact_manifest']['files'][relative]['sha256'] if 'artifact_manifest' in binding
                    else binding['saved_observer_outputs']['sha256'])
        digest = verified(path, expected)
        result['artifact_proofs'].append(dict(task=name, attempt_id=path.parts[-3] if provenance else
                                             str(evidence['local_artifact_root']).split('/')[-1],
                                             path=str(path), sha256=digest))
        return path

    name = 'gaussian1d_stability'
    saved = torch.load(artifact(name, 'observed-samples.pt'), map_location='cpu', weights_only=True)
    for filename, step, mean in [('initial-state.pt', 1000, 2.), ('pre-shift-state.pt', 4000, 2.), ('state.pt', 6000, 3.)]:
        state = torch.load(artifact(name, filename), map_location='cpu', weights_only=True)
        record = next(r for r in saved if r['step'] == step)
        points = record['samples'].double()
        training_state = {k: v for k, v in state.items() if k != 'streams'}
        training_state['trainer'] = {k: v for k, v in state['trainer'].items() if k != 'streams'}
        if state_digest(training_state) != record['training_state_sha256']:
            raise ValueError('Gaussian samples do not bind to examined training state')
        # Deterministic Gaussian quantiles, separately labelled from target draws.
        real = torch.tensor(mean + .5 * ndtri((np.arange(len(points)) + .5) / len(points)))[:, None]
        result['diagnostics'][f'{name}_{step}'] = distribution_probe(
            state, points, real, torch.tensor([[mean]], dtype=torch.float64), torch.tensor([.5]))
        result['diagnostics'][f'{name}_{step}']['sample_training_state_binding_verified'] = True
        print(f'Analyzed {name} state {step}', flush=True)
    for name in ('grid100', 'rotated100', 'staggered100'):
        state = torch.load(artifact(name, f'{name}/training-state.pt'), map_location='cpu', weights_only=True)
        saved = np.load(artifact(name, f'{name}/final_samples.npz'))
        centers, sigma = evaluation_geometry(name, dtype=torch.float64)
        result['diagnostics'][name] = distribution_probe(state, torch.from_numpy(saved['live']).double(),
            torch.from_numpy(saved['target']).double(), centers, torch.full((100,), sigma, dtype=torch.float64))
        print(f'Analyzed {name} endpoint', flush=True)
    for name in ('vector_unequal_mass', 'vector_unequal_width', 'vector_anisotropic', 'vector_overlap'):
        state = torch.load(artifact(name, 'provenance-state.pt', provenance=True), map_location='cpu', weights_only=True)
        original, _ = tasks[name]
        spec = original['evidence']['host']['definition']
        saved = torch.load(artifact(name, 'observed-samples.pt'), map_location='cpu', weights_only=True)[-1]
        centers = torch.tensor(spec['means'], dtype=torch.float64)
        covariance = torch.tensor(spec['covariances'], dtype=torch.float64)
        sigma = covariance.diagonal(dim1=-2, dim2=-1).mean(1).sqrt()
        if 'target' not in saved:
            # Symmetric cubature, equal points per mode; weighted loss is not
            # reconstructed. Used only for critic norms, explicitly scoped.
            offsets = torch.cat((torch.eye(2), -torch.eye(2))) * math.sqrt(2)
            real = (centers[:, None, :] + torch.einsum('kij,pj->kpi', torch.linalg.cholesky(covariance), offsets.double())).flatten(0, 1)
        else:
            real = saved['target'].double()
        result['diagnostics'][name] = distribution_probe(state, saved['samples'].double(), real, centers, sigma,
            spec['masses'], covariance=covariance,
            real_component_weights=None if 'target' in saved else spec['masses'])
        result['diagnostics'][name]['vector_real_probe_law'] = 'saved target draws' if 'target' in saved else \
            'deterministic second-moment cubature per component; game and penalty integrate declared target masses'
        print(f'Analyzed {name} endpoint', flush=True)
    for name in ('trajectory', 'residual_student'):
        state = torch.load(artifact(name, 'provenance-state.pt', provenance=True), map_location='cpu', weights_only=True)
        result['diagnostics'][name] = conditional_probe(state, name)
        published_mse = tasks[name][0]['metrics']['identity_mse']
        observed = result['diagnostics'][name]['identity_mse']
        if abs(observed - published_mse) > 1e-6:
            raise ValueError(f'Conditional endpoint does not reproduce: {name}: {observed} vs {published_mse}')
        result['diagnostics'][name]['endpoint_mse_absolute_error_vs_original'] = abs(observed - published_mse)
        print(f'Analyzed {name} objective balance', flush=True)
    for name in ('img_bars4', 'img_blobs4', 'img_intensity2'):
        saved = torch.load(artifact(name, 'observed-images.pt'), map_location='cpu', weights_only=True)
        result['diagnostics'][name] = image_probe(saved, contracts[name]['evaluation']['measurement']['quality_rmse'])
        print(f'Analyzed {name} pixel-error decomposition', flush=True)
    result['torch_global_rng_unchanged'] = bool(torch.equal(initial_rng, torch.random.get_rng_state()))
    if not result['torch_global_rng_unchanged']:
        raise ValueError('Diagnostic consumed global RNG')
    derivative_errors = [r['generator_motion']['jvp_relative_finite_difference_error']
                         for r in result['diagnostics'].values() if 'generator_motion' in r]
    derivative_errors.extend(m['jvp_relative_finite_difference_error'] for r in result['diagnostics'].values()
                             for m in r.get('motion_by_objective', {}).values())
    result['verification'] = dict(maximum_jvp_relative_finite_difference_error=max(derivative_errors),
        all_input_checkpoints_unchanged=all(r.get('input_checkpoint_unchanged', True)
                                          for r in result['diagnostics'].values()),
        gaussian_sample_state_bindings_verified=3, conditional_endpoint_mse_tolerance=1e-6)
    if max(derivative_errors) > 1e-5:
        raise ValueError('JVP finite difference verification failed')
    plot_diagnostics(result, args.figure)
    result['figure'] = dict(path=str(args.figure.relative_to(ROOT)), sha256=sha(args.figure))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}; zero optimizer steps and random draws', flush=True)


if __name__ == '__main__':
    main()
