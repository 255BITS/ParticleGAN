"""Scoped discriminator-signal experiments; all G gradients are adversarial.

The frozen hosts, sample/evaluation budgets and scoring functions are unchanged.
The legacy host's separate coverage objective is explicitly disabled. Noise is
an observation channel of D only; no target statistics or labels enter it.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import math
from unittest.mock import patch

import torch
from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer
from benchmarks.toy100.models import InputNoise
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, _InputAdapter
from benchmarks.transfer_suite import compare_defaults
from benchmarks.locked_shared.hosts import residual_student
from continuous_candidates import candidate_update as observe_adam


@contextmanager
def signal_policy(options):
    options = dict(options)
    allowed = {"noise", "sigma", "smooth", "zero_gp", "score_l2", "network_eps", "prior_eps",
               "antithetic_pairs", "real_only_gp", "noise_dimension_power",
               "consistency", "consistency_coeff", "consistency_sigma"}
    if set(options) - allowed:
        raise ValueError(f"unknown signal options: {set(options) - allowed}")
    kind = options.get("noise", "annealed")
    if kind not in ("annealed", "fixed", "mixture"):
        raise ValueError("unknown noise policy")
    for name in ("sigma", "smooth", "zero_gp", "score_l2", "consistency_coeff"):
        value = options.get(name, 0.)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
    if options.get("smooth", 0.) >= .5:
        raise ValueError("smoothing must be below one half")
    pairs = options.get("antithetic_pairs", 0)
    if type(pairs) is not int or pairs not in (0, 1, 2):
        raise ValueError("antithetic_pairs must be 0, 1, or 2")
    dimension_power = options.get("noise_dimension_power", 0.)
    if dimension_power not in (0., .25, .5):
        raise ValueError("noise_dimension_power must be 0, .25, or .5")
    consistency = options.get("consistency", "none")
    if consistency not in ("none", "mixup", "curvature"):
        raise ValueError("unknown critic consistency penalty")
    consistency_sigma = options.get("consistency_sigma", .1)
    if not math.isfinite(consistency_sigma) or consistency_sigma <= 0:
        raise ValueError("consistency_sigma must be positive")
    consistency_coeff = options.get("consistency_coeff", 0.)
    original_d = GANLoss.d_loss
    original_penalty = GradRegularizer.penalty
    original_candidate = compare_defaults.candidate
    sigma = options.get("sigma", 0.)
    smooth = options.get("smooth", 0.)
    zero_gp = options.get("zero_gp", 0.)
    score_l2 = options.get("score_l2", 0.)
    noise_receipt = {"calls": 0, "effective_sigma_min": None, "effective_sigma_max": None}
    extra = GradRegularizer("a_r1r2", coeff=zero_gp)
    work = {"consistency_forwards": 0, "consistency_applications": 0}

    def paired_scores(discriminator, inputs):
        # Use the same observation noise across the finite-difference stencil.
        # Otherwise noise variance would masquerade as critic curvature.
        streams = {}
        for module in discriminator.modules():
            stream = (module.policy.input_stream if isinstance(module, _InputAdapter)
                      else module.noise_stream if isinstance(module, InputNoise) else None)
            if stream is not None:
                streams[id(stream)] = stream
        saved = [(stream, stream.get_state()) for stream in streams.values()]
        scores = []
        for points in inputs:
            for stream, state in saved:
                stream.set_state(state)
            scores.append(discriminator(points))
        work["consistency_forwards"] += len(inputs)
        return scores

    def perturb(points, nominal, stream):
        amplitude = nominal if kind == "annealed" else sigma
        if dimension_power:
            amplitude = amplitude / (points[0].numel() ** dimension_power)
        noise_receipt["calls"] += 1
        for key, fn in (("effective_sigma_min", min), ("effective_sigma_max", max)):
            old = noise_receipt[key]
            noise_receipt[key] = amplitude if old is None else fn(old, amplitude)
        if amplitude == 0:
            return points
        noise = torch.randn(points.shape, generator=stream, device=points.device, dtype=points.dtype)
        if kind == "mixture":
            # Equal sharp and broad observation channels, independently drawn.
            mask = torch.rand((len(points),) + (1,) * (points.ndim - 1),
                              generator=stream, device=points.device) < .5
            noise = noise * mask
        return points + amplitude * noise

    def legacy_input(policy, points):
        key = "input_eval" if policy._evaluating else "input_train"
        policy._counts[key + "_calls"] += 1
        if (policy.input_sigma if kind == "annealed" else sigma) > 0:
            policy._counts[key + "_elements"] += points.numel()
        return perturb(points, policy.input_sigma, policy.input_stream)

    def native_input(wrapper, points):
        if not pairs:
            return wrapper.model(perturb(points, wrapper.sigma, wrapper.noise_stream))
        value = 0.
        for _ in range(pairs):
            noisy = perturb(points, wrapper.sigma, wrapper.noise_stream)
            value = value + wrapper.model(noisy) + wrapper.model(2 * points - noisy)
        return value / (2 * pairs)

    def legacy_forward(wrapper, *args, **kwargs):
        values = list(args)
        points = values[wrapper.data_index]
        value = 0.
        for _ in range(max(1, pairs)):
            noisy = wrapper.policy.input(points)
            values[wrapper.data_index] = noisy
            value = value + wrapper.model(*values, **kwargs)
            if pairs:
                values[wrapper.data_index] = 2 * points - noisy
                value = value + wrapper.model(*values, **kwargs)
        return value / (2 * pairs if pairs else 1)

    def discriminator_loss(loss, real, fake):
        value = original_d(loss, real, fake)
        if smooth:
            value = (1 - smooth) * value + smooth * original_d(loss, fake, real)
        if score_l2:
            value = value + score_l2 * .5 * (real.square().mean() + fake.square().mean())
        return value

    def penalty(reg, discriminator, real, fake, step=1, generator=None, collect_stats=True):
        if options.get("real_only_gp", False) and reg.arm == "a_r1r2":
            value = reg.coeff * reg._grad_norm(discriminator, real, squared=True).mean()
            stats = {"applied": True, "pen": float(value.detach()), "center": 0.} if collect_stats else {}
        else:
            value, stats = original_penalty(reg, discriminator, real, fake, step, generator, collect_stats)
        if zero_gp and reg is not extra:
            additional, _ = original_penalty(extra, discriminator, real, fake, step, generator, False)
            value = value + additional
        if consistency_coeff and consistency != "none":
            real, fake = real.detach(), fake.detach()
            if consistency == "mixup":
                alpha = torch.rand((len(real),) + (1,) * (real.ndim - 1),
                                   device=real.device, dtype=real.dtype)
                center = alpha * real + (1 - alpha) * fake
                dr, df, dc = paired_scores(discriminator, (real, fake, center))
                weight = alpha.reshape((len(real),) + (1,) * (dr.ndim - 1))
                target = weight * dr.detach() + (1 - weight) * df.detach()
                extra_value = (dc - target).square().mean()
            else:
                delta = torch.randn_like(fake) * (consistency_sigma / math.sqrt(fake[0].numel()))
                plus, center, minus = paired_scores(discriminator, (fake + delta, fake, fake - delta))
                extra_value = (plus + minus - 2 * center).square().mean() / consistency_sigma ** 4
            value = value + consistency_coeff * extra_value
            work["consistency_applications"] += 1
        return value, stats

    def adversarial_candidate(recipe):
        return replace(original_candidate(recipe), cover_weight=0., particle_l2=0., vicreg_weight=0.)

    eps = {k: options.get(k, 1e-8) for k in ("network_eps", "prior_eps")}
    with ExitStack() as stack:
        receipt = stack.enter_context(observe_adam(eps))
        receipt.update(signal_options=options, noise= noise_receipt,
                       signal_work=work,
                       generator_objective="GANLoss.g_loss only; no cover, L2, or VICReg",
                       nominal_noise_receipt="host schedule; actual channel is signal_options and noise")
        stack.enter_context(patch.object(GANLoss, "d_loss", discriminator_loss))
        stack.enter_context(patch.object(GradRegularizer, "penalty", penalty))
        stack.enter_context(patch.object(NoisePolicy, "input", legacy_input))
        stack.enter_context(patch.object(InputNoise, "forward", native_input))
        stack.enter_context(patch.object(_InputAdapter, "forward", legacy_forward))
        stack.enter_context(patch.object(compare_defaults, "candidate", adversarial_candidate))
        stack.enter_context(patch.object(residual_student, "RESIDUAL_WEIGHT", 0.))
        yield receipt
