"""Replay routed candidates against the noisy game without changing training.

This diagnostic is not a structural selector or a game-repair certificate.
It freezes a caller-owned model/controller/KA2 phase, reruns complete models
with matched private DV12 and paired-noise draws, and reports game gradients.
Temporal persistence and subsequent critic adaptation require external traces.
All copied models run in evaluation mode. This matches the linear/tanh and
frozen-host examples; it does not reproduce training-mode BatchNorm/Dropout.

Reconstruct ``capture_game`` after loading the ordinary policy checkpoint,
using the saved diagnostic RNG states and options. Nothing owns another policy
checkpoint or draws from training RNG. See ``main`` for a small runnable use.
"""
from copy import deepcopy
import json
import math

import torch

from particlegan import GANLoss, RoutedCandidate, RoutedRows

try:
    from examples.e22_routed_sites import apply_critic_penalty
except ModuleNotFoundError:
    from e22_routed_sites import apply_critic_penalty


def _states(values, device, name):
    if isinstance(values, torch.Tensor):
        values = (values,)
    states = []
    for value in values:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError(f"{name} must contain fixed torch.Generator states")
        state = value.detach().cpu().clone()
        try:
            torch.Generator(device=device).set_state(state)
        except RuntimeError as error:
            raise ValueError(f"{name} does not match the model device") from error
        states.append(state)
    if not states:
        raise ValueError("noisy game replay needs at least one fixed draw")
    return tuple(states)


def _generator(device, state):
    stream = torch.Generator(device=device)
    stream.set_state(state)
    return stream


def _cosine(left, right):
    denominator = left.double().norm() * right.double().norm()
    if not bool(denominator > 0):
        return None
    return float(((left.double().flatten() @ right.double().flatten()) / denominator).clamp(-1, 1))


def _default_rng(device):
    states = [torch.random.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return states


def _clone_bundle(bundle):
    clone = deepcopy(bundle)
    critic, optimizer, penalty = clone["models"]["critic"], clone["critic_optimizer"], clone["penalty"]
    if (optimizer.critic is not critic or penalty.optimizer is not optimizer
            or penalty.critic is not critic or penalty.regularizer.record is not optimizer.record):
        raise ValueError("capture the critic, its optimizer and its paired KA2 penalty together")
    # deepcopy preserves object aliases, but plain integer identity caches do
    # not follow the cloned modules. Rebuild the native penalty's view lookup.
    penalty._names = {id(module): name for name, module in critic.named_modules()}
    penalty.regularizer._critic_id = None
    penalty.collect_stats = True
    for module in clone["models"].values():
        module.eval()
        module.zero_grad(set_to_none=True)
    return clone


def capture_game(*, models, table, controller, critic_optimizer, penalty,
                 rows, output_sigma, latent_rng_states, paired_rng_states,
                 penalty_units="token"):
    """Capture explicit owners and fixed private draw states for paired replay.

    ``models`` supplies generator/critic and optional encoder/router modules.
    ``rows`` is their public complete-model RoutedRows callback contract.
    ``penalty`` and ``critic_optimizer`` must be a paired package KA2 setup;
    each arm evaluates the same NEXT penalty call on its own copy. Units are
    the caller's token/context convention, shared with the training example.
    Fixed states can be a single tensor or equal-length sequences of states.
    Callbacks must be deterministic in evaluation mode; private streams own
    all replay noise. Global model gradients are logged for temporal analysis.
    """
    if not isinstance(rows, RoutedRows) or not isinstance(models, dict):
        raise TypeError("capture_game requires explicit models and RoutedRows")
    if (not isinstance(table, torch.Tensor) or table.ndim != 2
            or not table.is_floating_point() or controller is None):
        raise ValueError("capture_game requires a floating table and a DV12 controller")
    if (penalty_units not in ("token", "context") or not math.isfinite(output_sigma)
            or output_sigma < 0 or controller.variant != "dv12"):
        raise ValueError("declare token/context penalty units, nonnegative noise and DV12")
    latent = _states(latent_rng_states, table.device, "latent_rng_states")
    paired = _states(paired_rng_states, table.device, "paired_rng_states")
    if len(latent) != len(paired):
        raise ValueError("latent and paired noise need the same number of fixed draws")
    bundle = _clone_bundle(dict(models=models, table=table, controller=controller,
                                critic_optimizer=critic_optimizer, penalty=penalty, rows=rows))
    return RoutedGameReplay(bundle, float(output_sigma), latent, paired, penalty_units)


class RoutedGameReplay:
    """Frozen diagnostic bundle; comparisons never advance this bundle."""

    def __init__(self, bundle, output_sigma, latent_states, paired_states, penalty_units):
        self._bundle, self.output_sigma = bundle, output_sigma
        self.latent_states, self.paired_states = latent_states, paired_states
        self.penalty_units = penalty_units

    def candidate(self):
        """Return a copied functional candidate for this captured model."""
        return self._bundle["rows"].candidate_for(self._bundle["models"], self._bundle["table"], copy=True)

    def _arm(self, context, targets, candidate, scale, latent_state, paired_state,
             prospective, output_log):
        rng_before = _default_rng(candidate.table.device)
        bundle = _clone_bundle(self._bundle)
        models, controller, rows = bundle["models"], bundle["controller"], bundle["rows"]
        parameter_names = {name for name, parameter in models["router"].named_parameters() if parameter.requires_grad}
        state = {name: value.detach().clone().requires_grad_(value.is_floating_point() and name in parameter_names)
                 for name, value in candidate.row_state.items()}
        table = candidate.table.detach().clone().requires_grad_(True)
        candidate = RoutedCandidate(table, state[rows.log_mass_key], state, candidate.averaged)
        prior = controller.routed_prior(table, candidate.log_mass)
        if prospective:
            controller.observe_prior(prior)
        latent_stream = _generator(table.device, latent_state)
        paired_stream = _generator(table.device, paired_state)
        applications = []
        stage = "critic"

        def perturb(codes):
            value = controller.perturb_latent(codes, latent_stream, prior, record=False)
            applications.append(dict(role=stage, shape=tuple(codes.shape),
                                     rms=float((value.detach().double() - codes.detach().double()).square().mean().sqrt())))
            return value

        # The actual paired loop uses independent critic/generator base draws
        # and two complete DV12 forwards, in this order. A fixed critic stays
        # fixed here: we measure its response gradient, not an optimizer step.
        critic_base = torch.randn(targets.shape, device=targets.device, dtype=targets.dtype, generator=paired_stream)
        generator_base = torch.randn(targets.shape, device=targets.device, dtype=targets.dtype, generator=paired_stream)
        with torch.no_grad():
            critic_output = rows.forward(models, context, candidate, perturb_fn=perturb)
        critic_real = self.output_sigma * critic_base
        critic_fake = critic_real + (critic_output - targets) / scale
        critic, loss = models["critic"], GANLoss()
        loss_d_game = loss.d_loss(critic(critic_real.detach()), critic(critic_fake.detach()))
        calls_before = bundle["critic_optimizer"].record.calls
        penalty = apply_critic_penalty(bundle["penalty"], critic, critic_real.detach(), critic_fake.detach(), units=self.penalty_units)
        loss_d = loss_d_game + penalty
        critic_parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
        critic_gradients = torch.autograd.grad(loss_d, critic_parameters, allow_unused=True)
        critic_gradient = torch.cat([gradient.detach().flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
                                     for parameter, gradient in zip(critic_parameters, critic_gradients)])
        stage = "generator"
        output, usage = rows.forward_with_usage(models, context, candidate, perturb_fn=perturb)
        if output.shape != targets.shape:
            raise ValueError("replayed outputs must match paired targets")
        noise = self.output_sigma * generator_base
        real, fake = noise, noise + (output - targets) / scale
        # Match the generator role: the real-logit path is fixed, while the
        # complete noisy routed fake path retains table/global-model gradients.
        real_logits = critic(real).detach()
        fake_logits = critic(fake)
        loss_g = loss.g_loss(fake_logits, real_logits)
        parameters, owners, seen = [table], ["table"], {id(table)}
        for role, module in models.items():
            if role == "critic":
                continue
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    owners.append(role)
                    seen.add(id(parameter))
        for name, value in state.items():
            if value.requires_grad:
                parameters.append(value)
                owners.append("router")
        generator_gradients = torch.autograd.grad(loss_g, parameters, allow_unused=True)
        if generator_gradients[0] is None:
            raise ValueError("the complete callback must use candidate.table in its noisy game")
        role_gradients = {}
        for role in dict.fromkeys(owners):
            # Keep a stable parameter coordinate system even when a conditional
            # branch leaves a parameter unused in one arm or observation.
            values = [gradient.detach().flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
                      for owner, parameter, gradient in zip(owners, parameters, generator_gradients) if owner == role]
            if values:
                role_gradients[role] = torch.cat(values)
        gradient = generator_gradients[0].detach().double()
        # Shared output noise reaches the same learned feature map as the game.
        # The public feature callback operates on raw outputs relative to target.
        noisy_features = rows.features(models, context, output.detach() + scale * noise, targets)
        reference = rows.features(models, context, targets + scale * noise, targets)
        feature_error = (noisy_features.double() - reference.double()).square().mean()
        clean = rows.forward(models, context, candidate)
        clean_features = rows.features(models, context, clean, targets)
        clean_reference = rows.features(models, context, targets, targets)
        clean_feature_error = (clean_features.double() - clean_reference.double()).square().mean()
        critic_features = rows.features(models, context, critic_output + scale * critic_real, targets)
        critic_reference = rows.features(models, context, targets + scale * critic_real, targets)
        critic_feature_error = (critic_features.double() - critic_reference.double()).square().mean()
        role_gradients["critic"] = critic_gradient
        mean_usage = usage.detach().double().mean(0)
        h = torch.where(mean_usage[:, None] > 0, gradient / mean_usage[:, None].clamp_min(1e-300).sqrt(),
                        torch.zeros_like(gradient))
        spectrum = torch.linalg.eigvalsh(h.T @ h).clamp_min(0)
        spectrum = spectrum / spectrum.sum().clamp_min(1e-300)
        nonzero = spectrum[spectrum > 0]
        rank = float((-nonzero * nonzero.log()).sum().exp()) if len(nonzero) else 0.
        stats = bundle["penalty"].last_stats
        gap = (fake_logits.detach() - real_logits).double()
        result = dict(loss_g=float(loss_g.detach()), loss_d_game=float(loss_d_game.detach()),
                      loss_d_total=float(loss_d.detach()), penalty=float(penalty.detach()),
                      penalty_phase=stats.get("phase", "lazy_skip"), penalty_stats=deepcopy(stats),
                      penalty_calls_before=calls_before, penalty_calls_after=bundle["critic_optimizer"].record.calls,
                      real_score=float(real_logits.double().mean()), fake_score=float(fake_logits.detach().double().mean()),
                      score_gap_mean=float(gap.mean()), score_gap_std=float(gap.std(unbiased=False)),
                      noisy_feature_error=float(feature_error.detach()), clean_feature_error=float(clean_feature_error.detach()),
                      critic_noisy_feature_error=float(critic_feature_error.detach()),
                      table_gradient_norm=float(gradient.norm()), mass_gradient_effective_rank=rank,
                      mass_gradient_concentration=float(spectrum.max()),
                      row_usage=mean_usage.tolist(),
                      row_context_ess=(usage.detach().double().sum(0).square()
                                       / usage.detach().double().square().sum(0).clamp_min(1e-300)).tolist(),
                      row_gradient_norm=gradient.norm(dim=1).tolist(), table_gradient=gradient.tolist(),
                      role_gradient_norm={role: float(value.double().norm()) for role, value in role_gradients.items()},
                      role_gradient={role: value.double().tolist() for role, value in role_gradients.items() if role != "table"},
                      dv12=applications, latent_bandwidth=controller.latent_bandwidth.detach().cpu().tolist())
        if output_log:
            result["output_mse_log"] = float((clean.detach().double() - targets.double()).square().mean())
        if any(not torch.equal(first, second) for first, second in zip(rng_before, _default_rng(table.device))):
            raise ValueError("replay callbacks must be deterministic and leave default RNG untouched")
        return result, role_gradients

    def compare(self, context, targets, before, after, *, residual_scale,
                prospective_bandwidth=False, log_output_error=False, split_rows=None):
        """Replay two complete candidates under identical private random draws.

        Each replicate freezes the same critic and next KA2 phase. Current
        bandwidth measures this update's noise law; prospective bandwidth
        applies one observe_prior(candidate) on each controller copy. These
        frozen-gradient diagnostics do not observe temporal persistence, critic
        adaptation, optimizer response, or future game repair. Output error is
        optional logging and is never used to rank, veto, or select a candidate.
        ``split_rows=(parent, child)`` also reports parent-vs-summed-child gradient
        cosine, avoiding a raw-coordinate comparison after mass refinement.
        """
        if type(prospective_bandwidth) is not bool or type(log_output_error) is not bool:
            raise ValueError("game replay options must be booleans")
        if not all(isinstance(candidate, RoutedCandidate) for candidate in (before, after)):
            raise TypeError("compare needs functional before/after RoutedCandidate states")
        table = self._bundle["table"]
        if (not isinstance(context, torch.Tensor) or not isinstance(targets, torch.Tensor)
                or context.device != table.device or targets.device != table.device
                or context.ndim < 2 or targets.ndim < 2 or not len(context) or len(context) != len(targets)
                or not targets.is_floating_point() or not bool(torch.isfinite(context).all())
                or not bool(torch.isfinite(targets).all())):
            raise ValueError("game replay needs paired nonempty contexts and targets on the model device")
        for candidate in (before, after):
            if (candidate.table.shape != table.shape or candidate.table.device != table.device
                    or candidate.table.dtype != table.dtype or candidate.row_state.keys() != self.candidate().row_state.keys()
                    or candidate.log_mass.shape != (len(table),)
                    or not torch.equal(candidate.log_mass, candidate.row_state[self._bundle["rows"].log_mass_key])):
                raise ValueError("functional candidates must retain captured table and row ownership shapes")
        if split_rows is not None:
            if (not isinstance(split_rows, (tuple, list)) or len(split_rows) != 2
                    or any(type(row) is not int or not 0 <= row < len(table) for row in split_rows)
                    or split_rows[0] == split_rows[1]):
                raise ValueError("split_rows must name distinct valid parent and child rows")
        scale = torch.as_tensor(residual_scale, device=targets.device, dtype=targets.dtype)
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("residual_scale must be finite and positive")
        devices = [table.device.index] if table.device.type == "cuda" else []
        draws = []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            for latent_state, paired_state in zip(self.latent_states, self.paired_states):
                first, first_gradients = self._arm(context, targets, before, scale, latent_state, paired_state,
                                                  prospective_bandwidth, log_output_error)
                second, second_gradients = self._arm(context, targets, after, scale, latent_state, paired_state,
                                                     prospective_bandwidth, log_output_error)
                if [entry["shape"] for entry in first["dv12"]] != [entry["shape"] for entry in second["dv12"]]:
                    raise ValueError("common-noise candidates must preserve per-site mixed-code shapes")
                delta = {name: second[name] - value for name, value in first.items()
                         if type(value) is float and type(second.get(name)) is float}
                cosines = {role: _cosine(value, second_gradients[role]) for role, value in first_gradients.items()
                           if role != "table" and role in second_gradients}
                draw = dict(before=first, after=second, delta=delta, global_gradient_cosine=cosines)
                if split_rows is not None:
                    parent, child = split_rows
                    g_before = torch.tensor(first["table_gradient"], dtype=torch.float64)
                    g_after = torch.tensor(second["table_gradient"], dtype=torch.float64)
                    draw["split_aggregate_table_gradient_cosine"] = _cosine(g_before[parent], g_after[parent] + g_after[child])
                draws.append(draw)
        names = draws[0]["delta"]
        return dict(probe="fixed_critic_gradient_response", noise_law="mass_atoms_v1",
                    bandwidth_mode="prospective" if prospective_bandwidth else "current",
                    penalty_units=self.penalty_units, output_sigma=self.output_sigma, draws=draws,
                    mean_delta={name: sum(draw["delta"][name] for draw in draws) / len(draws) for name in names})


def main():
    from e22_routed_sites import make_loop, update
    torch.set_num_threads(1)
    loop = make_loop(initialization="api", tokens=8, particles=16)
    update(loop)
    p = loop.policy
    # These diagnostic streams are caller-owned; saving their states suffices
    # to repeat the same observation after restoring the policy checkpoint.
    latent = torch.Generator(device=p.device).manual_seed(71)
    paired = torch.Generator(device=p.device).manual_seed(72)
    replay = capture_game(models={"generator": p.G, "encoder": p.encoder, "router": p.router, "critic": p.D},
                          table=p.table, controller=p.controller, critic_optimizer=p.opt_d, penalty=p.penalty,
                          rows=p.routed_control.spec, output_sigma=p.output_sigma(),
                          latent_rng_states=latent.get_state(), paired_rng_states=paired.get_state(),
                          penalty_units=loop.config["penalty_units"])
    candidate = replay.candidate()
    receipt = replay.compare(loop.guard_context[:8], loop.guard_targets[:8], candidate, candidate,
                             residual_scale=p.D.scale, log_output_error=True)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
