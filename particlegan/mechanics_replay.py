"""Same-batch extragradient of the realized normalized G/prior proposal.

The critic is fixed after its ordinary update. Probe optimizer clocks, public
state, streams and records roll back; G hooks observe only the final commit.
External D hooks/caller callback effects cannot roll back after an exception.
No game monotonicity, scalar Lyapunov function or convergence is assumed.
"""
from contextlib import contextmanager
from copy import deepcopy
import math

import torch
from torch import nn
from .optim.dualnorm import NormalizedOptimizer


def validate_recipe(recipe):
    if (recipe.model != "gan" or recipe.conditioning != "scalar" or recipe.encoder_mode != "none"
            or recipe.effective_critic_formulation != "bispf" or recipe.optimizer_family != "dualnorm"
            or recipe.optimizer_momentum != 0 or recipe.constraint_geometry_mode != "direction_blend"
            or recipe.bispf_thermal_application != "gradient"
            or recipe.input_noise_std or recipe.output_noise_std or recipe.standardize
            or recipe.continuous_policy is not None or recipe.lr_control != "mobility"
            or recipe.particle_birth_death or recipe.serve_average
            or recipe.row_evidence_gate or recipe.reopen_guard is not None
            or recipe.prior_proximal_strength):
        raise ValueError("secant_extragradient needs deterministic, zero-momentum DualNorm BISPF direction_blend without extra policy controls/noise")


def validate(trainer):
    validate_recipe(trainer.recipe)
    for owner in (trainer.G, trainer.D):
        for module in owner.modules():
            if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd,
                                   nn.RNNBase, nn.MultiheadAttention)):
                raise ValueError("secant_extragradient does not support stochastic or mutable-buffer models")


@contextmanager
def silent_optimizer_steps(optimizer):
    """Transient proposals cannot emit local/global optimizer step hooks."""
    import torch.optim.optimizer as implementation
    names = ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks")
    local = {name: getattr(optimizer, name) for name in names}
    global_names = ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks")
    global_hooks = {name: getattr(implementation, name) for name in global_names}
    try:
        for name, value in local.items():
            setattr(optimizer, name, type(value)())
        for name, value in global_hooks.items():
            setattr(implementation, name, type(value)())
        yield
    finally:
        for name, value in local.items():
            setattr(optimizer, name, value)
        for name, value in global_hooks.items():
            setattr(implementation, name, value)


class ReplayState:
    """Full public state plus deliberately transient autograd/policy details."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.public = trainer.state_dict()
        self.parameters = [p for group in trainer.opt_g.param_groups for p in group["params"]]
        self.gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters]
        self.critic_gradients = [(p, None if p.grad is None else p.grad.detach().clone())
                                 for p in trainer.D.parameters()]
        self.modes = [(m, m.training) for owner in (trainer.G, trainer.D, trainer.prior)
                      for m in owner.modules()]
        self.phase = trainer.policy._phase
        self.noise = trainer.policy._noise
        self.buffers = [(b, b.detach().clone()) for owner in (trainer.G, trainer.D, trainer.prior)
                        for b in owner.buffers()]
        self.record = trainer.penalty.regularizer.record
        self.last_logits = getattr(self.record, "_last_d_logits", None)

    def check_purity(self):
        trainer = self.trainer
        for name, before in self.public["streams"].items():
            if not torch.equal(getattr(trainer, name).get_state().cpu(), before.cpu()):
                raise ValueError("secant_extragradient replay consumed a named RNG stream")
        if not torch.equal(torch.get_rng_state(), self.public["cpu_rng"].cpu()):
            raise ValueError("secant_extragradient replay consumed global RNG")
        if trainer.device.type == "cuda" and not torch.equal(
                torch.cuda.get_rng_state(trainer.device), self.public["cuda_rng"].cpu()):
            raise ValueError("secant_extragradient replay consumed global CUDA RNG")
        for current, before in self.buffers:
            if not torch.equal(current, before):
                raise ValueError("secant_extragradient replay mutated a model buffer")

    def restore(self):
        self.trainer.load_state_dict(self.public)
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = None if gradient is None else gradient.clone()
        for parameter, gradient in self.critic_gradients:
            parameter.grad = None if gradient is None else gradient.clone()
        for module, training in self.modes:
            module.training = training
        self.trainer.policy._phase = self.phase
        self.trainer.policy._noise = self.noise
        self.record._last_d_logits = self.last_logits


def _norm(pieces):
    return math.sqrt(sum(float(piece.double().square().sum()) for piece in pieces))


@torch.no_grad()
def _put(parameters, anchor, targets, scale):
    for parameter, old, target in zip(parameters, anchor, targets):
        # Untouched rows, including signed zeros, retain their original bits.
        if scale == 1.:
            parameter.copy_(target)
        elif scale == 0.:
            parameter.copy_(old)
        else:
            parameter.copy_(torch.where(target == old, old, torch.lerp(old, target, scale)))


@contextmanager
def one_observable_commit(optimizer):
    """Emit one hook pair around the final applied move, not nested base calls.

    DirectionBlend invokes NormalizedOptimizer.step internally; PyTorch wraps
    both methods. Calling those nested wrappers normally notifies hooks twice.
    Probes notify neither, and this committed operation notifies each once.
    """
    import torch.optim.optimizer as implementation
    pre = [*implementation._global_optimizer_pre_hooks.values(),
           *optimizer._optimizer_step_pre_hooks.values()]
    post = [*optimizer._optimizer_step_post_hooks.values(),
            *implementation._global_optimizer_post_hooks.values()]
    args, kwargs = (optimizer,), {}
    for hook in pre:
        result = hook(optimizer, args, kwargs)
        if result is not None and result != (args, kwargs):
            raise ValueError("secant_extragradient does not support step hooks changing arguments")
    with silent_optimizer_steps(optimizer):
        yield
    for hook in post:
        hook(optimizer, args, kwargs)


def extragradient_step(trainer, objective, *, anchor_losses, critic_loss, collect_stats=False):
    """One committed G/prior clock and at most nine same-batch trial gradients.

    ``objective`` must replay existing real/latent/perturbation tensors without
    requesting data or samples. The secant certificate uses UNPROTECTED,
    realized normalized proposals; fixed-anchor direction protection can hide
    raw force reversals. Original anchor protection is applied only to the
    final committed proposal. The dyadic scale and .5 consistency margin are
    numerical method constants, never task-selected settings.
    """
    optimizer = trainer.opt_g
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    originals = [p.detach().clone() for p in parameters]
    state = ReplayState(trainer)
    anchor_normals = optimizer._protected.detach().clone()
    trials, accepted, scale, nonfinite_trials = 0, False, 0., 0
    consistency_lhs = consistency_rhs = None
    selected_gradients = state.gradients
    selected_losses = anchor_losses
    try:
        if (not bool(torch.isfinite(anchor_losses["loss_g"]).all())
                or not all(g is None or bool(torch.isfinite(g).all()) for g in state.gradients)):
            raise ValueError("secant_extragradient requires a finite anchor loss and gradient")
        with silent_optimizer_steps(optimizer):
            NormalizedOptimizer.step(optimizer)
        first_targets = [p.detach().clone() for p in parameters]
        p0 = [p.detach() - old for p, old in zip(parameters, originals)]
        if not all(bool(torch.isfinite(piece).all()) for piece in p0):
            raise ValueError("secant_extragradient requires a finite anchor normalized proposal")
        state.check_purity()
        state.restore()
        if _norm(p0) == 0:
            accepted, scale = True, 1.
            p1 = p0
        else:
            for trial in range(9):
                scale = 2. ** -trial
                _put(parameters, originals, first_targets, scale)
                predictor = [p.detach().clone() for p in parameters]
                losses = objective()
                trials += 1
                if not bool(torch.isfinite(losses["loss_g"]).all()):
                    nonfinite_trials += 1
                    state.check_purity()
                    state.restore()
                    continue
                optimizer.zero_grad()
                losses["loss_g"].backward()
                gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
                state.check_purity()
                state.restore()
                if not all(g is None or bool(torch.isfinite(g).all()) for g in gradients):
                    nonfinite_trials += 1
                    continue
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad = None if gradient is None else gradient.clone()
                optimizer._protected = anchor_normals.clone()
                with silent_optimizer_steps(optimizer):
                    NormalizedOptimizer.step(optimizer)
                # Both normalized proposals are realized from the SAME anchor.
                # Protection is excluded: reflecting a force reversal would
                # incorrectly make this secant appear small.
                p1 = [p.detach() - old for p, old in zip(parameters, originals)]
                prediction = [old - anchor for old, anchor in zip(predictor, originals)]
                difference = [one - zero for one, zero in zip(p1, p0)]
                finite = all(bool(torch.isfinite(piece).all()) for piece in p1)
                state.check_purity()
                state.restore()
                consistency_lhs, consistency_rhs = scale * _norm(difference), .5 * _norm(prediction)
                if finite and consistency_lhs <= consistency_rhs:
                    accepted = True
                    selected_gradients, selected_losses = gradients, losses
                    break
        # Probe state is fully restored. Only this call reaches step observers
        # and consumes optimizer clocks/sampled-row ownership.
        for parameter, gradient in zip(parameters, selected_gradients):
            parameter.grad = None if gradient is None else gradient.clone()
        optimizer._protected = anchor_normals.clone()
        trainer.policy.after_generator_backward(loss_gan=selected_losses["loss_gan"].detach(),
                                                loss_critic=critic_loss.detach())
        applied_scale = scale if accepted else 0.
        with one_observable_commit(optimizer):
            optimizer.step()
            committed = [p.detach() - old for p, old in zip(parameters, originals)]
            if not all(bool(torch.isfinite(piece).all()) for piece in committed):
                raise ValueError("secant_extragradient final protected proposal is nonfinite")
            targets = [p.detach().clone() for p in parameters]
            _put(parameters, originals, targets, applied_scale)
        trainer.policy.after_generator_step()
        applied = [p.detach().double() - old.double() for p, old in zip(parameters, originals)]
        stats = dict(trials=trials, accepted=accepted, applied_scale=applied_scale,
                     base_proposal_l2=_norm(p0), corrected_proposal_l2=_norm(committed),
                     raw_corrected_proposal_l2=_norm(p1) if accepted else None,
                     protection_delta_l2=_norm([one - raw for one, raw in zip(committed, p1)])
                     if accepted else None,
                     certificate_scope="pre_protection",
                     applied_l2=_norm(applied), field_evaluations=trials + 1,
                     consistency_lhs=consistency_lhs, consistency_rhs=consistency_rhs,
                     nonfinite_trials=nonfinite_trials,
                     optimizer_clocks=1, additional_sampling_draws=0)
        return stats
    except Exception:
        state.restore()
        raise
