"""Target-free, fixed positive particle geometry around the selected base.

No new training signal: only the existing discriminator G gradient is used.
Network updates, data, RNG, observation channels and budgets are unchanged.
Mean and zero-mean particle subspaces have constant strictly positive gains.
"""
from contextlib import contextmanager, ExitStack
import math
from unittest.mock import patch

import torch
from particlegan import ParticlePrior, GANTrainer
from benchmarks import learned_lr_evaluation as bridge
from selected_h_extension import extended_signal_policy


@contextmanager
def geometry_policy(options):
    options = dict(options)
    geometry = options.pop("particle_geometry")
    network_geometry = options.pop("network_geometry", "coordinate_adam")
    if network_geometry not in ("coordinate_adam", "tensor_rms", "row_rms", "radial_adam"):
        raise ValueError("unknown generator network geometry")
    kind = geometry["kind"]
    if kind not in ("contrast_adam", "split_adam"):
        raise ValueError("unknown particle geometry")
    basis = geometry.get("shape_basis", "isotropic")
    if basis not in ("isotropic", "radial"):
        raise ValueError("unknown shape basis")
    shape_gain = geometry["shape_gain"]
    mean_gain = geometry["mean_gain"]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in (shape_gain, mean_gain)):
        raise ValueError("all subspace gains must be fixed and strictly positive")
    prior_ids = set()
    optimizer_roles = {}
    original_init = ParticlePrior.__init__
    original_trainer_init = GANTrainer.__init__
    original_role = bridge.optimizer_role
    original_step = torch.optim.Adam.step
    work = dict(policy="particle_geometry_v1", **geometry,
                extra_forwards=0, extra_backwards=0, transformed_tensors=0,
                updates=[], signal="unchanged logistic relativistic discriminator objective",
                protocol="research only; every host uses this same policy from initialization")
    network_work = dict(policy=network_geometry, transformed_tensors=0, updates=[],
                        epsilon=1e-8, extra_forwards=0, extra_backwards=0)

    def prior_init(prior, *args, **kwargs):
        original_init(prior, *args, **kwargs)
        prior_ids.update(id(p) for p in prior.parameters())

    def trainer_init(trainer, *args, **kwargs):
        original_trainer_init(trainer, *args, **kwargs)
        optimizer_roles[id(trainer.opt_g)] = "g"
        optimizer_roles[id(trainer.opt_d)] = "d"

    def optimizer_role(optimizer, local_variables):
        role = original_role(optimizer, local_variables)
        optimizer_roles[id(optimizer)] = role
        return role

    def adam_step(optimizer, closure=None):
        if closure is not None:
            raise ValueError("explicit adversarial gradients required")
        before = []
        network_before = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                is_prior = id(p) in prior_ids or group.get("_comparison_prior", False)
                if p.grad is not None and is_prior:
                    if p.ndim < 2 or group["betas"][0] != 0:
                        raise ValueError("particle geometry requires particle axis and beta1=0")
                    before.append((p, p.detach().clone(), group))
                elif (p.grad is not None and network_geometry != "coordinate_adam"
                      and optimizer_roles.get(id(optimizer)) == "g"):
                    network_before.append((p, p.detach().clone(), group))
        result = original_step(optimizer)
        with torch.no_grad():
            for p, anchor, group in network_before:
                state = optimizer.state[p]
                step = int(state["step"])
                if network_geometry == "radial_adam":
                    ordinary = p - anchor
                    radial = anchor * ((anchor * ordinary).sum() / (anchor.square().sum() + 1e-12))
                    p.copy_(anchor + ordinary + (shape_gain - 1) * radial)
                    network_work["transformed_tensors"] += 1
                    if step == 1 or step % 50 == 0:
                        network_work["updates"].append(dict(step=step, shape=list(p.shape),
                            radial_rate=float(group["lr"]) * shape_gain,
                            tangent_rate=float(group["lr"]), applied_rms=float((p-anchor).square().mean().sqrt())))
                    continue
                # Preserve vector orientation with a shared positive second
                # moment. Matrix rows use a fan-in block in row_rms; tensor_rms
                # shares across the complete parameter. D is unchanged.
                second = state["exp_avg_sq"] / (1 - group["betas"][1] ** step)
                second = (second.mean(dim=-1, keepdim=True) if network_geometry == "row_rms" and p.ndim > 1
                          else second.mean())
                denom = second.sqrt() + group["eps"]
                first = state["exp_avg"] / (1 - group["betas"][0] ** step)
                p.copy_(anchor - group["lr"] * first / denom)
                network_work["transformed_tensors"] += 1
                if step == 1 or step % 50 == 0:
                    network_work["updates"].append(dict(step=step, shape=list(p.shape),
                        rate=float(group["lr"]), denom_min=float(denom.min()),
                        denom_max=float(denom.max()), applied_rms=float((p-anchor).square().mean().sqrt())))
            for p, anchor, group in before:
                state = optimizer.state[p]
                step = int(state["step"])
                ordinary = p - anchor
                if kind == "contrast_adam":
                    mean_direction = ordinary.mean(dim=0, keepdim=True)
                    shape_direction = ordinary - mean_direction
                else:
                    # Orthogonal projectors on BOTH sides of the adaptive metric:
                    # Q diag(v_shape^-1/2) Q + P diag(v_mean^-1/2) P is positive.
                    # Plain Adam moments stay intact; additional moments serialize
                    # with the optimizer for exact same-policy continuation.
                    mean = p.grad.mean(dim=0, keepdim=True)
                    shape = p.grad - mean
                    beta2 = group["betas"][1]
                    directions = []
                    for name, grad in (("mean", mean), ("shape", shape)):
                        key = "geometry_" + name + "_sq"
                        if key not in state:
                            if step != 1:
                                raise RuntimeError("split geometry needs its own acquired optimizer state")
                            state[key] = torch.zeros_like(grad)
                        state[key].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        denom = (state[key] / (1 - beta2 ** step)).sqrt() + group["eps"]
                        directions.append(grad / denom)
                    mean_direction, shape_direction = directions
                    shape_direction = shape_direction - shape_direction.mean(dim=0, keepdim=True)
                    mean_direction = -group["lr"] * mean_direction
                    shape_direction = -group["lr"] * shape_direction
                if basis == "isotropic":
                    shape_displacement = shape_gain * shape_direction
                else:
                    # A fixed geometry rule, not a time schedule: amplify the
                    # instantaneous radial direction of the centered cloud.
                    # All tangential directions retain gain 1; no radius target
                    # or support bound is imposed. At zero radius this is identity.
                    radial = anchor - anchor.mean(dim=0, keepdim=True)
                    projection = radial * ((radial * shape_direction).sum() /
                                           (radial.square().sum() + 1e-12))
                    shape_displacement = shape_direction + (shape_gain - 1) * projection
                displacement = shape_displacement + mean_gain * mean_direction
                p.copy_(anchor + displacement)
                work["transformed_tensors"] += 1
                if step == 1 or step % 50 == 0:
                    applied = p - anchor
                    work["updates"].append(dict(
                        step=step, shape=list(p.shape), nominal_rate=float(group["lr"]),
                        shape_rate=float(group["lr"]) * shape_gain,
                        tangent_rate=float(group["lr"]) * (shape_gain if basis == "isotropic" else 1.),
                        mean_rate=float(group["lr"]) * mean_gain,
                        ordinary_rms=float(ordinary.square().mean().sqrt()),
                        applied_rms=float(applied.square().mean().sqrt()),
                        applied_mean_rms=float(applied.mean(0).square().mean().sqrt()),
                        adversarial_gradient_dot_displacement=float((p.grad * applied).sum())))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(ParticlePrior, "__init__", prior_init))
        stack.enter_context(patch.object(GANTrainer, "__init__", trainer_init))
        stack.enter_context(patch.object(bridge, "optimizer_role", optimizer_role))
        stack.enter_context(patch.object(torch.optim.Adam, "step", adam_step))
        receipt = stack.enter_context(extended_signal_policy(options))
        receipt["particle_geometry"] = work
        receipt["network_geometry"] = network_work
        receipt["host_extension"]["archived_candidate_unchanged"] = False
        yield receipt
