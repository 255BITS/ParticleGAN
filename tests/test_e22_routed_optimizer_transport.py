"""Late routed splits retain a coherent Adam age and transported history."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan.k3p import K3PGeneratorAdam
from particlegan.routing import RoutedRows


class RowRouter(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.log_mass = nn.Parameter(torch.tensor([-.9, 0., .2, -.1], device=device, dtype=dtype))
        self.head = nn.Parameter(torch.arange(8, device=device, dtype=dtype).reshape(4, 2) * .03)
        self.query = nn.Parameter(torch.tensor([.2, -.1], device=device, dtype=dtype))


def route(models, context, candidate):
    query = context + models["router"].query
    logits = query @ candidate.table.T + context @ candidate.row_state["head"].T
    return (logits + candidate.log_mass).softmax(1)


def generate(models, context, candidate, weights):
    return models["generator"](candidate.codes)


def features(models, context, samples, targets):
    return samples - targets


def make_control(optimizer_type=torch.optim.Adam, *, betas=(0., .999), amsgrad=False,
                 weight_decay=0., device="cpu", dtype=torch.float64):
    table = nn.Parameter(torch.tensor([[-2., -.6], [1., .1], [1.1, .2], [.9, .3]],
                                      device=device, dtype=dtype))
    router = RowRouter(device, dtype)
    options = dict(lr=2e-5, betas=betas, amsgrad=amsgrad, weight_decay=weight_decay,
                   eps=1e-8, foreach=False)
    table_optimizer = optimizer_type([table], **options,
                                     **({"latent_table": table} if optimizer_type is K3PGeneratorAdam else {}))
    router_optimizer = optimizer_type(router.parameters(), **options)
    models = {"generator": nn.Identity(), "router": router, "critic": nn.Identity()}
    averages = {name: deepcopy(module).requires_grad_(False) for name, module in models.items() if name != "critic"}
    spec = RoutedRows(route=route, generate=generate, features=features, row_parameters=("head",))
    control = spec.bind(models=models, averaged_models=averages, table=table,
                        averaged_table=table.detach().clone(), optimizers=(table_optimizer, router_optimizer),
                        table_optimizer=table_optimizer)
    return control, options


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def checkpoint(control):
    return deepcopy(dict(control=control.state_dict(), table=control.table.detach(),
                         average_table=control.averaged_table,
                         models={name: module.state_dict() for name, module in control.models.items()},
                         averages={name: module.state_dict() for name, module in control.averaged_models.items()},
                         optimizers=[optimizer.state_dict() for optimizer in control.optimizers]))


def restore(control, saved):
    with torch.no_grad():
        control.table.copy_(saved["table"])
        control.averaged_table.copy_(saved["average_table"])
    for name, state in saved["models"].items():
        control.models[name].load_state_dict(state)
    for name, state in saved["averages"].items():
        control.averaged_models[name].load_state_dict(state)
    for optimizer, state in zip(control.optimizers, saved["optimizers"]):
        optimizer.load_state_dict(deepcopy(state))
    control.load_state_dict(saved["control"])


def linear_backward(parameters, gradients):
    # Actual differentiable objectives supply gradients during both aging and
    # next-step checks; no moment arrays or scalar clocks are synthetically aged.
    sum((parameter * gradient).sum() for parameter, gradient in zip(parameters, gradients)).backward()


CASES = [(torch.optim.Adam, (0., .999), False, 0.),
         (torch.optim.Adam, (.8, .999), True, .05),
         (torch.optim.AdamW, (.9, .999), True, .1),
         (K3PGeneratorAdam, (0., .999), True, 0.)]


@pytest.mark.parametrize("optimizer_type,betas,amsgrad,weight_decay", CASES,
                         ids=("e22-adam", "adam-momentum-amsgrad-coupled-decay", "adamw-amsgrad", "native-a2-adam"))
def test_late_split_next_update_matches_same_age_scaled_history(optimizer_type, betas, amsgrad, weight_decay):
    receipt = run_late_split(optimizer_type, betas, amsgrad, weight_decay, device="cpu", age=2048)
    print(f"{optimizer_type.__name__} beta1={betas[0]} amsgrad={amsgrad} decay={weight_decay}: {receipt}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_split_next_update_and_exact_checkpoint_recovery():
    receipt = run_late_split(torch.optim.Adam, (0., .999), True, 0., device="cuda:0", age=256)
    print(f"CUDA Adam AMSGrad: {receipt}")


def run_late_split(optimizer_type, betas, amsgrad, weight_decay, *, device, age):
    control, options = make_control(optimizer_type, betas=betas, amsgrad=amsgrad,
                                    weight_decay=weight_decay, device=device)
    parent, child = 2, 0
    rows = torch.tensor([parent, child], device=device)
    bindings = {name: (tensor, optimizer) for name, (tensor, _, optimizer) in control._bindings.items()
                if optimizer is not None}
    all_parameters = [control.table, *control.models["router"].parameters()]
    gradients = {parameter: (torch.arange(parameter.numel(), device=device, dtype=parameter.dtype)
                             .reshape(parameter.shape) + 1) * .07 for parameter in all_parameters}
    references = {name: nn.Parameter(tensor[parent].detach().expand(2, *tensor.shape[1:]).clone())
                  for name, (tensor, _) in bindings.items()}
    reference_optimizer = optimizer_type(references.values(), **options)

    for iteration in range(age):
        for optimizer in (*control.optimizers, reference_optimizer):
            optimizer.zero_grad(set_to_none=True)
        current_gradients = dict(gradients)
        if optimizer_type is K3PGeneratorAdam and iteration == 0:
            # Start the genuine A2 history once. Later dense steps exercise its
            # ordinary Adam path, leaving real previous-gradient row history.
            current_gradients[control.table] = gradients[control.table].clone()
            current_gradients[control.table][child] = 0
        reference_gradients = []
        for name, (tensor, _) in bindings.items():
            total_parent = current_gradients[tensor][parent]
            if optimizer_type is not torch.optim.AdamW and weight_decay:
                total_parent = total_parent + weight_decay * tensor[parent].detach()
                # The reference ages on half the parent's TOTAL regularized
                # gradients. Its own decay is cancelled only during aging.
                reference_gradients.append(total_parent.expand_as(references[name]) * .5
                                           - weight_decay * references[name].detach())
            else:
                reference_gradients.append(total_parent.expand_as(references[name]) * .5)
        linear_backward(all_parameters, [current_gradients[parameter] for parameter in all_parameters])
        linear_backward(list(references.values()), reference_gradients)
        reference_optimizer.step()
        for optimizer in control.optimizers:
            optimizer.step()

    moments = {name: deepcopy(optimizer.state[tensor]) for name, (tensor, optimizer) in bindings.items()}
    shared = control.models["router"].query
    shared_state = deepcopy(control.optimizers[1].state[shared])
    shared_weights = shared.detach().clone()
    regularizer = (deepcopy(control.table_optimizer.latent_damping.state_dict())
                   if optimizer_type is K3PGeneratorAdam else None)
    untouched = torch.ones(len(control.table), device=device, dtype=torch.bool)
    untouched[rows] = False
    fast, average = control.candidate(copy=True), control.candidate(averaged=True, copy=True)
    proposal = control._split(fast, child, parent, torch.zeros(control.table.shape[1], device=device, dtype=control.table.dtype))
    average_proposal = control._split(average, child, parent, torch.zeros_like(control.table[0]))
    control._commit(child, parent, proposal, average_proposal)

    tolerance = dict(rtol=3e-12, atol=1e-14)
    for name, (tensor, optimizer) in bindings.items():
        state = optimizer.state[tensor]
        assert float(state["step"]) == age
        for key, factor in (("exp_avg", .5), ("exp_avg_sq", .25), ("max_exp_avg_sq", .25)):
            if key not in state:
                continue
            same(state[key][untouched], moments[name][key][untouched])
            torch.testing.assert_close(state[key][rows],
                                       (moments[name][key][parent] * factor).expand_as(state[key][rows]), rtol=0, atol=0)
            torch.testing.assert_close(state[key][rows], reference_optimizer.state[references[name]][key], **tolerance)
        same(state["step"], moments[name]["step"])
        with torch.no_grad():
            references[name].copy_(tensor[rows])
    same(shared_state, control.optimizers[1].state[shared])
    same(shared_weights, shared)
    if regularizer is not None:
        same(regularizer, control.table_optimizer.latent_damping.state_dict())
        assert not control.table_optimizer.latent_history[rows].any()
        same(control.table_optimizer.latent_history[untouched], gradients[control.table][untouched])

    # Old behavior: zero moments on new rows while retaining the late age.
    old_parameters = {name: nn.Parameter(parameter.detach().clone()) for name, parameter in references.items()}
    old_optimizer = optimizer_type(old_parameters.values(), **options)
    old_optimizer.load_state_dict(deepcopy(reference_optimizer.state_dict()))
    for state in old_optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                state[key].zero_()

    saved = checkpoint(control)
    restored, _ = make_control(optimizer_type, betas=betas, amsgrad=amsgrad,
                               weight_decay=weight_decay, device=device)
    restore(restored, saved)
    same(saved, checkpoint(restored))
    next_gradients = {parameter: gradient.clone() for parameter, gradient in gradients.items()}
    for tensor, _ in bindings.values():
        next_gradients[tensor][rows] = gradients[tensor][parent] * .5
    previous = {name: tensor[rows].detach().clone() for name, (tensor, _) in bindings.items()}
    for optimizer in (*control.optimizers, reference_optimizer, old_optimizer, *restored.optimizers):
        optimizer.zero_grad(set_to_none=True)
    linear_backward(all_parameters, [next_gradients[parameter] for parameter in all_parameters])
    restored_parameters = [restored.table, *restored.models["router"].parameters()]
    linear_backward(restored_parameters, [next_gradients[parameter] for parameter in all_parameters])
    row_gradients = [next_gradients[tensor][rows] for tensor, _ in bindings.values()]
    linear_backward(list(references.values()), row_gradients)
    linear_backward(list(old_parameters.values()), row_gradients)
    for optimizer in (*control.optimizers, reference_optimizer, old_optimizer, *restored.optimizers):
        optimizer.step()
    for name, (tensor, optimizer) in bindings.items():
        torch.testing.assert_close(tensor[rows] - previous[name], references[name] - previous[name], **tolerance)
        assert float(optimizer.state[tensor]["step"]) == age + 1
    same(checkpoint(control), checkpoint(restored))
    actual_step = (control.table[rows] - previous["table"]).detach().norm()
    old_step = (old_parameters["table"] - previous["table"]).detach().norm()
    amplification = float(old_step / actual_step)
    if betas[0] == 0 and weight_decay == 0:
        expected = ((1 - betas[1] ** (age + 1)) / (1 - betas[1])) ** .5
        assert amplification > (25 if age >= 2048 else 10)
        assert amplification == pytest.approx(expected, rel=5e-6)
    reference_error = max(float((tensor[rows] - references[name]).detach().abs().max())
                          for name, (tensor, _) in bindings.items())
    return {"age": age, "moved_table_step_norm": float(actual_step), "old_zero_moment_step_norm": float(old_step),
            "old_amplification": amplification, "max_reference_error": reference_error}


@pytest.mark.parametrize("fault", ("sgd", "unknown_adam_subclass", "extra_layout", "shape", "scalar_age", "direct_response"))
def test_transport_rejects_unsupported_owner_or_state_before_any_mutation(fault):
    control, options = make_control()
    for optimizer in control.optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    if fault in ("sgd", "unknown_adam_subclass"):
        class UnknownAdam(torch.optim.Adam):
            pass
        owner = (torch.optim.SGD(control.models["router"].parameters(), lr=.01) if fault == "sgd"
                 else UnknownAdam(control.models["router"].parameters(), **options))
        control.optimizers = (control.table_optimizer, owner)
        for name, (tensor, averaged, optimizer) in control._bindings.items():
            if name != "table":
                control._bindings[name] = tensor, averaged, owner
    elif fault == "direct_response":
        owner = K3PGeneratorAdam([control.table], direct_particles=[control.table], **options)
        control.table_optimizer = owner
        control.optimizers = (owner, control.optimizers[1])
        control._bindings["table"] = control.table, control.averaged_table, owner
    else:
        # Corrupt a LATER owner: table preflight succeeds, but no table/moment
        # writes may occur before this router failure is detected.
        tensor = control.models["router"].head
        state = control.optimizers[1].state[tensor]
        if fault == "extra_layout":
            state["custom_row_state"] = torch.ones_like(tensor)
        elif fault == "shape":
            state["exp_avg_sq"] = state["exp_avg_sq"][:1]
        elif fault == "scalar_age":
            state["step"] = state["step"].reshape(1)
    before = checkpoint(control)
    parent, child = 2, 0
    proposal = control._split(control.candidate(copy=True), child, parent, torch.zeros_like(control.table[0]))
    average = control._split(control.candidate(averaged=True, copy=True), child, parent, torch.zeros_like(control.table[0]))
    with pytest.raises(ValueError, match="transport|Adam"):
        control.validate_optimizer_transport()
    same(before, checkpoint(control))
    with pytest.raises(ValueError, match="transport|Adam"):
        control._commit(child, parent, proposal, average)
    same(before, checkpoint(control))


def test_empty_supported_optimizer_state_validates_without_initializing_history():
    control, _ = make_control(torch.optim.AdamW, weight_decay=.1, amsgrad=True)
    before = checkpoint(control)
    control.validate_optimizer_transport()
    same(before, checkpoint(control))
