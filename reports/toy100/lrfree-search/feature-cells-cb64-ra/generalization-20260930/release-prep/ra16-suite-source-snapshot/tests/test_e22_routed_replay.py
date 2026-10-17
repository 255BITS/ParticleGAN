"""Bounded activation-checkpoint replay conformance for the full two-site model."""
import importlib
import io
import math
from pathlib import Path
import runpy

import pytest
import torch


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def api():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(EXAMPLES))
        sites = importlib.import_module("e22_routed_sites")
        replay = runpy.run_path(str(EXAMPLES / "e22_routed_replay.py"))
    return sites, replay["checkpointed_generate"]


def gradients(policy):
    parameters = {f"{role}.{name}": value
                  for role, model in policy._training_modules().items()
                  for name, value in model.named_parameters()}
    parameters.update(table=policy.table, noise=policy.log_output_sigma)
    return {name: None if value.grad is None else value.grad.detach().clone()
            for name, value in parameters.items()}


def clear_gradients(policy, context):
    for model in policy._training_modules().values():
        model.zero_grad(set_to_none=True)
    policy.table.grad = None
    policy.log_output_sigma.grad = None
    context.grad = None


def track_executions(policy):
    spec, executions = policy.routed_control.spec, []
    original = spec.model_forward

    def observed(models, context, candidate, routing):
        entry = {"execution": routing, "sites": []}
        executions.append(entry)
        mix = routing.mix

        def traced_mix(name, logits):
            entry["sites"].append(name)
            return mix(name, logits)

        routing.mix = traced_mix
        return original(models, context, candidate, routing)

    spec.model_forward = observed
    return executions


@pytest.mark.parametrize("backward_count", [1, 2])
def test_full_prediction_gradients_fresh_executions_and_rng_diagnostics_match(api, backward_count, monkeypatch):
    sites, checkpointed_generate = api
    native, replay = sites.make_loop(), sites.make_loop()
    contexts = [loop.fit_context[:4].detach().clone().requires_grad_(True) for loop in (native, replay)]
    executions = [track_executions(loop.policy) for loop in (native, replay)]
    perturbations = [[], []]
    owners = {id(loop.policy.controller): (loop.policy, calls)
              for loop, calls in zip((native, replay), perturbations)}
    controller_type = type(native.policy.controller)
    original_perturb = controller_type.perturb_latent

    def observed_perturb(controller, latent, stream, prior=None, record=False):
        policy, calls = owners[id(controller)]
        calls.append({"record": record, "stream": stream,
                      "training_stream": stream is policy.noise_generator, "shape": tuple(latent.shape)})
        return original_perturb(controller, latent, stream, prior, record=record)

    # Patch the class rather than adding a callable to a controller's serialized
    # instance dictionary. Recorded stream objects also establish identity.
    monkeypatch.setattr(controller_type, "perturb_latent", observed_perturb)
    for loop, context in zip((native, replay), contexts):
        batch = sites.RoutedBatch(context, loop.fit_targets[:4], loop.guard_context, loop.guard_targets)
        loop.policy.begin_step(loop.fit_targets[:4], routed=batch)
    default_rng = torch.get_rng_state().clone()
    with torch.autograd.set_multithreading_enabled(False):
        prediction = native.policy.routed_generate(contexts[0], sigma=0, perturb=True)
        checkpointed = checkpointed_generate(replay.policy, contexts[1])
        assert_tree_equal(prediction, checkpointed)
        assert len(executions[0]) == len(executions[1]) == 1
        training_rng = replay.policy.noise_generator.get_state().clone()
        diagnostics = replay.policy.controller.state_dict()
        owner = replay.policy.noise_generator
        probe = torch.linspace(-.3, .7, prediction.numel()).reshape_as(prediction)
        losses = [(prediction * probe).sum(), (checkpointed * probe).sum()]
        for index in range(backward_count):
            for loop, context in zip((native, replay), contexts):
                clear_gradients(loop.policy, context)
            for loss in losses:
                loss.backward(retain_graph=index + 1 < backward_count)
            assert_tree_equal(gradients(native.policy), gradients(replay.policy))
            assert_tree_equal(contexts[0].grad, contexts[1].grad)
            assert replay.policy.table.grad.norm(dim=-1).gt(0).all()
            assert replay.policy.noise_generator is owner
            assert_tree_equal(replay.policy.noise_generator.get_state(), training_rng)
            assert_tree_equal(replay.policy.controller.state_dict(), diagnostics)
            assert_tree_equal(native.policy.controller.state_dict(), diagnostics)
            assert_tree_equal(native.policy.noise_generator.get_state(), training_rng)
    assert_tree_equal(torch.get_rng_state(), default_rng)
    assert len(executions[0]) == 1
    assert len(executions[1]) == 1 + backward_count
    assert len(perturbations[0]) == 2
    assert len(perturbations[1]) == 2 * (1 + backward_count)
    assert all(call["record"] and call["training_stream"] for call in perturbations[0])
    assert all(call["record"] and call["training_stream"] for call in perturbations[1][:2])
    assert all(not call["record"] and not call["training_stream"] for call in perturbations[1][2:])
    assert all(call["shape"] == (32, 2) for calls in perturbations for call in calls)
    private_streams = [perturbations[1][2 + 2 * index]["stream"] for index in range(backward_count)]
    assert len({id(stream) for stream in private_streams}) == backward_count
    for index, stream in enumerate(private_streams):
        assert perturbations[1][3 + 2 * index]["stream"] is stream
    replay_objects = [entry["execution"] for entry in executions[1]]
    assert len({id(execution) for execution in replay_objects}) == len(replay_objects)
    for entry in executions[0] + executions[1]:
        assert entry["sites"] == ["first", "second"]
        assert entry["execution"]._candidate is None
        with pytest.raises(ValueError, match="only during"):
            entry["execution"].mix("first", torch.zeros(4, 8, 16))
    native.policy.abort_step()
    replay.policy.abort_step()


def adversarial_replay_conformance(api, *, device, steps=8):
    sites, checkpointed_generate = api
    native, replay = sites.make_loop(device=device), sites.make_loop(device=device)
    rows, before_move = [], None
    with torch.autograd.set_multithreading_enabled(False):
        for _ in range(steps):
            previous = sites.checkpoint(replay)
            expected = sites.update(native)
            actual = sites.update(replay, generator_forward=checkpointed_generate)
            assert_tree_equal(actual, expected)
            assert_tree_equal(gradients(replay.policy), gradients(native.policy))
            assert_tree_equal(sites.checkpoint(replay), sites.checkpoint(native))
            rows.append(actual)
            if actual["move"] and actual["move"].get("moves", 0):
                before_move = previous
    assert before_move is not None and before_move["policy"]["completed_steps"] > 0
    final = sites.checkpoint(replay)
    resumed = sites.make_loop(device=device)
    sites.restore(resumed, cpu_roundtrip(before_move))
    start = resumed.policy.completed_steps
    assert rows[start]["move"]["accepted"] and rows[start]["move"]["moves"] > 0
    with torch.autograd.set_multithreading_enabled(False):
        for expected in rows[start:]:
            assert_tree_equal(sites.update(resumed, generator_forward=checkpointed_generate), expected)
    assert_tree_equal(sites.checkpoint(resumed), final)
    assert_tree_equal(sites.evaluate(resumed), sites.evaluate(native))
    assert_tree_equal(resumed.policy.served_snapshot(), native.policy.served_snapshot())
    assert_tree_equal(resumed.policy.served_model().routed_forward(resumed.test_context),
                      native.policy.served_model().routed_forward(native.test_context))
    if device == "cuda":
        assert not resumed.policy.G.first_host.weight.requires_grad
        assert resumed.policy.G.first_host.weight.dtype == torch.bfloat16
        assert resumed.policy.G.second_host.weight.dtype == torch.bfloat16
        assert resumed.policy.G.first_adapter.weight.dtype == torch.float32
        assert resumed.policy.table.dtype == torch.float32


def test_checkpointed_adversarial_updates_moves_resume_and_serving_match_native(api):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        adversarial_replay_conformance(api, device="cpu")
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_checkpoint_replay_matches_native_and_cpu_loaded_event_resume(api):
    adversarial_replay_conformance(api, device="cuda", steps=6)
