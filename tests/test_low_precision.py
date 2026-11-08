"""Native-format software reference, gradient-penalty and checkpoint contracts."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.precision import prepare
from experiments.forge.state import state_digest
from particlegan import init
from particlegan.low_precision import (
    EmulatedLinear,
    _fp4,
    emulate_linear_training_,
    quantize_dequantize,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def task(format):
    return json.loads((ROOT / f"configs/forge/tasks/precision_gaussian_{format}.json").read_text())


def context_and_trainer(format):
    from experiments.forge.vectorprofiles import build_vector_models
    value = task(format)
    context = task_formulation_context({"recipe_preset": "bispf_optimized"}, value, {"seed": 0}, device="cpu")
    g, d = build_vector_models(context, value["execution"]["host_definition"])
    prepare(context, value, g, d)
    return context, context.build_trainer(g, d, max_steps=1000)


def test_e2m1_values_midpoint_ties_even_saturation_and_sign():
    values = torch.tensor([0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 20.])
    expected = torch.tensor([0., 0., 1., 1., 2., 2., 4., 4., 6.])
    torch.testing.assert_close(_fp4(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(_fp4(-values), -expected, rtol=0, atol=0)


def test_stochastic_fp4_rounding_is_replayable_and_unbiased():
    values = torch.full((32768,), .75)
    a, b = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    first = _fp4(values, generator=a)
    assert torch.equal(first, _fp4(values, generator=b))
    assert set(first.unique().tolist()) == {.5, 1.}
    assert abs(first.mean().item() - .75) < .01


def test_nvfp4_weight_tiles_are_transpose_equivalent_and_padding_is_zero():
    values = torch.randn(17, 35, generator=torch.Generator().manual_seed(0))
    left = quantize_dequantize(values, "nvfp4", weight=True)
    right = quantize_dequantize(values.T, "nvfp4", weight=True).T
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert left.shape == values.shape and torch.isfinite(left).all()
    assert torch.count_nonzero(quantize_dequantize(torch.zeros(1, 2), "nvfp4")) == 0


def test_nvfp4_uses_hierarchical_fp8_block_scales_and_not_integer_levels():
    values = torch.zeros(16, 32)
    values[0, 0] = 2688.  # global scale = 1; first block scale = 448
    values[0, 16:20] = torch.tensor([31., 7., 13., 21.])
    # Second block scale rounds (31/6) to E4M3 value 5. E2M1 then rounds
    # [6.2, 1.4, 2.6, 4.2] to [6, 1.5, 3, 4].
    result = quantize_dequantize(values, "nvfp4", weight=True)
    torch.testing.assert_close(result[0, 16:20], torch.tensor([30., 7.5, 15., 20.]), rtol=0, atol=0)
    assert result[0, 0] == 2688.


def test_conversion_preserves_shared_parameters_and_global_rng():
    linear = torch.nn.Linear(2, 3)
    network = torch.nn.ModuleDict({"left": linear, "right": linear})
    before = torch.get_rng_state().clone()
    parameters = dict(network.named_parameters())
    emulate_linear_training_(network, "nvfp4")
    assert network["left"] is network["right"]
    assert all(dict(network.named_parameters())[name] is value for name, value in parameters.items())
    assert torch.equal(before, torch.get_rng_state())
    assert isinstance(emulate_linear_training_(torch.nn.Linear(1, 2), "mxfp8"), EmulatedLinear)


def test_mxfp8_scale_is_power_of_two_rounded_up_and_blocks_are_independent():
    values = torch.cat((torch.linspace(-5., 5., 32), torch.linspace(-.001, .001, 32))).reshape(1, -1)
    actual = quantize_dequantize(values, "mxfp8")
    rounded = values.bfloat16().float().reshape(1, 2, 32)
    # These two independent scalar maxima require scales 2^-6 and 2^-18.
    scales = torch.tensor([2.**-6, 2.**-18]).reshape(1, 2, 1)
    expected = ((rounded / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mixed = torch.tensor([[1000., 1.], [.0001, .0001]])
    assert not torch.equal(quantize_dequantize(mixed.T, "mxfp8").T, quantize_dequantize(mixed, "mxfp8"))


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_forward_backward_and_critic_penalty_retain_a_finite_training_signal(format):
    layer = EmulatedLinear.from_linear(torch.nn.Linear(3, 5), format)
    init.deterministic_orthogonal_(layer)
    x = torch.randn(7, 3, generator=torch.Generator().manual_seed(0), requires_grad=True)
    output = layer(x)
    gradient = torch.autograd.grad(output.sum(), x, create_graph=True)[0]
    penalty_gradient = torch.autograd.grad(gradient.square().mean(), layer.weight)[0]
    assert torch.isfinite(penalty_gradient).all() and penalty_gradient.abs().max() > 0
    assert layer.calls == {"forward": 1, "dgrad": 1, "wgrad": 1}
    assert not torch.equal(output, torch.nn.functional.linear(x, layer.weight, layer.bias))


@pytest.mark.parametrize("format", ["nvfp4", "mxfp8"])
def test_public_trainer_resume_restores_rounding_streams_and_penalty_state(format):
    first, trainer = context_and_trainer(format)
    batches = torch.randn(3, 128, 1, generator=torch.Generator().manual_seed(0)) * .5 + 2.
    trainer.step(batches[0])
    checkpoint = deepcopy(first.state_dict())
    second, restored = context_and_trainer(format)
    second.load_state_dict(checkpoint)
    for batch in batches[1:]:
        trainer.step(batch)
        restored.step(batch)
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())
    before = {name: layer.rounding_rng.get_state().clone()
              for name, layer in trainer.G.named_modules() if isinstance(layer, EmulatedLinear)}
    trainer.sample(128, generator=torch.Generator().manual_seed(2), output_noise=False)
    assert all(torch.equal(before[name], layer.rounding_rng.get_state())
               for name, layer in trainer.G.named_modules() if isinstance(layer, EmulatedLinear))


def test_precision_conditions_keep_initial_master_parameters_and_other_streams_identical():
    contexts = [context_and_trainer(format) for format in ("fp32", "mxfp8", "nvfp4")]
    for role in ("G", "D", "prior"):
        states = [state_digest(dict(getattr(trainer, role).named_parameters())) for _, trainer in contexts]
        assert len(set(states)) == 1
    assert len({state_digest(context.streams.state_dict()) for context, _ in contexts}) == 1


def test_quantizer_rejects_unknown_formats_and_checkpoint_precision_changes():
    with pytest.raises(ValueError):
        quantize_dequantize(torch.ones(2, 2), "int4")
    a = EmulatedLinear.from_linear(torch.nn.Linear(2, 3), "nvfp4")
    b = EmulatedLinear.from_linear(torch.nn.Linear(2, 3), "mxfp8")
    with pytest.raises(ValueError, match="precision differs"):
        b.load_state_dict(a.state_dict())
