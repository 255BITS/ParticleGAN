"""The optional output stream leaves legacy training RNGs and gates intact."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy100.models import OUTPUT_NOISE_SEED_OFFSET
from benchmarks.transfer_suite import image_tasks, vector_tasks
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.toy100_compatibility import (
    _native_noise_receipt, declared_recipe, run_image, run_vector,
)
from particlegan import get_recipe
from benchmarks.gan_v3 import gan_v3_recipe


def test_isolated_selector_is_optional_exact_and_requires_positive_noise():
    config = json.loads(Path("configs/toy100/shared_candidate.json").read_text())
    _, old_noise, _ = declared_recipe(config)
    assert "output_noise_rng" not in old_noise
    selected = dict(config, output_noise_rng="isolated")
    _, new_noise, _ = declared_recipe(selected)
    assert new_noise == old_noise | {"output_noise_rng": "isolated"}
    for invalid in (None, False, "global", "ISOLATED", 1901):
        with pytest.raises(ValueError, match="output_noise_rng"):
            declared_recipe(dict(config, output_noise_rng=invalid))
    with pytest.raises(ValueError, match="output_noise_std"):
        declared_recipe(dict(selected, output_noise_std=0.0))


def test_legacy_isolated_output_draws_do_not_advance_global_or_input_stream():
    policy = NoisePolicy(0.029, 0.5, 0.5, 10, seed=17,
                         output_noise_warmup=0.5, output_noise_rng="isolated")
    assert OUTPUT_NOISE_SEED_OFFSET == 1901
    base = torch.zeros(32, 2)
    torch.manual_seed(777)
    policy.set_step(0)
    global_before = torch.random.get_rng_state().clone()
    output_before = policy.output_stream.get_state().clone()
    assert policy.output(base) is base
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(output_before, policy.output_stream.get_state())

    policy.set_step(5)
    input_before = policy.input_stream.get_state().clone()
    first = policy.output(base)
    expected_noise = torch.randn(
        base.shape, generator=torch.Generator().manual_seed(17 + OUTPUT_NOISE_SEED_OFFSET),
    )
    torch.testing.assert_close(first, 0.029 * expected_noise, rtol=0, atol=0)
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(input_before, policy.input_stream.get_state())

    output_training_state = policy.output_stream.get_state().clone()
    with policy.evaluation(7):
        eval_first = policy.output(base)
        policy.input(base)
    assert torch.equal(output_training_state, policy.output_stream.get_state())
    assert torch.equal(input_before, policy.input_stream.get_state())
    assert torch.equal(global_before, torch.random.get_rng_state())
    with policy.evaluation(7):
        eval_repeat = policy.output(base)
    torch.testing.assert_close(eval_first, eval_repeat, rtol=0, atol=0)
    receipt = policy.receipt()
    assert receipt["output_noise_seed"] == 17 + OUTPUT_NOISE_SEED_OFFSET
    assert receipt["output_noise_training_stream_isolated"] is True
    assert receipt["output_noise_eval_state_preserved"] is True
    assert len(receipt["output_noise_eval_state_pairs"]) == 2
    assert receipt["output_train_calls"] == 2
    assert receipt["output_eval_calls"] == 2


def test_image_route_draws_output_noise_from_the_runner_and_measures_without_it():
    """The image hosts train on benchmarks.toy_runner: its output noise has its
    own stream and measurement enumerates the particle table noise-free, so
    an isolated declaration is met by construction."""
    torch.set_num_threads(1)
    spec = deepcopy(image_tasks.TASKS[0])
    spec.update(runner="image", steps=24, batch_size=4, particles=8, width=4)
    noise = dict(output_noise_std=0.029, input_noise_std=0.5, input_noise_anneal_end=0.5,
                 output_noise_warmup=0.2, output_noise_rng="isolated")
    global_before = torch.random.get_rng_state().clone()
    result, context = run_image(spec, gan_v3_recipe(), noise)
    assert torch.equal(global_before, torch.random.get_rng_state())
    recipe = context["host_recipe"]
    assert (recipe.output_noise_std, recipe.input_noise_std) == (0.029, 0.5)
    assert len(result["observations"]) == 24
    assert all(isinstance(point["ema"], dict) for point in result["observations"])
    receipt = context["noise_receipt"]
    assert receipt["step_calls"] == 24 and receipt["input_nonzero_steps"] == 12
    assert receipt["input_train_calls"] == 4 * 12  # real + fake per D and G view
    assert receipt["schedule_mismatches"] == 0
    assert receipt["train_input_applied"] and receipt["train_output_applied"]
    assert receipt["output_train_calls"] == 2 * receipt["output_nonzero_steps"] > 0
    assert receipt["output_noise_eval_state_preserved"] is True
    learned = dict(noise, output_noise_learnable=True)
    with pytest.raises(NotImplementedError, match="learnable output noise"):
        run_image(spec, gan_v3_recipe(), learned)


@pytest.mark.parametrize("which", ["output", "input"])
def test_image_noise_receipt_is_observed_and_fails_on_a_wrong_draw(which, monkeypatch):
    """The receipt measures the noise actually drawn: a runner that applies
    twice the recipe's sigma fails the claim-vs-receipt check."""
    from benchmarks import toy_runner
    torch.set_num_threads(1)
    spec = deepcopy(image_tasks.TASKS[0])
    spec.update(runner="image", steps=24, batch_size=4, particles=8, width=4)
    noise = dict(output_noise_std=0.029, input_noise_std=0.5, input_noise_anneal_end=0.5,
                 output_noise_warmup=0.2, output_noise_rng="isolated")
    name = f"{which}_noise_std"
    original = getattr(toy_runner, name)
    monkeypatch.setattr(toy_runner, name, lambda recipe, step: 2 * original(recipe, step))
    _, context = run_image(spec, gan_v3_recipe(), noise)
    receipt = context["noise_receipt"]
    assert receipt["schedule_mismatches"] > 0
    assert not receipt["train_input_applied"] and not receipt["train_output_applied"]


def test_vector_route_records_real_private_draws_and_eval_restoration(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(vector_tasks, "EVAL_SAMPLES", 256)
    noise = dict(output_noise_std=0.029, input_noise_std=0.5,
                 input_noise_anneal_end=0.5, output_noise_warmup=0.2,
                 output_noise_rng="isolated")
    vector = deepcopy(vector_tasks.TASKS[0])
    vector.update(runner="vector", steps=24, batch=8, particles=16,
                  hidden=8, layers=1)
    for spec in (vector,):
        result, context = run_vector(spec, None, gan_v3_recipe(), noise)
        receipt = _native_noise_receipt(context, noise, spec, result)
        assert len(result["observations"]) == 24
        assert receipt["output_module"] == "IsolatedOutputNoise"
        assert receipt["input_module"] == "StatefulInputNoise"
        assert receipt["output_noise_seed"] == OUTPUT_NOISE_SEED_OFFSET
        assert receipt["output_train_calls"] == 46  # Warmup update zero; two G draws thereafter.
        assert receipt["output_eval_calls"] == 48  # Paired live and EMA measurements.
        assert receipt["output_train_elements"] > 0
        assert receipt["output_eval_elements"] > 0
        assert receipt["output_noise_eval_state_preserved"] is True
        assert len(receipt["output_noise_eval_state_pairs"]) == 24
        assert (receipt["output_noise_train_state_initial_sha256"]
                != receipt["output_noise_train_state_final_sha256"])
