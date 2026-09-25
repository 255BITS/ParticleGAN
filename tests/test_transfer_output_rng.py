"""The optional output stream leaves legacy training RNGs and gates intact."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy100.models import OUTPUT_NOISE_SEED_OFFSET, paired_output_noise
from benchmarks.transfer_suite import image_tasks, vector_tasks
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.toy100_compatibility import (
    _native_noise_receipt, declared_recipe, run_image, run_vector, setup_image,
)
from particlegan import get_recipe


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


def _image_step(*, isolated: bool):
    spec = deepcopy(image_tasks.TASKS[0])
    spec.update(runner="image", steps=24, batch_size=4, particles=8)
    noise = dict(output_noise_std=0.0, input_noise_std=0.5,
                 input_noise_anneal_end=0.5, output_noise_warmup=0.0)
    if isolated:
        noise.update(output_noise_std=0.029, output_noise_rng="isolated")
    context = setup_image(spec, get_recipe(), noise)
    trainer = context["trainer"]
    initial_global = torch.random.get_rng_state().clone()
    centers = context["centers"]
    real = centers[torch.randint(len(centers), (spec["batch_size"],))]
    real = (real + spec["noise_std"] * torch.randn_like(real)).clamp(0, 1)
    trainer.D.sigma = 0.5
    trainer.step(real, generator_real=real)
    return context, initial_global, torch.random.get_rng_state().clone()


def test_image_training_global_data_stream_matches_zero_noise_and_eval_restores_private_stream():
    torch.set_num_threads(1)
    zero, zero_initial, zero_final = _image_step(isolated=False)
    isolated, isolated_initial, isolated_final = _image_step(isolated=True)
    assert torch.equal(zero_initial, isolated_initial)
    assert torch.equal(zero_final, isolated_final)
    trainer = isolated["trainer"]
    private_before = trainer.G.noise_stream.get_state().clone()
    global_before = torch.random.get_rng_state().clone()
    with paired_output_noise((trainer.G, trainer.ema_G), seed=403):
        live = image_tasks.measure(trainer.G, trainer.prior, isolated["centers"],
                                   image_tasks.TASKS[0]["thresholds"])
        ema = image_tasks.measure(trainer.ema_G, trainer.ema_prior,
                                  isolated["centers"], image_tasks.TASKS[0]["thresholds"])
        assert live["modes"] >= 0 and ema["modes"] >= 0
    assert torch.equal(private_before, trainer.G.noise_stream.get_state())
    assert torch.equal(global_before, torch.random.get_rng_state())


def test_vector_and_image_routes_record_real_private_draws_and_eval_restoration(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(vector_tasks, "EVAL_SAMPLES", 256)
    noise = dict(output_noise_std=0.029, input_noise_std=0.5,
                 input_noise_anneal_end=0.5, output_noise_warmup=0.2,
                 output_noise_rng="isolated")
    vector = deepcopy(vector_tasks.TASKS[0])
    vector.update(runner="vector", steps=24, batch=8, particles=16,
                  hidden=8, layers=1)
    image = deepcopy(image_tasks.TASKS[0])
    image.update(runner="image", steps=24, batch_size=4, particles=8, width=4)
    for kind, spec in (("vector", vector), ("image", image)):
        if kind == "vector":
            result, context = run_vector(spec, None, get_recipe(), noise)
        else:
            result, context = run_image(spec, get_recipe(), noise)
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
