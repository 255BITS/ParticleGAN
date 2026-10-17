"""Private output-noise RNG, replay, and legacy compatibility contracts."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn

from benchmarks.toy100.config import validate_manifest
from benchmarks.toy100.models import (
    OUTPUT_NOISE_SEED_OFFSET, InputNoise, IsolatedOutputNoise, OutputNoise,
    StatefulInputNoise, linear_input_noise, paired_output_noise,
)
from benchmarks.toy100.train import (
    IsolatedNoiseGANTrainer, _noise_state_seed, _set_output_sigma,
    make_trainer, resolve_config, train, verify_source_archive,
)


def _config(**overrides):
    config = {
        "problem": "grid100", "steps": 4, "seed": 37, "device": "cpu", "threads": 1,
        "num_particles": 64, "batch_size": 16,
        "g_hidden": 16, "d_hidden": 16, "n_hidden": 1,
        "eval_samples": 128, "snapshot_samples": 32,
        "eval_interval": 4, "snapshot_interval": 4,
        "early_eval_steps": [0, 4], "log_interval": 4,
        "output_noise_std": .029, "output_noise_warmup": .5,
        "output_noise_rng": "isolated",
        "input_noise_std": .5, "input_noise_anneal_end": .5,
    }
    config.update(overrides)
    return config


def _train_step(trainer, config, real):
    _set_output_sigma(trainer, config, trainer.completed_steps)
    trainer.D.sigma = linear_input_noise(
        config["input_noise_std"], trainer.completed_steps,
        config["steps"], config["input_noise_anneal_end"],
    )
    result = trainer.step(real)
    _set_output_sigma(trainer, config, trainer.completed_steps)
    return {key: value.detach().clone() for key, value in result.items()
            if isinstance(value, torch.Tensor)}


def test_selector_is_global_optional_and_legacy_state_dict_is_unchanged():
    old, _ = resolve_config({"steps": 4})
    assert "output_noise_rng" not in old
    for bad in (None, "global", True, 1):
        with pytest.raises(ValueError, match="output_noise_rng"):
            resolve_config(_config(output_noise_rng=bad))
    with pytest.raises(ValueError, match="requires output_noise_std"):
        resolve_config(_config(output_noise_std=0))
    with pytest.raises(ValueError, match="cannot set output_noise_rng"):
        validate_manifest({**{key: value for key, value in _config().items() if key != "problem"},
                           "problem_overrides": {
            "grid100": {"output_noise_rng": "isolated"},
        }})

    fixed = OutputNoise(nn.Identity(), .029)
    assert not any("extra_state" in key or "noise_draw" in key for key in fixed.state_dict())
    zero = OutputNoise(nn.Identity(), 0)
    state = torch.get_rng_state().clone()
    points = torch.ones(3, 2)
    assert torch.equal(zero(points), points)
    assert torch.equal(state, torch.get_rng_state())

    legacy, recipe = resolve_config(
        {key: value for key, value in _config().items() if key != "output_noise_rng"}
    )
    trainer = make_trainer(legacy, recipe)
    assert type(trainer.G) is OutputNoise and type(trainer.D) is InputNoise
    assert "_extra_state" not in trainer.state_dict()["models"]["G"]
    assert "_extra_state" not in trainer.state_dict()["models"]["D"]


def test_private_stream_pairing_and_exception_restore_rng_and_counters():
    live = IsolatedOutputNoise(nn.Identity(), .029, seed=37, device=torch.device("cpu"))
    ema = deepcopy(live)
    assert live.noise_stream is not ema.noise_stream
    assert torch.equal(live.noise_stream.get_state(), ema.noise_stream.get_state())
    assert "_extra_state" in live.state_dict()
    global_before = torch.get_rng_state().clone()
    inputs = torch.zeros(11, 2)
    before = [model.noise_stream.get_state().clone() for model in (live, ema)]
    with paired_output_noise((live, ema), seed=439):
        a, b = live(inputs), ema(inputs)
        assert torch.equal(a, b)
        assert live.draw_receipt() == ema.draw_receipt() == {"calls": 1, "elements": 22}
    assert torch.equal(global_before, torch.get_rng_state())
    assert all(torch.equal(before[index], model.noise_stream.get_state())
               for index, model in enumerate((live, ema)))
    assert live.draw_receipt() == ema.draw_receipt() == {"calls": 0, "elements": 0}
    with pytest.raises(RuntimeError, match="sentinel"):
        with paired_output_noise((live, ema), seed=440):
            live(inputs)
            raise RuntimeError("sentinel")
    assert all(torch.equal(before[index], model.noise_stream.get_state())
               for index, model in enumerate((live, ema)))
    assert live.draw_receipt() == {"calls": 0, "elements": 0}

    live.std = 0.0
    assert torch.equal(live(inputs), inputs)
    assert live.draw_receipt() == {"calls": 0, "elements": 0}
    assert torch.equal(before[0], live.noise_stream.get_state())
    assert OUTPUT_NOISE_SEED_OFFSET == 1901


def test_positive_input_output_noise_checkpoint_resume_and_invalid_rng_preflight():
    config, recipe = resolve_config(_config())
    trainer = make_trainer(config, recipe)
    assert isinstance(trainer, IsolatedNoiseGANTrainer)
    assert isinstance(trainer.G, IsolatedOutputNoise)
    assert isinstance(trainer.D, StatefulInputNoise)
    data = torch.Generator().manual_seed(902)
    batches = [torch.randn(16, 2, generator=data) for _ in range(3)]
    global_before = torch.get_rng_state().clone()
    _train_step(trainer, config, batches[0])
    _train_step(trainer, config, batches[1])
    assert torch.equal(global_before, torch.get_rng_state())
    checkpoint = trainer.state_dict()
    for name in ("G", "D", "ema_G"):
        assert checkpoint["models"][name]["_extra_state"].dtype == torch.uint8
    assert checkpoint["models"]["G"]["noise_draw_calls"].item() == 2

    resumed = make_trainer(config, recipe)
    resumed.load_state_dict(checkpoint)
    assert torch.equal(trainer.G.noise_stream.get_state(), resumed.G.noise_stream.get_state())
    assert torch.equal(trainer.D.noise_stream.get_state(), resumed.D.noise_stream.get_state())
    assert trainer.G.draw_receipt() == resumed.G.draw_receipt()
    direct = _train_step(trainer, config, batches[2])
    replay = _train_step(resumed, config, batches[2])
    for key in direct:
        torch.testing.assert_close(direct[key], replay[key], rtol=0, atol=0)
    first, second = trainer.state_dict(), resumed.state_dict()
    for name in first["models"]:
        for key in first["models"][name]:
            torch.testing.assert_close(first["models"][name][key], second["models"][name][key], rtol=0, atol=0)

    bad = deepcopy(checkpoint)
    bad["models"]["G"]["noise_draw_calls"].fill_(-1)
    before = resumed.state_dict()
    with pytest.raises(ValueError, match="invalid noise RNG state"):
        resumed.load_state_dict(bad)
    after = resumed.state_dict()
    for name in before["models"]:
        for key in before["models"][name]:
            assert torch.equal(before["models"][name][key], after["models"][name][key])
    assert torch.equal(before["cpu_rng"], after["cpu_rng"])


def test_direct_sampling_varies_with_caller_state_but_preserves_training_streams():
    config, recipe = resolve_config(_config())
    trainer = make_trainer(config, recipe)
    output_before = trainer.G.noise_stream.get_state().clone()
    input_before = trainer.D.noise_stream.get_state().clone()
    latent_before = trainer.latent_generator.get_state().clone()
    global_before = torch.get_rng_state().clone()
    caller = torch.Generator().manual_seed(111)
    first_seed = _noise_state_seed(caller)
    first = trainer.sample(20, generator=caller)
    second_seed = _noise_state_seed(caller)
    second = trainer.sample(20, generator=caller)
    assert first_seed != second_seed and not torch.equal(first, second)
    assert torch.equal(output_before, trainer.G.noise_stream.get_state())
    assert torch.equal(input_before, trainer.D.noise_stream.get_state())
    assert torch.equal(latent_before, trainer.latent_generator.get_state())
    assert torch.equal(global_before, torch.get_rng_state())
    assert trainer.G.draw_receipt() == {"calls": 0, "elements": 0}


def test_native_evaluation_frequency_preserves_training_and_receipts(tmp_path):
    sparse = tmp_path / "sparse"
    dense = tmp_path / "dense"
    sparse_summary = train(_config(), sparse)
    dense_summary = train(_config(early_eval_steps=[0, 1, 2, 3, 4]), dense)
    for summary, directory in ((sparse_summary, sparse), (dense_summary, dense)):
        assert summary["status"] == "complete"
        assert summary["output_noise_rng"] == "isolated"
        receipt = summary["output_noise_rng_receipt"]
        assert receipt["namespace_offset"] == 1901
        assert receipt["training_seed"] == 37 + 1901
        assert receipt["generator_wrapper_class"] == "IsolatedOutputNoise"
        assert receipt["discriminator_wrapper_class"] == "StatefulInputNoise"
        assert receipt["final_live_draw_calls"] == 6  # First update has zero warmup noise.
        assert receipt["final_live_draw_elements"] == 6 * 16 * 2
        assert summary["provenance"]["source_archive_scope"] == "native-policy-public-package-v2"
        verify_source_archive(directory, summary["provenance"])
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        for event in events:
            if event["event"] == "eval":
                assert event["output_noise_rng_state_before_sha256"] == event["output_noise_rng_state_after_sha256"]
                assert event["output_noise_rng_draw_calls_before"] == event["output_noise_rng_draw_calls_after"]
                assert event["output_noise_rng_draw_elements_before"] == event["output_noise_rng_draw_elements_after"]
                assert event["output_noise_rng_eval_seed"] == 37 + 402
        assert [event["output_noise_rng_draw_calls"] for event in events if event["event"] == "train"] == [0, 2, 4, 6]
    assert sparse_summary["output_noise_rng_receipt"]["final_live_state_sha256"] == dense_summary["output_noise_rng_receipt"]["final_live_state_sha256"]
    assert sparse_summary["output_noise_rng_receipt"]["final_input_state_sha256"] == dense_summary["output_noise_rng_receipt"]["final_input_state_sha256"]
    with np.load(sparse / "final_samples.npz") as a, np.load(dense / "final_samples.npz") as b:
        for key in ("live", "ema", "target"):
            np.testing.assert_array_equal(a[key], b[key])


def test_isolated_policy_100k_holdout_is_independent_and_restores_streams(tmp_path):
    directory = tmp_path / "holdout"
    summary = train(_config(
        steps=1000, num_particles=64, batch_size=8,
        g_hidden=8, d_hidden=8, n_hidden=1,
        eval_samples=20_000, snapshot_samples=32,
        eval_interval=250, snapshot_interval=250,
        early_eval_steps=[0, 100], log_interval=1000,
    ), directory)
    receipt = summary["output_noise_rng_receipt"]
    assert summary["status"] == "complete"
    assert summary["holdout"]["live"]["n"] == 100_000
    assert receipt["holdout_live_before_state_sha256"] == receipt["holdout_live_after_state_sha256"]
    assert receipt["holdout_ema_before_state_sha256"] == receipt["holdout_ema_after_state_sha256"]
    assert receipt["final_live_state_sha256"] == receipt["holdout_live_after_state_sha256"]
    assert receipt["final_ema_state_sha256"] == receipt["holdout_ema_after_state_sha256"]
    with np.load(directory / "final_samples.npz") as final, np.load(
        directory / "holdout_samples.npz"
    ) as holdout:
        assert holdout["live"].shape == holdout["ema"].shape == holdout["target"].shape == (100_000, 2)
        assert not np.array_equal(final["target"], holdout["target"][:20_000])
