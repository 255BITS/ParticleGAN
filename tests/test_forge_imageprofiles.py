"""Image profile selection/constructor parity without optimizer updates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import FormulationContext
from experiments.forge.imageprofiles import (
    PROFILE_SOURCE, PROFILE_SHA256, build_image_models, image_profile_blockers,
    profile_declaration, resolve_image_spec, task_from_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def read_task(name):
    return json.loads((ROOT / "configs/forge/tasks" / f"{name}.json").read_text())


def context(task):
    return FormulationContext(seed=0, device="cpu", prior=task["execution"]["prior"])


@pytest.mark.parametrize("name", ["img_intensity2", "img_stripes2", "img_bars4", "img_blobs4"])
def test_variant_is_a_reproducible_explicit_materialization_without_mutating_base(name):
    path = ROOT / "configs/forge/tasks" / f"{name}.json"
    original = path.read_bytes()
    raw = json.loads(original)
    before = deepcopy(raw)
    variant = task_from_profile(raw, name + "_residual16")
    assert variant == read_task(name + "_residual16")
    assert raw == before and path.read_bytes() == original
    assert "image_profile" not in raw["execution"]
    assert raw["execution"]["host_definition"]["architecture"] == "transpose"
    assert raw["execution"]["host_definition"]["width"] == 12
    assert variant["execution"]["image_profile"] == profile_declaration()
    assert variant["evaluation"] == raw["evaluation"]
    assert variant["execution"]["steps"] == raw["execution"]["steps"] == 600
    assert variant["execution"]["prior"] == raw["execution"]["prior"]
    assert variant["resources"] == raw["resources"]
    assert variant["requires_capabilities"] == raw["requires_capabilities"]
    assert variant["dependencies"] == raw["dependencies"]
    differences = {key for key in raw["execution"]["host_definition"]
                   if raw["execution"]["host_definition"][key] != variant["execution"]["host_definition"][key]}
    assert differences == {"architecture", "width"}
    assert hashlib.sha256((ROOT / PROFILE_SOURCE).read_bytes()).hexdigest() == PROFILE_SHA256


@pytest.mark.parametrize("name", ["img_intensity2", "img_bars4", "img_blobs4", "img_stripes2"])
def test_raw_cards_remain_byte_identical_and_construction_preserves_draws(name):
    from benchmarks.transfer_suite.image_tasks import Generator, Discriminator
    task = read_task(name)
    original = deepcopy(task)
    spec = resolve_image_spec(task)
    assert spec == task["execution"]["host_definition"]
    new_context, old_context = context(task), context(task)
    global_before = torch.get_rng_state().clone()
    new_g, new_d = build_image_models(new_context, spec)
    old_g = old_context.construct(lambda: Generator(spec), component="generator")
    old_d = old_context.construct(lambda: Discriminator(spec), component="discriminator")
    assert torch.equal(global_before, torch.get_rng_state())
    assert new_context.streams.audit() == old_context.streams.audit()
    for actual, expected in ((new_g, old_g), (new_d, old_d)):
        assert actual.state_dict().keys() == expected.state_dict().keys()
        assert all(torch.equal(value, expected.state_dict()[key]) for key, value in actual.state_dict().items())
    assert task == original
    spec["width"] = 99
    assert task == original  # The resolver returns an independent value.


def test_residual16_constructor_matches_published_shapes_and_counts():
    task = read_task("img_intensity2_residual16")
    spec = resolve_image_spec(task)
    assert spec["architecture"] == "residual_upsample" and spec["width"] == 16
    g, d = build_image_models(context(task), spec)
    assert tuple(g.input.weight.shape) == (64, 8)
    assert tuple(g.first.weight.shape) == (16, 16, 3, 3)
    assert tuple(g.second.weight.shape) == (16, 16, 3, 3)
    assert tuple(g.output.weight.shape) == (1, 16, 3, 3)
    assert sum(p.numel() for p in g.parameters()) == 5361
    assert sum(p.numel() for p in d.parameters()) == 4929
    with torch.no_grad():
        images = g(torch.zeros(3, 8))
        assert images.shape == (3, 1, 8, 8)
        assert d(images).shape == (3,)


@pytest.mark.parametrize("mutation", ["id", "revision", "source_hash", "source_path", "unknown_profile_field",
                                     "architecture", "width", "noise", "batch", "unknown_architecture_option",
                                     "budget", "prior", "gates", "sampling"])
def test_unknown_or_mismatching_declarations_block_before_construction(mutation):
    task = read_task("img_intensity2_residual16")
    profile = task["execution"]["image_profile"]
    spec = task["execution"]["host_definition"]
    if mutation == "id": profile["id"] = "made_up"
    elif mutation == "revision": profile["revision"] = 2
    elif mutation == "source_hash": profile["source"]["sha256"] = "0" * 64
    elif mutation == "source_path": profile["source"]["path"] = "some-other-plan.json"
    elif mutation == "unknown_profile_field": profile["fallback"] = "transpose"
    elif mutation == "architecture": spec["architecture"] = "transpose"
    elif mutation == "width": spec["width"] = 12
    elif mutation == "noise": spec["noise_std"] = .02
    elif mutation == "batch": spec["batch_size"] = 64
    elif mutation == "unknown_architecture_option": spec["activation"] = "relu"
    elif mutation == "budget": task["execution"]["steps"] = 1200
    elif mutation == "prior": task["execution"]["prior"]["kind"] = "mog"
    elif mutation == "gates": task["evaluation"]["thresholds"][0][2] = 1
    else: task["evaluation"]["eval_output_noise"] = "public_recipe_schedule"
    assert image_profile_blockers(task)
    with pytest.raises(ValueError):
        resolve_image_spec(task)


def test_frozen_profile_file_tampering_is_rejected(tmp_path):
    path = tmp_path / PROFILE_SOURCE
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / PROFILE_SOURCE).read_bytes() + b"\n")
    task = read_task("img_intensity2_residual16")
    with pytest.raises(ValueError, match="source changed"):
        resolve_image_spec(task, root=tmp_path)


def test_unknown_raw_architecture_cannot_silently_fall_back_to_transpose():
    task = read_task("img_intensity2")
    task["execution"]["host_definition"]["architecture"] = "residul_typo"
    assert image_profile_blockers(task)
    with pytest.raises(ValueError, match="unsupported image architecture"):
        build_image_models(context(task), task["execution"]["host_definition"])


def test_profile_task_needs_distinct_identity():
    with pytest.raises(ValueError, match="distinct task id"):
        task_from_profile(read_task("img_intensity2"), "img_intensity2")


def test_runtime_uses_profile_shapes_before_first_update(tmp_path, monkeypatch):
    from experiments.forge.adapters import run_task
    from particlegan import GANTrainer

    class FirstUpdateReached(Exception):
        pass

    task = read_task("img_intensity2_residual16")
    captured = {}

    def stop(trainer, real, **kwargs):
        captured.update(architecture=trainer.G.architecture, width=trainer.G.width,
                        first_shape=list(trainer.G.first.weight.shape),
                        real_shape=list(real.shape), completed_steps=trainer.completed_steps)
        raise FirstUpdateReached

    monkeypatch.setattr(GANTrainer, "step", stop)
    request = {"candidate": {"prior": task["execution"]["prior"]},
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    with pytest.raises(FirstUpdateReached):
        run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert captured == {"architecture": "residual_upsample", "width": 16,
                        "first_shape": [16, 16, 3, 3], "real_shape": [32, 1, 8, 8],
                        "completed_steps": 0}


def test_preflight_uses_selected_execution_checkout(tmp_path):
    from experiments.forge.adapters import adapter_preflight

    task = read_task("img_intensity2_residual16")
    assert not adapter_preflight(task, {}, root=ROOT)
    assert adapter_preflight(task, {}, root=tmp_path)
