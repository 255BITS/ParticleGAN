"""Native adapter dispatch and parent checks, with zero optimizer updates."""
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.forge import adapters
from experiments.forge.contracts import read_json
from experiments.forge.planning import resolve_idea

ROOT = Path(__file__).resolve().parents[1]


def task(name):
    return read_json(ROOT / "configs/forge/tasks" / f"{name}.json")


def test_native_dispatch_applies_task_owned_public_initializers(tmp_path, monkeypatch):
    selected = task("grid100_affine_square_named_v1")
    captured = {}

    class Constructed(Exception):
        pass

    def stop(context, trainer, *args, **kwargs):
        captured.update(generator=type(trainer.G), identity=torch.equal(trainer.G.weight, torch.eye(2)),
            fourier=trainer.D.fourier, locations=list(trainer.prior.z.shape), sigma=float(trainer.prior.sigma),
            batch=context.recipe.batch_size, horizon=context.recipe.total_steps,
            methods={k: v["initializer"] for k, v in context.initialization.items()},
            steps=trainer.completed_steps)
        raise Constructed

    monkeypatch.setattr(adapters, "_Run", stop)
    request = {"candidate": {"prior": selected["execution"]["prior"]}, "protocol": {"seed": 0},
               "tasks": {selected["id"]: selected}}
    with pytest.raises(Constructed):
        adapters.run_task(request, {"task_id": selected["id"]}, tmp_path, "cpu")
    assert captured == {"generator": nn.Linear, "identity": True, "fourier": 3,
        "locations": [20000, 2], "sigma": pytest.approx(.025), "batch": 2048, "horizon": 7000,
        "methods": {"generator": "identity_linear_v1", "discriminator": "xavier_uniform_zero_bias_v1",
                    "prior": "sample_distributions_v1"}, "steps": 0}


def test_continuation_rejects_raw_parent_before_constructing_models(tmp_path, monkeypatch):
    selected = task("grid100_affine_square_named_v1_14k")
    parent = task("grid100")
    parent["id"] = selected["execution"]["continuation_of"]

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid profile parent must fail before construction")

    monkeypatch.setattr(adapters, "_context", forbidden)
    request = {"candidate": {"prior": selected["execution"]["prior"]}, "protocol": {"seed": 0},
               "tasks": {selected["id"]: selected, parent["id"]: parent}}
    with pytest.raises(ValueError, match="matching profile"):
        adapters.run_task(request, {"task_id": selected["id"]}, tmp_path, "cpu")


def test_all_declared_profile_hosts_plan_without_changing_default_view():
    request = resolve_idea(ROOT, "k3p", view_id="host_profile_transfer", through_tier=3,
                           execution_backend="cpu")
    assert not request["preflight_blockers"]
    assert all(not t["preflight_blockers"] for t in request["tasks"].values())
    profiled = {name for name, t in request["tasks"].items()
                if any(k.endswith("_profile") for k in t["execution"])}
    assert len(profiled) == 13  # Four image, six vector and three native parents.
    # The longer catalog variants retain their clock-free audit dependency and
    # do not enter this scheduled architecture-transfer view automatically.
    assert not any(name.endswith("_affine_square_named_v1_14k") for name in request["tasks"])
    assert all(a["importance"] == "diagnostic" for a in request["view"]["assignments"] if a["task"] in profiled)
    default = resolve_idea(ROOT, "k3p", through_tier=3, execution_backend="cpu")
    assert not profiled & default["tasks"].keys()
    assert request["candidate_revision"] == default["candidate_revision"]
