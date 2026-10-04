"""Structural scorer-boundary controls; no full-budget numerical qualification.

CPU fixtures retain the complete 200-update task declaration but execute at
most two updates. The CUDA regression is separately opt-in and is not part
of the CPU-only control run.
"""
from copy import deepcopy
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.locked_shared.hosts import unused_token_hold as host
from benchmarks.locked_shared.hosts import cover_leftover as cover_host
from experiments.forge import routed_policy_contracts as contract
from experiments.forge import multibank_policy_contracts as cover_contract
from experiments.forge.multibank_policy_adapters import CoverMultibankFixture, _SelectedResidual
from experiments.forge.policy_adapters import evaluation_state, typed_state_digest
from experiments.forge.routed_policy_adapters import (
    UnusedTokenRoutedFixture,
    _SelectedStudent,
    _global_rng,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def structural_one_thread():
    previous_threads = torch.get_num_threads()
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_num_threads(previous_threads)


def fixture(*, device="cpu"):
    task = contract.make_unused_variant(ROOT)
    request = {
        "candidate": {
            "id": "software_only_routed_scorer_device",
            "task_cohort": contract.COHORT,
            "trainer_family": contract.FAMILY,
            "recipe_preset": "atlas",
            "recipe_overrides": deepcopy(contract.SHARED_OVERRIDES),
        },
        "protocol": {"seed": 0},
    }
    value = UnusedTokenRoutedFixture(request, task, device=device)
    assert value.max_steps == host.STEPS == 200
    return value


def cover_fixture(*, device="cpu"):
    task = cover_contract.make_variant(ROOT)
    request = {
        "candidate": {
            "id": "software_only_multibank_scorer_device",
            "task_cohort": cover_contract.COHORT,
            "trainer_family": cover_contract.FAMILY,
            "recipe_preset": "atlas",
            "recipe_overrides": deepcopy(cover_contract.SHARED_OVERRIDES),
        },
        "protocol": {"seed": 0},
    }
    value = CoverMultibankFixture(request, task, device=device)
    assert value.max_steps == cover_host.GATE_STEPS == 800
    return value


@pytest.mark.parametrize("updates", [0, 2])
def test_original_scorer_numerical_parity_on_actual_selected_values(updates):
    value = fixture()
    for _ in range(updates):
        value.step()
    served = value.policy.served_model()
    # The original function is exact at this retained uniform routing mass.
    assert torch.equal(served.models["router"].slot_ids.flatten(), torch.arange(2))
    assert torch.equal(served.models["router"].log_mass, torch.zeros(2))
    original = host.SharedSlotStudent()
    with torch.no_grad():
        original.neu.copy_(served.generator.neu)
        original.shared.copy_(served.generator.shared)
        original.slot.copy_(served.table)
    selected = _SelectedStudent(served)
    for scale in (0., 1., -.5):
        torch.testing.assert_close(selected.embeds(scale), original.embeds(scale), rtol=0, atol=0)
    assert host.score_student(selected) == host.score_student(original)
    assert value.observe() == host.score_student(selected)


def test_snapshot_forward_uses_its_table_device_dtype_and_original_clean_flags(monkeypatch):
    value = fixture()
    value.step()
    served = value.policy.served_model()
    selected = _SelectedStudent(served)
    calls = []
    original = served.routed_forward

    def recorded(context, **kwargs):
        calls.append((context.detach().clone(), dict(kwargs)))
        assert context.device == served.table.device
        assert context.dtype == served.table.dtype
        return original(context, **kwargs)

    monkeypatch.setattr(served, "routed_forward", recorded)
    before = typed_state_digest(value.state_dict())
    host.score_student(selected)
    assert len(calls) == 2
    assert [row[0][:, 1].tolist() for row in calls] == [[1., 1.], [0., 0.]]
    assert all(torch.equal(row[0][:, 0], torch.tensor([0., 1.])) for row in calls)
    assert all(row[1] == {"perturb": False, "output_noise": False} for row in calls)
    assert typed_state_digest(value.state_dict()) == before


def test_scoring_boundary_uses_table_dtype_and_owns_detached_returned_storage():
    # A helper-only double-table/float-reference spy distinguishes the source
    # of the context dtype and detects missing detach or clone on CPU too.
    neu = host.NEU.clone().requires_grad_()
    raw = torch.tensor([[.2, .1], [1., 1.]], dtype=torch.float64, requires_grad=True)
    calls = []
    table = torch.zeros(2, 2, dtype=torch.float64)

    def forward(context, **kwargs):
        assert context.dtype == table.dtype and context.device == table.device
        calls.append(dict(kwargs))
        return raw

    served = SimpleNamespace(table=table, models={"generator": SimpleNamespace(neu=neu)},
                             routed_forward=forward)
    selected = _SelectedStudent(served)
    result = selected.embeds(1.)
    assert selected.neu.device.type == result.device.type == "cpu"
    assert not selected.neu.requires_grad and not result.requires_grad
    assert selected.neu.data_ptr() != neu.data_ptr() and result.data_ptr() != raw.data_ptr()
    torch.testing.assert_close(result, raw.detach(), rtol=0, atol=0)
    result.fill_(99.)
    selected.neu.fill_(-99.)
    assert torch.equal(neu, host.NEU) and torch.equal(raw.detach(), torch.tensor(
        [[.2, .1], [1., 1.]], dtype=torch.float64))
    assert calls == [{"perturb": False, "output_noise": False}]


def test_actual_scorer_view_is_independent_of_frozen_and_training_owners():
    value = fixture()
    value.step()
    served = value.policy.served_model()
    live_before = typed_state_digest(value.state_dict())
    snapshot_before = typed_state_digest({
        "models": {name: module.state_dict() for name, module in served.models.items()},
        "table": served.table,
    })
    assert served.table.data_ptr() != value.G.slot.data_ptr()
    assert served.generator.neu.data_ptr() != value.G.neu.data_ptr()
    selected = _SelectedStudent(served)
    returned = selected.embeds(1.)
    assert returned.device.type == selected.neu.device.type == "cpu"
    assert not returned.requires_grad and not selected.neu.requires_grad
    assert selected.neu.data_ptr() != served.generator.neu.data_ptr()
    returned.zero_()
    selected.neu.add_(100.)
    assert typed_state_digest(value.state_dict()) == live_before
    assert typed_state_digest({
        "models": {name: module.state_dict() for name, module in served.models.items()},
        "table": served.table,
    }) == snapshot_before


@pytest.mark.parametrize("kind", ["unused", "cover"])
def test_repeated_observations_preserve_full_training_state_modes_and_rng(kind):
    value = fixture() if kind == "unused" else cover_fixture()
    value.step()
    value.step()
    before = typed_state_digest(value.state_dict())
    train_before = typed_state_digest(evaluation_state(value.state_dict()))
    global_before = typed_state_digest(_global_rng())
    named_before = value.streams.audit()
    modes = {name: module.training for name, module in value.policy._training_modules().items()}
    outputs = [value.observe() for _ in range(3)]
    assert outputs[0] == outputs[1] == outputs[2]
    assert all(math.isfinite(number) for number in outputs[0].values())
    assert typed_state_digest(value.state_dict()) == before
    assert typed_state_digest(evaluation_state(value.state_dict())) == train_before
    assert typed_state_digest(_global_rng()) == global_before
    assert value.streams.compare(named_before, value.streams.audit())["unintended_rng_deviations"] == 0
    assert {name: module.training for name, module in value.policy._training_modules().items()} == modes
    assert all(row["pure"] for row in value.purity)
    assert all(array.device.type == "cpu" and not array.requires_grad for array in value.last_views.values())
    if kind == "unused":
        assert torch.equal(value.last_views["concept_target"], value.last_views["neu"][1] + host.CONCEPT_DIR)
    else:
        assert torch.equal(value.last_views["targets"], torch.stack((value.poles_p, value.poles_m)))


@pytest.mark.parametrize("kind", ["unused", "cover"])
def test_observation_and_checkpoint_do_not_change_the_exact_next_update(kind):
    build = fixture if kind == "unused" else cover_fixture
    plain = build()
    observed = build()
    assert plain.step() == observed.step()
    saved = observed.state_dict()
    saved_hash = typed_state_digest(saved)
    observed.observe()
    observed.observe()
    resumed = build()
    resumed.load_state_dict(saved)
    assert typed_state_digest(resumed.state_dict()) == saved_hash
    updates = [value.step() for value in (plain, observed, resumed)]
    assert updates[0] == updates[1] == updates[2]
    assert len({typed_state_digest(value.state_dict()) for value in (plain, observed, resumed)}) == 1
    assert typed_state_digest(saved) == saved_hash
    measurements = [value.observe() for value in (plain, observed, resumed)]
    assert measurements[0] == measurements[1] == measurements[2]
    for key in plain.last_views:
        assert torch.equal(plain.last_views[key], observed.last_views[key])
        assert torch.equal(plain.last_views[key], resumed.last_views[key])


@pytest.mark.parametrize("updates", [0, 2])
def test_cover_original_scorer_numerical_parity_on_actual_selected_outputs(updates):
    value = cover_fixture()
    for _ in range(updates):
        value.step()
    served = value.policy.served_model()
    original = cover_host.score_geometry(served.generator, value.field,
                                         value.poles_p, value.poles_m, value.neu)
    selected = _SelectedResidual(served.generator)
    for scale in (1., -1.):
        torch.testing.assert_close(selected.delta(scale), served.generator.delta(scale), rtol=0, atol=0)
    assert cover_host.score_geometry(selected, value.field, value.poles_p, value.poles_m, value.neu) == original
    assert value.observe() == {key: number for key, number in original.items() if type(number) in (int, float)}


def test_cover_scorer_output_storage_is_detached_and_independent():
    raw = torch.tensor([.2, .1, -.05, 0.], requires_grad=True)
    calls = []

    def delta(scale):
        calls.append(scale)
        return raw

    selected = _SelectedResidual(SimpleNamespace(delta=delta))
    first, second = selected.delta(1.), selected.delta(-1.)
    assert first.device.type == second.device.type == "cpu"
    assert not first.requires_grad and not second.requires_grad
    assert len({first.data_ptr(), second.data_ptr(), raw.data_ptr()}) == 3
    first.fill_(99.)
    second.fill_(-99.)
    assert torch.equal(raw.detach(), torch.tensor([.2, .1, -.05, 0.]))
    assert calls == [1., -1.]


def test_cover_scorer_references_and_outputs_cannot_mutate_selected_or_live_owners(monkeypatch):
    value = cover_fixture()
    value.step()
    served = value.policy.served_model()
    monkeypatch.setattr(value.policy, "served_model", lambda: served)
    original = cover_host.score_geometry
    before = typed_state_digest(value.state_dict())
    snapshot_before = typed_state_digest({name: module.state_dict() for name, module in served.models.items()})
    reference_before = typed_state_digest([value.poles_p, value.poles_m, value.neu])
    assert served.generator.w_odd.data_ptr() != value.G.w_odd.data_ptr()
    calls = []

    def recorded(student, field, poles_p, poles_m, neu):
        for source, copied in zip((value.poles_p, value.poles_m, value.neu), (poles_p, poles_m, neu)):
            assert copied.device.type == "cpu" and not copied.requires_grad
            assert source.data_ptr() != copied.data_ptr()
            torch.testing.assert_close(copied, source, rtol=0, atol=0)
        # This callback must use the exact frozen selected model, not a fresh
        # CPU module with copied or recomputed parameters.
        assert student.residual is served.generator
        answer = original(student, field, poles_p, poles_m, neu)
        for tensor in (poles_p, poles_m, neu, student.delta(1.), student.delta(-1.)):
            tensor.fill_(99.)
        calls.append(answer)
        return answer

    monkeypatch.setattr(cover_host, "score_geometry", recorded)
    measured = value.observe()
    assert len(calls) == 1
    assert measured == {key: number for key, number in calls[0].items() if type(number) in (int, float)}
    assert typed_state_digest(value.state_dict()) == before
    assert typed_state_digest([value.poles_p, value.poles_m, value.neu]) == reference_before
    assert typed_state_digest({name: module.state_dict() for name, module in served.models.items()}) == snapshot_before
    assert value.purity[-1]["pure"]


@pytest.mark.skipif(os.environ.get("PARTICLEGAN_ROUTED_SCORER_CUDA_REGRESSION") != "1",
                    reason="GPU-only regression requires explicit opt-in; CPU controls grant no CUDA credit")
def test_cuda_selected_forward_keeps_models_on_cuda_and_original_cpu_scorer_pure(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    value = fixture(device="cuda")
    value.step()
    served = value.policy.served_model()
    original = served.routed_forward
    contexts = []

    def recorded(context, **kwargs):
        assert context.device == served.table.device and context.device.type == "cuda"
        assert context.dtype == served.table.dtype
        contexts.append(context.detach().clone())
        return original(context, **kwargs)

    monkeypatch.setattr(served, "routed_forward", recorded)
    monkeypatch.setattr(value.policy, "served_model", lambda: served)
    before = typed_state_digest(value.state_dict())
    global_before = typed_state_digest(_global_rng())
    cuda_rng_before = torch.cuda.get_rng_state_all()
    named_before = value.streams.audit()
    first, second = value.observe(), value.observe()
    assert first == second and all(math.isfinite(number) for number in first.values())
    assert len(contexts) == 6
    assert served.generator.neu.device.type == served.table.device.type == "cuda"
    assert value.G.neu.device.type == value.G.slot.device.type == "cuda"
    assert typed_state_digest(value.state_dict()) == before
    assert typed_state_digest(_global_rng()) == global_before
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng_before, torch.cuda.get_rng_state_all()))
    assert value.streams.compare(named_before, value.streams.audit())["unintended_rng_deviations"] == 0
    assert all(row["pure"] for row in value.purity)
    assert all(array.device.type == "cpu" for array in value.last_views.values())


@pytest.mark.skipif(os.environ.get("PARTICLEGAN_ROUTED_SCORER_CUDA_REGRESSION") != "1",
                    reason="GPU-only regression requires explicit opt-in; CPU controls grant no CUDA credit")
def test_cuda_cover_forwards_keep_selected_models_and_references_on_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    value = cover_fixture(device="cuda")
    value.step()
    served = value.policy.served_model()
    original = served.generator.delta
    calls = []

    def recorded(scale):
        assert served.generator.w_odd.device.type == served.generator.w_even.device.type == "cuda"
        result = original(scale)
        assert result.device == served.table.device and result.device.type == "cuda"
        calls.append(scale)
        return result

    monkeypatch.setattr(served.generator, "delta", recorded)
    monkeypatch.setattr(value.policy, "served_model", lambda: served)
    before = typed_state_digest(value.state_dict())
    snapshot_before = typed_state_digest({name: module.state_dict() for name, module in served.models.items()})
    global_before = typed_state_digest(_global_rng())
    cuda_rng_before = torch.cuda.get_rng_state_all()
    named_before = value.streams.audit()
    first, second = value.observe(), value.observe()
    assert first == second and all(math.isfinite(number) for number in first.values())
    assert calls == [1., -1., 1., -1.] * 2
    assert all(tensor.device.type == "cuda" for tensor in (value.poles_p, value.poles_m, value.neu))
    assert value.G.w_odd.device.type == value.G.w_even.device.type == "cuda"
    assert typed_state_digest(value.state_dict()) == before
    assert typed_state_digest({name: module.state_dict() for name, module in served.models.items()}) == snapshot_before
    assert typed_state_digest(_global_rng()) == global_before
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng_before, torch.cuda.get_rng_state_all()))
    assert value.streams.compare(named_before, value.streams.audit())["unintended_rng_deviations"] == 0
    assert all(row["pure"] for row in value.purity)
    assert all(array.device.type == "cpu" for array in value.last_views.values())
