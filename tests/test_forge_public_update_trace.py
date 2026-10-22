"""The bulk observer retains earned scales without advancing public state."""
import json

import torch

from experiments.forge.update_trace import append_update, receipt


def test_scalar_trace_is_read_only_and_binds_original_bytes(tmp_path):
    generator = torch.Generator().manual_seed(0)
    parameter = torch.tensor([1., 2.], requires_grad=True)
    parameter.grad = torch.tensor([3., 4.])
    rng = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    values = dict(step=3, loss=parameter.sum(), omitted=parameter,
                  penalty_stats=dict(s_thermal=.25, ema_residual=.01),
                  thermal_displacement=dict(roles=dict(prior=dict(applied_to_base_l2=.25))))
    append_update(tmp_path, values)
    record = json.loads((tmp_path / "public-updates.jsonl").read_text())
    assert "omitted" not in record
    assert record["loss"] == 3.
    assert record["thermal_displacement"]["roles"]["prior"]["applied_to_base_l2"] == .25
    assert torch.equal(generator.get_state(), rng)
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(parameter, torch.tensor([1., 2.]))
    assert torch.equal(parameter.grad, torch.tensor([3., 4.]))
    first = receipt(tmp_path)
    assert first["rows"] == 1
    append_update(tmp_path, dict(step=4, loss=torch.tensor(0.)))
    assert receipt(tmp_path)["rows"] == 2
    assert receipt(tmp_path)["sha256"] != first["sha256"]


def test_trace_requires_finite_json_and_no_empty_evidence(tmp_path):
    assert receipt(tmp_path) is None
    import pytest
    with pytest.raises(ValueError):
        append_update(tmp_path, dict(step=1, loss=float("nan")))
