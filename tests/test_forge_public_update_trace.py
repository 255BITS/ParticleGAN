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
    assert first["path"] == "public-updates.jsonl" and first["path_base"] == "adapter_output"
    assert first["rows"] == 1
    append_update(tmp_path, dict(step=4, loss=torch.tensor(0.)))
    assert receipt(tmp_path)["rows"] == 2
    assert receipt(tmp_path)["sha256"] != first["sha256"]


def test_trace_receipt_is_equal_after_artifact_relocation(tmp_path):
    import shutil
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    append_update(left, dict(step=1, loss=torch.tensor(1.)))
    shutil.copytree(left, right)
    assert receipt(left) == receipt(right)


def test_trace_requires_finite_json_and_no_empty_evidence(tmp_path):
    assert receipt(tmp_path) is None
    import pytest
    with pytest.raises(ValueError):
        append_update(tmp_path, dict(step=1, loss=float("nan")))


def test_publisher_resolves_real_checkpoint_layout_and_rejects_changed_trace(tmp_path, monkeypatch):
    import copy
    from pathlib import Path
    import runpy

    import pytest
    from experiments.forge.artifacts import save_provenance_checkpoint

    publisher = Path(__file__).resolve().parents[1] / "reports/forge/first-principles-round2-20261022/publish.py"
    verify = runpy.run_path(str(publisher))["verify_update_trace"]
    output = tmp_path / "adapter-output"
    output.mkdir()
    append_update(output, dict(step=1001, loss=1.))
    append_update(output, dict(step=1002, loss=2.))
    checkpoint = save_provenance_checkpoint(output, {"streams": {"states": {}}}, completed_steps=1002)
    row = dict(evidence=dict(public_update_trace=receipt(output), provenance_checkpoint=checkpoint))
    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    (elsewhere / "public-updates.jsonl").write_text("wrong artifact\n")
    monkeypatch.chdir(elsewhere)
    result = verify(row)
    assert result["status"] == "VERIFIED"
    assert (result["first_step"], result["last_step"]) == (1001, 1002)
    assert result["scalar_aggregates"]["loss"]["mean"] == 1.5
    frozen = copy.deepcopy(row)
    frozen["evidence"]["public_update_trace"].pop("path_base")
    frozen["evidence"]["public_update_trace"]["path"] = str(output / "public-updates.jsonl")
    assert verify(frozen)["scalar_aggregates"] == result["scalar_aggregates"]
    append_update(output, dict(step=1003, loss=3.))
    with pytest.raises(ValueError, match="trace identity differs"):
        verify(row)
