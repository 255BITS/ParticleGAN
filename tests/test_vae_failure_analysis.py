"""Saved-state diagnostics must explain outputs without changing training state."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from benchmarks.toy_audit.api_model_vae import build_case


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/forge/vae-failure-investigation/analyze.py"
spec = importlib.util.spec_from_file_location("vae_failure_analysis", PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_row_attribution_rejects_opposite_explanation_and_counts_likelihood_separately():
    # Row0 is never inferred but all its independent means miss the target.
    mass = torch.tensor([0., 1.])
    rows = torch.tensor([0, 0, 1, 1])
    outputs = torch.tensor([[0., 0.], [0., 0.], [1.5, 0.], [1.5, 0.]])
    likelihood = torch.tensor([[-1.5, 0.], [-1.5, 0.], [0., 0.], [1.5, 0.]])
    centers = torch.tensor([[0., 0.], [1.5, 0.]])
    result = analysis.attribute_rows(mass, rows, outputs, likelihood, centers)
    assert result["association_prediction_supported"]
    assert result["low_usage_share_of_decoder_errors"] == 1
    assert result["effectively_unvisited"]["decoder_bad_draws"] == 2
    assert result["effectively_unvisited"]["likelihood_bad_draws"] == 0
    assert result["likelihood_bad_draws"] == 1
    opposite = analysis.attribute_rows(mass.flip(0), rows, outputs, likelihood, centers)
    assert not opposite["association_prediction_supported"]
    perfect = analysis.attribute_rows(mass, rows, torch.full_like(outputs, 0.) + outputs.new_tensor([1.5, 0.]),
                                      likelihood, centers)
    assert perfect["low_usage_share_of_decoder_errors"] is None
    assert not perfect["association_prediction_supported"]


@pytest.mark.parametrize("mode", ["hard", "categorical"])
def test_diagnostics_do_not_step_or_mutate_checkpoint_rng_or_existing_gradients(mode, monkeypatch):
    with analysis.deterministic_cpu():
        fixture = build_case(f"api-model-vae-{mode}", max_steps=1)
        for optimizer in (fixture.opt_g, fixture.opt_d):
            monkeypatch.setattr(optimizer, "step", lambda *a, **kw: pytest.fail("optimizer step attempted"))
        # Existing gradient buffers must survive autograd.grad unchanged too.
        next(fixture.E.parameters()).grad = torch.ones_like(next(fixture.E.parameters()))
        before = analysis.state_hash(fixture.state_dict())
        panel = dict(evaluation_seed=10000, evaluation_inputs=64, gradient_inputs=16,
                     posterior_seed=10001, prior_seed=10002, prior_noise_seed=10003, low_usage_mass=.01)
        result = analysis.analyze_fixture(fixture, panel)
        assert analysis.state_hash(fixture.state_dict()) == before
        assert result["state_unchanged"] and result["training_rng_unchanged"]
        assert result["parameter_gradients_unchanged"] and result["optimizer_updates"] == 0
        assert result["gradient_panel"]["terms"]["adversarial"]["gradient_norms"]["encoder"] == 0
        assert result["gradient_panel"]["terms"]["prior_regularization"]["gradient_norms"]["decoder"] == 0
        assert sum(row["prior_draws"] for row in result["row_attribution"]["rows"]) == 64


def test_determinism_scope_restores_caller_settings_even_on_error():
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError):
        with analysis.deterministic_cpu():
            torch.rand(3)
            raise RuntimeError("stop")
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_input_hashes_are_checked_before_checkpoint_loading(tmp_path):
    directory = tmp_path / "model_vae_hard/api"
    directory.mkdir(parents=True)
    (directory / "receipt.json").write_text("{}")
    protocol = dict(inputs={"hard": dict(receipt_sha256="wrong", checkpoint_sha256="wrong")})
    with pytest.raises(ValueError, match="archived evidence hash mismatch"):
        analysis.validate_inputs(protocol, tmp_path, tmp_path)
    (directory / "receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        analysis.validate_inputs(protocol, tmp_path, tmp_path)


def test_output_is_exclusive_before_loading_or_sampling(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("retained evidence")
    monkeypatch.setattr(analysis, "validate_inputs", lambda *a: pytest.fail("loaded inputs"))
    with pytest.raises(FileExistsError):
        analysis.execute({}, tmp_path, tmp_path, output)
    assert output.read_text() == "retained evidence"


def test_protocol_retains_original_failure_and_publication_bindings():
    protocol = json.loads(PATH.with_name("protocol.json").read_text())
    publication = json.loads((ROOT / "reports/toy_audit/api_contract/model_modes/publication.json").read_text())
    original = {row["id"]: row for row in publication["runs"]}
    assert protocol["optimizer_updates"] == 0 and protocol["cpu_seconds_cap"] == 60
    assert not protocol["qualification_input"]
    assert protocol["inputs"]["hard"]["archived_verdict"] == "FAIL"
    assert protocol["inputs"]["categorical"]["archived_verdict"] == "PASS"
    assert protocol["analysis_source_sha256"] == analysis.file_hash(PATH)
    for mode, pin in protocol["inputs"].items():
        row = original[f"api-model-vae-{mode}"]
        assert pin["receipt_sha256"] == row["raw_receipt_sha256"]
        assert pin["checkpoint_sha256"] == row["artifact_hashes"]["final-state.pt"]["sha256"]
