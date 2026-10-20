"""Saved-state diagnostics must narrow failures without changing evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.toy_audit import api_model_ddgan as api
from experiments.forge.rng import NamedStreams

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ddgan_diagnose",
    ROOT / "reports/forge/ddgan-failure-investigation/diagnose.py")
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_frozen_source_guard_accepts_only_exact_published_report_overlay(tmp_path):
    (tmp_path / "scientific.py").write_text("science")
    report = "experiments/forge/publication_memory.py"
    (tmp_path / report).parent.mkdir(parents=True)
    (tmp_path / report).write_text("repaired")
    manifest = dict(origin_commit=diagnose.FROZEN_COMMIT, digest="frozen",
        files={"scientific.py": _hash("science"), report: _hash("original")})
    repair = dict(reporting_only_changes={report: dict(
        frozen_sha256=_hash("original"), repaired_sha256=_hash("repaired"))})
    assert diagnose.verify_source(tmp_path, manifest, repair)["scientific_source_unchanged"]
    (tmp_path / "scientific.py").write_text("different")
    # A forged repair allowance cannot authorize a scientific file.
    repair["reporting_only_changes"]["scientific.py"] = dict(
        frozen_sha256=_hash("science"), repaired_sha256=_hash("different"))
    with pytest.raises(ValueError, match="scientific.py"):
        diagnose.verify_source(tmp_path, manifest, repair)


def test_original_artifact_hashes_are_required_before_checkpoint_load(tmp_path):
    (tmp_path / "api").mkdir()
    (tmp_path / "api/receipt.json").write_text("receipt")
    (tmp_path / "adapter-receipt.json").write_text("adapter")
    (tmp_path / "api/final-state.pt").write_text("original state")
    run = dict(raw_receipt_sha256=_hash("receipt"), adapter_receipt_sha256=_hash("adapter"),
        artifact_hashes={"final-state.pt": dict(sha256=_hash("original state"))})
    assert diagnose.verify_artifacts(tmp_path, run)["final-state.pt"] == _hash("original state")
    (tmp_path / "api/final-state.pt").write_text("changed state")
    with pytest.raises(ValueError, match="original artifact mismatch: final-state.pt"):
        diagnose.verify_artifacts(tmp_path, run)


def test_analytic_variance_budget_matches_independent_gaussian_precision():
    target, schedule = api.MixtureTarget(), api.DDGAN(api.ALPHA_BAR)
    for step in range(1, 5):
        value = diagnose.mechanism(target, schedule, step, diagnose.PANELS[0])
        a = schedule.ab[step].double()
        clean_var = 1 / (1 / target.variances[0].double() + a / (1 - a))
        signal = schedule.A[step].double().square() * clean_var
        expected = signal / (signal + schedule.posterior_var[step])
        assert torch.allclose(torch.tensor(value["learned_clean_share_of_exact_transition_variance"]),
                              expected.float(), atol=1e-7)
    assert value["learned_clean_share_of_exact_transition_variance"][1] < .005
    central = diagnose.mechanism(target, schedule, 1, diagnose.PANELS[1])
    assert central["gaussian_context_radial_tail"] > .97


def test_reverse_noise_can_mask_a_collapsed_clean_predictor():
    target, schedule = api.MixtureTarget(), api.DDGAN(api.ALPHA_BAR)
    n, generator = 8192, torch.Generator().manual_seed(0)
    for step in (1, 4):
        xt = torch.tensor(diagnose.PANELS[1]).expand(n, 2)
        weights, means, variances = target.posterior_parameters(xt[:1], schedule.ab[step])
        collapsed_clean = means[0, 0].expand(n, 2)
        clean_metrics = api.distribution_metrics(collapsed_clean, weights[0], means[0], variances[0])
        reverse = schedule.reverse(collapsed_clean, xt, torch.full((n,), step, dtype=torch.long),
            torch.randn(n, 2, generator=generator))
        metrics = api.distribution_metrics(reverse, *target.transition_parameters(diagnose.PANELS[1], step, schedule))
        assert clean_metrics["covariance_eigen_max"] < 1e-10
        if step == 1:
            assert metrics["covariance_eigen_min"] < 1e-10
        else:
            assert not api.distribution_failures(metrics)


@pytest.mark.parametrize("sign", [1., -1.])
def test_generator_loss_alignment_detects_helpful_and_opposed_critics(sign):
    class LinearCritic(nn.Module):
        def forward(self, x, **context):
            return sign * x[:, 0]
    fake = torch.zeros(32, 2)
    value = diagnose.critic_alignment(LinearCritic(), fake, torch.ones_like(fake), fake,
        torch.ones(32, dtype=torch.long), torch.tensor([1., 0.]))
    assert value["mean_correction_descent_alignment"] == pytest.approx(sign * .5)
    assert fake.grad is None


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_checkpoint_analysis_preserves_original_state_streams_and_caller_rng():
    fixture = api.build_case("api-model-ddgan-gaussian-chain")
    state = fixture.state_dict()
    before, rng = deepcopy(state), torch.get_rng_state().clone()
    receipt = dict(observations=[dict(step=0, metrics={}, passed=False, failed_bounds=["incomplete"])])
    value = diagnose.analyze(api, NamedStreams, state, receipt, n=16, context_count=1)
    assert _equal(state, before)
    assert torch.equal(rng, torch.get_rng_state())
    assert value["purity"]["training_updates_added"] == 0
    assert not value["purity"]["optimizer_constructed"]
    state["recipe"]["loss"] = "hinge"
    with pytest.raises(ValueError, match="requires non_saturating loss"):
        diagnose.analyze(api, NamedStreams, state, receipt, n=16, context_count=1)


def test_committed_diagnosis_preserves_source_failure_identity():
    path = ROOT / "reports/forge/ddgan-failure-investigation/results.json"
    if not path.exists():
        pytest.skip("saved analysis not yet published")
    results = json.loads(path.read_text())
    original = json.loads((ROOT / "reports/toy_audit/api_contract/model_modes/publication.json").read_text())
    assert results["source"]["origin_commit"] == diagnose.FROZEN_COMMIT
    assert results["analysis_code_sha256"] == diagnose.sha256(SPEC.origin)
    assert not results["qualification_input"]
    assert results["diagnostic_protocol"]["training_updates"] == 0
    for task, value in results["tasks"].items():
        run = next(r for r in original["runs"] if r["task_id"] == task)
        assert value["original_artifact_hashes"]["final-state.pt"] == run["artifact_hashes"]["final-state.pt"]["sha256"]
        assert value["saved_terminal_suffix"]["passing_joint_observations"] == 0
        assert value["purity"]["models_gradients_modes_unchanged"]
