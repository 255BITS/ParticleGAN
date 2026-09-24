"""Promotion and observer checks only; these tests never fork or train."""

import ast
from copy import deepcopy
import inspect
from pathlib import Path

import pytest
import torch

from reports.toy100 import allocation_continuous_probe as probe


def passing_filter():
    rows = []
    for start, end in probe.BRANCHES:
        steps = list(range(start, end + 1))
        candidate = dict(
            local_gate=dict(pass_all=True, checks=len(steps), passing_checks=len(steps)),
            points=[dict(step=s, grade=dict(modes=8, hq=.99)) for s in steps],
            rng_final_sha256="paired", noise={"total_steps": 1200}, moment_steps={"d": [end], "g": [end]},
            rates=[dict(step=s, role=role, rates=rates) for s in steps
                   for role, rates in (("d", [.00425]), ("g_prior", [.00425, .0085]))],
            dynamics=dict(method="candidate", outer_steps=len(steps), records=[{} for _ in steps],
                          corrections=[{} for _ in steps], correction_rng_checks=len(steps),
                          correction_owner_checks=len(steps)))
        rows.append(dict(start=start, end=end, variants=dict(original=deepcopy(candidate), reallocation=candidate)))
    return dict(status="PASS", warm_eligible=True, branches=rows,
                declaration=dict(method="candidate", states_sha256=probe.CAPTURE_SHA,
                                 nominal_rates=probe.RATES, seed=0, noise_horizon=1200))


def test_complete_saved44_is_consumed_and_true_summary_cannot_hide_bad_point():
    gate = passing_filter()
    probe.validate_filter(gate, "candidate")
    gate["branches"][1]["variants"]["reallocation"]["points"][-1]["grade"]["hq"] = .899
    with pytest.raises(RuntimeError, match="live quality"):
        probe.validate_filter(gate, "candidate")


@pytest.mark.parametrize("damage", ["missing_branch", "duplicate_step", "truthy_flag", "wrong_noise", "moment_twice", "rate_decay"])
def test_saved_gate_rejects_incomplete_or_invalid_training_receipts(damage):
    gate = passing_filter()
    candidate = gate["branches"][0]["variants"]["reallocation"]
    if damage == "missing_branch":
        gate["branches"].pop()
    elif damage == "duplicate_step":
        candidate["points"][1]["step"] = candidate["points"][0]["step"]
    elif damage == "truthy_flag":
        gate["warm_eligible"] = {"passed": False}
    elif damage == "wrong_noise":
        candidate["noise"]["total_steps"] = 2400
    elif damage == "moment_twice":
        candidate["moment_steps"]["g"] = [2670]
    else:
        candidate["rates"][1]["rates"][0] *= .5
    with pytest.raises(RuntimeError):
        probe.validate_filter(gate, "candidate")


def test_source_archive_and_live_bytes_both_required(tmp_path):
    root, archive = tmp_path / "live", tmp_path / "archive"
    root.mkdir()
    archive.mkdir()
    for directory in (root, archive):
        (directory / "source.py").write_text("frozen")
    sources = {"source.py": probe.sha(b"frozen")}
    probe.verify_sources(sources, root=root, archive=archive)
    (archive / "source.py").write_text("stale")
    with pytest.raises(RuntimeError, match="archived source"):
        probe.verify_sources(sources, root=root, archive=archive)
    with pytest.raises(ValueError, match="inside the repository"):
        probe.source_path("../escape", root)


def previous_gate():
    return dict(method="candidate", phase="hold", factory="module:factory", source={"x": "sha"},
                saved_state_filter_sha256="filter", identity_cold_parity=True,
                original_control_exact_parity=True, first200_parity=True, status="PASS",
                variants=dict(candidate=dict(status="PASS", local_stability=dict(checks=200, pass_all=True),
                                             long_hold=dict(checks=1200, pass_all=True))))


@pytest.mark.parametrize("damage", [None, "sparse_hold", "source_changed", "no_control", "failed_warm"])
def test_cold_requires_complete_dense_hold_and_unchanged_source(damage):
    previous = previous_gate()
    if damage == "sparse_hold":
        previous["variants"]["candidate"]["long_hold"]["checks"] = 120
    elif damage == "source_changed":
        previous["source"]["x"] = "different"
    elif damage == "no_control":
        previous["original_control_exact_parity"] = 1
    elif damage == "failed_warm":
        previous["variants"]["candidate"]["local_stability"]["pass_all"] = False
    kwargs = dict(phase="cold", method="candidate", factory="module:factory", sources={"x": "sha"}, filter_sha="filter")
    if damage is None:
        probe.require_previous(previous, **kwargs)
    else:
        with pytest.raises(RuntimeError):
            probe.require_previous(previous, **kwargs)


def test_dense_observer_changes_only_three_reviewed_diagnostic_expressions():
    from benchmarks.toy100 import warm_equilibrium_probe as warm
    _, source = probe.dense_warm_runner()  # Compiles only: no training/fork.
    modified = ast.parse(source)
    cadence = next(n for n in ast.walk(modified) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "cadence" for t in n.targets))
    assert ast.unparse(cadence.value) == "50"
    cadence.value = ast.parse("50 if steps == FROZEN_STEPS else 10", mode="eval").body
    end = next(n for n in ast.walk(modified) if isinstance(n, ast.keyword) and n.arg == "dense_until")
    assert ast.unparse(end.value) == "steps"
    end.value = ast.Name(id="FROZEN_STEPS", ctx=ast.Load())
    divisor = next(n for n in ast.walk(modified) if isinstance(n, ast.BinOp)
                   and isinstance(n.op, ast.FloorDiv) and ast.unparse(n.left) == "steps - FROZEN_STEPS")
    assert ast.unparse(divisor.right) == "1"
    divisor.right = ast.Name(id="cadence", ctx=ast.Load())
    original = ast.parse(inspect.getsource(warm.run_warm_variants))
    assert ast.dump(modified) == ast.dump(original)


def test_first200_requires_full_state_and_controller_parity():
    points = [dict(step=s, hq=1.) for s in range(1001, 1201)]
    old = dict(warm_state_sha256="warm", diagnostic=points,
               dynamics_receipt=dict(records=[{"g": s} for s in range(200)],
                                     corrections=[{"selected": "rest"} for _ in range(200)],
                                     first200_full_state_sha256="full-noise-and-state"))
    new = deepcopy(old)
    probe.compare_first200(old, new)
    new["dynamics_receipt"]["first200_full_state_sha256"] = "different-noise-history"
    with pytest.raises(RuntimeError, match="models/moments"):
        probe.compare_first200(old, new)


def test_original_hold_allows_only_declared_prefix_cadence_difference():
    data = {key: "same" for key in ("warm_state_sha256", "final_state_sha256", "observations", "noise",
                                     "final", "ema", "optimizer_final", "post_checkpoint_rate_ranges")}
    data["dynamics_receipt"] = dict(records=[])
    expected = deepcopy(data)
    expected["diagnostic"] = [dict(step=s, hq=1.) for s in (10, 20, 50, 1000, 1001, 1010, 1210)]
    disabled = deepcopy(data)
    disabled["diagnostic"] = [dict(step=s, hq=1.) for s in (50, 1000, 1001, 1010, 1210)]
    probe.compare_original(disabled, expected, [])
    disabled["diagnostic"].pop()
    with pytest.raises(RuntimeError, match="diagnostic"):
        probe.compare_original(disabled, expected, [])


def test_finite_audit_detects_optimizer_corruption_without_changing_state():
    local = {name: torch.nn.Linear(1, 1) for name in ("generator", "critic", "prior")}
    local["opt_g"] = torch.optim.Adam(local["generator"].parameters())
    local["opt_d"] = torch.optim.Adam(local["critic"].parameters())
    before = torch.get_rng_state().clone()
    assert probe.internal_state_metrics(local)["model_and_adam_finite"] is True
    assert torch.equal(before, torch.get_rng_state())
    parameter = next(local["critic"].parameters())
    local["opt_d"].state[parameter]["exp_avg_sq"] = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="Adam state"):
        probe.internal_state_metrics(local)
