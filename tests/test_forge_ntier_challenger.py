"""Adversarial stress-test suite for ParticleGAN Forge N-Tier core infrastructure.

Empirical Challenger verification for Milestone 1:
1. Dynamic mock views with Tier 4, Tier 5, Tier 10, and Tier 100.
2. Contiguity checks: missing intermediate tiers, gap at Tier 1, gap at Tier N-1.
3. Qualification reductions (qualify()):
   - Tier 4 required passes, ranking passes -> qualified tier 4
   - Tier 4 held, required passed up to 3 -> qualified tier 3, status HELD
   - Tier 4 required fails -> status INCOMPLETE / FAIL
   - Ranking failures, diagnostic isolation, intermediate held tiers, scaling to T10/T100.
4. CLI parsing for --through-tier and --tier under extreme values.
5. Dynamic report & table rendering up to Tier 100.
"""

from copy import deepcopy
import math
from pathlib import Path
import pytest

from experiments.forge.views import qualify, validate_view, IMPORTANCES
from experiments.forge.contracts import atomic_json
from experiments.forge.tier_report import build_report, render_markdown, tier_name, TIER_NAMES
from experiments.forge.family_reports import _table_header
from experiments.forge.__main__ import parser
from experiments.forge.dispositions import _tier_counts
from experiments.forge.configuration_search import select_configuration
from experiments.forge.studies import validate_study
from experiments.forge.decision_contracts import validate_shape


# ---------------------------------------------------------------------------
# Test Fixture Generators
# ---------------------------------------------------------------------------

def make_task(name: str) -> dict:
    """Generate a minimal valid task dictionary conforming to Forge schema."""
    return dict(
        schema_version=1,
        id=name,
        adapter="test_host",
        execution=dict(
            initializer="deterministic_orthogonal",
            steps=24,
            produces_state=True,
            prior=dict(kind="mog", sigma=0.025, standardize=False, learnable=True),
        ),
        evaluation=dict(
            kind="transfer_sustained",
            thresholds=[["error", "<=", 1.0]],
            scoring_weights="live",
        ),
        resources={},
        requires_capabilities=[],
        dependencies=[],
    )


def make_view(*assignments, view_id: str = "stress_view", revision: int = 1) -> dict:
    """Generate a valid view dictionary with given (task_id, tier, importance) tuples."""
    return dict(
        schema_version=1,
        id=view_id,
        revision=revision,
        goal="quality",
        eligibility={},
        assignments=[
            dict(task=task, qualification_tier=tier, importance=importance, order=order)
            for order, (task, tier, importance) in enumerate(assignments)
        ],
    )


def make_receipt(task: dict, *, bad_steps=(), final: float = 0.5) -> dict:
    """Generate a mock evaluation receipt for a task."""
    steps = task["execution"]["steps"]
    observations = [
        dict(step=math.ceil(i * steps / 24), error=2.0 if i in bad_steps else 0.5)
        for i in range(1, 25)
    ]
    return dict(
        task_id=task["id"],
        compatibility_key="compatible",
        gate_status="PASS" if final <= 1.0 and not bad_steps else "FAIL",
        evidence=dict(observations=observations, live=dict(error=final)),
        cost=dict(seconds=2.0),
    )


def make_hold(name: str, *, reason="no_global_recipe", blocked_by=()) -> dict:
    """Generate valid forge_hold_v1 metadata."""
    return dict(
        schema="forge_hold_v1",
        previous_importance="required",
        reason=reason,
        blocked_by=list(blocked_by),
        summary=f"Fixture: {name} is measured and shown, but not counted.",
        explanation=f"reports/forge/test-dispositions-20261015/README.md#{name}",
        reentry=f"Count {name} again when one global recipe passes it.",
    )


# ---------------------------------------------------------------------------
# Area 1: Dynamic Mock Views with Tier 4, Tier 5, Tier 10, Tier 100
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_tier", [4, 5, 10, 100])
def test_dynamic_mock_views_validation_scaling(max_tier):
    """Stress-test view validation with dynamic tiers scaling up to Tier 100."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, max_tier + 1)}
    assignments = [(f"t{i}", i, "required") for i in range(1, max_tier + 1)]
    view = make_view(*assignments, view_id=f"view_tier_{max_tier}")

    # Should validate without error
    validate_view(view, tasks)

    # Dispositions _tier_counts scaling
    counts = _tier_counts(view)
    assert len(counts["required"]) == max_tier
    assert counts["required"] == [1] * max_tier
    assert counts["held"] == [0] * max_tier
    assert counts["removed"] == [0] * max_tier


@pytest.mark.parametrize("tier_val, expected_name", [
    (1, "smoke"),
    (2, "quality"),
    (3, "endurance"),
    (4, "stress_and_formulations"),
    (5, "tier-5"),
    (10, "tier-10"),
    (100, "tier-100"),
    (99999, "tier-99999"),
])
def test_tier_naming_rule(tier_val, expected_name):
    """Verify tier_name adheres to exact specifications for T1-T4 and fallback tier-N for N>=5."""
    assert tier_name(tier_val) == expected_name


def test_markdown_rendering_scales_to_tier_10_and_100(tmp_path):
    """Verify build_report and render_markdown handle views scaling to Tier 10 and Tier 100."""
    for n in [10, 100]:
        subpath = tmp_path / f"catalog_{n}"
        subpath.mkdir(parents=True, exist_ok=True)
        # Create minimal tasks and view
        tasks_dir = subpath / "configs/forge/tasks"
        views_dir = subpath / "configs/forge/views"
        for i in range(1, n + 1):
            atomic_json(tasks_dir / f"task_{i}.json", make_task(f"task_{i}"))

        assignments = [
            {"task": f"task_{i}", "qualification_tier": i, "importance": "required", "order": 0}
            for i in range(1, n + 1)
        ]
        atomic_json(views_dir / f"view_{n}.json", {
            "schema_version": 1,
            "id": f"view_{n}",
            "revision": 1,
            "goal": "quality",
            "eligibility": {},
            "calibration": {"status": "provisional"},
            "assignments": assignments,
        })

        report = build_report(subpath)
        md = render_markdown(report, subpath, subpath / "reports/forge/EXPERIMENTS_BY_TIER.md")

        # Table header must contain all Tier columns
        assert f"Tier {n}" in md
        # Divider column count must match header column count
        for line in md.splitlines():
            if line.startswith("| View | Revision |"):
                header_cols = [c.strip() for c in line.split("|") if c.strip()]
            elif line.startswith("| --- | ---: |"):
                divider_cols = [c.strip() for c in line.split("|") if c.strip()]
                assert len(header_cols) == len(divider_cols)
                assert len(header_cols) == n + 3  # View, Revision, Tier 1..n, Declared calibration


# ---------------------------------------------------------------------------
# Area 2: Stress-Test Contiguity Checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_tier, max_tier", [
    (1, 4),    # gap at Tier 1 (starts at 2)
    (1, 5),    # gap at Tier 1 (starts at 2)
    (1, 10),   # gap at Tier 1
    (1, 100),  # gap at Tier 1
    (2, 4),    # missing intermediate Tier 2 in 1..4
    (3, 4),    # gap at Tier N-1 (missing 3 in 1..4)
    (4, 5),    # gap at Tier N-1 (missing 4 in 1..5)
    (9, 10),   # gap at Tier N-1 (missing 9 in 1..10)
    (99, 100), # gap at Tier N-1 (missing 99 in 1..100)
    (5, 10),   # missing intermediate Tier 5 in 1..10
    (50, 100), # missing intermediate Tier 50 in 1..100
])
def test_contiguity_rejects_missing_tiers(missing_tier, max_tier):
    """Stress-test contiguity enforcement rejecting missing intermediate tiers, gap at 1, and gap at N-1."""
    tiers = [t for t in range(1, max_tier + 1) if t != missing_tier]
    tasks = {f"t{t}": make_task(f"t{t}") for t in tiers}
    assignments = [(f"t{t}", t, "required") for t in tiers]
    view = make_view(*assignments)

    with pytest.raises(ValueError, match="empty required tier: qualification tiers must be contiguous from Tier 1"):
        validate_view(view, tasks)


@pytest.mark.parametrize("gap_role", ["ranking", "diagnostic"])
def test_contiguity_cannot_be_bridged_by_ranking_or_diagnostic(gap_role):
    """Stress-test that ranking or diagnostic tasks cannot bridge a contiguity gap at Tier 2 or Tier N-1."""
    # 4-tier view where Tier 2 is ranking/diagnostic
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    assignments = [
        ("t1", 1, "required"),
        ("t2", 2, gap_role),
        ("t3", 3, "required"),
        ("t4", 4, "required"),
    ]
    view = make_view(*assignments)
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view, tasks)

    # 4-tier view where Tier 3 (N-1) is ranking/diagnostic
    assignments_n_minus_1 = [
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, gap_role),
        ("t4", 4, "required"),
    ]
    view_n_minus_1 = make_view(*assignments_n_minus_1)
    with pytest.raises(ValueError, match="empty required tier"):
        validate_view(view_n_minus_1, tasks)


def test_contiguity_held_tasks_bridge_any_tier():
    """Verify held tasks in COUNTED_OR_HELD satisfy contiguity at intermediate, N-1, and top tier."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 6)}
    # Tier 2 held, Tier 4 (N-1) held, Tier 5 held
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "held"),
        ("t3", 3, "required"),
        ("t4", 4, "held"),
        ("t5", 5, "held"),
    )
    view["assignments"][1]["hold"] = make_hold("t2")
    view["assignments"][3]["hold"] = make_hold("t4")
    view["assignments"][4]["hold"] = make_hold("t5")

    validate_view(view, tasks)


# ---------------------------------------------------------------------------
# Area 3: Stress-Test Qualification Reductions (qualify())
# ---------------------------------------------------------------------------

def test_qualification_tier4_required_passes_ranking_passes():
    """qualify(): Tier 4 required passes, ranking passes -> qualified tier 4, status PASS."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    tasks["r4"] = make_task("r4")
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "required"),
        ("r4", 4, "ranking"),
    )
    results = [make_receipt(tasks[t]) for t in ["t1", "t2", "t3", "t4", "r4"]]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is True
    assert res["required_passed"] == 4
    assert res["required_total"] == 4
    assert res["task_statuses"]["t4"] == "PASS"
    assert res["task_statuses"]["r4"] == "PASS"


def test_qualification_tier4_held_required_passed_up_to_3():
    """qualify(): Tier 4 held, required passed up to 3 -> qualified tier 3, status HELD."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "held"),
    )
    view["assignments"][3]["hold"] = make_hold("t4")

    # Case A: held task passes
    results_pass = [make_receipt(tasks[f"t{i}"]) for i in range(1, 5)]
    res_pass = qualify(view, tasks, results_pass)
    assert res_pass["status"] == "HELD"
    assert res_pass["qualified_tier"] == 3
    assert res_pass["eligible"] is False
    assert res_pass["held_tiers"] == [4]
    assert res_pass["counted_tiers"] == [1, 2, 3]

    # Case B: held task fails
    results_fail = [
        make_receipt(tasks["t1"]),
        make_receipt(tasks["t2"]),
        make_receipt(tasks["t3"]),
        make_receipt(tasks["t4"], final=3.0),
    ]
    res_fail = qualify(view, tasks, results_fail)
    assert res_fail["status"] == "HELD"
    assert res_fail["qualified_tier"] == 3
    assert res_fail["eligible"] is False
    assert res_fail["task_statuses"]["t4"] == "FAIL"


def test_qualification_tier4_required_fails_gives_fail():
    """qualify(): Tier 4 required fails -> status FAIL, qualified tier 3."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "required"),
    )
    # Tiers 1-3 pass, Tier 4 required fails
    results = [
        make_receipt(tasks["t1"]),
        make_receipt(tasks["t2"]),
        make_receipt(tasks["t3"]),
        make_receipt(tasks["t4"], final=5.0),
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "FAIL"
    assert res["qualified_tier"] == 3
    assert res["eligible"] is False
    assert res["required_passed"] == 3
    assert res["required_total"] == 4
    assert any(b["task_id"] == "t4" for b in res["blockers"])


def test_qualification_tier4_required_incomplete_or_not_run():
    """qualify(): Tier 4 required not run -> status INCOMPLETE, qualified tier 3, next_task t4."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "required"),
    )
    # Tiers 1-3 pass, Tier 4 omitted from results
    results = [
        make_receipt(tasks["t1"]),
        make_receipt(tasks["t2"]),
        make_receipt(tasks["t3"]),
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "INCOMPLETE"
    assert res["qualified_tier"] == 3
    assert res["eligible"] is False
    assert res["next_task"] == "t4"


def test_qualification_tier4_ranking_failure_does_not_block():
    """qualify(): Tier 4 required passes, ranking fails -> qualified tier 4, status PASS."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 5)}
    tasks["r4"] = make_task("r4")
    view = make_view(
        ("t1", 1, "required"),
        ("t2", 2, "required"),
        ("t3", 3, "required"),
        ("t4", 4, "required"),
        ("r4", 4, "ranking"),
    )
    results = [
        make_receipt(tasks["t1"]),
        make_receipt(tasks["t2"]),
        make_receipt(tasks["t3"]),
        make_receipt(tasks["t4"]),
        make_receipt(tasks["r4"], final=5.0),  # ranking fails
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == 4
    assert res["eligible"] is True
    assert res["task_statuses"]["r4"] == "FAIL"


@pytest.mark.parametrize("max_tier", [10, 100])
def test_qualification_scaling_all_pass_tier10_and_100(max_tier):
    """Stress-test qualify() scaling to Tier 10 and Tier 100 with all passing."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, max_tier + 1)}
    assignments = [(f"t{i}", i, "required") for i in range(1, max_tier + 1)]
    view = make_view(*assignments)

    results = [make_receipt(tasks[f"t{i}"]) for i in range(1, max_tier + 1)]
    res = qualify(view, tasks, results)

    assert res["status"] == "PASS"
    assert res["qualified_tier"] == max_tier
    assert res["eligible"] is True
    assert res["required_passed"] == max_tier
    assert res["required_total"] == max_tier
    assert res["counted_tiers"] == list(range(1, max_tier + 1))


def test_qualification_partial_completion_stops_at_failing_tier():
    """In a 10-tier view, failure at Tier 6 stops progression at Tier 5."""
    tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 11)}
    assignments = [(f"t{i}", i, "required") for i in range(1, 11)]
    view = make_view(*assignments)

    # Tiers 1-5 pass, Tier 6 fails, Tiers 7-10 pass
    results = [
        make_receipt(tasks[f"t{i}"], final=(5.0 if i == 6 else 0.5))
        for i in range(1, 11)
    ]
    res = qualify(view, tasks, results)

    assert res["status"] == "FAIL"
    assert res["qualified_tier"] == 5
    assert res["eligible"] is False
    assert res["required_passed"] == 9
    assert any(b["task_id"] == "t6" for b in res["blockers"])


# ---------------------------------------------------------------------------
# Area 4: CLI Parsing Under Extreme Values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcommand", ["plan", "enqueue", "run"])
@pytest.mark.parametrize("extreme_tier", [1, 4, 5, 10, 100, 1000, 999999])
def test_cli_through_tier_extreme_valid_integers(subcommand, extreme_tier):
    """CLI --through-tier must accept arbitrary positive integers >= 1."""
    p = parser()
    args = p.parse_args([subcommand, "test_candidate", "--through-tier", str(extreme_tier)])
    assert args.through_tier == extreme_tier


@pytest.mark.parametrize("subcommand", ["plan", "enqueue", "run"])
@pytest.mark.parametrize("invalid_tier", [
    "0", "-1", "-5", "-99999", "foo", "1.5", "NaN", "inf", "", "None", "True", "False"
])
def test_cli_through_tier_extreme_invalid_values(subcommand, invalid_tier):
    """CLI --through-tier must reject non-integers, zero, and negative values."""
    p = parser()
    with pytest.raises(SystemExit):
        p.parse_args([subcommand, "test_candidate", "--through-tier", invalid_tier])


@pytest.mark.parametrize("extreme_tier", [0, 1, 4, 10, 100, 10000])
def test_cli_board_tier_extreme_valid(extreme_tier):
    """CLI board --tier accepts integers >= 0."""
    p = parser()
    args = p.parse_args(["board", "--tier", str(extreme_tier)])
    assert args.tier == extreme_tier


@pytest.mark.parametrize("invalid_tier", [
    "-1", "-100", "foo", "1.5", "NaN", "inf", "", "None"
])
def test_cli_board_tier_extreme_invalid(invalid_tier):
    """CLI board --tier rejects integers < 0 and non-integers."""
    p = parser()
    with pytest.raises(SystemExit):
        p.parse_args(["board", "--tier", invalid_tier])


@pytest.mark.parametrize("extreme_tier", [1, 4, 10, 100, 10000])
def test_cli_retier_tier_extreme_valid(extreme_tier):
    """CLI retier --tier accepts integers >= 1."""
    p = parser()
    args = p.parse_args([
        "retier", "view", "--task", "grid100_mlp16", "--tier", str(extreme_tier), "--reason", "stress"
    ])
    assert args.tier == extreme_tier


@pytest.mark.parametrize("invalid_tier", [
    "0", "-1", "-100", "foo", "1.5", "NaN", "inf", "", "None"
])
def test_cli_retier_tier_extreme_invalid(invalid_tier):
    """CLI retier --tier rejects integers < 1 and non-integers."""
    p = parser()
    with pytest.raises(SystemExit):
        p.parse_args([
            "retier", "view", "--task", "grid100_mlp16", "--tier", invalid_tier, "--reason", "stress"
        ])


# ---------------------------------------------------------------------------
# Area 5: Additional Edge Cases & Cross-Module Contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("through_tier", [4, 10, 100])
def test_configuration_search_scaling(through_tier):
    """select_configuration correctly handles through_tier scaling to 4, 10, 100."""
    trials = [
        {
            "configuration_id": f"cfg_{c}",
            "candidate_id": f"cand_{c}",
            "submission_status": "completed",
            "tasks": [
                {
                    "task": f"t_{t}",
                    "importance": "required",
                    "qualification_tier": t,
                    "gate_status": "PASS" if t <= (through_tier - c) else "FAIL",
                    "compatibility_key": f"k_{t}",
                }
                for t in range(1, through_tier + 1)
            ],
        }
        for c in range(2)
    ]
    summary = select_configuration(trials, through_tier=through_tier)
    # Candidate 0 passed all through_tier, candidate 1 failed the top tier
    assert summary["selected_candidate_id"] == "cand_0"
    assert summary["required_passes_by_tier"] == [1] * through_tier


def test_study_validation_accepts_tier_4_and_100():
    """validate_study accepts through_tier >= 4, rejects < 1 and booleans."""
    from experiments.forge.studies import scaffold
    base_study = scaffold(
        "test-study", "candidate_x", "base", "discriminator_stability",
        hypothesis="Testing dynamic tier support in study",
        execution_backend="cuda"
    )
    base_study["scope"]["through_tier"] = 4
    validate_study(base_study)

    # Tier 100
    study_100 = deepcopy(base_study)
    study_100["scope"]["through_tier"] = 100
    validate_study(study_100)

    # Invalid values
    for bad in [0, -1, True, False, "4", 4.0]:
        bad_study = deepcopy(base_study)
        bad_study["scope"]["through_tier"] = bad
        with pytest.raises(ValueError, match="study scope"):
            validate_study(bad_study)


def test_decision_contracts_validation_accepts_tier_4_and_100():
    """validate_shape accepts through_tier >= 4, rejects < 1 and booleans."""
    from experiments.forge import decision_contracts as decisions
    contract = {
        "schema_version": 1,
        "status": "ready",
        "prior_evidence": [{
            "path": "reports/prior.json", "sha256": "0" * 64, "selector": [],
            "identity": {"record_id": "r1"}, "use": "motivation_only"
        }],
        "control": {
            "candidate_id": "base",
            "task_map": {"t1": "t1"},
            "binding_sha256": "0" * 64,
        },
        "candidate_binding_sha256": "0" * 64,
        "substantive_delta": ["delta"],
        "scope": {
            "view": "test_view",
            "through_tier": 4,
            "task_ids": ["t1"],
            "max_rounds": 1,
            "candidate_budget_seconds": 100,
            "campaign_budget_seconds": 100,
            "protocol_sha256": "0" * 64,
            "source_digest": "0" * 64,
            "execution_backend": "cuda",
            "runtime_cohort_sha256": "0" * 64,
            "jobs_sha256": "0" * 64,
        },
        "prediction": {"task_id": "t1", "metric": "score", "op": ">=", "threshold": 0.8, "phase": "final"},
        "falsifier": {"task_id": "t1", "metric": "score", "op": "<", "threshold": 0.5, "phase": "final"},
        "competing_explanation": "Alternative explanation text here.",
        "terminal_rules": deepcopy(decisions.OUTCOMES),
    }
    validate_shape(contract)

    # Tier 100
    contract_100 = deepcopy(contract)
    contract_100["scope"]["through_tier"] = 100
    validate_shape(contract_100)

    # Invalid values
    for bad in [0, -1, True, False, "4", 4.0]:
        bad_contract = deepcopy(contract)
        bad_contract["scope"]["through_tier"] = bad
        with pytest.raises(ValueError, match="decision_contract scope"):
            validate_shape(bad_contract)

