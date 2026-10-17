"""Adversarial stress-test suite for ParticleGAN Forge N-Tier framework (Tier 5 hardening).

White-box adversarial challenge covering:
1. Arbitrary large N scaling (Tier 10, Tier 50, Tier 100, Tier 200).
2. Contiguity enforcement under shuffled, reversed, negative, and non-integer tiers.
3. Hold metadata edge cases: summaries (length/whitespace), explanations (syntax/missing files/missing anchors),
   re-entry criteria, schema/field completeness, and held dependency consistency.
4. Fallback rendering in markdown tables and headers for extreme tier numbers and multi-view depths.
"""

from copy import deepcopy
import math
from pathlib import Path
import random
import pytest

from experiments.forge.views import (
    qualify, validate_view, IMPORTANCES, HOLD_SCHEMA, HOLD_REASONS,
    plain_summary, explanation_reference
)
from experiments.forge.contracts import atomic_json
from experiments.forge.tier_report import (
    build_report, render_markdown, tier_name, TIER_NAMES
)
from experiments.forge.family_reports import role_label, role_note
from experiments.forge.dispositions import _tier_counts, unresolved_explanations


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_task(name: str, *, adapter: str = "test_host", dependencies=None) -> dict:
    """Generate a minimal valid task dictionary conforming to Forge schema."""
    return dict(
        schema_version=1,
        id=name,
        adapter=adapter,
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
        resources={"timeout_seconds": 60},
        requires_capabilities=[],
        dependencies=list(dependencies or []),
    )


def make_view(*assignments, view_id: str = "adversarial_view", revision: int = 1,
              evidence_scope: str | None = None) -> dict:
    """Generate a valid view dictionary with given (task_id, tier, importance) tuples."""
    view = dict(
        schema_version=1,
        id=view_id,
        revision=revision,
        goal="quality",
        eligibility={},
        calibration={"status": "provisional"},
        assignments=[
            dict(task=task, qualification_tier=tier, importance=importance, order=order)
            for order, (task, tier, importance) in enumerate(assignments)
        ],
    )
    if evidence_scope is not None:
        view["evidence_scope"] = evidence_scope
    return view


def make_receipt(task: dict, *, bad_steps=(), final: float = 0.5, cost_seconds: float = 2.0) -> dict:
    """Generate a mock evaluation receipt for a task."""
    steps = task["execution"]["steps"]
    observations = [
        dict(step=math.ceil(i * steps / 24), error=2.0 if i in bad_steps else 0.5)
        for i in range(1, 25)
    ]
    gate_status = "PASS" if final <= 1.0 and not bad_steps else "FAIL"
    return dict(
        task_id=task["id"],
        compatibility_key="compatible",
        gate_status=gate_status,
        evidence=dict(observations=observations, live=dict(error=final)),
        cost=dict(seconds=cost_seconds),
    )


def make_hold(name: str, *, reason="no_global_recipe", blocked_by=(),
              summary=None, explanation=None, reentry=None) -> dict:
    """Generate valid forge_hold_v1 metadata."""
    return dict(
        schema="forge_hold_v1",
        previous_importance="required",
        reason=reason,
        blocked_by=list(blocked_by),
        summary=summary or f"Fixture: {name} is measured and shown, but not counted.",
        explanation=explanation or f"reports/forge/test-dispositions-20261015/README.md#{name}",
        reentry=reentry or f"Count {name} again when one global recipe passes it.",
    )


# ---------------------------------------------------------------------------
# Area 1: Arbitrary Large N Scaling (Tier 5, 10, 25, 50, 100)
# ---------------------------------------------------------------------------

class TestArbitraryLargeNTiers:
    """Verify system scaling across arbitrary large N qualification tiers."""

    @pytest.mark.parametrize("n_tiers", [5, 10, 25, 50, 100])
    def test_validation_and_disposition_scaling_up_to_tier_100(self, n_tiers):
        """Stress-test view validation and disposition counts up to Tier N."""
        tasks = {f"task_{i}": make_task(f"task_{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"task_{i}", i, "required") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id=f"view_scale_{n_tiers}")

        validate_view(view, tasks)

        counts = _tier_counts(view)
        assert len(counts["required"]) == n_tiers
        assert counts["required"] == [1] * n_tiers
        assert counts["held"] == [0] * n_tiers
        assert counts["removed"] == [0] * n_tiers

    @pytest.mark.parametrize("n_tiers", [5, 10, 25, 50, 100])
    def test_qualification_full_pass_scaling(self, n_tiers):
        """Verify that passing all required tasks in 1..N grants Tier N qualification."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id=f"pass_scale_{n_tiers}")
        receipts = [make_receipt(tasks[f"t{i}"], final=0.4) for i in range(1, n_tiers + 1)]

        result = qualify(view, tasks, receipts)

        assert result["status"] == "PASS"
        assert result["qualified_tier"] == n_tiers
        assert result["eligible"] is True
        assert result["required_passed"] == n_tiers
        assert result["required_total"] == n_tiers
        assert len(result["tiers"]) == n_tiers
        assert result["counted_tiers"] == list(range(1, n_tiers + 1))
        assert result["held_tiers"] == []
        assert result["blockers"] == []

    @pytest.mark.parametrize("fail_tier, n_tiers", [
        (1, 10),
        (5, 10),
        (10, 10),
        (1, 50),
        (25, 50),
        (50, 50),
        (1, 100),
        (50, 100),
        (100, 100),
    ])
    def test_qualification_failure_at_arbitrary_tier_boundary(self, fail_tier, n_tiers):
        """Verify that a failure at arbitrary tier k halts qualification at k-1 and exposes blockers."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id=f"fail_at_{fail_tier}_of_{n_tiers}")

        # Task at fail_tier fails; others pass
        receipts = [
            make_receipt(tasks[f"t{i}"], final=2.0 if i == fail_tier else 0.4)
            for i in range(1, n_tiers + 1)
        ]

        result = qualify(view, tasks, receipts)

        assert result["status"] == "FAIL"
        assert result["qualified_tier"] == fail_tier - 1
        assert result["eligible"] is False
        assert len(result["blockers"]) == 1
        assert result["blockers"][0]["task_id"] == f"t{fail_tier}"
        assert result["blockers"][0]["status"] == "FAIL"

    @pytest.mark.parametrize("n_tiers", [5, 10, 25, 50, 100])
    def test_qualification_highest_tier_held_scaling(self, n_tiers):
        """When highest tier N is held and 1..N-1 pass, view status is HELD and qualified_tier is N-1."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"t{i}", i, "required" if i < n_tiers else "held") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id=f"held_highest_{n_tiers}")
        view["assignments"][-1]["hold"] = make_hold(f"t{n_tiers}")

        validate_view(view, tasks)
        receipts = [make_receipt(tasks[f"t{i}"], final=0.4) for i in range(1, n_tiers + 1)]

        result = qualify(view, tasks, receipts)

        assert result["status"] == "HELD"
        assert result["qualified_tier"] == n_tiers - 1
        assert result["eligible"] is False
        assert result["counted_tiers"] == list(range(1, n_tiers))
        assert result["held_tiers"] == [n_tiers]
        assert result["held_passed"] == 1
        assert result["held_total"] == 1

    @pytest.mark.parametrize("n_tiers", [6, 10, 20, 50])
    def test_qualification_interleaved_held_tiers_scaling(self, n_tiers):
        """Verify that intermediate held tiers do not halt qualification of subsequent required tiers."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        # Even tiers are held, odd tiers are required
        assignments = []
        for i in range(1, n_tiers + 1):
            imp = "held" if i % 2 == 0 else "required"
            assignments.append((f"t{i}", i, imp))

        view = make_view(*assignments, view_id=f"interleaved_{n_tiers}")
        for a in view["assignments"]:
            if a["importance"] == "held":
                a["hold"] = make_hold(a["task"])

        validate_view(view, tasks)
        receipts = [make_receipt(tasks[f"t{i}"], final=0.4) for i in range(1, n_tiers + 1)]

        result = qualify(view, tasks, receipts)

        highest_required = max(i for i in range(1, n_tiers + 1) if i % 2 != 0)
        expected_status = "PASS" if highest_required == n_tiers else "HELD"

        assert result["status"] == expected_status
        assert result["qualified_tier"] == highest_required
        expected_counted = [i for i in range(1, n_tiers + 1) if i % 2 != 0]
        expected_held = [i for i in range(1, n_tiers + 1) if i % 2 == 0]
        assert result["counted_tiers"] == expected_counted
        assert result["held_tiers"] == expected_held

    @pytest.mark.parametrize("n_tiers", [5, 10, 50])
    def test_qualification_all_tiers_held_scaling(self, n_tiers):
        """When all tiers 1..N are held, view reports HELD, qualified_tier 0, and counted_tiers []."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"t{i}", i, "held") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id=f"all_held_{n_tiers}")
        for a in view["assignments"]:
            a["hold"] = make_hold(a["task"])

        validate_view(view, tasks)
        receipts = [make_receipt(tasks[f"t{i}"], final=0.4) for i in range(1, n_tiers + 1)]

        result = qualify(view, tasks, receipts)

        assert result["status"] == "HELD"
        assert result["qualified_tier"] == 0
        assert result["eligible"] is False
        assert result["counted_tiers"] == []
        assert result["held_tiers"] == list(range(1, n_tiers + 1))
        assert result["required_total"] == 0

    def test_long_dependency_chain_across_arbitrary_tiers(self):
        """A linear dependency chain across 20 tiers: task_k depends on task_{k-1}."""
        n_tiers = 20
        tasks = {}
        for i in range(1, n_tiers + 1):
            deps = [dict(task=f"t{i-1}", kind="gate")] if i > 1 else []
            tasks[f"t{i}"] = make_task(f"t{i}", dependencies=deps)

        assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]
        view = make_view(*assignments, view_id="linear_chain_20")

        validate_view(view, tasks)

        # If task 1 fails, all downstream tasks must be marked BLOCKED
        receipts = [make_receipt(tasks["t1"], final=2.0)] + [
            make_receipt(tasks[f"t{i}"], final=0.4) for i in range(2, n_tiers + 1)
        ]
        result = qualify(view, tasks, receipts)
        assert result["status"] == "FAIL"
        assert result["qualified_tier"] == 0
        task_statuses = result["task_statuses"]
        assert task_statuses["t1"] == "FAIL"
        for i in range(2, n_tiers + 1):
            assert task_statuses[f"t{i}"] == "BLOCKED"

    def test_technique_inventory_tier_totals_scaling(self):
        """Verify dynamic required_tier_totals dictionary construction up to Tier 50."""
        n_tiers = 50
        tasks_list = [
            {"qualification_tier": i, "importance": "required", "task": f"t{i}"}
            for i in range(1, n_tiers + 1)
        ]
        row = {"tasks": tasks_list}
        through_tier = n_tiers

        max_tier = max([through_tier] + [t["qualification_tier"] for t in row["tasks"]])
        required_tier_totals = {
            str(tier): sum(t["qualification_tier"] == tier and t["importance"] == "required"
                           for t in row["tasks"]) for tier in range(1, max_tier + 1)
        }

        assert len(required_tier_totals) == n_tiers
        assert all(required_tier_totals[str(i)] == 1 for i in range(1, n_tiers + 1))


# ---------------------------------------------------------------------------
# Area 2: Contiguity Enforcement Under Shuffled, Reversed, Negative Tiers
# ---------------------------------------------------------------------------

class TestContiguityAndShuffledTiers:
    """Adversarial stress-testing of contiguity and ordering rules."""

    @pytest.mark.parametrize("seed", [42, 1337, 2026, 9999])
    def test_contiguity_with_shuffled_and_reversed_tier_order(self, seed):
        """View validation must succeed when assignments are presented in shuffled or reversed order."""
        n_tiers = 15
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]

        # Reversed order
        reversed_view = make_view(*reversed(assignments), view_id="reversed_view")
        validate_view(reversed_view, tasks)

        # Random shuffled order
        shuffled = list(assignments)
        random.Random(seed).shuffle(shuffled)
        shuffled_view = make_view(*shuffled, view_id=f"shuffled_view_{seed}")
        validate_view(shuffled_view, tasks)

    def test_qualification_invariance_to_assignment_ordering(self):
        """qualify() output must be strictly identical regardless of view assignment list ordering."""
        n_tiers = 10
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, n_tiers + 1)}
        base_assignments = [(f"t{i}", i, "required") for i in range(1, n_tiers + 1)]
        receipts = [make_receipt(tasks[f"t{i}"], final=0.5) for i in range(1, n_tiers + 1)]

        # Sorted ascending
        v_sorted = make_view(*base_assignments, view_id="v_sorted")
        res_sorted = qualify(v_sorted, tasks, receipts)

        # Sorted descending
        v_desc = make_view(*reversed(base_assignments), view_id="v_desc")
        res_desc = qualify(v_desc, tasks, receipts)

        # Shuffled
        shuffled_items = list(base_assignments)
        random.Random(123).shuffle(shuffled_items)
        v_shuffled = make_view(*shuffled_items, view_id="v_shuffled")
        res_shuffled = qualify(v_shuffled, tasks, receipts)

        for key in ["status", "qualified_tier", "eligible", "required_passed", "required_total",
                    "task_statuses", "counted_tiers", "held_tiers"]:
            assert res_sorted[key] == res_desc[key] == res_shuffled[key]

    @pytest.mark.parametrize("invalid_tier", [
        0, -1, -2, -10, -100, -999999, -2**63
    ])
    def test_rejection_of_negative_and_zero_tiers(self, invalid_tier):
        """validate_view must reject qualification_tier <= 0."""
        tasks = {"t1": make_task("t1")}
        view = make_view(("t1", 1, "required"))
        view["assignments"][0]["qualification_tier"] = invalid_tier

        with pytest.raises(ValueError, match="qualification_tier must be an integer >= 1"):
            validate_view(view, tasks)

    @pytest.mark.parametrize("invalid_type", [
        False, True, None, 1.0, 2.5, -1.0, float("inf"), float("-inf"), float("nan"),
        "1", "0", "-1", "Tier 1", [1], (1,), {"tier": 1}, set()
    ])
    def test_rejection_of_non_integer_tier_types(self, invalid_type):
        """validate_view must reject booleans, floats, strings, and collections as qualification_tier."""
        tasks = {"t1": make_task("t1")}
        view = make_view(("t1", 1, "required"))
        view["assignments"][0]["qualification_tier"] = invalid_type

        with pytest.raises(ValueError, match="qualification_tier must be an integer >= 1"):
            validate_view(view, tasks)

    @pytest.mark.parametrize("tiers", [
        [2, 3, 4],              # Missing Tier 1
        [1, 3, 4],              # Missing Tier 2
        [1, 2, 4],              # Missing Tier 3
        [1, 2, 3, 5],           # Missing Tier 4
        [1, 2, 3, 4, 6],        # Missing Tier 5
        [1, 5],                 # Missing 2, 3, 4
        [1, 2, 8],              # Missing 3..7
        [10],                   # Single disconnected tier
    ])
    def test_rejection_of_missing_intermediate_tiers_and_boundaries(self, tiers):
        """validate_view must reject any tier configuration with gaps from 1 to H."""
        tasks = {f"t{t}": make_task(f"t{t}") for t in tiers}
        assignments = [(f"t{t}", t, "required") for t in tiers]
        view = make_view(*assignments)

        with pytest.raises(ValueError, match="empty required tier: qualification tiers must be contiguous from Tier 1"):
            validate_view(view, tasks)

    @pytest.mark.parametrize("importance", ["ranking", "diagnostic"])
    def test_ranking_or_diagnostic_at_highest_tier_violates_contiguity(self, importance):
        """If highest tier H contains only ranking or diagnostic tasks, contiguity check must fail."""
        tasks = {f"t{i}": make_task(f"t{i}") for i in range(1, 4)}
        # Tiers 1 and 2 are required, Tier 3 is ranking/diagnostic
        view = make_view(
            ("t1", 1, "required"),
            ("t2", 2, "required"),
            ("t3", 3, importance),
        )
        with pytest.raises(ValueError, match="empty required tier: qualification tiers must be contiguous from Tier 1"):
            validate_view(view, tasks)

    def test_diagnostic_view_contiguity_and_tier_restrictions(self):
        """A diagnostic view cannot assign tasks to Tier > 1 or use non-diagnostic importance."""
        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}

        # Diagnostic task at Tier 2 in diagnostic view -> REJECT
        v_diag_t2 = make_view(("t1", 1, "diagnostic"), ("t2", 2, "diagnostic"),
                              evidence_scope="research_diagnostic")
        with pytest.raises(ValueError, match="diagnostic views require only diagnostic tasks in Tier 1"):
            validate_view(v_diag_t2, tasks)

        # Required task at Tier 1 in diagnostic view -> REJECT
        v_req_t1 = make_view(("t1", 1, "required"), evidence_scope="research_diagnostic")
        with pytest.raises(ValueError, match="diagnostic views require only diagnostic tasks in Tier 1"):
            validate_view(v_req_t1, tasks)

        # Valid diagnostic view -> ACCEPT
        v_valid = make_view(("t1", 1, "diagnostic"), ("t2", 1, "diagnostic"),
                            evidence_scope="research_diagnostic")
        validate_view(v_valid, tasks)


# ---------------------------------------------------------------------------
# Area 3: Hold Metadata Edge Cases & Validation Hardening
# ---------------------------------------------------------------------------

class TestHoldMetadataEdgeCases:
    """Stress-test hold metadata: summaries, explanations, anchors, schemas, dependencies."""

    @pytest.mark.parametrize("invalid_summary, match", [
        ("", "20-280"),
        ("Too short", "20-280"),
        ("x" * 19, "20-280"),
        ("x" * 281, "20-280"),
        ("x" * 500, "20-280"),
    ])
    def test_summary_length_boundaries_and_rejection(self, invalid_summary, match):
        """plain_summary must reject strings with length < 20 or > 280."""
        with pytest.raises(ValueError, match=match):
            plain_summary(invalid_summary, "test_summary")

    @pytest.mark.parametrize("valid_len", [20, 21, 150, 279, 280])
    def test_summary_valid_lengths(self, valid_len):
        """plain_summary must accept strings with length exactly between 20 and 280."""
        text = "a" * valid_len
        plain_summary(text, "valid_summary")

    @pytest.mark.parametrize("untrimmed_summary", [
        "  " + "x" * 25,
        "x" * 25 + "  ",
        " " * 30,
        "\n" + "x" * 25,
        "x" * 25 + "\t",
    ])
    def test_summary_whitespace_and_type_rejection(self, untrimmed_summary):
        """plain_summary must reject untrimmed strings and non-strings."""
        with pytest.raises(ValueError, match="plain-language summary of 20-280 characters"):
            plain_summary(untrimmed_summary, "test_trim")

        with pytest.raises(ValueError, match="plain-language summary of 20-280 characters"):
            plain_summary(None, "test_none")

        with pytest.raises(ValueError, match="plain-language summary of 20-280 characters"):
            plain_summary(12345678901234567890, "test_int")

    @pytest.mark.parametrize("invalid_ref", [
        "",
        "reports/forge/README.md",                   # Missing #
        "reports/forge/README.md#a#b",               # Multiple #
        "reports/forge/README.md#",                  # Empty anchor
        "reports/forge/README.txt#anchor",           # Non-.md extension
        "reports/forge/README.json#anchor",          # Non-.md extension
        "reports/forge/README#anchor",               # Missing extension
        "/reports/forge/README.md#anchor",           # Absolute path
        "/tmp/test.md#anchor",                       # Absolute path
        "../README.md#anchor",                       # Traversal ..
        "reports/../../README.md#anchor",            # Traversal ..
        "https://example.com/README.md#anchor",      # URL scheme
        "http://example.com/README.md#anchor",       # URL scheme
        "C:/README.md#anchor",                       # Drive letter
        "~/README.md#anchor",                        # Tilde home directory
        "reports\\forge\\README.md#anchor",          # Backslash
        "reports/forge/READ ME.md#anchor",           # Space in path
        "reports/forge/README.md#anchor with space", # Space in anchor
        " reports/forge/README.md#anchor",           # Leading space
    ])
    def test_explanation_reference_syntax_boundaries(self, invalid_ref):
        """explanation_reference must reject invalid syntax, traversal, non-.md, and whitespace."""
        with pytest.raises(ValueError, match="repo-relative path#anchor explanation"):
            explanation_reference(invalid_ref, "test_ref")

    def test_unresolved_explanations_detects_missing_file(self, tmp_path):
        """unresolved_explanations must catch when the referenced Markdown file does not exist."""
        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        view["assignments"][1]["hold"] = make_hold(
            "t2", explanation="reports/forge/nonexistent_file.md#my_anchor"
        )

        problems = unresolved_explanations(tmp_path, [view])
        assert len(problems) == 1
        assert "nonexistent_file.md" in problems[0]
        assert '<a name="my_anchor"></a>' in problems[0]

    def test_unresolved_explanations_detects_missing_anchor_in_existing_file(self, tmp_path):
        """unresolved_explanations must catch when file exists but explicit anchor is absent."""
        md_file = tmp_path / "reports/forge/doc.md"
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text("# Documentation\n\nSome other content here without anchor.\n")

        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        view["assignments"][1]["hold"] = make_hold(
            "t2", explanation="reports/forge/doc.md#missing_anchor"
        )

        problems = unresolved_explanations(tmp_path, [view])
        assert len(problems) == 1
        assert "reports/forge/doc.md" in problems[0]
        assert '<a name="missing_anchor"></a>' in problems[0]

    def test_unresolved_explanations_rejects_markdown_header_without_html_anchor(self, tmp_path):
        """A plain Markdown header `## my_anchor` does not satisfy Forge anchor requirement."""
        md_file = tmp_path / "reports/forge/doc.md"
        md_file.parent.mkdir(parents=True, exist_ok=True)
        # Markdown header without <a name="...">
        md_file.write_text("# Title\n\n## my_anchor\n\nDetails go here.\n")

        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        view["assignments"][1]["hold"] = make_hold(
            "t2", explanation="reports/forge/doc.md#my_anchor"
        )

        problems = unresolved_explanations(tmp_path, [view])
        assert len(problems) == 1
        assert '<a name="my_anchor"></a>' in problems[0]

    @pytest.mark.parametrize("anchor_html", [
        '<a name="test_anchor"></a>',
        "<a name='test_anchor'></a>",
        '<a id="test_anchor"></a>',
        "<a id='test_anchor'></a>",
        '<a  name = "test_anchor" ></a>',
    ])
    def test_unresolved_explanations_accepts_valid_anchor_variants(self, tmp_path, anchor_html):
        """Explicit <a name=...> and <a id=...> tags with quotes/spacing variants are recognized."""
        md_file = tmp_path / "reports/forge/doc.md"
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(f"# Doc\n\n{anchor_html}\n## Section\n")

        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        view["assignments"][1]["hold"] = make_hold(
            "t2", explanation="reports/forge/doc.md#test_anchor"
        )

        problems = unresolved_explanations(tmp_path, [view])
        assert problems == []

    @pytest.mark.parametrize("invalid_reentry", [
        "",
        "x" * 19,
        "   short   ",
        None,
        12345,
    ])
    def test_reentry_length_and_type_boundaries(self, invalid_reentry):
        """Hold re-entry criterion must be a string with stripped length >= 20."""
        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        view["assignments"][1]["hold"] = make_hold("t2")
        view["assignments"][1]["hold"]["reentry"] = invalid_reentry

        with pytest.raises(ValueError, match="concrete re-entry criterion"):
            validate_view(view, tasks)

    def test_hold_metadata_schema_and_field_completeness(self):
        """Hold metadata must strictly match exact required keys, schema name, and reason."""
        tasks = {"t1": make_task("t1"), "t2": make_task("t2")}
        view = make_view(("t1", 1, "required"), ("t2", 2, "held"))
        base_hold = make_hold("t2")

        # Extra field
        h_extra = dict(base_hold, extra_field="forbidden")
        view["assignments"][1]["hold"] = h_extra
        with pytest.raises(ValueError, match="held assignments need hold metadata with exactly"):
            validate_view(view, tasks)

        # Missing field
        h_missing = dict(base_hold)
        del h_missing["reentry"]
        view["assignments"][1]["hold"] = h_missing
        with pytest.raises(ValueError, match="held assignments need hold metadata with exactly"):
            validate_view(view, tasks)

        # Bad schema
        h_bad_schema = dict(base_hold, schema="forge_hold_v2")
        view["assignments"][1]["hold"] = h_bad_schema
        with pytest.raises(ValueError, match=f"only a previously required task can be held \\({HOLD_SCHEMA}\\)"):
            validate_view(view, tasks)

        # Bad previous_importance
        h_prev = dict(base_hold, previous_importance="ranking")
        view["assignments"][1]["hold"] = h_prev
        with pytest.raises(ValueError, match=f"only a previously required task can be held \\({HOLD_SCHEMA}\\)"):
            validate_view(view, tasks)

        # Bad reason
        h_reason = dict(base_hold, reason="too_difficult")
        view["assignments"][1]["hold"] = h_reason
        with pytest.raises(ValueError, match="hold reason must be one of"):
            validate_view(view, tasks)

    def test_hold_dependencies_consistency_and_blocked_by(self):
        """When a held task depends on another held task, blocked_by and reason must strictly match."""
        tasks = {
            "t1": make_task("t1"),
            "t2": make_task("t2", dependencies=[dict(task="t1", kind="gate")]),
        }
        view = make_view(("t1", 1, "held"), ("t2", 2, "held"))
        view["assignments"][0]["hold"] = make_hold("t1", reason="unsolved")
        # t2 has t1 as held prerequisite -> reason MUST be held_prerequisite, blocked_by MUST be ["t1"]
        view["assignments"][1]["hold"] = make_hold("t2", reason="held_prerequisite", blocked_by=["t1"])

        validate_view(view, tasks)

        # Mismatch in blocked_by: empty instead of ["t1"]
        view["assignments"][1]["hold"]["blocked_by"] = []
        with pytest.raises(ValueError, match="hold blocked_by must list exactly its held prerequisites"):
            validate_view(view, tasks)

        # Unsorted blocked_by: ["t2", "t1"]
        tasks["t3"] = make_task("t3", dependencies=[dict(task="t1", kind="gate"), dict(task="t2", kind="gate")])
        v3 = make_view(("t1", 1, "held"), ("t2", 1, "held"), ("t3", 2, "held"))
        v3["assignments"][0]["hold"] = make_hold("t1", reason="unsolved")
        v3["assignments"][1]["hold"] = make_hold("t2", reason="unsolved")
        v3["assignments"][2]["hold"] = make_hold("t3", reason="held_prerequisite", blocked_by=["t2", "t1"])
        with pytest.raises(ValueError, match="distinct task ids in sorted order"):
            validate_view(v3, tasks)

        # Duplicate in blocked_by
        v3["assignments"][2]["hold"]["blocked_by"] = ["t1", "t1"]
        with pytest.raises(ValueError, match="distinct task ids in sorted order"):
            validate_view(v3, tasks)

        # Required task depending on a held task -> REJECT
        v_req_dep = make_view(("t1", 1, "held"), ("t2", 2, "required"))
        v_req_dep["assignments"][0]["hold"] = make_hold("t1", reason="unsolved")
        with pytest.raises(ValueError, match="required task depends on held task t1"):
            validate_view(v_req_dep, tasks)


# ---------------------------------------------------------------------------
# Area 4: Fallback Rendering in Markdown Tables for Extreme Tier Numbers
# ---------------------------------------------------------------------------

class TestFallbackRenderingExtremeTierNumbers:
    """Stress-test markdown generation, table alignment, and tier naming under extreme tier counts."""

    @pytest.mark.parametrize("tier_val, expected", [
        (1, "smoke"),
        (2, "quality"),
        (3, "endurance"),
        (4, "stress_and_formulations"),
        (5, "tier-5"),
        (6, "tier-6"),
        (10, "tier-10"),
        (50, "tier-50"),
        (100, "tier-100"),
        (999, "tier-999"),
        (10000, "tier-10000"),
    ])
    def test_tier_name_mapping_and_fallback_property(self, tier_val, expected):
        """tier_name must return canonical names for 1..4 and tier-N fallback for N>=5."""
        assert tier_name(tier_val) == expected

    @pytest.mark.parametrize("n_tiers", [5, 10, 20, 50, 100])
    def test_markdown_table_column_alignment_and_headers_scaling(self, tmp_path, n_tiers):
        """Verify markdown report table columns remain strictly aligned for arbitrary N tiers."""
        root = tmp_path / f"repo_tier_{n_tiers}"
        tasks_dir = root / "configs/forge/tasks"
        views_dir = root / "configs/forge/views"

        for i in range(1, n_tiers + 1):
            atomic_json(tasks_dir / f"task_{i}.json", make_task(f"task_{i}"))

        assignments = [
            {"task": f"task_{i}", "qualification_tier": i, "importance": "required", "order": 0}
            for i in range(1, n_tiers + 1)
        ]
        atomic_json(views_dir / f"view_{n_tiers}.json", {
            "schema_version": 1,
            "id": f"view_{n_tiers}",
            "revision": 1,
            "goal": "stress",
            "eligibility": {},
            "calibration": {"status": "provisional"},
            "assignments": assignments,
        })

        report = build_report(root)
        md = render_markdown(report, root, root / "reports/forge/EXPERIMENTS_BY_TIER.md")

        # 1. Expected intro text for N >= 5
        assert "Tier 1 is smoke, Tier 2 is quality, Tier 3 is endurance, Tier 4 is stress_and_formulations, and subsequent tiers follow tier-N." in md

        # 2. Section headings
        assert f"### Tier {n_tiers}: tier-{n_tiers}" in md
        assert f"### Tier 4: stress_and_formulations" in md

        # 3. Table header and divider column counts
        header_cols = []
        divider_cols = []
        view_cols = []
        for line in md.splitlines():
            if line.startswith("| View | Revision |"):
                header_cols = [c.strip() for c in line.split("|") if c.strip()]
            elif line.startswith("| --- | ---: |"):
                divider_cols = [c.strip() for c in line.split("|") if c.strip()]
            elif line.startswith(f"| [view_{n_tiers}]"):
                view_cols = [c.strip() for c in line.split("|") if c.strip()]

        expected_col_count = n_tiers + 3  # View, Revision, Tier 1..N, Calibration
        assert len(header_cols) == expected_col_count
        assert len(divider_cols) == expected_col_count
        assert len(view_cols) == expected_col_count

    def test_multi_view_mixed_tier_depths_alignment(self, tmp_path):
        """When views have different max tiers (e.g. View A has 3 tiers, View B has 10 tiers),
        both view rows must format with matching column counts in the shared table.
        """
        root = tmp_path / "repo_multi_depth"
        tasks_dir = root / "configs/forge/tasks"
        views_dir = root / "configs/forge/views"

        for i in range(1, 11):
            atomic_json(tasks_dir / f"task_{i}.json", make_task(f"task_{i}"))

        # View A: Tiers 1..3
        atomic_json(views_dir / "view_short.json", {
            "schema_version": 1, "id": "view_short", "revision": 1, "goal": "short",
            "eligibility": {}, "calibration": {"status": "provisional"},
            "assignments": [{"task": f"task_{i}", "qualification_tier": i, "importance": "required", "order": 0}
                            for i in range(1, 4)],
        })

        # View B: Tiers 1..10
        atomic_json(views_dir / "view_deep.json", {
            "schema_version": 1, "id": "view_deep", "revision": 1, "goal": "deep",
            "eligibility": {}, "calibration": {"status": "provisional"},
            "assignments": [{"task": f"task_{i}", "qualification_tier": i, "importance": "required", "order": 0}
                            for i in range(1, 11)],
        })

        report = build_report(root)
        md = render_markdown(report, root, root / "reports/forge/EXPERIMENTS_BY_TIER.md")

        # Global max tier is 10
        expected_cols = 10 + 3
        short_row_cols = []
        deep_row_cols = []
        for line in md.splitlines():
            if line.startswith("| [view_short]"):
                short_row_cols = [c.strip() for c in line.split("|") if c.strip()]
            elif line.startswith("| [view_deep]"):
                deep_row_cols = [c.strip() for c in line.split("|") if c.strip()]

        assert len(short_row_cols) == expected_cols
        assert len(deep_row_cols) == expected_cols
        # In view_short, tiers 4..10 must render as "0 tasks"
        assert short_row_cols[5:12] == ["0 tasks"] * 7

    def test_held_status_text_generation_for_extreme_tiers(self, tmp_path):
        """When the highest tier (e.g. Tier 15) has only held tasks, the explanation specifically names Tier 15."""
        root = tmp_path / "repo_held_extreme"
        tasks_dir = root / "configs/forge/tasks"
        views_dir = root / "configs/forge/views"

        for i in range(1, 16):
            atomic_json(tasks_dir / f"task_{i}.json", make_task(f"task_{i}"))

        assignments = [
            {"task": f"task_{i}", "qualification_tier": i,
             "importance": "required" if i < 15 else "held", "order": 0}
            for i in range(1, 16)
        ]
        assignments[-1]["hold"] = make_hold("task_15")

        atomic_json(views_dir / "view_held15.json", {
            "schema_version": 1, "id": "view_held15", "revision": 1, "goal": "held_test",
            "eligibility": {}, "calibration": {"status": "provisional"},
            "assignments": assignments,
        })

        report = build_report(root)
        md = render_markdown(report, root, root / "reports/forge/EXPERIMENTS_BY_TIER.md")

        expected_msg = (
            "Tier 15, the highest tier, has only held tasks, so while they are held this view never shows PASS: "
            "a candidate that passes every counted tier gets HELD."
        )
        assert expected_msg in md

    def test_all_held_view_status_text_generation(self, tmp_path):
        """When all tiers in a view are held, render_markdown prints the all-held status notice."""
        root = tmp_path / "repo_all_held"
        tasks_dir = root / "configs/forge/tasks"
        views_dir = root / "configs/forge/views"

        for i in range(1, 6):
            atomic_json(tasks_dir / f"task_{i}.json", make_task(f"task_{i}"))

        assignments = [
            {"task": f"task_{i}", "qualification_tier": i, "importance": "held", "order": 0,
             "hold": make_hold(f"task_{i}")}
            for i in range(1, 6)
        ]

        atomic_json(views_dir / "view_all_held.json", {
            "schema_version": 1, "id": "view_all_held", "revision": 1, "goal": "all_held",
            "eligibility": {}, "calibration": {"status": "provisional"},
            "assignments": assignments,
        })

        report = build_report(root)
        md = render_markdown(report, root, root / "reports/forge/EXPERIMENTS_BY_TIER.md")

        expected_msg = "Nothing in this view is counted while its tasks are held, so it reports HELD for every candidate."
        assert expected_msg in md

    def test_markdown_cell_sanitization_with_special_characters(self):
        """Verify role_label and role_note correctly format held and removed explanations."""
        held_assignment = {
            "importance": "held",
            "hold": make_hold("special_task", reason="unsolved"),
        }
        note = role_note(held_assignment)
        assert note is not None
        assert note["reason"] == "unsolved"
        label = role_label(held_assignment)
        assert label == "held: no recipe has passed its current declaration"

        # Prerequisite held label
        prereq_assignment = {
            "importance": "held",
            "hold": make_hold("downstream_task", reason="held_prerequisite", blocked_by=["upstream_task"]),
        }
        label_prereq = role_label(prereq_assignment)
        assert label_prereq == "held: prerequisite upstream_task is held"
