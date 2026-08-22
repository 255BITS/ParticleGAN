#!/usr/bin/env python
"""
leaderboard.py

Promotion boards for the GAN gradient-regularizer study -> results/LEADERBOARD.md.

Where TABLE.md is the full descriptive dump, this file answers one question:
**which recipe should become the repo default?** It reads the same run
directories and reduces every group to a handful of promotion-relevant
numbers, then applies fixed eligibility rules rather than eyeballing a sort.

Two run trees are read:

  * ``results/runs``       -- the study proper. All core metrics come from here.
  * ``results/runs_audit`` -- re-runs of the key groups under the *same run
    names*, with the newer per-mode audit metrics. Training is deterministic,
    so an audit run reproduces the same model as its namesake; the audit tree
    is therefore used to *annotate* a group with ``per_mode_std_ratio`` and
    nothing else. It never supplies W1, hq, recall, collapses or timing, and
    an audit group with no counterpart in the main tree is ignored.

Boards:

  1. **Sharp board** (the promotion track). Eligible iff every seed clears the
     repo bar (100 modes and hq >= 0.9 by step 7000), mean hq >= 0.95, and no
     seed collapsed. Ranked by mean final exact W1, lowest first; the top row
     is the CHAMPION -- unless it is variance-compressed (see below).
  2. **Transport board**. Eligible iff mean mode_recall >= 0.95 with no
     collapses. Ranked the same way. This is the "covers the target" board;
     a recipe can win it while being too blurry for the sharp board.
  3. **Wasserstein section**. Informational only: a different objective, so its
     W1 numbers are not commensurable with the logistic runs and it is never
     promotable.

The variance-compression guard exists because the sharp board's own metrics
can be gamed: a generator that emits tight, under-dispersed blobs at every
mode centre scores excellent W1 and hq while not actually matching the target's
per-mode spread. ``per_mode_std_ratio`` (generated spread / real spread) below
0.8 flags exactly that failure, and a flagged row is barred from CHAMPION even
if it ranks first. Groups whose summaries predate the metric are unflagged --
absence of evidence, not evidence of absence -- and the board says so.

Usage:

    python experiments/leaderboard.py
    python experiments/leaderboard.py --runs_dir results/runs \
        --audit_dir results/runs_audit --out_dir results
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import-path shim for direct execution
    from analyze import DEFAULT_OBJECTIVE, Run, agg, fmt, group_runs, load_runs
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze import DEFAULT_OBJECTIVE, Run, agg, fmt, group_runs, load_runs

# --- the repo bar ----------------------------------------------------------
BAR_MODES = 100.0
BAR_HQ = 0.9
BAR_HORIZON = 7000  # a seed must clear the bar by this step to count

# --- board eligibility -----------------------------------------------------
SHARP_HQ_MIN = 0.95
TRANSPORT_RECALL_MIN = 0.95
VARIANCE_COMPRESSED_BELOW = 0.8

HEADER_NOTE = (
    "Champion = promotion candidate for the repo's default recipe. Sharp board "
    "requires: all seeds pass the repo bar (100 modes & hq>=0.9), mean hq>=0.95, "
    "zero collapses. If per_mode_std_ratio < 0.8, the row is flagged "
    "'variance-compressed' and cannot hold CHAMPION."
)


# ---------------------------------------------------------------------------
#  Per-run and per-group reduction
# ---------------------------------------------------------------------------


def steps_to_bar(run: Run) -> Tuple[float, bool]:
    """
    First eval step at which the run clears the repo bar: all 100 modes covered
    *and* hq >= 0.9, both read from metrics.jsonl at the same eval row.

    Returns (steps, reached). A run that never clears it -- including one with
    no usable timeseries at all -- is reported at ``BAR_HORIZON`` and marked
    not-reached, so the mean over such a group is a lower bound on speed.
    """
    steps = run.series.get("step")
    modes = run.series.get("modes")
    hq = run.series.get("hq")
    if steps is None or modes is None or hq is None or steps.size == 0:
        return float(BAR_HORIZON), False
    ok = (
        np.isfinite(steps)
        & np.isfinite(modes)
        & np.isfinite(hq)
        & (steps <= BAR_HORIZON)
        & (modes >= BAR_MODES)
        & (hq >= BAR_HQ)
    )
    if not ok.any():
        return float(BAR_HORIZON), False
    return float(steps[np.argmax(ok)]), True


@dataclass
class GroupStat:
    """One (variant, arm, coeff) group, reduced to promotion-relevant scalars."""

    label: str
    objective: str
    variant: str
    arm: str
    coeff_label: str
    n_seeds: int
    w1_mean: float
    hq_mean: float
    recall_mean: float
    pmsr_mean: float
    pmsr_std: float
    pmsr_n: int
    pmsr_source: str  # "" | "runs" | "audit"
    collapse_rate: float
    bar_steps_mean: float
    bar_reached: int
    bar_pass: float

    @property
    def coeff(self) -> str:
        return self.coeff_label.replace("p", ".").replace("m", "-")

    @property
    def variance_compressed(self) -> bool:
        """True only when the metric exists *and* falls below the guard."""
        return bool(
            self.pmsr_n
            and math.isfinite(self.pmsr_mean)
            and self.pmsr_mean < VARIANCE_COMPRESSED_BELOW
        )

    def sharp_eligible(self) -> bool:
        return (
            self.bar_pass >= 1.0
            and math.isfinite(self.hq_mean)
            and self.hq_mean >= SHARP_HQ_MIN
            and self.collapse_rate == 0.0
        )

    def transport_eligible(self) -> bool:
        return (
            math.isfinite(self.recall_mean)
            and self.recall_mean >= TRANSPORT_RECALL_MIN
            and self.collapse_rate == 0.0
        )


def group_label(runs: Sequence[Run]) -> str:
    """
    The group's run-name prefix, i.e. the run name with ``_s{seed}`` removed.

    Taken from an actual directory name rather than rebuilt from the parsed
    fields, so the label always matches what is on disk (``b_cap_c1p0_lr2p0``).
    """
    name = runs[0].name
    head, sep, tail = name.rpartition("_s")
    return head if sep and tail.isdigit() else name


def reduce_group(runs: Sequence[Run]) -> GroupStat:
    """Reduce one group's seeds to a GroupStat (all aggregations None-tolerant)."""
    first = runs[0]
    w1_mean, _, n = agg([r.final("w1_exact") for r in runs])
    hq_mean, _, _ = agg([r.final("hq") for r in runs])
    recall_mean, _, _ = agg([r.final("mode_recall") for r in runs])
    pmsr_mean, pmsr_std, pmsr_n = agg([r.final("per_mode_std_ratio") for r in runs])

    bars = [steps_to_bar(r) for r in runs]
    bar_mean, _, _ = agg([v for v, _ in bars])
    reached = sum(1 for _, ok in bars if ok)

    collapse = [1.0 if r.collapse_events > 0 else 0.0 for r in runs]
    return GroupStat(
        label=group_label(runs),
        objective=first.objective,
        variant=first.variant,
        arm=first.arm,
        coeff_label=first.coeff_label,
        n_seeds=n if n else len(runs),
        w1_mean=w1_mean,
        hq_mean=hq_mean,
        recall_mean=recall_mean,
        pmsr_mean=pmsr_mean,
        pmsr_std=pmsr_std,
        pmsr_n=pmsr_n,
        pmsr_source="runs" if pmsr_n else "",
        collapse_rate=float(np.mean(collapse)) if collapse else float("nan"),
        bar_steps_mean=bar_mean,
        bar_reached=reached,
        bar_pass=(reached / len(runs)) if runs else 0.0,
    )


def collect_stats(
    runs_dir: Path, audit_dir: Optional[Path]
) -> Tuple[List[GroupStat], Dict[str, int]]:
    """
    Build the group table from the main tree, then annotate it from the audit
    tree with ``per_mode_std_ratio`` only.
    """
    runs, skips = load_runs(runs_dir)
    stats = {key: reduce_group(rs) for key, rs in group_runs(runs).items()}

    info = {
        "runs": len(runs),
        "groups": len(stats),
        "audit_runs": 0,
        "audit_annotated": 0,
        "audit_orphans": 0,
    }
    info.update({f"skip_{k}": v for k, v in skips.items()})

    if audit_dir is None or not audit_dir.is_dir():
        return list(stats.values()), info

    audit_runs, _ = load_runs(audit_dir)
    info["audit_runs"] = len(audit_runs)
    for key, rs in group_runs(audit_runs).items():
        target = stats.get(key)
        if target is None:
            # Audit-only group: no core metrics to attach it to, so it stays out
            # of the boards entirely rather than appearing with empty columns.
            info["audit_orphans"] += 1
            continue
        mean, std, n = agg([r.final("per_mode_std_ratio") for r in rs])
        if not n:
            continue
        target.pmsr_mean, target.pmsr_std, target.pmsr_n = mean, std, n
        target.pmsr_source = "audit"
        info["audit_annotated"] += 1

    return list(stats.values()), info


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

BOARD_HEADER = (
    "| # | group | arm | coeff | n | W1 (exact) | hq | mode recall | "
    "per-mode σ ratio | collapse | steps→bar (n reached) | bar pass | notes |"
)
BOARD_SEP = "|" + "---|" * 13


def _rank(stats: Sequence[GroupStat]) -> List[GroupStat]:
    """Ascending mean W1; groups without a W1 sort last."""
    return sorted(
        stats,
        key=lambda s: s.w1_mean if math.isfinite(s.w1_mean) else float("inf"),
    )


def _pmsr_cell(s: GroupStat) -> str:
    if not s.pmsr_n:
        return ""
    cell = fmt(s.pmsr_mean, 3)
    if math.isfinite(s.pmsr_std):
        cell += f" ± {fmt(s.pmsr_std, 3)}"
    if s.pmsr_source == "audit":
        cell += " ᴬ"
    return cell


def board_rows(
    stats: Sequence[GroupStat], champion: Optional[GroupStat]
) -> List[str]:
    lines = [BOARD_HEADER, BOARD_SEP]
    for i, s in enumerate(stats, start=1):
        notes: List[str] = []
        if champion is not None and s is champion:
            notes.append("**CHAMPION**")
        if s.variance_compressed:
            notes.append("variance-compressed (ineligible for CHAMPION)")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    f"`{s.label}`",
                    s.arm,
                    s.coeff,
                    str(s.n_seeds),
                    fmt(s.w1_mean, 4),
                    fmt(s.hq_mean, 3),
                    fmt(s.recall_mean, 3),
                    _pmsr_cell(s),
                    fmt(s.collapse_rate, 2),
                    f"{fmt(s.bar_steps_mean, 0)} ({s.bar_reached}/{s.n_seeds})",
                    fmt(s.bar_pass, 2),
                    "; ".join(notes),
                ]
            )
            + " |"
        )
    return lines


def pick_champion(sharp: Sequence[GroupStat]) -> Optional[GroupStat]:
    """
    The best-ranked sharp-board row that is not variance-compressed.

    A flagged row keeps its rank -- it earned it -- but the title passes down
    the board, because a compressed generator wins on W1 by cheating the very
    quantity the sharp board is meant to certify.
    """
    for s in sharp:
        if not s.variance_compressed:
            return s
    return None


def champion_line(champion: Optional[GroupStat]) -> str:
    if champion is None:
        return "[leaderboard] CHAMPION: none -- no group met the sharp-board rules"
    return (
        f"[leaderboard] CHAMPION: {champion.label} "
        f"(W1 {fmt(champion.w1_mean, 4)}, hq {fmt(champion.hq_mean, 3)}, "
        f"recall {fmt(champion.recall_mean, 3)}, "
        f"bar {champion.bar_reached}/{champion.n_seeds} "
        f"@ mean {fmt(champion.bar_steps_mean, 0)} steps)"
    )


def render(stats: Sequence[GroupStat], info: Dict[str, int]) -> Tuple[str, Optional[GroupStat]]:
    logistic = [s for s in stats if s.objective == DEFAULT_OBJECTIVE]
    wgan = [s for s in stats if s.objective != DEFAULT_OBJECTIVE]

    sharp = _rank([s for s in logistic if s.sharp_eligible()])
    transport = _rank([s for s in logistic if s.transport_eligible()])
    champion = pick_champion(sharp)

    lines: List[str] = ["# Regularizer study -- promotion leaderboard\n"]
    lines.append(HEADER_NOTE + "\n")
    lines.append(
        f"{info['runs']} runs / {info['groups']} groups from the main tree"
        + (
            f"; {info['audit_runs']} audit runs annotated "
            f"{info['audit_annotated']} group(s) with `per_mode_std_ratio` "
            "(marked ᴬ)"
            if info["audit_runs"]
            else "; no audit tree found, so `per_mode_std_ratio` is blank "
            "wherever the original summaries predate it"
        )
        + (
            f"; {info['audit_orphans']} audit group(s) had no main-tree "
            "counterpart and were ignored"
            if info["audit_orphans"]
            else ""
        )
        + ".\n"
    )

    lines.append("\n## 1. Sharp board -- the promotion track\n")
    lines.append(
        "All seeds clear the repo bar (100 modes & hq >= 0.9 by step "
        f"{BAR_HORIZON}), mean hq >= {SHARP_HQ_MIN}, zero collapses. "
        "Ranked by mean final exact W1, lowest first.\n"
    )
    if sharp:
        lines.extend(board_rows(sharp, champion))
    else:
        lines.append("_No group met the sharp-board rules._")

    lines.append("\n## 2. Transport board -- coverage of the target\n")
    lines.append(
        f"Mean mode_recall >= {TRANSPORT_RECALL_MIN}, zero collapses. Ranked by "
        "mean final exact W1, lowest first. A group can top this board while "
        "being too blurry for the sharp board; CHAMPION is never awarded here.\n"
    )
    if transport:
        lines.extend(board_rows(transport, None))
    else:
        lines.append("_No group met the transport-board rules._")

    lines.append("\n## 3. Wasserstein runs -- informational, NOT promotable\n")
    lines.append(
        "`loss_type: wasserstein` optimizes a different objective, so these W1 "
        "numbers are not commensurable with the logistic-objective boards above. "
        "They are listed for reference only and are never pooled with, ranked "
        "against, or promoted over the logistic groups.\n"
    )
    if wgan:
        lines.extend(board_rows(_rank(wgan), None))
    else:
        lines.append("_No Wasserstein runs on disk yet._")

    lines.append("\n## Notes\n")
    lines.append(
        f"1. **steps→bar** is the first eval step in `metrics.jsonl` where "
        f"`modes >= {int(BAR_MODES)}` and `hq >= {BAR_HQ}` in the same row. Seeds "
        f"that never get there (or that have no usable timeseries) count as "
        f"{BAR_HORIZON}, so a group with `n reached` below `n` reports a lower "
        "bound on speed. **bar pass** is the fraction of seeds that did reach it.\n"
    )
    lines.append(
        "2. **per-mode σ ratio** is `final.per_mode_std_ratio` -- generated "
        "per-mode spread over the real per-mode spread. Blank means the metric "
        "predates the group's summaries, which is *not* treated as a failure: an "
        "unmeasured group is never flagged. Cells marked ᴬ come from the "
        "deterministic re-run in the audit tree; every other column on that row "
        "still comes from the original run.\n"
    )
    lines.append(
        "3. A row flagged **variance-compressed** (`per_mode_std_ratio` < "
        f"{VARIANCE_COMPRESSED_BELOW}) keeps its rank but cannot hold CHAMPION: "
        "tight, under-dispersed clusters flatter W1 and hq precisely by failing "
        "to reproduce the target's spread, so the title passes to the next "
        "unflagged row.\n"
    )
    lines.append(
        "4. `collapse` is the fraction of seeds with `collapse_events > 0`. Both "
        "promotion boards require it to be exactly 0.\n"
    )

    return "\n".join(lines) + "\n", champion


# ---------------------------------------------------------------------------
#  Entry points
# ---------------------------------------------------------------------------


def default_audit_dir(runs_dir: Path) -> Path:
    """``results/runs`` -> ``results/runs_audit`` (and the same for fixtures)."""
    return runs_dir.parent / f"{runs_dir.name}_audit"


def write_leaderboard(
    runs_dir: Path,
    out_dir: Path,
    audit_dir: Optional[Path] = None,
    quiet: bool = False,
) -> Tuple[Path, Optional[GroupStat]]:
    """Build results/LEADERBOARD.md; returns (path, champion-or-None)."""
    if audit_dir is None:
        audit_dir = default_audit_dir(runs_dir)
    stats, info = collect_stats(runs_dir, audit_dir)
    text, champion = render(stats, info)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "LEADERBOARD.md"
    with open(path, "w") as f:
        f.write(text)
    if not quiet:
        print(champion_line(champion))
    return path, champion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build results/LEADERBOARD.md from the run trees."
    )
    parser.add_argument("--runs_dir", type=str, default="results/runs")
    parser.add_argument(
        "--audit_dir",
        type=str,
        default=None,
        help="Audit run tree (default: <runs_dir>_audit). Absent is fine.",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    audit_dir = Path(args.audit_dir) if args.audit_dir else None
    path, _ = write_leaderboard(runs_dir, Path(args.out_dir), audit_dir)
    print(f"[leaderboard] wrote {path}")


if __name__ == "__main__":
    main()
