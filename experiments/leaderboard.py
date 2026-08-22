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

Penalty-curriculum runs (``b2a_c1p0_to0p1_s*``) train under b_cap and then
hard-switch to a_r1r2 at 60% of the run. They are logistic-objective runs, so
they compete on both promotion boards on exactly the same terms as any other
logistic group -- the only difference is that their ``arm`` cell reads ``b2a``,
because two arms trained the model and neither one alone names it.

The variance-compression guard exists because the sharp board's own metrics
can be gamed: a generator that emits tight, under-dispersed blobs at every
mode centre scores excellent W1 and hq while not actually matching the target's
per-mode spread. A spread ratio (generated / real) below 0.8 flags exactly that
failure, and a flagged row is barred from CHAMPION even if it ranks first.

**Which spread ratio.** Not ``per_mode_std_ratio``: results/metric_recon.md
showed that number is a second moment about a mean the far tail has itself
dragged, so a few per cent of the mass stranded whole grid cells away inflates
it several-fold -- ``c_eikonal_c0p1`` reports 4.80 while its cores sit at 0.55
of the true width, i.e. exactly the compression this guard exists to catch,
invisible to it. The guard therefore reads ``per_mode_core_ratio``
(lib.toy_metrics), the median-centered core width, recomputed here from each
seed's saved ``final_samples.npy``. The raw ratio keeps its column for
context, and the far-tail mass that separates the two is reported next to it.
Groups with no saved samples in either tree are unmeasured and so unflagged --
absence of evidence, not evidence of absence -- and the board says so.

Usage:

    python experiments/leaderboard.py
    python experiments/leaderboard.py --runs_dir results/runs \
        --audit_dir results/runs_audit --out_dir results
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.toy_metrics import TAIL_SIGMAS, per_mode_core_ratio  # noqa: E402

try:  # pragma: no cover - import-path shim for direct execution
    from analyze import (
        CURRICULUM_STEM,
        CURRICULUM_TITLE,
        DEFAULT_OBJECTIVE,
        Run,
        agg,
        fmt,
        group_runs,
        load_runs,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze import (
        CURRICULUM_STEM,
        CURRICULUM_TITLE,
        DEFAULT_OBJECTIVE,
        Run,
        agg,
        fmt,
        group_runs,
        load_runs,
    )

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
    "zero collapses. If the **core** σ ratio < 0.8, the row is flagged "
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
    # Core-width columns, recomputed from the seeds' saved sample clouds. Left
    # at "unmeasured" for groups with no final_samples.npy in either tree.
    core_mean: float = float("nan")
    core_std: float = float("nan")
    core_n: int = 0
    core_source: str = ""  # "" | "runs" | "audit" | "runs+audit"
    tail_mean: float = float("nan")

    @property
    def coeff(self) -> str:
        return self.coeff_label.replace("p", ".").replace("m", "-")

    @property
    def variance_compressed(self) -> bool:
        """
        True only when the core width exists *and* falls below the guard.

        Read from `core_mean`, never from `pmsr_mean`: the raw second moment is
        tail-inflated (see the module docstring), so a compressed core routinely
        hides behind a raw ratio of 4-5.
        """
        return bool(
            self.core_n
            and math.isfinite(self.core_mean)
            and self.core_mean < VARIANCE_COMPRESSED_BELOW
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


def samples_path(
    name: str, runs_dir: Path, audit_dir: Optional[Path]
) -> Tuple[Optional[Path], str]:
    """
    Where this run's saved 100k sample cloud lives: its own directory first,
    the audit re-run second (training is deterministic, so the audit tree's
    cloud is the same model's), or nowhere.
    """
    own = runs_dir / name / "final_samples.npy"
    if own.is_file():
        return own, "runs"
    if audit_dir is not None:
        alt = audit_dir / name / "final_samples.npy"
        if alt.is_file():
            return alt, "audit"
    return None, ""


def annotate_core(
    stat: GroupStat, runs: Sequence[Run], runs_dir: Path, audit_dir: Optional[Path]
) -> int:
    """
    Fill a group's core-width and far-tail columns from its seeds' sample
    clouds. Returns how many seeds contributed (0 leaves the columns blank).

    This is the only column recomputed from raw samples rather than read out of
    ``summary.json``: the core estimator postdates most of these runs, and it is
    the quantity the promotion guard depends on, so re-deriving it is cheaper
    and far less error-prone than retraining.
    """
    ratios: List[Optional[float]] = []
    tails: List[Optional[float]] = []
    sources: List[str] = []
    for run in runs:
        path, source = samples_path(run.name, runs_dir, audit_dir)
        if path is None:
            continue
        core = per_mode_core_ratio(np.load(path))
        ratios.append(core["per_mode_core_ratio"])
        tails.append(core["tail_frac_10sigma"])
        sources.append(source)
    if not ratios:
        return 0

    stat.core_mean, stat.core_std, stat.core_n = agg(ratios)
    stat.tail_mean, _, _ = agg(tails)
    stat.core_source = "+".join(sorted(set(sources)))
    return len(ratios)


def collect_stats(
    runs_dir: Path, audit_dir: Optional[Path]
) -> Tuple[List[GroupStat], Dict[str, int]]:
    """
    Build the group table from the main tree, annotate it from the audit tree
    with ``per_mode_std_ratio``, and recompute the core width from whichever
    tree holds each seed's sample cloud.
    """
    runs, skips = load_runs(runs_dir)
    grouped = group_runs(runs)
    stats = {key: reduce_group(rs) for key, rs in grouped.items()}

    info = {
        "runs": len(runs),
        "groups": len(stats),
        "audit_runs": 0,
        "audit_annotated": 0,
        "audit_orphans": 0,
        "core_seeds": 0,
        "core_groups": 0,
    }
    info.update({f"skip_{k}": v for k, v in skips.items()})

    for key, rs in grouped.items():
        n = annotate_core(stats[key], rs, runs_dir, audit_dir)
        info["core_seeds"] += n
        info["core_groups"] += 1 if n else 0

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
    "per-mode σ ratio (raw) | **core σ ratio** | tail > 10σ | collapse | "
    "steps→bar (n reached) | bar pass | notes |"
)
BOARD_SEP = "|" + "---|" * 15


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


def _core_cell(s: GroupStat) -> str:
    """The guarded column: blank when no seed of the group has a sample cloud."""
    if not s.core_n:
        return ""
    cell = f"**{fmt(s.core_mean, 3)}**"
    if math.isfinite(s.core_std):
        cell += f" ± {fmt(s.core_std, 3)}"
    if s.core_n != s.n_seeds:
        cell += f" ({s.core_n}/{s.n_seeds})"
    if "audit" in s.core_source:
        cell += " ᴬ"
    return cell


def _tail_cell(s: GroupStat) -> str:
    if not s.core_n or not math.isfinite(s.tail_mean):
        return ""
    return fmt(s.tail_mean, 4)


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
                    _core_cell(s),
                    _tail_cell(s),
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
        f"core σ ratio {fmt(champion.core_mean, 3) if champion.core_n else 'n/a'}, "
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
        + (
            f"; the core σ ratio was recomputed from {info['core_seeds']} saved "
            f"sample cloud(s) covering {info['core_groups']}/{info['groups']} "
            "group(s)"
            if info.get("core_seeds")
            else "; no saved sample clouds were found, so the core σ ratio is "
            "blank everywhere and nothing is flagged"
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
        "2. **per-mode σ ratio (raw)** is `final.per_mode_std_ratio` -- the "
        "second moment about each mode's sample mean, over the real per-mode "
        "spread. It is shown for context only and **nothing is judged on it**: "
        "`results/metric_recon.md` showed it is inflated several-fold by a far "
        "tail that also drags the mean it is taken about. Blank means the metric "
        "predates the group's summaries. Cells marked ᴬ come from the "
        "deterministic re-run in the audit tree; every other column on that row "
        "still comes from the original run.\n"
    )
    lines.append(
        "3. **core σ ratio** is `lib.toy_metrics.per_mode_core_ratio`: the "
        "median radial distance about each mode's coordinate-wise *median*, "
        "inverted through `median = σ·sqrt(2 ln 2)` and divided by the data's "
        "0.03. It ignores everything past the median, so no tail can reach it, "
        "and it is what `hq` actually reflects. It is recomputed here from each "
        "seed's saved `final_samples.npy` (the run's own directory first, then "
        "the audit tree), so it is blank for groups that predate the trainer "
        "saving that file -- an unmeasured group is never flagged. "
        f"**tail > 10σ** is the share of samples further than {TAIL_SIGMAS:.0f}× "
        "the true σ (0.30) from their nearest mode centre: the mass that buys "
        "the gap between the two ratio columns.\n"
    )
    lines.append(
        "4. A row flagged **variance-compressed** (core σ ratio < "
        f"{VARIANCE_COMPRESSED_BELOW}) keeps its rank but cannot hold CHAMPION: "
        "tight, under-dispersed clusters flatter W1 and hq precisely by failing "
        "to reproduce the target's spread, so the title passes to the next "
        "unflagged row.\n"
    )
    lines.append(
        "5. `collapse` is the fraction of seeds with `collapse_events > 0`. Both "
        "promotion boards require it to be exactly 0.\n"
    )
    curriculum = [s for s in stats if s.arm == CURRICULUM_STEM]
    if curriculum:
        lines.append(
            f"6. Rows whose arm reads `{CURRICULUM_STEM}` are the **penalty "
            f"curriculum** ({CURRICULUM_TITLE}): they train under b_cap at the "
            "coeff shown and hard-switch to a_r1r2 at 60% of the run, so no "
            "single arm names them. They optimize the same logistic objective "
            "as every other row here and are eligible for both boards, and for "
            f"CHAMPION, on identical terms ({len(curriculum)} such group(s) on "
            "disk).\n"
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
