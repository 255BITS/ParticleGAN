"""Read-only, source-bound score sections for the canonical family leaderboard.

These original Atlas and CPU research cohorts never enter ordinary selection.
Publication regeneration reads committed receipts only: no archive hydration,
model imports, sampling, grading, or training is performed here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


ATLAS = "reports/forge/atlas-compatible-coverage-20261013"
CATALOG = "reports/forge/research-catalog"
ATLAS_COMMIT = "3f970e106f66051c0054719b12c068681505da1a"
CATALOG_COMMIT = "c732db2c9353884764b336baa5fed525d2ed33f7"
SCORE_PINS = {
    f"{ATLAS}/results.json": "df1344be5fbb8631918c3ed2d11999fbef8508b1d57e88b561eb91c3785697ea",
    f"{ATLAS}/historical/evidence.json": "b885fc488d30334e647b366e7b11b245516ace94b94d244d68ccb4818c7d5f62",
    f"{ATLAS}/historical/native-real-cap/INVENTORY-RECEIPT-PACKET.json": "f14514230e11e79482dbc8fe2b3f75a29ce3b11ef2c2986ef10c0e82935b7e5e",
    f"{CATALOG}/coverage.json": "87f302153b052a784b17176f9fa847a058f7b578f1e0ccd0a47896a0d3eaaaf3",
    f"{CATALOG}/attempts.json": "17f5334c19959d64e382e95a61c8ad23c461f9992b5c39f8511dac87f0105317",
    f"{CATALOG}/studies.json": "2c8b40989eef7e4b8ca814440084152c2a9c2350f4c534a26cf5d50033a8cd53",
    f"{CATALOG}/archive-pins.json": "dab28c77f62304c20ec297fc5a83e3634c103d8c05555acb118302e84a6aab10",
    f"{CATALOG}/leaderboard-scores.json": "205fea128e07a96f12a98e450d877d68e114ee2f5682d2eceeff12b7309ed8c9",
}


def _read(root, relative):
    path = Path(root) / relative
    if not path.is_file():
        raise ValueError(f"missing preserved score evidence: {relative}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCORE_PINS[relative]:
        raise ValueError(f"preserved score evidence changed: {relative}")
    return json.loads(raw)


def _source(label, relative, commit):
    return f"[{label}](https://github.com/255BITS/ParticleGAN/blob/{commit}/{relative})"


def _number(value):
    return f"{value:.6f}"


def atlas_score_lines(root):
    """Return original Atlas scores, or nothing before PR404 is present."""
    if not (Path(root) / ATLAS / "results.json").is_file():
        return []
    result = _read(root, f"{ATLAS}/results.json")
    history = _read(root, f"{ATLAS}/historical/evidence.json")
    native = _read(root, f"{ATLAS}/historical/native-real-cap/INVENTORY-RECEIPT-PACKET.json")
    counts = Counter(row["status"] for row in result["rows"])
    counts["BLOCKED"] = len(result["blocked_unstarted"])
    if (counts != {"FAIL": 1, "INCOMPLETE": 2, "BLOCKED": 4}
            or result["parent_qualification_credit"] or result["global_winner"]
            or history["qualification_input"] or native["qualification"]):
        raise ValueError("Atlas preserved scores have conflicting scope")
    lines = [
        "## Preserved Atlas scores — separate original cohorts", "",
        "PR404's selected-policy particle-cloud scope has **0 PASS / 1 FAIL / "
        "2 INCOMPLETE / 4 BLOCKED across 7 questions**. The three started runs "
        "and four unstarted questions remain distinct. These scores add no "
        "ordinary MoG/live qualification or default-adoption credit. "
        + _source("Original receipts", f"{ATLAS}/results.json", ATLAS_COMMIT) + ".", "",
        "| Started question | Original gate | Completed / declared updates | Original measured score |",
        "|---|---|---:|---|",
    ]
    for row in result["rows"]:
        metrics = row["final_logged_metrics"]
        if row["task"].startswith("gaussian"):
            score = f"Logged CDF KS {_number(metrics['cdf_ks'])}; diagnostic only, checkpoint export ERROR"
        elif row["task"].startswith("ring16"):
            score = (f"Logged component covariance error {_number(metrics['component_covariance_error'])}; "
                     "diagnostic only, checkpoint export ERROR")
        else:
            matched = sum(c["changed_sha256"] == c["reference_sha256"] for c in row["clock_comparisons"])
            score = f"{matched}/{len(row['clock_comparisons'])} state-digest comparisons match; step-label and restart differ"
        lines.append(f"| `{row['task']}` | {row['status']} | {row['completed_steps']}/{row['declared_updates']} | {score} |")
    lines += ["", "Unstarted BLOCKED questions: two-pole, unused-token hold, AE hold, and five-word joint acquisition.", ""]
    panel = next(c for c in history["cohorts"] if c["id"] == "adapted_ra14_fast_panel_20261010_11")
    lines += [
        "The historical adapted RA14 panel reported **7 PASS / 4 FAIL across 11 cases**. "
        "These are its original reported verdicts: Blobs stopped at 300/600 updates "
        "and remains incomplete under its full horizon. Width/anisotropic transfers "
        "are counted once. This different source/prior/serving cohort cannot fill current cells. "
        + _source("Historical report", f"{ATLAS}/historical/FAST_PANEL_REPORT.md", ATLAS_COMMIT) + ".", "",
        "| Historical case | Reported verdict | Passing checks / observations | Actual / original updates |",
        "|---|---|---:|---:|",
    ]
    for case in panel["cases"]:
        verdict = case["reported_primary_status"] + ("; full horizon incomplete" if not case["full_training_horizon_complete"] else "")
        lines.append(f"| `{case['task']}` | {verdict} | {case['passing_checks']}/{case['observations']} | {case['actual_updates']}/{case['original_updates']} |")
    lines += [
        "", "The separate native seed-0 two-pole pilot retains a local RMS-cap win "
        "at its original **80-update** horizon; call-800 behavior and transfer were unmeasured. "
        + _source("Pilot receipts", f"{ATLAS}/historical/native-real-cap/INVENTORY-RECEIPT-PACKET.json", ATLAS_COMMIT) + ".", "",
        "| Native pilot arm | Original gate | Passing checks / observations | Final mean absolute position (≥0.3) | Final median critic gradient (≤1) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in native["new_rows"]:
        convergence = row["evaluator_result"]["convergence"]
        lines.append(f"| `{row['arm']}` | {row['status']} | {convergence['passing_observations']}/{convergence['observations']} | {_number(row['metrics']['mean_abs'])} | {_number(row['metrics']['grad_med'])} |")
    return lines + [""]


def catalog_score_lines(root, page):
    """Return preserved scored comparisons; catalog record counts are not scores."""
    if not (Path(root) / CATALOG / "coverage.json").is_file():
        return []
    coverage = _read(root, f"{CATALOG}/coverage.json")
    attempts = _read(root, f"{CATALOG}/attempts.json")
    scores = _read(root, f"{CATALOG}/leaderboard-scores.json")
    studies = {s["study_id"]: s for s in _read(root, f"{CATALOG}/studies.json")}
    pins = _read(root, f"{CATALOG}/archive-pins.json")
    if (coverage["ordinary_qualification"] or scores["qualification_input"]
            or scores["new_training_attempts"] != 0
            or len({a["attempt_id"] for a in attempts}) != coverage["verified_recent_attempts"]
            or Counter(a["status"] for a in attempts) != coverage["status_counts"]):
        raise ValueError("catalog score scope or canonical attempt counts changed")
    # This new compact receipt preserves original data missing from the old
    # catalog (passing observations/suffix); do not fabricate absent values.
    import os
    receipt_link = os.path.relpath(Path(root) / CATALOG / "leaderboard-scores.json", Path(page).parent)
    lines = [
        "## Preserved catalog scores — original CPU geometry comparisons", "",
        "PR406's catalog contains **285 distinct attempt IDs: 142 PASS / 136 FAIL / "
        "7 INCOMPLETE**. Those are record counts across different protocols, not "
        "a pooled performance score. The numerical comparisons below preserve "
        "two original CPU 2.13 studies on source `d378734f40b09ce223a389e8f54a9783ec6a0c75`; "
        "each seed, task and recipe remains separate. No new training or current-source "
        "qualification is claimed. "
        + _source("Original catalog", f"{CATALOG}/studies.json", CATALOG_COMMIT)
        + f"; [byte-pinned score receipts]({receipt_link}).", "",
        "| Recipe change / task / seed | Original gate, control → candidate | Passing checks / 24, control → candidate | Terminal passing suffix (required 5) | Original endpoint scores, control → candidate |",
        "|---|---|---|---|---|",
    ]
    for row in scores["comparisons"]:
        study = studies[row["study_id"]]
        if row["source_commit"] != study["source_commit"]:
            raise ValueError("catalog score source differs from original study")
        control, candidate = row["baseline"], row["candidate"]
        for arm in (control, candidate):
            if (pins[arm["artifact"]]["sha256"] != arm["sha256"]
                    or not arm["complete"] or arm["completed_steps"] != arm["horizon"]):
                raise ValueError("catalog score lacks a complete original pinned receipt")
        if row["task"] == "trajectory":
            keys = [("identity MSE", "identity_mse")]
        elif row["task"] == "vector_two_broad":
            keys = [("component covariance error", "component_covariance_error")]
        else:
            keys = [("covariance error", "covariance_error"), ("mean error", "mean_error"), ("SW1", "sw1_normalized")]
        measured = "; ".join(f"{label} {_number(control['metrics'][key])} → {_number(candidate['metrics'][key])}" for label, key in keys)
        label = "Full G Frobenius" if "e50af8b0" in row["study_id"] else "G half polar / half Frobenius"
        lines.append(f"| {label}; `{row['task']}`; seed {row['seed']} | {control['status']} → {candidate['status']} | {control['passing_observations']} → {candidate['passing_observations']} | {control['passing_suffix']} → {candidate['passing_suffix']} | {measured} |")
    lines += [
        "", "Overlap and broad use 1,200 updates; trajectory uses 400. The seed-0 "
        "full-Frobenius overlap gains a sustained PASS while mean/SW1 worsen; "
        "the half-sum repairs trajectory while worsening overlap endpoint errors. "
        "Full Frobenius alone fails the conditional trajectory control, and broader "
        "preservation and CUDA equivalence remain unmeasured. The reused broad "
        "control is not a fresh attempt. These five comparison rows and their "
        "control/candidate arms must not be added to the 285-ID register.", "",
    ]
    return lines


def atlas_catalog_score_lines(root, page):
    """Append to the single canonical leaderboard without changing its grades."""
    return atlas_score_lines(root) + catalog_score_lines(root, page)
