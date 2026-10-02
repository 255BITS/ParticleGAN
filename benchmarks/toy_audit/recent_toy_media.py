"""Strict, media-only reviews of three later public-API toy protocols.

Export reads saved observations, never trains or rescores a model. The optional
explicit --reproduce command invokes each original public caller in a child;
it does not copy a trainer loop and refuses incomplete protocols.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HEADS = {233: "6be1be53f8a8c2ac5d823aa8f63430d9c6a58a84",
         234: "ae7fd0bb2f61967f718c96ba9722cd4a7c9d1be0",
         235: "bcd4b1b201f8f154a00e2b432cf051c0e318813c"}
CARDS = {233: ("routed_film_code_preserved_20261002.json", "63fa23a0533acf88c54f3dced4a997590e025f38728ed47cdaffa03580739aac"),
         234: ("e22_routed_caption_antithetic_failure.json", "7fabe49df73cb5b8fc66b5f6e41888b78e3f5858bbecdbc52bf6c97d4d7879f8"),
         235: ("e22_routed_g_clean_results.json", "11089ea4af388899167740ed2f3ca45e2274c18b1a77352197329b2d2fa001bc")}
PROFILES = ("original_native", "shift_zero_native", "shift_time_zero_native")
JUDGES = ("native@128", "native@512", "G_clean@128", "G_clean@512")
ROLES = ("generator", "router", "table", "residual_upstream")
COMPONENTS = ("total_dv12_gaussian", "clean_fixed_gaussian")
NATIVE_SHA = "7b814e32627ac5a332637d336ebaa13b6baab0b9373f6841d9a22fb8ab39cde9"
COLORS = ("#74869a", "#d36b38", "#b93867")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_card(pr):
    name, expected = CARDS[pr]
    path = ROOT / "docs" / name
    if path.is_file():
        content = path.read_bytes()
    else:
        # Review worktrees may precede the source PR integration. A fresh
        # checkout uses the integrated public card, not a local archive path.
        try:
            content = subprocess.check_output(["git", "show", f"{HEADS[pr]}:docs/{name}"],
                                              cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError("integrate the selected source PR/card before reproduction or export") from error
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("selected public card differs from its reviewed exact head")
    return json.loads(content)


class Inputs:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        item = {"sha256": sha(path), "bytes": path.stat().st_size}
        if expected is not None and item["sha256"] != expected:
            raise ValueError(f"retained artifact identity mismatch: {path.name}")
        self.files[str(path)] = item
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def unchanged(self):
        return all(sha(path) == item["sha256"] and Path(path).stat().st_size == item["bytes"]
                   for path, item in self.files.items())


def finite(values):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError("nonfinite or missing numeric observation")


def film_data(root, card, inputs, fresh=False):
    root = Path(root)
    if not fresh:
        inputs.bind(root / "campaign-protocol.json", card["protocol_sha256"])
        inputs.bind(root / "readout.json", card["readout_sha256"])
    junit = inputs.bind(root / "contracts.xml", None if fresh else card["contracts"]["junit_sha256"])
    tree = ET.parse(junit)
    controls = [t for t in tree.iter("testcase") if t.get("classname", "").endswith("test_routed_conditioning_code_preserved")]
    required_controls = {"test_only_additive_time_columns_change_and_code_path_remains_live[4]",
                         "test_only_additive_time_columns_change_and_code_path_remains_live[16]",
                         "test_code_preserved_checkpoint_replays_exactly_and_refuses_whole_zero_owner"}
    if (len(controls) != 3 or {t.get("name") for t in controls} != required_controls
            or any(t.find(tag) is not None for t in tree.iter("testcase")
                               for tag in ("failure", "error", "skipped"))):
        raise ValueError("complete numerical code-retention/owner/replay controls are required")
    widths = {}
    for width in (4, 16):
        folder = root / f"width{width}"
        protocol = inputs.read(folder / "protocol.json")
        if (protocol["profiles"] != list(PROFILES) or protocol["external_update_cap"] != 1200
                or protocol["width"] != width or protocol["blocks"] != 1):
            raise ValueError("changed FiLM host, profile order or full budget")
        for name, expected in protocol["source_sha256"].items():
            if expected != card["source_sha256"][f"benchmarks/routed_conditioning/{name}"]:
                raise ValueError("changed FiLM source")
            inputs.bind(folder / name, expected)
        summary = inputs.read(folder / "summary.json")
        if summary["protocol"] != protocol or set(summary["profiles"]) != set(PROFILES):
            raise ValueError("incomplete or changed FiLM protocol")
        metadata = {arm: inputs.read(folder / f"{arm}-metadata.json") for arm in PROFILES}
        original, neutral, candidate = (metadata[arm] for arm in PROFILES)
        if (candidate["additive_initialization"] != "zero time columns and bias; retain code columns"
                or not candidate["initial_code_jacobian_frobenius"] > neutral["initial_code_jacobian_frobenius"]
                or any(candidate["initial_hashes"][role] != original["initial_hashes"][role]
                       for role in ("critic", "prior", "encoder", "router", "table"))
                or any(m["data_hashes"] != original["data_hashes"] or m["recipe"] != original["recipe"]
                       for m in metadata.values())):
            raise ValueError("code-retention/non-generator ownership controls failed")
        results = {}
        for arm in PROFILES:
            result = summary["profiles"][arm]
            rows = [json.loads(x) for x in inputs.bind(folder / f"{arm}.jsonl").read_text().splitlines()]
            updates = [row for row in rows if "step" in row]
            evaluations = [row["evaluation"] for row in rows if "evaluation" in row]
            if (result["completed_steps"] != 1200 or [r["step"] for r in updates] != list(range(1, 1201))
                    or [r["step"] for r in evaluations] != list(range(0, 1201, 100))
                    or evaluations != result["evaluations"]
                    or any(r["dense_gradient_rows"] != 128 for r in updates)):
                raise ValueError("incomplete FiLM budget, dense-bank invariant or observation schedule")
            finite([r[k] for r in updates for k in ("loss_d", "loss_g", "penalty")])
            finite([r[k] for r in evaluations for k in ("live_mse", "served_mse")])
            inputs.bind(folder / f"{arm}-final.pt")
            if not fresh and ({str(r["step"]): r["live_mse"] for r in evaluations}
                              != card["widths"][str(width)]["profiles"][arm]["clean_curve"]):
                raise ValueError("FiLM observations differ from the original published curve")
            results[arm] = {key: deepcopy(result[key]) for key in
                           ("live_mse", "served_mse", "served_source", "completed_steps", "evaluations")}
        widths[width] = {"profiles": results, "metadata": metadata}
    return {"native_updates_per_arm": 1200, "arms": 6, "complete": True,
            "structural_full_protocol_passed": True, "learned_scientific_status": "NO_FROZEN_GATE",
            "structural_gate": {"code_retention_source_testcases_passed": 3,
                "non_generator_initial_owner_mismatches": 0, "minimum_dense_bank_rows_each_update": 128,
                "fixed_updates_each": 1200, "scope": "Initial code-column byte equality, owners, Jacobian and complete finite protocol; not a learned-quality gate"},
            "widths": widths}


def variance_data(path, card, inputs, fresh=False):
    result = inputs.read(path, None if fresh else card["portable_fixed_fixture"]["first_receipt_sha256"])
    if (result["native_updates"] != 0 or result["pairs_per_batch"] != 16 or result["batches"] != 2
            or result["sigma"] != .125 or result["fixture"] != "routed-dv12"
            or result["ownership_checks"] != "PASS" or result["state_unchanged_before_restore"] is not True):
        raise ValueError("changed or incomplete zero-update variance protocol")
    stats = result["statistics"]
    for component in COMPONENTS:
        if set(stats[component]) != set(ROLES):
            raise ValueError("missing fixed gradient roles")
        for role in ROLES:
            value = stats[component][role]
            if len(value["batch_statistics"]) != 2:
                raise ValueError("both fixed variance batches are mandatory")
            finite([s[k] for s in [value, *value["batch_statistics"]]
                    for k in ("ratio", "single_variance", "antithetic_variance")])
    primary = stats[COMPONENTS[0]]["generator"]
    passed = primary["single_variance"] > 0 and 0 <= primary["antithetic_variance"] <= .75 * primary["single_variance"]
    if result["pass"] is not passed or result["variance_result"] != ("PASS" if passed else "FAIL"):
        raise ValueError("original variance verdict contradicts the frozen numerical gate")
    return {"complete": True, "native_updates": 0, "scientific_status": result["variance_result"],
            "variance_gate": {"maximum_full_generator_variance_ratio": .75,
                              "observed_ratio": primary["ratio"], "passed": passed},
            "observations": deepcopy(stats), "ownership_checks": "PASS"}


def clean_data(root, card, inputs, fresh=False):
    root = Path(root)
    bindings = card["bindings"]
    report = inputs.read(root / "report.json", None if fresh else bindings["report_sha256"])
    completion = inputs.read(root / "completion.json", None if fresh else bindings["completion_sha256"])
    retained = inputs.bind(root / "retained.pt", report["retained_sha256"])
    if (not report["complete"] or not completion["complete"] or report["fixed_updates_each"] != 512
            or report["quality_updates_total"] != 1024 or report["G_clean_extra_no_grad_forwards"] != 512
            or report["source_sha256"] != bindings["source_sha256"]
            or report["native_python_sha256"] != bindings["native_python_sha256"]
            or completion["report_sha256"] != sha(root / "report.json")
            or completion["scientific_status"] != report["scientific_status"]
            or report["matched_data_paired_and_DV12_schedules"] is not True
            or report["learned_owners_gradients_and_optimizer_moments_finite"] is not True):
        raise ValueError("changed or incomplete clean-G source/protocol/ownership evidence")
    schedules = []
    for arm in ("native", "G_clean"):
        path = inputs.bind(root / f"{arm}.jsonl", None if fresh else bindings[f"{arm}_trace_sha256"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["step"] for r in rows] != list(range(1, 513)):
            raise ValueError("missing clean-G full-budget native updates")
        schedules.append([{k: r[k] for k in ("step", "batch_indices", "paired_base_digest", "data_rng", "paired_rng", "dv12_rng")} for r in rows])
    if schedules[0] != schedules[1]:
        raise ValueError("clean-G/native draw cadence differs")
    for arm in ("native", "G_clean"):
        curves = report["curves"][arm]
        if list(curves) != [str(step) for step in range(0, 513, 64)]:
            raise ValueError("missing or reordered clean-G actual observations")
        for frame in curves.values():
            for cohort in ("clean", "DV12"):
                if set(frame[cohort]) != set(JUDGES):
                    raise ValueError("all four common judges are required at every checkpoint")
                finite(frame[cohort].values())
    gate = report["gate"]
    calibrated = (set(report["references"]) == set(JUDGES) and all(
        len(path) == 4 and all(math.isfinite(x) for x in path)
        and all(b >= a - 1e-5 for a, b in zip(path, path[1:])) and path[-1] > path[0] + 1e-4
        for path in report["references"].values()))
    coverage = report["coverage"]["G_clean"]
    if (gate["calibrated"] is not calibrated or gate["bank_live"] != (coverage["bank"] > 0)
            or gate["query_live"] != (coverage["query"] > 0)
            or gate["C_live"] != all(x > 0 for x in coverage["C_norms"])):
        raise ValueError("clean-G calibration or particle invariant contradicts retained observations")
    final = report["curves"]["G_clean"]["512"]
    native = report["curves"]["native"]["512"]["clean"]
    delta = max(final["clean"][j] - native[j] for j in JUDGES)
    gain = min(final["zero_code"][j] - final["clean"][j] for j in JUDGES)
    passed = delta < -1e-4 and gain > 1e-6 and all(gate[k] for k in ("calibrated", "bank_live", "query_live", "C_live"))
    if (gate["pass"] is not passed or gate["metric_max_clean_minus_native_game"] != delta
            or gate["minimum_code_gain"] != gain or gate["threshold"] != -1e-4
            or gate["code_threshold"] != 1e-6 or report["scientific_status"] != ("PASS" if passed else "FAIL")):
        raise ValueError("clean-G verdict differs from its exact frozen four-judge gate")
    if not fresh and (sha(retained) != bindings["retained_sha256"] or gate != card["gate"]):
        raise ValueError("clean-G retained qualification binding changed")
    return {"complete": True, "native_updates_per_arm": 512, "arms": 2,
            "scientific_status": report["scientific_status"], "gate": deepcopy(gate),
            "curves": deepcopy(report["curves"]), "ordinary_toy_comparison": "UNAVAILABLE",
            "actual_caption_status": "FAIL (separate frozen verification)"}


def render(pr, data, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    artifacts = []
    groups = list(data["widths"]) if pr == 233 else [None]
    for group in groups:
        steps = list(range(0, 1201, 100)) if pr == 233 else ([0, 1, 2] if pr == 234 else list(range(0, 513, 64)))
        images = []
        for index, step in enumerate(steps):
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.9)) if pr == 233 else plt.subplots(2, 2, figsize=(9, 6.8))
            if pr == 233:
                profiles = data["widths"][group]["profiles"]
                y_max = max(r["live_mse"] for p in profiles.values() for r in p["evaluations"]) * 1.08
                for color, arm in zip(COLORS, PROFILES):
                    rows = [r for r in profiles[arm]["evaluations"] if r["step"] <= step]
                    axes[0].plot([r["step"] for r in rows], [r["live_mse"] for r in rows], marker=".", color=color, label=arm.replace("_native", ""))
                axes[0].set(xlim=(0, 1200), ylim=(0, y_max), xlabel="Actual updates per arm", ylabel="Clean heldout MSE", title="All three actual training curves")
                axes[0].legend(fontsize=7)
                jacobians = [data["widths"][group]["metadata"][a]["initial_code_jacobian_frobenius"] for a in PROFILES]
                axes[1].bar(range(3), jacobians, color=COLORS)
                axes[1].set(ylim=(0, max(jacobians) * 1.18), ylabel="Initial code Jacobian norm", title="Frozen initial code sensitivity")
                axes[1].set_xticks(range(3), ["Original", "Whole shift\nzero", "Time only\nzero"])
                change = 100 * (profiles[PROFILES[2]]["live_mse"] / profiles[PROFILES[0]]["live_mse"] - 1)
                footer = f"Actual update {step}/1200 per arm | structural/full-protocol PASS | learned benefit NO_FROZEN_GATE\nFinal time-only change vs original: {change:+.2f}% (positive is worse). Both widths and all controls retained."
                title = f"Preserve additive code while neutralizing time: spatial width {group}"
            elif pr == 234:
                for ax, role in zip(axes.flat, ROLES):
                    values = [data["observations"][c][role] if step == 2 else data["observations"][c][role]["batch_statistics"][step] for c in COMPONENTS]
                    ratios = [r["ratio"] for r in values]
                    bound = max(data["observations"][c][role]["ratio"] for c in COMPONENTS)
                    bound = max(bound, *(r["ratio"] for c in COMPONENTS for r in data["observations"][c][role]["batch_statistics"]))
                    ax.bar((0, 1), ratios, color=(COLORS[2], COLORS[0]))
                    ax.axhline(.75, color="#333333", linestyle="--", linewidth=1)
                    ax.set_xticks((0, 1), ["DV12 + Gaussian", "Fixed clean + Gaussian"], fontsize=7)
                    ax.set(ylim=(0, max(1.1, bound * 1.12)), title=role.replace("_", " "), ylabel="Paired / single gradient variance")
                    for x, value in enumerate(ratios): ax.text(x, value + .02, f"{value:.5f}", ha="center", fontsize=8)
                phase = f"Fixed batch {step + 1} of 2" if step < 2 else "Final ratio of the two mean variances"
                footer = f"ZERO native training updates | {phase} | original variance gate {data['scientific_status']}\nPrimary gate applies to full generator with DV12: ratio <= .75. Other roles and clean-fixed panels are diagnostics."
                title = "Does Gaussian sign pairing remove total routed gradient variation?"
            else:
                all_deltas = [data["curves"]["G_clean"][str(s)][c][j] - data["curves"]["native"][str(s)][c][j]
                              for s in steps for c in ("clean", "DV12") for j in JUDGES]
                low, high = min(all_deltas + [-1e-4]), max(all_deltas + [0.])
                pad = max((high - low) * .1, 2e-5)
                for ax, judge in zip(axes.flat, JUDGES):
                    for color, cohort in zip((COLORS[2], COLORS[0]), ("clean", "DV12")):
                        xs = steps[:index + 1]
                        ys = [data["curves"]["G_clean"][str(s)][cohort][judge] - data["curves"]["native"][str(s)][cohort][judge] for s in xs]
                        ax.plot(xs, ys, marker=".", color=color, label=cohort)
                    ax.axhline(-1e-4, linestyle="--", color="#333333", linewidth=1, label="Terminal clean limit")
                    ax.axhline(0, color="#bbbbbb", linewidth=.6)
                    ax.set(xlim=(0, 512), ylim=(low - pad, high + pad), title=judge, xlabel="Actual native updates per arm", ylabel="Clean-G minus native paired game")
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    ax.legend(fontsize=7)
                footer = f"Actual update {step}/512 per arm | original frozen synthetic gate {data['scientific_status']}\nFour common judges, calibrated references, beneficial code and live bank/query required at terminal512.\nOrdinary toy comparison unavailable; actual caption transfer FAIL is a separate cohort."
                title = "Does cleaning only the differentiable G forward improve the common-critic game?"
            fig.suptitle(title, fontsize=11, y=.98)
            for ax in axes.flat: ax.grid(axis="y", alpha=.17)
            fig.tight_layout(rect=(0, .17 if pr == 233 else .16, 1, .95))
            fig.text(.025, .025, footer, fontsize=8, va="bottom")
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100)
            plt.close(fig)
            buffer.seek(0)
            images.append(Image.open(buffer).convert("RGB"))
        name = f"pr233-width{group}-goal.gif" if pr == 233 else f"pr{pr}-goal.gif"
        path = output / name
        images[0].save(path, save_all=True, append_images=images[1:], duration=450, loop=0, optimize=False)
        with Image.open(path) as gif:
            if gif.n_frames != len(steps): raise ValueError("actual observation frames were lost")
        artifacts.append({"file": name, "sha256": sha(path), "bytes": path.stat().st_size,
                          "frames": len(steps), "actual_updates": steps if pr != 234 else [0, 0, 0],
                          "variance_observation_phases": steps if pr == 234 else None})
    return artifacts


def active_sources(pr, card):
    names = {233: tuple(card["source_sha256"]),
             234: ("examples/e22_routed_antithetic_variance.py",),
             235: ("examples/e22_routed_g_clean.py", "tests/test_e22_routed_g_clean.py")}[pr]
    expected = (card["source_sha256"] if pr == 233 else
                {names[0]: card["provenance_sha256"]["executed_public_probe_source"]} if pr == 234 else
                {names[0]: card["bindings"]["source_sha256"], names[1]: card["bindings"]["test_sha256"]})
    hashes = {name: sha(ROOT / name) for name in names}
    native = hashlib.sha256()
    for path in sorted((ROOT / "particlegan").rglob("*.py")):
        native.update(str(path.relative_to(ROOT)).encode()); native.update(path.read_bytes())
    if hashes != expected or native.hexdigest() != NATIVE_SHA:
        raise ValueError("integrate exact reviewed public callers and compatible native source before reproduction")
    return {"public_callers_sha256": hashes, "native_python_sha256": native.hexdigest()}


def reproduce(pr, path, card):
    """Explicit future reproduction only; the media export never calls this."""
    path = Path(path).resolve()
    if path.exists(): raise ValueError("reproduction inputs must be a new artifact path")
    before = active_sources(pr, card)
    path.mkdir(parents=True)
    commands = []
    if pr == 233:
        commands.append(([sys.executable, "-m", "pytest", "-q", "tests/test_routed_conditioning_code_preserved.py", "tests/test_routed_conditioning_spatial.py", "tests/test_routed_conditioning_damping.py", "tests/test_paired_antithetic_math.py", "--junitxml", str(path / "contracts.xml")], 120))
        for width in (4, 16):
            command = [sys.executable, "-m", "benchmarks.routed_conditioning.spatial_damping", "--steps", "1200", "--width", str(width), "--output", str(path / f"width{width}")]
            for profile in PROFILES: command += ["--profile", profile]
            commands.append((command, 900))
    elif pr == 234:
        commands.append(([sys.executable, "examples/e22_routed_antithetic_variance.py", "--fixture", "routed-dv12", "--bf16", "--out", str(path / "probe.json")], 120))
    else:
        commands.append(([sys.executable, "-m", "examples.e22_routed_g_clean", "--run", "--out", str(path / "quality")], 300))
    receipts = []
    import os
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    for index, (command, cap) in enumerate(commands):
        start = time.monotonic()
        with (path / f"caller-{index}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=cap)
        receipts.append({"command": command, "cap_seconds": cap, "returncode": result.returncode,
                         "elapsed_seconds": time.monotonic() - start})
        if result.returncode not in (0, 1) or pr == 233 and result.returncode != 0:
            raise RuntimeError("public caller failed; no complete qualification/media receipt")
    if active_sources(pr, card) != before: raise ValueError("public sources changed during reproduction")
    write(path / "reproduction.json", {"source": before, "commands": receipts,
                                      "scope": "New exact-protocol cohort; does not replace original evidence"})
    return path / "probe.json" if pr == 234 else path / "quality" if pr == 235 else path


def export(pr, source, output, *, fresh=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("review output must be new and outside the original input archive")
    card, inputs = load_card(pr), Inputs()
    if fresh:
        active_sources(pr, card)
        proof_path = source.parent / "reproduction.json" if pr in (234, 235) else source / "reproduction.json"
        proof = inputs.read(proof_path)
        if proof["source"] != active_sources(pr, card): raise ValueError("missing exact public-caller reproduction binding")
    data = {233: film_data, 234: variance_data, 235: clean_data}[pr](source, card, inputs, fresh)
    if not inputs.unchanged(): raise ValueError("input archive changed during verification")
    output.mkdir(parents=True, exist_ok=False)
    module_hash = sha(__file__)
    numeric_identity = json.dumps(data, sort_keys=True, allow_nan=False)
    artifacts = render(pr, data, output)
    if (not inputs.unchanged() or sha(__file__) != module_hash
            or json.dumps(data, sort_keys=True, allow_nan=False) != numeric_identity):
        raise ValueError("original evidence, metrics or media exporter changed")
    exporter_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True, stderr=subprocess.DEVNULL).strip()
    receipt = {"schema": "particlegan_recent_toy_media_v1", "pr": pr, "reviewed_head": HEADS[pr],
        "cohort": "fresh_exact_protocol" if fresh else "retained_original",
        "quality_tier": "4/5 bounded diagnostic definition", "recommendation": "PRESERVE_AND_ADD",
        "frozen_176_campaign_changed": False, "training_or_model_rescoring_during_export": False,
        "raw_inputs": inputs.files, "raw_files_unchanged": True,
        "exporter_source": {"commit": exporter_commit, "file": "benchmarks/toy_audit/recent_toy_media.py", "sha256": module_hash},
        "data": data, "media": artifacts}
    write(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, choices=(233, 234, 235), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduce", action="store_true", help="Explicitly invoke the selected fixed public caller before exporting a new cohort")
    args = parser.parse_args(argv)
    try:
        source = reproduce(args.pr, args.input, load_card(args.pr)) if args.reproduce else args.input
        receipt = export(args.pr, source, args.output, fresh=args.reproduce)
        print(json.dumps({"pr": args.pr, "media": receipt["media"], "data_status": receipt["data"].get("scientific_status", receipt["data"].get("learned_scientific_status"))}))
        # A completed variance FAIL remains scientific exit1. FiLM's structural
        # variant can pass without supplying a learned-benefit qualification.
        return int(receipt["data"].get("scientific_status") == "FAIL")
    except (ValueError, KeyError, FileNotFoundError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"INCOMPLETE: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
