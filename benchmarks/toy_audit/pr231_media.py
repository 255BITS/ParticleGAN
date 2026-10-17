"""Render PR231's retained actual training measurements, without new science.

Only archive JSON and final checkpoint dictionaries are read. No model is
constructed, sampled, replayed or scored. Original scientific gates are absent;
execution completeness and software geometry evidence remain separate statuses.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "68cf68e599d4ac8f8608434c417ed021647e2ff4"
SOURCE_REPORT = "docs/routed_generator_damping_diagnostic_20261002.json"
ARCHIVE = Path("/ml2/hypergan/routed-generator-damping-toy-artifacts-20261002")
STEPS = list(range(0, 1201, 100))
PALETTE = ["#b94352", "#137d69", "#00a8c6", "#8064a2", "#b77817"]
LABELS = {"original_native": "original", "shift_zero_native": "whole shift zero",
          "shift_zero_g_bypass": "shift zero + G bypass", "shift_zero_antithetic": "shift zero + antithetic",
          "shift_zero_antithetic_g_bypass": "antithetic + G bypass"}
QUESTIONS = {
    "initial-width16": "Historical outer-identity architecture control; weaker backfilled provenance",
    "native-handoff-width16": "Recipient handoff with large target; ordinary G cuts not reproduced",
    "calibrated-handoff-width16": "Calibrated target; ordinary G cuts not reproduced",
    "faithful-width64": "Capacity/antithetic control; ordinary G cuts not reproduced",
    "faithful-width64-antithetic-bypass": "Matched narrow antithetic rate intervention; no real-fix qualification",
    "spatial-native-width4": "Small BF16 spatial handoff; original finishes better",
    "spatial-native-width16": "Wider spatial counterexample; shift zero finishes better",
    "publication-width4-replay": "Final toy-source reproduction on 70de5a3e; two spatial width4 arms",
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fafafa", "axes.facecolor": "#fafafa"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def committed(relative):
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)


def parse_trace(path, expected_evaluations):
    updates, evaluations, normalized_rows = [], [], []
    with path.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index:
                require(type(row["step"]) is int and row["step"] == index, "missing/duplicate training update")
                ratio, sigma = row["applied_rates"]["generator"]["applied_ratio"], row["output_sigma"]
                require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (ratio, sigma)),
                        "nonfinite applied rate or noise observation")
                updates.append({"step": index, "ratio": ratio, "sigma": sigma})
            require(index or set(row) == {"evaluation"}, "missing actual initial evaluation")
            if "evaluation" in row:
                evaluation = row["evaluation"]
                require(evaluation["step"] == index, "evaluation attached to wrong training state")
                require(all(type(evaluation[k]) in (int, float) and math.isfinite(evaluation[k])
                            and evaluation[k] >= 0 for k in ("live_mse", "served_mse")), "nonfinite clean MSE")
                evaluations.append(evaluation)
            # The only acknowledged publication reporting correction is D's
            # gradient energy, previously read after freezing D. Never remove
            # a scientific field or another gradient measurement.
            normalized = deepcopy(row)
            if "gradient_energy" in normalized:
                normalized["gradient_energy"].pop("critic", None)
            normalized_rows.append(hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest())
    require(len(updates) == 1200 and [e["step"] for e in evaluations] == STEPS, "incomplete training/evaluation stream")
    require(evaluations == expected_evaluations, "trace and retained summary evaluations disagree")
    return updates, evaluations, normalized_rows


def load_case(archive, declaration, *, publication=False):
    case = "publication-width4-replay" if publication else declaration["artifact_id"]
    directory = archive / case
    summary = read(directory / "summary.json")
    protocol_name = "protocol.json" if (directory / "protocol.json").exists() else "source-receipt.json"
    protocol = read(directory / protocol_name)
    require(protocol["external_update_cap"] == 1200, "changed original update cap")
    require(protocol.get("external_seconds_cap", protocol.get("inner_seconds_cap")) == 900, "changed original wall cap")
    sources = declaration["source_sha256"]
    if isinstance(sources, str):
        sources = {"source.py": sources}
    files = {"summary.json": sha(directory / "summary.json"), protocol_name: sha(directory / protocol_name)}
    for name, expected in sources.items():
        require(sha(directory / name) == expected, "archived executed source changed")
        files[name] = expected
    for relative, expected in protocol["package_files"].items():
        relative = relative if relative.startswith("particlegan/") else "particlegan/" + relative
        require(hashlib.sha256(committed(relative)).hexdigest() == expected, "publication/native source mismatch")
    require(len(protocol["package_files"]) == 29, "unexpected native source coverage")
    expected_profiles = declaration["profiles"]
    expected_profiles = list(expected_profiles) if isinstance(expected_profiles, dict) else expected_profiles
    require(set(summary["profiles"]) == set(expected_profiles), "missing or extra executed profile")
    curves, profiles, normalized = {}, {}, {}
    case_recipe = None
    caller_rng_before = torch.get_rng_state().clone()
    for profile in expected_profiles:
        record = summary["profiles"][profile]
        require(record["completed_steps"] == 1200, "incomplete arm")
        metadata_name, state_name, trace_name = profile + "-metadata.json", profile + "-final.pt", profile + ".jsonl"
        metadata = read(directory / metadata_name)
        if case_recipe is None:
            case_recipe = metadata["recipe"]
        require(metadata["recipe"] == case_recipe, "unmatched effective recipes within cohort")
        # Deserialize dictionary owners solely to establish a legitimate final
        # saved state. No policy/model construction or metric computation.
        state = torch.load(directory / state_name, map_location="cpu", weights_only=True)
        require(state["profile"] == profile and json.loads(json.dumps(state["metadata"])) == metadata
                and state["policy"]["completed_steps"] == 1200, "saved owner/recipe/profile mismatch")
        require(json.loads(json.dumps(state["policy"]["recipe"])) == metadata["recipe"], "saved effective recipe mismatch")
        require(metadata["recipe"]["batch_size"] == 16 and metadata["prior_kind"] == "particle_cloud"
                and metadata["prior_sigma"] == 0, "unexpected actual prior/batch law")
        updates, evaluations, normalized[profile] = parse_trace(directory / trace_name, record["evaluations"])
        for key in ("live_mse", "served_mse"):
            require(record[key] == evaluations[-1][key], "final trace/summary measurement mismatch")
            if not publication:
                require(record[key] == declaration["profiles"][profile][key], "committed scientific endpoint mismatch")
        if not publication:
            require(record["generator_cut_events"] == declaration["profiles"][profile]["generator_cut_events"],
                    "committed organic cut count mismatch")
        if "data_sha256" in declaration:
            require(metadata["data_hashes"] == declaration["data_sha256"], "captured data law changed")
        for role, expected in declaration.get("unaffected_initial_owner_sha256", {}).items():
            require(metadata["initial_hashes"][role] == expected, "unaffected initialization binding differs")
        for name in (metadata_name, state_name, trace_name):
            files[name] = sha(directory / name)
        curves[profile] = {"updates": updates, "evaluations": evaluations}
        profiles[profile] = {"completed_steps": 1200, "live_mse": record["live_mse"], "served_mse": record["served_mse"],
                             "generator_cut_events": record["generator_cut_events"],
                             "generator_damped_updates": record["generator_damped_updates"],
                             "actual_metadata_sha256": files[metadata_name],
                             "initial_hashes": metadata["initial_hashes"], "data_hashes": metadata["data_hashes"]}
    require(torch.equal(caller_rng_before, torch.get_rng_state()), "archive observer consumed ambient RNG")
    if publication:
        review_name = "independent-publication-replay.json"
        review = read(directory / review_name)
        require(review == declaration, "committed publication reproduction receipt differs")
        files[review_name] = sha(directory / review_name)
    else:
        for name in ("independent-review.json", "independent-comparison.json", "recovered-source-verification.json"):
            if (directory / name).exists():
                files[name] = sha(directory / name)
                evidence = read(directory / name)
                if "summary_sha256" in evidence:
                    require(evidence["summary_sha256"] == files["summary.json"], "independent summary hash mismatch")
    metadata = {"case_id": case, "question": QUESTIONS[case], "archive": str(directory),
                "execution_status": "COMPLETE", "original_scientific_status": "NO_FROZEN_CONVERGENCE_GATE",
                "source_scope": "final toy source at publication base 70de5a3e" if publication else "exact executed historical snapshot",
                "protocol_timing": "publication reproduction" if publication else declaration["protocol_receipt_timing"],
                "files_sha256": files, "frame_steps": STEPS, "profiles": profiles,
                "effective_recipe": case_recipe,
                "final_checkpoints_deserialized": len(profiles), "model_instances_constructed": 0,
                "metrics_recomputed": 0, "training_updates": 0, "interpolation": False,
                "runtime": {key: summary[key] for key in ("python", "torch", "device")}}
    return curves, metadata, normalized


def plot_case(case, curves, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), dpi=90)
    fig.subplots_adjust(left=.075, right=.975, bottom=.15, top=.76, hspace=.6, wspace=.25)
    mse_values = [e[k] for c in curves.values() for e in c["evaluations"] for k in ("live_mse", "served_mse")]
    min_mse, max_mse = min(mse_values), max(mse_values)
    mse_limits = [min_mse * .8 if min_mse > 0 else 0, max_mse * 1.08]
    max_sigma = max(u["sigma"] for c in curves.values() for u in c["updates"])
    frames = []
    for step in STEPS:
        for ax in axes.flat:
            ax.clear()
        for (profile, curve), color in zip(curves.items(), PALETTE):
            style = "--" if "g_bypass" in profile else "-"
            observed = [e for e in curve["evaluations"] if e["step"] <= step]
            updates = curve["updates"][:step]
            for ax, key in zip(axes.flat[:2], ("live_mse", "served_mse")):
                ax.plot([e["step"] for e in observed], [e[key] for e in observed], color=color,
                        linestyle=style, marker="x" if "g_bypass" in profile else "o",
                        markersize=2.5, linewidth=1.3, label=LABELS[profile])
                if min_mse > 0:
                    ax.set_yscale("log")
                ax.set_ylim(*mse_limits)
            for ax, key in zip(axes.flat[2:], ("ratio", "sigma")):
                ax.plot([u["step"] for u in updates], [u[key] for u in updates], color=color, linestyle=style, linewidth=1.)
        for ax, title in zip(axes.flat, ("Clean live held-out MSE", "Clean public-served held-out MSE",
                                       "Actually applied G LR / initial G LR", "Recorded training output sigma")):
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-15, 1220)
            ax.set_xlabel("actual training update")
            ax.grid(alpha=.2)
        axes[1, 0].set_ylim(0, 1.06)
        axes[1, 1].set_ylim(0, max_sigma * 1.06)
        fig.suptitle(f"PR231 · {case['case_id']} · actual update {step} / 1200", y=.97, fontsize=12)
        legend = fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                            bbox_to_anchor=(.5, .925), ncol=min(4, len(curves)), fontsize=8, frameon=False)
        note = fig.text(.5, .836, case["question"], fontsize=9, ha="center")
        footer = fig.text(.06, .035,
                          "13 actual clean evaluations; rate/noise lines use every recorded update. Final saved owners verified; no reconstructed clouds.\n"
                          "No frozen convergence gate, new training, rescoring, interpolation, or Forge/default qualification.", fontsize=8)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128))
        legend.remove(); note.remove(); footer.remove()
    plt.close(fig)
    path = output / (case["case_id"] + ".gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[220] * 12 + [1800], loop=0, optimize=True)
    frames[-1].convert("RGB").save(path.with_suffix(".png"))
    return {"gif": path.name, "sha256": sha(path), "poster": path.with_suffix(".png").name,
            "poster_sha256": sha(path.with_suffix(".png")), "frames": len(frames), "frame_steps": STEPS,
            "bytes": path.stat().st_size, "fixed_mse_limits": mse_limits, "mse_axis_scale": "log" if min_mse > 0 else "linear",
            "fixed_sigma_limits": [0, max_sigma * 1.06], "interpolation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/toy_audit/pr231_media")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.archive.resolve()), "output must preserve original archive")
    source_bytes = committed(SOURCE_REPORT)
    report = json.loads(source_bytes)
    declarations = [(r, False) for r in report["runs"]] + [(report["publication_replay"], True)]
    records, normalized = [], {}
    if not args.validate_only:
        args.output.mkdir(parents=True, exist_ok=True)
    for declaration, publication in declarations:
        curves, record, signatures = load_case(args.archive, declaration, publication=publication)
        if not args.validate_only:
            record["media"] = plot_case(record, curves, args.output)
        normalized[record["case_id"]] = signatures
        records.append(record)
        print(json.dumps({"case": record["case_id"], "execution": record["execution_status"], "frames": 13,
                          "profiles": len(curves), "gate": record["original_scientific_status"]}), flush=True)
    for profile, signatures in normalized["publication-width4-replay"].items():
        require(signatures == normalized["spatial-native-width4"][profile], "publication scientific trace differs")
    native = normalized["faithful-width64"]["shift_zero_antithetic"]
    bypass = normalized["faithful-width64-antithetic-bypass"]["shift_zero_antithetic_g_bypass"]
    require(native[:505] == bypass[:505] and native[505] != bypass[505], "matched rate intervention prefix differs")
    for record in records:
        for name, expected in record["files_sha256"].items():
            require(sha(Path(record["archive"]) / name) == expected, "renderer altered original evidence")
    if args.validate_only:
        return
    receipt = {"schema": "toy_audit_pr231_supplemental_training_media_v1", "source_commit": SOURCE_COMMIT,
               "committed_report_sha256": hashlib.sha256(source_bytes).hexdigest(), "renderer_sha256": sha(__file__),
               "new_training_updates": 0, "model_instances_constructed": 0, "metrics_recomputed": 0,
               "software_replay_updates": 0, "historical_receipts_unchanged": True,
               "verified_publication_trace_exclusion": ["gradient_energy.critic"],
               "verified_publication_scientific_updates_per_arm": 1200,
               "verified_width64_antithetic_first_rate_divergence": 505, "records": records}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    lines = ["# PR231 supplemental actual training media", "",
             "Every GIF uses all 13 recorded clean held-out evaluations and all 1,200 recorded rate/noise updates for its exact source cohort. "
             "Final checkpoint dictionaries verify completed updates, actual metadata, recipe and prior without constructing a model. "
             "No intermediate sample clouds were captured, so these are measurement animations. No new training, metric computation or replay occurred.", "",
             "The diagnostic asks whether erasing the whole additive FiLM branch removes code sensitivity, and whether that initialization "
             "explains rate cuts or late quality loss. The zero-hidden analytic Jacobian tests answer the geometry question with independent "
             "controls. The full native toys do not reproduce the real repeated ordinary G cuts. Spatial width16 reverses the initialization "
             "ordering, and the matched real GPU G-rate bypass worsens LPIPS by11.47%. Keep those counterexamples. "
             "Raw gradient energy and Adam displacement are separate measurements; neither is a convergence guarantee.", "",
             "No frozen trained-convergence acceptance gate exists for these cohorts. `COMPLETE` means the budget and measurement stream "
             "completed; it does not mean a model PASS. Historical source snapshots, early backfilled protocols, and the separately "
             "verified final toy-source width4 reproduction on70de5a3e remain distinct. The earlier report's no-final-source-rerun statement refers "
             "to historical cohorts; its later publication reproduction does execute the final common source for two arms.", "",
             "| Cohort | Purpose | Actual training media |", "|---|---|---|"]
    for record in records:
        lines.append(f"| {record['case_id']} | {record['question']} | [13 frames]({record['media']['gif']}) |")
    lines += ["", "The [receipt](receipt.json) binds every raw source/protocol/metadata/trace/final-state file hash. "
              "All29 native package files match published PR231. A separate read-only check reproduces every scientific "
              "publication trace row after only the acknowledged corrected D-gradient reporting field is excluded, "
              "and verifies the matched antithetic prefix through504 with first applied-rate divergence505. "
              "Ambient global checkpoint RNG differences remain documented in the original publication receipt. "
              "No full real-image rerun, real fix, robustness, public-default or Forge qualification follows.", "",
              "```sh", "python -m benchmarks.toy_audit.pr231_media --validate-only",
              "python -m benchmarks.toy_audit.pr231_media", "```", "",
              "Original raw archive: `" + str(args.archive) + "`. Source report is pinned to PR231 head `" + SOURCE_COMMIT + "`."]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
