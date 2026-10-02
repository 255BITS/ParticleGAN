"""Compact receipts and real-checkpoint GIFs for conditional source coverage."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .source_conditional_capture import CASES, REVISION, load_module, terminal
from .source_demos import digest, state_hash, write


def read(path):
    return json.loads(Path(path).read_text())


def observations(path):
    stream = path / "full/raw/observations.jsonl"
    return [json.loads(line) for line in stream.read_text().splitlines()] if stream.exists() else []


def error_receipt(error, path):
    """Keep exact brief errors in Git and full tracebacks in the raw archive."""
    if not error:
        return None, None
    message = error.rstrip().splitlines()[-1]
    frames = re.findall(r'File "([^"]+/source/[^\"]+)", line (\d+), in ([^\n]+)', error)
    frame = dict(path=frames[-1][0], line=int(frames[-1][1]), function=frames[-1][2]) if frames else None
    return message, frame


def frame(shot, history, raw, label, outcome):
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    requested, reference = raw["requested"], raw["reference_x"]
    target = np.stack([np.abs(reference[requested == c]).mean(0) for c in range(8)], axis=1)
    limit = max(1., target.max())
    for index, cohort in enumerate(("live", "ema")):
        samples = raw[cohort + "_x"]
        values = np.stack([np.abs(samples[requested == c]).mean(0) for c in range(8)], axis=1)
        heatmap = axes[0, index].imshow(values, aspect="auto", vmin=0, vmax=limit, cmap="magma")
        axes[0, index].set(title=f"{cohort}: mean |x| by requested class", xlabel="class", ylabel="coordinate")
        figure.colorbar(heatmap, ax=axes[0, index], shrink=.8)
        confusion = axes[1, index].imshow(shot[cohort]["confusion"], aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axes[1, index].set(title=f"{cohort}: p(symbol | class)", xlabel="emitted symbol", ylabel="requested class")
        figure.colorbar(confusion, ax=axes[1, index], shrink=.8)
    target_plot = axes[0, 2].imshow(target, aspect="auto", vmin=0, vmax=limit, cmap="magma")
    axes[0, 2].set(title="Reference: exact sparse target", xlabel="class", ylabel="coordinate")
    figure.colorbar(target_plot, ax=axes[0, 2], shrink=.8)
    curve = axes[1, 2]
    for cohort, color in (("live", "#ba1658"), ("ema", "#087e9b")):
        curve.plot([row["step"] for row in history], [row[cohort]["max_conditional_quality_mass_tv"] for row in history],
                   color=color, label=cohort + " worst class TV")
        curve.plot([row["step"] for row in history], [row[cohort]["exact_zero_frac"] for row in history],
                   color=color, linestyle="--", label=cohort + " exact inactive zeros")
    curve.axhline(.10, color="black", linestyle=":", linewidth=1)
    curve.set(xlim=(0, 5000), ylim=(-.03, 1.03), xlabel="completed update pairs", title="Joint mass and exact zeros")
    curve.legend(fontsize=7)
    numeric = " / ".join(f"{cohort}: HQ {shot[cohort]['hq']:.3f}, TV {shot[cohort]['max_conditional_quality_mass_tv']:.3f}, "
                         f"zeros {shot[cohort]['exact_zero_frac']:.3f}" for cohort in ("live", "ema"))
    figure.suptitle(f"{label} · observed update {shot['step']}/5,000\nRun outcome: {outcome}\n{numeric}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .88))
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=85, facecolor="white")
    plt.close(figure)
    return Image.open(stream).convert("RGB")


def media(path, output, identifier, label, outcome, rows):
    frames = []
    for index, shot in enumerate(rows):
        cloud = path / f"full/raw/cloud-{index:03d}.npz"
        if digest(cloud) != shot["cloud_sha256"]:
            raise AssertionError("Source observation cloud identity changed")
        with np.load(cloud, allow_pickle=False) as arrays:
            frames.append(frame(shot, rows[:index+1], arrays, label, outcome))
    if not frames:
        return None
    destination = output / "media"
    destination.mkdir(parents=True, exist_ok=True)
    gif, poster = destination / f"{identifier}.gif", destination / f"{identifier}.png"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[180]*(len(frames)-1)+[1500], loop=0)
    frames[-1].save(poster)
    return dict(path="media/" + gif.name, poster="media/" + poster.name,
                gif_sha256=digest(gif), poster_sha256=digest(poster), frames=len(frames),
                actual_checkpoint_steps=[shot["step"] for shot in rows], interpolation=False)


def summary(path):
    if (path / "execution-summary.json").exists():
        return read(path / "execution-summary.json")
    return dict(status="TIMEOUT" if (path / "hard-timeout.json").exists() else "INTERRUPTED",
                error="No durable terminal worker receipt; inspect retained launcher/log/raw evidence",
                measurement_complete=False, completed_update_pairs=read(path / "full/raw/progress.json").get("completed_update_pairs", 0)
                if (path / "full/raw/progress.json").exists() else 0)


def markdown(records, controls):
    lines = ["# Conditional source-family coverage", "",
             "Four formerly source-only entries now have one bounded exact-source attempt each. Scientific "
             "sources/configurations come from develop `" + REVISION + "`; CPU uses one thread and each entry "
             "has a hard 120-second cap including its software parity probes. Original catalog source-only "
             "receipts and quality ratings remain unchanged. No public library, recipe/configuration, batch, "
             "seed, initializer or original update budget was repaired or reduced.", "",
             "| Entry | Original budget | Execution / completed pairs | Original source gate | Added live / EMA | Training GIF |",
             "| --- | --- | --- | --- | --- | --- |"]
    for record in records:
        protocol = record["protocol"]
        gif = f"[actual checkpoints]({record['media']['path']})" if record["media"] else "none: source prerequisite blocks training"
        pair = record["added_gate_status"]
        lines.append(f"| {record['name']} | {protocol['original_update_budget']:,} | "
                     f"{record['fresh_execution_status']} / ≥{record['completed_update_pairs']:,} | "
                     f"{record['original_scientific_status']} | {pair['live']} / {pair['ema']} | {gif} |")
    lines.extend(["", "## What these definitions verify", "",
                  "Sparse identity asks for eight class-conditional distributions over 64 real modes in 24 "
                  "coordinates, three active coordinates per mode with Gaussian noise, exact inactive zeros, "
                  "and the class's deterministic symbol. Split replaces that symbol law with a balanced "
                  "two-symbol law in each class. Continuous mode and symbol must describe the same draw. "
                  "The source's convergence bar checks coverage/HQ/class/symbol and near-zero sparsity; it "
                  "does not require exact zeros, active Gaussian shape or the full split-symbol mass. "
                  "The new separately frozen gate adds those checks, including a reject bin in each class's "
                  "joint quality mass and fixed radial/projected CDF checks on active standardized residuals. "
                  "The residual-shape checks pool assigned modes; they do not independently certify every "
                  "mode's covariance.", "",
                  "Denoising one/four classes asks for repeated clean draws from the analytic multimodal "
                  "`q(x0 | xt,c)` at fixed observations and diffusion times. Four classes select a checkerboard "
                  "subset of 25 modes. Recovering just a posterior mean, ignoring the noisy observation, or "
                  "sampling a different class is insufficient. The original source trains Gaussian reverse "
                  "transitions and reports posterior sliced-W1 diagnostics, but declares no single scientific "
                  "acceptance gate. Analytic controls support the added definition; they are not trained "
                  "model evidence or a substitute for the unavailable GPU execution.", "",
                  f"The versioned evaluator has {len(controls['controls'])} exact-law/control results: exact "
                  "oracle samples pass, while tiny inactive smear, zero-width centres, wrong requested class, "
                  "single-symbol collapse, incoherent balanced symbols, posterior-mean collapse and ignored "
                  "observations are rejected where applicable. Tiny smear and zero-width centres both "
                  "pass the original sparse bar on the same cloud, demonstrating its specific blind spots.", "",
                  "## Source-bound outcomes", ""])
    for record in records:
        lines.extend(["### " + record["name"], "",
                      f"Original source: `{record['protocol']['script']}`; config: "
                      f"`{record['protocol']['config'] or 'source DEFAULTS'}`. "
                      f"Seed {record['protocol']['resolved_config']['seed']}, "
                      f"batch {record['protocol']['resolved_config']['batch_size']}, "
                      f"full budget {record['protocol']['original_update_budget']:,}.", ""])
        if record["error"]:
            lines.extend(["Preserved terminal error: `" + record["error"] + "`.", ""])
            source_frame = record["last_source_error_frame"]
            if source_frame:
                lines.extend([f"Last source frame: `{source_frame['path']}:{source_frame['line']}` "
                              f"in `{source_frame['function']}`.", ""])
            log = record["external_full_log"]
            lines.extend([f"Full external log: `{log['path']}` (SHA-256 `{log['sha256']}`).", ""])
        if record["last_observation"]:
            shot = record["last_observation"]
            for cohort in ("live", "ema"):
                metrics = shot[cohort]
                lines.append(f"Last scored update {shot['step']} {cohort}: {metrics['modes']}/64 HQ modes, "
                             f"HQ {metrics['hq']:.4f}, worst conditional joint TV {metrics['max_conditional_quality_mass_tv']:.4f}, "
                             f"exact inactive zeros {metrics['exact_zero_frac']:.4f}, maximum symbol TV {metrics['max_symbol_tv']:.4f}. "
                             "This partial checkpoint cannot certify the original full-budget gate.")
            lines.append("")
        if record["effective_training"]:
            lines.extend(["The effective recipe, actual prior type/buffers, original initializer and optimizer "
                          "group overrides are recorded alongside source/config hashes. Live and EMA reads "
                          "use the original `sample_z`/`draw_fakes` closures with separate evaluation RNGs. "
                          "Exact short-prefix parity compares model/optimizer/gradient/input tensors, module "
                          "modes, requires-grad flags and owned/global RNG states before the full attempt.", ""])
    lines.extend(["The identity prefix has no HQ modes and severe joint/support defects at its last frame. "
                  "The split prefix has much stronger class/symbol correspondence but lacks HQ coverage, "
                  "leaks onto inactive coordinates and has overly broad active residuals. Both use the "
                  "source's linear real head, which provides no structural exact-zero mask. These are "
                  "measured prefix defects and a source-level limitation; neither capped prefix establishes "
                  "the unknown 5,000-update scientific outcome or isolates an optimizer cause.", ""])
    lines.extend(["## Reproduction and receipts", "",
                  "`coverage.json` retains all four entries, terminal statuses, protocol/effective settings, "
                  "source hashes, actual frame steps, known costs and original errors. Bulk source snapshots, "
                  "checkpoints and tail-able logs stay outside Git under "
                  "`/ml2/hypergan/toy-conditional-sources-20261001`. No retry or failed-prerequisite extension "
                  "was performed. A complete original budget and five passing terminal observations are "
                  "required for an added-gate PASS; blocked or partial runs retain their required denominator.", "",
                  "```sh", "python -m benchmarks.toy_audit.source_conditional_capture \\",
                  "  --case source-family-00 --output /ml2/hypergan/new-conditional-case", "",
                  "python -m benchmarks.toy_audit.source_conditional_report \\",
                  "  --artifacts /ml2/hypergan/toy-conditional-sources-20261001 \\",
                  "  --output reports/toy_audit/conditional_sources", "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for identifier, entry in CASES.items():
        path = args.artifacts / identifier
        protocol, receipt, result = read(path / "protocol.json"), read(path / "source-receipt.json"), summary(path)
        rows = observations(path)
        complete = result["measurement_complete"]
        blocked = result["status"] == "BLOCKED"
        gate = {cohort: "BLOCKED" if blocked else "INCOMPLETE" if not complete else
                "PASS" if terminal(rows, cohort)["passed"] else "FAIL" for cohort in ("live", "ema")}
        original = ("NO_FROZEN_GATE" if entry["family"] == "denoising" else "BLOCKED" if blocked else
                    "INCOMPLETE" if not complete else "PASS" if result["original_summary"]["final"]["bar_all"] else "FAIL")
        outcome = result["status"] + " / live " + gate["live"] + " / EMA " + gate["ema"]
        image = None if blocked else media(path, args.output, identifier, entry["label"], outcome, rows)
        effective_path, parity_path = path / "full/raw/effective.json", path / "parity/parity.json"
        launcher = read(path / "launcher.json")
        dependencies = {name: receipt["source_sha256"][name] for name in
                        (protocol["script"], protocol["config"], "experiments/config.py", "lib/sparse_toy.py",
                         "lib/sparse_models.py", "lib/sparse_metrics.py", "lib/denoising_toy.py", "lib/toy_metrics.py")
                        if name is not None}
        package_digest = state_hash({name: value for name, value in receipt["source_sha256"].items()
                                     if name.startswith("particlegan/")})
        brief_error, source_error_frame = error_receipt(result["error"], path)
        records.append(dict(catalog_id=identifier, name=entry["label"], historical_catalog_status="SOURCE_REVIEW_ONLY",
                            historical_quality_rating=4, historical_receipt_unchanged=True,
                            fresh_execution_status=result["status"], original_scientific_status=original,
                            added_gate_status=gate, media=image, protocol=protocol,
                            protocol_sha256=digest(path / "protocol.json"), source_receipt_sha256=digest(path / "source-receipt.json"),
                            scientific_source_revision=REVISION, source_package_digest=package_digest,
                            source_dependency_sha256=dependencies, launcher_receipt=launcher,
                            hard_timeout_receipt=read(path / "hard-timeout.json") if (path / "hard-timeout.json").exists() else None,
                            observer_sha256=receipt["runner_sha256"], helper_sha256=receipt["helper_sha256"],
                            evaluator_sha256=receipt["evaluator_sha256"], known_wall_seconds=launcher["elapsed_seconds"],
                            original_update_budget=protocol["original_update_budget"],
                            completed_update_pairs=result["completed_update_pairs"], completed_update_pairs_are_lower_bound=not complete,
                            last_observation=rows[-1] if rows else None, last_actual_frame_id=len(rows)-1 if rows else None,
                            effective_training=read(effective_path) if effective_path.exists() else None,
                            parity=read(parity_path) if parity_path.exists() else None, error=brief_error,
                            last_source_error_frame=source_error_frame,
                            external_full_log=dict(path=str(path / "execution.log"), sha256=digest(path / "execution.log")),
                            raw_terminal_receipt=dict(path=str(path / "execution-summary.json"),
                                                      sha256=digest(path / "execution-summary.json"))
                            if (path / "execution-summary.json").exists() else None,
                            raw_artifact_directory=str(path), terminal_live=terminal(rows, "live"), terminal_ema=terminal(rows, "ema")))
    # Controls use the same pinned definitions as the trained/blocked attempts.
    first = args.artifacts / "source-family-00"
    import sys
    sys.path.insert(0, str(first / "source"))
    from lib.sparse_toy import SparseMixedToy
    from lib.denoising_toy import GaussianGrid
    frozen = load_module("_conditional_report_scoring", first / "scoring.py")
    controls = frozen.calibration(SparseMixedToy, GaussianGrid)
    write(args.output / "controls.json", controls)
    write(args.output / "coverage.json", dict(version="conditional-source-coverage-v1", records=records,
          frozen_evaluator_version=frozen.VERSION, frozen_evaluator_sha256=digest(first / "scoring.py"),
          analytic_controls_sha256=digest(args.output / "controls.json"), scientific_retries=0,
          attempts=4, configuration_repairs=0, known_wall_seconds=sum(r["known_wall_seconds"] for r in records)))
    (args.output / "README.md").write_text(markdown(records, controls))
    print(json.dumps({record["catalog_id"]: record["fresh_execution_status"] for record in records}), flush=True)


if __name__ == "__main__":
    main()
