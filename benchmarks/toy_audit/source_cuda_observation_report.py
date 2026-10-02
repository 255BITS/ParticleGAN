"""Publish only compact CUDA receipts and GIFs of actual captured states."""
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

from .source_cuda_observation import ENTRIES, LABELS, SOURCE_SHA, VERSION, digest, sha, write


def read(path):
    return json.loads(Path(path).read_text())


def rows(directory):
    path = directory / "training/raw/observations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def terminal(observations, cohort):
    count = 0
    for shot in reversed(observations):
        if not shot["metrics"][cohort]["passed"]:
            break
        count += 1
    return dict(passing_suffix=count, minimum_required=5, passed=count >= 5)


def assess(family, status, paid, observations, budget):
    complete = bool(status == "COMPLETE" and paid.get("complete_original_endpoint") and
                    paid.get("completed_updates") == budget and observations and observations[-1]["step"] == budget)
    if family == "denoising":
        added = {cohort: "BLOCKED" if status == "BLOCKED" else "INCOMPLETE" if not complete else
                 "PASS" if terminal(observations, cohort)["passed"] else "FAIL" for cohort in ("live", "ema")}
        suffix = {cohort: terminal(observations, cohort) for cohort in ("live", "ema")}
    else:
        added, suffix = "NO_FROZEN_GATE; source metrics only", None
    return complete, added, suffix


def compact_execution(directory):
    path = directory / "execution-summary.json"
    if path.exists():
        return read(path)
    return dict(fresh_execution_status="INCOMPLETE", full_campaign_count=int((directory / "training").exists()),
                source_unchanged=None, error="No durable worker summary; consult hard-timeout and retained execution log")


def error_brief(value):
    if not value:
        return None, None
    frames = re.findall(r'File "([^\"]+)", line (\d+), in ([^\n]+)', value)
    source = [f for f in frames if "/source/" in f[0]]
    site = dict(path=source[-1][0], line=int(source[-1][1]), function=source[-1][2]) if source else None
    return value.rstrip().splitlines()[-1], site


def figure(shot, history, cloud, family, label, budget, status):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    if family == "denoising":
        c, real = cloud["requested"], cloud["reference"]
        for k, cohort in enumerate(("live", "ema")):
            sample = cloud[cohort + "_sample"]
            axes[0, k].scatter(*sample[:4096].T, c=c[:4096], cmap="tab10", s=2, alpha=.4, linewidths=0)
            axes[0, k].scatter(*cloud["means"].T, c="black", marker="+", s=8, linewidths=.5)
            native = shot["metrics"][cohort]["original_marginal"]
            axes[0, k].set(title=f"{cohort}: HQ {native['joint_hq']:.3f}, modes {native['modes']}/100",
                           xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect="equal")
        axes[0, 2].scatter(*cloud["posterior_reference"][:2048].T, c="black", s=2, alpha=.3, label="exact posterior")
        for cohort, color in (("live", "#ba1658"), ("ema", "#087e9b")):
            axes[0, 2].scatter(*cloud[cohort + "_posterior"][:2048].T, c=color, s=2, alpha=.3, label=cohort)
        axes[0, 2].set(title="Clean posterior: class0, xt=0, t=2", xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), aspect="equal")
        axes[0, 2].legend(fontsize=7)
        curves = [("Marginal joint HQ", lambda v: v["original_marginal"]["joint_hq"]),
                  ("Fixed clean-posterior quality TV", lambda v: v["added_clean_posterior"]["conditional_quality_mass_tv"]),
                  ("Original all-panel reverse posterior SW1", lambda v: v["original_reverse_probe"]["posterior_sw1"])]
        fixed = [None, .10, None]
    else:
        mask = cloud["test_group"] == 0
        geom = cloud["test_geom"][mask][0]
        samples = [cloud[cohort + "_test_sample"][mask] for cohort in ("live", "ema")]
        samples.append(cloud["test_reference"][mask])
        for k, (title, points) in enumerate(zip(("live", "ema", "Exact reference"), samples)):
            for sample in points[:24]:
                axes[0, k].plot(sample[0], sample[1], alpha=.35, linewidth=.8)
            axes[0, k].add_patch(plt.Circle((0, geom[1]), geom[2], color="#666666", alpha=.25))
            axes[0, k].set(title=title + ": first held-out class0 context",
                           xlim=(-1.4, 1.4), ylim=(-1.15, 1.15), aspect="equal")
        curves = [("All held-out contexts: valid fraction", lambda v: v["test"]["valid"]),
                  ("All held-out contexts: route TV", lambda v: v["test"]["route_tv"]),
                  ("All held-out contexts: collision fraction", lambda v: v["test"]["collision"])]
        fixed = [None, None, None]
    for index, ((title, metric), bound) in enumerate(zip(curves, fixed)):
        ax = axes[1, index]
        for cohort, color in (("live", "#ba1658"), ("ema", "#087e9b")):
            ax.plot([r["step"] for r in history], [metric(r["metrics"][cohort]) for r in history], color=color, marker=".", label=cohort)
        if bound is not None:
            ax.axhline(bound, linestyle=":", color="black", linewidth=1)
        ax.set(title=title, xlabel="actual completed update pairs", xlim=(0, max(1, shot["step"])))
        ax.legend(fontsize=8)
    fig.suptitle(f"{label} · CUDA · actual update {shot['step']:,}/{budget:,}\n"
                 f"Execution: {status} · original scientific gate: NO_FROZEN_GATE", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .91))
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=85, facecolor="white")
    plt.close(fig)
    return Image.open(stream).convert("RGB")


def media(directory, output, identifier, family, budget, status, observations):
    frames = []
    for index, shot in enumerate(observations):
        cloud = directory / f"training/raw/cloud-{index:03d}.npz"
        state = directory / f"training/raw/state-{index:03d}.pt"
        assert sha(cloud) == shot["cloud_sha256"] and sha(state) == shot["state_sha256"]
        assert shot["complete_owner_and_all_visible_cuda_rng_pure"]
        with np.load(cloud, allow_pickle=False) as arrays:
            frames.append(figure(shot, observations[:index + 1], arrays, family, LABELS[identifier], budget, status))
    if not frames:
        return None
    destination = output / "media"
    destination.mkdir(parents=True, exist_ok=True)
    gif, poster = destination / (identifier + ".gif"), destination / (identifier + ".png")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[180] * (len(frames) - 1) + [1500], loop=0)
    frames[-1].save(poster)
    assert Image.open(gif).n_frames == len(frames)
    return dict(path="media/" + gif.name, poster="media/" + poster.name,
                gif_sha256=sha(gif), poster_sha256=sha(poster), frames=len(frames),
                actual_checkpoint_steps=[r["step"] for r in observations], interpolation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for identifier, (family, config, budget) in ENTRIES.items():
        directory = args.artifacts / identifier
        provenance = read(directory / "source-receipt.json")
        assert provenance["source_revision"] == SOURCE_SHA
        assert sha(directory / "source.tar") == provenance["archive_sha256"]
        for name, expected in provenance["source_sha256"].items():
            assert sha(directory / "source" / name) == expected, name
        for name, expected in provenance["observer_source_sha256"].items():
            assert sha(Path(__file__).resolve().parents[2] / name) == expected, name
        execution, launcher = compact_execution(directory), read(directory / "launcher.json")
        assert execution.get("source_unchanged") is not False
        protocol = read(directory / "protocol.json") if (directory / "protocol.json").exists() else None
        if protocol:
            assert protocol["original_update_budget"] == budget and protocol["resolved_config"]["seed"] == 24002
            assert protocol["config_mutations"] == [] and protocol["runtime"]["visible_gpu_count"] == 1
        observations = rows(directory)
        status = execution["fresh_execution_status"]
        if launcher["hard_killed"]:
            status = "INCOMPLETE"
        paid = execution.get("training", {})
        progress = read(directory / "training/progress.json") if (directory / "training/progress.json").exists() else {}
        complete, added, suffix = assess(family, status, paid, observations, budget)
        brief, site = error_brief(execution.get("error"))
        image = media(directory, args.output, identifier, family, budget, status, observations)
        artifacts = {name: dict(path=str(directory / name), sha256=sha(directory / name)) for name in
                     ("source-receipt.json", "source.tar", "protocol.json", "execution.log", "execution-summary.json", "launcher.json",
                      "training/receipt.json", "training/raw/observations.jsonl", "training/raw/effective.json", "training/original-summary.json")
                     if (directory / name).exists()}
        critical = {name: provenance["source_sha256"][name] for name in
                    ("experiments/train_" + family + ".py", "lib/" + ("denoising_toy" if family == "denoising" else "trajectory") + ".py", config)}
        records.append(dict(catalog_id=identifier, label=LABELS[identifier], name=LABELS[identifier], family=family,
                    historical_catalog_receipt_unchanged=True, original_cpu_blocker_preserved=True,
                    fresh_execution_status=status, original_scientific_status="NO_FROZEN_GATE",
                    added_gate_status=added, added_scope=protocol["added_gate"] if protocol else "NOT_MEASURED",
                    original_update_budget=budget, completed_updates=paid.get("completed_updates", progress.get("completed_updates", 0)),
                    completed_count_definition="Completed original update pairs at the following loop boundary; interrupted iteration may contain a partial update.",
                    full_original_endpoint_complete=complete, full_source_attempts=execution["full_campaign_count"],
                    config_path=config, resolved_config=protocol["resolved_config"] if protocol else None,
                    resolved_recipe=protocol["resolved_recipe"] if protocol else None,
                    runtime=protocol["runtime"] if protocol else None, source_revision=SOURCE_SHA,
                    source_manifest_sha256=digest(provenance["source_sha256"]), critical_source_sha256=critical,
                    observer_source_sha256=provenance["observer_source_sha256"], prefix_parity=execution.get("prefix_parity"),
                    imported_scientific_source_count=len(execution.get("imported_scientific_sources", {})),
                    imported_scientific_source_manifest_sha256=digest(execution.get("imported_scientific_sources", {})),
                    observation_count=len(observations), final_observed_step=observations[-1]["step"] if observations else None,
                    final_observed_metrics=observations[-1]["metrics"] if observations else None,
                    terminal_suffix=suffix, all_captured_observer_reads_pure=
                        all(r["complete_owner_and_all_visible_cuda_rng_pure"] for r in observations) if observations else None,
                    media=image["path"] if image else None, media_receipt=image,
                    error=brief, last_source_error_frame=site, blocked_phase=execution.get("blocked_phase"),
                    wall_cap_seconds=120, subprocess_wall_seconds=launcher["elapsed_seconds"], hard_killed=launcher["hard_killed"],
                    allocator_peak_bytes=execution.get("allocator_peak_bytes"), artifacts=artifacts,
                    original_absolute_gate="NONE", qualification_credit="none", scientific_retries=0, configuration_repairs=0))
    result = dict(version=VERSION + "-publication", cohort="CUDA: one visible GPU, sequential original-source attempts",
                  historical_cpu_and_catalog_receipts_unchanged=True, records=records, full_metric_streams_in_git=False,
                  report_generator_sha256=sha(__file__), total_subprocess_wall_seconds=sum(r["subprocess_wall_seconds"] for r in records),
                  limitation="Scientific quality PASS is distinct from completed execution. Original source endpoints have no frozen aggregate acceptance gate. Added denoising gate covers only the declared single clean posterior, and requires the full original endpoint plus five terminal checks.")
    write(args.output / "coverage.json", result)
    lines = ["Four exact-source CUDA follow-ups are separate from the preserved CPU-blocked receipts. Each entry retained its original 6ec7 source/config, seed24002, initializer, full update budget and sampling law. One visible GPU ran sequential attempts, with a 120-second subprocess cap including startup and the observed/unobserved three-update prefix controls. No source, configuration or optimizer was repaired or tuned.", "",
             "| Entry | Full original budget | Execution / completed updates | Added gate | Actual training GIF |",
             "| --- | --- | --- | --- | --- |"]
    for record in records:
        gate = record["added_gate_status"]
        if isinstance(gate, dict):
            gate = "live " + gate["live"] + "; EMA " + gate["ema"]
        gif = f"[actual states]({record['media']})" if record["media"] else "none: prerequisite blocked or no durable state"
        lines.append(f"| {record['label']} | {record['original_update_budget']:,} | {record['fresh_execution_status']} / {record['completed_updates']:,} | {gate} | {gif} |")
    lines.extend(["", "All original scientific statuses remain `NO_FROZEN_GATE`; the original trainer's marginal, reverse-transition and train/test route diagnostics are retained. The stronger denoising check is exactly the existing analytic-control law: 4096 repeated clean predictions at class0, observation0, diffusion step2 (alpha_bar .5), live and EMA separately. It rejects posterior-mean collapse and wrong/ignored conditioning on that panel. It does not certify every class/time/observation posterior. Trajectory metrics retain all original train/test contexts, 512 rows per context, fresh prior particles at each reverse step and configured Gaussian reverse noise; no new aggregate binary threshold is invented.", "",
                 "Every captured read checks model/buffer/optimizer/gradient ownership, module flags, nested named generators, Python/NumPy/CPU RNGs and all visible CUDA RNGs. Four exact boundary hashes must match the unobserved source prefix before the sole full attempt begins. The full attempt also checks its initial owner hash against that baseline. A cap, API error or failed observer prerequisite cannot yield a convergence PASS. Complete budgets still require the source's final diagnostics/render/save, while denoising's added PASS requires five passing terminal observations.", ""])
    for record in records:
        if record["error"]:
            lines.extend([record["label"] + ": `" + record["error"] + "`.", ""])
    lines.extend(["`coverage.json` contains compact final metrics, actual frame indices, terminal suffixes, source/runtime/config/recipe identities and external archive hashes. Raw observations, full tracebacks, optimizer checkpoints and original source outputs stay outside Git under the supplied artifact directory. GIFs show only actual captured states, with no interpolation or selected best checkpoint. CUDA timing is a separate hardware/contention cohort and supplies no default or Forge qualification.", ""])
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(records=len(records), execution=[r["fresh_execution_status"] for r in records],
                          media=sum(r["media"] is not None for r in records)), sort_keys=True))


if __name__ == "__main__":
    main()
