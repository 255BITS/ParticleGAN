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
import torch

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


def bound(path):
    return dict(path=str(path), sha256=sha(path))


def prefix_diagnosis(directory, execution):
    """Explain a prerequisite failure using archived software evidence only."""
    baseline, observed = (execution.get(name, {}) for name in ("prefix_baseline", "prefix_observed"))
    parity = execution.get("prefix_parity", {})
    path = directory / "prefix-observed/raw/observations.jsonl"
    observations = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    result = dict(role="SOFTWARE_PREFIX; not a scientific quality campaign",
                  baseline_completed_updates=baseline.get("completed_updates", 0),
                  observed_completed_updates=observed.get("completed_updates", 0),
                  software_updates=parity.get("software_updates", 0), observation_count=len(observations),
                  actual_observed_steps=[shot["step"] for shot in observations],
                  archived_metric_stream=bound(path) if path.exists() else None,
                  root_mechanism="UNRESOLVED", root_mechanism_confidence="not established")
    if execution.get("blocked_phase") != "observer parity prerequisite" or not observations:
        return result
    first = observations[0]
    step, before = first["step"], first["owner_state_sha256"]
    assert step == 1 and first["complete_owner_and_all_visible_cuda_rng_pure"]
    state, cloud = directory / "prefix-observed/raw/state-000.pt", directory / "prefix-observed/raw/cloud-000.npz"
    assert sha(state) == first["state_sha256"] and sha(cloud) == first["cloud_sha256"]
    # The frozen Capture records the hash BEFORE entering the read, verifies it
    # unchanged afterward, and saves these owners. This CPU check independently
    # binds the saved tensors to the hash used by the original prerequisite.
    saved = torch.load(state, map_location="cpu", weights_only=False)
    saved_hash = digest(saved)
    after = observed["boundaries"][str(step)]
    assert before == after == saved_hash
    initial_equal = baseline["boundaries"]["0"] == observed["boundaries"]["0"]
    already_differs = baseline["boundaries"][str(step)] != before
    assert initial_equal and already_differs
    ids = saved["owners"].get("ids")
    ids_summary = None
    if ids is not None:
        _, counts = torch.unique(ids[0], return_counts=True)
        ids_summary = dict(rows=ids[0].numel(), unique_rows=counts.numel(),
                           repeated_rows=int((counts > 1).sum()), maximum_multiplicity=int(counts.max()))
    missing = directory / "prefix-baseline/raw/state-000.pt"
    assert not missing.exists(), "Amended baseline evidence requires a separate cohort"
    result.update(
        observed_failure_class="EXACT_SOURCE_PREFIX_DIVERGENCE_BEFORE_FIRST_OBSERVER",
        observed_failure_confidence="high; archived before/after hashes and saved owner tensors agree",
        initial_tracked_owners_match=True, initial_owner_sha256=baseline["boundaries"]["0"],
        first_divergent_completed_update=step,
        baseline_owner_sha256=baseline["boundaries"][str(step)],
        observed_before_read_owner_sha256=before, observed_after_read_owner_sha256=after,
        independently_recomputed_checkpoint_owner_sha256=saved_hash,
        first_read_preserved_tracked_owners_and_rngs=True,
        divergence_predates_first_read=True, checkpoint=bound(state), cloud=bound(cloud),
        final_generator_batch_prior_id_multiplicity=ids_summary,
        proven_cause_of_acquisition_stop="The frozen prerequisite requires four exact state matches; only initialization matches. The full source campaign is therefore prohibited by that protocol.",
        causal_limit="The first measurement cannot explain a difference already present before that measurement. This does not identify the differing owner or prove a particular CUDA kernel, source optimizer defect or omitted runtime state.",
        kernel_hypothesis="Learned particle tables use indexed differentiable reads. A CUDA numerical reduction mechanism is a hypothesis, not a demonstrated cause; generator prior IDs are unique in both trajectory prefixes.",
        missing_artifacts=[dict(path=str(missing), status="NOT_RECORDED", purpose="Baseline first-update owner tensors or per-owner hashes are needed to locate the difference; only the aggregate baseline hash exists."),
                           dict(status="NOT_RECORDED", artifact="Per-control CUDA deterministic-algorithm, backend and stream settings", purpose="Separate numerical execution behavior from a missed non-RNG runtime setting.")],
        bounded_next_diagnostic="A separate bounded per-owner software control would need baseline and observed initialization/update1 tensors or subhashes plus CUDA backend/determinism/stream settings. Compare RNGs, gradients, parameters and optimizer tensors before considering an isolated kernel control. Such a control would confer no scientific qualification; these receipts do not warrant a full quality retry, changed config, new seed or weakened parity gate.")
    return result


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
    parser.add_argument("--software-test-log", type=Path,
                        help="External log of the independent software CUDA controls")
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
            assert sha(directory / "protocol.json") == execution["protocol_sha256"]
            assert protocol["source_manifest_sha256"] == digest(provenance["source_sha256"])
        for binding in execution.get("imported_scientific_sources", {}).values():
            assert provenance["source_sha256"][binding["path"]] == binding["sha256"]
            assert sha(directory / "source" / binding["path"]) == binding["sha256"]
        observations = rows(directory)
        status = execution["fresh_execution_status"]
        if launcher["hard_killed"]:
            status = "INCOMPLETE"
        paid = execution.get("training", {})
        progress = read(directory / "training/progress.json") if (directory / "training/progress.json").exists() else {}
        complete, added, suffix = assess(family, status, paid, observations, budget)
        if family == "trajectory" and not complete:
            added = "NO_FROZEN_GATE; full-budget quality observation " + status
        diagnosis = prefix_diagnosis(directory, execution)
        brief, site = error_brief(execution.get("error"))
        image = media(directory, args.output, identifier, family, budget, status, observations)
        artifacts = {name: dict(path=str(directory / name), sha256=sha(directory / name)) for name in
                     ("source-receipt.json", "source.tar", "protocol.json", "execution.log", "execution-summary.json", "launcher.json",
                      "prefix-baseline/receipt.json", "prefix-observed/receipt.json", "prefix-observed/raw/effective.json",
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
                    source_acquisition_attempts=execution.get("source_attempts", 1),
                    scientific_measurement_status="NOT_MEASURED; software prerequisite " + status
                        if not execution["full_campaign_count"] else status,
                    config_path=config, resolved_config=protocol["resolved_config"] if protocol else None,
                    resolved_recipe=protocol["resolved_recipe"] if protocol else None,
                    actual_source_contract=read(directory / "prefix-observed/raw/effective.json")
                        if (directory / "prefix-observed/raw/effective.json").exists() else None,
                    runtime=protocol["runtime"] if protocol else None, source_revision=SOURCE_SHA,
                    source_manifest_sha256=digest(provenance["source_sha256"]), critical_source_sha256=critical,
                    observer_source_sha256=provenance["observer_source_sha256"], prefix_parity=execution.get("prefix_parity"),
                    prefix_diagnosis=diagnosis, software_updates=diagnosis["software_updates"],
                    imported_scientific_source_count=len(execution.get("imported_scientific_sources", {})),
                    imported_scientific_source_manifest_sha256=digest(execution.get("imported_scientific_sources", {})),
                    observation_count=len(observations), final_observed_step=observations[-1]["step"] if observations else None,
                    final_observed_metrics=observations[-1]["metrics"] if observations else None,
                    terminal_suffix=suffix, all_captured_observer_reads_pure=
                        all(r["complete_owner_and_all_visible_cuda_rng_pure"] for r in observations) if observations else None,
                    media=image["path"] if image else None, media_receipt=image,
                    error=brief, last_source_error_frame=site, blocked_phase=execution.get("blocked_phase"),
                    wall_cap_seconds=120, subprocess_wall_seconds=launcher["elapsed_seconds"], hard_killed=launcher["hard_killed"],
                    worker_wall_seconds=execution.get("worker_wall_seconds"),
                    wall_cap_scope="Worker subprocess including startup and software controls; source export preparation is outside this clock.",
                    allocator_peak_bytes=execution.get("allocator_peak_bytes"), artifacts=artifacts,
                    original_absolute_gate="NONE", qualification_credit="none", scientific_retries=0, configuration_repairs=0))
    software_tests = None
    if args.software_test_log:
        content = args.software_test_log.read_text()
        summary = re.search(r"(\d+) passed in ([\d.]+)s", content)
        assert summary and int(summary[1]) == 9 and "failed" not in content.lower()
        software_tests = dict(status="PASS", passed=9, elapsed_seconds=float(summary[2]), artifact=bound(args.software_test_log),
                              scope="RNG/mode/gradient/owner/import controls and a small separate prefix fixture; this supplies no source convergence result.")
    result = dict(version=VERSION + "-publication", cohort="CUDA: one visible GPU, sequential original-source attempts",
                  historical_cpu_and_catalog_receipts_unchanged=True, records=records, full_metric_streams_in_git=False,
                  coverage_catalog_ids=list(ENTRIES), scientific_quality_media_count=sum(r["media"] is not None for r in records),
                  total_software_updates=sum(r["software_updates"] for r in records),
                  total_full_source_attempts=sum(r["full_source_attempts"] for r in records),
                  software_controls=software_tests, qualification_credit="none",
                  report_generator_sha256=sha(__file__), total_subprocess_wall_seconds=sum(r["subprocess_wall_seconds"] for r in records),
                  total_worker_wall_seconds=sum(r["worker_wall_seconds"] or 0 for r in records),
                  limitation="Scientific quality PASS is distinct from completed execution. Original source endpoints have no frozen aggregate acceptance gate. Added denoising gate covers only the declared single clean posterior, and requires the full original endpoint plus five terminal checks.")
    write(args.output / "coverage.json", result)
    all_parity_blocked = all(r["blocked_phase"] == "observer parity prerequisite" and r["full_source_attempts"] == 0 for r in records)
    lines = []
    if all_parity_blocked:
        lines.extend(["All four CUDA follow-ups are **BLOCKED before a full quality attempt**. CUDA availability was restored, but the original source controls failed the next prerequisite: only the initial tracked state matches between the unobserved and observed prefixes. Each control ran three source update pairs, for six software updates per entry and 24 total. There were zero full campaigns, zero scientific quality checkpoints, zero training GIFs and zero qualification credit. The single step1 capture in each observed prefix remains external software evidence.", ""])
    lines.extend(["These follow-ups preserve the earlier CPU-blocked receipts and original catalog results. Each entry retained its original 6ec7 source/config, seed24002, initializer, full update budget and sampling law. One visible RTX A6000 ran sequential attempts under Torch2.13.0+cu126 and Python3.12.13, with a 120-second subprocess cap including startup and the observed/unobserved three-update prefix controls. Source archive preparation was outside that subprocess clock. No source, configuration or optimizer was repaired or tuned.", "",
             "| Entry | Full original budget | Execution / quality updates | Worker / subprocess seconds | Software updates | Actual training GIF |",
             "| --- | --- | --- | --- | --- | --- |"])
    for record in records:
        gif = f"[actual states]({record['media']})" if record["media"] else "none: prerequisite blocked or no durable state"
        lines.append(f"| {record['label']} | {record['original_update_budget']:,} | {record['fresh_execution_status']} / {record['completed_updates']:,} | {record['worker_wall_seconds']:.4f} / {record['subprocess_wall_seconds']:.4f} | {record['software_updates']} | {gif} |")
    lines.extend(["", f"Total worker time: {result['total_worker_wall_seconds']:.4f}s; supervised subprocess time: {result['total_subprocess_wall_seconds']:.4f}s. Neither number is a trained-model throughput result.", ""])
    if software_tests:
        lines.extend([f"The independent CUDA software controls passed **9 tests in {software_tests['elapsed_seconds']:.2f}s**, with the full log bound in `coverage.json`. They verify the observer helper and a small separate prefix fixture; they do not establish repeatability or convergence of these four original sources.", ""])
    if all_parity_blocked:
        lines.extend(["The saved chronology locates the first divergence before the first metric read. For each entry, baseline boundary0 equals observed boundary0; baseline boundary1 differs from the observed **before-read** hash. That before-read hash equals both the after-read hash and an independently recomputed hash of the saved model/optimizer/gradient/RNG owners. The first read therefore preserved the tracked owners and cannot explain the difference already present before it. The compact proof hashes and external checkpoint/stream identities are in each record's `prefix_diagnosis`.", "",
            "The exact source/CUDA mechanism remains **unresolved**. The frozen learned particle tables perform indexed differentiable reads, but this alone does not establish a numerical reduction cause. The saved first generator batch had 255 unique IDs in 256 rows for source02 and 1943 in 2048 for source03; source04 and source05 each had 128 unique IDs in 128 rows. These counts do not prove a common repeated-prior reduction explanation. No kernel intervention or further training was run.", "",
            "The baseline's first-update owner tensors were not saved: only its aggregate hash exists. Per-control CUDA backend, deterministic-algorithm and stream metadata were also not captured. A separate bounded per-owner software control would need those tensors/subhashes and settings at initialization and update1 to locate the differing state and separate numerical behavior from an omitted runtime setting. It would confer no qualification. The present failure establishes an acquisition blocker, not a trained generator's scientific FAIL.", ""])
    lines.extend(["All original scientific statuses remain `NO_FROZEN_GATE`. The observer retains the original marginal, reverse-transition and train/test route diagnostic laws; their only captured values in this cohort belong to the external software-prefix control. The separate denoising check uses the existing analytic-control law: 4096 repeated clean predictions at class0, observation0, diffusion step2 (alpha_bar .5), live and EMA separately. It measures posterior mass and local shape on that panel and rejects the calibrated posterior-mean-collapse witness. A single fixed panel cannot demonstrate that conditioning is used or certify every class/time/observation posterior. Its scientific status here is BLOCKED for both live and EMA because the full acquisition never began. Trajectory panels retain all original train/test contexts, 512 rows per context, fresh prior particles at each reverse step and configured Gaussian reverse noise; no new aggregate binary threshold is invented, and the full-budget quality observation remains BLOCKED.", "",
                 "Every captured read checks model/buffer/optimizer/gradient ownership, module flags, nested named generators, Python/NumPy/CPU RNGs and all visible CUDA RNGs. Four exact boundary hashes must match the unobserved source prefix before the sole full attempt begins. The full attempt also checks its initial owner hash against that baseline. A cap, API error or failed observer prerequisite cannot yield a convergence PASS. Complete budgets still require the source's final diagnostics/render/save, while denoising's added PASS requires five passing terminal observations.", ""])
    for record in records:
        if record["error"]:
            lines.extend([record["label"] + ": `" + record["error"] + "`.", ""])
    lines.extend(["`coverage.json` contains compact execution and diagnosis receipts, source/runtime/config/recipe/resource identities and external archive hashes. Scientific observation counts are zero, final scientific metrics are null and media paths are null. Raw software-prefix metrics, checkpoints and original source outputs stay outside Git under the supplied artifact directory. CUDA timing is a separate hardware/contention cohort and supplies no default or Forge qualification.", ""])
    (args.output / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(records=len(records), execution=[r["fresh_execution_status"] for r in records],
                          media=sum(r["media"] is not None for r in records)), sort_keys=True))


if __name__ == "__main__":
    main()
