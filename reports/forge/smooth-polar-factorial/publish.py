"""Reduce saved evidence, verify matched inputs on CUDA, and export training GIFs.

Adds no model updates, evaluation draws or scientific qualification.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result, load_tasks
from experiments.forge.gaussian_tasks import bounds, grade
from experiments.forge.tier1_media import render

HERE = Path(__file__).resolve().parent
GAUSSIAN_ARMS = dict(zip(("truncated-serial", "full-serial", "truncated-threaded", "full-threaded"),
                        ("current", "old_polar_serial", "truncated_parallel", "historical")))


def historical(binding):
    content = subprocess.check_output(["git", "show", binding["source_commit"] + ":" + binding["readout_path"]], cwd=ROOT)
    import hashlib
    if hashlib.sha256(content).hexdigest() != binding["readout_sha256"]:
        raise ValueError("historical readout differs from frozen bytes")
    return json.loads(content)


def load_state(path):
    return torch.load(path, map_location="cuda:0", weights_only=False)


def verify_committed_source(source):
    if stable_hash(source["files"]) != source["digest"]:
        raise ValueError("source manifest digest differs")
    entries = sorted(source["files"].items())
    identifiers = "".join(source["origin_commit"] + ":" + path + "\n" for path, _ in entries)
    result = subprocess.run(["git", "cat-file", "--batch"], input=identifiers.encode(),
                            stdout=subprocess.PIPE, check=True, cwd=ROOT)
    content = io.BytesIO(result.stdout)
    for path, expected in entries:
        header = content.readline().decode().split()
        if len(header) != 3 or header[1] != "blob":
            raise ValueError("missing committed source: " + path)
        data = content.read(int(header[2]))
        if content.read(1) != b"\n" or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("committed source bytes differ: " + path)
    if content.read():
        raise ValueError("unexpected committed source data")


def state_at(directory, task, initial=False):
    if task == "gaussian1d_smoke":
        return load_state(directory / "evaluator" / ("initial-state.pt" if initial else "state.pt"))
    if task == "five_word_joint_acquisition":
        return load_state(directory / "state.pt")
    return load_state(directory / "provenance/provenance-state.pt")


def quality(task, receipt):
    observations, evaluation = receipt["observations"], task["evaluation"]
    def passes(row):
        return all((row[name] >= value if op == ">=" else row[name] <= value if op == "<=" else row[name] == value)
                   for name, op, value in evaluation["thresholds"])
    flags = [passes(row) for row in observations]
    suffix = next((i for i, flag in enumerate(reversed(flags)) if not flag), len(flags))
    result = dict(passing_observations=sum(flags), observations=len(flags), passing_suffix=suffix,
                  first_full_pass=next((row["step"] for row, flag in zip(observations, flags) if flag), None),
                  final_metrics=observations[-1])
    if task["id"] == "gaussian1d_smoke":
        result.update(confirmed_steps=receipt["grade"]["evaluator_result"]["confirmed_steps"],
            best_primary_cdf_ks=min(row["cdf_ks"] for row in observations),
            final_confirmation_cdf_ks=receipt["confirmations"][-1]["metrics"]["cdf_ks"],
            confirmed_states=[dict(step=row["step"], primary_cdf_ks=row["cdf_ks"],
                confirmation_cdf_ks=receipt["confirmations"][i]["metrics"]["cdf_ks"])
                for i, row in enumerate(observations) if row["step"] in receipt["grade"]["evaluator_result"]["confirmed_steps"]])
    result["failed_bound_counts"] = {name + " " + op + " " + str(value): sum(
        not (row[name] >= value if op == ">=" else row[name] <= value if op == "<=" else row[name] == value)
        for row in observations) for name, op, value in evaluation["thresholds"]}
    return result


def reduce(args):
    protocol = read_json(HERE / "protocol.json")
    tasks = load_tasks(ROOT)
    controls = {key: historical(binding) for key, binding in protocol["historical_controls"].items()}
    gaussian_controls = {row["arm"]: row for row in controls["gaussian"]["arms"]}
    rows, checks, states, audits, media = {}, {}, {}, {}, {}
    source_identities = set()
    for task_id in protocol["tasks"]:
        rows[task_id], states[task_id], audits[task_id] = {}, {}, {}
        for arm in protocol["arms"]:
            name = task_id + "--" + arm["id"]
            directory = args.runs / name
            path = directory / "receipt.json"
            if not path.exists():
                if args.partial:
                    continue
                raise ValueError("missing completed receipt: " + name)
            receipt = read_json(path)
            raw = read_json(directory / "raw-result.json")
            for artifact, expected in receipt["artifacts"].items():
                p = directory / artifact
                if file_hash(p) != expected["sha256"] or p.stat().st_size != expected["bytes"]:
                    raise ValueError("saved input changed: " + name + "/" + artifact)
            source = read_json(directory / "source-manifest.json")
            if (source["origin_commit"], source["digest"]) != (receipt["source_commit"], receipt["source_digest"]):
                raise ValueError("receipt source binding differs")
            source_id = (source["origin_commit"], source["digest"])
            if source_id not in source_identities:
                verify_committed_source(source)
                source_identities.add(source_id)
            task = tasks[task_id]
            actual_grade = grade(task, raw["evidence"]) if task_id == "gaussian1d_smoke" else grade_result(task, raw)
            if actual_grade != receipt["grade"]:
                raise ValueError("original gate recomputation changed")
            audit = read_json(directory / "factor-audit.json")
            baseline_recipe = deepcopy(receipt["recipe"])
            if baseline_recipe.pop("optimizer_smoothing") != protocol["smoothing_lambda"]:
                raise ValueError("smoothing differs")
            check = dict(all_finite=receipt["guards"]["all_finite"],
                exercised=receipt["guards"]["hooks_exercised"], zero_rng_deviations=receipt["guards"]["unintended_rng_deviations"] == 0,
                completed_roles=all(value == task["execution"]["steps"] for value in receipt["guards"]["optimizer_updates"].values()),
                actual_mode=all(key.endswith(str(arm["autograd_multithreading"])) for key in receipt["mode_counts"]),
                gate_recomputed=True, source_files_verified=True)
            control = None
            if task_id == "gaussian1d_smoke":
                old_name = GAUSSIAN_ARMS[arm["id"]]
                control = gaussian_controls[old_name]
                old = args.gaussian_controls / old_name
                old_raw = read_json(old / "adapter-receipt.json")
                if file_hash(old / "adapter-receipt.json") != control["raw_receipt_sha256"]:
                    raise ValueError("archived Gaussian receipt changed")
                prefix = "bcap-gaussian-regression-v1/" + old_name + "/"
                for member in protocol["historical_controls"]["gaussian"]["archive"]["members"]:
                    if member["path"].startswith(prefix):
                        p = old / member["path"][len(prefix):]
                        if file_hash(p) != member["sha256"] or p.stat().st_size != member["bytes"]:
                            raise ValueError("archived Gaussian input bytes changed")
                baseline = state_at(old, task_id, initial=True)
                initial = state_at(directory, task_id, initial=True)
                check.update(baseline_recipe=baseline_recipe == old_raw["recipe"],
                    baseline_prior=receipt["prior"] == old_raw["prior"], baseline_data=receipt["data_sha256"] == old_raw["evidence"]["data_sha256"],
                    baseline_initial_models=state_digest(initial["trainer"]["models"]) == state_digest(baseline["trainer"]["models"]),
                    baseline_initial_streams=state_digest(initial["streams"]) == state_digest(baseline["streams"]))
                saved = torch.load(directory / "evaluator/observed-samples.pt", map_location="cuda:0", weights_only=True)
                from benchmarks.toy_audit.gaussian1d_quality import score_samples
                spec = task["execution"]["host_definition"]
                check["all_saved_metrics_recomputed"] = all(score_samples(point["samples"], spec, point["step"]) == point["metrics"] and
                    score_samples(point["confirmation_samples"], spec, point["step"]) == point["confirmation"]["metrics"] for point in saved)
                check["initial_scored_samples_equal"] = torch.equal(saved[0]["samples"], torch.load(
                    old / "evaluator/observed-samples.pt", map_location="cuda:0", weights_only=True)[0]["samples"])
            elif task_id == "five_word_joint_acquisition":
                control = controls["words"]["arms"][arm["id"]]
                old = args.word_controls / arm["id"]
                old_raw = read_json(old / "raw-result.json")
                if file_hash(old / "raw-result.json") != control["artifacts"]["raw-result.json"]["sha256"]:
                    raise ValueError("archived word receipt changed")
                for artifact, expected in control["artifacts"].items():
                    p = old / artifact
                    if file_hash(p) != expected["sha256"] or p.stat().st_size != expected["bytes"]:
                        raise ValueError("archived word input bytes changed")
                check.update(baseline_recipe=stable_hash(baseline_recipe) == control["resolved_recipe_sha256"],
                    baseline_prior=receipt["prior"] == control["prior"], baseline_initialization=receipt["initialization"] == control["initialization"],
                    baseline_initial_models=receipt["host"] == control["host"],
                    baseline_data=receipt["data_sha256"] == control["data_sequence_sha256"])
                from benchmarks.toy_audit.api_images import score_words
                saved = torch.load(directory / "observed-records.pt", map_location="cuda:0", weights_only=True)
                if [point["step"] for point in saved] != [point["step"] for point in receipt["observations"]]:
                    raise ValueError("saved word observation schedule differs")
                verified = []
                for point, observed in zip(saved, receipt["observations"]):
                    views = {view["title"]: view for view in point["views"] if view["kind"] == "text"}
                    generated = views["Actual generated word strings"]["samples"]
                    reconstructed = views["Actual matched word reconstructions"]["samples"]
                    scored = score_words(generated.cpu().numpy(), reconstructed.cpu().numpy())
                    expected = {key: observed[key] for key in scored["metrics"]}
                    verified.append(scored["metrics"] == expected and scored["passed"] == point["passed"]
                                    and scored["failed_bounds"] == point["failed_bounds"])
                check["all_saved_metrics_recomputed"] = len(verified) == 24 and all(verified)
            if control is not None:
                check["original_control_artifacts_verified"] = True
            final = state_at(directory, task_id)
            states[task_id][arm["id"]] = final
            audits[task_id][arm["id"]] = audit
            if control is not None:
                old_final = state_at(old, task_id)
                check["baseline_final_streams"] = state_digest(final["streams"]) == state_digest(old_final["streams"])
                if task_id == "five_word_joint_acquisition":
                    check["baseline_word_data_generator"] = torch.equal(final["fixture"]["data_generator"], old_final["fixture"]["data_generator"])
            if not all(check.values()):
                raise ValueError("matched evidence verification failed: " + name + " " + str(check))
            checks[name] = check
            rows[task_id][arm["id"]] = dict(arm=arm, gate=receipt["grade"].get("gate_status"),
                quality=quality(task, receipt), original_unsmoothed_gate=control["grade"].get("gate_status") if control else None,
                original_unsmoothed_quality=(control["grade"].get("evaluator_result") if control else None),
                guards=receipt["guards"], source_commit=receipt["source_commit"], source_digest=receipt["source_digest"],
                receipt_sha256=file_hash(path), source_manifest_sha256=file_hash(directory / "source-manifest.json"),
                baseline_recipe_sha256=stable_hash(baseline_recipe), data_sha256=receipt["data_sha256"],
                initial_models=receipt["initial_models"], cost=receipt["cost"],
                rank_scope="only steps 1,2 and 24 evenly spaced checkpoints",
                sampled_rank_summary=dict(matrix_updates=len(audit["ranks"]),
                    below_threshold=sum(x["below_threshold"] for x in audit["ranks"]),
                    actual_removed=sum(x["actual_removed"] for x in audit["ranks"])))
            if args.media:
                media_path = HERE / "media" / (name + ".gif")
                if media_path.exists() and media_path.with_suffix(".json").exists():
                    cached = read_json(media_path.with_suffix(".json"))
                    if cached["gif_sha256"] != file_hash(media_path) or cached["observations_sha256"] != stable_hash(receipt["observations"]):
                        raise ValueError("existing media changed")
                    media[name] = cached
                else:
                    media[name] = render(task, dict(gate_status=receipt["grade"].get("gate_status"), evidence=raw["evidence"]),
                                         directory, media_path)
    if len(source_identities) != 1:
        raise ValueError("factorial must retain one exact executed source")
    cross_arm = {}
    for task_id, variants in states.items():
        if not variants:
            continue
        cross_arm[task_id] = dict(same_final_named_streams=len({state_digest(s["streams"]) for s in variants.values()}) == 1,
            same_initial_models=len({stable_hash(r["initial_models"]) for r in rows[task_id].values()}) == 1,
            same_data=len({stable_hash(r["data_sha256"]) for r in rows[task_id].values()}) == 1,
            same_base_recipe=len({r["baseline_recipe_sha256"] for r in rows[task_id].values()}) == 1)
        if not all(cross_arm[task_id].values()):
            raise ValueError("cross-arm mismatch: " + task_id)
    result = dict(schema_version=1, id=protocol["id"], scope=protocol["scope"],
        qualification_input=False, default_adoption=False, partial=args.partial,
        protocol_sha256=file_hash(HERE / "protocol.json"), smoothing_lambda=protocol["smoothing_lambda"],
        paper_epsilon=protocol["paper_epsilon"], original_control_bindings={name: {
            key: binding[key] for key in ("source_commit", "readout_path", "readout_sha256", "pull_request")}
            for name,binding in protocol["historical_controls"].items()},
        cells=rows, verification=checks, cross_arm_invariants=cross_arm, media=media,
        completed_runs=sum(len(v) for v in rows.values()),
        actual_updates=sum(protocol["tasks"][t]["updates"] * len(v) for t,v in rows.items()),
        paid_seconds=sum(r["cost"]["wall_seconds"] for v in rows.values() for r in v.values()),
        reserved_seconds=protocol["reserved_seconds"], new_analysis_updates=0, new_sampling=0,
        scientific_retries=0, seeds=[0], ordinary_qualification_changes=0)
    current_word = audits.get("five_word_joint_acquisition", {}).get("truncated-serial")
    if current_word:
        role_by_shape = {(64, 2): "G", (128, 64): "G", (168, 128): "G",
                         (128, 168): "E", (64, 128): "E", (2, 64): "E",
                         (256, 170): "D", (128, 256): "D", (1, 128): "D"}
        result["retrospective_word_scale_example"] = dict(
            arm="truncated-serial", step=16668, scope="saved network matrix directions only; no causal role isolation",
            rows=[dict(role=role_by_shape[tuple(row["shape"])], shape=row["shape"],
                       singular_max=row["singular_max"], leading_direction_weight=row["singular_max"] / math.hypot(
                           row["singular_max"], protocol["smoothing_lambda"]))
                  for row in current_word["ranks"] if row["step"] == 16668])
    atomic_json(args.output, result)
    print(json.dumps(dict(completed_runs=result["completed_runs"], cells={t:{a:r["gate"] for a,r in v.items()} for t,v in rows.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/forge/smooth-polar-factorial-v1")
    parser.add_argument("--gaussian-controls", type=Path, required=True)
    parser.add_argument("--word-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "readout.json")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--media", action="store_true")
    options = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("saved tensor verification requires one visible CUDA GPU")
    reduce(options)
