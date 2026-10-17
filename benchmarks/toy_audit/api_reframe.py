"""Review retained API goal media without training or model rescoring.

The original receipt, GIF, numeric observations and checkpoint stay immutable.
Reviewed GIFs and their separate renderer bindings belong in a new archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from . import api_contract, api_publish, api_run


def renderer_source():
    root = api_contract.ROOT
    names = ("api_run.py", "api_reframe.py", "api_contract.py")
    files = {f"benchmarks/toy_audit/{name}": api_run.file_hash(root / "benchmarks/toy_audit" / name)
             for name in names}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     text=True, stderr=subprocess.DEVNULL).strip()
    return {"commit": commit, "files_sha256": files}


def _raw_identity(path, names=("receipt.json", "goal.gif", "observations.npz", "final-state.pt")):
    return {name: {"sha256": api_run.file_hash(path / name), "bytes": (path / name).stat().st_size}
            for name in names}


def _observation_identity(records):
    """Bind exact numeric bytes and original metric/view metadata in memory."""
    digest = hashlib.sha256()
    for record in records:
        compact = {key: value for key, value in record.items() if key != "views"}
        compact["views"] = []
        for view in record["views"]:
            compact["views"].append({key: value for key, value in view.items()
                                     if key not in {"target", "samples"}})
            for role in ("target", "samples"):
                values = np.ascontiguousarray(view[role])
                digest.update(json.dumps([role, values.dtype.str, list(values.shape)]).encode())
                digest.update(values.tobytes())
        digest.update(json.dumps(compact, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def reconstruct_media(path, receipt, *, steps=None):
    """Restore exactly the original media-step arrays, not newly sampled data."""
    observations = {record["step"]: record for record in receipt["observations"]}
    records = []
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        for step in receipt["protocol"]["media_steps"] if steps is None else steps:
            record = deepcopy(observations[step])
            for index, view in enumerate(record["views"]):
                for role in ("target", "samples"):
                    view[role] = arrays[f"step{step}_view{index}_{role}"].copy()
            records.append(record)
    return records


def _separate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("review output must be a new directory outside the original archive")
    return source, output


def review_run(path, output):
    path, output = _separate_output(path, output)
    before = _raw_identity(path)
    receipt = api_publish.verify_run(path)
    case_id = receipt["case"]["id"]
    if case_id != path.name or output.name != case_id:
        raise ValueError("review directory must retain the actually observed case ID")
    source = renderer_source()
    records = reconstruct_media(path, receipt)
    original_observations = _observation_identity(records)
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "_rendering.gif"
    try:
        annotations = api_run.render_gif(
            receipt["case"], records, temporary,
            full_budget=receipt["default_protocol_complete"],
            requested_steps=receipt["protocol"]["updates"], final_verdict=receipt["verdict"])
        if _observation_identity(records) != original_observations:
            raise ValueError("renderer changed retained observations or metric/view metadata")
        if (not isinstance(annotations, dict)
                or annotations.get("numeric_observations_changed") is not False
                or annotations.get("default_verdict_displayed") != receipt["verdict"]):
            raise ValueError("renderer must preserve observations and display the original verdict")
        with Image.open(temporary) as gif:
            frames = gif.n_frames
        if frames != len(receipt["protocol"]["media_steps"]):
            raise ValueError("reviewed GIF lost original media steps")
        if _raw_identity(path) != before:
            raise ValueError("original raw files changed during media review")
        if renderer_source() != source:
            raise ValueError("renderer source changed during media review")
        target = output / "reviewed-goal.gif"
        temporary.rename(target)
        review = {
            "schema": "particlegan_api_toy_media_review_v1", "case_id": case_id,
            "raw_receipt": str(path / "receipt.json"),
            "raw_receipt_sha256": before["receipt.json"]["sha256"],
            "raw_artifacts": deepcopy(receipt["artifacts"]),
            "training_source_commit": receipt["source"]["commit"], "renderer_source": source,
            "verdict": receipt["verdict"], "metric_passed": receipt["metric_passed"],
            "sustained_metric_passed": receipt["sustained_metric_passed"],
            "default_protocol_complete": receipt["default_protocol_complete"],
            "failed_bounds": deepcopy(receipt["failed_bounds"]),
            "final_metrics": deepcopy(receipt["observations"][-1]["metrics"]),
            "media_steps": list(receipt["protocol"]["media_steps"]),
            "reviewed_gif": {"file": target.name, "sha256": api_run.file_hash(target),
                             "bytes": target.stat().st_size, "frames": frames},
            "annotations": annotations, "training_or_rescoring": False,
            "raw_files_unchanged": True}
        api_run.write_json(output / "media-review.json", review)
        return review
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


_FAILURE_SOURCE = "5d75e6556bb65726e71a602dbce708986a5b49f5"
_EXPORT_CASES = {"api-critic-lag-current", "api-critic-lag-even_critic", "api-critic-lag-d_antithetic"}
_PREREQUISITE_CASES = {"api-ring8-hold", "api-ring8-shift"}
_EXPORT_ERROR = "goal media/state error: ValueError: image goal views require NCHW grayscale/RGB values"
_RECIPE_HASHES = {
    "export": "3a8ee8e7899731ad1872f95ba3fd562f30dd29e50670434c12e6e1ec42bfcad8",
    "prerequisite": "db46a759e3abbeb55baaee68fed6db9e862af273c0423f725dc5ac64a559a6c2"}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=256)
def _frozen_file_hash(name):
    """Bind retained source bytes to the known original Git object."""
    if Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".py"):
        raise ValueError("unknown frozen source path")
    try:
        content = subprocess.check_output(["git", "show", f"{_FAILURE_SOURCE}:{name}"],
                                          cwd=api_contract.ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"unavailable frozen source object: {name}") from error
    return hashlib.sha256(content).hexdigest()


def verify_failure_view(path):
    """Recognize five frozen failed attempts; never qualify an ERROR receipt."""
    path = Path(path)
    receipt = api_publish.read(path / "receipt.json")
    name = receipt["case"]["id"]
    if name != path.name or name not in _EXPORT_CASES | _PREREQUISITE_CASES:
        raise ValueError("failure views support only the five named frozen attempts")
    if (receipt.get("status") != "ERROR" or receipt.get("verdict") != "FAIL"
            or receipt.get("passed") is not False or receipt.get("source_unchanged") is not True):
        raise ValueError("original ERROR/FAIL and unchanged source must be retained")
    source = receipt["source"]
    manifest = source.get("files_sha256")
    if source.get("commit") != _FAILURE_SOURCE or not isinstance(manifest, dict) or not manifest:
        raise ValueError("unknown failure source cohort")
    case = api_contract.discover()[name]
    if receipt["case"] != api_run.json_value(case):
        raise ValueError("failed attempt differs from the declared question/protocol")
    required_source = {"benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_contract.py",
                       f"benchmarks/toy_audit/{case['provider']}.py", "particlegan/recipes.py"}
    if not required_source <= manifest.keys() or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in manifest.values()):
        raise ValueError("failed source manifest must retain its explicit code bindings")
    if any(_frozen_file_hash(name) != expected for name, expected in manifest.items()):
        raise ValueError("failed source manifest differs from the frozen Git objects")
    export = name in _EXPORT_CASES
    if _json_hash(receipt["recipe"]) != _RECIPE_HASHES["export" if export else "prerequisite"]:
        raise ValueError("failure views require the exact originally resolved recipe")
    components = (["Recipe", "E22Policy", "RoutedRows", "GANLoss", "native policy lifecycle", "public initialization"]
                  if export else ["particlegan.Recipe", "particlegan.GANTrainer", "particlegan.ParticlePrior"])
    if (receipt.get("api_components") != components or receipt.get("seed") != 24002
            or receipt.get("runtime", {}).get("device") != "cpu"
            or receipt.get("runtime", {}).get("torch_threads") != 1):
        raise ValueError("failed attempt must retain its API and frozen seed bindings")
    protocol = receipt["protocol"]
    updates = case["default_steps"]
    metric_count = api_contract.metric_observations(case)
    metric_steps = api_contract.evaluation_steps(updates, metric_count + 1)
    media_steps = api_contract.evaluation_steps(updates, 9)
    planned = sorted(set(metric_steps) | set(media_steps))
    expected_protocol = {
        "updates": updates, "default_updates": updates, "evaluation_samples": case["eval_samples"],
        "default_evaluation_samples": case["eval_samples"], "metric_observations": metric_count,
        "terminal_observations": case.get("terminal_observations", 5), "media_frames": 9,
        "metric_evaluation_steps": metric_steps, "media_steps": media_steps, "evaluation_steps": planned}
    if protocol != expected_protocol:
        raise ValueError("failed attempt must retain the exact frozen evaluation schedule")
    completed = 800 if export else 1200
    records = receipt["observations"]
    if (receipt["completed_updates"] != completed or not records
            or [record.get("step") for record in records] != [step for step in planned if step <= completed]):
        raise ValueError("missing, duplicated or unordered actual failure-prefix observations")
    for record in records:
        api_publish._flags(record, f"failure observation{record['step']}")
        if (not isinstance(record.get("metrics"), dict) or not record["metrics"]
                or not all(isinstance(value, (int, float)) and np.isfinite(value)
                           for value in record["metrics"].values())
                or not isinstance(record.get("views"), list) or not record["views"]):
            raise ValueError("failure views require original finite metrics and numeric goal views")
    final = records[-1]
    if export:
        terminal = [record for record in records if record["step"] > 0 and record["step"] in metric_steps][-5:]
        if (receipt["failed_bounds"] != [_EXPORT_ERROR] or receipt.get("metric_passed") is not True
                or receipt.get("sustained_metric_passed") is not True
                or receipt.get("default_protocol_complete") is not True
                or len(terminal) != 5 or not all(record["passed"] for record in terminal)
                or receipt["artifacts"] != {} or receipt.get("gif_frames") != 0
                or (path / "goal.gif").exists() or (path / "final-state.pt").exists()):
            raise ValueError("unknown export failure or inconsistent original numeric gate")
        required_artifacts = {"observations.npz"}
        display = "FAIL (original export error; numeric gate PASS)"
    else:
        error = f"API execution or metric error: RuntimeError: scientific prerequisite failed at update1200: {final['failed_bounds']}"
        if (receipt["failed_bounds"] != [error] or final["passed"]
                or receipt.get("default_protocol_complete") is not False
                or case["phase_prerequisites"]["acquisition_step"] != completed
                or set(receipt["artifacts"]) != {"goal.gif", "observations.npz", "final-state.pt"}):
            raise ValueError("unknown or inconsistent scientific prerequisite failure")
        required_artifacts = {"goal.gif", "observations.npz", "final-state.pt"}
        display = "FAIL (prerequisite; continuation unattempted)"
    for name in required_artifacts:
        if not (path / name).is_file():
            raise ValueError(f"missing retained failure artifact: {name}")
    available = _raw_identity(path, ["receipt.json", *sorted(required_artifacts)])
    for name, expected in receipt["artifacts"].items():
        if available[name] != expected:
            raise ValueError(f"original failure artifact identity mismatch: {name}")
    if not export:
        prefix_frames = len([step for step in media_steps if step <= completed])
        with Image.open(path / "goal.gif") as gif:
            if gif.n_frames != prefix_frames or receipt.get("gif_frames") != prefix_frames:
                raise ValueError("original failure GIF differs from its actual retained media prefix")
    keys = {f"step{record['step']}_view{index}_{role}" for record in records
            for index in range(len(record["views"])) for role in ("target", "samples")}
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != keys:
            raise ValueError("failure numeric observations differ from the exact retained prefix")
        for record in records:
            for index, view in enumerate(record["views"]):
                for role in ("target", "samples"):
                    values = arrays[f"step{record['step']}_view{index}_{role}"]
                    if not values.size or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                        raise ValueError("failure views require retained finite numeric references/outputs")
                    if export and (view.get("kind") != "image" or values.shape != (8, 2, 1, 32)):
                        raise ValueError("unknown residual feature-block view layout")
    selected = [step for step in media_steps if step <= completed]
    if completed not in selected:
        selected.append(completed)
    return receipt, available, selected, display


def review_failure_run(path, output):
    """Create an explicitly unqualified supplement, separate from normal media."""
    path, output = _separate_output(path, output)
    receipt, before, steps, display = verify_failure_view(path)
    if output.name != receipt["case"]["id"]:
        raise ValueError("failure review must retain the actually observed case ID")
    source = renderer_source()
    records = reconstruct_media(path, receipt, steps=steps)
    identity = _observation_identity(records)
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / "_rendering.gif"
    try:
        annotations = api_run.render_gif(receipt["case"], records, temporary,
            full_budget=receipt["default_protocol_complete"], requested_steps=receipt["protocol"]["updates"],
            final_verdict=display)
        if (_observation_identity(records) != identity or _raw_identity(path, before.keys()) != before
                or renderer_source() != source):
            raise ValueError("retained failure observations, raw files or renderer changed during review")
        if (annotations.get("numeric_observations_changed") is not False
                or annotations.get("default_verdict_displayed") != display):
            raise ValueError("failure renderer must preserve data and the explicit original failure")
        with Image.open(temporary) as gif:
            frames = gif.n_frames
        if frames != len(steps):
            raise ValueError("failure supplement lost actual retained media steps")
        target = output / "failure-goal.gif"
        temporary.rename(target)
        review = {
            "schema": "particlegan_api_toy_failure_media_review_v1", "case_id": receipt["case"]["id"],
            "raw_receipt": str(path / "receipt.json"), "raw_receipt_sha256": before["receipt.json"]["sha256"],
            "raw_artifacts": deepcopy(receipt["artifacts"]),
            "available_raw_artifacts": {name: value for name, value in before.items() if name != "receipt.json"},
            "artifact_attestations": {name: "original_receipt" if name in receipt["artifacts"] else "review_time_only"
                                      for name in before if name != "receipt.json"},
            "training_source": deepcopy(receipt["source"]), "recipe": deepcopy(receipt["recipe"]),
            "original_execution_status": receipt["status"], "verdict": receipt["verdict"],
            "original_failed_bounds": deepcopy(receipt["failed_bounds"]),
            "original_numeric_flags": {key: receipt.get(key) for key in
                                       ("metric_passed", "sustained_metric_passed", "default_protocol_complete")},
            "completed_updates": receipt["completed_updates"], "planned_protocol": deepcopy(receipt["protocol"]),
            "original_media_steps": list(receipt["protocol"]["media_steps"]), "media_steps": steps,
            "terminal_frame_added": steps[-1] not in receipt["protocol"]["media_steps"],
            "final_metrics": deepcopy(receipt["observations"][-1]["metrics"]),
            "observations": deepcopy(receipt["observations"]), "displayed_verdict": display,
            "reviewed_gif": {"file": target.name, "sha256": api_run.file_hash(target),
                             "bytes": target.stat().st_size, "frames": frames},
            "renderer_source": source, "annotations": annotations,
            "qualification_upgrade": False, "training_or_rescoring": False, "raw_files_unchanged": True}
        api_run.write_json(output / "failure-review.json", review)
        return review
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _review_case(path, output, include_failure_views=False):
    torch.set_num_threads(1)
    path = Path(path)
    record = {"id": path.name}
    try:
        receipt_path = path / "receipt.json"
        if not receipt_path.is_file():
            return dict(record, status="SKIPPED", execution_status="PENDING",
                        reason="No completed receipt; no reviewed media created")
        receipt = api_publish.read(receipt_path)
        record["execution_status"] = receipt.get("status", "UNKNOWN")
        if receipt.get("status") != "COMPLETE":
            if include_failure_views and path.name in _EXPORT_CASES | _PREREQUISITE_CASES:
                review = review_failure_run(path, Path(output) / path.name)
                return dict(record, status="FAILURE_VIEW", verdict=review["verdict"],
                            failure_review=f"{path.name}/failure-review.json")
            return dict(record, status="SKIPPED", reason="Original execution is not COMPLETE")
        review = review_run(path, Path(output) / path.name)
        return dict(record, status="REVIEWED", verdict=review["verdict"],
                    media_review=f"{path.name}/media-review.json")
    except Exception as error:
        return dict(record, status="INVALID", reason=f"{type(error).__name__}: {error}")


def review_archive(archive, output, *, jobs=1, include_failure_views=False):
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    archive, output = _separate_output(archive, output)
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    paths = sorted(path for path in archive.iterdir() if path.is_dir())
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def retain(result):
        results.append(result)
        results.sort(key=lambda item: item["id"])
        summary = {"schema": "particlegan_api_toy_media_review_archive_v1",
                   "source_archive": str(archive), "cases": results,
                   "training_or_rescoring": False}
        api_run.write_json(output / "summary.json", summary)
        print(json.dumps(result, allow_nan=False), flush=True)

    if jobs == 1:
        for path in paths:
            retain(_review_case(path, output, include_failure_views))
    else:
        # Matplotlib has process-global state; independent spawned processes
        # avoid sharing figures/backend state between case renderings.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_review_case, path, output, include_failure_views) for path in paths]
            for future in as_completed(futures):
                retain(future.result())
    if not results:
        api_run.write_json(output / "summary.json", {
            "schema": "particlegan_api_toy_media_review_archive_v1",
            "source_archive": str(archive), "cases": [], "training_or_rescoring": False})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include-failure-views", action="store_true",
                        help="Separate unqualified views for the five known frozen failure attempts")
    args = parser.parse_args(argv)
    records = review_archive(args.runs, args.output, jobs=args.jobs, include_failure_views=args.include_failure_views)
    return int(any(record["status"] == "INVALID" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
