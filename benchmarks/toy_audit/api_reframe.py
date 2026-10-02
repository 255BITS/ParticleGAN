"""Review retained API goal media without training or model rescoring.

The original receipt, GIF, numeric observations and checkpoint stay immutable.
Reviewed GIFs and their separate renderer bindings belong in a new archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
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


def _raw_identity(path):
    return {name: {"sha256": api_run.file_hash(path / name), "bytes": (path / name).stat().st_size}
            for name in ("receipt.json", "goal.gif", "observations.npz", "final-state.pt")}


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


def reconstruct_media(path, receipt):
    """Restore exactly the original media-step arrays, not newly sampled data."""
    observations = {record["step"]: record for record in receipt["observations"]}
    records = []
    with np.load(path / "observations.npz", allow_pickle=False) as arrays:
        for step in receipt["protocol"]["media_steps"]:
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


def _review_case(path, output):
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
            return dict(record, status="SKIPPED", reason="Original execution is not COMPLETE")
        review = review_run(path, Path(output) / path.name)
        return dict(record, status="REVIEWED", verdict=review["verdict"],
                    media_review=f"{path.name}/media-review.json")
    except Exception as error:
        return dict(record, status="INVALID", reason=f"{type(error).__name__}: {error}")


def review_archive(archive, output, *, jobs=1):
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
            retain(_review_case(path, output))
    else:
        # Matplotlib has process-global state; independent spawned processes
        # avoid sharing figures/backend state between case renderings.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_review_case, path, output) for path in paths]
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
    args = parser.parse_args(argv)
    records = review_archive(args.runs, args.output, jobs=args.jobs)
    return int(any(record["status"] == "INVALID" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
