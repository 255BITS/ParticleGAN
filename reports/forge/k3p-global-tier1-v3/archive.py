"""Archive or independently verify this finite study's exact saved evidence."""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from reports.forge.regenerate_technique_inventory import _evaluator_summary
DEFAULT_ARCHIVE = Path("/home/martyn/dev/ParticleGAN/artifacts/forge/k3p-global-tier1-v3-v1.tar.gz")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def stable(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def safe_name(name):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and ".." not in path.parts and str(path) == name, name


def create(output):
    summaries = [json.loads(path.read_text()) for path in (REPORT / "summary.json", REPORT / "direct-moments/summary.json")]
    plans = [json.loads((REPORT / name).read_text()) for name in ("plans.json", "plans-direct-moments.json")]
    candidates = {row["candidate_id"] for summary in summaries for row in summary["candidates"]}
    members, attempts, sources, run_roots = {}, [], {}, set()

    def add(name, data):
        safe_name(name)
        assert name not in members or members[name] == data, name
        members[name] = data

    measured = []
    for summary in summaries:
        for candidate in summary["candidates"]:
            receipt = json.loads((ROOT / candidate["receipt"]).read_text())
            measured.extend(task for task in receipt["tasks"] if "attempt_id" in task)
    attempt_ids = sorted({task["attempt_id"] for task in measured})
    for attempt_id in attempt_ids:
        directory = ROOT / "reports/forge/attempts" / attempt_id
        envelope = json.loads((directory / "request.json").read_text())
        request = envelope["request"]
        certificate = json.loads((directory / "evidence.json").read_text())
        raw = Path(certificate["local_artifact_root"]).resolve()
        queue_root = Path(request["queue_root"]).resolve()
        assert request["candidate"]["id"] in candidates and raw.is_relative_to(queue_root)
        assert queue_root.is_relative_to(ROOT / "runs/forge")
        run_roots.add(queue_root.parent)
        for prefix, directory in ((f"durable/{attempt_id}", directory), (f"raw/{attempt_id}", raw)):
            for file in sorted(directory.rglob("*")):
                assert not file.is_symlink(), file
                if file.is_file() and file.suffix != ".lock":
                    add(f"{prefix}/{file.relative_to(directory)}", file.read_bytes())
        manifest = request["source"]
        commit = manifest["origin_commit"]
        assert commit not in sources or sources[commit] == manifest
        sources[commit] = manifest
        snapshot = Path(request["queue_root"]) / "snapshots" / manifest["digest"]
        for relative, expected in manifest["files"].items():
            data = (snapshot / relative).read_bytes()
            assert digest(data) == expected
            add(f"sources/{commit}/{relative}", data)
        add(f"sources/{commit}/forge-source.json", encode(manifest))
        attempts.append({"attempt_id": attempt_id, "candidate_id": request["candidate"]["id"],
            "candidate_revision": request["candidate_revision"], "source_commit": commit,
            "source_digest": manifest["digest"], "task_ids": envelope["job"]["task_ids"]})
    assert len(measured) == sum(summary["measured_task_count"] for summary in summaries) == len(attempts)
    assert len(candidates) == sum(summary["configuration_count"] for summary in summaries) == sum(plan["configuration_count"] for plan in plans)
    for run in sorted(run_roots):
        for file in sorted(run.rglob("*")):
            if (file.is_file() and not file.is_symlink() and file.suffix in {".json", ".jsonl", ".log"}
                    and "snapshots" not in file.parts and not any(attempt_id in file.parts for attempt_id in attempt_ids)):
                add(f"queue/{run.name}/{file.relative_to(run)}", file.read_bytes())
    for file in sorted(REPORT.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.name not in {"archive.json", "archive-audit.json"}:
            add(f"publication/{file.relative_to(REPORT)}", file.read_bytes())
    declarations = set()
    for plan in plans:
        spec_value = json.loads((ROOT / plan["spec"]).read_text())
        declarations.update({plan["spec"], f"configs/forge/ideas/{spec_value['base_candidate']}.json",
            f"reports/forge/configuration-search/{plan['study']}.json"})
        declarations.update(trial["declaration"] for trial in plan["trials"])
    for relative in sorted(declarations):
        add(f"declarations/{relative}", (ROOT / relative).read_bytes())
    renderer_names = {"reports/forge/family-wide-word-repairs/render.py",
        "benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_reframe.py", "benchmarks/toy_audit/api_contract.py"}
    for relative in sorted(renderer_names):
        add(f"renderers/{relative}", (ROOT / relative).read_bytes())
    add("publication/evaluator-projection.py", (ROOT / "reports/forge/regenerate_technique_inventory.py").read_bytes())
    inventory = {"schema_version": 1, "scope": "exact_ordinary_evidence_and_executed_sources",
        "entries": [{"path": name, "bytes": len(data), "sha256": digest(data)} for name, data in sorted(members.items())],
        "attempts": attempts, "sources": {commit: {"digest": manifest["digest"], "files": len(manifest["files"])}
                                          for commit, manifest in sorted(sources.items())}}
    add("inventory.json", encode(inventory))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Archive identity is immutable; choose a new version")
    with output.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
    card = {"schema_version": 1, "scope": "ordinary_global_k3p_tier1_configuration_search",
        "primary_archive_path": str(output.resolve()), "archive_sha256": digest(output.read_bytes()),
        "archive_bytes": output.stat().st_size, "inventory_sha256": digest(members["inventory.json"]),
        "inventory_entries": len(inventory["entries"]), "regular_files": len(members),
        "attempt_ids": attempt_ids, "candidate_count": len(candidates), "measured_task_count": len(measured),
        "source_manifests": inventory["sources"], "studies": [summary["study"] for summary in summaries],
        "charged_wall_seconds": sum(summary["charged_wall_seconds"] for summary in summaries),
        "original_local_logs_retained": True, "training_updates_added": 0, "sampling_draws_added": 0,
        "restore": "Extract into an isolated directory. durable/ retains certified envelopes; raw/ retains exact worker outputs, scored tensors, states and logs; sources/<commit>/ retains each actual executed source. Queue metadata, frozen declarations and publication reproduction sources are included. Preserve embedded absolute paths and scientific identities.",
        "qualification": "Archive preserves certified ordinary PASS/FAIL and UNKNOWN cells. This card, readout and GIFs do not independently qualify or adopt a candidate."}
    (REPORT / "archive.json").write_bytes(encode(card))
    return card


def verify(card, output, root):
    """Check bytes, source Git blobs, certificates and every media/receipt binding."""
    from PIL import Image
    archive_bytes = Path(card["primary_archive_path"]).read_bytes()
    assert digest(archive_bytes) == card["archive_sha256"] and len(archive_bytes) == card["archive_bytes"]
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        assert len({member.name for member in members}) == len(members)
        for member in members:
            safe_name(member.name)
            assert member.isfile(), member.name
        data = {member.name: archive.extractfile(member).read() for member in members}
    assert digest(data["inventory.json"]) == card["inventory_sha256"]
    inventory = json.loads(data["inventory.json"])
    assert len({entry["path"] for entry in inventory["entries"]}) == len(inventory["entries"])
    assert set(data) == {"inventory.json"} | {entry["path"] for entry in inventory["entries"]}
    assert len(data) == card["regular_files"] and len(inventory["entries"]) == card["inventory_entries"]
    for entry in inventory["entries"]:
        value = data[entry["path"]]
        assert len(value) == entry["bytes"] and digest(value) == entry["sha256"]
    source_counts = {}
    for commit, source in inventory["sources"].items():
        prefix = f"sources/{commit}/"
        manifest = json.loads(data[prefix + "forge-source.json"])
        assert manifest["origin_commit"] == commit and manifest["digest"] == source["digest"] == stable(manifest["files"])
        names = list(manifest["files"])
        assert source["files"] == len(names)
        blobs = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
            input="".join(f"{commit}:{name}\n" for name in names).encode())
        cursor = 0
        for name in names:
            safe_name(name)
            end = blobs.index(b"\n", cursor)
            header = blobs[cursor:end].split()
            assert len(header) == 3 and header[1] == b"blob", name
            size = int(header[2]); value = blobs[end + 1:end + 1 + size]
            assert digest(value) == manifest["files"][name] and data[prefix + name] == value
            cursor = end + size + 2
        assert cursor == len(blobs)
        source_counts[commit] = len(names)
    assert inventory["sources"] == card["source_manifests"]
    summaries = [json.loads(data[name]) for name in ("publication/summary.json", "publication/direct-moments/summary.json")]
    summary = {"candidates": [candidate for report in summaries for candidate in report["candidates"]],
               **{key: sum(report[key] for report in summaries) for key in ("measured_task_count", "measured_pass", "measured_fail", "unknown_count", "task_cells", "charged_wall_seconds")}}
    assert all(report["default_adoption"] is False for report in summaries)
    compact_tasks = {}
    outcomes = Counter()
    for candidate in summary["candidates"]:
        receipt_name = "publication/" + str(Path(candidate["receipt"]).relative_to("reports/forge/k3p-global-tier1-v3"))
        receipt = json.loads(data[receipt_name])
        assert receipt["candidate_id"] == candidate["candidate_id"] and receipt["qualification"] == candidate["qualification"]
        assert len(receipt["tasks"]) == len({task["task_id"] for task in receipt["tasks"]}) == 26
        assert sum(task["tier"] == 1 and task["importance"] == "required" for task in receipt["tasks"]) == 5
        for task in receipt["tasks"]:
            outcomes[task["status"]] += 1
            assert task["tier"] == 1 or task["status"] == "UNKNOWN"
            if "attempt_id" in task:
                key = (task["attempt_id"], task["task_id"])
                assert key not in compact_tasks
                compact_tasks[key] = task
        tier1_pass = sum(task["tier"] == 1 and task["status"] == "PASS" for task in receipt["tasks"])
        assert receipt["qualification"]["qualified_tier"] == (1 if tier1_pass == 5 else 0)
        assert receipt["default_adoption"] is False and receipt["prior_word_qualification_reuse"] is False
    rows, requests = {}, {}
    for attempt in inventory["attempts"]:
        attempt_id = attempt["attempt_id"]
        prefix, raw_prefix = f"durable/{attempt_id}/", f"raw/{attempt_id}/"
        envelope = json.loads(data[prefix + "request.json"])
        request = envelope["request"]
        result = json.loads(data[prefix + "result.json"])
        evidence = json.loads(data[prefix + "evidence.json"])
        assert evidence["result_hash"] == stable(result) and evidence["source"] == request["source"]
        assert evidence["runtime"] == request["runtime"] and result["attempt_id"] == attempt_id
        assert result["candidate_revision"] == request["candidate_revision"] == attempt["candidate_revision"]
        assert request["candidate"]["id"] == attempt["candidate_id"]
        assert request["source"] == json.loads(data[f"sources/{attempt['source_commit']}/forge-source.json"])
        raw, grading = [json.loads(data[raw_prefix + name]) for name in ("raw-result.json", "graded-result.json")]
        assert grading["raw_hash"] == stable(raw) and grading["source_digest"] == request["source"]["digest"]
        assert result["raw"]["grading"] == grading
        assert data[raw_prefix + "request.json"] == data[prefix + "request.json"]
        assert data[raw_prefix + "result.json"] == data[prefix + "result.json"]
        requests[attempt_id] = request
        for row in result["task_results"]:
            task_id = row["task_id"]
            compact = compact_tasks[(attempt_id, task_id)]
            assert row["raw_status"] == "completed" and row["gate_status"] == grading["grades"][task_id]["gate_status"] == compact["status"]
            assert row["compatibility_key"] == envelope["job"]["compatibility_key"] == compact["compatibility_key"]
            assert compact["evaluator_result"] in (row["evaluator_result"], _evaluator_summary(row["evaluator_result"]))
            assert row["metrics"] == compact["metrics"]
            assert row["evaluator_result"]["convergence"]["observations"] == len(row["evidence"]["observations"]) == 24
            assert row["evidence"]["guards"]["all_finite"] and row["evidence"]["guards"]["unintended_rng_deviations"] == 0
            effective = row if "recipe" in row else row["applied"]
            for field, target in (("recipe", "effective_recipe"), ("prior", "prior"), ("initializer", "initializer"),
                                  ("initialization", "initialization"), ("field_ownership", "field_ownership")):
                assert effective[field] == compact[target]
            assert stable(effective["rng"]) == compact["rng_manifest_sha256"]
            assert effective["field_ownership"]["task_contract"]["prior"]["value"] == effective["prior"]
            assert row["evidence"]["sampling_law"] == compact["sampling_law"]
            rows[(attempt_id, task_id)] = row
    assert set(rows) == set(compact_tasks)
    assert sorted(requests) == card["attempt_ids"]
    media = 0
    for name, value in data.items():
        if not name.startswith("publication/") or "media" not in PurePosixPath(name).parts or not name.endswith(".json"):
            continue
        receipt = json.loads(value)
        if not receipt.get("kind", "").startswith("actual_training_"):
            continue
        matches = [attempt for attempt in inventory["attempts"] if attempt["candidate_id"] == receipt["candidate"]
                   and receipt["task"] in attempt["task_ids"]]
        assert len(matches) == 1
        attempt_id = matches[0]["attempt_id"]
        request, row = requests[attempt_id], rows[(attempt_id, receipt["task"])]
        raw_prefix = f"raw/{attempt_id}/"
        assert digest(data[raw_prefix + "raw-result.json"]) == receipt["raw_result"]["sha256"]
        assert digest(data[raw_prefix + "request.json"]) == receipt["resolved_request"]["sha256"]
        observations = row["evidence"]["observations"]
        assert stable(observations) == receipt["observations_sha256"] and receipt["observation_count"] == len(observations) == 24
        assert receipt["updates"] == [observations[index]["step"] for index in receipt["selected_observation_indices"]]
        assert receipt["thresholds"] == request["tasks"][receipt["task"]]["evaluation"]["thresholds"]
        assert receipt["candidate_revision"] == request["candidate_revision"]
        assert receipt["optimizer_updates"] == receipt["sampling_draws"] == 0
        gif = data[name.removesuffix(".json") + ".gif"]
        assert digest(gif) == receipt["gif"]["sha256"] and len(gif) == receipt["gif"]["bytes"]
        with Image.open(io.BytesIO(gif)) as image:
            assert image.n_frames == len(receipt["selected_observation_indices"])
        if receipt["kind"] == "actual_training_numerical_goal_gif":
            assert receipt["renderer"]["source_sha256"] == digest(data["renderers/reports/forge/family-wide-word-repairs/render.py"])
        else:
            assert receipt["kind"] == "actual_training_saved_observer_outputs_gif"
            descriptor = row["evidence"]["saved_observer_outputs"]
            assert descriptor == receipt["saved_observer_outputs"]
            retained = data[raw_prefix + descriptor["path"]]
            assert digest(retained) == descriptor["sha256"] == receipt["retained_outputs"]["sha256"]
            assert len(retained) == descriptor["bytes"] == receipt["retained_outputs"]["bytes"]
            assert descriptor["observation_count"] == 24 and descriptor["optimizer_updates_added"] == descriptor["sampling_draws_added"] == 0
            for relative, expected in receipt["renderer"]["files_sha256"].items():
                assert digest(data["renderers/" + relative]) == expected
            assert receipt["publication_source"]["sha256"] == digest(data["publication/materialize.py"])
        media += 1
    assert len(rows) == media == summary["measured_task_count"] == card["measured_task_count"]
    assert outcomes["PASS"] == summary["measured_pass"] and outcomes["FAIL"] == summary["measured_fail"]
    assert outcomes["UNKNOWN"] == summary["unknown_count"] and sum(outcomes.values()) == summary["task_cells"] == 26 * card["candidate_count"]
    assert math.isclose(sum(row["cost"]["wall_seconds"] for row in rows.values()), summary["charged_wall_seconds"], abs_tol=1e-6)
    assert summary["charged_wall_seconds"] == card["charged_wall_seconds"]
    # The immutable archive predates this additive concurrency disclosure.
    for addendum in card.get("publication_addenda", []):
        value = (root / addendum["path"]).read_bytes()
        assert digest(value) == addendum["sha256"] and len(value) == addendum["bytes"]
        if not addendum["path"].endswith("execution-concurrency.json"):
            continue
        disclosure = json.loads(value)
        assert disclosure["archive_sha256"] == card["archive_sha256"]
        for phase in disclosure["phases"]:
            events = data[phase["archived_events"]["path"]]
            assert digest(events) == phase["archived_events"]["sha256"]
            assert len(events) == phase["archived_events"]["bytes"]
            active, intervals, peak = {}, {}, 0
            for event in sorted(map(json.loads, events.splitlines()), key=lambda event: event["timestamp"]):
                if event["event"] == "claimed":
                    active[event["attempt"]] = True
                    peak = max(peak, len(active))
                    intervals[event["attempt"]] = {"start": event["timestamp"], "backend": "cpu" if event["gpu"] == "cpu" else "cuda:" + event["gpu"], "task": event["task"]}
                elif event["event"] == "completed":
                    intervals[event["attempt"]]["end"] = event["timestamp"]
                    del active[event["attempt"]]
            assert not active and len(intervals) == phase["attempt_count"] and peak == phase["observed_maximum_workers"]
            expected = {}
            for first, left in intervals.items():
                for second, right in intervals.items():
                    start, end = max(left["start"], right["start"]), min(left["end"], right["end"])
                    if first < second and start < end:
                        expected[frozenset((first, second))] = (start, end)
            assert len(expected) == len(phase["overlaps"])
            for overlap in phase["overlaps"]:
                assert expected[frozenset(overlap["attempts"])] == (overlap["start"], overlap["end"])
                assert overlap["seconds"] == (datetime.fromisoformat(overlap["end"]) - datetime.fromisoformat(overlap["start"])).total_seconds()
                assert overlap["backends"] == [intervals[attempt]["backend"] for attempt in overlap["attempts"]]
                assert overlap["tasks"] == [intervals[attempt]["task"] for attempt in overlap["attempts"]]
    proof = {"schema_version": 1, "status": "PASS", "archive_sha256": card["archive_sha256"],
        "archive_bytes": len(archive_bytes), "inventory_entries": len(inventory["entries"]),
        "regular_files": len(data), "ordinary_certificates": len(requests), "source_git_blob_files": source_counts,
        "actual_training_media_receipts": media, "measured_pass": outcomes["PASS"], "measured_fail": outcomes["FAIL"],
        "unknown_count": outcomes["UNKNOWN"], "reproducer_sha256": digest(Path(__file__).read_bytes()),
        "publication_addenda_checked": len(card.get("publication_addenda", [])),
        "training_updates_added": 0, "sampling_draws_added": 0,
        "scope": "Exact member hashes, safe unique archive paths, ordinary durable/grader certificates, executed Git blobs, complete recipe/prior/init metadata and saved-observation/media identities. This checks provenance without training, resampling, rescoring or independent qualification."}
    output.write_bytes(encode(proof))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--verify", type=Path, help="Existing external archive.json card")
    parser.add_argument("--audit-output", type=Path, default=REPORT / "archive-audit.json")
    parser.add_argument("--root", type=Path, default=ROOT, help="Git repository containing the executed source commits")
    args = parser.parse_args()
    if args.verify:
        proof = verify(json.loads(args.verify.read_text()), args.audit_output, args.root)
        print(json.dumps(proof))
    else:
        card = create(args.output)
        print(json.dumps({key: card[key] for key in ("primary_archive_path", "archive_sha256", "archive_bytes", "inventory_entries")}))


if __name__ == "__main__":
    main()
