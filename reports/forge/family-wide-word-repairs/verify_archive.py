"""Replay inventory/certificate/source/media identity checks without training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_bytes = args.archive.read_bytes()
    assert digest(archive_bytes) == args.sha256
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data["inventory.json"])
    assert set(data) == {"inventory.json"} | {e["path"] for e in inventory["entries"]}
    for entry in inventory["entries"]:
        value = data[entry["path"]]
        assert len(value) == entry["bytes"] and digest(value) == entry["sha256"]
    source_counts = {}
    for commit, source in inventory["sources"].items():
        prefix = f"sources/{commit}/"
        manifest = json.loads(data[prefix + "forge-source.json"])
        assert manifest["digest"] == source["digest"] == stable(manifest["files"])
        assert manifest["origin_commit"] == commit
        # One git cat-file batch validates exact executed blobs efficiently.
        names = list(manifest["files"])
        queries = "".join(f"{commit}:{name}\n" for name in names).encode()
        result = subprocess.check_output(["git", "cat-file", "--batch"], cwd=args.root, input=queries)
        cursor = 0
        for name in names:
            end = result.index(b"\n", cursor)
            header = result[cursor:end].split()
            assert len(header) == 3 and header[1] == b"blob", name
            size = int(header[2]); value = result[end + 1:end + 1 + size]
            assert digest(value) == manifest["files"][name]
            assert data[prefix + name] == value
            cursor = end + size + 2
        source_counts[commit] = len(names)
    tasks = []
    for attempt in inventory["attempts"]:
        prefix = f"durable/{attempt['attempt_id']}/"
        envelope = json.loads(data[prefix + "request.json"])
        request = envelope["request"]
        result = json.loads(data[prefix + "result.json"])
        evidence = json.loads(data[prefix + "evidence.json"])
        assert evidence["result_hash"] == stable(result)
        assert evidence["source"] == request["source"]
        assert evidence["runtime"] == request["runtime"]
        assert result["attempt_id"] == attempt["attempt_id"]
        assert result["candidate_revision"] == request["candidate_revision"] == attempt["candidate_revision"]
        assert request["source"]["origin_commit"] == attempt["source_commit"]
        assert request["source"] == json.loads(data[f"sources/{attempt['source_commit']}/forge-source.json"])
        raw_prefix = f"raw/{attempt['attempt_id']}/"
        raw = json.loads(data[raw_prefix + "raw-result.json"])
        grading = json.loads(data[raw_prefix + "graded-result.json"])
        assert grading["raw_hash"] == stable(raw)
        assert grading["source_digest"] == request["source"]["digest"]
        assert result["raw"]["grading"] == grading
        assert data[raw_prefix + "request.json"] == data[prefix + "request.json"]
        assert data[raw_prefix + "result.json"] == data[prefix + "result.json"]
        for row in result["task_results"]:
            task = row["task_id"]
            assert row["gate_status"] == grading["grades"][task]["gate_status"]
            assert row["compatibility_key"] == envelope["job"]["compatibility_key"]
            assert row["evaluator_result"]["convergence"]["observations"] == 24
            assert row["evidence"]["guards"]["all_finite"]
            assert row["evidence"]["guards"]["unintended_rng_deviations"] == 0
            tasks.append(row["gate_status"])
    media = 0
    for name, value in data.items():
        if not name.startswith("publication/media/") or not name.endswith(".json"):
            continue
        receipt = json.loads(value)
        if receipt.get("kind") != "actual_training_numerical_goal_gif":
            continue
        assert receipt["optimizer_updates"] == receipt["sampling_draws"] == 0
        candidates = [a for a in inventory["attempts"] if a["candidate_id"] == receipt["candidate"]
                      and receipt["task"] in a["task_ids"]]
        assert len(candidates) == 1
        attempt = candidates[0]
        raw = data[f"raw/{attempt['attempt_id']}/raw-result.json"]
        envelope = data[f"raw/{attempt['attempt_id']}/request.json"]
        assert digest(raw) == receipt["raw_result"]["sha256"]
        assert digest(envelope) == receipt["resolved_request"]["sha256"]
        rows = json.loads(raw)["evidence"]["observations"]
        assert stable(rows) == receipt["observations_sha256"]
        assert receipt["observation_count"] == len(rows) == 24
        assert receipt["updates"] == [rows[i]["step"] for i in receipt["selected_observation_indices"]]
        request = json.loads(envelope)["request"]
        assert receipt["thresholds"] == request["tasks"][receipt["task"]]["evaluation"]["thresholds"]
        assert receipt["candidate_revision"] == request["candidate_revision"]
        gif = data[name.removesuffix(".json") + ".gif"]
        assert digest(gif) == receipt["gif"]["sha256"] and len(gif) == receipt["gif"]["bytes"]
        assert receipt["renderer"]["source_sha256"] == digest(data["publication/render.py"])
        media += 1
    assert len(tasks) == media == 12 and tasks.count("PASS") == 3 and tasks.count("FAIL") == 9
    proof = {"schema_version": 1, "status": "PASS", "archive_sha256": args.sha256,
        "archive_bytes": len(archive_bytes), "inventory_entries": len(inventory["entries"]),
        "regular_files": len(data), "ordinary_certificates": len(inventory["attempts"]),
        "source_git_blob_files": source_counts, "actual_training_media_receipts": media,
        "measured_pass": 3, "measured_fail": 9,
        "reproducer_sha256": digest(Path(__file__).read_bytes()),
        "training_updates_added": 0, "sampling_draws_added": 0,
        "scope": "Exact archived member hashes, ordinary durable/grader certificates, executed Git/source blobs, saved observation/media identities. This does not regrade science or independently qualify a candidate; archive/card remains external."}
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
