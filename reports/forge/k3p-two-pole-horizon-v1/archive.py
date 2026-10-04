"""Create or verify an immutable archive of the four horizon diagnostic arms."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "artifacts/forge/k3p-two-pole-horizon-v1-v1.tar.gz"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def safe(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and ".." not in p.parts and str(p) == name, name


def create(output):
    summary = json.loads((REPORT / "summary.json").read_text())
    assert summary["qualification_input"] is summary["default_adoption"] is False
    assert len(summary["arms"]) == 4
    assert (REPORT / "README.md").is_file(), "Final readout must exist before archiving"
    assert not output.exists() and not (REPORT / "archive.json").exists(), "Archive identity is immutable; use a new version"
    members, attempts, sources, queue_roots = {}, [], {}, set()

    def add(name, data):
        safe(name)
        assert name not in members or members[name] == data, name
        members[name] = data

    def add_tree(prefix, directory):
        for file in sorted(directory.rglob("*")):
            assert not file.is_symlink(), file
            if file.is_file() and file.suffix != ".lock":
                add(f"{prefix}/{file.relative_to(directory)}", file.read_bytes())

    for arm in summary["arms"]:
        receipt = json.loads((ROOT / arm["receipt"]["path"]).read_text())
        attempt = receipt["attempt_id"]
        durable = ROOT / "reports/forge/attempts" / attempt
        envelope = json.loads((durable / "request.json").read_text())
        request = envelope["request"]
        certificate = json.loads((durable / "evidence.json").read_text())
        local = Path(certificate["local_artifact_root"]).resolve()
        queue = Path(request["queue_root"]).resolve()
        assert local.is_relative_to(queue) and queue.is_relative_to(ROOT / "runs/forge")
        queue_roots.add(queue)
        add_tree(f"durable/{attempt}", durable)
        add_tree(f"raw/{attempt}", local)
        manifest = request["source"]
        commit = manifest["origin_commit"]
        assert manifest["digest"] == summary["source_digest"] == stable(manifest["files"])
        assert commit not in sources or sources[commit]["files"] == manifest["files"]
        sources[commit] = manifest
        snapshot = queue / "snapshots" / manifest["digest"]
        for name, expected in manifest["files"].items():
            data = (snapshot / name).read_bytes()
            assert sha(data) == expected, name
            add(f"sources/{commit}/{name}", data)
        add(f"sources/{commit}/forge-source.json", encode(manifest))
        attempts.append({"arm_id": arm["arm_id"], "attempt_id": attempt,
                         "candidate_id": request["candidate"]["id"], "candidate_revision": request["candidate_revision"],
                         "task_ids": envelope["job"]["task_ids"], "source_commit": commit, "source_digest": manifest["digest"]})
    assert len({a["attempt_id"] for a in attempts}) == 4
    for index, queue in enumerate(sorted(queue_roots)):
        for file in sorted(queue.parent.rglob("*")):
            if (file.is_file() and not file.is_symlink() and file.suffix in {".json", ".jsonl", ".log"}
                    and "snapshots" not in file.parts and not any(a["attempt_id"] in file.parts for a in attempts)):
                add(f"queue/{index}/{file.relative_to(queue.parent)}", file.read_bytes())
    for file in sorted(REPORT.rglob("*")):
        if file.is_file() and "__pycache__" not in file.parts and file.name not in {"archive.json", "archive-audit.json"}:
            add(f"publication/{file.relative_to(REPORT)}", file.read_bytes())
    for name in summary["declarations"]:
        add(f"declarations/{name}", (ROOT / name).read_bytes())
    for control in summary["historical_prefix_controls"]:
        for artifact in control["reference_artifacts"]:
            data = Path(artifact["path"]).read_bytes()
            assert sha(data) == artifact["sha256"] and len(data) == artifact["bytes"]
            add(artifact["archive_member"], data)
    inventory = {"schema_version": 1, "scope": "exact_nonqualifying_horizon_diagnostic_evidence",
                 "attempts": attempts,
                 "sources": {commit: {"digest": m["digest"], "file_count": len(m["files"])} for commit, m in sources.items()},
                 "entries": [{"path": name, "bytes": len(data), "sha256": sha(data)} for name, data in sorted(members.items())]}
    add("inventory.json", encode(inventory))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
    card = {"schema_version": 1, "scope": inventory["scope"], "primary_archive_path": str(output.resolve()),
            "archive_sha256": sha(output.read_bytes()), "archive_bytes": output.stat().st_size,
            "inventory_sha256": sha(members["inventory.json"]), "inventory_entries": len(inventory["entries"]),
            "regular_files": len(members), "attempts": attempts, "sources": inventory["sources"],
            "charged_wall_seconds": summary["charged_wall_seconds"], "qualification_input": False,
            "default_adoption": False, "training_updates_added": 0, "sampling_draws_added": 0,
            "restore": "Extract into an isolated directory. Durable certificates, exact raw traces/checkpoints/logs, queue metadata, executed source snapshots, declarations and final publication are retained. Preserve embedded scientific identities and original absolute paths."}
    (REPORT / "archive.json").write_bytes(encode(card))
    return card


def verify(card, output):
    archive_bytes = Path(card["primary_archive_path"]).read_bytes()
    assert sha(archive_bytes) == card["archive_sha256"] and len(archive_bytes) == card["archive_bytes"]
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        files = tar.getmembers()
        assert len({f.name for f in files}) == len(files)
        for f in files:
            safe(f.name)
            assert f.isfile(), f.name
        data = {f.name: tar.extractfile(f).read() for f in files}
    assert sha(data["inventory.json"]) == card["inventory_sha256"]
    inventory = json.loads(data["inventory.json"])
    assert len(inventory["entries"]) == card["inventory_entries"] and len(data) == card["regular_files"]
    assert set(data) == {"inventory.json"} | {e["path"] for e in inventory["entries"]}
    for e in inventory["entries"]:
        assert len(data[e["path"]]) == e["bytes"] and sha(data[e["path"]]) == e["sha256"]
    assert inventory["attempts"] == card["attempts"] and inventory["sources"] == card["sources"]
    for commit, source in inventory["sources"].items():
        prefix = f"sources/{commit}/"
        manifest = json.loads(data[prefix + "forge-source.json"])
        assert manifest["digest"] == source["digest"] == stable(manifest["files"])
        assert manifest["origin_commit"] == commit and len(manifest["files"]) == source["file_count"]
        names = sorted(manifest["files"])
        blobs = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT,
            input="".join(f"{commit}:{name}\n" for name in names).encode())
        cursor = 0
        for name in names:
            end = blobs.index(b"\n", cursor)
            header = blobs[cursor:end].split()
            assert len(header) == 3 and header[1] == b"blob", name
            size = int(header[2]); value = blobs[end + 1:end + 1 + size]
            assert sha(value) == manifest["files"][name] and value == data[prefix + name], name
            cursor = end + size + 2
        assert cursor == len(blobs)
    summary = json.loads(data["publication/summary.json"])
    assert summary["qualification_input"] is summary["default_adoption"] is False and len(summary["arms"]) == 4
    assert len(summary["historical_prefix_controls"]) == 2
    for control in summary["historical_prefix_controls"]:
        assert control["all24prefix_observations_exactly_match"] and control["qualification_input"] is False
        for artifact in control["reference_artifacts"]:
            value = data[artifact["archive_member"]]
            assert sha(value) == artifact["sha256"] and len(value) == artifact["bytes"]
        original = json.loads(data[f"references/{control['recipe_label']}/raw/raw-result.json"])
        assert stable(original["evidence"]["observations"]) == control["original_prefix_observations_sha256"]
    from PIL import Image
    for arm in summary["arms"]:
        receipt_name = "publication/" + str(Path(arm["receipt"]["path"]).relative_to(REPORT.relative_to(ROOT)))
        receipt = json.loads(data[receipt_name])
        assert sha(data[receipt_name]) == arm["receipt"]["sha256"]
        attempt = receipt["attempt_id"]
        envelope = json.loads(data[f"durable/{attempt}/request.json"])
        result = json.loads(data[f"durable/{attempt}/result.json"])
        cert = json.loads(data[f"durable/{attempt}/evidence.json"])
        request = envelope["request"]
        assert cert["result_hash"] == stable(result) == receipt["durable_certificate"]["result_stable_hash"]
        assert cert["source"] == request["source"] and cert["runtime"] == request["runtime"]
        assert request["source"]["digest"] == summary["source_digest"]
        assert request["candidate_revision"] == receipt["candidate_revision"] == result["candidate_revision"]
        raw = json.loads(data[f"raw/{attempt}/raw-result.json"])
        grading = json.loads(data[f"raw/{attempt}/graded-result.json"])
        assert grading["raw_hash"] == stable(raw) and grading["source_digest"] == summary["source_digest"]
        assert result["raw"]["grading"] == grading
        assert data[f"raw/{attempt}/request.json"] == data[f"durable/{attempt}/request.json"]
        assert data[f"raw/{attempt}/result.json"] == data[f"durable/{attempt}/result.json"]
        row = next(r for r in result["task_results"] if r["task_id"] == receipt["task_id"])
        assert row["metrics"] == receipt["final_metrics"] and row["gate_status"] == receipt["diagnostic_gate_status"]
        assert receipt["qualification_input"] is False
        for artifact in receipt["retained_diagnostic_artifacts"]:
            value = data[f"raw/{attempt}/{artifact['path']}"]
            assert sha(value) == artifact["sha256"] and len(value) == artifact["bytes"]
        media = receipt["media"]
        media_name = "publication/" + str(Path(media["path"]).relative_to(REPORT.relative_to(ROOT)))
        assert sha(data[media_name]) == media["sha256"] and len(data[media_name]) == media["bytes"]
        with Image.open(io.BytesIO(data[media_name])) as gif:
            assert gif.n_frames == receipt["media_frames"]
    audit = {"schema_version": 1, "status": "PASS", "archive_sha256": card["archive_sha256"],
             "regular_files_verified": len(data), "certified_attempts_verified": 4,
             "executed_source_git_blobs_verified": sum(s["file_count"] for s in inventory["sources"].values()),
             "actual_training_gifs_verified": 4, "qualification_input": False,
             "training_updates_added": 0, "sampling_draws_added": 0}
    output.write_bytes(encode(audit))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--card", type=Path, default=REPORT / "archive.json")
    parser.add_argument("--audit-output", type=Path, default=REPORT / "archive-audit.json")
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.card.read_text()), args.audit_output) if args.verify else create(args.output), sort_keys=True))
