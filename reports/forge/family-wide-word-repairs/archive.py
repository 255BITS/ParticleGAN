"""Archive exact twelve ordinary attempts and their executed source manifests.

Bulk logs, observations and durable envelopes remain outside Git. This reads
saved evidence only; it does not train, draw samples, regrade or qualify a row.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/forge/family-wide-word-repairs"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = json.loads((REPORT / "initial-summary.json").read_text())
    rates = json.loads((REPORT / "rates-summary.json").read_text())
    candidates = {r["candidate_id"] for r in initial["candidates"] + rates["candidates"]}
    members, attempts, sources = {}, [], {}

    def add(name, data):
        if name in members:
            assert members[name] == data
        else:
            members[name] = data

    for directory in sorted((ROOT / "reports/forge/attempts").iterdir()):
        path = directory / "request.json"
        if not path.exists():
            continue
        envelope = json.loads(path.read_text())
        request = envelope["request"]
        if request["candidate"]["id"] not in candidates:
            continue
        raw_directory = Path(json.loads((directory / "evidence.json").read_text())["local_artifact_root"])
        assert raw_directory.is_relative_to(ROOT)
        for file in directory.rglob("*"):
            if file.is_file():
                add(f"durable/{directory.name}/{file.relative_to(directory)}", file.read_bytes())
        for file in raw_directory.rglob("*"):
            if file.is_file() and file.suffix != ".lock":
                add(f"raw/{directory.name}/{file.relative_to(raw_directory)}", file.read_bytes())
        commit = request["source"]["origin_commit"]
        manifest = request["source"]
        if commit in sources:
            assert sources[commit] == manifest
        sources[commit] = manifest
        snapshot = Path(request["queue_root"]) / "snapshots" / manifest["digest"]
        for relative, expected in manifest["files"].items():
            data = (snapshot / relative).read_bytes()
            assert digest(data) == expected
            add(f"sources/{commit}/{relative}", data)
        add(f"sources/{commit}/forge-source.json", encode(manifest))
        attempts.append({"attempt_id": directory.name, "candidate_id": request["candidate"]["id"],
            "candidate_revision": request["candidate_revision"], "source_commit": commit,
            "source_digest": manifest["digest"], "task_ids": envelope["job"]["task_ids"]})
    assert len(attempts) == 12 and len(candidates) == 9 and len(sources) == 2
    for queue_name in ("family-wide-word-repairs-v1", "family-wide-word-repair-rates-v1"):
        run = ROOT / "runs/forge" / queue_name
        for name in ("worker.log", "publication.log"):
            if (run / name).exists():
                add(f"queue/{queue_name}/{name}", (run / name).read_bytes())
        for file in (run / "queue").rglob("*"):
            if (file.is_file() and "snapshots" not in file.parts and
                    file.suffix in {".json", ".jsonl"} and
                    not any(a["attempt_id"] in file.parts for a in attempts)):
                add(f"queue/{queue_name}/{file.relative_to(run / 'queue')}", file.read_bytes())
    for relative in ("initial-summary.json", "rates-summary.json", "plans.json", "rates-plans.json",
                     "prepare.py", "prepare_rates.py", "run_rates.py", "render.py",
                     "materialize_initial.py", "materialize_rates.py", "archive.py", "verify_archive.py"):
        add(f"publication/{relative}", (REPORT / relative).read_bytes())
    for directory in ("initial", "rates", "media"):
        for file in (REPORT / directory).rglob("*"):
            if file.is_file():
                add(f"publication/{file.relative_to(REPORT)}", file.read_bytes())
    for family in ("k3p", "ka2", "r1r2"):
        for relative in (f"configs/forge/ideas/{family}-global-repair-v1.json",
                         f"configs/forge/searches/{family}-global-repair-rates-v1.json",
                         f"reports/forge/configuration-search/{family}-global-repair-rates-v1.json"):
            add(f"declarations/{relative}", (ROOT / relative).read_bytes())
    for row in rates["candidates"]:
        relative = f"configs/forge/configurations/{row['candidate_id']}.json"
        add(f"declarations/{relative}", (ROOT / relative).read_bytes())
    for relative in ("configs/forge/rounds/family-wide-word-repairs-v1.json",
                     "configs/forge/rounds/family-wide-word-repair-rates-v1.json",
                     "configs/forge/campaigns/family-wide-word-repairs-v1.json"):
        add(f"declarations/{relative}", (ROOT / relative).read_bytes())
    inventory = {"schema_version": 1, "scope": "exact_ordinary_evidence_and_executed_sources",
        "entries": [{"path": name, "bytes": len(data), "sha256": digest(data)}
                    for name, data in sorted(members.items())], "attempts": attempts,
        "sources": {commit: {"digest": manifest["digest"], "files": len(manifest["files"])}
                    for commit, manifest in sorted(sources.items())}}
    add("inventory.json", encode(inventory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Archive identity is immutable; choose a new name")
    with args.output.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
    card = {"schema_version": 1, "scope": "ordinary_global_candidates",
        "primary_archive_path": str(args.output.resolve()), "archive_sha256": digest(args.output.read_bytes()),
        "archive_bytes": args.output.stat().st_size, "inventory_sha256": digest(members["inventory.json"]),
        "inventory_entries": len(inventory["entries"]), "regular_files": len(members),
        "attempt_ids": [a["attempt_id"] for a in attempts], "candidate_count": 9,
        "measured_task_count": 12, "source_manifests": inventory["sources"],
        "charged_wall_seconds": initial["charged_wall_seconds"] + rates["charged_wall_seconds"],
        "original_local_logs_retained": True, "training_updates_added_by_archive": 0,
        "sampling_draws_added_by_archive": 0,
        "restore": "Extract into an isolated directory. durable/<attempt> retains the ordinary request/result/evidence certificate; raw/<attempt> retains exact saved observations and worker logs; sources/<commit> is the actual executed source. Full original queue metadata and declarations are included. Do not rewrite raw absolute paths or transplant this evidence into another candidate/source cohort.",
        "qualification": "Archive preserves the ordinary certified outcomes and UNKNOWN prerequisite-gated cells. Compact summaries, media and this card do not independently qualify a candidate."}
    (REPORT / "archive.json").write_bytes(encode(card))
    print(json.dumps({"archive": str(args.output), "sha256": card["archive_sha256"],
                      "bytes": card["archive_bytes"], "entries": card["inventory_entries"]}))


if __name__ == "__main__":
    main()
