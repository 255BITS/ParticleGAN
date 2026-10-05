"""Run the finite BCAP optimizer screen through Forge's unchanged public hosts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.configuration_search import enqueue_search, materialize_search, plan_search, report_search
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.queue import Queue, drain
from experiments.forge.planning import resolve_idea

CAMPAIGN = "bcap-dualnorm-tier1-v1"
REPORT = ROOT / "reports/forge/dualnorm-tier1"
ARMS = ("adam", "sgda", "nsgda-global", "nsgda-layer", "ada-nsgda", "dualnorm-zero",
        "dualnorm-momentum", "dualnorm-d-only", "particle-rownorm-only")
SPECS = tuple("bcap-optim-" + arm + "-tier1-v1" for arm in ARMS)


def emit(event, **data):
    print(json.dumps({"event": event, **data}, sort_keys=True), flush=True)


def verify_source(source):
    names = list(source["files"])
    references = "".join(source["origin_commit"] + ":" + name + "\n" for name in names)
    data = subprocess.run(["git", "cat-file", "--batch"], input=references.encode(), cwd=ROOT,
        stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    for name in names:
        end = data.index(b"\n", cursor)
        header = data[cursor:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("captured source is not committed: " + name)
        size = int(header[2]); cursor = end + 1
        payload = data[cursor:cursor + size]; cursor += size + 1
        if hashlib.sha256(payload).hexdigest() != source["files"][name]:
            raise ValueError("captured source differs from committed bytes: " + name)


def summaries(queue_root, *, final=False):
    values = [report_search(ROOT, queue_root, spec) if final else plan_search(ROOT, queue_root, spec) for spec in SPECS]
    trials = [{"arm": arm, "study": value["study_id"], **trial}
        for arm, value in zip(ARMS, values) for trial in value["trials"]]
    report = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
        "scope": "optimizer-only BCAP, protocol seed 0, complete current Tier 1; no task repairs or later tiers",
        "diagnostics": "RNG-free actual-step observers enabled for scalar/ring/joint-word hosts; excluded from grading",
        "deferred": ["R1/R2", "five-seed confirmation", "7k native100", "sparse177", "width/depth transfer"],
        "predictions": {"P1": "unscored: original native benchmarks and five-seed equivalence absent",
            "P2": "screening evidence only; no five-seed falsification",
            "P3": "unscored: native HQ/core-width comparison deferred", "P4": "screening evidence only",
            "P5": "screening evidence only; native five-seed equivalence deferred",
            "P6": "unscored: scale transfer deferred"},
        "studies": [{"id": value["study_id"], "selection": value["selection"]} for value in values],
        "trials": trials, "logs": str(queue_root / "events.jsonl")}
    if final:
        state = Queue(queue_root, report_root=ROOT / "reports/forge").inspect()
        report["accounting"] = state["campaigns"].get(CAMPAIGN, {})
        report["counts"] = dict(Counter(task["gate_status"] for trial in trials
            for task in trial["tasks"] if task["qualification_tier"] == 1))
    atomic_json(REPORT / ("results.json" if final else "plan.json"), report)
    emit("reported" if final else "planned", configurations=len(trials),
        counts=report.get("counts"), reservation_seconds=sum(value["declared_worst_case_seconds"] for value in values))
    return report


def attempts(queue_root):
    state = Queue(queue_root, report_root=ROOT / "reports/forge").inspect()
    rows = []
    seen = set()
    for entry in state["submissions"].values():
        request = entry["request"]
        if request["campaign_id"] != CAMPAIGN:
            continue
        for job in request["jobs"]:
            for attempt in state["jobs"][job["compatibility_key"]]["attempts"]:
                name = attempt["attempt_id"]
                if name not in seen:
                    rows.append((request, attempt)); seen.add(name)
    return rows


def media(queue_root):
    from experiments.forge.tier1_media import export_attempt
    rows = []
    for request, attempt in attempts(queue_root):
        name = attempt["attempt_id"]
        original = ROOT / "reports/forge/attempts" / name
        if not (original / "evidence.json").is_file():
            continue
        output = REPORT / "media" / name
        for receipt in export_attempt(original, output):
            rows.append({"candidate": request["candidate"]["id"], "attempt": name, **receipt,
                "gif": (output / (receipt["task_id"] + ".gif")).relative_to(ROOT).as_posix()})
    atomic_json(REPORT / "media.json", {"schema_version": 1, "qualification_input": False, "items": rows})
    emit("media", gifs=len(rows))


def archive(queue_root):
    destination = ROOT / "artifacts/forge" / (CAMPAIGN + ".tar.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("archive already exists; preserve its original identity")
    members, sources = {}, set()
    with tarfile.open(destination, "w:gz") as bundle:
        def add_tree(path, prefix):
            if not path.exists():
                return
            for member in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if not member.is_file() or member.is_symlink() or "__pycache__" in member.parts:
                    continue
                name = str(Path(prefix) / member.relative_to(path)) if path.is_dir() else prefix
                if name in members:
                    continue
                bundle.add(member, arcname=name, recursive=False)
                members[name] = {"sha256": file_hash(member), "bytes": member.stat().st_size}
        for request, attempt in attempts(queue_root):
            name = attempt["attempt_id"]; sources.add(request["source"]["digest"])
            add_tree(Path(attempt["path"]), "attempts/" + name)
            add_tree(ROOT / "reports/forge/attempts" / name, "durable/" + name)
            add_tree(queue_root / "queue/requests" / (request["request_id"] + ".json"), "requests/" + request["request_id"] + ".json")
        for source in sorted(sources):
            add_tree(queue_root / "snapshots" / source, "snapshots/" + source)
        add_tree(queue_root / CAMPAIGN, "campaign/" + CAMPAIGN)
        add_tree(queue_root / "events.jsonl", "events.jsonl")
    atomic_json(REPORT / "artifact-inventory.json", {"schema_version": 1,
        "archive": {"path": str(destination), "sha256": file_hash(destination), "bytes": destination.stat().st_size},
        "members": members, "source_digests": sorted(sources)})
    emit("archived", bytes=destination.stat().st_size, members=len(members))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "enqueue", "run", "report", "media", "archive"))
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge")
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args(); queue_root = args.queue_root.resolve()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        for spec in SPECS:
            materialize_search(ROOT, spec)
        return
    if args.stage == "plan":
        summaries(queue_root)
        return
    if args.stage in ("enqueue", "run"):
        plans = [plan_search(ROOT, queue_root, spec) for spec in SPECS]
        # All declarations/source are reviewed before the first paid admission.
        first = plans[0]["trials"][0]
        request = resolve_idea(ROOT, first["candidate_id"], view_id="discriminator_stability", through_tier=1,
            execution_backend="cuda", cuda_model="NVIDIA RTX A6000", queue_root=queue_root)
        verify_source(request["source"])
        for spec in SPECS:
            value = enqueue_search(ROOT, queue_root, spec)
            emit("enqueued", study=spec, configurations=len(value["trials"]))
    if args.stage == "run":
        os.environ["PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS"] = "1"
        drain(Queue(queue_root, report_root=ROOT / "reports/forge"), args.gpus.split(","),
            workers_per_gpu=1, campaign=CAMPAIGN)
    if args.stage in ("run", "report"):
        summaries(queue_root, final=True)
    elif args.stage == "media":
        media(queue_root)
    elif args.stage == "archive":
        archive(queue_root)


if __name__ == "__main__":
    main()
