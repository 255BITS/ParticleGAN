"""CPU-only dynamic checker; validation inputs stay read-only.

Calls the frozen lane's original canonical collector for completed jobs. Its
JSON writes are redirected to this review directory; report()/manifest() are
never called because they write into the validation lane. Pending jobs remain
pending until root's completion event or a finalized execution receipt exists.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
REVIEW = Path(__file__).resolve().parent
STUDY = REVIEW.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--validation",type=Path,default=STUDY / "validation")
parser.add_argument("--watch",action="store_true")
parser.add_argument("--output",type=Path,default=REVIEW / "validation-monitor")
args = parser.parse_args()
args.validation = args.validation.resolve()
args.output = args.output.resolve()
if not args.output.is_relative_to(REVIEW):
    raise SystemExit("monitor outputs must be inside integration/review")
args.output.mkdir(parents=True,exist_ok=True)
captured_global_sha = None


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+"\n")


def import_file(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_global():
    global captured_global_sha
    frozen_path = args.validation / "source-freeze.json"
    current_sha = sha(frozen_path)
    if captured_global_sha is not None and current_sha != captured_global_sha:
        raise RuntimeError("validation source-freeze receipt changed after checker startup")
    frozen = read(frozen_path)
    failures = []
    for name,expected in frozen["local_sources"].items():
        path = args.validation / name
        if not path.exists() or sha(path)!=expected:failures.append(str(path))
    for name,expected in frozen["external_sources"].items():
        path = Path(name)
        if not path.exists() or sha(path)!=expected:failures.append(str(path))
    if failures:raise RuntimeError("frozen validation inputs changed: "+", ".join(failures))
    captured_global_sha = current_sha
    # The completed baseline used the same immutable maps under its variants.
    identity = frozen if "package_sha256" in frozen else frozen["variants"]["CB64-RA"]
    return dict(status="VALID",source_freeze_sha256=current_sha,
                package_sha256=identity["package_sha256"],config_sha256=identity["config_sha256"],
                checked_local=len(frozen["local_sources"]),checked_external=len(frozen["external_sources"]))


def queue_events():
    path = args.validation / "run.log"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue  # A final line may be in flight.
    return rows


def redirected_write(path,value):
    path = Path(path).resolve()
    if not path.is_relative_to(args.validation):
        raise RuntimeError("collector attempted a write outside validation")
    write(args.output / "canonical-receipts" / path.relative_to(args.validation),value)


def snapshot_report(lane,records,queue,integrity):
    rows = [records.get(task,dict(task=task,primary_status="PENDING",acceptance_status="PENDING",
            canonical_fixture_validity="UNVERIFIED")) for task in lane.TASKS]
    counts = {name:dict(Counter(row["acceptance_status"] for row in rows if row["task"] in tasks))
              for name,tasks in (("portability",lane.PORTABILITY),("native",lane.NATIVE))}
    complete = all(row["acceptance_status"]!="PENDING" for row in rows)
    status = "PENDING"
    if complete:
        status = "ERROR" if any(row["acceptance_status"]=="ERROR" for row in rows) else (
                 "PASS" if all(row["acceptance_status"]=="PASS" for row in rows) else "FAIL")
    summary = dict(status=status,counts=counts,source_integrity=integrity,records=rows,
                   completed=len(records),total=len(rows),scope="CPU saved-artifact canonical check; validation bytes read-only",
                   completed_other_jobs=[row for row in queue if row.get("event")=="job_complete"
                                         and not row.get("name","").startswith("screen-")])
    write(args.output / "summary.json",summary)
    lines = ["# Corrected CUDA validation artifact review","",f"State: {status}; {len(records)}/{len(rows)} screens collected.","",
             "Primary verdicts and mandatory validity come from the frozen original collector. Active tasks remain pending.","",
             "| Task | Primary | Canonical validity | Accepted | Steps | Peak reserved MiB |",
             "|---|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['task']} | {row['primary_status']} | {row['canonical_fixture_validity']} | {row['acceptance_status']} | {row.get('completed_steps','—')} | {row.get('gpu_memory',{}).get('peak_reserved_mib','—')} |")
    errors = [row for row in rows if row.get("validity_reasons") or row.get("error")]
    if errors:
        lines += ["","## Runtime or fixture errors",""]
        for row in errors:lines.append(f"- {row['task']}: {row.get('validity_reasons')}; {row.get('error')}")
    (args.output / "REPORT.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    lane = collector = None
    records = {}
    while True:
        initialized = False
        if not (args.validation / "source-freeze.json").exists():
            if not args.watch:
                print(json.dumps(dict(status="NOT_PREPARED",validation=str(args.validation))),flush=True)
                return
            time.sleep(10);continue
        if lane is None:
            integrity = verify_global()
            lane = import_file("lane",args.validation / "screens/lane.py")
            collector = import_file("review_original_canonical_collector",args.validation / "screens/collect.py")
            collector.write = redirected_write
            canonical = lane.verify_frozen()
            write(args.output / "CHECKER-IDENTITY.json",dict(global_integrity=integrity,canonical_integrity=canonical,
                  lane_source_sha256=sha(args.validation / "screens/lane.py"),
                  collector_source_sha256=sha(args.validation / "screens/collect.py"),
                  checker_source_sha256=sha(Path(__file__)),write_redirection_only=True,
                  original_canonical_checks_unchanged=True))
            print(json.dumps(dict(event="validation_freeze_verified",**integrity)),flush=True)
            initialized = True
        queue = queue_events()
        done = {row["name"].removeprefix("screen-") for row in queue
                if row.get("event")=="job_complete" and row.get("name","").startswith("screen-")}
        changed = initialized
        for task in lane.TASKS:
            if task in records:continue
            receipt_path = args.validation / "screens/runs" / task / "execution-receipt.json"
            finalized = False
            if receipt_path.exists():
                try:
                    finalized = isinstance(read(receipt_path).get("process_exit_code"),int)
                except json.JSONDecodeError:
                    if task in done:raise  # Finished evidence must be readable.
                    continue              # The running wrapper may be writing.
            if task not in done and not finalized:continue
            integrity = verify_global()
            record = collector.collect(task,require_attempt=True)
            records[task] = record;changed = True
            print(json.dumps(dict(event="completed_canonical_screen",**{key:record.get(key) for key in
                ("task","primary_status","canonical_fixture_validity","acceptance_status","validity_reasons","gpu_memory")})),flush=True)
        if changed or not args.watch:
            summary = snapshot_report(lane,records,queue,integrity)
            if len(records)==len(lane.TASKS):
                verify_global()
                artifact_map = {str(path.relative_to(args.validation)):dict(sha256=sha(path),bytes=path.stat().st_size)
                                for task in lane.TASKS for path in sorted((args.validation / "screens/runs" / task).rglob("*"))
                                if path.is_file() and "__pycache__" not in path.parts}
                write(args.output / "READ-ONLY-ARTIFACT-MANIFEST.json",dict(
                    validation=str(args.validation),source_integrity=integrity,
                    inputs=artifact_map,original_canonical_checks_unchanged=True))
                print(json.dumps(dict(event="all_canonical_screens_collected",status=summary["status"],counts=summary["counts"])),flush=True)
                return
        if not args.watch:return
        if any(row.get("event")=="queue_aborted" for row in queue):
            snapshot_report(lane,records,queue,integrity)
            print(json.dumps(dict(event="root_queue_aborted",records=len(records))),flush=True)
            return
        time.sleep(10)


try:
    main()
except Exception as error:
    failure = dict(status="ERROR",type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    write(args.output / "CHECKER-ERROR.json",failure)
    print(json.dumps(failure),flush=True)
    raise SystemExit(1)
