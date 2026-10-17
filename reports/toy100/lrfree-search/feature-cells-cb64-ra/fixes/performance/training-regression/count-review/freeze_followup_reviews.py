"""Freeze the independent joint/API/RA3 follow-up receipts, stdlib only."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

prior = read(HERE / "FROZEN.json")
for name, expected in prior["local_artifact_sha256"].items():
    assert sha(HERE / name) == expected, name
for path, expected in prior["reviewed_source_sha256"].items():
    assert sha(path) == expected, path
for path, expected in prior["inspected_owner_receipt_sha256"].items():
    assert sha(path) == expected, path


def freeze(output, receipt, local_files, archive=None, extra_files=None):
    path = HERE / output
    assert not path.exists(), "preserve the existing frozen manifest"
    result = read(HERE / receipt)
    assert result["status"] == "PASS" and not result["cuda_initialized"]
    archive = archive or {}
    for original, expected in result["source_sha256"].items():
        assert sha(archive.get(original, original)) == expected, original
    files = [HERE / name for name in local_files] + [Path(__file__)]
    extras = {str(p): sha(p) for p in extra_files or []}
    value = dict(status="PASS_CPU_EVIDENCE", receipt=receipt,
                 local_artifact_sha256={str(p.relative_to(HERE)):sha(p) for p in files},
                 reviewed_source_sha256=result["source_sha256"],
                 archived_original_path_to_retained_copy={str(k):str(v) for k,v in archive.items()},
                 external_artifact_sha256=extras,
                 previous_independent_freeze_sha256=sha(HERE / "FROZEN.json"),
                 previous_frozen_artifacts_and_sources_unchanged=True,
                 cuda_initialized=False, optimizer_updates=0, new_seeds=0,
                 scope=result["scope"])
    if "checkpoint_endpoint_sha256" in result:
        value["checkpoint_endpoint_sha256"] = result["checkpoint_endpoint_sha256"]
        for p,h in value["checkpoint_endpoint_sha256"].items():
            assert sha(p) == h, p
    path.write_text(json.dumps(value, indent=2)+"\n")
    print(json.dumps(dict(path=str(path), sha256=sha(path), status=value["status"])), flush=True)


joint = ROOT / "integration/review/training-regression/joint-count"
freeze("JOINT-FROZEN.json", "joint-review.json",
       ["JOINT-REPORT.md", "audit_joint.py", "joint-review.json", "joint-review-attempt1.log"],
       archive={str(joint/"inputs.pt"):joint/"inputs-attempt1.pt",
                str(joint/"prepare-inputs.json"):joint/"prepare-inputs-attempt1.json"})
freeze("AXIS-API-FROZEN.json", "axis-api-review.json",
       ["AXIS-API-REPORT.md", "audit_axis_api.py", "axis-api-review.json", "axis-api-review-attempt1.log"])
audit = ROOT / "integration/review/ra3-artifact-audit-state-review"
freeze("RA3-ARTIFACT-FROZEN.json", "ra3-graph-review.json",
       ["RA3-ARTIFACT-REPORT.md", "audit_ra3_graph.py", "ra3-graph-review.json",
        "ra3-graph-review-attempt1.log", "ra3-graph-review-attempt2.log"],
       extra_files=[audit/name for name in ("summary.json", "REPORT.md", "AUDITOR-IDENTITY.json", "audit.log")])
