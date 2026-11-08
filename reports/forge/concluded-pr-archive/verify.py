"""Verify retained Git evidence and nonqualifying scope without training or hydration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BASE = "7d9498d32a96bf0d7e61239fea51ee613a844d22"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def verify() -> dict:
    pins = 0
    for number in (320, 360, 395, 397, 398):
        paths = list((ROOT / "reports/forge/records").glob(f"archive-pr{number}-*.json"))
        assert len(paths) == 1, (number, paths)
        record = json.loads(paths[0].read_text())
        assert record["record_type"] == "family_context"
        assert record["evidence_scope"] == "historical"
        assert not record["qualification_input"] and not record["qualification_reuse"]
        assert not record["reuse"] and record["task_results"] == []
        entries = [record["source"], *record["provenance"]["documents"].values()]
        for entry in entries:
            revision, path = entry["archive_commit"], entry["path"]
            content = git("show", f"{revision}:{path}")
            assert hashlib.sha256(content).hexdigest() == entry["sha256"], path
            assert git("rev-parse", f"{revision}:{path}").decode().strip() == entry["git_blob"], path
            assert entry["url"] == f"https://github.com/255BITS/ParticleGAN/blob/{revision}/{path}"
            if path.endswith(".gif"):
                assert content[:6] in (b"GIF87a", b"GIF89a"), path
            pins += 1
    for relative in ("concluded-pr-archive/README.md", "stall-reach/README.md",
                     "bcap-generator-idle/README.md"):
        path = ROOT / "reports/forge" / relative
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (relative, target)
    unchanged = ("reports/forge/technique-inventory.json", "reports/forge/technique-inventory.md",
                 "reports/forge/leaderboards", "reports/forge/families", "reports/forge/automation.json")
    assert not git("diff", "--name-only", BASE, "--", *unchanged), "Published snapshots changed"
    return {"status": "PASS", "nonqualifying_records": 5, "immutable_document_pins": pins,
            "local_links": "PASS", "published_snapshots_unchanged_from": BASE,
            "training_updates_added": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
