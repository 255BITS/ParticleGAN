"""One current generated leaderboard per goal (AGENTS.md), declared in reports/leaderboard-index.json.

Every tracked leaderboard page (a path or H1 title naming a leaderboard) is either a goal's
single current page, a declared view/diagnostic of it, a non-leaderboard, or a superseded page
that is bannered, byte-frozen by a verified manifest pin, or archived in reports/log-archive.json.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = json.loads((ROOT / "reports/leaderboard-index.json").read_text())
INVENTORY = "reports/forge/technique-inventory.md"
LINK = re.compile(r"\]\(([^)\s#]+)")


@pytest.fixture(scope="module")
def tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-s", "-z", "--", "reports", "results"], cwd=ROOT,
                             capture_output=True, check=True).stdout.decode()
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"not a Git checkout: {error}")
    blobs = {}
    for item in filter(None, out.split("\0")):
        meta, path = item.split("\t", 1)
        blobs[path] = meta.split()[1]
    return blobs


def _goals():
    return INDEX["goals"]


def _superseded():
    return [(goal, entry) for goal in _goals() for entry in goal.get("superseded", [])]


def _title(path):
    with open(ROOT / path, errors="replace") as handle:
        for line in handle:
            if line.strip():
                return line.strip()
    return ""


def _is_leaderboard_page(path):
    if not path.endswith(".md") or "/attempts/" in path:
        return False
    return "leaderboard" in path.lower() or bool(re.match(r"#\s.*leaderboard", _title(path), re.I))


def test_every_tracked_leaderboard_page_is_declared_once(tracked):
    declared = []
    for goal in _goals():
        declared += [goal["current"], *goal.get("readouts", []), *goal.get("diagnostics", [])]
        declared += [entry["path"] for entry in goal.get("superseded", [])]
    declared += [entry["path"] for entry in INDEX["not_leaderboards"]]
    assert len(declared) == len(set(declared)), sorted(p for p in declared if declared.count(p) > 1)
    pages = {path for path in tracked if _is_leaderboard_page(path)}
    assert not pages - set(declared), sorted(pages - set(declared))


def test_each_goal_has_one_unbannered_current_page(tracked):
    names = [goal["goal"] for goal in _goals()]
    assert len(names) == len(set(names))
    for goal in _goals():
        assert goal["current"] in tracked, goal["goal"]
        assert INDEX["banner"] not in (ROOT / goal["current"]).read_text(errors="replace"), goal["current"]


def test_bannered_pages_link_their_goal_and_the_inventory(tracked):
    for goal, entry in _superseded():
        if entry["disposition"] != "bannered":
            continue
        path = ROOT / entry["path"]
        assert entry["path"] in tracked, entry["path"]
        text = path.read_text(errors="replace")
        assert INDEX["banner"] in text, entry["path"]
        targets = {(path.parent / link).resolve() for link in LINK.findall(text)}
        for required in {goal["current"], INVENTORY}:
            assert (ROOT / required).resolve() in targets, (entry["path"], required)


def test_frozen_pages_keep_their_pinned_bytes_and_carry_no_banner(tracked):
    for _, entry in _superseded():
        if entry["disposition"] != "frozen":
            continue
        data = (ROOT / entry["path"]).read_bytes()
        assert INDEX["banner"].encode() not in data, entry["path"]
        identities = (hashlib.sha256(data).hexdigest(), tracked[entry["path"]])
        assert entry["pinned_by"], entry["path"]
        for pin in entry["pinned_by"]:
            text = (ROOT / pin).read_text(errors="replace")
            assert any(identity in text for identity in identities), (entry["path"], pin)


def test_archived_pages_are_out_of_the_tree_and_in_the_log_archive(tracked):
    archives = {a["commit"]: {p: blob for p, _, blob in a["files"]}
                for a in json.loads((ROOT / "reports/log-archive.json").read_text())["archives"]}
    for goal in _goals():
        archived = goal.get("archived")
        if not archived:
            continue
        recorded = archives[archived["log_archive_commit"]]
        for path in archived["paths"]:
            assert path not in tracked, path
            assert path in recorded, path


def test_declared_dispositions_are_known():
    for _, entry in _superseded():
        assert entry["disposition"] in {"bannered", "frozen"}, entry


def test_history_pins_keep_original_blobs_and_record_superseded_page_edits(tracked):
    sources = {s["source_id"]: s for s in
               json.loads((ROOT / "reports/forge/history-sources.json").read_text())["sources"]}
    edits = {e["path"]: e for e in
             json.loads((ROOT / "reports/forge/history-pin-edits.json").read_text())["edits"]}
    for path, edit in edits.items():
        source = sources[edit["source_id"]]
        assert source["revision"] == edit["pinned_revision"], path
        assert source["files"][path] == edit["pinned_blob"], path  # pins are never rewritten
        assert edit["current_blob"] == tracked.get(path), path
    pinned = {path: blob for source in sources.values() for path, blob in source["files"].items()}
    touched = [e["path"] for _, e in _superseded() if e["disposition"] == "bannered"]
    touched += [p for goal in _goals() for p in goal.get("archived", {}).get("paths", [])]
    for path in touched:
        if path in pinned and tracked.get(path) != pinned[path]:
            assert path in edits, path


def test_live_protocol_pages_are_not_bannered():
    for protocol in INDEX["protocols"]:
        if protocol["status"] != "live":
            continue
        assert INDEX["banner"] not in (ROOT / protocol["path"]).read_text(), protocol["path"]
        for consumer in protocol["consumers"]:
            assert (ROOT / consumer.split(" ")[0]).exists(), consumer
