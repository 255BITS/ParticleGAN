"""Prepare unchanged seed-zero research variants from pinned local requests."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .contracts import atomic_json, file_hash, read_json, stable_hash
from .views import task_execution_fingerprint, task_evaluation_fingerprint


def scope_id(source, revision):
    return stable_hash([source, revision, 0])


def variants(root):
    """Use exactly the research cohorts already displayed by research_tiers."""
    from .pr399_scores import publications
    from .test_performance import SUPPLEMENTAL
    from .source_bound_board import BOARD_SOURCE, validate_board
    from .confidence_coherence_scores import validate_original_rows
    board = validate_board(read_json(root / BOARD_SOURCE))
    validate_original_rows(root, board)
    required = sorted({a["task"] for a in board["recorded_view"]["assignments"] if a["importance"] == "required"})
    packets = []
    names = {"pr399-confidence-lr-tier1-6-20261013", "pr399-confidence-lr-own-retention-2-20261013",
             "pr399-gaussian-finite-cap-1-20261013", "pr399-critic-orientation-2-20261013",
             "pr399-signed-confidence-2-20261013", "pr399-positive-evidence-2-20261013"}
    for pin in read_json(root / SUPPLEMENTAL)["publications"]:
        document = read_json(root / pin["path"])
        if document.get("discovery_summary", {}).get("id") in names:
            if file_hash(root / pin["path"]) != pin["sha256"]:
                raise ValueError("research packet changed: " + pin["path"])
            packets.append((pin["path"], document))
    packets += [(path, document) for path, document in publications(root) if "-atlas-" not in path]
    groups = {}
    for path, document in packets:
        for cell in document["cells"]:
            condition = document["condition_contracts"][cell["conditions_sha256"]]
            if condition["seed"] != 0:
                continue
            source, revision = condition["source_digest"], cell["candidate_revision"]
            key = scope_id(source, revision)
            group = groups.setdefault(key, dict(scope=key, source_digest=source, candidate_revision=revision,
                label=cell["label"], required=required, records={}))
            value = dict(status=cell["status"], attempt_id=cell["result"]["attempt_id"], evidence=path,
                         task=deepcopy(condition["task"]), request_sha256=cell["result"].get("request_file_sha256"))
            previous = group["records"].get(cell["task_id"])
            if previous and previous["status"] != value["status"]:
                raise ValueError("conflicting original research task outcomes")
            group["records"].setdefault(cell["task_id"], value)
    for row in board["rows"]:
        key = scope_id(row["source_digest"], row["candidate_revision"])
        groups[key] = dict(scope=key, source_digest=row["source_digest"], candidate_revision=row["candidate_revision"],
            label=row["label"], required=required,
            records={name: dict(status=cell["status"], attempt_id=cell["attempt"], evidence=BOARD_SOURCE.as_posix())
                     for name, cell in row["benchmark_tasks"].items() if cell["status"] is not None})
    return list(groups.values())


def request_index(root, archive_base):
    result = {}
    for uri, pin in read_json(root / "reports/forge/research-catalog/archive-pins.json").items():
        if uri.startswith("archive://") and uri.endswith("/request.json"):
            path = archive_base / uri.removeprefix("archive://")
            result[path.parent.name] = (path, pin["sha256"])
    # The original G-only source and raw artifacts have a separate archive.
    archive = read_json(root / "reports/forge/g-coherent-mean-seed0/artifact-archive.json")
    base = Path(archive["archive_local_path"]).parent
    for relative, pin in archive["included_files"].items():
        if relative.endswith("/request.json"):
            attempt = relative.split("/")[0]
            matches = list((base / "queue").glob(f"*/{attempt}/request.json"))
            if len(matches) == 1:
                result[attempt] = (matches[0], pin["sha256"])
    return result


def _request(path, digest, group):
    if file_hash(path) != digest:
        raise ValueError("original research request bytes changed: " + str(path))
    wrapper = read_json(path)
    request = wrapper.get("request", wrapper)
    if (request["candidate_revision"] != group["candidate_revision"]
            or request["source"]["digest"] != group["source_digest"]
            or request["protocol"]["seed"] != 0):
        raise ValueError("research source/recipe/seed differs from the archived row")
    return request


def prepare(root, directory, *, archive_base=Path("/ml2/codex-storage")):
    from .attempt_coverage import identity
    index = request_index(root, archive_base)
    jobs = {}
    for group in variants(root):
        requests = []
        for record in group["records"].values():
            if record["attempt_id"] not in index:
                raise ValueError("no pinned original request locator: " + record["attempt_id"])
            path, digest = index[record["attempt_id"]]
            requests.append((_request(path, digest, group), path, digest))
        request = deepcopy(requests[0][0])
        source = Path(request["source"]["snapshot_path"])
        if not source.is_dir():
            raise ValueError("original research source snapshot unavailable: " + str(source))
        full = next((parent / "source" for parent in source.parents
                     if (parent / "source/configs/forge/tasks").is_dir()), None)
        cards = {}
        if full:
            for path in sorted((full / "configs/forge/tasks").glob("*.json")):
                card = read_json(path); cards[card["id"]] = card
        original_jobs = {}
        for original, _, _ in requests:
            cards.update(original["tasks"])
            for job in original["jobs"]:
                for task in job.get("task_ids", [job["task_id"]]):
                    original_jobs[task] = job
        target = directory / "research-requests" / (group["scope"] + ".json")
        if target.is_file():
            # Freeze the first source-bound task catalog. An unrelated edit to
            # the archival worktree must not silently schedule a changed test.
            request = read_json(target)
            if (request["candidate_revision"] != group["candidate_revision"]
                    or request["source"]["digest"] != group["source_digest"]):
                raise ValueError("cached research request identity changed")
            cards = request["tasks"]
            original_jobs = {task: job for job in request["jobs"] for task in job.get("task_ids", [job["task_id"]])}
        missing = set(group["required"]) - cards.keys()
        if missing:
            raise ValueError(f"{group['label']}: no original task declarations for {sorted(missing)}")
        request["tasks"] = {name: deepcopy(cards[name]) for name in group["required"]}
        request["jobs"] = list({j["compatibility_key"]: j for j in original_jobs.values()}.values())
        for name, card in request["tasks"].items():
            if name not in original_jobs:
                job = dict(task_id=name, compatibility_key=identity(group["source_digest"], group["candidate_revision"], name,
                    task_execution_fingerprint(card), task_evaluation_fingerprint(card)))
                original_jobs[name] = job; request["jobs"].append(job)
        atomic_json(target, request)
        for name, card in request["tasks"].items():
            execution, evaluation = task_execution_fingerprint(card), task_evaluation_fingerprint(card)
            key = identity(group["source_digest"], group["candidate_revision"], name, execution, evaluation)
            job = dict(key=key, family="research-" + group["scope"][:12], label=group["label"],
                research_scope=group["scope"], candidate_id=request["candidate"]["id"],
                candidate_revision=group["candidate_revision"], source_digest=group["source_digest"],
                task=name, execution_sha256=execution, evaluation_sha256=evaluation, request_path=str(target),
                request_sha256=file_hash(target),
                source_root=str(source), timeout_seconds=card["resources"]["timeout_seconds"], resources=card["resources"],
                dependencies=card.get("dependencies", []), status="PENDING", diagnostic=True)
            record = group["records"].get(name)
            if record:
                path, digest = index[record["attempt_id"]]
                job.update(status=record["status"], attempted=True, origin="retained_research_receipt",
                    evidence=record["evidence"], original_attempt=record["attempt_id"],
                    original_request=str(path), original_request_sha256=digest)
                result_path = path.with_name("result.json")
                if record["status"] == "PASS" and result_path.is_file():
                    result = read_json(result_path)
                    row = next((r for r in result.get("task_results", []) if r["task_id"] == name), None)
                    if row and row.get("gate_status") == "PASS" and result.get("candidate_revision") == group["candidate_revision"]:
                        job["certified_parent"] = dict(attempt_id=record["attempt_id"], result_hash=stable_hash(result),
                            result=row, candidate_revision=group["candidate_revision"],
                            compatibility_key=original_jobs[name]["compatibility_key"])
            jobs[key] = job
    return jobs
