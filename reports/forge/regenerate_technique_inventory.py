"""Update the single current technique leaderboard from validated evidence.

Default regeneration uses committed numerical snapshots and compact receipts.
Current selection pins one whole ordinary evidence row per formulation family.
--source-commit independently regrades hydrated original receipts and registers
new measured rows before updating the same leaderboard. No command trains.
--recorded-policy rebuilds existing rows under their exact archived view policy.
--advance-policy explicitly archives an earlier policy before registering new
source evidence under the current view. Archived outcomes are never regraded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.forge.contracts import atomic_text, file_hash, identifier, read_json, stable_hash
from experiments.forge.technique_board import render_markdown, write_report


def _scalars(value):
    """Preserve scalar final summaries and mappings, never tensor/trace arrays."""
    if isinstance(value, dict):
        return {key: _scalars(item) for key, item in value.items()
                if not isinstance(item, list) and key not in {"raw", "applied", "stream", "streams", "state", "states"}}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("summary field is not a scalar or scalar mapping")


def _evaluator_summary(evaluator):
    summary = _scalars(evaluator)
    # Transfer evaluators return one final threshold row per metric, not a
    # trajectory. Store these by metric name while dropping arbitrary arrays.
    metrics = evaluator.get("metrics")
    if isinstance(metrics, list):
        summary["metric_checks"] = {item["metric"]: _scalars(item) for item in metrics
                                    if isinstance(item, dict) and isinstance(item.get("metric"), str)}
    terminal = evaluator.get("terminal_checks")
    if isinstance(terminal, list):
        summary["terminal_summary"] = {
            "checks": len(terminal), "passing_checks": sum(item.get("passed") is True for item in terminal),
            "status_counts": dict(sorted(Counter(item.get("status", "PASS" if item.get("passed") else "FAIL")
                                                    for item in terminal).items()))}
    return summary


def project_receipt(root: Path | str, attempt_id: str) -> dict:
    """Validate an original durable receipt, then project display-only content."""
    root = Path(root).resolve()
    identifier(attempt_id, "attempt")
    directory = root / "reports/forge/attempts" / attempt_id
    paths = {name: directory / f"{name}.json" for name in ("request", "evidence", "result")}
    if not all(path.is_file() for path in paths.values()):
        raise ValueError(f"hydrate original request/evidence/result receipts before publication: {attempt_id}")
    resolved, certificate, result = (read_json(paths[name]) for name in ("request", "evidence", "result"))
    request = resolved.get("request", resolved)
    result_hash = stable_hash(result)
    if certificate.get("result_hash") != result_hash:
        raise ValueError(f"invalid result hash certificate: {attempt_id}")
    if certificate.get("source") != request.get("source"):
        raise ValueError(f"invalid source certificate: {attempt_id}")
    if certificate.get("runtime") != request.get("runtime"):
        raise ValueError(f"invalid runtime certificate: {attempt_id}")
    source = request.get("source", {})
    if "files" in source and stable_hash(source["files"]) != source.get("digest"):
        raise ValueError(f"invalid source manifest digest: {attempt_id}")
    if result.get("candidate_revision") != request.get("candidate_revision"):
        raise ValueError(f"result/request candidate revision mismatch: {attempt_id}")
    if result.get("attempt_id") != attempt_id:
        raise ValueError(f"result/directory attempt identity mismatch: {attempt_id}")
    if result.get("retry_of") != resolved.get("retry_of"):
        raise ValueError(f"result/request retry identity mismatch: {attempt_id}")
    keys = {member: job["compatibility_key"] for job in request.get("jobs", [])
            for member in job.get("task_ids", [job["task_id"]])}
    projected = []
    for row in result.get("task_results", []):
        if row.get("task_id") not in keys or row.get("compatibility_key") != keys[row["task_id"]]:
            raise ValueError(f"task result/request compatibility mismatch: {attempt_id}")
        compact = {key: _scalars(row[key]) for key in (
            "task_id", "compatibility_key", "gate_status", "status", "raw_status", "reason", "metrics", "cost",
            "device", "execution_path", "api_version", "claim_contract") if key in row}
        compact["reasons"] = [reason for reason in row.get("reasons", []) if isinstance(reason, str)]
        compact["sampling"] = {key: _scalars(row.get("evidence", {})[key]) for key in (
            "sampling_contract_version", "sampling_law", "eval_output_noise", "scoring_weights")
                               if key in row.get("evidence", {})}
        compact["evaluator_summary"] = _evaluator_summary(row.get("evaluator_result", {}))
        projected.append(compact)
    return {
        "schema_version": 1, "summary_version": "forge-technique-receipt-summary-v1",
        "evidence_scope": "published_summary", "qualification_reuse": False,
        "qualification_input": False, "certificate_validated": True,
        "note": "Display summary only. Regrading requires the byte-exact archived original receipts; this projection cannot qualify a technique.",
        "attempt_id": attempt_id, "candidate_id": request.get("candidate", {}).get("id"),
        "candidate_revision": request.get("candidate_revision"), "request_id": request.get("request_id", resolved.get("request_id")),
        "campaign_id": request.get("campaign_id"), "runtime": request.get("runtime"),
        "attempt_status": result.get("raw", {}).get("attempt_status"), "cost_owner": result.get("cost_owner"),
        "task_results": projected,
        "provenance": {
            "canonical_result_hash": result_hash, "source_digest": source.get("digest"),
            "source_origin_commit": source.get("origin_commit"),
            "original_files": {name: {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
                               for name, path in paths.items()}}}


def _json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _write_changed(path, content):
    if not path.exists() or path.read_text() != content:
        atomic_text(path, content)


def _publication_provenance(result, summaries):
    """Bind publication identity to recorded science, excluding mutable Git HEAD."""
    origins = {}
    for summary in summaries.values():
        provenance = summary["provenance"]
        if provenance.get("source_origin_commit") is not None:
            origins.setdefault(provenance["source_digest"], set()).add(provenance["source_origin_commit"])
    for row in [*result["rows"], *result.get("configuration_rows", [])]:
        bindings = row.setdefault("bindings", {})
        recorded = sorted(origins.get(bindings.get("source_digest"), set()))
        bindings["source_origin_commit"] = recorded[0] if len(recorded) == 1 else None
        bindings["recorded_source_origin_commits"] = recorded
    upstream = result.get("provenance", {})
    # The upstream board digest includes freshly resolved request metadata such
    # as HEAD. It cannot identify an immutable publication of unchanged science.
    result["provenance"] = {
        key: upstream[key] for key in ("view_sha256", "reducer_sha256") if key in upstream}
    result["provenance"].update(
        publication_reducer_version="forge-technique-publication-v1",
        publication_reducer_sha256=file_hash(Path(__file__)),
        qualified_receipts={attempt: {
            "canonical_result_hash": summary["provenance"]["canonical_result_hash"],
            "source_digest": summary["provenance"]["source_digest"],
            "original_file_sha256": {name: item["sha256"] for name, item in
                                      summary["provenance"]["original_files"].items()}}
            for attempt, summary in sorted(summaries.items())})
    result["provenance"]["input_digest"] = stable_hash(result)


def _resolve_commit(root, source_commit):
    try:
        return subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                        str(source_commit) + "^{commit}"], cwd=root,
                                       text=True, stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"unknown source commit: {source_commit}") from error


def _receipt_manifests(root, commit):
    """Locate validated original science recorded at this exact Git origin."""
    manifests = {}
    for path in sorted((root / "reports/forge/attempts").glob("*/request.json")):
        resolved = read_json(path)
        request = resolved.get("request", resolved)
        source = request.get("source", {})
        if source.get("origin_commit") != commit:
            continue
        project_receipt(root, path.parent.name)
        if not isinstance(source.get("files"), dict):
            raise ValueError("frozen source publication requires the original complete source manifest")
        previous = manifests.setdefault(source["digest"], source)
        if previous["files"] != source["files"]:
            raise ValueError("conflicting original frozen source manifests")
    if not manifests:
        raise ValueError(f"no hydrated original receipts bind source commit {commit}")
    return manifests


def _extract_source(root, commit, destination, manifests):
    # Historical reports dominate this repository's 924 MB checkout. Extract
    # all scientific roots/configs plus explicitly bound support instead.
    top_level = set(subprocess.check_output(["git", "ls-tree", "--name-only", commit], cwd=root, text=True).splitlines())
    roots = {name for name in ("particlegan", "experiments", "benchmarks", "lib", "configs", "docs", "pyproject.toml")
             if name in top_level}
    extras = {name for source in manifests.values() for name in source["files"]
              if name.split("/", 1)[0] not in roots}
    archive = destination.parent / "frozen-source.tar"
    try:
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", "--format=tar", commit, "--", *sorted(roots | extras)],
                           cwd=root, stdout=output, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("source commit cannot reconstruct its recorded scientific files") from error
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not (member.isdir() or member.isfile()):
                raise ValueError("unsafe member in frozen source archive")
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as stream:
                    target.write_bytes(stream.read())
    _overlay_reports(root / "reports", destination / "reports")


def _overlay_reports(original, destination):
    """Expose current knowledge read-only while retaining frozen report helpers."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in original.iterdir():
        target = destination / source.name
        if not target.exists():
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        elif source.is_dir() and target.is_dir():
            _overlay_reports(source, target)


def _publication_rows(result):
    """Retain every independently graded configuration before family selection.

    The frozen registry may predate a study's activation. Its family default
    rows are a display selection, while configuration_rows retain all trials.
    A numerical snapshot must preserve that full roster and its receipt proofs.
    """
    if "configuration_rows" not in result:
        return result["rows"]
    display_fields = {"technique", "selection", "selected_configuration", "alternative_scope"}
    rows, seen = [], {}
    for row in [*result["configuration_rows"], *result["rows"]]:
        identity = (row["candidate_id"], row.get("candidate_revision"), row.get("cohort"))
        science = {key: value for key, value in row.items() if key not in display_fields}
        if identity in seen:
            if seen[identity] != science:
                raise ValueError("conflicting scientific rows for one publication identity")
            continue
        seen[identity] = science
        rows.append(deepcopy(row))
    return rows


def _frozen_report(root, source_commit, *, view_id, execution_backend, temporary):
    commit = _resolve_commit(root, source_commit)
    manifests = _receipt_manifests(root, commit)
    frozen_root = temporary / "source"
    _extract_source(root, commit, frozen_root, manifests)
    contract = temporary / "manifests.json"
    atomic_text(contract, _json_text(manifests))
    prefix = temporary / "inventory"
    # A fresh process is essential: already imported live model/resolver
    # modules cannot grade a reconstructed older checkout faithfully.
    program = """
import json, sys
from pathlib import Path
from experiments.forge.sources import inspect_source
from experiments.forge.technique_board import write_report
root, contract, prefix, goal, backend = sys.argv[1:]
root = Path(root)
manifests = json.loads(Path(contract).read_text())
for expected in manifests.values():
    actual = inspect_source(root, list(expected['files']))
    if actual['digest'] != expected['digest'] or actual['files'] != expected['files']:
        raise ValueError('reconstructed scientific source differs from original receipt manifest')
result = write_report(root, goal, execution_backend=None if backend == 'all' else backend, output_prefix=prefix)
print(json.dumps(result, sort_keys=True))
"""
    environment = {**os.environ, "PYTHONPATH": str(frozen_root), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        output = subprocess.check_output([os.path.abspath(sys.executable), "-c", program, str(frozen_root),
                                          str(contract), str(prefix), view_id, execution_backend or "all"],
                                         cwd=frozen_root, env=environment, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError("frozen independent report failed: " + error.stderr.strip()) from error
    metadata = json.loads(output)
    result = read_json(metadata["json"])
    result["rows"] = _publication_rows(result)
    if "configuration_rows" in result:
        result["snapshot_row_scope"] = "all_configuration_variants"
    allowed = set(manifests)
    if any(row.get("bindings", {}).get("source_digest") not in allowed for row in result["rows"]):
        raise ValueError("frozen report contains a source cohort absent from the validated original receipts")
    return metadata, result, commit, sorted(manifests)


def regenerate(root: Path | str = REPOSITORY_ROOT, *, view_id="discriminator_stability", execution_backend="cuda",
               output_prefix: Path | str = "reports/forge/technique-inventory", source_commit=None) -> dict:
    """Regrade original evidence and publish deterministic compact artifacts."""
    root = Path(root).resolve()
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    published_attempts = set()
    if json_path.exists():
        previous = read_json(json_path)
        published_attempts = {attempt for row in previous.get("rows", []) for attempt in row.get("attempt_ids", [])}
        for attempt in sorted(published_attempts):
            identifier(attempt, "attempt")
            original = root / "reports/forge/attempts" / attempt
            if not all((original / f"{name}.json").is_file() for name in ("request", "evidence", "result")):
                raise ValueError(f"hydrate original receipts referenced by the published report before regeneration: {attempt}")
    # Stage the existing independent reducer before publishing anything. Invalid
    # originals cannot overwrite a previously measured report with a projection.
    with tempfile.TemporaryDirectory(prefix="forge-technique-publication-") as temporary:
        if source_commit is None:
            metadata = write_report(root, view_id, execution_backend=execution_backend,
                                    output_prefix=Path(temporary) / "inventory")
            result = read_json(metadata["json"])
            result["rows"] = _publication_rows(result)
            if "configuration_rows" in result:
                result["snapshot_row_scope"] = "all_configuration_variants"
            commit, frozen_digests = None, []
        else:
            metadata, result, commit, frozen_digests = _frozen_report(
                root, source_commit, view_id=view_id, execution_backend=execution_backend, temporary=Path(temporary))
    attempt_ids = sorted({attempt for row in result["rows"] for attempt in row.get("attempt_ids", [])})
    if published_attempts - set(attempt_ids):
        if source_commit is None:
            raise ValueError("live source no longer retains the published measured cohort; use --source-commit or a different output prefix")
        raise ValueError("frozen regrade no longer retains the published measured cohort; restore the recorded runtime/hardware or use a different output prefix")
    summaries = {attempt: project_receipt(root, attempt) for attempt in attempt_ids}
    result["publication_scope"] = "frozen_source" if commit else "live_current"
    if commit:
        result["frozen_source"] = {"commit": commit, "source_digests": frozen_digests,
                                   "qualifies_latest_checkout": False}
    _publication_provenance(result, summaries)
    repo_prefix = os.path.relpath(root, markdown_path.parent.resolve())
    markdown = render_markdown(result, json_link=json_path.name, repo_link_prefix=repo_prefix)
    if commit:
        markdown = markdown.replace("exact current cohort", "exact recorded cohort").replace(
            "Current rows bind", "Recorded rows bind")
        first, rest = markdown.split("\n", 1)
        markdown = first + f"\n\nThis report grades the **frozen source cohort `{commit}`** reconstructed from Git and " \
            "verified against original receipt manifests. Its outcomes do not qualify the latest checkout.\n" + rest
    for attempt in attempt_ids:
        markdown = markdown.replace(f"{repo_prefix}/reports/forge/attempts/{attempt}/result.json",
                                    f"{repo_prefix}/reports/forge/technique-receipts/{attempt}.json")
    prefix_argument = os.path.relpath(prefix, root)
    command = ("python reports/forge/regenerate_technique_inventory.py --root . --goal " + shlex.quote(view_id) +
               (" --device " + execution_backend if execution_backend else " --device all") +
               " --output-prefix " + shlex.quote(prefix_argument) +
               (" --source-commit " + commit if commit else ""))
    lines = markdown.splitlines()
    lines = [command if line.startswith("python -m experiments.forge techniques ") else line for line in lines]
    markdown = "\n".join(lines) + "\n\nPublished receipt summaries retain final metrics, gate outcomes and original file hashes. " \
        "They are display artifacts and supply no qualification input. Hydrate byte-exact original " \
        "request/evidence/result receipts from the artifact archive before a full independent regrade.\n"
    for attempt, summary in summaries.items():
        _write_changed(root / "reports/forge/technique-receipts" / f"{attempt}.json", _json_text(summary))
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    return {**metadata, "report": str(markdown_path), "json": str(json_path),
            "input_digest": result["provenance"]["input_digest"],
            "publication_scope": result["publication_scope"], "source_commit": commit,
            "summary_receipts": len(summaries), "qualification_reuse": False}


def _published_report(root, path):
    """Read an intact display publication; composing never regrades evidence."""
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    report = read_json(path)
    if report.get("publication_scope") not in {"live_current", "frozen_source", "recorded_cohort_composition"}:
        raise ValueError("composition requires an independently published current or frozen report")
    copy = deepcopy(report)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("published report input digest mismatch")
    markdown = path.with_suffix(".md")
    if not markdown.is_file():
        raise ValueError("published report Markdown companion is missing")
    return report, {"json": path, "markdown": markdown,
                    "json_sha256": file_hash(path), "markdown_sha256": file_hash(markdown)}


def _validate_published_row(root, report, row, visited=None):
    """Bind each measured display row to its already-published receipt proofs."""
    if report.get("publication_scope") == "recorded_cohort_composition":
        pointer = report.get("source_publications", {}).get(row.get("publication_key"))
        if not isinstance(pointer, dict):
            raise ValueError("composed row has no source publication")
        path = (root / pointer["json"]).resolve()
        visited = set(visited or ())
        if path in visited:
            raise ValueError("cyclic source publication composition")
        visited.add(path)
        source, paths = _published_report(root, path)
        if any(pointer.get(name + "_sha256") != paths[name + "_sha256"]
               for name in ("json", "markdown")) or pointer.get("input_digest") != source["provenance"]["input_digest"]:
            raise ValueError("composed source publication hash mismatch")
        identities = ("candidate_id", "candidate_revision", "cohort")
        matches = [item for item in source.get("rows", [])
                   if all(item.get(key) == row.get(key) for key in identities)]
        if len(matches) != 1:
            raise ValueError("composed row has no unique original publication row")
        # Labels and display flags can change; scientific row contents cannot.
        display_fields = {"publication_key", "qualification_reuse", "qualification_input", "technique"}
        science = lambda item: {key: value for key, value in item.items() if key not in display_fields}
        if science(row) != science(matches[0]):
            raise ValueError("composed row differs from its original scientific publication")
        _validate_published_row(root, source, matches[0], visited)
        return
    proofs = report.get("provenance", {}).get("qualified_receipts", {})
    for attempt in row.get("attempt_ids", []):
        identifier(attempt, "attempt")
        proof = proofs.get(attempt)
        path = root / "reports/forge/technique-receipts" / f"{attempt}.json"
        if not isinstance(proof, dict) or not path.is_file():
            raise ValueError("measured publication row lacks its compact receipt proof")
        summary = read_json(path)
        provenance = summary.get("provenance", {})
        actual = {"canonical_result_hash": provenance.get("canonical_result_hash"),
                  "source_digest": provenance.get("source_digest"),
                  "original_file_sha256": {name: item.get("sha256") for name, item in
                                            provenance.get("original_files", {}).items()}}
        if actual != proof or summary.get("certificate_validated") is not True:
            raise ValueError("compact receipt proof differs from its publication")
        if (summary.get("qualification_input") is not False or summary.get("qualification_reuse") is not False
                or summary.get("candidate_id") != row.get("candidate_id")
                or summary.get("candidate_revision") != row.get("candidate_revision")
                or provenance.get("source_digest") != row.get("bindings", {}).get("source_digest")):
            raise ValueError("compact receipt candidate/source cohort differs from its publication row")
    for tier, required in report.get("tier_requirements", {}).items():
        cell = row.get("tiers", {}).get(tier, {})
        if (cell.get("total") != len(required) or type(cell.get("passed")) is not int
                or not 0 <= cell["passed"] <= cell["total"]):
            raise ValueError("published row changed a declared tier denominator")


def _compose_markdown(result, paths, root, markdown_path, original_path, current_path, candidate_id, prefix):
    def link(path):
        return os.path.relpath(path, markdown_path.parent)

    def cell(value):
        return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")

    tiers = list(result["tier_requirements"])
    lines = ["# Forge technique inventory: recorded source cohorts", "",
             "Each cell is **passes / full required total** in its row's recorded source and runtime. "
             "The retained inventory and appended technique keep their separate evidence identities.", "",
             "| Technique | Recorded source | Exact revision / cohort | Compute | " +
             " | ".join(f"Tier {tier}" for tier in tiers) + " | Recorded tier | Other outcomes | Paid seconds |",
             "| --- | --- | --- | --- | " + " | ".join("---:" for _ in tiers) + " | ---: | --- | ---: |"]
    for row in result["rows"]:
        source = row["bindings"]["source_digest"]
        pointer = paths[row["publication_key"]]
        source_link = f"[`{source[:12]}`]({link(pointer['markdown'])})"
        runtime = row.get("runtime_cohort", {})
        models = sorted({profile.get("model") for profile in runtime.get("compute_profiles", {}).values()
                         if profile and profile.get("model")})
        compute = runtime.get("execution_backend", "unrecorded") + (" / " + ", ".join(models) if models else "")
        counts = Counter(task["status"] for task in row.get("tasks", []) if task["status"] != "PASS")
        others = ", ".join(f"{status} {count}" for status, count in sorted(counts.items())) or "all required tasks PASS"
        seconds = row.get("cost", {}).get("wall_seconds")
        values = [row["technique"], source_link,
                  f"{row['candidate_revision'][:12]} / {row['cohort'][:12]}", compute,
                  *[f"{row['tiers'][tier]['passed']}/{row['tiers'][tier]['total']}" for tier in tiers],
                  row["qualified_tier"], others, round(seconds, 3) if seconds is not None else "unknown"]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "Recorded tiers describe each independently graded publication. This display does not pool passes, "
              "rank across source cohorts, or qualify the latest checkout. Recipes, priors, initialization, budgets, "
              "clean/noisy sampling and hardware remain bound to their original rows.", "",
              "UNKNOWN means unmeasured or unrun evidence. Failed or blocked prerequisite gates stop further work; "
              "every declared task remains in its tier denominator.", "",
              f"[Previous recorded inventory]({link(paths['original']['markdown'])}) · "
              f"[Full current inventory, including all technique denominator rows]({link(paths['current']['markdown'])}) · "
              f"[Exact row bindings and publication hashes]({markdown_path.with_suffix('.json').name})", "",
              "Regenerate this display from the committed publications without launching training or requiring raw receipt hydration:",
              "", "```sh",
              "python reports/forge/regenerate_technique_inventory.py --root . --compose-original " +
              shlex.quote(os.path.relpath(original_path, root)) + " --compose-current " +
              shlex.quote(os.path.relpath(current_path, root)) + " --append-candidate " + shlex.quote(candidate_id) +
              " --output-prefix " + shlex.quote(os.path.relpath(prefix, root)), "```", "",
              f"Publication input digest `{result['provenance']['input_digest']}`. "
              "Full independent regrading uses each linked publication's own regeneration command and byte-exact original receipts.", ""]
    return "\n".join(lines)


def compose(root: Path | str = REPOSITORY_ROOT, *, original_report="reports/forge/technique-inventory.json",
            current_report="reports/forge/r3gan-technique-inventory.json",
            candidate_id="r3gan-stacked-training-toy-v1",
            output_prefix="reports/forge/technique-inventory-expanded") -> dict:
    """Append one new technique to recorded rows without pooling qualification.

    Source publications are immutable inputs. This display can regenerate in a
    fresh checkout using compact receipt proofs; it supplies no gate evidence.
    """
    root = Path(root).resolve()
    identifier(candidate_id, "candidate")
    original, original_paths = _published_report(root, original_report)
    current, current_paths = _published_report(root, current_report)
    if original.get("publication_scope") not in {"frozen_source", "recorded_cohort_composition"}:
        raise ValueError("original publication must preserve a frozen source cohort")
    # Earlier compositions recorded runtime on every row but omitted this field.
    def backend(report):
        if report.get("execution_backend") is not None:
            return report["execution_backend"]
        cohorts = {row.get("runtime_cohort", {}).get("execution_backend") for row in report.get("rows", [])}
        return next(iter(cohorts)) if len(cohorts) == 1 else None
    for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements", "execution_backend"):
        left, right = (backend(original), backend(current)) if key == "execution_backend" else (original.get(key), current.get(key))
        if left != right:
            raise ValueError(f"source publications have incompatible {key}; keep them in separate reports")
    if any(row.get("candidate_id") == candidate_id for row in original.get("rows", [])):
        raise ValueError("appended technique is already present in the original publication")
    additions = [row for row in current.get("rows", []) if row.get("candidate_id") == candidate_id]
    if len(additions) != 1:
        raise ValueError("current publication must contain exactly one requested technique/runtime cohort")
    current_candidates = {row.get("candidate_id") for row in current.get("rows", [])}
    if any(row.get("candidate_id") not in current_candidates for row in original.get("rows", [])):
        raise ValueError("current publication must retain every original technique denominator row")
    prefix = Path(output_prefix)
    if not prefix.is_absolute():
        prefix = root / prefix
    json_path, markdown_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".md")
    paths = {"original": original_paths, "current": current_paths}
    protected = {pointer[name] for pointer in paths.values() for name in ("json", "markdown")}
    if json_path.resolve() in protected or markdown_path.resolve() in protected:
        raise ValueError("composition output cannot overwrite an input publication")
    rows = []
    for key, report, selected in (("original", original, original["rows"]), ("current", current, additions)):
        for original_row in selected:
            _validate_published_row(root, report, original_row)
            if not original_row.get("bindings", {}).get("source_digest"):
                raise ValueError("composition row is missing its recorded source identity")
            row = deepcopy(original_row)
            row.update(publication_key=key, qualification_reuse=False, qualification_input=False)
            if key == "current" and candidate_id == "r3gan-stacked-training-toy-v1":
                row["technique"] = "R3GAN Stacked-MNIST recipe (toy-host adaptation)"
            rows.append(row)
    result = {"schema_version": 1, "reducer_version": "forge-technique-composition-v1",
              "publication_scope": "recorded_cohort_composition", "qualification_reuse": False,
              "qualification_input": False, "view": original["view"], "view_revision": original["view_revision"],
              "policy_fingerprint": original["policy_fingerprint"],
              "execution_backend": backend(original),
              "tier_requirements": deepcopy(original["tier_requirements"]), "rows": rows,
              "source_publications": {key: {
                  "json": os.path.relpath(pointer["json"], root), "markdown": os.path.relpath(pointer["markdown"], root),
                  "json_sha256": pointer["json_sha256"], "markdown_sha256": pointer["markdown_sha256"],
                  "publication_scope": report["publication_scope"],
                  "input_digest": report["provenance"]["input_digest"], "frozen_source": report.get("frozen_source")}
                  for key, report, pointer in (("original", original, original_paths), ("current", current, current_paths))}}
    for catalog in ("recipe_contracts", "protocol_contracts", "task_contracts", "status_reasons"):
        combined = {}
        for report in (original, current):
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract):
                    raise ValueError(f"invalid {catalog} identity in input publication")
                if digest in combined and combined[digest] != contract:
                    raise ValueError("conflicting scientific contract identities between publications")
                combined[digest] = deepcopy(contract)
        result[catalog] = dict(sorted(combined.items()))
    result["provenance"] = {"publication_reducer_sha256": file_hash(Path(__file__)),
                            "input_publication_sha256": {key: pointer["json_sha256"] for key, pointer in paths.items()},
                            "selected_rows_sha256": stable_hash(rows)}
    result["provenance"]["input_digest"] = stable_hash(result)
    markdown = _compose_markdown(result, paths, root, markdown_path, original_paths["json"],
                                 current_paths["json"], candidate_id, prefix)
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(rows),
            "input_digest": result["provenance"]["input_digest"], "publication_scope": result["publication_scope"],
            "qualification_reuse": False}


EVIDENCE_MANIFEST = Path("reports/forge/technique-evidence/manifest.json")
CURRENT_PREFIX = Path("reports/forge/technique-inventory")
CONTRACT_CATALOGS = ("recipe_contracts", "protocol_contracts", "task_contracts", "status_reasons")
POLICY_FIELDS = ("view", "view_revision", "policy_fingerprint", "tier_requirements")


def _snapshot(root, entry, manifest):
    """Numerical evidence needs its own digest/proofs, not another Markdown board."""
    path = root / entry["snapshot"]
    if file_hash(path) != entry["json_sha256"]:
        raise ValueError("technique evidence snapshot hash mismatch")
    report = read_json(path)
    copy = deepcopy(report)
    claimed = copy.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(copy):
        raise ValueError("technique evidence input digest mismatch")
    if report.get("publication_scope") != "frozen_source":
        raise ValueError("technique evidence must retain a frozen source cohort")
    if report.get("frozen_source", {}).get("commit") != entry["source_commit"]:
        raise ValueError("technique evidence source commit mismatch")
    for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
        if manifest[key] != report.get(key):
            raise ValueError(f"technique evidence has incompatible {key}")
    selected = {}
    for name, recorded_at in entry["candidates"].items():
        matches = [row for row in report["rows"] if row.get("candidate_id") == name
                   and (bool(row.get("attempt_ids")) if isinstance(recorded_at, str) else True)]
        if len(matches) != 1:
            raise ValueError("technique evidence needs one exact candidate/runtime row")
        row = matches[0]
        if row.get("attempt_ids") and not isinstance(recorded_at, str):
            raise ValueError("measured technique evidence needs its recorded completion time")
        _validate_published_row(root, report, row)
        selected[name] = row
    return report, selected


def _archived_reports(root, manifest):
    """Validate historical policies without admitting their rows to selection."""
    reports, fingerprints = [], {manifest["policy_fingerprint"]}
    for policy in manifest.get("archived_policies", []):
        if (policy.get("view") != manifest["view"] or
                not isinstance(policy.get("view_revision"), int) or
                policy["view_revision"] >= manifest["view_revision"] or
                policy.get("policy_fingerprint") in fingerprints):
            raise ValueError("invalid archived technique evidence policy")
        fingerprints.add(policy["policy_fingerprint"])
        reports.extend((policy, entry, *_snapshot(root, entry, policy)) for entry in policy["cohorts"])
    return reports


def _current_tier_cell(row, tier, required, json_link):
    """Display recorded states without turning missing/blocked work into FAIL."""
    summary = row["tiers"][tier]
    passed, total = summary["passed"], summary["total"]
    counts = summary.get("counts")
    if counts is None:
        # Some older snapshots retain only the aggregate and task statuses.
        # Unknown tasks stay in the denominator; the aggregate is not regraded.
        required = set(required)
        counts = Counter(task["status"] for task in row.get("tasks", [])
                         if task["task_id"] in required and task["status"] != "PASS")
        counts["PASS"] = passed
        remaining = total - sum(counts.values())
        if remaining >= 0:
            counts["UNKNOWN"] += remaining
    counts = {status: count for status, count in counts.items() if count}
    if sum(counts.values()) != total or counts.get("PASS", 0) != passed:
        # An incomplete legacy breakdown cannot establish an all-failed state.
        return f"{passed}/{total}<br>[{total} required; task statuses]({json_link})"
    if len(counts) == 1:
        status = next(iter(counts))
        if status == "PASS":
            return f"{passed}/{total} PASS"
        if status != "FAIL":
            return f"{status.replace('_', ' ')} ({total} required)"
    if not passed and set(counts) <= {"UNKNOWN", "NOT_RUN"}:
        return f"NOT RUN/UNKNOWN ({total} required)"
    other = " · ".join(f"{status.replace('_', ' ')} {count}"
                       for status, count in sorted(counts.items()) if status != "PASS")
    return f"{passed}/{total}" + (f"<br>{other}" if other else "")


ATLAS_PROGRESS_INPUTS = {
    "baseline": ("atlas-current-gpu-diagnostics-native-v2-20261003/summary.json",
                 "9976a462fa3c8fbbb33bcdb4dd6b7697c65167f7c5a9c2bdf2c8ed6f6f5cd488",
                 "particlegan_atlas_current_gpu_native_v2_publication_v1"),
    "adaptations_v3": ("atlas-named-gpu-diagnostics-native-v3-20261003/results.json",
                       "8ba7230a71041e3a0185528c9e76c5f39bff6eeadfb98b59d715ef72ff76b9e0",
                       "particlegan_atlas_named_gpu_diagnostics_publication_v3"),
    "adaptations_v4b": ("atlas-named-gpu-diagnostics-native-v4b-20261004/results.json",
                        "0a175519b59b5180078d9669519060b79dc30d596daca3c8be1aca2dd9ce1d5f",
                        "particlegan_atlas_named_gpu_diagnostics_publication_v4b"),
    "word_context": ("atlas-word-retained-context-20261004/context.json",
                     "54d61041c72befd8b4196f78d368374ac1c98470fd5742d4772f9cb6da5bcbdd",
                     "pg_atlas_word_retained_context_v1"),
}

# A separate, accepted numerical diagnostic, bound to its immutable passive
# export. The earlier C6 INVALID remains a different row.
WORD_HALF_BASE_INPUT = ("word-half-base-20261004/results.json",
                       "d2bc198d65c935ca31160505510b8efc7aa2cf9e5f62ac6429f758173e7ca8b7",
                       "pg_word_half_base_publication_v1")
WORD_HALF_BASE_READOUT_SHA256 = "08db1485b912d00558af0f3496dff6332c9b27d2cb13dfca68496f8a90d54bbe"
WORD_HALF_BASE_RECIPE_SHA256 = "ea470cca5c55a31e6f726945402f0dfad945fea2ec95b654afba922e93c8ac69"
WORD_HALF_BASE_GIF_SHA256 = "2d94247b902d6384ef172faa99db0ae4d655bd0171ae6a4ef4d77eaadbc1012b"


def _word_half_base_diagnostic(root):
    """Read one sealed completed export; never grade, sample or select a winner."""
    root = Path(root)
    relative, digest, schema = WORD_HALF_BASE_INPUT
    report_path = root / "reports/forge" / relative
    if not report_path.exists():
        if report_path.parent.exists():
            raise ValueError("half_base retained report is unavailable")
        return {}
    report = read_json(report_path)
    if file_hash(report_path) != digest or report.get("schema") != schema:
        raise ValueError("half_base display requires the exact retained report")
    cohort = "word_joint_policy_min11_rates_v1"
    task = "five_word_joint_acquisition_" + cohort
    flags = ("qualification_input", "ordinary_tier_credit", "historical_credit",
             "default_adoption", "speed_ranking", "cross_tuple_pooling")
    if any(report.get(flag) is not False for flag in flags):
        raise ValueError("half_base display cannot confer qualification, default or speed credit")
    if (report.get("candidate_id") != "word-min11-half_base-rates-v1"
            or report.get("profile") != "half_base"
            or report.get("family") != "atlas_word_joint_min11_rates"
            or report.get("task_cohort") != cohort or report.get("task_id") != task
            or report.get("parent_task_id") != "five_word_joint_acquisition"
            or report.get("status") != "COMPLETE" or report.get("accepted_numeric") != "FAIL"):
        raise ValueError("half_base configuration or accepted numerical status changed")
    parents = ("two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition",
               "five_word_joint_acquisition", "trajectory", "residual_student", "unipolar",
               "cover_leftover", "mid_scale_identity", "mode_hold", "vector_two_broad",
               "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
               "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2",
               "grid100", "rotated100", "staggered100", "ring_hold", "ring_extension")
    slots = {task if parent == "five_word_joint_acquisition" else parent:
             {"status": "FAIL" if parent == "five_word_joint_acquisition" else "NOT_RUN"}
             for parent in parents}
    if (report.get("slots") != slots or report.get("counts") != {"FAIL": 1, "NOT_RUN": 25}
            or report.get("required_slots") != 26 or report.get("scheduled_slots") != 1
            or report.get("not_run_slots") != 25 or report.get("tiers") != {"1": 5, "2": 19, "3": 2}):
        raise ValueError("half_base separate 26-slot denominator changed")
    source = report.get("source", {})
    if source != dict(origin_commit="f9f7ed9d7a06c48d4ec56999107658983d7e8efc",
                      digest="5995590d3c303207c664fe3b0ba7dc1e09dd3da7a90abf87086634c789175026",
                      files=1444,
                      supervisor_sha256="dae842935a1ea3c4c7aae1694cb91e2f6c8b55815095f7230e64c7dc0ec368a1"):
        raise ValueError("half_base executed source identity changed")
    recipe = report.get("resolved_recipe", {})
    overrides = dict(lr=.00265625, prior_lr_mult=1.5, d_lr_mult=1.)
    if (report.get("recipe_overrides") != overrides
            or report.get("resolved_recipe_sha256") != WORD_HALF_BASE_RECIPE_SHA256
            or stable_hash(recipe) != WORD_HALF_BASE_RECIPE_SHA256
            or any(recipe.get(key) != value for key, value in overrides.items())
            or recipe.get("prior_kind") != "particles" or recipe.get("sigma_rel") != 0.0
            or recipe.get("standardize") is not False or recipe.get("encoder_mode") != "none"):
        raise ValueError("half_base actual Recipe or raw particle representation changed")
    steps = [(i * 20001 + 23) // 24 for i in range(1, 25)]
    protocol = report.get("protocol", {})
    if (protocol.get("resources") != dict(num_particles=11, z_dim=2, batch_size=256)
            or protocol.get("seed") != 0 or protocol.get("updates") != 20001
            or protocol.get("schedule_horizon") != 20000 or protocol.get("eval_samples") != 1024
            or protocol.get("metric_steps") != steps or protocol.get("terminal_steps") != steps[-5:]
            or protocol.get("thresholds") != [["sample_count", ">=", 1024], ["quality_fraction", ">=", .95],
                ["modes", "==", 5], ["mass_tv", "<=", .1], ["reconstruction_exact", "==", 1],
                ["minimum_reconstruction_token_probability", ">=", .9]]
            or protocol.get("serving") != "actual policy-selected G/E/raw ParticlePrior; DV12 retained, output noise off"
            or protocol.get("objective") != "original joint RpGAN plus original regularizers; free continuous E; no reconstruction training loss"):
        raise ValueError("half_base original word protocol or selected serving law changed")
    result = report.get("result", {})
    owners = sorted(("continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
                     "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard"))
    hooks = ("begin_step", "after_critic_step", "after_generator_backward", "after_generator_step", "finish_step")
    summary = result.get("original_grader_summary", {})
    if (result.get("original_gate") != "FAIL" or result.get("completed_steps") != 20001
            or result.get("policy_owner") != "particlegan.UpdatePolicy"
            or result.get("optimizer_updates") != {key: 20001 for key in ("generator", "encoder", "prior", "discriminator")}
            or result.get("lifecycle_calls") != {key: 20001 for key in hooks}
            or result.get("requested_owners") != owners or result.get("enabled_owners") != owners
            or result.get("actual_birth_death") != dict(rows=11, neighbours=5, reference_half=6, isolation=True)
            or result.get("metric_steps") != steps or result.get("terminal_steps") != steps[-5:]
            or result.get("selected_source") not in {"fast", "averaged"}
            or summary.get("complete") is not True or summary.get("observations") != 24
            or summary.get("minimum_stable_checks") != 5 or summary.get("passing_observations") != 0
            or summary.get("passing_suffix") != 0 or any(result.get(flag) is not False for flag in flags)):
        raise ValueError("half_base full public ownership, clocks or numerical decision changed")
    metrics = result.get("final_metrics", {})
    if (not {"sample_count", "step", "quality_fraction", "modes", "mass_tv", "reconstruction_exact",
             "minimum_reconstruction_token_probability"}.issubset(metrics)
            or metrics.get("step") != 20001 or metrics.get("sample_count") != 1024
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values())
            or result.get("independent_atlas_qualification") is not False
            or result.get("synthetic_mechanism_probes_are_learning_credit") is not False):
        raise ValueError("half_base endpoint metrics must retain scalar numbers")
    historical = report.get("historical_word", {})
    if (historical.get("status") != "INVALID" or historical.get("accepted_numeric") != "UNAVAILABLE"
            or historical.get("paid_seconds") != 558.5739127129782
            or historical.get("already_in_prior_total") is not True or historical.get("recertified") is not False):
        raise ValueError("half_base display cannot replace the original C6 INVALID")
    cost = report.get("cost", {})
    expected_cost = dict(paid_seconds=556.4301753160544, reserved_seconds=0., charged_seconds=556.4301753160544,
                         overrun_seconds=0., prior_charged_seconds=910.2391431590077,
                         inclusive_charged_seconds=1466.669318475062, original_cap_seconds=10500,
                         allowance_seconds=900, export_grace_seconds=0)
    lanes = {"0": dict(prior_charged_seconds=234.82608077581972, current_charged_seconds=0.,
                       inclusive_charged_seconds=234.82608077581972, cap_seconds=7500),
             "1": dict(prior_charged_seconds=675.4130623831879, current_charged_seconds=556.4301753160544,
                       inclusive_charged_seconds=1231.8432376992423, cap_seconds=3000)}
    if (any(cost.get(key) != value for key, value in expected_cost.items()) or cost.get("lanes") != lanes
            or cost.get("paid_vs_reserve_separate") is not True or cost.get("speed_comparison_available") is not False):
        raise ValueError("half_base inclusive cost or unchanged lane ceilings changed")
    media = result.get("actual_goal_gif", {})
    media_steps = [steps[i] for i in (0, 3, 6, 9, 12, 14, 17, 20, 23)]
    if (media.get("path") != "media/goal.gif" or media.get("frames") != 9
            or media.get("actual_steps") != media_steps or media.get("sha256") != WORD_HALF_BASE_GIF_SHA256):
        raise ValueError("half_base accepted goal media identity changed")
    gif_path = report_path.parent / media["path"]
    readout = report_path.with_name("README.md")
    if (not gif_path.is_file() or gif_path.stat().st_size != media.get("bytes")
            or file_hash(gif_path) != media["sha256"] or not readout.is_file()
            or file_hash(readout) != WORD_HALF_BASE_READOUT_SHA256):
        raise ValueError("half_base retained goal media or readout bytes changed")
    return dict(schema_version=1, scope="separate_named_configuration_diagnostic",
                candidate_id=report["candidate_id"], profile=report["profile"], family=report["family"],
                task_cohort=cohort, task_id=task, status="COMPLETE", numerical_gate="FAIL",
                required=26, passed=0, counts=report["counts"], source=source,
                recipe_overrides=overrides, resolved_recipe=recipe, resolved_recipe_sha256=WORD_HALF_BASE_RECIPE_SHA256,
                protocol=protocol, result=result, historical_word=historical, cost=cost,
                representation=dict(label="Particles (N11 joint cloud; free encoder)",
                    prior_kind=recipe["prior_kind"], sigma_rel=recipe["sigma_rel"], standardize=recipe["standardize"],
                    actual_prior_rows=11, canonical_target_words=5, encoder_mode=recipe["encoder_mode"]),
                media=dict(media, path=gif_path.relative_to(root).as_posix()),
                readout=readout.relative_to(root).as_posix(),
                input=dict(report=report_path.relative_to(root).as_posix(), sha256=digest,
                           readout_sha256=WORD_HALF_BASE_READOUT_SHA256),
                accounting=dict(inclusive_charged_seconds=cost["inclusive_charged_seconds"],
                    gpu0_charged_seconds=lanes["0"]["inclusive_charged_seconds"],
                    gpu1_charged_seconds=lanes["1"]["inclusive_charged_seconds"]),
                qualification_input=False, qualification_reuse=False, ordinary_tier_credit=False,
                cross_cohort_pooling=False, default_adoption=False, speed_ranking=False)


def _representation_kind(prior):
    """Read the declared law; family names are not representation evidence."""
    kind = (prior or {}).get("kind")
    return {"mog": "MoG", "particle_cloud": "Particles", "particles": "Particles"}.get(kind, "UNKNOWN")


def _ordinary_representation(result, row):
    reference = _representation_kind(row.get("bindings", {}).get("prior"))
    if reference != "MoG":
        return reference
    contracts = result.get("task_contracts", {})
    task_kinds = {_representation_kind(contracts.get(digest, {}).get("prior"))
                  for digest in row.get("bindings", {}).get("task_contracts", {}).values()}
    return "MoG + particles (per task)" if "Particles" in task_kinds else reference


def _atlas_unblocking_progress(root):
    """Project separate source-scoped model scores, never qualification rows."""
    directory = Path(root) / "reports/forge"
    paths = {key: directory / values[0] for key, values in ATLAS_PROGRESS_INPUTS.items()}
    if not any(path.exists() for path in paths.values()):
        return {}
    reports, inputs = {}, {}
    for key, (relative, digest, schema) in ATLAS_PROGRESS_INPUTS.items():
        report = read_json(paths[key])
        if file_hash(paths[key]) != digest or report.get("schema") != schema:
            raise ValueError("Atlas progress requires the exact retained report: " + key)
        if any(report.get(flag) is not False for flag in (
                "qualification_input", "ordinary_tier_credit", "default_adoption", "speed_ranking",
                "cross_cohort_pooling")):
            raise ValueError("Atlas progress cannot confer qualification, default or speed credit")
        readout = paths[key].with_name("README.md")
        if not readout.is_file():
            raise ValueError("Atlas progress readout is unavailable: " + key)
        inputs[key] = dict(report=(Path("reports/forge") / relative).as_posix(), sha256=digest,
                           readout=readout.relative_to(root).as_posix(), readout_sha256=file_hash(readout))
        reports[key] = report
    baseline = reports["baseline"]
    counts = dict(Counter(case["diagnostic_status"] for case in baseline["cases"]))
    if (counts != {"PASS": 7, "FAIL": 11, "BLOCKED": 8} or baseline["counts"]["required"] != 26
            or len({case["parent_id"] for case in baseline["cases"]}) != 26
            or {tier: row["required"] for tier, row in baseline["tier_counts"].items()}
               != {"1": 5, "2": 19, "3": 2}):
        raise ValueError("Atlas baseline diagnostic denominator changed")
    executed = [case for case in baseline["cases"] if case["diagnostic_status"] in {"PASS", "FAIL"}]
    if (len(executed) != 18 or any(
            _representation_kind(case["applied"].get("prior")) != "Particles"
            or case["applied"]["prior"].get("sigma") != 0.0
            or case["applied_recipe"].get("prior_kind") != "particles"
            or case["applied_recipe"].get("standardize") is not False
            or case["applied_recipe"].get("sigma_rel") != 0.0 for case in executed)):
        raise ValueError("Atlas baseline representation requires its actual applied particle law")
    baseline_representation = dict(label="Particles", evidence_source="actual applied priors and complete Recipes",
        report=inputs["baseline"]["report"], report_sha256=inputs["baseline"]["sha256"],
        cases={case["id"]: dict(prior=case["applied"]["prior"],
                               applied_recipe_sha256=case["applied_recipe_sha256"],
                               frozen_definition=case["frozen_definition"])
               for case in executed})
    for key in ("adaptations_v3", "adaptations_v4b"):
        report = reports[key]
        if (report["counts"]["required_family_cells"] != 130
                or report["counts"]["required_slots_per_family"] != 26 or len(report["families"]) != 5
                or any(len(family["slots"]) != 26
                       or dict(Counter(slot["diagnostic_status"] for slot in family["slots"].values()))
                          != family["counts"] for family in report["families"].values())):
            raise ValueError("Atlas adaptation family denominators changed")
    groups = (
        ("Conditional", "atlas_conditional", "adaptations_v3", ["trajectory", "residual_student", "unipolar", "mid_scale_identity"],
         "conditional_policy_selected_cloud_v1"),
        ("AE routed", "atlas_ae_routed", "adaptations_v3", ["ae_gan_hold"], "ae_routed_policy_v1"),
        ("Unused token", "atlas_routed", "adaptations_v4b", ["unused_token_hold"], "routed_policy_selected_cloud_v1"),
        ("Cover leftover", "atlas_multibank", "adaptations_v4b", ["cover_leftover"], "multibank_policy_v1"),
        ("Word joint N11", "atlas_word_joint_min11", "adaptations_v4b", ["five_word_joint_acquisition"], "word_joint_policy_min11_v1"),
    )
    rows = []
    for label, family, key, parents, cohort in groups:
        report = reports[key]
        family_report = report["families"][family]
        required = [parent + "_" + cohort for parent in parents]
        statuses = [family_report["slots"][task]["diagnostic_status"] for task in required]
        declarations = {task: report["questions"][task] for task in required}
        kinds = {_representation_kind(question["prior"]) for question in declarations.values()}
        if len(kinds) != 1 or "UNKNOWN" in kinds:
            raise ValueError("Named model representation requires its declared question laws")
        representation = next(iter(kinds))
        if family == "atlas_ae_routed":
            representation = "MoG (fixed σ .025; routed AE)"
        elif family == "atlas_conditional":
            representation = "Particles (conditional clouds / role banks)"
        elif family == "atlas_routed":
            representation = "Particles (shared / slot parameter bank)"
        elif family == "atlas_multibank":
            representation = "Particles (two routed clouds)"
        else:
            representation = "Particles (N11 joint cloud; free encoder)"
        media, applied = {}, {}
        for attempt in family_report["attempts"]:
            outcome = attempt.get("outcome", {})
            task = outcome.get("task_id")
            if task in required:
                actual = outcome["applied"]
                if (_representation_kind({"kind": actual["recipe"].get("prior_kind")})
                        != _representation_kind(declarations[task]["prior"])
                        or actual.get("prior") and _representation_kind(actual["prior"])
                           != _representation_kind(declarations[task]["prior"])):
                    raise ValueError("Named representation conflicts with its actual applied owner law")
                applied[task] = dict(recipe=outcome["applied"]["recipe"],
                                     table_ownership=outcome["applied"].get("table_ownership"),
                                     prior=outcome["applied"].get("prior"))
                media[task] = dict(outcome["gif_publication"],
                    path=(Path(inputs[key]["report"]).parent / outcome["gif_publication"]["path"]).as_posix())
        rows.append(dict(label=label, family=family, cohort=cohort, task_ids=required, required=26,
                         targeted_required=len(required), passed=statuses.count("PASS"),
                         targeted_counts=dict(Counter(statuses)), counts=family_report["counts"],
                         source=report["source"], readout=inputs[key]["readout"], media=media,
                         representation=dict(label=representation, report=inputs[key]["report"],
                            report_sha256=inputs[key]["sha256"],
                            declarations={task: dict(prior=q["prior"], policy_contract=q["policy_contract"],
                                execution_sha256=q["execution_sha256"], evaluation_sha256=q["evaluation_sha256"])
                                for task, q in declarations.items()}, applied=applied),
                         qualification_input=False))
    word = reports["word_context"]
    if (rows[-1]["targeted_counts"] != {"INVALID": 1} or word["execution_status"] != "INVALID"
            or word["certified_numerical_gate"] != "UNAVAILABLE"
            or any(word["source"].get(key) != value
                   for key, value in reports["adaptations_v4b"]["source"].items())
            or word["raw_reported_clock"]["completed_steps"] != 20001
            or word["actual_prior_rows"] != 11 or word["canonical_target_words"] != 5
            or set(word["raw_reported_clock"]["optimizer_updates"].values()) != {20001}):
        raise ValueError("Atlas retained word clock/grade/source identity changed")
    illustration = word["retained_illustration"]
    rows[-1]["media"][word["task_id"]] = dict(path=(Path(inputs["word_context"]["report"]).parent /
        illustration["path"]).as_posix(), sha256=illustration["sha256"], bytes=illustration["bytes"],
        accepted_numeric_verdict="UNAVAILABLE", caption_required=True)
    return dict(schema_version=2, scope="separate_source_scoped_model_scores",
                baseline=dict(required=26, counts=counts, source=baseline["source"],
                              candidate_id=baseline["candidate_id"],
                              shared_recipe_overrides=baseline["shared_recipe_overrides"],
                              representation=baseline_representation,
                              readout=inputs["baseline"]["readout"]),
                adaptations=rows,
                word=dict(status="INVALID", numerical_gate="UNAVAILABLE", raw_completed_steps=20001,
                          recorded_endpoint=word["recorded_endpoint"], readout=inputs["word_context"]["readout"]),
                inputs=inputs, qualification_input=False, qualification_reuse=False,
                ordinary_tier_credit=False, cross_cohort_pooling=False, default_adoption=False, speed_ranking=False)


def _original_pr223_atlas(result):
    """Expose the verified original-law replay without changing selected rows."""
    studies = result.get("completed_api_studies", {}).get("rows", [])
    matches = [row for row in studies if row.get("id") == "atlas19_original"]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Original PR223 display requires one complete replay publication")
    study = matches[0]
    config_sha = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
    reference_sha = "1e1b536bfd66e20a240436b417d429059d22ea55c2b6ecfff31e796b96c470fa"
    recipe = study.get("recipe", {})
    options = recipe.get("overrides", {})
    requested = dict(lr=.00425, prior_lr_mult=2., d_lr_mult=1., prior_kind="particles",
                     sigma_rel=0., output_noise_mode="learnable", output_noise_std=.029,
                     serve_average=4., ema_decay=.995, total_steps=None)
    if (study.get("required_cells") != 19 or study.get("counts") != {"PASS": 19}
            or study.get("declared_updates") != 48800
            or study.get("qualification_input") is not False or study.get("reuse") is not False
            or study.get("cross_cohort_pooling") is not False
            or recipe.get("sha256") != config_sha
            or recipe.get("path_in_snapshot") != "configs/100gaussians/atlas.json"
            or any(options.get(key) != value for key, value in requested.items())
            or study.get("source", {}).get("protected_files_sha256", {}).get(
                "configs/100gaussians/atlas.json") != config_sha):
        raise ValueError("Original PR223 display recipe, source or original scope changed")
    cells = study.get("cells", [])
    expected = {"native": (3, 1234), "moving": (3, 1234), "portability": (13, 0)}
    if len(cells) != 19 or len({cell["id"] for cell in cells}) != 19:
        raise ValueError("Original PR223 display requires all 19 distinct full questions")
    for group, (count, seed) in expected.items():
        members = [cell for cell in cells if cell["definition"].get("group") == group]
        if len(members) != count:
            raise ValueError("Original PR223 display question groups changed")
        for cell in members:
            definition = cell["definition"]
            if (cell.get("full_protocol_complete") is not True
                    or cell.get("execution_status") != "PASS" or cell.get("scientific_status") != "PASS"
                    or definition.get("config_sha256") != config_sha
                    or definition.get("reference_sha256") != reference_sha
                    or definition.get("sampling") != "original noisy state-selected primary; clean diagnostics separate"
                    or definition.get("original_host", {}).get("seed") != seed
                    or not definition.get("original_requirements") or not definition.get("observation_steps")
                    or not cell.get("media")):
                raise ValueError("Original PR223 display full gates, seeds or noisy law changed")
            if group == "native" and cell.get("native_gates", {}).get("noisy") != {
                    "coverage": "PASS", "accuracy": "PASS"}:
                raise ValueError("Original PR223 display requires noisy native coverage and accuracy")
    original = dict(label="Original PR223 Atlas FULL", representation="Particles", required=19,
                counts={"PASS": 19}, readout=study["readout"], publication=deepcopy(study["publication"]),
                source=deepcopy(study["source"]), recipe=deepcopy(recipe),
                seeds={group: seed for group, (_, seed) in expected.items()},
                question_ids=[cell["id"] for cell in cells],
                scope="Faithful replay of the full original 19-question noisy-law protocol",
                qualification_input=False, qualification_reuse=False, default_adoption=False,
                speed_ranking=False, new_current_retest_credit=False)
    fresh = result.get("passive_publications", {}).get("latest", {}).get("pg_pr223_full19_passive_publication_v1")
    if fresh:
        original["fresh_retest"] = deepcopy(fresh)
    return original


def _original_pr223_score_line(original, root, path):
    link = os.path.relpath(root / original["readout"], path.parent)
    fresh = original.get("fresh_retest")
    context = "fresh retest PENDING"
    if fresh:
        readout = os.path.relpath(root / fresh["readout"], path.parent)
        cost_link = os.path.relpath(root / fresh["final_cost"]["path"], path.parent)
        counts = fresh["counts"]
        remaining = ", ".join(f"{n} {status}" for status, n in counts.items() if status not in {"PASS", "FAIL"})
        context = (f"[fresh retest]({readout}): {fresh['accepted_counts']['PASS']}/19 PASS · "
                   f"{fresh['accepted_counts']['FAIL']} FAIL" + (f" · {remaining}" if remaining else "") +
                   f" · overall {fresh['status']} · source `{fresh['source']['origin_commit'][:8]}`/"
                   f"`{fresh['source']['digest'][:8]}` · [final charge {fresh['cost']['charged_seconds']:.6f} / 10800 s]({cost_link})")
    return (f"| [Original PR223 Atlas FULL · LR .00425 / prior2]({link}) | "
            f"{original['representation']} | **19/19 PASS** · full original recipe/law replay "
            f"· [19 original goal GIFs]({link})<br>Policy-selected; averaging enabled; learned output kernel (init .029) "
            f"· native/moving seed1234, portability seed0 · {context}; no current-26/default credit |")


def _half_base_score_line(half_base, root, path):
    def link(relative):
        return os.path.relpath(root / relative, path.parent)
    metrics = half_base.get("result", {}).get("final_metrics")
    endpoint = (f"<br>Final quality {metrics['quality_fraction']:.6f} · modes {metrics['modes']}/5 · "
                f"mass TV {metrics['mass_tv']:.6f} · all-five exact {metrics['reconstruction_exact']}"
                if metrics else "")
    return (f"| [half_base LR .00265625 / prior1.5 / D1 · source `{half_base['source']['origin_commit'][:8]}`]"
            f"({link(half_base['readout'])}) | {half_base['representation']['label']} | "
            f"**0/26 PASS** · FAIL 1 · NOT_RUN 25 · COMPLETE{endpoint}<br>"
            f"[accepted 20,001-update goal GIF]({link(half_base['media']['path'])}) |")


def _atlas_progress_markdown(progress, root, path, *, half_base=None):
    """Rows in the single current table, each with its own measured scope."""
    def link(relative):
        return os.path.relpath(root / relative, path.parent)
    baseline = progress["baseline"]
    counts = baseline["counts"]
    lines = [f"| [Atlas C6 CHANGED-rate / noise-OFF · LR .0053125 / prior1.5 · source `{baseline['source']['origin_commit'][:8]}`]({link(baseline['readout'])}) "
             f"| {baseline['representation']['label']} | **{counts['PASS']}/26 PASS** · "
             f"FAIL {counts['FAIL']} · BLOCKED {counts['BLOCKED']} "
             f"<br>Selected-policy diagnostic/reference · seed0 · [source and 18 goal GIFs]({link(baseline['readout'])}) |"]
    for row in progress["adaptations"]:
        gates = f"**{row['passed']}/{row['required']} PASS** · NOT_RUN {row['counts'].get('NOT_RUN', 0)}"
        readout = row["readout"]
        if row["family"] == "atlas_word_joint_min11":
            gates = "**INVALID 1** · NOT_RUN 25 · numerical gate UNAVAILABLE"
            readout = progress["word"]["readout"]
        media_label = "retained INVALID illustration" if row["family"] == "atlas_word_joint_min11" else "goal GIFs"
        media = " · ".join(f"[{task.split('_' + row['cohort'])[0]}]({link(pin['path'])})"
                           for task, pin in sorted(row["media"].items()))
        lines.append(f"| [`{row['family']}` · C6 · `{row['source']['origin_commit'][:8]}`]({link(readout)}) "
                     f"| {row['representation']['label']} | {gates}<br>Named-family diagnostic · {media_label}: {media} |")
    if half_base:
        lines.append(_half_base_score_line(half_base, root, path))
    return lines


def _policy_score_scope(root, path, *, half_base=None):
    link = os.path.relpath(root / "reports/forge/technique-inventory.json", path.parent)
    return ["The C6 diagnostic/reference changes rates to LR .0053125 / prior-rate 1.5 and disables "
              "evaluation output noise, with seed0 and declared host-specific "
              "Recipe fields. Representation labels come from the pinned applied priors, Recipes and "
              "routed table owners; a parameter bank is not a sampled MoG. "
              "[Full source and representation bindings](" + link + "). "
              "Original N5 word execution remains BLOCKED. " +
              ("The original C6 N11 illustration has no accepted numerical grade. The separate half_base run "
               "completed all 20,001 updates and 24 reads, with zero passing reads and an accepted numerical FAIL."
               if half_base else "The N11 illustration has no accepted numerical grade."), ""]


def _separate_baseline_links(result, row, root, path):
    """Navigate separate Atlas/C6 evidence without granting current tier credit."""
    family = row.get("trainer_family", row["candidate_id"].split("--", 1)[0])
    if family not in {"atlas", "e22"}:
        return []
    studies = {study["id"]: study for study in
               result.get("completed_api_studies", {}).get("rows", [])}
    links = []
    historical = studies.get("atlas19_original")
    if historical:
        counts = historical["counts"]
        if counts == {"PASS": historical["required_cells"]}:
            link = os.path.relpath(root / historical["readout"], path.parent)
            links.append(f"[Atlas history: {counts['PASS']}/{historical['required_cells']} PASS]({link})")
    hold = studies.get("c6_hold")
    if hold and hold["counts"] == {"FAIL": hold["required_cells"]}:
        debug = result.get("baseline_debugging", {}).get("readout")
        link = os.path.relpath(root / (debug or hold["readout"]), path.parent)
        label = {"atlas": "Atlas", "e22": "E22"}[family]
        links.append(f"[C6 {label} hold FAIL]({link})")
    return links


def _current_markdown(result, root, path):
    def cell(value):
        return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")
    tiers = list(result["tier_requirements"])
    recorded_policy = result.get("recorded_policy")
    ordinary_current = not recorded_policy and result["view"] == "discriminator_stability"
    title = "Recorded" if recorded_policy else "Current"
    lines = ["# Current model/configuration scores" if ordinary_current else
             f"# {title} Forge trainer-family leaderboard", ""]
    if recorded_policy:
        denominators = ", ".join(f"Tier {tier}: {len(result['tier_requirements'][tier])}" for tier in tiers)
        policy_link = os.path.relpath(root / recorded_policy, path.parent)
        lines += [f"This table preserves [{result['view']} revision {result['view_revision']}]({policy_link}) "
                  f"with recorded required denominators **{denominators}**. "
                  "Current task placement is listed in [experiments by tier](EXPERIMENTS_BY_TIER.md). "
                  "The recorded outcomes supply no qualification for a later view revision.", ""]
    if result.get("completed_api_studies") and not ordinary_current:
        lines += ["Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has "
                  "**2/2 hold FAIL**, with six other domains UNKNOWN per family. "
                  "See [completed source-bound studies](#completed-source-bound-studies) for the exact "
                  "protocols and original goal GIFs. These separate results do not fill the ordinary "
                  "qualification cells below.", ""]
    progress = result.get("atlas_unblocking_progress") if ordinary_current else None
    half_base = result.get("word_half_base_diagnostic") if ordinary_current else None
    original_pr223 = result.get("original_pr223_atlas") if ordinary_current else None
    if ordinary_current:
        lines += ["Each row keeps its own configuration, source, representation and measured scope. "
                  "Required denominators stay fixed; diagnostic and ordinary qualification scores remain separate.", "",
                  "| Model/configuration | Actual representation | Measured score/status and scope |",
                  "| --- | --- | --- |"]
        if original_pr223:
            lines.append(_original_pr223_score_line(original_pr223, root, path))
        if progress:
            lines += _atlas_progress_markdown(progress, root, path, half_base=half_base)
        elif half_base:
            lines.append(_half_base_score_line(half_base, root, path))
    else:
        lines += ["Each cell retains **passes / full required total** or **status (N required)** from one selected configuration. "
             "Mixed cells show the counts of failed, blocked and unmeasured tasks. " +
             ("Each trainer family and runtime has one archived row; its alternatives remain recorded separately. "
              if recorded_policy else "Each formulation family has one current selected row; source and runtime alternatives remain unranked. ") +
             "Expand the configuration details below for selection, provenance, other outcomes and cost.", "",
             "| Trainer family / runtime | " +
             " | ".join(f"Tier {tier}" for tier in tiers) + " | Recorded tier |",
                 "| --- | " + " | ".join("---:" for _ in tiers) + " | ---: |"]
    details = ["<details>", "<summary>Selected configurations and provenance</summary>", ""]
    for row in result["rows"]:
        # The actual C6 Atlas result occupies the displayed Atlas row. Its
        # incompatible ordinary reference is retained verbatim in the JSON.
        if ordinary_current and progress and row.get("trainer_family", row["candidate_id"]) == "atlas":
            continue
        source = row.get("bindings", {}).get("source_digest")
        pointer = result["evidence_sources"].get(row.get("publication_key"))
        if pointer and source:
            source_link = f"[`{source[:12]}`]({os.path.relpath(root / pointer['snapshot'], path.parent)})"
        else:
            source_link = f"`{source[:12]}` (unmeasured)" if source else "unresolved"
        runtime = row.get("runtime_cohort", {})
        models = sorted({p.get("model") for p in runtime.get("compute_profiles", {}).values() if p and p.get("model")})
        compute = runtime.get("execution_backend", "unrecorded") + (" / " + ", ".join(models) if models else "")
        counts = Counter(task["status"] for task in row.get("tasks", []) if task["status"] != "PASS")
        others = ", ".join(f"{status} {count}" for status, count in sorted(counts.items())) or "all required tasks PASS"
        seconds = row.get("cost", {}).get("wall_seconds")
        revision, cohort = (str(row.get(key) or "unresolved")[:12] for key in ("candidate_revision", "cohort"))
        selection = row.get("selection", {})
        selection_label = selection.get("selection_kind", "canonical_fallback").replace("_", " ")
        if selection.get("selection_kind") == "qualified_winner":
            selection_label += f" (tuning through tier {selection['tuning_through_tier']})"
        if not selection.get("qualified", False):
            selection_label += "; no qualified winner"
        name = row["candidate_id"]
        card = root / "configs/forge/configurations" / (name + ".json")
        if not card.is_file():
            card = root / "configs/forge/ideas" / (name + ".json")
        # Keep content-addressed IDs in the link target, with a readable label.
        label = name.rsplit("--", 1)
        configuration_label = " · ".join([label[0], label[1][:12]]) if len(label) == 2 else name
        configuration_link = f"[{configuration_label}]({os.path.relpath(card, path.parent)})"
        backend = runtime.get("execution_backend", "unrecorded")
        baseline_links = _separate_baseline_links(result, row, root, path)
        family_label = "<br>".join([f"{row['technique']}", backend, *baseline_links])
        values = [family_label,
                  *[_current_tier_cell(row, tier, result["tier_requirements"][tier],
                                      path.with_suffix('.json').name) for tier in tiers],
                  row["qualified_tier"]]
        if ordinary_current:
            json_link = os.path.relpath(root / CURRENT_PREFIX.with_suffix(".json"), path.parent)
            scores = "<br>".join(f"Tier {tier}: " + _current_tier_cell(
                row, tier, result["tier_requirements"][tier], json_link) for tier in tiers)
            unresolved = f"<br>Revision/cohort: {revision} / {cohort}" if not row.get("candidate_revision") or not row.get("cohort") else ""
            values = [f"{row['technique']} · {configuration_link}<br>{backend} · ordinary qualification",
                      _ordinary_representation(result, row),
                      scores + f"<br>Recorded tier: {row['qualified_tier']} · [source]({json_link})" + unresolved]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
        details += [f"### {cell(row['technique'])} ({cell(backend)})", "",
                    f"- **Selected configuration:** {configuration_link}",
                    f"- **Selection:** {cell(selection_label)}",
                    f"- **Evidence source:** {source_link}",
                    f"- **Exact revision / cohort:** {revision} / {cohort}",
                    f"- **Compute:** {cell(compute)}",
                    f"- **Other outcomes:** {cell(others)}",
                    f"- **Paid seconds:** {round(seconds, 3) if seconds is not None else 'unknown'}", ""]
        if baseline_links:
            details += ["- **Separate baseline evidence:** " + " · ".join(baseline_links) +
                        "; no current tier credit.", ""]
    lines.append("")
    if progress:
        lines += _policy_score_scope(root, path, half_base=half_base)
    if ordinary_current:
        lines += ["## Qualification and scope", "",
                  "MoG + particles (per task) denotes separate declared host laws within a suite; "
                  "it does not assert a hybrid model. The release-0.7 cloud-labelled row retains its "
                  "archived MoG declaration. BLOCKED rows show requested representations, not successful execution.", "",
                  "Before shipping defaults, one unchanged family/configuration tuple must satisfy all 26 required "
                  "5/19/2 gates with compatible source, runtime and serving evidence, followed by the separate "
                  "calibration and robustness requirements. Named diagnostic rows do not pool passing cells "
                  "across families or sources and grant no ordinary tier, default or speed credit.", ""]
        if result.get("completed_api_studies"):
            lines += ["Atlas's historical study has **19/19 original PASS**. The later C6 broad hold has "
                      "**2/2 hold FAIL**, with six other domains UNKNOWN per family. "
                      "See [completed source-bound studies](#completed-source-bound-studies) for the exact "
                      "protocols and original goal GIFs. These separate results do not fill ordinary qualification.", ""]
    if not ordinary_current:
        lines += details + ["</details>", ""]
    lines += ["Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws "
              "and hardware. They do not pool qualification across sources or qualify the latest checkout. "
              "Selection never combines passing tasks or tiers from different configurations. A failed best-observed "
              "configuration is not a qualified winner. Search qualification covers only its declared tuning tiers; "
              "later-tier outcomes are reported separately. The screening profile remains provisional and does not "
              "confer calibrated robustness or public-default adoption.", "",
              "BLOCKED means execution was incompatible or a prerequisite was unavailable. "
              "NOT RUN and UNKNOWN mean unexecuted or unmeasured; FAIL records an executed gate failure. "
              "Failed or blocked prerequisites stop later work; required denominators stay fixed. "
              "The Atlas-history and C6-hold links retain their separate protocols and supply no current tier credit.", "",
              f"[All configuration alternatives, trials, task statuses and exact bindings]({path.with_suffix('.json').name}) · "
              f"[Evidence and archived publication identities]({os.path.relpath(root / EVIDENCE_MANIFEST, path.parent)})", "",
              "Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:", "",
              "```sh", "python reports/forge/regenerate_technique_inventory.py" +
              (" --recorded-policy " + shlex.quote(recorded_policy) if recorded_policy else ""), "```", "",
              ("This command uses the exact archived policy and registered snapshots; it does not resolve new declarations. "
               "New evidence for a later view revision requires its own compatible evidence registration."
               if recorded_policy else
               "For ordinary Forge qualification, use `--source-commit <executed-commit>` to independently regrade "
               "hydrated original receipts and update this leaderboard. Reusable candidates use the same complete task ladder. "
               "Historical task-only diagnostics remain motivation and reproduction evidence. Source snapshots are provenance, not additional leaderboards."), "",
              f"Publication input digest `{result['provenance']['input_digest']}`.", ""]
    if not recorded_policy and result["view"] == "discriminator_stability":
        lines += ["The [whole-family repair readout](family-wide-word-repairs/README.md) records the ordinary "
                  "candidate attempts and bounded global configuration search. Their complete rows remain "
                  "unranked alternatives below and in the companion JSON; a failed replacement does not "
                  "make its historical incumbent a qualified standard.", ""]
    if result.get("archived_policies"):
        lines += ["Earlier view policies retain their exact numerical snapshots and receipt proofs in the companion JSON. "
                  "Their outcomes do not fill current requirements:", ""]
        for policy in result["archived_policies"]:
            totals = "/".join(str(len(policy["tier_requirements"][tier])) for tier in tiers)
            lines.append(f"- `{policy['view']}` revision {policy['view_revision']}: recorded denominators {totals}; "
                         f"{len(policy['cohorts'])} source cohorts.")
        lines.append("")
    archived = [row for row in result.get("configuration_rows", []) if row.get("alternative_scope") == "archived_alternative"]
    if archived:
        lines += ["Unranked alternatives retain their original outcomes and exact source/runtime bindings:", ""]
        for row in archived:
            source = row.get("bindings", {}).get("source_digest", "unresolved")
            lines.append(f"- `{row['candidate_id']}` ({row['trainer_family']}), source `{source[:12]}`; "
                         f"recorded tier {row.get('qualified_tier', 0)}. Full evidence is in the companion JSON.")
        lines.append("")
    if result.get("completed_api_studies"):
        if ordinary_current:
            lines += ["## Completed source-bound studies", "",
                      "Historical protocols and distinct contrasts retain their exact scores, source and GIFs "
                      "in their original readouts. They do not fill cells in the table above.", ""]
            for study in result["completed_api_studies"]["rows"]:
                link = os.path.relpath(root / study["readout"], path.parent)
                lines.append(f"- [{study['label']}]({link}) · own {study['required_cells']}-cell scope.")
            lines.append("")
        else:
            from experiments.forge.completed_studies import render_completed_studies
            lines += ["", render_completed_studies(result["completed_api_studies"], root, path), ""]
    if result.get("baseline_debugging"):
        link = os.path.relpath(root / result["baseline_debugging"]["readout"], path.parent)
        lines += [f"[Exact C6 baseline and retained persistence diagnosis]({link}): "
                  "two original smoke passes per family, six later domains UNKNOWN; "
                  "the added broad hold fails on projected-CDF shape excursions. "
                  "The source-bound diagnostic preserves all original gates and supplies no default or speed credit.", ""]
    return "\n".join(lines)


def _shared_score_archive(root):
    """Move the original historical suffix intact out of the current chart."""
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    if not path.is_file():
        return
    archive = path.with_name("ARCHIVED_REPORTS.md")
    original = path.read_bytes()
    marker = b"## Completed current Atlas GPU diagnostic\n"
    if marker in original:
        suffix = marker + original.split(marker, 1)[1]
        if archive.exists() and archive.read_bytes() != suffix:
            raise ValueError("Shared score index historical archive differs from its preserved report boundary")
    elif b"[Historical report archive](ARCHIVED_REPORTS.md)" in original and archive.is_file():
        suffix = archive.read_bytes()
    else:
        raise ValueError("Shared score index is missing its preserved report boundary")
    return archive, suffix.decode("utf-8")


def _shared_score_intro(root, result):
    """Render the one current table and link the separate unchanged archive."""
    path = root / "reports/forge/shared-score-index-20261003/README.md"
    if not path.is_file():
        return
    _shared_score_archive(root)  # Validate the legacy or already-moved boundary.
    leading = _current_markdown(result, root, path).split("## Qualification and scope\n", 1)[0]
    leading = leading.replace("# Current model/configuration scores", "# Shared ParticleGAN score index", 1)
    half_base = result.get("word_half_base_diagnostic")
    if half_base:
        accounting = half_base["accounting"]
        account_text = (
            f"The named campaign's predecessor-inclusive charge is **{accounting['inclusive_charged_seconds']} / 10500 seconds**, "
            f"reserve zero: GPU0 **{accounting['gpu0_charged_seconds']} / 7500**, "
            f"GPU1 **{accounting['gpu1_charged_seconds']} / 3000**. "
            f"The separate half_base attempt charged **{half_base['cost']['paid_seconds']} / 900 seconds**, "
            "with zero reserve and overrun. Prior charges are included once; the original C6 word INVALID is unchanged.\n\n")
    else:
        account_text = ("The named campaign's predecessor-inclusive charge remains **910.2391431590077 / 10500 seconds**, "
                        "reserve zero: GPU0 **234.82608077581972 / 7500**, GPU1 **675.4130623831879 / 3000**. "
                        "The half-base contrast has no completed measurement in these tables.\n\n")
    scope = ("## Qualification and accounting\n\n"
             "Each model/configuration keeps its own source, law and full 26-slot denominator. "
             "MoG + particles (per task) is a mixture of separate suite hosts, not a hybrid-model claim. "
             "Named diagnostic passes do not fill ordinary cells or combine into a family score. "
             "INVALID supplies no accepted numerical grade; original N5 word execution remains BLOCKED. "
             "No family default or comparable speed winner is established.\n\n"
             + account_text +
             "[Historical report archive](ARCHIVED_REPORTS.md) · [Original snapshot index](index.json). "
             "The archived source, scores, costs and links retain their original bytes and separate scopes.\n")
    return path, leading + scope


def publish_current(root=REPOSITORY_ROOT, *, source_commit=None, view_id="discriminator_stability",
                    execution_backend=None, recorded_policy=None, advance_policy=False):
    """Maintain one current table; registered source snapshots retain the history."""
    root = Path(root).resolve()
    manifest_path = root / EVIDENCE_MANIFEST
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if recorded_policy is not None and source_commit is not None:
        raise ValueError("recorded policy publication cannot register new source evidence")
    if advance_policy and (source_commit is None or recorded_policy is not None):
        raise ValueError("advancing the evidence policy requires --source-commit and no recorded policy")
    recorded_view = None
    if recorded_policy is not None:
        if manifest is None:
            raise ValueError("recorded policy publication requires registered technique evidence")
        recorded_path = Path(recorded_policy)
        if not recorded_path.is_absolute():
            recorded_path = root / recorded_path
        recorded_view = read_json(recorded_path)
        recorded_policy = os.path.relpath(recorded_path.resolve(), root)
    reports, archived_reports, advancing = [], [], False
    policy = None
    if manifest is not None:
        if manifest.get("schema_version") != 1 or manifest.get("view") != view_id:
            raise ValueError("unsupported current technique evidence manifest/view")
        policy_path = root / "configs/forge/views" / (view_id + ".json")
        if recorded_view is not None or policy_path.is_file():
            policy = recorded_view if recorded_view is not None else read_json(policy_path)
            if (policy.get("id") != view_id or policy.get("revision") != manifest["view_revision"]
                    or stable_hash(policy) != manifest["policy_fingerprint"]):
                label = "recorded" if recorded_view is not None else "current"
                if not advance_policy:
                    raise ValueError(f"{label} view policy differs from registered technique evidence")
                if (policy.get("id") != view_id or type(policy.get("revision")) is not int
                        or policy["revision"] <= manifest["view_revision"]):
                    raise ValueError("advancing requires a later revision of the same view policy")
                advancing = True
        reports = [(entry, *_snapshot(root, entry, manifest)) for entry in manifest["cohorts"]]
        archived_reports = _archived_reports(root, manifest)
    pending_snapshot = None
    if source_commit is not None:
        with tempfile.TemporaryDirectory(prefix="forge-technique-evidence-") as temporary:
            metadata = regenerate(root, view_id=view_id, execution_backend=execution_backend,
                                  source_commit=source_commit, output_prefix=Path(temporary) / "evidence")
            report = read_json(metadata["json"])
        candidates = {}
        for row in report["rows"]:
            if not row.get("attempt_ids"):
                continue
            times = [read_json(root / "reports/forge/attempts" / attempt / "result.json")["raw"]["finished_at"]
                     for attempt in row["attempt_ids"]]
            if row["candidate_id"] in candidates:
                raise ValueError("select one runtime cohort; current rows cannot pool runtimes")
            candidates[row["candidate_id"]] = max(times)
        if not candidates:
            raise ValueError("source regrade selected no measured technique rows; restore its recorded runtime/hardware")
        # Freeze declared blockers on the explicitly selected backend alongside
        # measured rows. A null completion time records no execution or gate
        # credit; it prevents cached regeneration from inventing CPU shadows or
        # resolving those same blockers against later source changes.
        if execution_backend is not None:
            unmeasured = {}
            for row in report["rows"]:
                if row["candidate_id"] in candidates or row.get("attempt_ids") or row.get("status") != "BLOCKED":
                    continue
                unmeasured.setdefault(row["candidate_id"], []).append(row)
            for name, rows in unmeasured.items():
                if len(rows) != 1 or rows[0].get("runtime_cohort", {}).get("execution_backend") != execution_backend:
                    raise ValueError("select one runtime cohort; blocked current rows cannot pool runtimes")
                candidates[name] = None
        if advancing:
            if (report.get("view") != policy["id"] or report.get("view_revision") != policy["revision"]
                    or report.get("policy_fingerprint") != stable_hash(policy)):
                raise ValueError("new source evidence differs from the current view policy")
            previous = {key: deepcopy(manifest[key]) for key in POLICY_FIELDS}
            previous["cohorts"] = deepcopy(manifest["cohorts"])
            archived_reports.extend((previous, entry, old_report, rows) for entry, old_report, rows in reports)
            manifest = {**deepcopy(manifest), **{key: deepcopy(report[key]) for key in POLICY_FIELDS},
                        "cohorts": [], "archived_policies": [*manifest.get("archived_policies", []), previous]}
            reports = []
        if manifest is None:
            manifest = {"schema_version": 1, **{key: report[key] for key in
                        ("view", "view_revision", "policy_fingerprint", "tier_requirements")},
                        "cohorts": [], "retired_publications": {}}
        for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
            if report.get(key) != manifest[key]:
                raise ValueError(f"new technique evidence has incompatible {key}")
        data = _json_text(report)
        relative = EVIDENCE_MANIFEST.parent / (report["provenance"]["input_digest"] + ".json")
        entry = {"snapshot": relative.as_posix(), "json_sha256": hashlib.sha256(data.encode()).hexdigest(),
                 "source_commit": metadata["source_commit"], "candidates": candidates}
        # Keep exact evidence across CPU/GPU models and other runtime cohorts,
        # including cohorts from the same source commit. Regrading an identical
        # snapshot is idempotent; a new snapshot never retires another runtime.
        if entry not in manifest["cohorts"]:
            manifest["cohorts"].append(entry)
        # A frozen all-backend roster also contains unmeasured configurations
        # on other runtimes. Those cannot overwrite a measured registration
        # of the same candidate just because they appear later in the roster.
        selected = {row["candidate_id"]: row for row in report["rows"]
                    if row["candidate_id"] in candidates and
                    (bool(row.get("attempt_ids")) if isinstance(candidates[row["candidate_id"]], str) else True)}
        for row in selected.values():
            _validate_published_row(root, report, row)
        if not any(old == entry for old, _, _ in reports):
            reports.append((entry, report, selected))
        pending_snapshot = root / relative, data
    if manifest is None:
        raise ValueError("no registered technique evidence; use --source-commit after a completed experiment")
    from experiments.forge.planning import declaration_paths
    from experiments.forge.trainer_families import (CURRENT_SELECTION, family_for_candidate, load_current_selection,
                                                    scientific_row_hash, select_family_rows)
    declarations = {path.stem: read_json(path) for path in declaration_paths(root)}
    candidates = set(declarations)
    current_pins = (load_current_selection(root, view_id=view_id, policy_fingerprint=manifest["policy_fingerprint"])
                    if recorded_view is None else {})
    selected = {}
    for entry, report, rows in reports:
        for name, row in rows.items():
            backend = row.get("runtime_cohort", {}).get("execution_backend", report.get("execution_backend"))
            if name not in candidates or (execution_backend is not None and backend != execution_backend):
                continue
            key = name, backend
            rank = entry["candidates"][name] or ""
            previous = selected.get(key)
            if previous is None or rank > previous[0]:
                selected[key] = rank, entry, report, deepcopy(row)
            elif (rank == previous[0] and any(row.get(field) != previous[3].get(field)
                                             for field in ("candidate_revision", "cohort", "runtime_cohort"))
                  and family_for_candidate(root, name, declarations.get(name),
                                           current_presentation=True)["id"] not in current_pins):
                raise ValueError("ambiguous latest recorded technique cohort")
    # A search runtime needs the actual canonical declaration as its fallback;
    # never present an arbitrary unmeasured tuning trial as the family default.
    canonical_missing = set()
    for (name, backend), (_, entry, report, row) in list(selected.items()):
        canonical = family_for_candidate(root, name, declarations.get(name),
                                         current_presentation=recorded_view is None)["canonical_candidate"]
        runtime_key = stable_hash(row.get("runtime_cohort"))
        if any(item[3]["candidate_id"] == canonical and item[3].get("runtime_cohort") == row.get("runtime_cohort")
               for item in selected.values()):
            continue
        matches = [item for item in report["rows"] if item["candidate_id"] == canonical
                   and item.get("runtime_cohort") == row.get("runtime_cohort")
                   and item.get("bindings", {}).get("source_digest") == row.get("bindings", {}).get("source_digest")]
        if len(matches) == 1 and not matches[0].get("attempt_ids"):
            _validate_published_row(root, report, matches[0])
            selected[canonical, backend, runtime_key] = "", entry, report, deepcopy(matches[0])
        else:
            canonical_missing.add((canonical, backend, runtime_key))
    missing = candidates - {key[0] for key in selected}
    live = None
    if (missing or canonical_missing) and recorded_view is None:
        with tempfile.TemporaryDirectory(prefix="forge-technique-current-") as temporary:
            metadata = write_report(root, view_id, execution_backend=execution_backend,
                                    output_prefix=Path(temporary) / "current")
            live = read_json(metadata["json"])
        for key in ("view", "view_revision", "policy_fingerprint", "tier_requirements"):
            if live.get(key) != manifest[key]:
                raise ValueError(f"current declarations have incompatible {key}; update the evidence view explicitly")
        for row in live["rows"]:
            backend = row.get("runtime_cohort", {}).get("execution_backend")
            runtime_key = stable_hash(row.get("runtime_cohort"))
            if row["candidate_id"] in missing or (row["candidate_id"], backend, runtime_key) in canonical_missing:
                # Measured rows must first be frozen and registered, so a later
                # code change cannot silently erase their published results.
                if row.get("attempt_ids"):
                    raise ValueError("register new measured technique evidence with --source-commit")
                row.setdefault("bindings", {})["source_origin_commit"] = None
                selected[row["candidate_id"], backend, runtime_key] = "", None, live, deepcopy(row)
    if current_pins:
        # Exact historical incumbents must survive newer observations of the
        # same card. Keep complete source/runtime alternatives, never cells.
        retained = {}
        for entry, report, rows in reports:
            for name, row in rows.items():
                backend = row.get("runtime_cohort", {}).get("execution_backend")
                if name not in candidates or (execution_backend is not None and backend != execution_backend):
                    continue
                retained.setdefault(scientific_row_hash(row), ("", entry, report, deepcopy(row)))
        for item in selected.values():
            retained.setdefault(scientific_row_hash(item[3]), item)
        selected = retained
    from experiments.forge.technique_board import DEFAULT_LABELS
    result = {"schema_version": 1, "reducer_version": "forge-current-technique-inventory-v1",
              "publication_scope": "current_technique_inventory", "qualification_reuse": False,
              "qualification_input": False, **{key: manifest[key] for key in
              ("view", "view_revision", "policy_fingerprint", "tier_requirements")},
              "execution_backend": execution_backend, "rows": [], "evidence_sources": {}}
    if manifest.get("archived_policies"):
        result["archived_policies"] = deepcopy(manifest["archived_policies"])
    if recorded_policy is not None:
        result["recorded_policy"] = recorded_policy
    for _, entry, report, row in selected.values():
        if entry is not None:
            row["publication_key"] = entry["json_sha256"]
            result["evidence_sources"][row["publication_key"]] = deepcopy(entry)
        else:
            row.pop("publication_key", None)
        row.update(qualification_reuse=False, qualification_input=False,
                   technique=DEFAULT_LABELS.get(row["candidate_id"], row.get("technique", row["candidate_id"])))
        result["rows"].append(row)
    result["rows"].sort(key=lambda row: (-row["qualified_tier"], row["technique"], row.get("cohort") or ""))
    for catalog in CONTRACT_CATALOGS:
        combined = {}
        for report in ([data_report for _, data_report, _ in reports]
                       + [data_report for _, _, data_report, _ in archived_reports] + ([live] if live else [])):
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract):
                    raise ValueError(f"invalid {catalog} identity in technique evidence")
                if digest in combined and combined[digest] != contract:
                    raise ValueError("conflicting scientific contract identities")
                combined[digest] = deepcopy(contract)
        result[catalog] = dict(sorted(combined.items()))
    family_result = select_family_rows(root, result["rows"], result, view_id=view_id,
                                       policy_fingerprint=manifest["policy_fingerprint"], declarations=declarations,
                                       view_policy=recorded_view, execution_backend=execution_backend)
    result.update(family_result)
    # Immutable scientific history stays numerical, including earlier revisions
    # of the same configuration. It cannot fill cells in the selected row.
    result["evidence_rows"] = []
    for entry, report, rows in reports:
        result["evidence_sources"][entry["json_sha256"]] = deepcopy(entry)
        for original in rows.values():
            if execution_backend is not None and original.get("runtime_cohort", {}).get("execution_backend") != execution_backend:
                continue
            evidence = deepcopy(original)
            evidence.update(publication_key=entry["json_sha256"], qualification_reuse=False, qualification_input=False)
            result["evidence_rows"].append(evidence)
    if archived_reports:
        result["archived_evidence_rows"] = []
        for policy, entry, _, rows in archived_reports:
            result["evidence_sources"][entry["json_sha256"]] = deepcopy(entry)
            for original in rows.values():
                evidence = deepcopy(original)
                evidence.update(publication_key=entry["json_sha256"], qualification_reuse=False,
                                qualification_input=False,
                                evidence_policy={key: deepcopy(policy[key]) for key in POLICY_FIELDS})
                result["archived_evidence_rows"].append(evidence)
    # Compact API studies retain their own gates, laws and full denominators.
    # They are display evidence and never enter ordinary family selection.
    from experiments.forge import completed_studies as completed_studies_projection
    completed_studies = completed_studies_projection.load_completed_studies(root)
    if completed_studies:
        result["completed_api_studies"] = completed_studies
    if not recorded_policy and view_id == "discriminator_stability":
        from experiments.forge import publication_memory
        passive = publication_memory.load_passive_publications(root)
        if passive:
            result["passive_publications"] = passive
        original_pr223 = _original_pr223_atlas(result)
        if original_pr223:
            result["original_pr223_atlas"] = original_pr223
        atlas_progress = _atlas_unblocking_progress(root)
        if atlas_progress:
            result["atlas_unblocking_progress"] = atlas_progress
        half_base = _word_half_base_diagnostic(root)
        if half_base:
            result["word_half_base_diagnostic"] = half_base
    debug_root = root / "reports/forge/c6-baseline-debug-20261003"
    if debug_root.is_dir():
        debug_names = ("README.md", "BASELINE_SELECTION.md", "BASELINE_SELECTION.json",
                       "RETAINED_HOLD_DEBUG.md", "retained-hold-debug.json",
                       "analyze_retained_hold.py", "copy-verification.json")
        result["baseline_debugging"] = {
            "readout": (debug_root / "README.md").relative_to(root).as_posix(),
            "files_sha256": {(debug_root / name).relative_to(root).as_posix(): file_hash(debug_root / name)
                             for name in debug_names},
            "qualification_input": False, "qualification_reuse": False,
            "scope": "Exact original C6 baseline selection and retained-data causal diagnosis; no new qualification",
        }
    result["provenance"] = {"publication_reducer_sha256": file_hash(Path(__file__)),
                            "evidence_manifest_sha256": stable_hash(manifest),
                            "trainer_family_registry_sha256": file_hash(root / "configs/forge/trainer-families.json")
                                if (root / "configs/forge/trainer-families.json").is_file() else None,
                            "selected_rows_sha256": stable_hash(result["rows"]),
                            "family_current_selection_sha256": file_hash(root / CURRENT_SELECTION)
                                if current_pins else None}
    if completed_studies:
        result["provenance"]["completed_studies_projector_sha256"] = file_hash(
            Path(completed_studies_projection.__file__))
    if result.get("passive_publications"):
        result["provenance"]["passive_publications_reducer_sha256"] = file_hash(Path(publication_memory.__file__))
    result["provenance"]["input_digest"] = stable_hash(result)
    json_path, markdown_path = root / CURRENT_PREFIX.with_suffix(".json"), root / CURRENT_PREFIX.with_suffix(".md")
    markdown = _current_markdown(result, root, markdown_path)
    shared_archive = (_shared_score_archive(root)
                      if not recorded_policy and view_id == "discriminator_stability" else None)
    shared_intro = (_shared_score_intro(root, result)
                    if not recorded_policy and view_id == "discriminator_stability" else None)
    # Validate all inputs before changing evidence registry or public outputs.
    if pending_snapshot:
        _write_changed(*pending_snapshot)
        _write_changed(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    if shared_archive:
        _write_changed(*shared_archive)
    if shared_intro:
        _write_changed(*shared_intro)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(result["rows"]),
            "input_digest": result["provenance"]["input_digest"], "qualification_reuse": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--goal", default="discriminator_stability")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--source-commit", help="reconstruct and grade an exact recorded Git source cohort, independently of live HEAD")
    parser.add_argument("--recorded-policy", type=Path,
                        help="rebuild registered rows under this exact archived view policy, without resolving live declarations")
    parser.add_argument("--advance-policy", action="store_true",
                        help="with --source-commit, archive earlier policy cohorts and register evidence for the current view revision")
    args = parser.parse_args(argv)
    print(_json_text(publish_current(args.root, view_id=args.goal,
                                execution_backend=None if args.device == "all" else args.device,
                                source_commit=args.source_commit, recorded_policy=args.recorded_policy,
                                advance_policy=args.advance_policy)), end="")


if __name__ == "__main__":
    main()
