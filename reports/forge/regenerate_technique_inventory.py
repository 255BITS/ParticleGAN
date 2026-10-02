"""Update the single current technique leaderboard from validated evidence.

Default regeneration uses committed numerical snapshots and compact receipts.
--source-commit independently regrades hydrated original receipts and registers
new measured rows before updating the same leaderboard. No command trains.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
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
    for row in result["rows"]:
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
        matches = [row for row in report["rows"] if row.get("candidate_id") == name]
        if len(matches) != 1:
            raise ValueError("technique evidence needs one exact candidate/runtime row")
        row = matches[0]
        if row.get("attempt_ids") and not isinstance(recorded_at, str):
            raise ValueError("measured technique evidence needs its recorded completion time")
        _validate_published_row(root, report, row)
        selected[name] = row
    return report, selected


def _current_markdown(result, root, path):
    def cell(value):
        return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")
    tiers = list(result["tier_requirements"])
    lines = ["# Current Forge trainer-family leaderboard", "",
             "Each cell is **passes / full required total** from one complete selected configuration. "
             "Each trainer family and runtime has one row; its alternatives remain recorded separately.", "",
             "| Trainer family | Selected configuration | Selection | Evidence source | Exact revision / cohort | Compute | " +
             " | ".join(f"Tier {tier}" for tier in tiers) + " | Recorded tier | Other outcomes | Paid seconds |",
             "| --- | --- | --- | --- | --- | --- | " + " | ".join("---:" for _ in tiers) + " | ---: | --- | ---: |"]
    for row in result["rows"]:
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
        if not selection.get("qualified", False):
            selection_label += "; no qualified winner"
        name = row["candidate_id"]
        card = root / "configs/forge/configurations" / (name + ".json")
        if not card.is_file():
            card = root / "configs/forge/ideas" / (name + ".json")
        configuration_link = f"[`{name}`]({os.path.relpath(card, path.parent)})"
        values = [row["technique"], configuration_link, selection_label, source_link, f"{revision} / {cohort}", compute,
                  *[f"{row['tiers'][tier]['passed']}/{row['tiers'][tier]['total']}" for tier in tiers],
                  row["qualified_tier"], others, round(seconds, 3) if seconds is not None else "unknown"]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "Recorded results remain bound to their actual recipes, priors, initialization, budgets, sampling laws "
              "and hardware. They do not pool qualification across sources or qualify the latest checkout. "
              "Selection never combines passing tasks or tiers from different configurations. A failed best-observed "
              "configuration is not a qualified winner. Search evidence is provisional, without independent confirmation "
              "or public-default adoption. The screening profile remains provisional.", "",
              "UNKNOWN means unmeasured. Failed or blocked prerequisites stop later work; required denominators stay fixed.", "",
              f"[All configuration alternatives, trials, task statuses and exact bindings]({path.with_suffix('.json').name}) · "
              f"[Evidence and archived publication identities]({os.path.relpath(root / EVIDENCE_MANIFEST, path.parent)})", "",
              "Regenerate this same leaderboard from committed evidence, without training or raw-log hydration:", "",
              "```sh", "python reports/forge/regenerate_technique_inventory.py", "```", "",
              "After a new experiment, use `--source-commit <executed-commit>` to independently regrade its "
              "hydrated original receipts and update this leaderboard. Source snapshots are provenance, not additional leaderboards.", "",
              f"Publication input digest `{result['provenance']['input_digest']}`.", ""]
    archived = [row for row in result.get("configuration_rows", []) if row.get("alternative_scope") == "archived_alternative"]
    if archived:
        lines += ["Archived alternatives retain their original outcomes and incompatible source/runtime bindings:", ""]
        for row in archived:
            source = row.get("bindings", {}).get("source_digest", "unresolved")
            lines.append(f"- `{row['candidate_id']}` ({row['trainer_family']}), source `{source[:12]}`; "
                         f"recorded tier {row.get('qualified_tier', 0)}. Full evidence is in the companion JSON.")
        lines.append("")
    return "\n".join(lines)


def publish_current(root=REPOSITORY_ROOT, *, source_commit=None, view_id="discriminator_stability",
                    execution_backend=None):
    """Maintain one current table; registered source snapshots retain the history."""
    root = Path(root).resolve()
    manifest_path = root / EVIDENCE_MANIFEST
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    reports = []
    if manifest is not None:
        if manifest.get("schema_version") != 1 or manifest.get("view") != view_id:
            raise ValueError("unsupported current technique evidence manifest/view")
        policy_path = root / "configs/forge/views" / (view_id + ".json")
        if policy_path.is_file():
            policy = read_json(policy_path)
            if (policy.get("id") != view_id or policy.get("revision") != manifest["view_revision"]
                    or stable_hash(policy) != manifest["policy_fingerprint"]):
                raise ValueError("current view policy differs from registered technique evidence")
        reports = [(entry, *_snapshot(root, entry, manifest)) for entry in manifest["cohorts"]]
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
        selected = {row["candidate_id"]: row for row in report["rows"] if row["candidate_id"] in candidates}
        for row in selected.values():
            _validate_published_row(root, report, row)
        if not any(old == entry for old, _, _ in reports):
            reports.append((entry, report, selected))
        pending_snapshot = root / relative, data
    if manifest is None:
        raise ValueError("no registered technique evidence; use --source-commit after a completed experiment")
    from experiments.forge.planning import declaration_paths
    from experiments.forge.trainer_families import family_for_candidate, select_family_rows
    declarations = {path.stem: read_json(path) for path in declaration_paths(root)}
    candidates = set(declarations)
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
            elif rank == previous[0] and any(row.get(field) != previous[3].get(field)
                                            for field in ("candidate_revision", "cohort", "runtime_cohort")):
                raise ValueError("ambiguous latest recorded technique cohort")
    # A search runtime needs the actual canonical declaration as its fallback;
    # never present an arbitrary unmeasured tuning trial as the family default.
    canonical_missing = set()
    for (name, backend), (_, entry, report, row) in list(selected.items()):
        canonical = family_for_candidate(root, name, declarations.get(name))["canonical_candidate"]
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
    if missing or canonical_missing:
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
    from experiments.forge.technique_board import DEFAULT_LABELS
    result = {"schema_version": 1, "reducer_version": "forge-current-technique-inventory-v1",
              "publication_scope": "current_technique_inventory", "qualification_reuse": False,
              "qualification_input": False, **{key: manifest[key] for key in
              ("view", "view_revision", "policy_fingerprint", "tier_requirements")},
              "execution_backend": execution_backend, "rows": [], "evidence_sources": {}}
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
        for report in [data_report for _, data_report, _ in reports] + ([live] if live else []):
            for digest, contract in report.get(catalog, {}).items():
                if digest != stable_hash(contract):
                    raise ValueError(f"invalid {catalog} identity in technique evidence")
                if digest in combined and combined[digest] != contract:
                    raise ValueError("conflicting scientific contract identities")
                combined[digest] = deepcopy(contract)
        result[catalog] = dict(sorted(combined.items()))
    family_result = select_family_rows(root, result["rows"], result, view_id=view_id,
                                       policy_fingerprint=manifest["policy_fingerprint"], declarations=declarations)
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
    result["provenance"] = {"publication_reducer_sha256": file_hash(Path(__file__)),
                            "evidence_manifest_sha256": stable_hash(manifest),
                            "trainer_family_registry_sha256": file_hash(root / "configs/forge/trainer-families.json")
                                if (root / "configs/forge/trainer-families.json").is_file() else None,
                            "selected_rows_sha256": stable_hash(result["rows"])}
    result["provenance"]["input_digest"] = stable_hash(result)
    json_path, markdown_path = root / CURRENT_PREFIX.with_suffix(".json"), root / CURRENT_PREFIX.with_suffix(".md")
    markdown = _current_markdown(result, root, markdown_path)
    # Validate all inputs before changing evidence registry or public outputs.
    if pending_snapshot:
        _write_changed(*pending_snapshot)
        _write_changed(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    _write_changed(json_path, _json_text(result))
    _write_changed(markdown_path, markdown)
    return {"report": str(markdown_path), "json": str(json_path), "rows": len(result["rows"]),
            "input_digest": result["provenance"]["input_digest"], "qualification_reuse": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--goal", default="discriminator_stability")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--source-commit", help="reconstruct and grade an exact recorded Git source cohort, independently of live HEAD")
    args = parser.parse_args(argv)
    print(_json_text(publish_current(args.root, view_id=args.goal,
                                execution_backend=None if args.device == "all" else args.device,
                                source_commit=args.source_commit)), end="")


if __name__ == "__main__":
    main()
