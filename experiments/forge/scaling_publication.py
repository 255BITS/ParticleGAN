"""Validate compact DCGAN performance receipts and render the existing board.

This reducer never imports torch, runs training, regrades a scientific task, or
selects a recipe. The manifest is an evidence index, not a second leaderboard.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

from .contracts import atomic_json, atomic_text, file_hash, identifier, read_json, stable_hash

MANIFEST = Path("reports/forge/scaling-performance.json")
SCOPE = "dcgan_mock_performance"
BENCHMARK = "dcgan_performance_v1"
START = "<!-- Forge scaling performance: start -->"
END = "<!-- Forge scaling performance: end -->"
TIMING_SCOPE = ("Synchronized wall time for one public trainer update, including its "
                "critic/generator/prior work; warmup, construction, mock-data generation, "
                "evaluation, GIFs and checkpoint writes are excluded.")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _fail(message):
    raise ValueError("scaling performance: " + message)


def _digest(value, label):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        _fail("invalid " + label)
    return value


def _path(root, name):
    if not isinstance(name, str):
        _fail("receipt path must be a relative string")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        _fail("unsafe receipt path")
    path = root / relative
    if (not path.resolve().is_relative_to(root)
            or any(parent.is_symlink() for parent in (path, *path.parents)
                   if parent != root and parent.is_relative_to(root))):
        _fail("receipt symlink or path escape")
    return path


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        _fail(label + " must be finite and " + ("positive" if positive else "nonnegative"))


def performance_status(receipt):
    """Keep execution failures explicit without suggesting image quality passed."""
    detail = str(receipt.get("benchmark_status", "")).upper()
    if "OOM" in detail or "OUT_OF_MEMORY" in detail:
        return "OOM"
    if "UNSUPPORTED" in detail or receipt.get("status") == "BLOCKED":
        return "UNSUPPORTED"
    if "ERROR" in detail or receipt.get("status") == "ERROR":
        return "ERROR"
    if receipt.get("status") == "PASS":
        return "COMPLETE"
    if "NONFINITE" in detail or receipt.get("status") == "FAIL":
        return "NONFINITE" if "NONFINITE" in detail else "FAIL"
    return "INCOMPLETE"


def validate_receipt(receipt):
    """Check internal exact-source/task/recipe/runtime bindings, even for failures."""
    if receipt.get("schema_version") != 1 or receipt.get("benchmark") != BENCHMARK:
        _fail("unsupported benchmark receipt")
    for key in ("spec", "identity", "runtime", "hardware", "recipe", "metrics", "validity"):
        if not isinstance(receipt.get(key), dict):
            _fail("receipt missing " + key)
    identity = receipt["identity"]
    source = deepcopy(identity.get("source", {}))
    source_hash = source.pop("sha256", None)
    if not isinstance(source.get("files_sha256"), dict) or not source["files_sha256"]:
        _fail("missing benchmark source file map")
    for name, digest in source["files_sha256"].items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            _fail("unsafe source file path")
        _digest(digest, "source file digest")
    if _digest(source_hash, "source digest") != stable_hash(source):
        _fail("source digest mismatch")
    for name, document in (("task", receipt["spec"]), ("recipe", receipt["recipe"]),
                           ("runtime", receipt["runtime"])):
        if identity.get(name + "_sha256") != stable_hash(document):
            _fail(name + " digest mismatch")
    global_recipe = {key: value for key, value in receipt["recipe"].items()
                     if key not in {"batch_size", "num_particles", "z_dim", "total_steps"}}
    if identity.get("global_recipe_sha256") != stable_hash(global_recipe):
        _fail("global recipe digest mismatch")
    spec = receipt["spec"]
    if spec.get("seed") != 0:
        _fail("performance comparisons require protocol seed 0")
    for key in ("image_size", "batch_size", "base_channels", "warmup_updates", "measured_updates"):
        _number(spec.get(key), "spec." + key, positive=key != "warmup_updates")
    status = performance_status(receipt)
    if status == "COMPLETE":
        _digest(identity.get("initialization_sha256"), "initialization digest")
        for kind, components in (("initial_model_sha256", ("generator", "discriminator", "prior")),
                                 ("batch_sequence_sha256", ("critic", "generator"))):
            for component in components:
                _digest(identity.get(kind, {}).get(component), kind + "." + component)
        validity = receipt["validity"]
        if (validity.get("passed") is not True or validity.get("full_budget") is not True
                or validity.get("finite_update_fraction") != 1):
            _fail("complete run lacks finite full-budget execution")
        metrics = receipt["metrics"]
        for key in ("update_ms_median", "update_ms_p95", "images_per_second"):
            _number(metrics.get(key), key, positive=True)
        if metrics["update_ms_p95"] < metrics["update_ms_median"]:
            _fail("p95 update time is below median")
        if metrics.get("measured_updates") != spec["measured_updates"]:
            _fail("complete run has a partial measurement budget")
        if receipt["hardware"].get("type") == "cuda":
            for key in ("peak_allocated_mib", "peak_reserved_mib"):
                _number(metrics.get(key), key)
    # Reject NaN and infinity in any nested provenance or metric, including errors.
    stable_hash(receipt)
    return receipt


def hardware_key(receipt):
    """Physical GPUs and CPU/runtime cohorts never share aggregated timings."""
    hardware = receipt["hardware"]
    return {key: value for key, value in hardware.items()
            if key not in {"contention_before", "contention_after", "contention_status"}}


def comparison_key(receipt, *, receipt_sha256=None):
    identity = receipt["identity"]
    hardware = receipt["hardware"]
    contention = hardware.get("contention_status", "unknown")
    # Unknown or contended observations remain displayable. They cannot silently
    # become a matched speed comparison with another run.
    private = None if contention == "clear" else receipt_sha256 or stable_hash(receipt)
    return stable_hash({"source": identity["source"]["sha256"],
                        "runtime": identity["runtime_sha256"], "hardware": hardware_key(receipt),
                        "task": identity["task_sha256"],
                        "initialization": identity.get("initialization_sha256"),
                        "initial_model": identity.get("initial_model_sha256"),
                        "batches": identity.get("batch_sequence_sha256"),
                        "timing_scope": TIMING_SCOPE, "contention": contention,
                        "unmatched_receipt": private})


def build_manifest(root, entries, *, tasks=None):
    """Index already compact receipts; entries own family/candidate/task labels.

    entries: [{trainer_family, candidate_id, task_id, receipt: relative/path.json}]
    tasks: optional [{id, spec}] for declared cells that have no result yet.
    No receipt is copied and no artifact is hydrated by this function.
    """
    root = Path(root).resolve()
    task_specs = {item["id"]: deepcopy(item["spec"]) for item in tasks or []}
    records = []
    for entry in entries:
        row = {key: entry[key] for key in ("trainer_family", "candidate_id", "task_id", "receipt")}
        for key in ("trainer_family", "candidate_id", "task_id"):
            identifier(row[key], key)
        path = _path(root, row["receipt"])
        receipt = validate_receipt(read_json(path))
        if receipt.get("candidate_id", row["candidate_id"]) != row["candidate_id"]:
            _fail("candidate label differs from execution receipt")
        if receipt.get("task_id", row["task_id"]) != row["task_id"]:
            _fail("task label differs from execution receipt")
        if row["task_id"] in task_specs and task_specs[row["task_id"]] != receipt["spec"]:
            _fail("one task id has different specifications")
        task_specs[row["task_id"]] = receipt["spec"]
        row["receipt_sha256"] = file_hash(path)
        row["identity_sha256"] = stable_hash(receipt["identity"])
        row["comparison_key"] = comparison_key(receipt, receipt_sha256=row["receipt_sha256"])
        records.append(row)
    manifest = {"schema_version": 1, "scope": SCOPE, "benchmark": BENCHMARK,
                "qualification_input": False, "default_adoption": False,
                "timing_scope": TIMING_SCOPE,
                "tasks": [{"id": key, "spec": spec, "spec_sha256": stable_hash(spec)}
                          for key, spec in sorted(task_specs.items())],
                "entries": sorted(records, key=lambda row: (row["trainer_family"], row["candidate_id"],
                                                            row["task_id"], row["receipt"]))}
    manifest["input_digest"] = stable_hash(manifest)
    _validate_manifest(root, manifest)
    return manifest


def _validate_manifest(root, manifest):
    check = deepcopy(manifest)
    digest = check.pop("input_digest", None)
    if digest != stable_hash(check):
        _fail("manifest digest mismatch")
    if (manifest.get("schema_version") != 1 or manifest.get("scope") != SCOPE
            or manifest.get("benchmark") != BENCHMARK
            or manifest.get("qualification_input") is not False
            or manifest.get("default_adoption") is not False
            or manifest.get("timing_scope") != TIMING_SCOPE):
        _fail("manifest scope differs from performance-only contract")
    tasks = {}
    for task in manifest["tasks"]:
        identifier(task["id"], "task id")
        if task["id"] in tasks or task["spec_sha256"] != stable_hash(task["spec"]):
            _fail("duplicate task or task digest mismatch")
        tasks[task["id"]] = task["spec"]
    recipes, seen, records = {}, set(), []
    for row in manifest["entries"]:
        for key in ("trainer_family", "candidate_id", "task_id"):
            identifier(row[key], key)
        path = _path(root, row["receipt"])
        if file_hash(path) != row["receipt_sha256"]:
            _fail("receipt bytes changed: " + row["receipt"])
        receipt = validate_receipt(read_json(path))
        if receipt.get("candidate_id", row["candidate_id"]) != row["candidate_id"]:
            _fail("candidate label differs from execution receipt")
        if receipt.get("task_id", row["task_id"]) != row["task_id"]:
            _fail("task label differs from execution receipt")
        if (row["task_id"] not in tasks or receipt["spec"] != tasks[row["task_id"]]
                or stable_hash(receipt["identity"]) != row["identity_sha256"]
                or comparison_key(receipt, receipt_sha256=row["receipt_sha256"]) != row["comparison_key"]):
            _fail("receipt task, identity or comparison key mismatch")
        recipe_key = (row["trainer_family"], row["candidate_id"], receipt["identity"]["source"]["sha256"])
        recipe = receipt["identity"]["global_recipe_sha256"]
        if recipe_key in recipes and recipes[recipe_key] != recipe:
            _fail("one candidate uses different global recipes across tasks")
        recipes[recipe_key] = recipe
        cell = (row["trainer_family"], row["candidate_id"], row["task_id"],
                receipt["identity"]["source"]["sha256"], receipt["identity"]["runtime_sha256"],
                stable_hash(hardware_key(receipt)))
        if cell in seen:
            _fail("duplicate performance cell; choose one current receipt explicitly")
        seen.add(cell)
        records.append({**row, "result": receipt})
    return records


def load_performance(root):
    root = Path(root).resolve()
    path = root / MANIFEST
    if not path.is_file():
        return None
    manifest = read_json(path)
    records = _validate_manifest(root, manifest)
    return {"manifest": MANIFEST.as_posix(), "manifest_sha256": file_hash(path),
            "input_digest": manifest["input_digest"], "tasks": manifest["tasks"],
            "timing_scope": manifest["timing_scope"], "records": records,
            "qualification_input": False, "default_adoption": False}


def attach_performance(root, publication):
    """Refresh display-only evidence, leaving selected rows and gates untouched."""
    evidence = load_performance(root)
    publication.pop("scaling_performance", None)
    publication.get("provenance", {}).pop("scaling_performance_reducer_sha256", None)
    if evidence is not None:
        publication["scaling_performance"] = evidence
        publication.setdefault("provenance", {})["scaling_performance_reducer_sha256"] = file_hash(Path(__file__))


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number_cell(value):
    return "unknown" if value is None else f"{value:,.2f}"


def _link(root, page, label, target):
    return "[" + label + "](" + os.path.relpath(root / target, page.parent) + ")"


def _selected(publication):
    labels = {family["id"]: family["label"]
              for family in publication.get("family_progress", {}).get("families", [])}
    return [(row["trainer_family"], row["candidate_id"], labels.get(row["trainer_family"], row["trainer_family"]))
            for row in publication.get("rows", []) if row.get("trainer_family")]


def render_performance(root, publication, page, *, family_id=None):
    """One section of the existing board, optionally filtered for a family page."""
    root, page = Path(root).resolve(), Path(page)
    evidence = publication.get("scaling_performance")
    lines = ["## Performance", "",
             "DCGAN on deterministic mock RGB data measures finite-update speed and memory. "
             "These separate performance-source receipts confer no image-quality, scientific qualification "
             "or default-promotion credit. Standard BatchNorm is retained in generator and critic; "
             "critic input gradients therefore include batch coupling. The shared critic uses a flattened "
             "4×4 Linear scalar head for policy feature extraction; its module layout is fixed across recipes.", ""]
    if evidence is None:
        return lines + ["Speed and scale are **unknown** until compact performance receipts are published. " +
                        _link(root, page, "Run and publication protocol", "docs/forge-scaling-performance.md") + ".", ""]
    lines += [evidence["timing_scope"], "",
              "Every cell is one run; timings are never pooled across devices or contention states. "
              "Compare only rows with the same task and matched key (source, runtime/dtype, physical device, "
              "initialization and batch streams). Unknown or contended observations are unpaired. "
              "Resolution progression changes the task, while each candidate keeps one global recipe. "
              "Images/s is B divided by full update time (one D and one G/prior update with separate real batches). "
              "Three warmup and ten measured updates give short-window median/p95 measurements; "
              "long-horizon policy costs and component phase times remain unknown.", "",
              _link(root, page, "Compact performance evidence index", evidence["manifest"]) + " · " +
              _link(root, page, "Protocol and bounded reproduction", "docs/forge-scaling-performance.md"), ""]
    selected = [item for item in _selected(publication) if family_id is None or item[0] == family_id]
    wanted = {(family, candidate) for family, candidate, _ in selected}
    records = [row for row in evidence["records"] if (row["trainer_family"], row["candidate_id"]) in wanted]
    groups = defaultdict(list)
    for row in records:
        result = row["result"]
        key = (result["identity"]["source"]["sha256"], result["identity"]["runtime_sha256"],
               stable_hash(hardware_key(result)))
        groups[key].append(row)
    if not groups:
        lines += ["No current selected-recipe performance cells are measured; all declared tasks are **unknown**.", ""]
    for (source, runtime_hash, _), rows in sorted(groups.items()):
        sample = rows[0]["result"]
        hardware, runtime = sample["hardware"], sample["runtime"]
        physical = hardware.get("physical_uuid") or hardware.get("pci_bus_id") or hardware.get("device", "unknown")
        prior = sample["spec"].get("prior", {})
        lines += ["### " + _cell(hardware.get("name", hardware.get("type", "unknown hardware"))) +
                  " · " + _cell(physical), "",
                  f"Performance source `{source[:12]}` (commit `{sample['identity']['source'].get('commit', 'unknown')}`); "
                  f"runtime `{runtime_hash[:12]}`; {_cell(runtime.get('dtype', 'unknown dtype'))}; "
                  f"PyTorch {_cell(runtime.get('torch', 'unknown'))}; "
                  f"{_cell(hardware.get('total_memory_mib', 'unknown'))} MiB device memory.", "",
                  f"Task prior: {_cell(prior.get('kind', 'unknown'))}, sigma {_cell(prior.get('sigma', 'unknown'))}, "
                  f"{sample['spec'].get('n_particles', 'unknown')} initial locations. " +
                  ("The common finite-table performance exception retains each recipe's prior update policy and regularizer. "
                   if prior.get("kind") == "particle_cloud" else "") +
                  "Exact prior and resource specifications remain part of every task key.", "",
                  "| Family / recipe | Task (RGB size, batch, width) | Execution | Median / p95 update ms | Images/s | GPU peak allocated / reserved MiB | Contention | Matched key / evidence |",
                  "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
        lookup = {(row["trainer_family"], row["candidate_id"], row["task_id"]): row for row in rows}
        for family, candidate, label in selected:
            for task in sorted(evidence["tasks"], key=lambda item: (item["spec"]["image_size"], item["spec"]["batch_size"], item["spec"]["base_channels"], item["id"])):
                spec = task["spec"]
                task_label = f"{spec['image_size']}², B{spec['batch_size']}, W{spec['base_channels']}"
                row = lookup.get((family, candidate, task["id"]))
                prefix = f"| {_cell(label)} · `{_cell(candidate)}` | {_cell(task['id'])}: {task_label} | "
                if row is None:
                    lines.append(prefix + "UNKNOWN | unknown | unknown | unknown | unknown | — |")
                    continue
                result, metrics = row["result"], row["result"]["metrics"]
                status = performance_status(result)
                reason = result.get("error", {}).get("message")
                if reason:
                    status += ": " + reason[:180]
                timings = _number_cell(metrics.get("update_ms_median")) + " / " + _number_cell(metrics.get("update_ms_p95"))
                memory = ("n/a (CPU)" if result["hardware"].get("type") != "cuda" else
                          _number_cell(metrics.get("peak_allocated_mib")) + " / " + _number_cell(metrics.get("peak_reserved_mib")))
                cells = [status, timings, _number_cell(metrics.get("images_per_second")), memory,
                         result["hardware"].get("contention_status", "unknown"),
                         "`" + row["comparison_key"][:12] + "` · " + _link(root, page, "receipt", row["receipt"])]
                lines.append(prefix + " | ".join(_cell(value) for value in cells) + " |")
        lines += [""]
    return lines


def performance_section(root, publication, page, *, family_id=None):
    return [START, "", *render_performance(root, publication, page, family_id=family_id), END, ""]


def refresh_performance(root):
    """Update only the current board's performance block and display provenance.

    Preserve all existing scientific rows, historical sections and family pages.
    This bounded path needs no archived-source extraction or model dependencies.
    """
    root = Path(root).resolve()
    json_path = root / "reports/forge/technique-inventory.json"
    page = root / "reports/forge/technique-inventory.md"
    publication = read_json(json_path)
    original = deepcopy(publication)
    claimed = original.get("provenance", {}).pop("input_digest", None)
    if claimed != stable_hash(original):
        _fail("current technique inventory digest mismatch")
    attach_performance(root, publication)
    publication["provenance"].pop("input_digest", None)
    publication["provenance"]["input_digest"] = stable_hash(publication)
    markdown = page.read_text()
    section = "\n".join(performance_section(root, publication, page))
    if START in markdown or END in markdown:
        if markdown.count(START) != 1 or markdown.count(END) != 1 or markdown.index(START) > markdown.index(END):
            _fail("current board has malformed performance section markers")
        start, end = markdown.index(START), markdown.index(END) + len(END)
        markdown = markdown[:start] + section.rstrip() + markdown[end:]
    else:
        # Put performance near the selected results while retaining the existing
        # scientific Markdown bytes and all links to historical evidence.
        anchor = "## Research recipes and separate diagnostics"
        offset = markdown.find(anchor)
        if offset < 0:
            offset = markdown.find("## Refresh")
        if offset < 0:
            offset = len(markdown)
        markdown = markdown[:offset] + section + "\n" + markdown[offset:]
    old_digest = claimed
    markdown = markdown.replace(f"Publication input digest `{old_digest}`.",
                                f"Publication input digest `{publication['provenance']['input_digest']}`.")
    json_text = json.dumps(publication, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if json_path.read_text() != json_text:
        atomic_text(json_path, json_text)
    if page.read_text() != markdown:
        atomic_text(page, markdown)
    return {"report": page.relative_to(root).as_posix(), "training_launched": False,
            "scientific_rows_preserved": True, "qualification_regraded": False,
            "records": len(publication.get("scaling_performance", {}).get("records", []))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--entries", type=Path, help="JSON object with entries and optional declared tasks")
    parser.add_argument("--check", action="store_true", help="validate the current compact index without writing")
    args = parser.parse_args(argv)
    if args.check:
        evidence = load_performance(args.root)
        print(json.dumps({"available": evidence is not None,
                          "records": len(evidence["records"]) if evidence else 0,
                          "training_launched": False}, sort_keys=True))
    elif args.entries:
        declaration = read_json(args.entries)
        manifest = build_manifest(args.root, declaration["entries"], tasks=declaration.get("tasks"))
        atomic_json(args.root / MANIFEST, manifest)
        print(json.dumps({"manifest": MANIFEST.as_posix(), "records": len(manifest["entries"]),
                          "input_digest": manifest["input_digest"], "training_launched": False}, sort_keys=True))
    else:
        parser.error("pass --entries or --check")


if __name__ == "__main__":
    main()
