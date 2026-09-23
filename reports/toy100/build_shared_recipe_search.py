"""Regenerate the audited transfer-search ledger from saved episode evidence.

This is a read-only audit of training evidence. The output files live beside
this script; original protocols, configs, and compressed episodes are never
changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_suite import _episode_rows
from benchmarks.transfer_suite.compare_defaults import plan


ARTIFACTS = ROOT / "artifacts/toy100-accuracy/compatibility"
HERE = Path(__file__).resolve().parent
ALL_NAMES = tuple(job["spec"]["name"] for job in plan())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local(path: Path) -> str:
    return "`" + str(path.relative_to(ROOT)) + "`"


def _legacy_schedule_matches_archive(directory: Path, recipe: dict, names: list[str]):
    if not any(name in ALL_NAMES[:9] for name in names):
        return True, "trainer-only screen"
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        control = archive.extractfile("benchmarks/transfer_suite/compare_defaults.py")
        if control is None:
            raise ValueError("control implementation absent from source archive")
        source = control.read()
    class_start = source.index(b"class RecipeControl")
    class_end = source.index(b"with ExitStack", class_start)
    controller = source[class_start:class_end]
    if (b"learning_rate_scale(" in controller
            and b"recipe.lr_anneal_start" in controller
            and b"recipe.lr_floor" in controller):
        return True, "legacy controller read candidate schedule"
    matched = recipe["lr_anneal_start"] == 0.6 and recipe["lr_floor"] == 0.05
    return matched, ("historical fixed .6/.05 schedule matches candidate"
                     if matched else "historical legacy .6/.05 schedule differs from candidate")


def build():
    rows = []
    for summary_path in sorted(ARTIFACTS.glob("*/summary.json")):
        directory = summary_path.parent
        summary = json.loads(summary_path.read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        names = summary.get("tasks") or [row["name"] for row in summary["cases"]]
        control = protocol["version"] == "public-default-verification-v1"
        recipe = (summary.get("global_recipe") or protocol.get("base_get_recipe")
                  or protocol.get("global_recipe"))
        noise = summary.get("noise") or protocol.get("noise")
        if control:
            schedule_valid, schedule_reason = True, "installed-wheel public v3 control"
        else:
            schedule_valid, schedule_reason = _legacy_schedule_matches_archive(
                directory, recipe, names,
            )
        grade = _episode_rows(directory, tuple(names), candidate=not control)
        complete19 = len(names) == 19 and set(names) == set(ALL_NAMES)
        mechanism = bool(summary.get("full_mechanism")) if not control else None
        reasons = []
        if not control:
            if not complete19:
                reasons.append("subset only")
            if not mechanism:
                reasons.append("noise mechanism not verified on every selected host")
            if not schedule_valid:
                reasons.append(schedule_reason)
        if grade["status"] == "INVALID":
            reasons.append("strict regrade: " + grade["reason"])
        config_file = protocol.get("config_file")
        config_path = directory / config_file if config_file else None
        source_archive = directory / "source.tar.gz"
        rows.append(dict(
            run=directory.name,
            kind="installed-wheel control" if control else "shared candidate screen",
            observed_live_passes=summary["passed"], attempted=summary["attempted"],
            observed_overall=summary["overall"],
            strict_regrade_status=grade["status"],
            strict_regrade_passes=grade["passed"],
            strict_regrade_reason=grade.get("reason"),
            valid_shared_schedule=schedule_valid,
            schedule_reason=schedule_reason,
            noise_mechanism_verified=mechanism,
            complete_19=complete19,
            eligible_full19_evidence=(not control and complete19 and mechanism
                                      and schedule_valid and grade["status"] in ("PASS", "FAIL")),
            limitations=reasons,
            selected_tasks=names,
            passing_tasks=[case["name"] for case in summary["cases"] if case["live"] == "PASS"],
            failing_tasks=[case["name"] for case in summary["cases"] if case["live"] != "PASS"],
            recipe=recipe,
            noise=noise,
            provenance=dict(
                summary=str(summary_path.relative_to(ROOT)),
                protocol=str((directory / "protocol.json").relative_to(ROOT)),
                config=None if config_path is None else str(config_path.relative_to(ROOT)),
                config_sha256=protocol.get("config_sha256"),
                source_archive=str(source_archive.relative_to(ROOT)),
                source_archive_sha256=_digest(source_archive),
                compare_defaults_sha256=protocol["source_sha256"].get(
                    "benchmarks/transfer_suite/compare_defaults.py"),
                noise_source_sha256=protocol.get("noise_source_sha256"),
                protocol_version=protocol["version"],
                seed=protocol.get("seed"), device=protocol.get("device"),
                threads=protocol.get("threads"),
            ),
        ))
    output = dict(protocol="shared-recipe-search-ledger-v1", candidate_selection="no seed search",
                  note="Observed subset and schedule-mismatched scores are diagnostics, not 22-case passes.",
                  rows=rows)
    (HERE / "shared-recipe-search.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
    )
    controls = [row for row in rows if row["kind"] == "installed-wheel control"]
    candidates = [row for row in rows if row["kind"] != "installed-wheel control"]
    lines = ["# Shared-recipe transfer search ledger", "",
             "Every row uses seed 0 and one CPU thread, frozen host budgets, data,",
             "architectures, 24 checkpoints, and live thresholds. Raw observed",
             "counts remain visible even when an old custom-host controller applied",
             "`.6/.05` instead of the candidate's `.4/.05` LR schedule. Such rows",
             "are diagnostics and cannot prove one recipe across 22 toys. Raw",
             "artifacts are retained in the local workspace; their code-formatted",
             "paths below are not GitHub links.", "",
             "The installed-wheel public-v3 control is separate from candidate evidence:", ""]
    for row in controls:
        lines.append(f"- `{row['run']}`: "
                     f"{row['observed_live_passes']}/19 live, strict regrade "
                     f"{row['strict_regrade_status']}; raw summary "
                     f"{_local(ROOT / row['provenance']['summary'])}; source archive "
                     f"{_local(ROOT / row['provenance']['source_archive'])}.")
    lines += ["", "| Candidate evidence | Raw live | Strict regrade | Validity | Exact config |",
              "| --- | ---: | --- | --- | --- |"]
    for row in candidates:
        validity = ("full 19, same schedule/noise" if row["eligible_full19_evidence"]
                    else "; ".join(row["limitations"]) or "subset diagnostic")
        config = row["provenance"]["config"]
        config_cell = (f"{_local(ROOT / config)} · `{row['provenance']['config_sha256'][:12]}`"
                       if config else "—")
        lines.append(f"| `{row['run']}`; {_local(ROOT / row['provenance']['summary'])} "
                     f"| {row['observed_live_passes']}/{row['attempted']} "
                     f"| {row['strict_regrade_status']} "
                     f"| {validity} | {config_cell} |")
    lines += ["", "The [JSON ledger](shared-recipe-search.json) records every selected task,",
              "pass/fail case, full resolved recipe and noise policy, exact config digest,",
              "source-archive digest, and custom-host schedule validity. A PASS on a",
              "subset never counts as 19/19 or 22/22. The combined 22-toy gate is",
              "[`benchmarks/toy_suite.py`](../../benchmarks/toy_suite.py).", ""]
    (HERE / "shared-recipe-search.md").write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    built = build()
    print(json.dumps(dict(rows=len(built["rows"]), report=str(HERE / "shared-recipe-search.md"))))
