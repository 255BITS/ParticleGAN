"""Regrade retained geometry evidence and write the compact result ledger.

This is a read-only audit of the ignored local raw artifacts. It does not
train, select seeds, change a threshold, or infer passes from log text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_gate
from benchmarks.toy100.gate import evaluate_suite as coverage_gate
from benchmarks.toy_suite import _check_toy100_policy, _episode_rows
from benchmarks.transfer_suite.public_default_verification import load_declaration
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "artifacts/toy100-constraints-geometry-v1"
MOMENT = ROOT / "artifacts/toy100-constraints-geometry-moment-v1"
RAM_V1 = Path("/dev/shm/particlegan-constraints-geometry-v1-346b1ed")
RAM_MOMENT = Path("/dev/shm/particlegan-constraints-geometry-moment-v1-3eb182f")
OUTPUT = ROOT / "reports/toy100/constraints_geometry_results.json"


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> tuple[dict[str, str], int]:
    files = {str(path.relative_to(root)): digest(path)
             for path in root.rglob("*") if path.is_file()}
    return files, sum((root / name).stat().st_size for name in files)


def retention(original: Path, durable: Path) -> dict:
    source, source_bytes = inventory(original)
    copied, copied_bytes = inventory(durable)
    if source != copied or source_bytes != copied_bytes:
        raise ValueError(f"retained files differ: {original}, {durable}")
    tree = hashlib.sha256()
    for name, file_hash in sorted(copied.items()):
        tree.update(name.encode() + b"\0" + file_hash.encode() + b"\n")
    return dict(durable_local_path=str(durable.relative_to(ROOT)),
                original_ram_path=str(original), files=len(copied),
                bytes=copied_bytes, tree_sha256=tree.hexdigest(),
                all_file_sha256_equal=True)


def native(directory: Path, problem: str, config_path: Path, sources: dict) -> dict:
    config = read(config_path)
    episode = directory / problem
    summary = read(episode / "summary.json")
    if summary["provenance"]["source_sha256"] != sources:
        raise ValueError(f"source identity differs: {directory}")
    _check_toy100_policy(
        episode, summary, summary["config"], declared_model_policy(config),
    )
    coverage = coverage_gate(directory, problem=problem, write=False)["problems"][problem]
    accuracy = accuracy_gate(directory, problem=problem, write=False)["problems"][problem]
    strict = coverage["status"] == "PASS" and accuracy["status"] == "PASS"
    return dict(
        strict_status="PASS" if strict else "FAIL",
        coverage_status=coverage["status"], accuracy_status=accuracy["status"],
        terminal_passes=sum(check["passed"] for check in accuracy["terminal_checks"]),
        terminal_steps=[check["step"] for check in accuracy["terminal_checks"]],
        terminal_pass_flags=[check["passed"] for check in accuracy["terminal_checks"]],
        final_coverage=summary["final"]["live"],
        final_accuracy=accuracy["final_metrics"],
        holdout_accuracy=accuracy["holdout_metrics"],
        model_policy=summary["model_policy"],
        source_archive_sha256=summary["provenance"]["source_archive_sha256"],
        local_evidence_path=str(directory.relative_to(ROOT)),
    )


def transfer(directory: Path, names: tuple[str, ...]) -> dict:
    result = _episode_rows(directory, names, candidate=True)
    if result["status"] in ("INVALID", "MISSING"):
        raise ValueError(f"transfer archive invalid: {directory}: {result.get('reason')}")
    return dict(status=result["status"], passed=result["passed"],
                required=result["required"],
                cases={name: row["status"] for name, row in result["cases"].items()},
                local_evidence_path=str(directory.relative_to(ROOT)))


def main() -> None:
    v1_manifest = read(ROOT / "reports/toy100/constraints_geometry_v1_manifest.json")
    simple_manifest = read(ROOT / "reports/toy100/constraints_geometry_simple_interaction_manifest.json")
    moment_manifest = read(ROOT / "reports/toy100/constraints_geometry_moment_manifest.json")
    rows = []
    configs = {}
    for entry in v1_manifest["rows"]:
        path = ROOT / entry["config"]
        if digest(path) != entry["config_sha256"]:
            raise ValueError(f"grid config changed: {path}")
        configs[entry["slug"]] = path
        rows.append(dict(slug=entry["slug"], delta=entry["delta"],
                         config_path=entry["config"],
                         config_sha256=entry["config_sha256"],
                         grid100=native(V1 / "rows" / entry["slug"] / "grid100",
                                        "grid100", path, v1_manifest["source_sha256"])))
    promotions = {}
    for slug, folder in (("affine_empirical_box", "empirical"),
                         ("critic64", "critic64"),
                         ("critic_depth2", "critic_depth2"),
                         ("lean_joint", "lean_joint")):
        promotions[slug] = {
            name: native(V1 / f"promotion_{folder}" / name, name,
                         configs[slug], v1_manifest["source_sha256"])
            for name in ("rotated100", "staggered100")
        }
    simple_config = ROOT / simple_manifest["candidate_config_path"]
    moment_config = ROOT / moment_manifest["candidate_config"]
    if digest(simple_config) != simple_manifest["candidate_config_sha256"]:
        raise ValueError("simple empirical config changed")
    if digest(moment_config) != moment_manifest["candidate_sha256"]:
        raise ValueError("moment config changed")
    old19_names = tuple(job["spec"]["name"] for job in load_declaration()[0])
    simple_root = V1 / "simple_interaction"
    simple = dict(
        config_path=str(simple_config.relative_to(ROOT)),
        config_sha256=simple_manifest["candidate_config_sha256"],
        old_six=transfer(simple_root / "old6", tuple(simple_manifest["old_screen"])),
        old_nineteen=transfer(simple_root / "old19", old19_names),
        native={name: native(simple_root / name, name, simple_config,
                             simple_manifest["source_sha256"])
                for name in simple_manifest["native_order"]},
    )
    moment = dict(
        config_path=str(moment_config.relative_to(ROOT)),
        config_sha256=moment_manifest["candidate_sha256"],
        old_six=transfer(MOMENT / "old6", tuple(moment_manifest["old_screen"])),
        grid100=native(MOMENT / "grid100", "grid100", moment_config,
                       moment_manifest["source_sha256"]),
        rotated100="SKIPPED: strict grid gate failed",
        staggered100="SKIPPED: strict grid gate failed",
    )
    ledger = dict(
        protocol="toy100-geometry-constraint-results-v1",
        generation="independent regrade of frozen archives; no training",
        fixed_native_seed=1234, fixed_native_steps=7000,
        fixed_terminal_checks=5, terminal_samples_per_check=20_000,
        independent_holdout_samples=100_000,
        original_recipe_grid_rows=rows, original_recipe_promotions=promotions,
        simple_core_empirical_interaction=simple, simple_core_moment_box=moment,
        retention=dict(
            v1=retention(RAM_V1, V1),
            moment=retention(RAM_MOMENT, MOMENT),
        ),
    )
    OUTPUT.write_text(json.dumps(ledger, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(f"wrote {OUTPUT}: {len(rows)} grid rows, "
          f"{sum(row['grid100']['strict_status']=='PASS' for row in rows)} grid passes, "
          f"{len(promotions) * 2} native promotions, "
          f"simple19={simple['old_nineteen']['passed']}/19, "
          f"moment6={moment['old_six']['passed']}/6, "
          f"moment_grid={moment['grid100']['strict_status']}")


if __name__ == "__main__":
    main()
