"""Record the independently checked composition; writes only this review directory."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
BASELINE = ROOT.parent / "feature-cells-cuda-retest-20260929"
SHARED = ROOT / "pkg-CB64-RA2"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nodes(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        result[node.name + "." + method.name] = ast.dump(method, include_attributes=False)
    return result


source_map = {str(p.relative_to(SHARED)): sha(p) for p in sorted(SHARED.rglob("*.py"))}
digest = hashlib.sha256()
for path in sorted((SHARED / "particlegan").rglob("*.py")):
    digest.update(str(path.relative_to(SHARED / "particlegan")).encode() + b"\0" + path.read_bytes() + b"\0")
assert source_map["particlegan/feature_cells.py"] == "080dccbc404e1fe4112cd18275eaf94b8f68e6086996b88b5f9b5718770573c1"
assert source_map["particlegan/training.py"] == "28f6cc3f01308829531d4fab655db4cfd1870354d9c81f9374175a8b324088d9"
assert source_map["particlegan/birth_death.py"] == "447f67de1ed80eec0e260c88808cd40e9f682bf55e1ef6c39e6dc513594c8b72"

owners = {
    "geometry": (ROOT / "geometry/pkg-CB64-RA2/particlegan/feature_cells.py", [
        "BoundedLatentGeometry", "FeatureCellBirthDeath._jitter", "FeatureCellBirthDeath.perturb_latent",
        "FeatureCellBirthDeath._capture_generated"]),
    "performance": (ROOT / "performance/pkg/particlegan/feature_cells.py", [
        "conditional_count_pvalues", "FeatureCellSnapshot.fit", "FeatureCellSnapshot._pool"]),
    "stability": (ROOT / "stability/pkg/particlegan/feature_cells.py", [
        "population_policy", "SmallPopulationReferenceBirthDeath", "make_feature_cell_birth_death",
        "_integer_allocate", "FeatureCellSnapshot._mass_targets", "FeatureCellSnapshot._mass_topology",
        "FeatureCellSnapshot._group_counts", "FeatureCellSnapshot.select_parents"]),
    "nonflagged_correction": (REVIEW / "pkg-nonflagged-correction/particlegan/feature_cells.py", [
        "FeatureCellSnapshot.ordinary_transport", "FeatureCellBirthDeath.check_state", "FeatureCellBirthDeath.load_state_dict"]),
}
shared_nodes = nodes(SHARED / "particlegan/feature_cells.py")
composition = {}
for owner, (path, methods) in owners.items():
    expected = nodes(path)
    checks = {name: shared_nodes[name] == expected[name] for name in methods}
    assert all(checks.values()), (owner, checks)
    composition[owner] = dict(source=str(path), sha256=sha(path), identical_ast=checks)
assert sha(SHARED / "particlegan/training.py") == sha(ROOT / "stability/pkg/particlegan/training.py")

old_ready = read(ROOT / "stability/READY.json")
preserved = {}
for key in ("package_source_sha256", "local_source_sha256", "evidence_sha256"):
    preserved[key] = all(sha(ROOT / "stability" / path) == expected for path, expected in old_ready[key].items())
assert all(preserved.values()), preserved
preserved["ready_sha256_unchanged"] = sha(ROOT / "stability/READY.json") == "0d2104e017b2871cfb16e8f8b5c1fc643de284767238aa142ea292685631a373"
assert preserved["ready_sha256_unchanged"]

baseline_manifest = read(BASELINE / "screens/artifact_manifest.json")
baseline_preserved = all(sha(BASELINE / "screens" / name) == value["sha256"] for name, value in baseline_manifest["files"].items())
assert baseline_preserved

stability = read(REVIEW / "regressions.json")
geometry = read(REVIEW / "test-results.json")
flagged = read(REVIEW / "flagged-regression-shared.json")
assert stability["success"] and stability["tests_run"] == 8
assert geometry["passed"] and geometry["tests"] == 5 and not geometry["cuda_initialized"]
assert flagged["success"] and flagged["tests"] == 2 and Path(flagged["package_root"]) == SHARED
assert {name.removeprefix("pkg-CB64-RA2/"): value for name, value in geometry["source_sha256"].items()} == source_map
gpu = read(REVIEW / "mass-gpu-after/result.json")
before = read(REVIEW / "mass-gpu-before/result.json")
assert gpu["status"] == "PASS" and gpu["source_unchanged"]
assert gpu["source_sha256_before"] == gpu["source_sha256_after"] == source_map
assert all(all(row["checks"].values()) for row in gpu["records"])
assert before["status"] == "ERROR"
failed_checks = {row["scenario"]: [name for name, passed in row["checks"].items() if not passed] for row in before["records"]}
assert failed_checks == {"nominal": [], "rare_hole": ["zero_cross_group_ordinary"]}

monitor = read(REVIEW / "monitor-baseline-check/summary.json")
old_board = read(BASELINE / "screens/leaderboard.json")
expected = {row["task"]: (row["primary_status"], row["canonical_fixture_validity"], row["acceptance_status"]) for row in old_board["tasks"]}
actual = {row["task"]: (row["primary_status"], row["canonical_fixture_validity"], row["acceptance_status"]) for row in monitor["records"]}
assert actual == expected and monitor["completed"] == monitor["total"] == 16
assert monitor["counts"] == old_board["acceptance_counts"]
monitor_identity = read(REVIEW / "monitor-baseline-check/CHECKER-IDENTITY.json")
assert monitor_identity["checker_source_sha256"] == sha(REVIEW / "monitor_validation.py")

case_fields = ("scenario", "ordinary_moves", "isolation_moves", "unique_parents", "mass_tv", "rare_ratio",
               "repair_recall", "intended_parent_agreement", "groups", "ordinary_between_group_moves", "checks")
evidence_files = ["regressions.json", "test-results.json", "flagged-regression-shared.json",
                  "stability-shared-after.log", "geometry-shared-after.log", "flagged-regression-shared.log",
                  "mass-gpu-before/result.json", "mass-gpu-after/result.json", "mass-cpu-before/result.json", "mass-cpu-after/result.json",
                  "monitor-baseline-check/summary.json", "monitor-baseline-check/CHECKER-IDENTITY.json",
                  "monitor-baseline-check/READ-ONLY-ARTIFACT-MANIFEST.json", "monitor-baseline-check.log"]
local_sources = ["test_stability_shared.py", "test_geometry_shared.py", "test_flagged_surplus.py",
                 "mass_check_review.py", "monitor_validation.py", "NONFLAGGED-SURPLUS.patch", "record_review.py"]
receipt = dict(
    status="INTEGRATED_CONTRACTS_PASS_FULL_QUALITY_PENDING", created_at=datetime.now(timezone.utc).isoformat(),
    scope="Reviewed shared geometry/performance/stability composition plus nonflagged-surplus correction; later support changes require their own review.",
    package_root=str(SHARED), package_sha256=digest.hexdigest(), source_sha256=source_map,
    composition=composition, training_source_exact_stability=True, original_reference_birth_death_unchanged=True,
    private_stability_evidence_preserved=preserved, old_baseline_manifest_unchanged=baseline_preserved,
    old_baseline_checked_files=len(baseline_manifest["files"]),
    cpu_regressions=dict(total=15, failures=0, errors=0, stability=stability, geometry=geometry, flagged_surplus=flagged),
    gpu_mass_contract=dict(status=gpu["status"], device=gpu["device"], uuid=gpu["uuid"], seed=gpu["seed"],
                           inputs_sha256=gpu["inputs_sha256"], source_unchanged=True, peak_reserved_mib=gpu["peak_reserved_mib"],
                           cases=[{key: row[key] for key in case_fields} for row in gpu["records"]]),
    integration_bug=dict(failed_checks=failed_checks,
        cause="Flagged holes were counted as ordinary population surplus, permitting a legitimate supported survivor to be deleted before isolation.",
        correction="Ordinary surplus/vacancy counts use every nonflagged survivor; parent eligibility stays stricter. Full table counts remain diagnostic. Mass policy v3 rejects v2 continuation.",
        count_evidence_and_quality_gates_unchanged=True,
        original_failed_receipts_preserved=[str(ROOT / "stability/gpu-integrated/result.json"), str(REVIEW / "mass-gpu-before/result.json")]),
    collector_check=dict(status="PASS", completed_saved_baseline_screens=16, exact_primary_and_canonical_verdicts=True,
                         writes_only_review=True, baseline_bytes_unchanged=True, quality_verdict="none; saved-artifact verification",
                         upcoming_outputs=str(REVIEW / "validation-monitor")),
    local_source_sha256={name: sha(REVIEW / name) for name in local_sources},
    evidence_sha256={name: sha(REVIEW / name) for name in evidence_files},
    config_recommendation="Retain existing feature_cells / cells64 / rank8 / chunk256 / real_anchor settings; no recipe changes.",
    pending=["Support-score correction review and final freeze", "Full learned and canonical CUDA quality validation"])
(REVIEW / "REVIEW.json").write_text(json.dumps(receipt, indent=2) + "\n")
lines = [
    "# Integrated stability review", "", "Status: focused contracts PASS; full CUDA quality validation pending.", "",
    "The shared package passes 8 stability, 5 adaptive geometry and 2 flagged-surplus CPU regressions. The original reference backend remains byte-identical. Factory routing, small-population reference law, matching sampler, checkpoint rejection, caches, gradients, optimizer rows and unique parent supply are covered.", "",
    "## Concrete integration correction", "",
    "The original frozen rare-hole input exposed one cross-group ordinary move: flagged contamination was treated as legitimate population surplus. Excluding flagged rows from ordinary population counts preserves every supported survivor, including p≤Q rows that cannot seed births. Isolation accounts for the combined planned ordinary moves. Parents selected for deletion cannot seed ordinary or isolation births. Count evidence, Q, guard arithmetic and quality gates remain unchanged. Mass-policy v3 rejects incompatible v2 continuation.", "",
    "The original GPU failure and its traceback remain preserved. Root applied NONFLAGGED-SURPLUS.patch and reran every unchanged named assertion on physical GPU0.", "",
    "| Frozen case | Ordinary | Repairs | Distinct parents | Mass TV | Rare/target | Intended agreement |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in gpu["records"]:
    lines.append(f"| {row['scenario']} | {row['ordinary_moves']} | {row['isolation_moves']} | {row['unique_parents']} | {row['mass_tv']} | {row['rare_ratio']} | {row['intended_parent_agreement']:.6f} |")
lines += ["", "The mass contract reserves 34 MiB on GPU0. This is planning on fixed saved inputs, without training. Rare-hole intended identity is erased by the fixture; its agreement is disclosed and has no identity-recovery gate. Geometry and performance function ASTs match their separately verified implementations.", "",
    "## Read-only validation monitor", "",
    "monitor_validation.py waits for the future validation freeze, checks its immutable local/external maps, and imports that lane’s original canonical collector. Only collector JSON writes are redirected to integration/review; validation bytes stay read-only. Active jobs remain PENDING. Completed jobs retain exact original PASS/FAIL/ERROR and mandatory fixture validity, including official native clouds, schedules and holdout. The source-freeze identity is retained across polls. Final saved inputs receive a read-only hash manifest.", "",
    "The monitor reproduced all 16 completed baseline verdicts (9/13 portability PASS and 0/3 native PASS). Every file in the baseline screen manifest and every declared original stability source/evidence hash remains unchanged.", "",
    "Root may run:", "", "```bash",
    f"/tmp/pr38-default-env/bin/python -u -B {REVIEW / 'monitor_validation.py'} --watch",
    "```", "", "Reports: integration/review/validation-monitor/summary.json and REPORT.md. The monitor creates no CUDA context or numerical job.", "",
    f"Reviewed feature_cells.py SHA: {source_map['particlegan/feature_cells.py']}",
    f"Reviewed training.py SHA: {source_map['particlegan/training.py']}", "",
    "Recommendation: retain the recipe fields and perform the authorized complete CUDA validation after the support decision and final freeze. These focused contracts do not establish training quality.", ""]
(REVIEW / "REVIEW.md").write_text("\n".join(lines))
print(json.dumps(dict(status=receipt["status"], package_sha256=receipt["package_sha256"],
                     cpu_tests=15, gpu_contract="PASS", checker_sha256=sha(REVIEW / "monitor_validation.py"),
                     review_sha256=sha(REVIEW / "REVIEW.json"))), flush=True)
