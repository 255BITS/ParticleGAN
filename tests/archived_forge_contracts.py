"""Exact archived declarations for software controls, never current qualification.

The fixtures preserve original UTF-8 bytes and Git blob identities. Tests replay
the published checkout's contracts in private directories; live declarations,
published selections and saved evidence are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/archived-forge-contracts.json"
COMMITS = {
    "published_develop": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "published_develop_views": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "tier1_completion": "928b485ffbe6e17d79b41d307ae8b6275489b37a",
    "shallow_gaussian": "1853cfdaf31c243c8a1e75a86f370ba143691d5a",
    "shallow_gaussian_cards": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "bcap_legacy_questions": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "bcap_prior_baseline_questions": "03466efa4de8271b9a2c964406dc6f4f0f260792",
}


def archived_contract_bytes(scope, relative):
    packet = json.loads(FIXTURE.read_text())
    assert packet["schema"] == "forge_archived_software_contract_fixtures_v1"
    assert packet["qualification_input"] is False
    cohort = packet[scope]
    assert cohort["source_commit"] == COMMITS[scope]
    record = cohort["files"][relative]
    # Unchanged committed declarations need only their original byte pins. Any
    # later edit fails here and requires explicitly archiving the old bytes.
    data = ((ROOT / relative).read_bytes() if record.get("repository_bytes") is True
            else record["text"].encode("utf-8"))
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == record["git_blob"]
    return data


def restore_archived_contracts(root, scope):
    """Restore only private fixture bytes, with an explicit immutable origin."""
    root = Path(root).resolve()
    assert root != ROOT.resolve()
    packet = json.loads(FIXTURE.read_text())
    for relative in packet[scope]["files"]:
        path = root / relative
        assert path.resolve().is_relative_to(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archived_contract_bytes(scope, relative))
    return root


def restore_published_view_roster(root):
    """Replay exactly the views declared by the original published report."""
    root = Path(root).resolve()
    assert root != ROOT.resolve()
    packet = json.loads(FIXTURE.read_text())
    originals = set(packet["published_develop_views"]["files"])
    directory = root / "configs/forge/views"
    assert directory.resolve().is_relative_to(root)
    for declaration in directory.glob("*.json"):
        assert declaration.resolve().is_relative_to(root)
        if declaration.relative_to(root).as_posix() not in originals:
            declaration.unlink()
    return restore_archived_contracts(root, "published_develop_views")


def commit_available(commit):
    """True when this checkout can read the archived commit (shallow clones cannot)."""
    return subprocess.run(["git", "cat-file", "-e", commit + "^{commit}"], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def require_history(scope):
    """Skip archived-history controls with an explicit reason instead of erroring."""
    import pytest
    commit = COMMITS[scope]
    if not commit_available(commit):
        pytest.skip(f"archived {scope} commit {commit[:12]} is unavailable in this checkout "
                    "(shallow clone); fetch full history to run this control")


def published_develop_checkout(tmp_path):
    """Read original committed reports against their published develop cards."""
    # These report controls only read their inputs; share one complete physical
    # report mirror per pytest run so containment guards retain their meaning.
    require_history("published_develop")
    root = tmp_path.parent / "published-develop-checkout"
    complete = root.with_name(root.name + ".complete")
    if complete.is_file() and root.is_dir():
        return root
    if root.exists():
        # An earlier call failed part-way; never hand out a partial mirror.
        shutil.rmtree(root)
    root.mkdir()
    # Reports and artifact links are read-only inputs in these controls. Configs
    # are private copies, allowing exact original task identities to be restored.
    for path in ROOT.iterdir():
        if path.name not in {".git", "configs", "reports"}:
            destination = root / path.name
            if path.name in {"particlegan", "experiments", "benchmarks", "docs"}:
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copyfile(path, destination)
            else:
                destination.symlink_to(path, target_is_directory=path.is_dir())
    shutil.copytree(ROOT / "configs", root / "configs")
    # The publication's recorded input manifest is sufficient for these
    # read-only controls. Do not copy unrelated research logs or checkpoints.
    commit = COMMITS["published_develop"]
    report_path = "reports/forge/technique-inventory.json"
    report_bytes = subprocess.check_output(["git", "show", f"{commit}:{report_path}"], cwd=ROOT)
    publication = json.loads(report_bytes)
    for relative, digest in publication["family_progress"]["input_hashes"].items():
        if not relative.startswith("reports/"):
            continue
        source, destination = ROOT / relative, root / relative
        data = source.read_bytes() if source.is_file() else None
        if data is None or hashlib.sha256(data).hexdigest() != digest:
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(data).hexdigest() == digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    restore_archived_contracts(root, "published_develop")
    restore_archived_contracts(root, "bcap_legacy_questions")
    # Current publication refreshes must not enter these historical controls.
    # Restore their original compact report and whole-selection inputs from
    # the exact immutable publication commit, without hydrating raw artifacts.
    for relative in ("reports/forge/technique-inventory.json",
                     "reports/forge/technique-inventory.md",
                     "configs/forge/selections/family-current-v1.json",
                     "configs/forge/trainer-families.json"):
        data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        blob = subprocess.check_output(["git", "rev-parse", f"{commit}:{relative}"], cwd=ROOT,
                                       text=True).strip()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blob
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)
    # Documentation validates its local source and illustration targets too;
    # those navigation-only files are outside the numerical input manifest.
    family_ids = {row["trainer_family"] for row in
                  publication["rows"] + publication.get("historical_family_rows", [])}
    navigation = set()
    for family_id in family_ids:
        documentation = json.loads((root / "configs/forge/family-documentation" /
                                    (family_id + ".json")).read_text())
        navigation.update(path for path in documentation["sources"] if path.startswith("reports/"))
        illustration = documentation.get("illustration", {}).get("path", "")
        if illustration.startswith("reports/"):
            navigation.add(illustration)
    for relative in sorted(navigation):
        destination = root / relative
        if not destination.is_file():
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    result = restore_published_view_roster(root)
    complete.write_text("complete\n")
    return result


def bcap_legacy_software_checkout(directory):
    """Current code with exact original BCAP questions, only for software replay.

    The frozen comparator reads Git history but never writes it. Its private
    checkout therefore has a read-only history pointer; no admission helper
    may use this checkout. Current migrated questions are tested separately.
    """
    root = Path(directory).resolve()
    assert root != ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("particlegan", "experiments", "benchmarks", "lib", "configs"):
        shutil.copytree(ROOT / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Frozen evaluators may import report-owned Python. No bulk logs, tensor
    # dumps, old metrics or qualification receipts enter this software replay.
    report_sources = list((ROOT / "reports").rglob("*.py"))
    report_sources.append(ROOT / "reports/transfer_suite/unadjusted/leading_profile.json")
    for source in report_sources:
        destination = root / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    restore_archived_contracts(root, "bcap_legacy_questions")
    # Current code has explicit source guards. Rebind only their byte identity;
    # preserve the original question and exact ancestry expected by the frozen
    # comparator's unchanged source-revision assertions.
    cohort = json.loads(FIXTURE.read_text())["bcap_legacy_questions"]
    rebinding = {}
    for relative in cohort["files"]:
        if "/tasks/" not in relative:
            continue
        path = root / relative
        card = json.loads(path.read_text())
        sources = card["evaluation"].get("sources", {})
        changed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name, old in sources.items() if (root / name).is_file()
                   and hashlib.sha256((root / name).read_bytes()).hexdigest() != old}
        if changed:
            previous = dict(sources)
            sources.update(changed)
            card["evaluation"]["evaluator_revision"] = {
                "id": "bcap-legacy-prior-software-source-binding-v1",
                "qualification_input": False,
                "original_task_commit": cohort["source_commit"],
                "original_task_path": relative,
                "previous_revision": card["evaluation"].get("evaluator_revision"),
                "previous_source_sha256": {name: previous[name] for name in changed},
            }
            path.write_text(json.dumps(card, indent=2) + "\n")
            rebinding[relative] = {name: {"original_sha256": previous[name],
                                        "current_sha256": digest}
                                  for name, digest in changed.items()}
    (root / "legacy-software-source-bindings.json").write_text(json.dumps({
        "qualification_input": False, "original_task_source_commit": cohort["source_commit"],
        "source_rebindings": rebinding}, indent=2) + "\n")
    gitdir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                     cwd=ROOT, text=True).strip()
    (root / ".git").write_text("gitdir: " + gitdir + "\n")
    return root


def current_policy_software_checkout(tmp_path):
    """Original parent questions plus freshly bound public-policy source bytes."""
    from experiments.forge.planning import resolve_idea
    from experiments.forge.tier1_policy import write_declarations
    root = tmp_path / "policy-software-checkout"
    source = resolve_idea(ROOT, "k3p")["source"]
    for relative in source["files"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / "configs/forge", root / "configs/forge", dirs_exist_ok=True)
    restore_archived_contracts(root, "published_develop")
    write_declarations(root)
    return root
