"""Exact archived declarations for software controls, never current qualification.

The fixtures preserve original UTF-8 bytes and Git blob identities. Tests replay
the published checkout's contracts in private directories; live declarations,
published selections and saved evidence are never rewritten.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/archived-forge-contracts.json"
COMMITS = {
    "published_develop": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "published_develop_views": "5737ade47dca89b04d338ada20667781d1f2b5df",
    "tier1_completion": "928b485ffbe6e17d79b41d307ae8b6275489b37a",
    "shallow_gaussian": "1853cfdaf31c243c8a1e75a86f370ba143691d5a",
    "shallow_gaussian_cards": "5737ade47dca89b04d338ada20667781d1f2b5df",
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


def published_develop_checkout(tmp_path):
    """Read original committed reports against their published develop cards."""
    # These report controls only read their inputs; share one complete physical
    # report mirror per pytest run so containment guards retain their meaning.
    root = tmp_path.parent / "published-develop-checkout"
    if root.is_dir():
        return root
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
    shutil.copytree(ROOT / "reports", root / "reports")
    restore_archived_contracts(root, "published_develop")
    return restore_published_view_roster(root)


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
