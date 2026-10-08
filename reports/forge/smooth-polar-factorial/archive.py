"""Archive exact local bulk evidence and all executed source bytes; no training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
HERE = Path(__file__).resolve().parent


def archive(runs, destination):
    files = {}
    for path in sorted(runs.rglob("*")):
        if path.is_file():
            files["raw/" + path.relative_to(runs).as_posix()] = path.read_bytes()
    manifests = [read_json(path) for path in sorted(runs.glob("*/source-manifest.json"))]
    if len(manifests) != 12 or len({row["origin_commit"] for row in manifests}) != 1 or len({row["digest"] for row in manifests}) != 1:
        raise ValueError("twelve runs must retain one exact executed source")
    source = manifests[0]
    for name, digest in source["files"].items():
        content = subprocess.check_output(["git", "show", source["origin_commit"] + ":" + name], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("executed source differs from exact commit: " + name)
        files["source/" + name] = content
    protocol = read_json(HERE / "protocol.json")
    additional = [row["task_file"] for row in protocol["tasks"].values()]
    additional += ["pyproject.toml", "tests/test_dualnorm_smoothing.py",
                   "tests/test_dualnorm_optimizers.py", "tests/test_forge_boundaries.py",
                   "tests/test_forge_techniques.py", "tests/test_recipe_defaults.py",
                   "tests/test_forge_dualnorm_contracts.py"]
    for name in additional:
        files["source/" + name] = subprocess.check_output(
            ["git", "show", source["origin_commit"] + ":" + name], cwd=ROOT)
    files["references/develop-dualnorm.py"] = subprocess.check_output(
        ["git", "show", protocol["base_commit"] + ":particlegan/optim/dualnorm.py"], cwd=ROOT)
    for name in ("controller.py", "protocol.json", "screening-protocol.json", "run.py"):
        path = HERE / name
        content = subprocess.check_output(["git", "show", source["origin_commit"] + ":" + str(path.relative_to(ROOT))], cwd=ROOT)
        files["reproduction/" + name] = content
    members = {name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest()) for name,content in files.items()}
    receipt = dict(schema_version=1, file_count=len(files), members_sha256=stable_hash(members), members=members)
    files["member-manifest.json"] = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("refusing to overwrite an existing archive")
    with tarfile.open(destination, "w:gz", compresslevel=6) as output:
        for name, content in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mtime = 0
            output.addfile(member, io.BytesIO(content))
    with tarfile.open(destination, "r:gz") as saved:
        for name, expected in files.items():
            if saved.extractfile(name).read() != expected:
                raise ValueError("archive byte verification failed: " + name)
    compact = dict(schema_version=1, path=str(destination.resolve()), sha256=file_hash(destination),
        bytes=destination.stat().st_size, file_count=len(files), members_sha256=receipt["members_sha256"],
        source_commit=source["origin_commit"], source_digest=source["digest"], all_member_bytes_verified=True,
        additional_pinned_sources=additional,
        bulk_artifacts_committed=False, new_updates=0, new_sampling=0)
    prior = HERE / "archive.json"
    if prior.exists():
        old = read_json(prior)
        compact["previous_export"] = {key: old[key] for key in ("path", "sha256", "bytes", "file_count")}
        compact["previous_export"]["reason"] = "Supplement exact task declarations and software fixtures; original training evidence unchanged."
    atomic_json(HERE / "archive.json", compact)
    print(json.dumps(compact))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/forge/smooth-polar-factorial-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/smooth-polar-factorial-v1-complete.tar.gz")
    args = parser.parse_args()
    archive(args.runs, args.output)
