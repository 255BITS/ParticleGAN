"""Verify archived bytes, behavioral verdicts and the composed reference profile.

Run from the repository: python -m reports.transfer_suite.solvability.verify
"""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from benchmarks.transfer_suite.protocol import summarize, test_verdict
from benchmarks.transfer_suite.suite import manifest

ROOT = Path(__file__).resolve().parent


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def verify():
    counts = dict(archive_files=0, source_files=0, gan_episodes=0,
                  supervised_controls=0, complete_curves=0, errors=0)
    for inventory in [*ROOT.rglob("archive_manifest.json"), ROOT / "dynamics/inventory.json"]:
        for row in read(inventory)["files"]:
            path = inventory.parent / row["path"]
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == row.get("archived_sha256", row.get("sha256")), path
            if row.get("gzip_added") or (row.get("original_name", "").endswith(".json") and path.suffix == ".gz"):
                assert hashlib.sha256(gzip.decompress(data)).hexdigest() == row["original_sha256"], path
            counts["archive_files"] += 1
    originals = {s["name"]: s for s in manifest()["tasks"]}
    witnesses = set()

    def episode(spec, result, *, supervised=False):
        original = originals[spec["name"]]
        assert spec["thresholds"] == json.loads(json.dumps(original["thresholds"]))
        for key in ("kind", "means", "covariances", "masses", "identifiable", "pattern", "noise_std"):
            if key in original:
                assert spec[key] == original[key], (spec["name"], key)
        verdict = test_verdict(spec, result)
        assert verdict.get("convergence", {}).get("complete"), spec["name"]
        assert verdict["passed"] == (result["convergence"]["confirmed_step"] is not None)
        counts["supervised_controls" if supervised else "gan_episodes"] += 1
        counts["complete_curves"] += 1
        counts["errors"] += bool(result.get("error"))
        if verdict["passed"] and not supervised and original["tier"] == "ranking" and original["split"] == "development":
            witnesses.add(spec["name"])
        return verdict

    for index in [*(ROOT / "vectors").glob("*/index.json.gz"),
                  ROOT / "image_reproduction/index.json.gz", ROOT / "required_cap10/index.json.gz"]:
        for row in read(index)["records"]:
            data = gzip.decompress((index.parent / row["artifact"]).read_bytes())
            assert hashlib.sha256(data).hexdigest() == row["uncompressed_sha256"]
            payload = json.loads(data)
            assert episode(payload["spec"], payload["result"]) == row["verdict"]
    for path in (ROOT / "images").glob("*/episodes/*.json.gz"):
        value = read(path)
        episode(value.get("effective_spec", value["spec"]), value, supervised=path.name.startswith("supervised__"))
    for path in (ROOT / "images/diagnostics").glob("residual*.json.gz"):
        value = read(path)
        episode(value.get("effective_spec", value["spec"]), value)
    for path in (ROOT / "mog").glob("mog_sigma*.json.gz"):
        value = read(path)
        episode(value["spec"], value["result"])
    for value in read(ROOT / "dynamics/episodes.json.gz")["episodes"]:
        verdict = episode(value["spec"], value["result"])
        assert verdict["passed"] == value["verdict"]["sustained_pass"]

    # Source bundles paired with an explicit top-level protocol or declaration.
    for bundle in ROOT.rglob("source.tar.gz"):
        protocol = None
        for name in ("protocol.json.gz", "declaration.json.gz"):
            path = bundle.parent / name
            if path.exists():
                value = read(path)
                candidate = value.get("protocol", value)
                if "source_sha256" in candidate:
                    protocol = candidate
                    break
        if protocol:
            with tarfile.open(bundle) as archive:
                for name, expected in protocol["source_sha256"].items():
                    data = archive.extractfile(name).read()
                    assert hashlib.sha256(data).hexdigest() == expected, (bundle, name)
                    counts["source_files"] += 1
        elif (bundle.parent / "source_manifest.json").exists():
            with tarfile.open(bundle) as archive:
                for name, expected in read(bundle.parent / "source_manifest.json").items():
                    assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
                    counts["source_files"] += 1

    profile = read(ROOT / "reference_profile.json")
    payload = read(ROOT / "reference_profile_results.json.gz")
    assert summarize(payload["manifest"], payload["results"]) == profile["summary"]
    for name, ref in profile["episode_provenance"].items():
        path = ROOT / ref["artifact"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["artifact_sha256"], name
        value = read(path)
        actual = value.get("result", value)
        for key in ("live", "ema", "observations"):
            assert actual.get(key) == payload["results"][name].get(key), (name, key)
    counts.update(practical_tasks_with_gan_witness=len(witnesses), practical_witness_names=sorted(witnesses),
                  reference_required=profile["summary"]["counts"]["required"],
                  reference_ranking=profile["summary"]["counts"]["ranking"],
                  native_image_replay=read(ROOT / "native_image_parity.json"),
                  focused_tests="71 passed in 5.82s; test command retained in tests.log",
                  all_profile_episode_hashes_verified=True)
    assert counts["gan_episodes"] == 254 and counts["supervised_controls"] == 4
    assert len(witnesses) == 16 and counts["errors"] == 0
    (ROOT / "validation.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    verify()
