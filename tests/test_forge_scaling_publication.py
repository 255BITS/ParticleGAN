"""Performance publication preserves source and protocol boundaries without training."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge import scaling_publication as publication


def receipt(*, size=32, gpu="GPU-one", status="PASS", candidate="bispf-v2"):
    spec = {"image_size": size, "batch_size": 16, "base_channels": 64,
            "max_channels": 512, "latent_dim": 128, "n_particles": 256,
            "warmup_updates": 3, "measured_updates": 10, "seed": 0,
            "prior": {"kind": "mog", "sigma": .1, "standardize": False},
            "max_seconds": 300., "profile_updates": 0}
    recipe = {"name": candidate, "batch_size": 16, "num_particles": 256, "z_dim": 128,
              "total_steps": 13, "prior_update": "learned", "prior_regularizer": "sbnp"}
    source = {"commit": "a" * 40, "files_sha256": {"particlegan/bispf.py": "a" * 64}}
    source["sha256"] = stable_hash(source)
    runtime = {"torch": "2.14.0", "dtype": "torch.float32", "threads": 1}
    identity = {"source": source, "task_sha256": stable_hash(spec),
                "recipe_sha256": stable_hash(recipe), "runtime_sha256": stable_hash(runtime),
                "global_recipe_sha256": stable_hash({key: value for key, value in recipe.items()
                    if key not in {"batch_size", "num_particles", "z_dim", "total_steps"}}),
                "initialization_sha256": "b" * 64,
                "initial_model_sha256": {"generator": "c" * 64, "discriminator": "d" * 64, "prior": "e" * 64},
                "batch_sequence_sha256": {"critic": "f" * 64, "generator": "0" * 64}}
    return {"schema_version": 1, "benchmark": publication.BENCHMARK, "candidate_id": candidate,
            "status": status, "benchmark_status": "COMPLETED" if status == "PASS" else status,
            "spec": spec, "recipe": recipe, "identity": identity, "runtime": runtime,
            "hardware": {"name": "RTX A6000", "type": "cuda", "device": "cuda:0",
                         "physical_uuid": gpu, "pci_bus_id": "0000:01:00.0",
                         "total_memory_mib": 49140, "contention_status": "clear",
                         "contention_before": {"pid": 1}, "contention_after": {"pid": 2}},
            "validity": {"passed": status == "PASS", "full_budget": status == "PASS",
                         "finite_update_fraction": 1. if status == "PASS" else 0.},
            "metrics": {"update_ms_median": 100., "update_ms_p95": 120.,
                        "images_per_second": 150., "peak_allocated_mib": 512.,
                        "peak_reserved_mib": 768., "measured_updates": 10}}


def write_receipt(root, result, *, name="result.json", family="bispf", task="r32"):
    path = "reports/forge/scaling-receipts/" + name
    atomic_json(root / path, result)
    return {"trainer_family": family, "candidate_id": result["candidate_id"],
            "task_id": task, "receipt": path}


def index(root, entries):
    manifest = publication.build_manifest(root, entries)
    atomic_json(root / publication.MANIFEST, manifest)
    return manifest


def seal(root, manifest):
    manifest.pop("input_digest", None)
    manifest["input_digest"] = stable_hash(manifest)
    atomic_json(root / publication.MANIFEST, manifest)


def test_index_roundtrip_and_current_board_render_no_training(tmp_path):
    entry = write_receipt(tmp_path, receipt())
    manifest = index(tmp_path, [entry])
    data = publication.load_performance(tmp_path)
    assert data["input_digest"] == manifest["input_digest"]
    board = {"rows": [{"trainer_family": "bispf", "candidate_id": "bispf-v2"},
                      {"trainer_family": "atlas", "candidate_id": "atlas"}],
             "family_progress": {"families": [{"id": "bispf", "label": "BISPF"}]}}
    publication.attach_performance(tmp_path, board)
    text = "\n".join(publication.render_performance(tmp_path, board, tmp_path / "reports/forge/technique-inventory.md"))
    assert "100.00 / 120.00" in text and "512.00 / 768.00" in text
    assert "Atlas" not in text and "atlas" in text and "UNKNOWN" in text
    assert "GPU-one" in text and "separate performance-source" in text
    assert "no image-quality" in text and "BatchNorm" in text
    assert "receipt](scaling-receipts/result.json)" in text
    assert not (tmp_path / "runs").exists()


def test_tampered_manifest_or_compact_receipt_is_rejected(tmp_path):
    entry = write_receipt(tmp_path, receipt())
    manifest = index(tmp_path, [entry])
    manifest["default_adoption"] = True
    atomic_json(tmp_path / publication.MANIFEST, manifest)
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        publication.load_performance(tmp_path)
    seal(tmp_path, manifest)
    with pytest.raises(ValueError, match="performance-only"):
        publication.load_performance(tmp_path)
    manifest["default_adoption"] = False
    seal(tmp_path, manifest)
    result = receipt()
    result["metrics"]["update_ms_median"] = 1.
    atomic_json(tmp_path / entry["receipt"], result)
    with pytest.raises(ValueError, match="receipt bytes changed"):
        publication.load_performance(tmp_path)


@pytest.mark.parametrize("section,key", [("runtime", "dtype"), ("spec", "seed"), ("recipe", "prior_regularizer")])
def test_receipt_internal_hashes_cannot_be_relabelled(section, key):
    result = receipt()
    result[section][key] = "changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        publication.validate_receipt(result)


def test_source_binding_and_initialization_proof_are_required():
    result = receipt()
    result["identity"]["source"]["files_sha256"]["particlegan/bispf.py"] = "1" * 64
    with pytest.raises(ValueError, match="source digest mismatch"):
        publication.validate_receipt(result)
    result = receipt()
    result["identity"].pop("initial_model_sha256")
    with pytest.raises(ValueError, match="initial_model_sha256"):
        publication.validate_receipt(result)


def test_physical_device_dtype_initialization_batch_and_contention_split_comparisons():
    original = receipt()
    base = publication.comparison_key(original)
    mutations = [lambda r: r["hardware"].update(physical_uuid="GPU-two"),
                 lambda r: r["identity"].update(runtime_sha256="1" * 64),
                 lambda r: r["identity"].update(initialization_sha256="1" * 64),
                 lambda r: r["identity"]["batch_sequence_sha256"].update(critic="1" * 64),
                 lambda r: r["hardware"].update(contention_status="unknown"),
                 lambda r: r["hardware"].update(contention_status="contended")]
    for mutate in mutations:
        result = deepcopy(original)
        mutate(result)
        assert publication.comparison_key(result) != base
    result = deepcopy(original)
    result["hardware"]["contention_before"] = {"pid": 900}
    assert publication.comparison_key(result) == base
    result["hardware"]["contention_status"] = "unknown"
    assert publication.comparison_key(result, receipt_sha256="a" * 64) != publication.comparison_key(result, receipt_sha256="b" * 64)


def test_cross_resolution_global_recipe_changes_rejected(tmp_path):
    first = write_receipt(tmp_path, receipt())
    result = receipt(size=64)
    result["recipe"]["prior_regularizer"] = "none"
    result["identity"]["recipe_sha256"] = stable_hash(result["recipe"])
    result["identity"]["global_recipe_sha256"] = stable_hash({key: value for key, value in result["recipe"].items()
        if key not in {"batch_size", "num_particles", "z_dim", "total_steps"}})
    second = write_receipt(tmp_path, result, name="r64.json", task="r64")
    with pytest.raises(ValueError, match="different global recipes"):
        publication.build_manifest(tmp_path, [first, second])


def test_duplicate_runs_require_explicit_current_choice(tmp_path):
    first = write_receipt(tmp_path, receipt())
    second = write_receipt(tmp_path, receipt(), name="retry.json")
    with pytest.raises(ValueError, match="duplicate performance cell"):
        publication.build_manifest(tmp_path, [first, second])


@pytest.mark.parametrize("detail,expected", [("OOM", "OOM"), ("UNSUPPORTED", "UNSUPPORTED"),
                                            ("ERROR", "ERROR"), ("WALL_LIMIT", "INCOMPLETE")])
def test_failure_reason_stays_visible_without_complete_metrics(tmp_path, detail, expected):
    result = receipt(status="INCOMPLETE")
    result.update(benchmark_status=detail, metrics={}, error={"type": "RuntimeError", "message": "scope reason"})
    entry = write_receipt(tmp_path, result)
    index(tmp_path, [entry])
    board = {"rows": [{"trainer_family": "bispf", "candidate_id": "bispf-v2"}]}
    publication.attach_performance(tmp_path, board)
    text = "\n".join(publication.render_performance(tmp_path, board, tmp_path / "reports/forge/technique-inventory.md"))
    assert expected + ": scope reason" in text and "unknown / unknown" in text


def test_receipt_path_escape_and_symlink_rejected(tmp_path):
    entry = write_receipt(tmp_path, receipt())
    entry["receipt"] = "../result.json"
    with pytest.raises(ValueError, match="unsafe receipt path"):
        publication.build_manifest(tmp_path, [entry])
    target = tmp_path / "target.json"
    atomic_json(target, receipt())
    (tmp_path / "link.json").symlink_to(target)
    entry["receipt"] = "link.json"
    with pytest.raises(ValueError, match="symlink"):
        publication.build_manifest(tmp_path, [entry])


def test_refresh_only_changes_performance_and_provenance_and_is_idempotent(tmp_path):
    entry = write_receipt(tmp_path, receipt())
    index(tmp_path, [entry])
    original = {"rows": [{"trainer_family": "bispf", "candidate_id": "bispf-v2", "qualified_tier": 3}],
                "provenance": {}, "qualification": {"unchanged": True}}
    original["provenance"]["input_digest"] = stable_hash(original)
    path = tmp_path / "reports/forge/technique-inventory.json"
    page = path.with_suffix(".md")
    atomic_json(path, original)
    page.write_text("# Current scientific results\n\n81/81 recorded.\n\n## Research recipes and separate diagnostics\n\nOriginal archive.\n")
    publication.refresh_performance(tmp_path)
    after = read_json(path)
    assert after["rows"] == original["rows"] and after["qualification"] == original["qualification"]
    assert page.read_text().count(publication.START) == 1
    assert "81/81 recorded." in page.read_text() and "Original archive." in page.read_text()
    before = (path.read_bytes(), page.read_bytes())
    publication.refresh_performance(tmp_path)
    assert before == (path.read_bytes(), page.read_bytes())
    assert not (tmp_path / "reports/forge/families").exists()


def test_wrapper_exposes_no_training_performance_refresh():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("scaling_wrapper", root / "reports/forge/regenerate_technique_inventory.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert callable(wrapper.main)
    assert "--refresh-performance" in (root / "reports/forge/regenerate_technique_inventory.py").read_text()


def test_missing_manifest_remains_unknown(tmp_path):
    assert publication.load_performance(tmp_path) is None
    board = {"rows": [], "provenance": {}}
    publication.attach_performance(tmp_path, board)
    text = "\n".join(publication.render_performance(tmp_path, board, tmp_path / "reports/forge/technique-inventory.md"))
    assert "**unknown**" in text and "promotion" in text
