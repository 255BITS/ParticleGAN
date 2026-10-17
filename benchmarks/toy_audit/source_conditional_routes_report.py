"""Join exact source attempts and verified media into a compact audit receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from PIL import Image


NAMES = {
    "source-family-04": "Conditional two-route trajectories, discrete geometry",
    "source-family-05": "Conditional two-route trajectories, continuous geometry",
    "source-family-06": "Route transitions, discrete geometry",
    "source-family-07": "Route transitions, continuous geometry",
}
CLAIMS = {
    "trajectory": "Match class-dependent upper-route probability (0.8/0.3), continuous route variation, endpoints and segment obstacle clearance on held-out geometry; route IDs are hidden.",
    "transition": "Match a joint (state, action, next_state) law conditional on geometry, class and physical tick; next_state = state + action, and separately matching block marginals does not establish this relationship.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    media_path = args.out / "media/media.json"
    media = json.loads(media_path.read_text())
    records, source_bindings, recipes = [], None, {}
    for catalog_id, name in NAMES.items():
        root = args.artifacts / catalog_id
        path = root / "receipt.json"
        raw = json.loads(path.read_text())
        assert raw["catalog_id"] == catalog_id and raw["source_unchanged"]
        assert raw["seed_tuning"] is False and raw["qualification_credit"] == "none"
        if source_bindings is None:
            source_bindings = raw["source_sha256"]
        assert source_bindings == raw["source_sha256"]
        family = raw["family"]
        recipes.setdefault(family, raw["resolved_recipe"])
        assert recipes[family] == raw["resolved_recipe"]
        for rel, digest in raw["observer_sources"].items():
            assert sha(Path(__file__).resolve().parents[2] / rel) == digest
        config = raw["execution_config"]
        assert config["seed"] == 24002
        assert config["steps"] == (10000 if family == "trajectory" else 28000)
        record = dict(catalog_id=catalog_id, name=name, label=name,
                      catalog_original_status="SOURCE REVIEW ONLY", catalog_original_rating=4,
                      fresh_execution_status=raw["fresh_execution_status"],
                      original_scientific_status=raw["original_scientific_status"],
                      added_gate_status=raw["added_gate_status"], media=None,
                      scientific_question=CLAIMS[family], original_acceptance_gate=raw["gate"],
                      full_source_campaigns=raw["full_campaign_count"], source_attempts=1,
                      original_budget=config["steps"], geometry=raw["geometry"],
                      config_path=raw["input_config_path"], problem_selection=raw["problem_selection"],
                      execution_config=config, resolved_recipe_key=family,
                      wall_cap_seconds=raw["wall_cap_seconds"], wall_seconds_including_cleanup=raw["wall_seconds"],
                      raw_receipt=dict(path=str(path), sha256=sha(path)),
                      frozen_inputs=dict(path=str(root / "frozen-inputs.json"), sha256=sha(root / "frozen-inputs.json")),
                      raw_stdout=dict(path=str(args.artifacts.parent / ("toy-route-source-" + catalog_id[-2:] + ".log"))),
                      observer_sources=raw["observer_sources"], runtime=raw["runtime"],
                      qualification_credit="none", original_sources_unchanged=True)
        record["raw_stdout"]["sha256"] = sha(record["raw_stdout"]["path"])
        if raw["fresh_execution_status"] == "BLOCKED":
            phase = raw["prefix_baseline"]
            assert phase["completed_updates"] == phase["observer_states"] == raw["full_campaign_count"] == 0
            record.update(completed_updates=0, software_prefix_updates=0, captured_training_states=0,
                          blocked_phase=raw["blocked_phase"], exact_error="RuntimeError: CUDA is required",
                          original_error_site="experiments/train_trajectory.py:88", prefix_parity="NOT RUN: original source CUDA prerequisite blocked",
                          interpretation="Original source cannot execute on CPU; no model convergence evidence was produced.")
        else:
            assert raw["fresh_execution_status"] == "INCOMPLETE"
            assert raw["prefix_parity"]["passed"] and raw["prefix_parity"]["exact_matches"] == 4
            training = raw["training"]
            data = root / "training"
            rows = [json.loads(line) for line in (data / "observations.jsonl").read_text().splitlines()]
            assert len(rows) == training["observer_states"]
            assert all(r["complete_owner_and_rng_pure"] for r in rows)
            for field in ("observations.npz", "observations.jsonl"):
                assert sha(data / field) == training[field + "_sha256"]
            info = media[catalog_id]
            assert info["actual_states"] == [r["step"] for r in rows]
            assert info["frames"] == len(rows) and info["interpolation"] is False
            assert info["observations_npz_sha256"] == training["observations.npz_sha256"]
            assert info["observations_jsonl_sha256"] == training["observations.jsonl_sha256"]
            assert sha(args.out / "media" / info["gif"]) == info["sha256"]
            assert sha(args.out / "media" / info["poster"]) == info["poster_sha256"]
            assert Image.open(args.out / "media" / info["gif"]).n_frames == len(rows)
            record.update(completed_updates=training["completed_updates"],
                          completed_count_definition="Completed loop iterations observed at the next original loop boundary; the cap can interrupt the following iteration after some owner updates.",
                          interrupted_iteration=training["completed_updates"] + 1,
                          interruption_site="experiments/train_transition.py:355 (EMA update)" if catalog_id.endswith("06") else "experiments/train_transition.py:325 (fake composition before D update)",
                          software_prefix_updates=raw["prefix_parity"]["software_updates"],
                          prefix_parity=raw["prefix_parity"],
                          prefix_baseline_owner_sha256=raw["prefix_baseline"]["boundaries"],
                          prefix_observed_owner_sha256=raw["prefix_observed"]["boundaries"],
                          captured_training_states=len(rows),
                          captured_prefix_states=raw["prefix_observed"]["observer_states"],
                          exact_purity_assertions=len(rows) + raw["prefix_observed"]["observer_states"],
                          actual_saved_updates=info["actual_states"], final_observed_update=rows[-1]["step"],
                          final_observed_metrics=rows[-1]["metrics"], media="media/" + info["gif"],
                          media_sha256=info["sha256"], poster="media/" + info["poster"],
                          sampling=rows[0]["metrics"]["panel"],
                          exact_error="TimeoutError: 120-second entry wall cap exhausted",
                          training_seconds_including_save=training["elapsed_seconds"],
                          paid_observer_seconds=training["observer_seconds"],
                          prefix_observer_seconds=raw["prefix_observed"]["observer_seconds"],
                          prior_at_initialization=json.loads((data / "run/prior.json").read_text()),
                          normalization=json.loads((data / "run/normalization.json").read_text()),
                          artifacts={field: dict(path=str(data / field), sha256=sha(data / field))
                                     for field in ("observations.npz", "observations.jsonl", "run/config.yaml", "run/recipe.json", "run/source.zip")},
                          interpretation="Partial progress at the cap; original 28,000-step budget and original endpoint evaluation were not completed. No convergence conclusion or merge qualification.")
        records.append(record)
    archive_path = args.artifacts / "source.tar"
    with tarfile.open(archive_path) as archive:
        for name, expected in source_bindings.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    revision = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
    queries = "".join(f"{revision}:{name}\n" for name in source_bindings).encode()
    objects = subprocess.run(["git", "cat-file", "--batch"], input=queries,
                             cwd=Path(__file__).resolve().parents[2], capture_output=True, check=True).stdout
    offset = 0
    for name, expected in source_bindings.items():
        newline = objects.index(b"\n", offset)
        header = objects[offset:newline].split()
        assert len(header) == 3 and header[1] == b"blob", name
        size = int(header[2])
        start = newline + 1
        assert hashlib.sha256(objects[start:start + size]).hexdigest() == expected, name
        offset = start + size + 1
    assert offset == len(objects)
    package = {k: v for k, v in source_bindings.items() if k.startswith("particlegan/")}
    critical = {k: source_bindings[k] for k in ("experiments/train_trajectory.py", "experiments/train_transition.py",
                "lib/trajectory.py", "lib/transition.py", "lib/toy_metrics.py",
                "configs/trajectory/default.yaml", "configs/trajectory/diversity/confirm_10k/mlp_continuous.yaml",
                "configs/transition/default.yaml")}
    result = dict(format="route_source_coverage_compact_v1", source_revision=revision,
                  source_archive=dict(path=str(archive_path), sha256=sha(archive_path), bound_source_files=len(source_bindings),
                                      git_revision_blobs_verified=len(source_bindings),
                                      binding_algorithm="canonical SHA256 of sorted compact JSON path-to-SHA256 map",
                                      source_manifest_sha256=canonical_sha(source_bindings),
                                      package_manifest_sha256=canonical_sha(package), critical_sha256=critical),
                  recipes=recipes, records=records,
                  counts=dict(catalog_entries=4, blocked=2, incomplete=2, completed_original_budgets=0,
                              full_campaigns=2, completed_campaign_updates=sum(r["completed_updates"] for r in records),
                              prefix_software_updates=sum(r["software_prefix_updates"] for r in records),
                              prefix_exact_owner_matches=8, total_purity_assertions=24,
                              captured_training_states=16, captured_prefix_states=8, actual_gif_frames=16,
                              inferred_convergence_passes=0, qualification_credit=0),
                  wall_cap_definition="120-second SIGALRM per entry, including source import, short baseline parity and the sole full-budget attempt. Save/receipt cleanup follows without additional optimizer updates.",
                  runtime_device="CPU1; separate from CUDA reports", original_catalog_unchanged=True,
                  full_streams_in_git=False, media_receipt=dict(path="media/media.json", sha256=sha(media_path)),
                  report_generator_sha256=sha(__file__))
    (args.out / "coverage.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
