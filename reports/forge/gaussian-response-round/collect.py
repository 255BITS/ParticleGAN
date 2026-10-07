"""Project completed source-bound readouts; no training, sampling or regrading."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(directories, refs):
    sources, rows = {}, []
    total_updates = total_seconds = 0
    data_sequences = {}
    for cohort, directory in directories.items():
        result_path = directory / "results.json"
        publication = json.loads(result_path.read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        if publication["new_training_updates"] != 30000 or len(publication["results"]) != 9:
            raise ValueError("incomplete declared study: " + cohort)
        total_updates += publication["new_training_updates"]
        total_seconds += publication["new_training_loop_seconds"]
        sources[cohort] = {
            "publication_commit": refs[cohort],
            "report_url": f"https://github.com/255BITS/ParticleGAN/blob/{refs[cohort]}/reports/forge/{directory.name}/README.md",
            "compact_results_sha256": digest(result_path),
            "provenance_sha256": digest(directory / "provenance.json"),
            "protocol_sha256": digest(directory / "protocol.json"),
            "executed_scientific_commit": (provenance.get("executed_scientific_commit")
                                          or provenance["source_commit"]),
            "archive": provenance["archive"],
        }
        for result in publication["results"]:
            if result["combined_verdict"] != "FAIL" or result["additional_updates"] not in (2000, 4000):
                raise ValueError("projection assumptions changed; review readout")
            key = (result["task"], result["phase"])
            data_sequences.setdefault(key, set()).add(result["data_sha256"])
            rows.append({"cohort": cohort, **{key: result[key] for key in (
                "arm", "task", "phase", "acquisition_verdict", "acquisition_suffix",
                "hold_pass_checks", "hold_total_checks", "combined_verdict",
                "longest_pass_streak", "first_five_pass_window", "total_pass_checks",
                "total_checks", "final_metrics", "loop_seconds")}})
    if any(len(values) != 1 for values in data_sequences.values()):
        raise ValueError("training batch sequences differ across studies")
    return {
        "schema_version": 1, "id": "gaussian-response-round-v1",
        "scope": "display_only_cross_pr_synthesis",
        "qualification_input": False, "qualification_reuse": False,
        "new_training_updates_for_this_projection": 0,
        "referenced_cost": {"new_training_updates": total_updates,
            "new_real_training_examples": total_updates * 128,
            "new_training_loop_seconds": total_seconds,
            "reserved_seconds": 13500, "scientific_retries": 0,
            "accounting": "Costs belong to the three original scientific records; do not debit again."},
        "matching_real_batch_sequences": {f"{task}__{phase}": next(iter(values))
            for (task, phase), values in data_sequences.items()},
        "sources": sources, "results": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in ("frozen", "prior", "network"):
        parser.add_argument("--" + cohort, type=Path, required=True)
        parser.add_argument("--" + cohort + "-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = ("frozen", "prior", "network")
    result = collect({name: getattr(args, name) for name in cohorts},
                     {name: getattr(args, name + "_ref") for name in cohorts})
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"readouts": len(result["results"]), "new_projection_training_updates": 0,
                      "referenced_cost": result["referenced_cost"]}))
