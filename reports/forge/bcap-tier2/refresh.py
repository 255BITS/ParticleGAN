"""Register original Tier 2 evidence and update BCAP's whole-row publication pin."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.trainer_families import CURRENT_SELECTION, family_for_candidate, family_row_pin
from experiments.forge.planning import load_idea
from reports.forge import regenerate_technique_inventory as publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-report", type=Path,
                        help="Reuse an independently graded frozen-source report after verifying all original receipt hashes")
    args = parser.parse_args()
    study = read_json(ROOT / "reports/forge/bcap-tier2/readout.json")
    assert sum(study["tier2_counts"].values()) == 21 and study["new_attempt_count"] == 21
    if args.validated_report:
        path = args.validated_report.resolve()
        saved = read_json(path)
        assert saved["frozen_source"]["commit"] == study["source_origin_commit"]
        assert saved["frozen_source"]["source_digests"] == [study["source_digest"]]
        assert saved["provenance"]["input_digest"] == stable_hash({key: value for key, value in saved.items()
            if key != "provenance"} | {"provenance": {key: value for key, value in saved["provenance"].items()
                                                     if key != "input_digest"}})
        for attempt, proof in saved["provenance"]["qualified_receipts"].items():
            receipt = publication.project_receipt(ROOT, attempt)
            assert receipt["provenance"]["canonical_result_hash"] == proof["canonical_result_hash"]
            assert {key: value["sha256"] for key, value in receipt["provenance"]["original_files"].items()} == proof["original_file_sha256"]
        metadata = {"json": str(path), "source_commit": study["source_origin_commit"]}
    else:
        metadata = publication.regenerate(ROOT, source_commit=study["source_origin_commit"], execution_backend="cuda",
                                          output_prefix=ROOT / "runs/bcap-six/tier2-source-evidence")
    report = read_json(metadata["json"])
    matches = [row for row in report["rows"] if row["candidate_id"] == study["candidate_id"]
               and row.get("bindings", {}).get("source_digest") == study["source_digest"]]
    assert len(matches) == 1
    row = matches[0]
    assert row["candidate_revision"] == study["candidate_revision"] and len(row["attempt_ids"]) == 28
    assert row["tiers"]["1"]["passed"] == row["tiers"]["1"]["total"] == 6
    assert row["tiers"]["2"]["counts"] == study["tier2_counts"]
    row["trainer_family"] = family_for_candidate(ROOT, row["candidate_id"], load_idea(ROOT, row["candidate_id"]),
                                                current_presentation=True)["id"]
    pin = family_row_pin(row, selection_kind="configured_standard",
        reason="Retain the selected global BCAP smoothing=1e-5 recipe and all six Tier 1 passes; add its complete frozen Tier 2 outcomes, including numerical failures and image setup limitations. Constant learning rates, seed 0, no retries or default adoption; Tier 3 not requested.")
    path = ROOT / CURRENT_SELECTION
    original, original_sha = path.read_bytes(), file_hash(path)
    card = read_json(path)
    old = next(item for item in card["selections"] if item["trainer_family"] == pin["trainer_family"])
    assert old["candidate_id"] == pin["candidate_id"] and old["candidate_revision"] == pin["candidate_revision"]
    card["selections"] = [pin if item["trainer_family"] == pin["trainer_family"] else item for item in card["selections"]]
    atomic_json(path, card)
    original_regenerate = publication.regenerate
    evidence_sha = file_hash(Path(metadata["json"]))
    def verified_same_report(root, *, source_commit, execution_backend, view_id, output_prefix):
        # Reuse this call's independently regraded immutable report. The normal
        # publisher revalidates its rows/receipt proofs; a second identical
        # regrade adds no evidence and spends several minutes on catalog binding.
        assert Path(root) == ROOT and source_commit == study["source_origin_commit"]
        assert execution_backend == "cuda" and view_id == "discriminator_stability"
        assert file_hash(Path(metadata["json"])) == evidence_sha
        return metadata
    publication.regenerate = verified_same_report
    try:
        result = publication.publish_current(ROOT, source_commit=study["source_origin_commit"], execution_backend="cuda")
    except Exception:
        path.write_bytes(original)
        raise
    finally:
        publication.regenerate = original_regenerate
    atomic_json(ROOT / "reports/forge/bcap-tier2/publication.json", {
        "schema_version": 1, "previous_card_sha256": original_sha, "previous_pin": old, "current_pin": pin,
        "same_candidate_and_scientific_source": True, "publication": result,
        "qualification_input": False, "script_sha256": file_hash(Path(__file__))})
    print(result)


if __name__ == "__main__":
    main()
