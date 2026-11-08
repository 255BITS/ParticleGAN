"""New update-boundary task identities retain each exact ancestor and old result."""
import hashlib
import json

from update_boundary_source_bindings import ROOT, before_update_boundary_refresh, receipts


def test_update_boundary_source_bindings_preserve_all_questions_and_original_receipts():
    for receipt in receipts():
        assert receipt["schema"] == "update_boundary_source_binding_v1"
        assert receipt["qualification_results_changed"] is False
        assert receipt["scientific_training_launched"] is False
        assert receipt["automatic_cross_source_reuse"] is False
        assert len(receipt["tasks"]) == 10
        for row in receipt["tasks"]:
            path = ROOT / row["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["current_task_sha256"]
            current = json.loads(path.read_text())
            assert current["evaluation"]["evaluator_revision"]["qualification_input"] is False
            previous, _ = before_update_boundary_refresh(current, row["path"])
            assert previous["evaluation"]["evaluator_revision"] != current["evaluation"]["evaluator_revision"]
