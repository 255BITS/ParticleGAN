"""Metadata-only checks for explicit interrupted archives; no model imports."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("inventory_archive", Path(__file__).with_name("archive_final.py"))
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class InterruptedArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.state = {"jobs": {"job": {"status": "terminal", "attempts": [{"attempt_id": "a"}]}},
                      "submissions": {"request": {"status": "queued"}},
                      "campaigns": {ARCHIVE.ROUND: {"reserved_seconds": 0, "spent_seconds": 1.25, "paused": False}},
                      "charges": [{"owner": {"campaign": ARCHIVE.ROUND}, "attempt_id": "a", "seconds": 1.25}]}

    def test_default_refuses_queued_cut_and_explicit_mode_preserves_it(self):
        original = deepcopy(self.state)
        with self.assertRaisesRegex(ValueError, "active campaign requests"):
            ARCHIVE.final_accounting(self.state)
        self.assertEqual(ARCHIVE.final_accounting(self.state, allow_interrupted=True)[1:], (1.25, ["a"]))
        self.assertEqual(self.state, original)

    def test_interrupted_mode_still_refuses_active_work_and_incomplete_charges(self):
        for field, value in (("worker", "running"), ("submission", "running"),
                             ("submission", "paused"), ("reserve", 1), ("charges", [])):
            with self.subTest(field=field, value=value):
                state = deepcopy(self.state)
                if field == "worker":
                    state["jobs"]["job"]["status"] = value
                elif field == "submission":
                    state["submissions"]["request"]["status"] = value
                elif field == "reserve":
                    state["campaigns"][ARCHIVE.ROUND]["reserved_seconds"] = value
                else:
                    state["charges"] = value
                with self.assertRaises(ValueError):
                    ARCHIVE.final_accounting(state, allow_interrupted=True)

    def test_unknown_partial_trial_preserves_exact_attempt_and_request_binding(self):
        self.state["jobs"]["job"]["subscribers"] = ["request"]
        trial = {"request_id": "request", "status": "UNKNOWN", "attempt_ids": ["a"]}
        self.assertFalse(ARCHIVE.trial_attempts_match(trial, self.state))
        self.assertTrue(ARCHIVE.trial_attempts_match(trial, self.state, allow_interrupted=True))
        for changed in ({**trial, "attempt_ids": []}, {**trial, "request_id": "missing"}):
            self.assertFalse(ARCHIVE.trial_attempts_match(changed, self.state, allow_interrupted=True))

    def test_stop_receipt_binds_bytes_source_cost_and_no_new_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            (queue / "queue").mkdir()
            (queue / "queue/state.json").write_bytes(ARCHIVE.encoded(self.state))
            receipt = {"campaign_id": ARCHIVE.ROUND, "source_origin_commit": ARCHIVE.SOURCE,
                       "source_digest": ARCHIVE.SOURCE_DIGEST, "active_workers": 0, "reserved_seconds": 0,
                       "scientific_retries": 0, "old_results_are_new_source_credit": False,
                       "new_source_credit": False, "numerical_verdicts_unchanged": True,
                       "dispatch_stopped": True, "paid_seconds": 1.25,
                       "queue_state_sha256": ARCHIVE.digest(queue / "queue/state.json"),
                       "job_status_counts": {"terminal": 1}, "submission_status_counts": {"queued": 1},
                       "successor_campaign_id": "separate-source"}
            campaign = self.state["campaigns"][ARCHIVE.ROUND]
            (queue / "interruption-receipt.json").write_bytes(ARCHIVE.encoded(receipt))
            self.assertEqual(ARCHIVE.interrupted_receipt(queue, self.state, campaign, 1.25), receipt)
            for field, value in (("new_source_credit", True), ("paid_seconds", 0),
                                 ("source_digest", "other"), ("successor_campaign_id", ARCHIVE.ROUND)):
                with self.subTest(field=field):
                    altered = {**receipt, field: value}
                    (queue / "interruption-receipt.json").write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        ARCHIVE.interrupted_receipt(queue, self.state, campaign, 1.25)
            (queue / "interruption-receipt.json").write_bytes(ARCHIVE.encoded(receipt))
            (queue / "queue/state.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "exact inactive source-bound"):
                ARCHIVE.interrupted_receipt(queue, self.state, campaign, 1.25)


if __name__ == "__main__":
    unittest.main()
