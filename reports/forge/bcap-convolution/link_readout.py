"""Point compiled recall at the committed readout, retaining original provenance."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, read_json


def main():
    report_path = Path("reports/forge/bcap-convolution/readout.json")
    report = read_json(ROOT / report_path)
    record_path = ROOT / report["concluded_readout_record"]
    record = read_json(record_path)
    assert record["evidence_scope"] == "research_diagnostic"
    assert record["qualification_input"] is record["qualification_reuse"] is False
    assert {row["task_id"]: row["gate_status"] for row in record["task_results"]} == {
        row["task_id"]: row["gate_status"] for row in report["tasks"]}
    record.setdefault("original_source", record["source"])
    record["source"] = {"path": report_path.as_posix()}
    record["summary_curation"] = "Committed compact readout navigation; original receipt identities and numerical verdicts unchanged."
    atomic_json(record_path, record)


if __name__ == "__main__":
    main()
