"""Receipt-backed G-coherence scores for the main board, without qualification."""
from pathlib import Path
from posixpath import relpath
import re

from .contracts import file_hash, read_json


ORIGINAL = Path("reports/forge/g-coherent-mean-seed0/results.json")
PAIRED = Path("reports/forge/g-coherent-mean-paired-research/results.json")
PINS = {
    ORIGINAL: "c53a30178dbf9fd5741387f682222232baa9b4956c36f10c524a62de3df0a86d",
    PAIRED: "bf31dff3bb12a4f112e6d432773642464930b2a28a844ff68ce019ea46cad189",
}
SOURCE = "3e6ab1939a8f24ae4f874fe5d5c668d8291c681d762177eab191763b8dfd71c4"
REVISIONS = {
    "control": "cb1671ed7174fdd2eaf66df4b70c55bdeb9f691e5c885a1d4d752e4b79d49448",
    "candidate": "10df0bea444a45793fd77757f53453d909421c00283c5594d6613d220449a275",
}


def _pinned(root, path):
    target = Path(root) / path
    if file_hash(target) != PINS[path]:
        raise ValueError(f"G-coherence original receipt changed: {path}")
    return read_json(target)


def validate_original_rows(root, board):
    """Require the seed-zero display to agree with its actual original grades."""
    original = _pinned(root, ORIGINAL)
    records = {record["attempt_id"]: record for record in original["records"]}
    if (original["qualification_input"] is not False or original["qualification_reuse"] is not False
            or len(records) != 4 or len(board["rows"]) != 2):
        raise ValueError("G-coherence original scope or attempt set changed")
    for row in board["rows"]:
        for task, cell in row["benchmark_tasks"].items():
            if cell["status"] is None:
                continue
            record = records[cell["attempt"]]
            if (record["task_id"] != task or record["source_digest"] != row["source_digest"]
                    or record["candidate_revision"] != row["candidate_revision"]
                    or record["original_grade"]["status"] != cell["status"]
                    or record["gate_status"] != cell["status"]):
                raise ValueError("G-coherence display disagrees with original source, revision or grade")
            if task == "gaussian1d_stability":
                checks = record["original_grade"]["evaluator_result"]
                phases = row["retention_phases"]
                for phase in ("stationary", "shift_hold"):
                    if (phases[phase]["passed"] != checks[phase + "_passes"]
                            or phases[phase]["checks"] != checks[phase + "_checks"]):
                        raise ValueError("G-coherence display disagrees with original observation checks")
                if phases["reacquisition"]["status"] != checks["reacquisition_status"]:
                    raise ValueError("G-coherence display disagrees with original reacquisition grade")


def validate_paired_results(data):
    """Derive each pair from original attempts, retaining its own failed gate."""
    if (data["schema_version"] != 1 or data["scope"] != "registered_nonqualifying_research"
            or data["qualification_input"] is not False or data["qualification_reuse"] is not False
            or data["source_digest"] != SOURCE or len(data["records"]) != 24):
        raise ValueError("G-coherence paired research scope changed")
    records, attempts = {}, set()
    for record in data["records"]:
        match = re.fullmatch(r"paired-g-coherence-seed([1-6])-(control|candidate)", record["study"])
        if match is None:
            raise ValueError("G-coherence paired study identity changed")
        seed, arm = int(match[1]), match[2]
        task = record["task"]
        key = seed, arm, task
        expected = "PASS" if task == "gaussian1d_smoke" else "FAIL"
        if (task not in {"gaussian1d_smoke", "gaussian1d_stability"} or key in records
                or record["attempt"] in attempts or record["source_digest"] != SOURCE
                or record["candidate_revision"] != REVISIONS[arm]
                or record["qualification_input"] is not False
                or record["gate"]["status"] != expected or record["gate"]["gate_status"] != expected
                or record["cost"]["completed_steps"] != (1000 if expected == "PASS" else 6000)):
            raise ValueError("G-coherence paired source, revision, attempt or original grade changed")
        records[key] = record
        attempts.add(record["attempt"])
    pairs = []
    for seed in range(1, 7):
        pair = {"seed": seed}
        for arm in ("control", "candidate"):
            acquisition = records[seed, arm, "gaussian1d_smoke"]
            retention = records[seed, arm, "gaussian1d_stability"]
            continuity = retention["continuity"]
            checks = retention["gate"]["evaluator_result"]
            if (continuity["restored_exactly"] is not True or continuity["history_reset"] is not False
                    or continuity["parent_checkpoint_sha256"] != acquisition["checkpoint"]["sha256"]
                    or continuity["parent_state_sha256"] != acquisition["checkpoint"]["state_sha256"]
                    or continuity["parent_candidate_revision"] != acquisition["candidate_revision"]
                    or continuity["parent_smoke_status"] != "PASS" or continuity["prefix_steps"] != 1000
                    or checks["stationary_checks"] != 72 or checks["shift_hold_checks"] != 24
                    or checks["frozen_checks"] != 48 or checks["frozen_passes"] != 0
                    or checks["reacquisition_status"] != "FAIL"
                    or any(not 0 <= checks[phase + "_passes"] <= checks[phase + "_checks"]
                           for phase in ("stationary", "shift_hold"))):
                raise ValueError("G-coherence paired continuation or observation denominator changed")
            pair[arm] = checks
            pair[arm + "_gate"] = retention["gate"]["status"]
        pair["stationary_gain"] = pair["candidate"]["stationary_passes"] - pair["control"]["stationary_passes"]
        pair["shift_gain"] = pair["candidate"]["shift_hold_passes"] - pair["control"]["shift_hold_passes"]
        pairs.append(pair)
    if (pairs != data["paired_comparisons"] or data["acquisition_passes"] != 12
            or data["retention_passes"] != 0 or data["candidate_reacquisition_passes"] != 0
            or data["control_reacquisition_passes"] != 0
            or data["mean_stationary_gain"] != sum(p["stationary_gain"] for p in pairs) / 6
            or data["mean_shift_hold_gain"] != sum(p["shift_gain"] for p in pairs) / 6):
        raise ValueError("G-coherence paired summary differs from original attempt grades")
    return pairs


def main_board_rows(root, page):
    """Render the six research pairs separately from seed-zero and family scores."""
    if not (Path(root) / PAIRED).exists():
        return []
    data = _pinned(root, PAIRED)
    pairs = validate_paired_results(data)
    link = relpath(str(Path(root) / PAIRED), str(Path(page).parent))
    lines = ["## G-only coherence: registered paired research", "",
             f"[Original receipt scores]({link}), source `{SOURCE[:12]}`; control revision "
             f"`{REVISIONS['control'][:12]}`, candidate revision `{REVISIONS['candidate'][:12]}`. "
             "Each arm passes acquisition and fails its own complete retention task: **1/2 tasks per arm**. "
             "The six research seeds remain separate from the original seed-zero screening result and from selected family scores.", "",
             "| Research seed | Stationary checks, control → candidate | Shifted-hold checks, control → candidate | Reacquisition, control / candidate | Complete retention, control / candidate |",
             "| --- | --- | --- | --- | --- |"]
    for pair in pairs:
        control, candidate = pair["control"], pair["candidate"]
        lines.append(f"| {pair['seed']} | {control['stationary_passes']}/72 → {candidate['stationary_passes']}/72 | "
                     f"{control['shift_hold_passes']}/24 → {candidate['shift_hold_passes']}/24 | "
                     f"{control['reacquisition_status']} / {candidate['reacquisition_status']} | "
                     f"**{pair['control_gate']} / {pair['candidate_gate']}** |")
    lines += ["", "Stationary checks improve in all six pairs; shifted-hold checks improve in three and regress in three. "
              "These fractions count observations inside each retention task, not additional benchmark passes. "
              "Every continuation restores its own acquisition checkpoint. No pooled qualification or default promotion is inferred.", ""]
    return lines
