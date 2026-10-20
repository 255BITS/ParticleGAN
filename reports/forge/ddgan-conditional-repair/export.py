"""Export exact saved Forge evidence and the single current research board.

No model is constructed, no training/sampling is performed, and archived grades
are copied instead of recomputed. Missing full receipts remain INCOMPLETE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(queue, campaigns):
    runs = []
    for campaign in campaigns:
        for path in sorted((queue / campaign).glob("*/request.json")):
            resolved = read(path)
            request, job = resolved["request"], resolved["job"]
            if not str(job["task_id"]).startswith("ddgan_repair_"):
                continue
            directory = path.parent
            raw_path, adapter_path = directory / "api/receipt.json", directory / "adapter-receipt.json"
            terminal_path = directory / "terminal.json"
            terminal = read(terminal_path) if terminal_path.exists() else {}
            row = dict(candidate=request["candidate"]["id"], task=job["task_id"],
                attempt=directory.name, campaign=campaign,
                request_id=resolved["request_id"], candidate_revision=request["candidate_revision"],
                source=request["source"], task_contract=request["tasks"][job["task_id"]],
                protocol_seed=0, qualification_input=False,
                attempt_status=terminal.get("attempt_status", "running"),
                terminal_sha256=sha(terminal_path) if terminal_path.exists() else None,
                artifact_directory=str(directory), completed_updates=0,
                verdict="INCOMPLETE", terminal_passes=0, final_metrics={}, failed_bounds=[])
            # The full source file map is already content-addressed in Forge;
            # retain compact manifest identities here, not another bulk copy.
            row["source"] = {key: value for key, value in row["source"].items() if key != "files"}
            if raw_path.exists():
                raw = read(raw_path)
                complete = raw.get("default_protocol_complete", False) and raw.get("status") == "COMPLETE"
                scheduled = raw["protocol"]["metric_evaluation_steps"]
                checks = [obs for obs in raw["observations"] if obs["step"] > 0 and obs["step"] in scheduled]
                row.update(completed_updates=raw["completed_updates"],
                    verdict=raw["verdict"] if complete else "INCOMPLETE",
                    full_protocol_complete=complete, terminal_passes=sum(obs["passed"] for obs in checks[-5:]),
                    scheduled_checks=len(checks), final_metrics=raw["observations"][-1]["metrics"],
                    failed_bounds=raw["failed_bounds"], recipe=raw["recipe"],
                    reproducibility=raw["reproducibility"], runtime=raw["runtime"],
                    raw_receipt_sha256=sha(raw_path), elapsed_seconds=raw["elapsed_seconds"],
                    artifact_hashes=raw["artifacts"])
                gif = directory / "api/goal.gif"
                if gif.exists():
                    if sha(gif) != raw["artifacts"]["goal.gif"]["sha256"]:
                        raise ValueError("actual-training GIF differs from saved receipt")
                    destination = REPORT / "media" / f'{row["candidate"]}-{job["task_id"]}.gif'
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(gif, destination)
                    row["gif"] = str(destination.relative_to(ROOT))
                if adapter_path.exists():
                    adapter = read(adapter_path)
                    row.update(adapter_receipt_sha256=sha(adapter_path),
                        guards=adapter["evidence"]["guards"],
                        prior_policy=adapter["evidence"]["prior_policy"],
                        initial_models=adapter["evidence"]["initial_models"],
                        batch_sequence_sha256=adapter["evidence"]["batch_sequence_sha256"])
            runs.append(row)
    # Check the actual task contract and consumed clean/timestep sequence agree
    # across all recipe arms; timeout-only variants compare as their own tasks.
    for task in {row["task"] for row in runs}:
        measured = [row for row in runs if row["task"] == task and row.get("full_protocol_complete")]
        for name in ("initial_models", "batch_sequence_sha256"):
            if len({json.dumps(row.get(name), sort_keys=True) for row in measured}) > 1:
                raise ValueError(f"unmatched {name} for {task}")
    result = dict(schema="ddgan_conditional_repair_publication_v1", qualification_input=False,
        goal="ddgan_conditional_repair", seed=0,
        baseline="bispf-optimized-quintic-v1", campaigns=campaigns, runs=runs)
    (REPORT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    board = ROOT / "reports/forge/leaderboards/ddgan_conditional_repair.md"
    board.parent.mkdir(exist_ok=True)
    lines = ["# DDGAN conditional repair: current research leaderboard", "",
        "GPU0 shared-load diagnostic only; no ordinary qualification or speed ranking. "
        "All runs use protocol seed0, clean live four-step sampling and unchanged scientific gates.", "",
        "| Recipe | Task | Verdict | Updates | Last5 passes | Final CDF | Worst conditional CDF | GIF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in runs:
        metrics = row["final_metrics"]
        final = f'{metrics["final_projection_cdf"]:.4f}' if "final_projection_cdf" in metrics else "—"
        worst = f'{metrics["worst_transition_cdf"]:.4f}' if "worst_transition_cdf" in metrics else "—"
        gif = f'[actual training](../../{Path(row["gif"]).relative_to("reports")})' if row.get("gif") else "—"
        lines.append(f'| `{row["candidate"]}` | `{row["task"]}` | **{row["verdict"]}** | '
            f'{row["completed_updates"]}/{row["task_contract"]["execution"]["steps"]} | '
            f'{row["terminal_passes"]}/5 | {final} | {worst} | {gif} |')
    lines += ["", "CDF bounds: each final and conditional panel <=0.12. A PASS also requires all mean, "
        "covariance and bimodal mode/local-spread bounds at every final five scheduled checks. "
        "[Full failed bounds, provenance and exact final metrics](../ddgan-conditional-repair/results.json).", ""]
    board.write_text("\n".join(lines))
    print(json.dumps(dict(runs=len(runs), outcomes=[(r["candidate"], r["task"], r["verdict"]) for r in runs])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--campaign", action="append", required=True)
    args = parser.parse_args()
    export(args.queue.resolve(), args.campaign)
