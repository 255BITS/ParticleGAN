"""Retain actual observations after export errors; never train, sample or regrade."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forge.contracts import atomic_json, file_hash, file_lock, read_json, stable_hash
from experiments.forge.attempt_coverage import publish


def summarize(root, state_path, *, coordinator_stopped=False):
    import json
    from experiments.forge.tier1_media import render
    state = read_json(state_path)
    if not coordinator_stopped and any(j["status"] in {"PENDING", "RUNNING"} for j in state["jobs"].values()):
        raise ValueError("observation summary requires a drained coordinator")
    for job in state["jobs"].values():
        if not job.get("policy_scope") or job["status"] != "ERROR":
            continue
        directory = Path(job["directory"])
        observations = []
        for line in (directory / "run.log").read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "observation" and event.get("task") == job["task"]:
                observations.append(dict(step=event["step"], **event["metrics"]))
        if not observations:
            continue
        request = read_json(Path(job["request_path"]))
        card = request["tasks"][job["task"]]
        evidence = dict(observations=observations, sampling_law=card["evaluation"]["sampling_law"],
                        scoring_weights=card["evaluation"]["scoring_weights"])
        receipt = render(card, dict(gate_status=job["status"], evidence=evidence), directory, directory / "training.gif")
        receipt["source_inputs"] = {"run.log": file_hash(directory / "run.log"),
                                   "request.json": file_hash(directory / "request.json")}
        atomic_json(directory / "training.json", receipt)
        job["training_gif_sha256"] = receipt["gif_sha256"]
        job["saved_observations"] = dict(count=len(observations), completed_updates=observations[-1]["step"],
            final_metrics={k: v for k, v in observations[-1].items() if k != "step"},
            observations_sha256=stable_hash(observations), log_sha256=receipt["source_inputs"]["run.log"],
            note="Actual saved measurements; export ERROR remains unchanged and grants no qualification.")
        atomic_json(directory / "saved-observations-summary.json", job["saved_observations"])
    atomic_json(state_path, state)
    publish(root, state)
    print("Saved policy observations and training GIFs retained; original errors and grades unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--coordinator-stopped", action="store_true",
                        help="acquire the coordinator lock to summarize terminal jobs while supervisors continue")
    args = parser.parse_args()
    lock = file_lock(args.state_root / "coordinator.lock", blocking=False) if args.coordinator_stopped else nullcontext()
    with lock:
        summarize(Path(__file__).resolve().parents[1], args.state_root / "state.json",
                  coordinator_stopped=args.coordinator_stopped)


if __name__ == "__main__":
    main()
