"""Read certified published checkpoints; never construct/train/sample models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from experiments.forge.state import state_digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def optimizer_packets(value, path="root"):
    if isinstance(value, dict):
        if "dualnorm" in value:
            yield path, value
        for key, child in value.items():
            yield from optimizer_packets(child, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from optimizer_packets(child, f"{path}/{index}")


def extract(results):
    rows = []
    for task in results["task_results"]:
        proof = task.get("provenance_checkpoint")
        row = {key: task[key] for key in ("role", "task_id", "attempt_id", "gate_status")}
        if proof is None:
            rows.append({**row, "status": "UNAVAILABLE", "reason": "No certified final checkpoint"})
            continue
        path = Path(proof["artifact_root"]) / proof["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == proof["sha256"], f"{path}: certificate SHA mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(state_digest(saved) == proof["state_sha256"], f"{path}: state SHA mismatch")
        optimizers, totals = [], Counter()
        for location, packet in optimizer_packets(saved):
            meta = packet["dualnorm"].get("optimism")
            states = packet["state"]
            histories = {key: state for key, state in states.items()
                         if "optimism_previous_gradient" in state}
            active = task["role"] == "candidate"
            require((meta is not None) == active, f"{location}: wrong active mechanism")
            if active:
                require(meta["mode"] == "raw_gradient" and meta["schema"] == 1,
                        f"{location}: wrong optimism law")
                require(len(histories) == len(states), f"{location}: missing consumed history")
                applications = sum(int(s["step"]) for s in states.values())
                require(applications == meta["stats"]["parameter_applications"],
                        f"{location}: consumed history/counters disagree")
                totals.update(meta["stats"])
            else:
                require(not histories, f"{location}: inactive history consumed")
            seen = 0
            masks = 0
            for state in histories.values():
                gradient = state["optimism_previous_gradient"]
                require(bool(torch.isfinite(gradient).all()), f"{location}: nonfinite raw history")
                mask = state.get("optimism_seen_rows")
                if mask is not None:
                    masks += 1
                    require(mask.dtype == torch.bool and len(mask) == len(gradient),
                            f"{location}: malformed row ownership")
                    require(not bool((gradient[~mask] != 0).any()),
                            f"{location}: unseen row history")
                    seen += int(mask.sum())
            optimizers.append(dict(path=location, active=active,
                roles=[group["role"] for group in packet["param_groups"]],
                stats=None if meta is None else meta["stats"],
                parameter_histories=len(histories), row_masks=masks, seen_rows=seen,
                history_sha256=state_digest(histories)))
        require(optimizers, f'{row["task_id"]}: no public normalized optimizer checkpoint')
        rows.append({**row, "status": "PASS", "completed_steps": proof["completed_steps"],
            "checkpoint_path": str(path), "checkpoint_sha256": digest,
            "checkpoint_state_sha256": proof["state_sha256"], "optimizers": optimizers,
            "totals": dict(totals)})
    return dict(schema_version=1, scope="saved_optimizer_history_audit", qualification_input=False,
        source_commit=results["source_commit"], source_digest=results["source_digest"],
        optimizer_updates_added=0, sampling_draws_added=0,
        status="PASS", task_checkpoints=rows,
        interpretation="Counters come from final saved state, including consumed producer history in continuations. Summing task totals double-counts such prefixes; these are per-checkpoint counters, not campaign work totals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = extract(json.loads(args.results.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"], checkpoints=len(result["task_checkpoints"]))))
