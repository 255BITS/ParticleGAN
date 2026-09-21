#!/usr/bin/env python
"""Score slow→fast checkpoints on the shared Lunar seeds.

Landings and mean steps among successes are the report. Mean episode length is
printed and is not the speed metric: a crash is short and still a failure.
A candidate that lands less often, crashes more, or flies out of bounds more
than the #18 baseline is ineligible, even if the successes that remain are quick.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_gym_transition import sha256, write_json
from lib.gym_particle_finetune import load_paired_controller, playback_action
from lib.slow_fast_lunar import (EVAL_SEEDS, protocol_seeds, read_jsonl, speed_decision, speed_stats)

DEFAULT_OUT = ROOT / "reports/gym/lunar_lander_slow_fast"
DEFAULT_EPISODES = ROOT / "results/gym/lunar_lander/data/episodes.json"
DEFAULT_ROLLOUTS = ROOT / "results/gym/lunar_lander_slow_fast/rollouts.jsonl"
DEFAULT_LOG = ROOT / "results/gym/lunar_lander_slow_fast/live.log"


def _check_device(device):
    if str(device).startswith("cuda") and device != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")


def _overlap_note(seeds, episodes_path, rollouts_path):
    notes = []
    seeds = set(int(seed) for seed in seeds)
    if episodes_path.is_file():
        used = {int(episode["seed"]) for episode in json.loads(episodes_path.read_text())}
        overlap = sorted(seeds.intersection(used))
        if overlap:
            raise ValueError(f"eval seeds overlap source episodes: {overlap[:5]}")
        notes.append(f"disjoint from {len(used)} source reset seeds")
    else:
        notes.append(f"source episodes absent at {episodes_path}; overlap was not checked")
    if rollouts_path.is_file():
        used = {int(episode["seed"]) for episode in read_jsonl(rollouts_path)}
        overlap = sorted(seeds.intersection(used))
        if overlap:
            raise ValueError(f"eval seeds overlap collected rollouts: {overlap[:5]}")
        notes.append(f"disjoint from {len(used)} collected rollouts")
    else:
        notes.append(f"rollouts absent at {rollouts_path}; overlap was not checked")
    leaked = seeds.intersection(EVAL_SEEDS) ^ seeds
    if leaked:
        raise ValueError("eval seeds left the shared validation/test protocol")
    return "; ".join(notes)


def _score(checkpoint, seeds, device, trace_path, label):
    bundle = load_paired_controller(checkpoint, device)
    action = lambda env, state, previous, terrain: playback_action(bundle, state, previous, terrain)
    from lib.gym_control_evaluation import evaluate_controller
    row = evaluate_controller(action, list(seeds), trace_path, label)
    row["speed"] = speed_stats(row["episodes"])
    row["checkpoint"] = str(Path(checkpoint).resolve())
    row["checkpoint_sha256"] = sha256(checkpoint)
    row["step"] = int(bundle["step"])
    row["adv_weight"] = float(bundle["config"]["adv_weight"])
    row["safe_fast_weight"] = float(bundle["config"].get("safe_fast_weight", 0.) or 0.)
    residual = bundle.get("residual")
    row["playback"] = ("residual scale=0.15 added to frozen tanh(G2)" if residual is not None
                       else "no residual; E_control -> G2 only")
    return row


def _load_or_score(checkpoint, split, role, out, seeds, device, log):
    destination = out / "evaluations" / f"{role}_{split}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(checkpoint)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old.get("checkpoint_sha256") == digest and old.get("seeds") == list(seeds):
            log(f"REUSE {role} {split} {destination.name}")
            return old
        raise RuntimeError(f"Existing score does not match this checkpoint: {destination}")
    log(f"SCORE {role} {split} checkpoint={checkpoint}")
    row = _score(checkpoint, seeds, device, out / "traces" / f"{role}_{split}.npz", f"slow-fast/{role}/{split}")
    log(f"PLAYBACK {row['playback']}")
    row.update(role=role, split=split, seeds=list(seeds), device=str(device))
    kept = {key: value for key, value in row.items() if key != "episodes"}
    kept["episodes"] = row["episodes"]
    write_json(destination, kept)
    speed = kept["speed"]
    log(f"RESULT {role} landings={speed['landing_count']}/{speed['episodes']} "
        f"success_steps={speed['mean_success_steps']} crashes={speed['crash_count']} "
        f"timeouts={speed['timeout_count']} oob={speed['oob_count']} "
        f"mean_episode_steps={speed['mean_episode_steps']:.1f}")
    return kept


def render_report(split, baseline, rows):
    """Markdown table. Eligibility uses successes only, never crash-shortened episodes."""
    lines = [
        "# Lunar slow→fast evaluation",
        "",
        f"Split `{split}`. Seeds are the shared control protocol "
        f"({rows[0]['seeds'][0]}–{rows[0]['seeds'][-1]}, n={len(rows[0]['seeds'])})." if rows else
        f"Split `{split}`.",
        "",
        "Speed is mean steps among `successful_landing` only. "
        "`mean_episode_steps` includes crashes and is not a ranking key. "
        "A candidate is eligible only when landings did not fall, crashes did not rise, "
        "out-of-bounds did not rise, and success steps fell versus the #18 baseline.",
        "",
        "| Role | Step | Landings | Success steps | Crashes | OOB | Timeouts | Episode steps | Eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def cells(role, row, decision):
        speed = row["speed"]
        steps = "—" if speed["mean_success_steps"] is None else f"{speed['mean_success_steps']:.1f}"
        eligible = "baseline" if decision is None else ("yes" if decision["eligible"] else "no: " + "; ".join(decision["reasons"]))
        return (f"| {role} | {row['step']} | {speed['landing_count']}/{speed['episodes']} | {steps} | "
                f"{speed['crash_count']} | {speed['oob_count']} | {speed['timeout_count']} | "
                f"{speed['mean_episode_steps']:.1f} | {eligible} |")

    lines.append(cells("baseline", baseline, None))
    any_win = False
    for row in rows:
        decision = speed_decision(row["speed"], baseline["speed"])
        row["decision"] = decision
        any_win = any_win or decision["eligible"]
        lines.append(cells(row["role"], row, decision))
    lines.append("")
    if any_win:
        lines.append("At least one candidate is faster among successes without a crash shortcut.")
    else:
        lines.append("No speed win. Crash shortcuts and slower successes are refused.")
    lines.append("")
    lines.append("These rows are this directory's rollouts. Do not copy the 2D toy board here.")
    lines.append("")
    return "\n".join(lines)


def select_winner(rows):
    eligible = [row for row in rows if row["decision"]["eligible"]]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row["speed"]["mean_success_steps"],
                                          -row["speed"]["landing_count"], int(row["step"])))


def evaluate(baseline, checkpoints, split, out, device, episodes_path, rollouts_path, live_path):
    _check_device(device)
    if not checkpoints:
        raise ValueError("pass at least one slow-fast checkpoint")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = protocol_seeds(split)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with live_path.open("a", buffering=1) as live:
        def log(message):
            text = f"[slow-fast-eval] {message}"
            print(text, flush=True)
            live.write(text + "\n")
            live.flush()

        note = _overlap_note(seeds, Path(episodes_path), Path(rollouts_path))
        log(f"START split={split} seeds={seeds[0]}-{seeds[-1]} n={len(seeds)} device={device}")
        log(f"SHARED {note}")
        log("SPEED mean steps among successes. Crashes are refused as a shortcut.")
        started = time.perf_counter()
        base = _load_or_score(baseline, split, "baseline", out, seeds, device, log)
        rows = []
        for index, checkpoint in enumerate(checkpoints):
            role = f"candidate_{index}_{Path(checkpoint).stem}"
            rows.append(_load_or_score(checkpoint, split, role, out, seeds, device, log))
        report = render_report(split, base, rows)
        (out / "README.md").write_text(report)
        winner = select_winner(rows)
        selection = dict(split=split, seeds=list(seeds), baseline=str(Path(baseline).resolve()),
                         winner=None if winner is None else winner["checkpoint"],
                         eligible=winner is not None)
        if winner is None:
            log("SELECTED none. No candidate beat the baseline without a crash shortcut.")
        elif split == "validation":
            best = Path(winner["checkpoint"]).parent / "best.pt"
            if best.exists() and sha256(best) != winner["checkpoint_sha256"]:
                raise RuntimeError("Refusing to overwrite a different best.pt")
            if not best.exists():
                best.write_bytes(Path(winner["checkpoint"]).read_bytes())
            selection["best"] = str(best.resolve())
            log(f"SELECTED step={winner['step']} success_steps={winner['speed']['mean_success_steps']:.1f} "
                f"landings={winner['speed']['landing_count']}/{winner['speed']['episodes']} best={best}")
        else:
            log(f"ELIGIBLE step={winner['step']} success_steps={winner['speed']['mean_success_steps']:.1f} "
                f"landings={winner['speed']['landing_count']}/{winner['speed']['episodes']}. "
                "Test does not write best.pt.")
        write_json(out / "selection.json", selection)
        log(f"WROTE {out / 'README.md'} elapsed_s={time.perf_counter() - started:.1f}")
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--live-log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("pass at least one --checkpoint")
    evaluate(args.baseline, args.checkpoint, args.split, args.out, args.device,
             args.episodes, args.rollouts, args.live_log)


if __name__ == "__main__":
    main()
