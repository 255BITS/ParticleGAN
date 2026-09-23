"""Collect, train, extract and fly a fast Lunar lander in one command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lib.lunar_flight import (VARIANT, collect_counterfactuals, fast_expert,
                              rollout_episode, slow_expert, summarize_episodes)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compact_episode(episode):
    return {key: jsonable(value) for key, value in episode.items()
            if not isinstance(value, np.ndarray)}


def records(episodes, *, successful_only=False):
    selected = [e for e in episodes if not successful_only or e["outcome"] == "successful_landing"]
    if not selected:
        raise ValueError("No successful expert trajectories; refusing to train on crashes")
    result = {key: np.concatenate([e[key] for e in selected])
              for key in ("states", "actions", "next_states")}
    result["episode_seeds"] = np.concatenate([np.full(e["steps"], e["seed"], np.int64) for e in selected])
    result["episode_ids"] = np.concatenate([np.full(e["steps"], i, np.int64) for i, e in enumerate(selected)])
    result["episode_steps"] = np.concatenate([np.arange(e["steps"]) for e in selected])
    result["controllers"] = np.concatenate([np.repeat(e.get("controller", "unspecified"), e["steps"]) for e in selected])
    return result


def extract_pairs(slow, fast):
    """Both-land same-reset evidence plus physically aligned fast transitions.

    Progress pairs are exported for inspection. Policy targets keep their own
    real fast states: substituting another flight's state changes the task.
    """
    fast_by_seed = {e["seed"]: e for e in fast}
    pairs, fast_episodes = [], []
    for left in slow:
        right = fast_by_seed.get(left["seed"])
        if right is None or any(e["outcome"] != "successful_landing" for e in (left, right)):
            continue
        if right["steps"] >= left["steps"]:
            continue
        index = np.rint(np.linspace(0, right["steps"] - 1, left["steps"])).astype(int)
        pairs.append({"slow_states": left["states"], "slow_actions": left["actions"],
                      "fast_states": right["states"][index], "fast_actions": right["actions"][index],
                      "slow_seed": np.full(left["steps"], left["seed"], np.int64),
                      "fast_seed": np.full(left["steps"], right["seed"], np.int64)})
        fast_episodes.append(right)
    if not pairs:
        raise ValueError("No same-seed successful faster trajectories to extract")
    return ({key: np.concatenate([p[key] for p in pairs]) for key in pairs[0]},
            records(fast_episodes), len(pairs))


def paired_speed(slow, fast):
    left = {e["seed"]: e for e in slow if e["outcome"] == "successful_landing"}
    pairs = [(left[e["seed"]]["steps"], e["steps"]) for e in fast
             if e["outcome"] == "successful_landing" and e["seed"] in left]
    return {"matched_landings": len(pairs),
            "paired_speedup": float(np.mean([s for s, _ in pairs]) / np.mean([f for _, f in pairs])) if pairs else None,
            "fast_wins": sum(f < s for s, f in pairs),
            "median_steps_saved": float(np.median([s-f for s, f in pairs])) if pairs else None}


def passes_gate(slow, fast, *, minimum_success=0.9, minimum_speedup=1.1):
    baseline, candidate = summarize_episodes(slow), summarize_episodes(fast)
    paired = paired_speed(slow, fast)
    return bool(candidate["success_rate"] >= minimum_success
                and candidate["success_count"] >= baseline["success_count"]
                and paired["matched_landings"] >= minimum_success * len(slow)
                and paired["paired_speedup"] >= minimum_speedup)


class Mission:
    def __init__(self, out):
        self.out = out
        self.console = Console(highlight=False)
        self.start = time.monotonic()
        self.logfile = (out / "run.log").open("a", buffering=1)
        self.metrics = (out / "metrics.jsonl").open("a", buffering=1)

    def log(self, message=None, **values):
        if isinstance(message, dict):
            values = {**message, **values}
            message = None
        entry = {"elapsed_seconds": round(time.monotonic() - self.start, 2), **values}
        if message is not None:
            entry["message"] = str(message)
        line = json.dumps(jsonable(entry), allow_nan=False)
        self.metrics.write(line + "\n")
        text = str(message) if message is not None else "  ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())
        rendered = f"[{entry['elapsed_seconds']:8.1f}s] {text}"
        self.logfile.write(rendered + "\n")
        self.console.print(rendered, markup=False)

    def stage(self, number, title):
        self.console.rule(f"[bold cyan]{number}/7 · {title}")
        self.log(title, stage=number)

    def flights(self, seeds, policy, label):
        episodes = []
        for i, seed in enumerate(seeds):
            ep = rollout_episode(seed, policy)
            ep["controller"] = label
            episodes.append(ep)
            if (i + 1) % 8 == 0 or i + 1 == len(seeds):
                self.log(f"{label}: {i + 1}/{len(seeds)} flights · {sum(e['outcome'] == 'successful_landing' for e in episodes)} landings")
        return episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/gym/fast_lander"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--world-steps", type=int, default=2500)
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--slow-gan-steps", type=int, default=1200)
    parser.add_argument("--gan-steps", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="Small integration run; cannot establish merge readiness")
    args = parser.parse_args(argv)
    for name in ("threads", "train_episodes", "validation_episodes", "test_episodes", "world_steps", "warmup_steps", "slow_gan_steps", "gan_steps", "rounds", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_episodes < 2:
        parser.error("--train-episodes must be at least 2 for episode-disjoint world validation")
    if args.smoke:
        args.train_episodes, args.validation_episodes, args.test_episodes = 8, 3, 3
        args.world_steps, args.warmup_steps, args.gan_steps, args.rounds = 20, 20, 8, 1
        args.slow_gan_steps = 8
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}; use a fresh --out to preserve evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    from lib.lunar_training import train_world_model, train_fast_policy, load_fast_policy
    mission = Mission(args.out)
    mission.console.print(Panel("[bold cyan]PARTICLEGAN · LUNAR FLIGHT SCHOOL[/]\nCollect → learn dynamics → learn landing → extract → accelerate → prove → replay", border_style="cyan"))
    seed_sets = {"train": list(range(24000, 24000 + args.train_episodes)),
                 "validation": list(range(34000, 34000 + args.validation_episodes)),
                 # The original 84000 cohort informed the failure investigation.
                 # Freeze a new final cohort before evaluating the correction.
                 "test": list(range(94000, 94000 + args.test_episodes))}
    if args.smoke:
        seed_sets = {key: [seed + 1000000 for seed in seeds] for key, seeds in seed_sets.items()}
    if any(set(seed_sets[a]) & set(seed_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        parser.error("Episode cohorts overlap; reduce episode counts")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1], text=True).strip())
    source_dir = args.out / "source"
    source_dir.mkdir()
    source_paths = [Path(__file__), *(Path(__file__).resolve().parents[1] / "lib").glob("lunar_*.py")]
    for source in source_paths:
        shutil.copy2(source, source_dir / source.name)
    config = {**vars(args), "variant": VARIANT, "seed_sets": seed_sets, "revision": revision, "git_dirty": dirty,
              "weight_kind": "live", "ema_decay": 0.,
              "world_action_features": "engine_power",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()},
              "dependencies": {name: version(name) for name in ("torch", "numpy", "gymnasium", "Box2D", "Pillow")},
              "started_at": datetime.now(timezone.utc).isoformat(),
              "selection": "validation success count first, then mean steps on successful landings",
              "gate": "at least 90% success, no success count loss, >=1.1x paired speedup"}
    write_json(args.out / "config.json", config)
    mission.stage(1, "Gather expert flights and transition data")
    slow = mission.flights(seed_sets["train"], slow_expert, "slow expert")
    fast = mission.flights(seed_sets["train"], fast_expert, "fast expert")
    # Episode-disjoint world-model validation inside the training cohort.
    cut = max(1, len(slow) * 4 // 5)
    world_data = records(slow[:cut] + fast[:cut])
    world_validation = records(slow[cut:] + fast[cut:])
    slow_data = records(slow, successful_only=True)
    np.savez_compressed(args.out / "transitions.npz", **world_data)
    np.savez_compressed(args.out / "world_validation.npz", **world_validation)
    np.savez_compressed(args.out / "slow_expert.npz", **slow_data)
    write_json(args.out / "expert_episodes.json", {"slow": [compact_episode(e) for e in slow], "fast": [compact_episode(e) for e in fast]})
    # Branch only inside the world-training cohort. Off/down/ignition examples
    # teach dynamics; their alternative actions never become imitation labels.
    branches = collect_counterfactuals(slow[:cut] + fast[:cut], log=mission.log)
    np.savez_compressed(args.out / "counterfactuals.npz", **branches)
    dynamics_data = {key: np.concatenate([world_data[key], branches[key]])
                     for key in ("states", "actions", "next_states")}
    mission.stage(2, "Train the world model")
    world = args.out / "world.pt"
    world_metrics = train_world_model(dynamics_data, world, validation_records=world_validation, steps=args.world_steps,
        batch_size=args.batch_size, device=args.device, seed=7301, log=mission.log)
    world_metrics["expert_transitions"] = len(world_data["states"])
    world_metrics["counterfactual_transitions"] = len(branches["states"])
    write_json(args.out / "world_metrics.json", world_metrics)
    mission.stage(3, "Train the slow RpGAN landing policy")
    slow_path = args.out / "slow.pt"
    slow_metrics = train_fast_policy(slow_data, world, slow_path, warmup_steps=args.warmup_steps,
        steps=args.slow_gan_steps, batch_size=args.batch_size, device=args.device, seed=7302, log=mission.log)
    write_json(args.out / "slow_metrics.json", slow_metrics)
    slow_policy = load_fast_policy(slow_path, device=args.device)
    slow_validation = mission.flights(seed_sets["validation"], slow_policy.act, "slow policy validation")
    write_json(args.out / "slow_validation.json", [compact_episode(e) for e in slow_validation])
    mission.stage(4, "Extract successful slow → fast trajectories")
    pairs, fast_data, count = extract_pairs(slow, fast)
    np.savez_compressed(args.out / "slow_fast_pairs.npz", **pairs)
    np.savez_compressed(args.out / "fast_expert.npz", **fast_data)
    mission.log(f"Extracted {count} successful same-world pairs and {len(fast_data['states'])} real fast transitions")
    mission.stage(5, "Refine fast policies and select by real validation landings")
    candidates, previous = [], slow_path
    for round_index in range(1, args.rounds + 1):
        path = args.out / f"fast_round_{round_index}.pt"
        train_fast_policy(fast_data, world, path, initial_policy=previous,
            warmup_steps=args.warmup_steps, steps=args.gan_steps, batch_size=args.batch_size,
            device=args.device, seed=7303, log=mission.log)
        policy = load_fast_policy(path, device=args.device)
        episodes = mission.flights(seed_sets["validation"], policy.act, f"fast round {round_index}")
        metrics = {**summarize_episodes(episodes), **paired_speed(slow_validation, episodes)}
        candidates.append({"checkpoint": path.name, "metrics": metrics, "episodes": episodes})
        write_json(args.out / "validation.json", [{**c, "episodes": [compact_episode(e) for e in c["episodes"]]} for c in candidates])
        previous = path
    winner = max(candidates, key=lambda c: (c["metrics"]["success_count"], -(c["metrics"]["mean_success_steps"] or float("inf"))))
    shutil.copy2(args.out / winner["checkpoint"], args.out / "fast.pt")
    validation_pass = passes_gate(slow_validation, winner["episodes"])
    fast_policy = load_fast_policy(args.out / "fast.pt", device=args.device)
    slow_test, fast_test, test_pass = [], [], False
    mission.stage(6, "Evaluate the frozen winner on untouched test worlds")
    if validation_pass and not args.smoke:
        slow_test = mission.flights(seed_sets["test"], slow_policy.act, "slow policy test")
        fast_test = mission.flights(seed_sets["test"], fast_policy.act, "fast policy test")
        test_pass = passes_gate(slow_test, fast_test)
    else:
        mission.log("Test cohort remains untouched: validation gate not established or smoke run")
    leaderboard = [{"controller": name, **summarize_episodes(episodes)} for name, episodes in (
        ("slow expert (train)", slow), ("fast expert (train)", fast),
        ("learned slow (validation)", slow_validation), ("learned fast (validation)", winner["episodes"]),
        ("learned slow (test)", slow_test), ("learned fast (test)", fast_test)) if episodes]
    full_evaluation = args.validation_episodes >= 20 and args.test_episodes >= 30
    report = {"variant": VARIANT, "weight_kind": "live", "ema_decay": 0.,
              "smoke": args.smoke, "merge_ready": bool(validation_pass and test_pass and not args.smoke and full_evaluation),
              "full_evaluation": full_evaluation,
              "validation_pass": validation_pass, "test_pass": test_pass, "leaderboard": leaderboard,
              "paired_test": paired_speed(slow_test, fast_test), "winner": winner["checkpoint"],
              "test_episodes": {"slow": [compact_episode(e) for e in slow_test], "fast": [compact_episode(e) for e in fast_test]},
              "artifacts": {"slow_gif": "slow.gif", "fast_gif": "fast.gif", "comparison_gif": "comparison.gif", "dashboard": "index.html"},
              "slow_checkpoint": "slow.pt", "fast_checkpoint": "fast.pt", "world_checkpoint": "world.pt",
              "config": "config.json", "world_metrics": world_metrics}
    table = Table(title="Lunar flight leaderboard")
    for column in ("Controller", "Landings", "Success steps", "Return"):
        table.add_column(column)
    for row in leaderboard:
        table.add_row(row["controller"], f"{row['success_count']}/{row['episodes']}",
                      f"{row['mean_success_steps']:.1f}" if row["mean_success_steps"] else "—", f"{row['mean_return']:.1f}")
    mission.console.print(table)
    mission.stage(7, "Export real flight GIFs and the demo page")
    demo_slow, demo_fast = (slow_test, fast_test) if slow_test else (slow_validation, winner["episodes"])
    common = [e["seed"] for e in demo_fast if e["outcome"] == "successful_landing" and any(s["seed"] == e["seed"] and s["outcome"] == "successful_landing" for s in demo_slow)]
    demo_seed = common[0] if common else demo_slow[0]["seed"]
    from lib.lunar_artifacts import render_comparison
    slow_demo = rollout_episode(demo_seed, slow_policy.act, render=True)
    fast_demo = rollout_episode(demo_seed, fast_policy.act, render=True)
    report["demo_seed"] = demo_seed
    report["demo_split"] = "test" if slow_test else "validation"
    report["demo_is_success_pair"] = bool(common)
    report["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.suffix in (".pt", ".npz")}
    report["artifacts"] = render_comparison(args.out, slow_demo, fast_demo, report)
    write_json(args.out / "report.json", report)
    mission.log("PASS · fast landing demonstrated" if report["merge_ready"] else "NOT READY · speed/success gate not established; inspect report.json", merge_ready=report["merge_ready"])
    mission.console.print(f"[bold cyan]Demo:[/] {args.out / 'index.html'}\n[bold cyan]Logs:[/] tail -F {args.out / 'run.log'}")
    return 0 if report["merge_ready"] or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
