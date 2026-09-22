#!/usr/bin/env python
"""Roll the safe teacher and the fast teacher and write same-seed Lunar pairs.

Each seed is flown by both checkpoints. A pair is kept only when both land and
the fast landing is strictly sooner. Rows are aligned by progress t/T.
Cross-episode nearest matching is disabled. A legacy pairs.npz in the output
directory is renamed to pairs_stranger_do_not_train.npz before anything new
is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import read_config
from lib.gym_particle_finetune import load_paired_controller, playback_action
from lib.slow_fast_lunar import (COLLECTION_SEED_START, SUCCESS, archive_stranger_pairs,
    assert_collection_seeds, build_progress_pairs, read_jsonl, save_pairs,
    synthetic_teacher_rollouts)

DEFAULTS = dict(
    safe_checkpoint="results/gym/lunar_lander_particle_finetune/particle/best.pt",
    fast_checkpoint="results/gym/lunar_lander_slow_fast/fast_teacher/held.pt",
    out="results/gym/lunar_lander_slow_fast/pairs_progress.npz",
    rollouts="results/gym/lunar_lander_slow_fast/rollouts_progress.jsonl",
    live_log="results/gym/lunar_lander_slow_fast/collect.log",
    device="cuda:1",
    seed_start=COLLECTION_SEED_START,
    episodes=200,
)


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    for key in ("safe_checkpoint", "fast_checkpoint", "out", "rollouts", "live_log", "device"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")
    for key in ("seed_start", "episodes"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["episodes"] < 1:
        raise ValueError("episodes must be positive")
    if Path(cfg["out"]).name == "pairs.npz":
        raise ValueError("refusing to write pairs.npz; that name was the stranger file. "
                         "Use pairs_progress.npz")


def _log_to(live, message):
    text = f"[slow-fast-collect] {message}"
    print(text, flush=True)
    live.write(text + "\n")
    live.flush()


def rollout_teacher(bundle, seeds, teacher, rollouts_path, live):
    """Append one json object per finished episode. The teacher tag is part of the key."""
    from lib.gym_control_evaluation import OFF_ACTION, terminal_reason
    from lib.gym_data import make_env, terrain_context

    if teacher not in ("safe", "fast"):
        raise ValueError("teacher must be safe or fast")
    rollouts_path = Path(rollouts_path)
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rollouts_path.exists():
        for episode in read_jsonl(rollouts_path):
            done.add((episode.get("teacher"), int(episode["seed"])))
    env = make_env()
    written = 0
    try:
        with rollouts_path.open("a", buffering=1) as handle:
            for seed in seeds:
                if (teacher, seed) in done:
                    _log_to(live, f"teacher={teacher} seed={seed} skip=already_recorded")
                    continue
                state, _ = env.reset(seed=int(seed))
                initial, terrain = state.copy(), terrain_context(env)
                previous = OFF_ACTION.copy()
                states, actions = [], []
                total = 0.
                while True:
                    action, _route = playback_action(bundle, state, previous, terrain)
                    action = action.astype("float32", copy=False)
                    following, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    states.append(state.tolist())
                    actions.append(action.tolist())
                    state, previous = following, action
                    if terminated or truncated:
                        outcome = terminal_reason(env, state, terminated, truncated)
                        episode = dict(
                            teacher=teacher, seed=int(seed), steps=len(actions), outcome=outcome,
                            initial_state=initial.tolist(), terrain=terrain.tolist(),
                            states=states, actions=actions, game_over=bool(env.unwrapped.game_over),
                            lander_awake=bool(env.unwrapped.lander.awake), terminated=bool(terminated),
                            truncated=bool(truncated), **{"return": total})
                        handle.write(json.dumps(episode) + "\n")
                        handle.flush()
                        written += 1
                        _log_to(live, f"teacher={teacher} seed={seed} outcome={outcome} "
                                      f"steps={episode['steps']} success={outcome == SUCCESS}")
                        break
                    if len(actions) >= 1000:
                        raise RuntimeError("Pinned TimeLimit failed to end episode")
    finally:
        env.close()
    return written


def _split_teachers(episodes):
    safe, fast = [], []
    for episode in episodes:
        teacher = episode.get("teacher")
        if teacher == "safe":
            safe.append(episode)
        elif teacher == "fast":
            fast.append(episode)
        else:
            raise ValueError("rollout is missing teacher=safe|fast; this file is not a two-teacher collect")
    if not safe or not fast:
        raise ValueError("need both safe and fast rollouts")
    return safe, fast


def pair_from_episodes(safe_episodes, fast_episodes, cfg, live):
    assert_collection_seeds(int(episode["seed"]) for episode in safe_episodes)
    assert_collection_seeds(int(episode["seed"]) for episode in fast_episodes)
    archived = archive_stranger_pairs(Path(cfg["out"]).parent)
    if archived is not None:
        _log_to(live, f"ARCHIVED stranger pairs -> {archived}")
    try:
        built = build_progress_pairs(safe_episodes, fast_episodes)
    except ValueError as exc:
        _log_to(live, f"PAIR FAILED {exc}")
        raise
    path, sidecar = save_pairs(cfg["out"], built)
    manifest = built["manifest"]
    _log_to(live, "PAIRS "
            f"pairing={manifest['pairing']} alignment={manifest['alignment']} "
            f"pairs={manifest['pairs']} rows={manifest['rows']} "
            f"crashes_excluded={manifest['crashes_excluded']} "
            f"timeouts_excluded={manifest['timeouts_excluded']} "
            f"oob_excluded={manifest['oob_excluded']} not_faster={manifest['not_faster']} "
            f"fast_steps_max={manifest['fast_steps_max']} slow_steps_min={manifest['slow_steps_min']} "
            f"mean_action_edit={manifest['mean_action_edit']:.4f}")
    _log_to(live, f"WROTE pairs={path} manifest={sidecar}")
    return manifest


def collect(cfg, smoke=False, from_jsonl=False):
    validate(cfg)
    live_path = Path(cfg["live_log"])
    live_path.parent.mkdir(parents=True, exist_ok=True)
    with live_path.open("a", buffering=1) as live:
        if smoke:
            _log_to(live, "SMOKE synthetic same-seed pairs. Not Lunar landings and not a checkpoint rollout.")
            safe, fast = synthetic_teacher_rollouts()
            return pair_from_episodes(safe, fast, cfg, live)
        if from_jsonl:
            safe, fast = _split_teachers(read_jsonl(cfg["rollouts"]))
            _log_to(live, f"PAIR from {cfg['rollouts']} safe={len(safe)} fast={len(fast)}")
            return pair_from_episodes(safe, fast, cfg, live)
        seeds = assert_collection_seeds(range(cfg["seed_start"], cfg["seed_start"] + cfg["episodes"]))
        _log_to(live, f"START safe={cfg['safe_checkpoint']} fast={cfg['fast_checkpoint']} "
                      f"device={cfg['device']} seeds={seeds[0]}-{seeds[-1]} n={len(seeds)} "
                      "pairing=progress_same_seed")
        started = time.perf_counter()
        safe_bundle = load_paired_controller(cfg["safe_checkpoint"], cfg["device"])
        fast_bundle = load_paired_controller(cfg["fast_checkpoint"], cfg["device"])
        if fast_bundle.get("format") != "gym_fast_teacher_v1":
            raise ValueError("fast checkpoint must be gym_fast_teacher_v1, the held speed teacher")
        if safe_bundle.get("format") == "gym_fast_teacher_v1":
            raise ValueError("safe checkpoint must be the frozen #18 lander, not the fast teacher")
        if fast_bundle.get("residual") is not None or safe_bundle.get("residual") is not None:
            raise ValueError("teachers are E_control -> G2 checkpoints, not residual students")
        written = rollout_teacher(safe_bundle, seeds, "safe", cfg["rollouts"], live)
        written += rollout_teacher(fast_bundle, seeds, "fast", cfg["rollouts"], live)
        _log_to(live, f"ROLLOUTS wrote={written} file={cfg['rollouts']} "
                      f"elapsed_s={time.perf_counter() - started:.1f}")
        safe, fast = _split_teachers(read_jsonl(cfg["rollouts"]))
        return pair_from_episodes(safe, fast, cfg, live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--safe-checkpoint")
    parser.add_argument("--fast-checkpoint")
    parser.add_argument("--out")
    parser.add_argument("--rollouts")
    parser.add_argument("--live-log")
    parser.add_argument("--device")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="Write pairs from synthetic same-seed successes. Not a Lunar result.")
    parser.add_argument("--from-jsonl", action="store_true",
                        help="Rebuild pairs from the two-teacher jsonl. Does not touch the simulator.")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    cli = dict(safe_checkpoint=args.safe_checkpoint, fast_checkpoint=args.fast_checkpoint,
               out=args.out, rollouts=args.rollouts, live_log=args.live_log, device=args.device,
               seed_start=args.seed_start, episodes=args.episodes)
    for key, value in cli.items():
        if value is not None:
            cfg[key] = value
    if args.smoke and args.from_jsonl:
        parser.error("Choose one of --smoke or --from-jsonl")
    collect(cfg, smoke=args.smoke, from_jsonl=args.from_jsonl)


if __name__ == "__main__":
    main()
