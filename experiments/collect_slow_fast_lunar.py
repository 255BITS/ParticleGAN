#!/usr/bin/env python
"""Roll a frozen paired-error controller and write matched slow/fast Lunar pairs.

Successful landings only. Crashes, timeouts, and flyaways are counted and then
dropped. The fast pool is the short successes; the slow pool is the long ones.
Pairs require a nearby start and nearby terrain. Rollouts append to a jsonl
file before pairing, so a distance retune does not need another simulator pass.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import read_config
from lib.gym_particle_finetune import load_paired_controller
from lib.slow_fast_lunar import (COLLECTION_SEED_START, SUCCESS, assert_collection_seeds,
    build_slow_fast_pairs, read_jsonl, save_pairs, synthetic_episodes)

DEFAULTS = dict(
    checkpoint="results/gym/lunar_lander_particle_finetune/particle/best.pt",
    out="results/gym/lunar_lander_slow_fast/pairs.npz",
    rollouts="results/gym/lunar_lander_slow_fast/rollouts.jsonl",
    live_log="results/gym/lunar_lander_slow_fast/live.log",
    device="cuda:1",
    seed_start=COLLECTION_SEED_START,
    episodes=200,
    fast_quantile=0.30,
    slow_quantile=0.70,
    max_start_distance=0.50,
    max_terrain_distance=0.50,
    max_state_distance=0.75,
)


def validate(cfg):
    if set(cfg) != set(DEFAULTS):
        raise ValueError(f"Unexpected config keys: {set(cfg) ^ set(DEFAULTS)}")
    for key in ("checkpoint", "out", "rollouts", "live_log", "device"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if str(cfg["device"]).startswith("cuda") and cfg["device"] != "cuda:1":
        raise ValueError("Experiments must use cuda:1; GPU 0 belongs to the user")
    for key in ("seed_start", "episodes"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["episodes"] < 4:
        raise ValueError("episodes must be at least 4 so the pools can split")
    if not 0 < cfg["fast_quantile"] < cfg["slow_quantile"] < 1:
        raise ValueError("need 0 < fast_quantile < slow_quantile < 1")
    for key in ("max_start_distance", "max_terrain_distance", "max_state_distance"):
        if isinstance(cfg[key], bool) or type(cfg[key]) not in (int, float) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative number")


def _log_to(live, message):
    text = f"[slow-fast-collect] {message}"
    print(text, flush=True)
    live.write(text + "\n")
    live.flush()


def rollout_seeds(bundle, seeds, rollouts_path, live):
    """Append one json object per finished episode. Crashes are stored, not paired."""
    from lib.gym_control import control_action
    from lib.gym_control_evaluation import OFF_ACTION, terminal_reason
    from lib.gym_data import make_env, terrain_context

    rollouts_path = Path(rollouts_path)
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if rollouts_path.exists():
        for episode in read_jsonl(rollouts_path):
            done.add(int(episode["seed"]))
    env = make_env()
    written = 0
    try:
        with rollouts_path.open("a", buffering=1) as handle:
            for seed in seeds:
                if seed in done:
                    _log_to(live, f"seed={seed} skip=already_recorded")
                    continue
                state, _ = env.reset(seed=int(seed))
                initial, terrain = state.copy(), terrain_context(env)
                previous = OFF_ACTION.copy()
                states, actions = [], []
                total = 0.
                while True:
                    action, _route = control_action(bundle, state, previous, terrain)
                    action = action.astype("float32", copy=False)
                    following, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    states.append(state.tolist())
                    actions.append(action.tolist())
                    state, previous = following, action
                    if terminated or truncated:
                        outcome = terminal_reason(env, state, terminated, truncated)
                        episode = dict(
                            seed=int(seed), steps=len(actions), outcome=outcome,
                            initial_state=initial.tolist(), terrain=terrain.tolist(),
                            states=states, actions=actions, game_over=bool(env.unwrapped.game_over),
                            lander_awake=bool(env.unwrapped.lander.awake), terminated=bool(terminated),
                            truncated=bool(truncated))
                        episode["return"] = total
                        handle.write(json.dumps(episode) + "\n")
                        handle.flush()
                        written += 1
                        _log_to(live, f"seed={seed} outcome={outcome} steps={episode['steps']} "
                                      f"success={outcome == SUCCESS}")
                        break
                    if len(actions) >= 1000:
                        raise RuntimeError("Pinned TimeLimit failed to end episode")
    finally:
        env.close()
    return written


def pair_from_episodes(episodes, cfg, live):
    assert_collection_seeds(int(episode["seed"]) for episode in episodes)
    try:
        built = build_slow_fast_pairs(
            episodes, fast_quantile=cfg["fast_quantile"], slow_quantile=cfg["slow_quantile"],
            max_start_distance=cfg["max_start_distance"], max_terrain_distance=cfg["max_terrain_distance"],
            max_state_distance=cfg["max_state_distance"])
    except ValueError as exc:
        _log_to(live, f"PAIR FAILED {exc}")
        _log_to(live, "Rollouts stay on disk. Retune with --from-jsonl, or collect more landings.")
        raise
    path, sidecar = save_pairs(cfg["out"], built)
    manifest = built["manifest"]
    _log_to(live, "POOLS "
            f"successes={manifest['successes']} fast={manifest['fast_pool']} slow={manifest['slow_pool']} "
            f"pairs={manifest['pairs']} rows={manifest['rows']} "
            f"crashes_excluded={manifest['crashes_excluded']} "
            f"timeouts_excluded={manifest['timeouts_excluded']} "
            f"oob_excluded={manifest['oob_excluded']} "
            f"fast_steps_max={manifest['fast_steps_max']} slow_steps_min={manifest['slow_steps_min']} "
            f"mean_action_edit={manifest['mean_action_edit']:.4f}")
    nearest = manifest["nearest"]
    _log_to(live, "NEAREST "
            f"start_p50={nearest.get('start_p50')} start_p90={nearest.get('start_p90')} "
            f"terrain_p50={nearest.get('terrain_p50')} terrain_p90={nearest.get('terrain_p90')} "
            f"unmatched_slow={manifest['unmatched_slow']} dropped_state_rows={manifest['dropped_state_rows']}")
    _log_to(live, f"WROTE pairs={path} manifest={sidecar}")
    return manifest


def collect(cfg, smoke=False, from_jsonl=False):
    validate(cfg)
    live_path = Path(cfg["live_log"])
    live_path.parent.mkdir(parents=True, exist_ok=True)
    with live_path.open("a", buffering=1) as live:
        if smoke:
            _log_to(live, "SMOKE synthetic pairs. Not Lunar landings and not a checkpoint rollout.")
            return pair_from_episodes(synthetic_episodes(), cfg, live)
        if from_jsonl:
            episodes = read_jsonl(cfg["rollouts"])
            _log_to(live, f"PAIR from {cfg['rollouts']} episodes={len(episodes)}")
            return pair_from_episodes(episodes, cfg, live)
        seeds = assert_collection_seeds(range(cfg["seed_start"], cfg["seed_start"] + cfg["episodes"]))
        _log_to(live, f"START checkpoint={cfg['checkpoint']} device={cfg['device']} "
                      f"seeds={seeds[0]}-{seeds[-1]} n={len(seeds)} adv_weight=1 safe_fast_weight=0")
        started = time.perf_counter()
        bundle = load_paired_controller(cfg["checkpoint"], cfg["device"])
        written = rollout_seeds(bundle, seeds, cfg["rollouts"], live)
        _log_to(live, f"ROLLOUTS wrote={written} file={cfg['rollouts']} "
                      f"elapsed_s={time.perf_counter() - started:.1f}")
        return pair_from_episodes(read_jsonl(cfg["rollouts"]), cfg, live)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    parser.add_argument("--rollouts")
    parser.add_argument("--device")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--fast-quantile", type=float)
    parser.add_argument("--slow-quantile", type=float)
    parser.add_argument("--max-start-distance", type=float)
    parser.add_argument("--max-terrain-distance", type=float)
    parser.add_argument("--max-state-distance", type=float)
    parser.add_argument("--smoke", action="store_true",
                        help="Write pairs from tiny synthetic successes. Not a Lunar result.")
    parser.add_argument("--from-jsonl", action="store_true",
                        help="Rebuild pairs from the rollouts jsonl. Does not touch the simulator.")
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    cli = dict(checkpoint=args.checkpoint, out=args.out, rollouts=args.rollouts, device=args.device,
               seed_start=args.seed_start, episodes=args.episodes, fast_quantile=args.fast_quantile,
               slow_quantile=args.slow_quantile, max_start_distance=args.max_start_distance,
               max_terrain_distance=args.max_terrain_distance, max_state_distance=args.max_state_distance)
    for key, value in cli.items():
        if value is not None:
            cfg[key] = value
    if args.smoke and args.from_jsonl:
        parser.error("Choose one of --smoke or --from-jsonl")
    collect(cfg, smoke=args.smoke, from_jsonl=args.from_jsonl)


if __name__ == "__main__":
    main()
