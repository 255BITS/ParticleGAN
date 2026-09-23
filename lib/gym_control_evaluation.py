"""Actual simulator control metrics, separate from world-prediction metrics."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time

import numpy as np

from lib.gym_data import make_env, sha256, simulator_provenance, terrain_context

OFF_ACTION = np.array([-1., 0.], dtype=np.float32)
OUTCOMES = ("successful_landing", "crash", "out_of_bounds", "time_limit")


def terminal_reason(env, state, terminated, truncated):
    """Follow pinned simulator reward/termination precedence, including sleep.

    LunarLander's final sleep check overwrites crash/out-of-bounds reward with
    +100. Keep raw flags in the episode record so this precedence is auditable.
    """
    base = env.unwrapped
    if terminated:
        if not base.lander.awake:
            return "successful_landing"
        if base.game_over:
            return "crash"
        if abs(float(state[0])) >= 1.:
            return "out_of_bounds"
        raise RuntimeError("Unrecognized terminal condition in pinned simulator")
    return "time_limit" if truncated else None


def termination_reason(env, terminated, truncated):
    from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_W
    half_width = VIEWPORT_W / SCALE / 2
    state = [(env.unwrapped.lander.position.x-half_width)/half_width]
    return terminal_reason(env, state, terminated, truncated)


def wilson_interval(successes, count, z=1.959963984540054):
    if not 0 <= successes <= count or count <= 0:
        raise ValueError("Expected 0 <= successes <= positive count")
    rate = successes/count
    denominator = 1 + z*z/count
    center = (rate + z*z/(2*count))/denominator
    radius = z*math.sqrt(rate*(1-rate)/count + z*z/(4*count*count))/denominator
    return [0. if successes == 0 else max(0., center-radius),
            1. if successes == count else min(1., center+radius)]


def engine_regimes(actions):
    actions = np.asarray(actions)
    return np.stack([actions[..., 0] > 0,
                     np.where(np.abs(actions[..., 1]) > .5, np.sign(actions[..., 1]), 0)], -1)


def freeze_protocol(path, episodes_path):
    path, episodes_path = Path(path), Path(episodes_path)
    episodes = json.loads(episodes_path.read_text())
    used = {int(e["seed"]) for e in episodes}
    validation, test = list(range(391000, 391020)), list(range(491000, 491050))
    if used.intersection(validation+test) or set(validation).intersection(test):
        raise ValueError("Control evaluation reset seeds overlap existing data")
    value = dict(version=1, validation_seeds=validation, test_seeds=test,
        candidate_updates=[250,1000,2500], training_updates=2500, training_batch_size=256,
        source_episodes_sha256=sha256(episodes_path), source_episodes=str(episodes_path.resolve()),
        source_reset_seeds=sorted(used), simulator=simulator_provenance(),
        selection=["validation landing rate descending", "validation mean return descending",
                   "earlier update in an exact tie"],
        success="terminated and not env.unwrapped.lander.awake; final simulator sleep check takes precedence",
        outcome_precedence=list(OUTCOMES), landing_interval="Wilson score 95%; finite episode sample",
        previous_action_at_reset=OFF_ACTION.tolist(), max_episode_steps=1000,
        inference="E_control(state, previous action, terrain) -> z -> G2; actual simulator advances state",
        training_seed_repeats=False,
        inference_timing="Controller call including routing/offset diagnostics; simulator and rendering excluded",
        sources={str(p): sha256(p) for p in ([Path(__file__)] +
            [Path(__file__).parents[1]/name for name in ["experiments/evaluate_gym_control.py",
             "experiments/train_gym_transition.py", "lib/gym_control.py", "lib/gym_transition.py", "lib/gym_data.py"]] +
            sorted((Path(__file__).parents[1]/"particlegan").glob("*.py")))})
    if path.exists():
        old = json.loads(path.read_text())
        if old != value:
            raise RuntimeError("Frozen control protocol differs; choose a new report directory")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")
    return value


def verify_protocol(path):
    value = json.loads(Path(path).read_text())
    if simulator_provenance() != value["simulator"]:
        raise RuntimeError("Simulator differs from frozen protocol")
    if sha256(value["source_episodes"]) != value["source_episodes_sha256"]:
        raise RuntimeError("Source episodes changed")
    for filename, digest in value["sources"].items():
        if sha256(filename) != digest:
            raise RuntimeError(f"Evaluation source changed: {filename}")
    return value


def summarize_episodes(episodes):
    if not episodes:
        raise ValueError("No episodes")
    counts = {k: sum(e["outcome"] == k for e in episodes) for k in OUTCOMES}
    returns = np.asarray([e["return"] for e in episodes])
    steps = sum(e["steps"] for e in episodes)
    inference = sum(e["inference_seconds"] for e in episodes)
    return dict(episodes=len(episodes), landing_count=counts["successful_landing"],
        landing_rate=counts["successful_landing"]/len(episodes),
        landing_rate_wilson95=wilson_interval(counts["successful_landing"], len(episodes)),
        mean_return=float(returns.mean()), median_return=float(np.median(returns)),
        outcomes=counts, mean_episode_steps=float(np.mean([e["steps"] for e in episodes])),
        main_engine_fraction=sum(e["main_engine_steps"] for e in episodes)/steps,
        lateral_engine_fraction=sum(e["lateral_engine_steps"] for e in episodes)/steps,
        mean_main_power=sum(e["main_power_sum"] for e in episodes)/steps,
        mean_lateral_power=sum(e["lateral_power_sum"] for e in episodes)/steps,
        repeated_command_fraction=sum(e["repeated_commands"] for e in episodes)/steps,
        inference_ms_per_step=1000*inference/steps,
        simulator_steps=steps, simulator_resets=len(episodes),
        simulator_steps_including_reset=steps+len(episodes),
        terminated_count=sum(e["terminated"] for e in episodes),
        truncated_count=sum(e["truncated"] for e in episodes))


def paired_comparison(episodes, reference):
    left, right = {e["seed"]: e for e in episodes}, {e["seed"]: e for e in reference}
    if set(left) != set(right) or len(left) != len(episodes) or len(right) != len(reference):
        raise ValueError("Paired comparisons require identical unique reset seeds")
    differences = np.array([left[s]["return"]-right[s]["return"] for s in sorted(left)])
    land = [(left[s]["outcome"] == "successful_landing", right[s]["outcome"] == "successful_landing") for s in sorted(left)]
    return dict(episodes=len(left), return_wins=int((differences > 1e-6).sum()),
        return_losses=int((differences < -1e-6).sum()), return_ties=int((np.abs(differences) <= 1e-6).sum()),
        mean_return_difference=float(differences.mean()), median_return_difference=float(np.median(differences)),
        landing_wins=sum(a and not b for a,b in land), landing_losses=sum(b and not a for a,b in land),
        landing_ties=sum(a == b for a,b in land),
        per_episode_return_difference={str(s): float(d) for s,d in zip(sorted(left), differences)})


def evaluate_controller(action_fn, seeds, trace_path, label):
    """action_fn(env,state,previous,terrain) returns (physical action, metadata)."""
    env, episodes, traces = make_env(), [], []
    start = time.perf_counter()
    try:
        for seed in seeds:
            state, _ = env.reset(seed=int(seed))
            initial_state, terrain = state.copy(), terrain_context(env)
            previous = OFF_ACTION.copy()
            total = elapsed = main_power = side_power = 0.
            main_steps = side_steps = repeated = steps = 0
            while True:
                before = time.perf_counter()
                action, metadata = action_fn(env, state, previous, terrain)
                elapsed += time.perf_counter()-before
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (2,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
                    raise ValueError("Controller produced an invalid physical action")
                following, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                main_steps += bool(action[0] > 0)
                side_steps += bool(abs(action[1]) > .5)
                main_power += float((action[0]+1)/2 if action[0] > 0 else 0)
                side_power += float(abs(action[1]) if abs(action[1]) > .5 else 0)
                repeated += bool(np.allclose(action, previous, atol=1e-6, rtol=0))
                traces.append((int(seed), steps, state.copy(), previous.copy(), action.copy(), following.copy(),
                               float(reward), int(metadata.get("component_id", -1)),
                               float(metadata.get("offset_saturation", float("nan"))),
                               float(metadata.get("offset_norm", float("nan")))))
                steps += 1
                state, previous = following, action
                if terminated or truncated:
                    outcome = terminal_reason(env, state, terminated, truncated)
                    episode = dict(seed=int(seed), steps=steps, outcome=outcome, terminated=bool(terminated),
                        truncated=bool(truncated), final_reward=float(reward), initial_state=initial_state.tolist(),
                        final_state=state.tolist(), terrain=terrain.tolist(), game_over=bool(env.unwrapped.game_over),
                        lander_awake=bool(env.unwrapped.lander.awake), inference_seconds=elapsed,
                        main_engine_steps=main_steps, lateral_engine_steps=side_steps, main_power_sum=main_power,
                        lateral_power_sum=side_power, repeated_commands=repeated)
                    episode["return"] = total
                    episodes.append(episode)
                    logging.info("CONTROL %s seed=%d outcome=%s steps=%d return=%.3f", label, seed, outcome, steps, total)
                    break
                if steps >= 1000:
                    raise RuntimeError("Pinned TimeLimit failed to end episode")
    finally:
        env.close()
    names = ("seeds", "steps", "states", "previous_actions", "actions", "next_states", "rewards", "component_ids", "offset_saturation", "offset_norm")
    arrays = {name: np.asarray([row[i] for row in traces]) for i,name in enumerate(names)}
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **arrays)
    summary = summarize_episodes(episodes)
    routes, counts = np.unique(arrays["component_ids"][arrays["component_ids"] >= 0], return_counts=True)
    summary["route_count"] = len(routes)
    summary["route_counts"] = {str(k): int(v) for k,v in zip(routes,counts)}
    for name in ("offset_saturation", "offset_norm"):
        finite = arrays[name][np.isfinite(arrays[name])]
        summary["mean_"+name] = float(finite.mean()) if len(finite) else None
    summary["evaluation_seconds"] = time.perf_counter()-start
    summary["realtime_speed_ratio"] = summary["simulator_steps"]/50/summary["evaluation_seconds"]
    return dict(summary=summary, episodes=episodes, traces=str(Path(trace_path).resolve()), traces_sha256=sha256(trace_path))


def selection_key(row):
    score = row["summary"]
    return score["landing_rate"], score["mean_return"], -int(row.get("step", 0))
