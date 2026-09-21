"""Collect finite, episode-disjoint Lunar Lander transitions with replay branches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.gym_data import (PHASE_NAMES, branch_episode, make_env, phases, sha256,
                          simulator_provenance, terrain_context)


def collect_episode(seed, episode_id, split):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    env = make_env()
    rng = np.random.default_rng(seed + 10_000_000)
    # Deterministic mixture: 60% heuristic, 30% noisy heuristic, 10% exploratory.
    mode = "heuristic" if episode_id % 10 < 6 else ("perturbed" if episode_id % 10 < 9 else "exploratory")
    episode = dict(seed=seed, episode_id=episode_id, split=split, behavior=mode,
                   states=[], actions=[], next_states=[], rewards=[], terminated=[], truncated=[])
    try:
        state, _ = env.reset(seed=seed)
        episode["terrain"] = terrain_context(env).tolist()
        for step in range(1000):
            action = heuristic(env, state)
            if mode == "perturbed":
                action = np.clip(action + rng.normal(0, .35, 2), -1, 1)
            elif mode == "exploratory":
                # Correlated exploratory controls make physically distinct failures.
                if step % 8 == 0:
                    exploratory_action = rng.uniform(-1, 1, 2)
                action = exploratory_action
            action = np.asarray(action, np.float32)
            following, reward, terminated, truncated, _ = env.step(action)
            for name, value in (("states", state.tolist()), ("actions", action.tolist()),
                                ("next_states", following.tolist()), ("rewards", float(reward)),
                                ("terminated", bool(terminated)), ("truncated", bool(truncated))):
                episode[name].append(value)
            state = following
            if terminated or truncated:
                break
        return episode
    finally:
        env.close()


def select_anchors(episodes, count, rng):
    candidates = [[], [], []]
    for ep in episodes:
        labels = phases(np.asarray(ep["states"]), np.asarray(ep["next_states"]))
        for step, label in enumerate(labels):
            candidates[int(label)].append((ep["episode_id"], step))
    quotas = [int(count * .50), int(count * .35)]
    quotas.append(count - sum(quotas))
    selected, leftovers = [], []
    natural = [len(c) for c in candidates]
    for group, quota in zip(candidates, quotas):
        rng.shuffle(group)
        selected.extend(group[:quota])
        leftovers.extend(group[quota:])
    rng.shuffle(leftovers)
    selected.extend(leftovers[:count - len(selected)])
    if len(selected) != count:
        raise ValueError("Insufficient distinct source anchors")
    by_episode = {}
    for episode_id, step in selected:
        by_episode.setdefault(episode_id, []).append(step)
    return by_episode, dict(zip(PHASE_NAMES, natural))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/gym/lunar_lander/data"))
    parser.add_argument("--counts", nargs=3, type=int, default=[32768, 4096, 8192])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if any(n <= 0 or n % 4 for n in args.counts):
        parser.error("Each split size must be positive and divisible by four")
    if (args.outdir / "metadata.json").exists():
        parser.error("Refusing to overwrite a completed dataset")
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Freeze the executed collector/adapter before asynchronous collection.
    (args.outdir / "collector_source.py").write_bytes(Path(__file__).read_bytes())
    (args.outdir / "adapter_source.py").write_bytes((Path(__file__).resolve().parents[1] / "lib/gym_data.py").read_bytes())
    logpath = args.outdir.parent / ("pilot_collection.log" if args.pilot else "collection.log")
    start = time.monotonic()
    with logpath.open("a", buffering=1) as logfile:
        def log(message):
            line = f"[{time.monotonic()-start:.1f}s] {message}"
            print(line, flush=True)
            print(line, file=logfile, flush=True)
        metadata = dict(protocol="lunar_lander_replay_v1", simulator=simulator_provenance(),
                        behavior_mixture={"heuristic": .6, "perturbed": .3, "exploratory": .1},
                        target_anchor_phase_mixture={"flight": .5, "approach": .35, "contact": .15},
                        phase_definition="contact in current or next behavior state; otherwise approach y<.25, else flight",
                        shortfall_policy="Sample without replacement; redistribute phase shortfall uniformly among remaining anchors",
                        actions_per_anchor=4, behavior_action_included=True,
                        replay="Reset same episode seed and replay exact float32 prefix for each alternative",
                        dataset_seed=91000, splits={}, pilot=args.pilot,
                        simulator_step_calls=0, simulator_reset_calls=0)
        all_episodes = []
        for split_index, (split, count) in enumerate(zip(("train", "validation", "test"), args.counts)):
            episodes = []
            total = 0
            minimum = 3 if args.pilot else (64 if split == "train" else 16)
            while total < count // 4 * 2 or len(episodes) < minimum:
                episode_id = split_index * 100_000 + len(episodes)
                ep = collect_episode(91000 + episode_id, episode_id, split)
                episodes.append(ep)
                total += len(ep["actions"])
                metadata["simulator_step_calls"] += len(ep["actions"])
                metadata["simulator_reset_calls"] += 1
                if len(episodes) % 10 == 0:
                    log(f"{split}: source episodes={len(episodes)} transitions={total}")
            anchors, natural = select_anchors(episodes, count // 4, np.random.default_rng(88000 + split_index))
            tasks = [(ep, sorted(anchors[ep["episode_id"]]), 77000 + ep["episode_id"])
                     for ep in episodes if ep["episode_id"] in anchors]
            log(f"{split}: replay {count//4} anchors from {len(episodes)} episodes; natural phases={natural}")
            chunks = []
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(branch_episode, task) for task in tasks]
                for done, future in enumerate(as_completed(futures), 1):
                    chunk, calls, resets = future.result()
                    chunks.append(chunk)
                    metadata["simulator_step_calls"] += calls
                    metadata["simulator_reset_calls"] += resets
                    log(f"{split}: replay episodes {done}/{len(tasks)}, tuples={sum(len(c['states']) for c in chunks)}")
            data = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
            # Scheduling cannot affect frozen dataset order or hashes.
            order = np.lexsort((data["actions"][:, 1], data["actions"][:, 0], data["anchor_steps"], data["episode_ids"]))
            data = {key: value[order] for key, value in data.items()}
            np.savez_compressed(args.outdir / f"{split}.npz", **data)
            labels = phases(data["states"], data["next_states"])
            metadata["splits"][split] = dict(tuples=count, anchors=count//4, episodes=len(episodes),
                                             behavior_episode_counts={name: sum(ep["behavior"] == name for ep in episodes) for name in ("heuristic", "perturbed", "exploratory")},
                                             natural_phase_counts=natural,
                                             selected_transition_phase_counts={name: int((labels == i).sum()) for i, name in enumerate(PHASE_NAMES)},
                                             main_active=int((data["actions"][:, 0] > 0).sum()),
                                             lateral_active=int((np.abs(data["actions"][:, 1]) > .5).sum()),
                                             sha256=sha256(args.outdir / f"{split}.npz"))
            all_episodes.extend(episodes)
            log(f"{split}: saved {count} tuples")
        (args.outdir / "episodes.json").write_text(json.dumps(all_episodes, separators=(",", ":")))
        metadata["episodes_sha256"] = sha256(args.outdir / "episodes.json")
        metadata["wall_seconds"] = time.monotonic() - start
        metadata["simulator_calls_including_reset_internal_step"] = metadata["simulator_step_calls"] + metadata["simulator_reset_calls"]
        metadata["collector_source_sha256"] = sha256(args.outdir / "collector_source.py")
        metadata["adapter_source_sha256"] = sha256(args.outdir / "adapter_source.py")
        (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        log(f"Complete: {sum(args.counts)} tuples, {metadata['simulator_step_calls']} explicit steps, {metadata['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
