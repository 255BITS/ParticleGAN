"""Evaluation-only Lunar Lander replay, engine-boundary and noise diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.gym_data import (PHASE_NAMES, load_split, make_env, phases, replay_anchor,
                          replay_episode, sha256, simulator_provenance)


COMMANDS = {
    "off": [-1., 0.], "main_boundary": [0., 0.], "main_above": [1e-6, 0.],
    "main_full": [1., 0.], "left_boundary": [-1., -.5],
    "left_above": [-1., -.500001], "left_full": [-1., -1.],
    "right_boundary": [-1., .5], "right_above": [-1., .500001],
    "right_full": [-1., 1.], "combined": [1., 1.],
}


def state_scale(training):
    # Same population std, six fields and 1e-3 floor as GymTransitionScaler.fit.
    values = np.concatenate([training["states"][:, :6], training["next_states"][:, :6]])
    return np.maximum(values.astype(np.float64).std(0), 1e-3)


def choose_anchors(test, episodes, maximum_episodes=8):
    chosen = []
    episode_map = {ep["episode_id"]: ep for ep in episodes if ep["split"] == "test"}
    for episode_id in sorted(np.unique(test["episode_ids"]))[:maximum_episodes]:
        episode = episode_map[int(episode_id)]
        anchor_steps = np.unique(test["anchor_steps"][test["episode_ids"] == episode_id])
        labels = phases(np.asarray(episode["states"]), np.asarray(episode["next_states"]))
        for phase_id, phase_name in enumerate(PHASE_NAMES):
            matching = anchor_steps[labels[anchor_steps] == phase_id]
            if len(matching):
                chosen.append((episode, int(matching[0]), phase_name))
    return chosen


def verify_episodes(episodes, count=3):
    results = []
    for episode in sorted((ep for ep in episodes if ep["split"] == "test"), key=lambda ep: ep["episode_id"])[:count]:
        replayed = replay_episode(episode["seed"], episode["actions"])
        for key in ("states", "actions", "next_states", "rewards", "terminated", "truncated"):
            expected = np.asarray(episode[key], dtype=replayed[key].dtype)
            if not np.array_equal(replayed[key], expected):
                raise RuntimeError(f"Saved episode {episode['episode_id']} replay mismatch: {key}")
        results.append(dict(episode_id=episode["episode_id"], steps=len(replayed["states"]),
                            contact_rows=int(np.any(replayed["states"][:, 6:] > .5, axis=1).sum()),
                            exact_replay=True))
    return results


def probe_anchor(env, episode, step, phase, scale, noise_samples=16):
    prefix = episode["actions"][:step]
    expected = np.asarray(episode["states"][step], np.float32)
    outcomes, rng_hashes = {}, []
    for name, command in COMMANDS.items():
        state = replay_anchor(env, episode["seed"], prefix)
        if not np.array_equal(state, expected):
            raise RuntimeError("Counterfactual replay changed the physical anchor")
        encoded_rng = json.dumps(env.unwrapped.np_random.bit_generator.state, sort_keys=True).encode()
        rng_hashes.append(hashlib.sha256(encoded_rng).hexdigest())
        following, reward, terminated, truncated, _ = env.step(np.asarray(command, np.float32))
        outcomes[name] = dict(next_state=following.tolist(), reward=float(reward),
                              terminated=bool(terminated), truncated=bool(truncated))
    if len(set(rng_hashes)) != 1:
        raise RuntimeError("Counterfactual commands did not share pre-step RNG state")
    off = np.asarray(outcomes["off"]["next_state"])
    for name in ("main_boundary", "left_boundary", "right_boundary"):
        if not np.array_equal(outcomes[name]["next_state"], off):
            raise RuntimeError(f"Pinned engine boundary changed: {name}")
    for name, outcome in outcomes.items():
        delta = (np.asarray(outcome["next_state"])[:6] - off[:6]) / scale
        outcome["standardized_squared_effect_vs_off"] = float(np.mean(delta ** 2))
    noisy = []
    for index in range(noise_samples):
        restored = replay_anchor(env, episode["seed"], prefix)
        if not np.array_equal(restored, expected):
            raise RuntimeError("Noise probe changed the pre-action physical anchor")
        # Replace only the RNG after replay, preserving the Box2D world. Wind is
        # off, so these draws perturb engine dispersion rather than terrain.
        env.unwrapped.np_random = np.random.default_rng(450_000 + index)
        following, _, _, _, _ = env.step(np.asarray(COMMANDS["combined"], np.float32))
        noisy.append(following)
    noisy = np.asarray(noisy)
    continuous_variance = np.var(noisy[:, :6].astype(np.float64) / scale, axis=0, ddof=1)
    return dict(episode_id=episode["episode_id"], anchor_step=step, phase=phase,
                current_state=expected.tolist(), terrain=episode["terrain"],
                shared_pre_step_rng_sha256=rng_hashes[0], commands=outcomes,
                noise=dict(samples=noise_samples, command=COMMANDS["combined"],
                           rng_seeds=list(range(450_000, 450_000 + noise_samples)),
                           standardized_variance_by_coordinate=continuous_variance.tolist(),
                           mean_standardized_variance=float(continuous_variance.mean()),
                           contact_mean=noisy[:, 6:].mean(0).tolist(),
                           continuous_outcomes=noisy[:, :6].tolist()))


def run(directory, outpath, maximum_episodes=8, noise_samples=16):
    directory, outpath = Path(directory), Path(outpath)
    train, test = load_split(directory, "train"), load_split(directory, "test")
    episodes = json.loads((directory / "episodes.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    for split in ("train", "test"):
        if sha256(directory / f"{split}.npz") != metadata["splits"][split]["sha256"]:
            raise RuntimeError(f"Dataset hash mismatch: {split}")
    if sha256(directory / "episodes.json") != metadata["episodes_sha256"]:
        raise RuntimeError("Episode replay file hash mismatch")
    verified = verify_episodes(episodes)
    scale = state_scale(train)
    anchors = choose_anchors(test, episodes, maximum_episodes)
    env = make_env()
    try:
        records = [probe_anchor(env, ep, step, phase, scale, noise_samples) for ep, step, phase in anchors]
    finally:
        env.close()
    report = dict(protocol="lunar_lander_simulator_probe_v1", simulator=simulator_provenance(),
                  source_sha256=sha256(__file__), dataset_metadata_sha256=sha256(directory / "metadata.json"),
                  train_sha256=metadata["splits"]["train"]["sha256"], test_sha256=metadata["splits"]["test"]["sha256"],
                  episodes_sha256=metadata["episodes_sha256"], train_state_scale=scale.tolist(),
                  selection="First stored anchor per behavior phase in each of first eight sorted test episodes (missing phases skipped)",
                  interpretation="Evaluation-only variability of engine dispersion at exactly replayed physical states; this is not a Bayes-error floor or a training seed-repeat experiment.",
                  aggregation="Equal weight per selected anchor; sample variance ddof=1 over noise outcomes, then mean of six train-standardized coordinates",
                  command_inputs=COMMANDS, verified_complete_episodes=verified, anchors=records,
                  summary=dict(anchors=len(records), exact_replay=True, boundary_checks_passed=True,
                               mean_standardized_noise_variance=float(np.mean([r["noise"]["mean_standardized_variance"] for r in records])),
                               by_phase={name: dict(anchors=sum(r["phase"] == name for r in records),
                                                    mean_standardized_noise_variance=float(np.mean([r["noise"]["mean_standardized_variance"] for r in records if r["phase"] == name])))
                                         for name in PHASE_NAMES if any(r["phase"] == name for r in records)}))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/gym/lunar_lander/data"))
    parser.add_argument("--out", type=Path, default=Path("reports/gym/lunar_lander/simulator_probe.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.out)["summary"], indent=2), flush=True)
