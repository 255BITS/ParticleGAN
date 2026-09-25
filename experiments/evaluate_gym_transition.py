#!/usr/bin/env python
"""Evaluate pinned checkpoints and persistence on the frozen Lunar Lander data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_data import load_split, sha256
from lib.gym_evaluation import (PROTOCOL, action_metrics, distribution_metrics, fit_scales,
    normalize_joint, phases, prediction_metrics, score_predictions)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def predictor(checkpoint, device):
    if checkpoint is None:
        return lambda s, a, c: np.asarray(s).copy(), None
    from experiments.train_gym_transition import load_checkpoint, predict
    bundle = load_checkpoint(checkpoint, device=device)
    def call(s, a, c):
        return predict(bundle, torch.as_tensor(s, dtype=torch.float32, device=device),
                       torch.as_tensor(a, dtype=torch.float32, device=device),
                       torch.as_tensor(c, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return call, bundle


def rollout_metrics(call, test, episodes, scale):
    """Stored reference commands with no observation refresh, ending at reference end."""
    horizons = PROTOCOL["rollout_horizons"]
    predictions, targets = {h: [] for h in horizons}, {h: [] for h in horizons}
    selected, scenes = [], []
    by_id = {int(e["episode_id"]): e for e in episodes}
    for episode_id in sorted(np.unique(test["episode_ids"]))[:8]:
        episode = by_id[int(episode_id)]
        behavior_phases = phases(np.asarray(episode["states"]), np.asarray(episode["next_states"]))
        episode_rows = np.flatnonzero(test["episode_ids"] == episode_id)
        for phase in ("flight", "approach", "contact"):
            candidates = episode_rows[behavior_phases[test["anchor_steps"][episode_rows]] == phase]
            if not len(candidates):
                continue
            anchor = int(test["anchor_steps"][candidates].min())
            actions = np.asarray(episode["actions"][anchor:anchor+max(horizons)], np.float32)
            real = np.asarray(episode["next_states"][anchor:anchor+max(horizons)], np.float32)
            terrain = np.asarray(episode["terrain"], np.float32)[None]
            state = np.asarray(episode["states"][anchor], np.float32)[None]
            trajectory = []
            for step, action in enumerate(actions, 1):
                state = call(state, action[None], terrain)
                trajectory.append(state[0].copy())
                if step in horizons:
                    predictions[step].append(state[0].copy())
                    targets[step].append(real[step-1])
                state[:, 6:] = (state[:, 6:] >= .5).astype(np.float32)
            selected.append({"episode_id": int(episode_id), "anchor": anchor,
                             "phase": phase, "length": len(actions)})
            scenes.append({**selected[-1], "terrain": terrain[0].tolist(),
                "initial_state": episode["states"][anchor], "actions": actions.tolist(),
                "real": real.tolist(), "predicted": np.asarray(trajectory).tolist()})
    scores = {str(h): prediction_metrics(np.asarray(predictions[h]), np.asarray(targets[h]), scale)
              for h in horizons}
    return {"horizons": scores, "anchors": selected, "termination": "stop at saved reference episode end"}, scenes


def counterfactual_metrics(call, probe, scale):
    commands = list(probe["command_inputs"])
    actions = np.asarray([probe["command_inputs"][name] for name in commands], np.float32)
    errors, scenes = {name: [] for name in commands}, []
    magnitudes = {name: [] for name in commands}
    for anchor in probe["anchors"]:
        predicted = call(np.repeat(np.asarray(anchor["current_state"], np.float32)[None], len(actions), 0),
                         actions, np.repeat(np.asarray(anchor["terrain"], np.float32)[None], len(actions), 0))
        real = np.asarray([anchor["commands"][name]["next_state"] for name in commands])
        off = commands.index("off")
        real_delta = (real[:, :6]-real[off, :6])/scale
        predicted_delta = (predicted[:, :6]-predicted[off, :6])/scale
        for i, name in enumerate(commands):
            errors[name].append(float(((predicted_delta[i]-real_delta[i])**2).mean()))
            magnitudes[name].append(float((real_delta[i]**2).mean()))
        scenes.append({"episode_id": anchor["episode_id"], "anchor": anchor["anchor_step"],
            "phase": anchor["phase"], "terrain": anchor["terrain"], "initial_state": anchor["current_state"],
            "command_names": commands, "actions": actions.tolist(), "real": real.tolist(), "predicted": predicted.tolist()})
    return {name: {"anchors": len(values), "effect_mse_vs_off": float(np.mean(values)),
                    "zero_response_mse": float(np.mean(magnitudes[name]))}
            for name, values in errors.items()}, scenes


@torch.no_grad()
def generation_metrics(bundle, train, test, device):
    from lib.gym_transition import contact_record, composed_transition, encoded_transition
    mean, scale = fit_scales(train)
    action_mean, action_scale = train["actions"].mean(0), train["actions"].std(0).clip(.001)
    normalize = lambda s, a, n: normalize_joint(s, a, n, mean, scale, action_mean, action_scale)
    rng = torch.Generator(device=device).manual_seed(92173)
    subset_rng = np.random.default_rng(92173)
    g, e, prior, scaler = [bundle[k] for k in ("G", "E", "prior", "scaler")]
    results = []
    gallery = []
    for episode_id in sorted(np.unique(test["episode_ids"])):
        ids = np.flatnonzero(test["episode_ids"] == episode_id)
        if len(ids) < 64:
            continue
        ids = subset_rng.choice(ids, min(512, len(ids)), replace=False)
        c = torch.as_tensor(test["terrain"][ids], device=device)
        fake = contact_record(g(prior.sample(len(ids), rng)[0], c), rng=rng)
        composed, _, _ = composed_transition(e, g, prior, fake, c, rng=rng)
        real = normalize(test["states"][ids], test["actions"][ids], test["next_states"][ids])
        row = {"episode_id": int(episode_id), "count": len(ids)}
        for name, output in (("prior", fake), ("composed", composed)):
            physical = scaler.inverse(output).cpu().numpy()
            normalized = normalize(physical[:, :8], physical[:, 8:10], physical[:, 10:])
            row[name] = distribution_metrics(normalized, real, device)
            real_phases = phases(test["states"][ids], test["next_states"][ids])
            fake_phases = phases(physical[:, :8], physical[:, 10:])
            row[name]["phase_frequencies"] = {p: float(np.mean(fake_phases == p)) for p in ("flight", "approach", "contact")}
            row[name]["reference_phase_frequencies"] = {p: float(np.mean(real_phases == p)) for p in ("flight", "approach", "contact")}
            if len(gallery) < 4:
                gallery.append({"episode_id": int(episode_id), "mode": name,
                                "terrain": test["terrain"][ids[0]].tolist(), "transitions": physical[:24].tolist()})
        # Random disjoint halves characterize finite-reference differences. The
        # halves may contain sibling actions, so this is not an iid error floor.
        n = len(real)//2
        row["reference_halves"] = distribution_metrics(real[:n], real[n:2*n], device)
        results.append(row)
        if len(results) >= 16:
            break
    if not results:
        raise ValueError("No held-out terrain has 64 reference records")
    summary = {name: {key: float(np.mean([r[name][key] for r in results]))
                       for key in ("joint_sw1", "coverage", "precision", "contact_pattern_tv")}
               for name in ("prior", "composed", "reference_halves")}
    # Real-input reconstruction and encoder routing diagnostic; no target next
    # state is passed into E.
    real = torch.as_tensor(np.concatenate([test["states"], test["actions"], test["next_states"]], 1), device=device)
    terrain = torch.as_tensor(test["terrain"], device=device)
    decoded, ids_all, offset_all = [], [], []
    for start in range(0, len(real), 512):
        x = scaler(real[start:start+512])
        out, encoding = encoded_transition(e, g, prior, x[:, :10], terrain[start:start+512])
        decoded.append(scaler.inverse(contact_record(out, mode="probability")).cpu().numpy())
        ids_all.append(encoding.indices.cpu().numpy().reshape(-1))
        bounded_offset = (encoding.codes[:, 0] - prior.means()[encoding.indices[:, 0]]) / prior.sigma
        offset_all.append(bounded_offset.cpu().numpy())
    decoded = np.concatenate(decoded)
    counts = np.bincount(np.concatenate(ids_all), minlength=prior.num_particles)
    frequencies = counts[counts > 0]/counts.sum()
    offsets = np.concatenate(offset_all)
    summary["reconstruction"] = {
        "state": prediction_metrics(decoded[:, :8], test["states"], scale),
        "action_standardized_mse": float((((decoded[:, 8:10]-test["actions"])/action_scale)**2).mean()),
        "used_components": len(np.unique(np.concatenate(ids_all))),
        "route_counts": counts.tolist(),
        "route_entropy_nats": float(-(frequencies*np.log(frequencies)).sum()),
        "bounded_offset_l2_mean": float(np.linalg.norm(offsets, axis=1).mean()),
        "bounded_offset_abs_p95": float(np.quantile(np.abs(offsets), .95)),
        "offset_coordinates_near_bound_fraction": float((np.abs(offsets) >= 2.85).mean()),
    }
    summary["contexts"] = results
    return summary, gallery


def verify_checkpoint(bundle, checkpoint, frozen):
    """Reject mismatched training data or inference code before ranking a run."""
    provenance = bundle["provenance"]
    if provenance["dataset"] != frozen["references"]:
        raise ValueError("Checkpoint dataset hashes differ from the frozen evaluation references")
    for name, expected in provenance["sources"].items():
        if name in ("lib/gym_transition.py", "experiments/train_gym_transition.py") or name.startswith("particlegan/"):
            if sha256(ROOT/name) != expected:
                raise ValueError(f"Checkpoint inference source changed: {name}")
    summary = json.loads((Path(checkpoint).parent/"summary.json").read_text())
    if summary["checkpoints"].get(Path(checkpoint).name) != sha256(checkpoint):
        raise ValueError("Checkpoint hash differs from completed training summary")
    return summary


def evaluate(data_dir, checkpoints, out_dir, device="cpu", probe_path="reports/gym/lunar_lander/simulator_probe.json"):
    torch.set_num_threads(1)
    out = Path(out_dir)
    if (out/"leaderboard.json").exists():
        raise FileExistsError("Use a fresh evaluation directory; boards are immutable")
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_split(data_dir, "train"), load_split(data_dir, "test")
    episodes_file = Path(data_dir)/"episodes.json"
    episodes = json.loads(episodes_file.read_text())
    if isinstance(episodes, dict):
        episodes = episodes["episodes"]
    _, scale = fit_scales(train)
    probe = json.loads(Path(probe_path).read_text())
    frozen = {"definitions": PROTOCOL,
              "references": {name: sha256(Path(data_dir)/name) for name in
                             ("train.npz", "validation.npz", "test.npz", "episodes.json", "metadata.json")},
              "evaluation_sources": {name: sha256(ROOT/name) for name in
                                     ("lib/gym_evaluation.py", "experiments/evaluate_gym_transition.py")},
              "simulator_probe_sha256": sha256(probe_path)}
    for key, filename in (("train_sha256", "train.npz"), ("test_sha256", "test.npz"), ("episodes_sha256", "episodes.json")):
        if probe[key] != frozen["references"][filename]:
            raise ValueError("Simulator probe belongs to a different frozen dataset")
    write_json(out/"protocol.json", frozen)
    rows, artifacts = [], {}
    for name, checkpoint in [("persistence", None)] + list(checkpoints):
        print(f"Evaluating {name}", flush=True)
        call, bundle = predictor(checkpoint, device)
        if checkpoint:
            verify_checkpoint(bundle, checkpoint, frozen)
        call(test["states"][:16], test["actions"][:16], test["terrain"][:16])
        start = time.perf_counter()
        predicted = call(test["states"], test["actions"], test["terrain"])
        seconds = time.perf_counter()-start
        conditional = score_predictions(predicted, test, scale)
        row = {"name": name, "conditional": conditional, "action_response": action_metrics(predicted, test, scale),
               "inference_samples_per_second": len(test["states"])/max(seconds, 1e-9)}
        row["rollout"], scenes = rollout_metrics(call, test, episodes, scale)
        row["counterfactual_commands"], action_scenes = counterfactual_metrics(call, probe, scale)
        gallery = []
        if checkpoint:
            row["checkpoint"] = str(Path(checkpoint))
            row["checkpoint_sha256"] = sha256(checkpoint)
            summary_path = Path(checkpoint).parent/"summary.json"
            row["training"] = json.loads(summary_path.read_text())
            row["training_summary_sha256"] = sha256(summary_path)
            if bundle.get("G") is not None:
                row["generation"], gallery = generation_metrics(bundle, train, test, device)
        rows.append(row)
        np.savez_compressed(out/f"{name}_predictions.npz", next_states=predicted)
        artifacts[name] = {"scenes": scenes, "gallery": gallery, "action_scenes": action_scenes}
        write_json(out/f"{name}.json", row)
        print(f"{name}: continuous MSE={conditional['continuous_mse']:.8f}, contact Brier={conditional['contact_brier']:.6f}", flush=True)
    baseline = rows[0]["conditional"]["continuous_mse"]
    for row in rows:
        row["improvement_over_persistence_percent"] = 100*(1-row["conditional"]["continuous_mse"]/baseline)
    rows.sort(key=lambda row: row["conditional"]["continuous_mse"])
    write_json(out/"leaderboard.json", {"protocol": frozen, "rows": rows})
    write_json(out/"demo_data.json", artifacts)
    text = ["# Lunar Lander baseline leaderboard", "",
        "Single transitions; 11 terrain heights are privileged context. Lower errors are better.", "",
        "| Model/checkpoint | Next-state MSE | vs persistence | Contact Brier | Action-effect MSE | 20-step MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        horizon = r["rollout"]["horizons"]["20"].get("continuous_mse")
        horizon_text = "n/a" if horizon is None else f"{horizon:.6f}"
        text.append(f"| {r['name']} | {r['conditional']['continuous_mse']:.6f} | {r['improvement_over_persistence_percent']:+.1f}% | "
                    f"{r['conditional']['contact_brier']:.5f} | {r['action_response']['all']['effect_mse']:.6f} | {horizon_text} |")
    text += ["", "Checkpoints selected by validation continuous MSE; final checkpoints also listed. "
             "Recursive contacts use p >= .5, and evaluation ends at reference termination. "
             "The deterministic encoder predicts a point; joint generation does not establish conditional uncertainty.", ""]
    (out/"README.md").write_text("\n".join(text))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="results/gym/lunar_lander/data")
    parser.add_argument("--checkpoint", action="append", default=[], help="name=checkpoint path; repeat")
    parser.add_argument("--out-dir", default="reports/gym/lunar_lander/baseline")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--probe", default="reports/gym/lunar_lander/simulator_probe.json")
    args = parser.parse_args()
    evaluate(args.data_dir, [item.split("=", 1) for item in args.checkpoint], args.out_dir, args.device, args.probe)


if __name__ == "__main__":
    main()
