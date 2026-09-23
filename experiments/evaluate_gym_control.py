#!/usr/bin/env python
"""Frozen paired Lunar Lander control evaluation and validation-only selection."""
import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_data import sha256
from lib.gym_control_evaluation import (engine_regimes, evaluate_controller,
    freeze_protocol, paired_comparison, selection_key, verify_protocol)

ORIGINAL = ROOT/"results/gym/lunar_lander/adversarial/best.pt"
REPORTS = ROOT/"reports/gym/lunar_lander_control"
EPISODES = ROOT/"results/gym/lunar_lander/data/episodes.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def make_controller(kind, checkpoint=None, device="cpu"):
    if kind == "expert":
        from gymnasium.envs.box2d.lunar_lander import heuristic
        return lambda env,s,a,c: (heuristic(env.unwrapped, s), {}), None
    if kind == "original":
        from experiments.train_gym_transition import load_checkpoint
        from lib.gym_control import control_action_details
        bundle = load_checkpoint(checkpoint, device)
        bundle["E_control"] = bundle["E"]
        for filename,digest in bundle["provenance"]["sources"].items():
            if sha256(ROOT/filename) != digest:
                raise RuntimeError(f"Original checkpoint source changed: {filename}")
        return lambda env,s,a,c: control_action_details(bundle,s,a,c), bundle
    from lib.gym_control import load_control_checkpoint, control_action_details
    bundle = load_control_checkpoint(checkpoint, device)
    verify_control_checkpoint(bundle, checkpoint)
    if bundle["config"]["arm"] != kind:
        raise ValueError("Checkpoint arm differs from requested controller")
    return lambda env,s,a,c: control_action_details(bundle, s, a, c), bundle


def verify_control_checkpoint(bundle, checkpoint):
    provenance = bundle["provenance"]
    summary = bundle.get("training_summary")
    if not summary or sha256(checkpoint) not in summary["checkpoints"].values():
        raise RuntimeError("Control checkpoint is not a hashed completed training output")
    if summary["provenance"] != provenance:
        raise RuntimeError("Checkpoint and training summary provenance differ")
    for filename,digest in provenance["sources"].items():
        if sha256(ROOT/filename) != digest:
            raise RuntimeError(f"Control training/inference source changed: {filename}")
    for key in ("initial_checkpoint", "episodes"):
        if sha256(ROOT/provenance[key]["path"]) != provenance[key]["sha256"]:
            raise RuntimeError(f"Control source artifact changed: {key}")
    if sha256(Path(checkpoint).parent/"expert_records.npz") != provenance["expert_data"]["npz_sha256"]:
        raise RuntimeError("Frozen control expert records changed")
    if sha256(Path(checkpoint).parent/"source.zip") != provenance["source_archive_sha256"]:
        raise RuntimeError("Frozen training source archive changed")


@torch.no_grad()
def expert_action_diagnostics(bundle, episodes_path, split):
    episodes = [e for e in json.loads(Path(episodes_path).read_text())
                if e["split"] == split and e["behavior"] == "heuristic"]
    if not episodes:
        raise ValueError("Missing held-out expert episodes")
    states, previous, targets, contexts = [], [], [], []
    for episode in episodes:
        actions = np.asarray(episode["actions"], np.float32)
        states.extend(episode["states"])
        previous.extend(np.concatenate([np.array([[-1.,0.]], np.float32), actions[:-1]]))
        targets.extend(actions)
        contexts.extend([episode["terrain"]]*len(actions))
    device, scaler = bundle["device"], bundle["scaler"]
    encoder = bundle.get("E_control", bundle["E"])
    predictions = []
    for start in range(0, len(states), 1024):
        s,a,c = [torch.as_tensor(np.asarray(x[start:start+1024]), dtype=torch.float32, device=device)
                 for x in (states, previous, contexts)]
        encoding = encoder(torch.cat([scaler.state(s), scaler.action(a)], 1), c, bundle["prior"])
        predictions.append(bundle["G"].branches[1](torch.cat([encoding.codes[:,0], c], 1)).tanh().cpu().numpy())
    prediction, target = np.concatenate(predictions), np.asarray(targets, np.float32)
    p,t = [torch.as_tensor(x, device=device) for x in (prediction,target)]
    same = engine_regimes(prediction) == engine_regimes(target)
    return dict(source_split=split, source_sha256=sha256(episodes_path),
        episode_ids=[e["episode_id"] for e in episodes], records=len(target),
        standardized_action_mse=float((scaler.action(p)-scaler.action(t)).square().mean()),
        physical_action_mse=float(np.mean((prediction-target)**2)),
        main_regime_agreement=float(same[:,0].mean()), lateral_regime_agreement=float(same[:,1].mean()),
        joint_regime_agreement=float(same.all(1).mean()),
        previous_commands="expert demonstration commands; rollout uses learner commands")


def score(kind, checkpoint, split, name, out, protocol, device):
    destination = out/"evaluations"/f"{name}_{split}.json"
    digest = sha256(checkpoint) if checkpoint else None
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["checkpoint_sha256"] != digest or old["protocol_sha256"] != sha256(out/"protocol.json"):
            raise RuntimeError(f"Existing score provenance differs: {destination}")
        if sha256(old["traces"]) != old["traces_sha256"]:
            raise RuntimeError("Cached evaluation traces changed")
        return old
    action, bundle = make_controller(kind, checkpoint, device)
    if kind in ("imitation", "joint"):
        if (bundle["config"]["steps"] != protocol["training_updates"] or
                bundle["config"]["batch_size"] != protocol["training_batch_size"]):
            raise RuntimeError("Training budget differs from frozen control protocol")
    row = evaluate_controller(action, protocol[split+"_seeds"],
                              out/"traces"/f"{name}_{split}.npz", name+"/"+split)
    row.update(name=name, arm=kind, split=split, checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
        checkpoint_sha256=digest, protocol_sha256=sha256(out/"protocol.json"), device=device,
        step=int(bundle.get("step",0)) if bundle else 0)
    if bundle:
        row["expert_action_diagnostics"] = expert_action_diagnostics(bundle, protocol["source_episodes"], split)
        row["training"] = bundle.get("training_summary")
        row["checkpoint_provenance"] = bundle.get("provenance")
    write_json(destination, row)
    logging.info("SCORE %s %s landing=%d/%d mean_return=%.3f", name, split,
                 row["summary"]["landing_count"], row["summary"]["episodes"], row["summary"]["mean_return"])
    return row


def refresh_leaderboard(out):
    selections = [json.loads(p.read_text()) for p in sorted((out/"selections").glob("*.json"))]
    rows = []
    for selected in selections:
        for key in ("selected_test", "final_test"):
            if selected.get(key):
                row = json.loads(Path(selected[key]).read_text())
                row["report_role"] = "selected" if key == "selected_test" else "final"
                rows.append(row)
    references = {r["arm"]: r for r in rows if r["report_role"] == "selected"}
    for row in rows:
        row["paired"] = {name: paired_comparison(row["episodes"], references[name]["episodes"])
                         for name in ("imitation", "original") if name in references}
    # Test table order is descriptive, never used to select checkpoints/default.
    rows.sort(key=lambda r: (r["summary"]["landing_rate"], r["summary"]["mean_return"]), reverse=True)
    write_json(out/"leaderboard.json", dict(protocol_sha256=sha256(out/"protocol.json"), rows=rows))
    lines = ["# Lunar Lander control baseline", "",
        "Checkpoints and playable default are chosen on validation landing rate, then mean return. "
        "Test results below are held out; final checkpoints are reported separately. "
        "Intervals are 95% Wilson intervals over the finite paired reset episodes.", "",
        "| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit | Main / side usage |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |"]
    for row in rows:
        s = row["summary"]
        lo,hi = s["landing_rate_wilson95"]
        counts = s["outcomes"]
        lines.append(f"| {row['arm']} | {row['report_role']} ({row['step']}) | {s['landing_count']}/{s['episodes']} | {lo:.1%}–{hi:.1%} | {s['mean_return']:.2f} | {s['median_return']:.2f} | {counts['crash']} / {counts['out_of_bounds']} / {counts['time_limit']} | {s['main_engine_fraction']:.1%} / {s['lateral_engine_fraction']:.1%} |")
    lines += ["", "Per-episode results, paired return/landing wins, action error, route counts, simulator cost, and inference latency are in `leaderboard.json`. Full state/action traces are in `traces/`. The expert is a reference and is excluded from the learned-controller default selection.", ""]
    (out/"README.md").write_text("\n".join(lines))
    eligible = [s for s in selections if s["arm"] != "expert"]
    if eligible:
        winner = max(eligible, key=lambda s: selection_key(s["validation"]))
        controllers = {}
        for selected in selections:
            controllers[selected["arm"]] = dict(checkpoint=selected.get("selected_checkpoint"),
                validation=selected["validation"]["summary"], step=selected["validation"].get("step",0))
        write_json(out/"controllers.json", dict(default_controller=winner["arm"], controllers=controllers,
            selection="Validation landing rate, then mean return; expert excluded from playable default",
            protocol_sha256=sha256(out/"protocol.json")))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORTS)
    parser.add_argument("--episodes", type=Path, default=EPISODES)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--arm", choices=("imitation", "joint"))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--final", type=Path)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, default=ROOT/"results/gym/lunar_lander_control/live.log")
    args = parser.parse_args()
    if not (args.freeze or args.baseline or args.arm):
        parser.error("Choose --freeze, --baseline, or --arm")
    if args.arm and (not args.checkpoint or not args.final):
        parser.error("--arm requires --checkpoint candidate(s) and --final")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log)])
    torch.set_num_threads(1)
    if args.freeze:
        freeze_protocol(args.out/"protocol.json", args.episodes)
        logging.info("FROZEN %s", args.out/"protocol.json")
    protocol = verify_protocol(args.out/"protocol.json")
    if args.baseline:
        for arm, checkpoint in [("expert", None), ("original", args.original)]:
            validation = score(arm, checkpoint, "validation", arm, args.out, protocol, args.device)
            score(arm, checkpoint, "test", arm, args.out, protocol, args.device)
            write_json(args.out/"selections"/f"{arm}.json", dict(arm=arm, validation=validation,
                selected_checkpoint=str(checkpoint.resolve()) if checkpoint else None,
                selected_test=str((args.out/"evaluations"/f"{arm}_test.json").resolve())))
            refresh_leaderboard(args.out)
    if args.arm:
        candidates = []
        for checkpoint in args.checkpoint:
            # Content digest makes candidate identities unambiguous across names.
            name = f"{args.arm}_{checkpoint.stem}_{sha256(checkpoint)[:10]}"
            candidates.append(score(args.arm, checkpoint, "validation", name, args.out, protocol, args.device))
        if sorted(r["step"] for r in candidates) != protocol["candidate_updates"]:
            raise RuntimeError("Candidate updates differ from frozen control protocol")
        selected = max(candidates, key=selection_key)
        best_path = args.final.parent/"best.pt"
        if best_path.exists() and sha256(best_path) != selected["checkpoint_sha256"]:
            raise RuntimeError("Refusing to overwrite different previously selected best checkpoint")
        if not best_path.exists():
            shutil.copyfile(selected["checkpoint"], best_path)
        # Persist selection before any test evaluation.
        selection = dict(arm=args.arm, validation=selected,
            candidates=[dict(name=r["name"], checkpoint=r["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"],
                             step=r["step"], summary=r["summary"]) for r in candidates],
            selected_checkpoint=str(best_path.resolve()))
        selection_path = args.out/"selections"/f"{args.arm}.json"
        write_json(selection_path, selection)
        for role, checkpoint in [("selected", best_path), ("final", args.final)]:
            name = f"{args.arm}_{role}"
            if role == "final" and sha256(checkpoint) == sha256(best_path):
                row = json.loads(Path(selection["selected_test"]).read_text())
                row.update(name=name, checkpoint=str(checkpoint.resolve()), reused_identical_selected_test=True)
                write_json(args.out/"evaluations"/f"{name}_test.json", row)
            else:
                score(args.arm, checkpoint, "test", name, args.out, protocol, args.device)
            selection[role+"_test"] = str((args.out/"evaluations"/f"{name}_test.json").resolve())
        write_json(selection_path, selection)
        refresh_leaderboard(args.out)


if __name__ == "__main__":
    main()
