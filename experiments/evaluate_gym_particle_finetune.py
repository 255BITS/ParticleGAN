#!/usr/bin/env python
"""Score the default YuE2 particle finetune on the shared Lunar control protocol.

Training does not select a checkpoint, and diag_action_mse is not a landing
rate. This script freezes the same protocol as experiments/evaluate_gym_control.py
(validation seeds 391000-391019, test seeds 491000-491049) and selects by
validation landing rate, then mean return, then the earlier update.

It loads gym_particle_finetune_v1 checkpoints only. adv_weight must be 1.
The cited board (val 20/20, test 50/50, mean return 287.7, step 2500) is
PR #18; this script recomputes landings from rollouts in --out.
"""
import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_control import control_action_details
from lib.gym_control_evaluation import (evaluate_controller, freeze_protocol,
    selection_key, verify_protocol)
from lib.gym_data import sha256
from lib.gym_particle_finetune import load_particle_checkpoint, require_live_adversary

REPORTS = ROOT / "reports/gym/lunar_lander_yue2"
EPISODES = ROOT / "results/gym/lunar_lander/data/episodes.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assert_yue2_checkpoint(bundle):
    """Reject anything other than the default paired-error controller step."""
    cfg = bundle["config"]
    require_live_adversary(cfg["adv_weight"])
    if cfg.get("arm") != "particle":
        raise ValueError("arm must be particle")
    if cfg.get("train_scope") != "control":
        raise ValueError("train_scope stays control")
    summary = bundle.get("training_summary")
    if not summary:
        raise RuntimeError("Particle checkpoint is missing summary.json")
    if summary.get("adv_weight") != 1.:
        raise ValueError("summary adv_weight must be 1")
    recipe = summary.get("recipe") or {}
    if recipe.get("critic") != "gmix_t8_w48_l1" or recipe.get("gan_mode") != "rp":
        raise ValueError("checkpoint recipe is not YuE2 paired-error RpGAN")
    if recipe.get("adv_weight") != 1. or recipe.get("l2_aux_weight") != 0.:
        raise ValueError("checkpoint recipe left the adversarial controller step")
    if summary.get("provenance") != bundle.get("provenance"):
        raise RuntimeError("Checkpoint and training summary provenance differ")
    return bundle


def verify_checkpoint(checkpoint, device):
    bundle = load_particle_checkpoint(checkpoint, device)
    assert_yue2_checkpoint(bundle)
    if sha256(checkpoint) not in bundle["training_summary"]["checkpoints"].values():
        raise RuntimeError("Particle checkpoint is not a hashed completed training output")
    return bundle


def score(checkpoint, split, name, out, protocol, device):
    destination = out / "evaluations" / f"{name}_{split}.json"
    digest = sha256(checkpoint)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["checkpoint_sha256"] != digest or old["protocol_sha256"] != sha256(out / "protocol.json"):
            raise RuntimeError(f"Existing score provenance differs: {destination}")
        return old
    bundle = verify_checkpoint(checkpoint, device)
    if (bundle["config"]["steps"] != protocol["training_updates"]
            or bundle["config"]["batch_size"] != protocol["training_batch_size"]):
        raise RuntimeError("Training budget differs from the frozen protocol")
    action = lambda env, s, a, c: control_action_details(bundle, s, a, c)
    row = evaluate_controller(action, protocol[split + "_seeds"], out / "traces" / f"{name}_{split}.npz",
                              "particle/" + split)
    row.update(name=name, arm="particle", split=split, checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=digest, protocol_sha256=sha256(out / "protocol.json"), device=device,
        step=int(bundle["step"]), adv_weight=1., recipe="yue2_paired_error_rpgan")
    write_json(destination, row)
    logging.info("SCORE %s %s landing=%d/%d mean_return=%.3f", name, split,
                 row["summary"]["landing_count"], row["summary"]["episodes"], row["summary"]["mean_return"])
    return row


def write_leaderboard(out, rows):
    rows = sorted(rows, key=lambda r: (r["summary"]["landing_rate"], r["summary"]["mean_return"]), reverse=True)
    write_json(out / "leaderboard.json", dict(protocol_sha256=sha256(out / "protocol.json"), rows=rows,
        recipe="yue2_paired_error_rpgan", adv_weight=1.))
    lines = ["# Lunar Lander YuE2 particle finetune", "",
        "Validation landing rate, then mean return, selects the checkpoint. Test rows are held out. "
        "The controller step is paired-error RpGAN at adv_weight 1. "
        "These rows are from this directory's rollouts.", "",
        "| Role | Update | Landings | Mean return | Median return |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        s = row["summary"]
        lines.append(
            f"| {row['report_role']} | {row['step']} | {s['landing_count']}/{s['episodes']} | "
            f"{s['mean_return']:.2f} | {s['median_return']:.2f} |")
    lines.append("")
    (out / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORTS)
    parser.add_argument("--episodes", type=Path, default=EPISODES)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--final", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, default=ROOT / "results/gym/lunar_lander_particle_finetune/live.log")
    args = parser.parse_args()
    if not (args.freeze or args.checkpoint):
        parser.error("Choose --freeze and/or --checkpoint")
    if args.checkpoint and not args.final:
        parser.error("--checkpoint requires --final")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log)])
    torch.set_num_threads(1)
    if args.freeze:
        freeze_protocol(args.out / "protocol.json", args.episodes)
        logging.info("FROZEN %s", args.out / "protocol.json")
    if not args.checkpoint:
        return
    protocol_path = args.out / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Freeze the evaluation protocol before scoring: {protocol_path}")
    protocol = verify_protocol(protocol_path)
    candidates = []
    for checkpoint in args.checkpoint:
        name = f"particle_{checkpoint.stem}_{sha256(checkpoint)[:10]}"
        candidates.append(score(checkpoint, "validation", name, args.out, protocol, args.device))
    if sorted(r["step"] for r in candidates) != protocol["candidate_updates"]:
        raise RuntimeError("Candidate updates differ from the frozen protocol")
    selected = max(candidates, key=selection_key)
    best_path = args.final.parent / "best.pt"
    if best_path.exists() and sha256(best_path) != selected["checkpoint_sha256"]:
        raise RuntimeError("Refusing to overwrite a different previously selected best checkpoint")
    if not best_path.exists():
        shutil.copyfile(selected["checkpoint"], best_path)
    selected_test = score(best_path, "test", "particle_selected", args.out, protocol, args.device)
    if sha256(args.final) == sha256(best_path):
        final_test = {**selected_test, "name": "particle_final", "reused_identical_selected_test": True}
        write_json(args.out / "evaluations" / "particle_final_test.json", final_test)
    else:
        final_test = score(args.final, "test", "particle_final", args.out, protocol, args.device)
    rows = [{**selected_test, "report_role": "selected"}, {**final_test, "report_role": "final"}]
    selection = dict(arm="particle", recipe="yue2_paired_error_rpgan", adv_weight=1., validation=selected,
        selected_checkpoint=str(best_path.resolve()),
        candidates=[dict(name=r["name"], checkpoint=r["checkpoint"], step=r["step"], summary=r["summary"])
                    for r in candidates],
        selected_test=str((args.out / "evaluations" / "particle_selected_test.json").resolve()),
        final_test=str((args.out / "evaluations" / "particle_final_test.json").resolve()))
    write_json(args.out / "selections" / "particle.json", selection)
    write_leaderboard(args.out, rows)


if __name__ == "__main__":
    main()
