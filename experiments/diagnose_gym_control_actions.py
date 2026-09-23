#!/usr/bin/env python
"""Posthoc expert labels on frozen learner traces; no simulator rollouts or fitting."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_control_evaluation import engine_regimes, verify_protocol
from lib.gym_data import make_env, phases, sha256, simulator_provenance


def action_metrics(actions, expert, scale, mask):
    actions, expert = actions[mask], expert[mask]
    if not len(actions):
        return {"records": 0}
    learner_regime, expert_regime = engine_regimes(actions), engine_regimes(expert)
    same = learner_regime == expert_regime
    learner_on, expert_on = learner_regime[:,0].astype(bool), expert_regime[:,0].astype(bool)
    false_off, false_on = ~learner_on & expert_on, learner_on & ~expert_on
    return dict(records=len(actions), standardized_action_mse=float(np.mean(((actions-expert)/scale)**2)),
        physical_action_mse=float(np.mean((actions-expert)**2)),
        main_standardized_mse=float(np.mean(((actions[:,0]-expert[:,0])/scale[0])**2)),
        lateral_standardized_mse=float(np.mean(((actions[:,1]-expert[:,1])/scale[1])**2)),
        main_regime_agreement=float(same[:,0].mean()), lateral_regime_agreement=float(same[:,1].mean()),
        joint_regime_agreement=float(same.all(1).mean()),
        expert_main_on_fraction=float(expert_on.mean()), learner_main_on_fraction=float(learner_on.mean()),
        main_false_off_count=int(false_off.sum()), main_false_on_count=int(false_on.sum()),
        main_false_off_fraction=float(false_off.mean()), main_false_on_fraction=float(false_on.mean()),
        main_false_off_given_expert_on=float(false_off.sum()/expert_on.sum()) if expert_on.any() else None,
        main_false_on_given_expert_off=float(false_on.sum()/(~expert_on).sum()) if (~expert_on).any() else None,
        mean_main_command_error=float(np.mean(actions[:,0]-expert[:,0])),
        mean_abs_main_command_error=float(np.mean(np.abs(actions[:,0]-expert[:,0]))),
        mean_abs_lateral_command_error=float(np.mean(np.abs(actions[:,1]-expert[:,1]))))


def diagnose(report_dir, normalization):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    protocol = verify_protocol(report_dir/"protocol.json")
    scales = np.asarray(json.loads(normalization.read_text())["action_scale"], np.float32)
    env = make_env()  # Heuristic only reads env.unwrapped.continuous. No reset/step.
    output = dict(kind="posthoc descriptive diagnostics, no checkpoint selection or training",
        protocol_sha256=sha256(report_dir/"protocol.json"), simulator=simulator_provenance(),
        source_sha256=sha256(__file__), heuristic_source_sha256=sha256(inspect.getfile(heuristic)),
        normalization=dict(path=str(normalization.resolve()), sha256=sha256(normalization), action_scale=scales.tolist()),
        simulator_resets=0, simulator_steps=0, new_training_records_used=0,
        phase_definition="Current observed contact if either leg contacts; otherwise approach y<0.25; otherwise flight",
        early_definition="First 20 commands of each episode, zero-based trace step <20",
        weights="Overall and phase metrics weight saved transitions equally; episode_mean metrics weight episodes equally",
        limitations=["Controllers visit different states, so between-controller on-policy errors are descriptive, not a matched-state causal comparison.",
            "Heuristic labels are actions this expert would take on the recorded state, not proof of the optimal recovery action.",
            "Expert was not rolled out from learner states; these labels do not establish whether it would recover.",
            "DAgger is a proposed future comparison; no labels here were added to training."], controllers={})
    try:
        for arm,name in [("original", "original"), ("imitation", "imitation_selected"), ("joint", "joint_selected")]:
            row_path = report_dir/"evaluations"/f"{name}_test.json"
            row = json.loads(row_path.read_text())
            if row["protocol_sha256"] != output["protocol_sha256"] or sha256(row["traces"]) != row["traces_sha256"]:
                raise RuntimeError("Frozen trace/protocol provenance mismatch")
            with np.load(row["traces"], allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            if sorted(np.unique(arrays["seeds"]).tolist()) != protocol["test_seeds"]:
                raise ValueError("Unexpected test episode identities")
            states, actions = arrays["states"], arrays["actions"]
            expert = np.asarray([heuristic(env.unwrapped, state) for state in states], np.float32)
            phase, early = phases(states), arrays["steps"] < 20
            masks = dict(overall=np.ones(len(states), bool), first20=early, later=~early,
                flight=phase == 0, approach=phase == 1, contact=phase == 2)
            for timing, time_mask in [("first20", early), ("later", ~early)]:
                for number,pname in enumerate(("flight", "approach", "contact")):
                    masks[timing+"_"+pname] = time_mask & (phase == number)
            groups = {name: {**action_metrics(actions, expert, scales, mask),
                             "fraction_of_all_records": float(mask.mean())} for name,mask in masks.items()}
            per_episode = []
            outcome = {e["seed"]: e["outcome"] for e in row["episodes"]}
            for seed in protocol["test_seeds"]:
                mask = arrays["seeds"] == seed
                per_episode.append(dict(seed=seed, outcome=outcome[seed],
                    **action_metrics(actions, expert, scales, mask)))
            output["controllers"][arm] = dict(evaluation=str(row_path.resolve()), evaluation_sha256=sha256(row_path),
                traces=row["traces"], traces_sha256=row["traces_sha256"],
                checkpoint_sha256=row["checkpoint_sha256"], step=row["step"],
                control_summary=row["summary"], demonstration_action_diagnostics=row["expert_action_diagnostics"],
                groups=groups, per_episode=per_episode,
                episode_mean_standardized_mse=float(np.mean([e["standardized_action_mse"] for e in per_episode])),
                episode_mean_joint_regime_agreement=float(np.mean([e["joint_regime_agreement"] for e in per_episode])))
    finally:
        env.close()
    return output


def markdown(value):
    rows = value["controllers"]
    lines = ["# Expert action disagreement on learner-visited states", "",
        "Posthoc labels on the frozen selected test traces; no new rollouts, resets, training, or checkpoint selection. "
        "The installed heuristic was queried using each recorded state. Overall metrics weight transitions equally; "
        "episode-weighted summaries and detailed counts are in the JSON.", "",
        "| Controller | Landings | Demo MSE | On-policy MSE | First 20 MSE | Later MSE | Joint engine-regime agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm,row in rows.items():
        g = row["groups"]
        lines.append(f"| {arm} | {row['control_summary']['landing_count']}/50 | {row['demonstration_action_diagnostics']['standardized_action_mse']:.5f} | {g['overall']['standardized_action_mse']:.5f} | {g['first20']['standardized_action_mse']:.5f} | {g['later']['standardized_action_mse']:.5f} | {g['overall']['joint_regime_agreement']:.1%} |")
    lines += ["", "MSE uses the unchanged training action scaler. Demo MSE uses held-out expert states and expert previous commands; on-policy MSE uses learner states and learner previous commands.", "",
        "| Controller / phase | Occupancy | Action MSE | Main false-off given expert-on | Main false-on given expert-off | Lateral regime agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def pct(v):
        return "—" if v is None else f"{v:.1%}"
    for arm,row in rows.items():
        for phase in ("flight", "approach", "contact"):
            g = row["groups"][phase]
            if not g["records"]:
                continue
            lines.append(f"| {arm} / {phase} | {g['fraction_of_all_records']:.1%} | {g['standardized_action_mse']:.5f} | {pct(g['main_false_off_given_expert_on'])} | {pct(g['main_false_on_given_expert_off'])} | {g['lateral_regime_agreement']:.1%} |")
    lines += ["", "Engine regimes follow the simulator dead zones: main on iff command >0; lateral off iff absolute command ≤0.5, otherwise signed direction. Phase uses the current state: either leg contact first, then approach y<0.25, otherwise flight.", "",
        "These measurements show how each controller disagrees with this expert on states it actually visits. Different state distributions and episode lengths prevent a causal attribution from this comparison alone. The heuristic was not tested for recovery from these states. A later DAgger comparison can test whether labeling learner-visited states improves control; this analysis does not establish that it will.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT/"reports/gym/lunar_lander_control")
    parser.add_argument("--normalization", type=Path, default=ROOT/"results/gym/lunar_lander/adversarial/normalization.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.reports/"on_policy_action_diagnostics.json"
    if out.exists() or out.with_suffix(".md").exists():
        raise FileExistsError("Choose fresh diagnostic output")
    value = diagnose(args.reports, args.normalization)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    out.with_suffix(".md").write_text(markdown(value))
    print(out.with_suffix(".md").read_text(), flush=True)


if __name__ == "__main__":
    main()
