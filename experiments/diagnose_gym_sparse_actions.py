#!/usr/bin/env python
"""Posthoc heuristic disagreement on frozen sparse-controller states; no rollouts."""
import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.diagnose_gym_control_actions import action_metrics
from experiments.evaluate_gym_sparse_action import verify_protocol
from lib.gym_control_evaluation import engine_regimes
from lib.gym_data import make_env, phases, sha256


def lateral_metrics(actions, expert, mask):
    """Separate missed firing, extra firing, and wrong direction, with denominators."""
    learner, target = [engine_regimes(x[mask])[:, 1] for x in (actions, expert)]
    if not len(learner):
        return {}
    on, expert_on = learner != 0, target != 0
    missed, extra = ~on & expert_on, on & ~expert_on
    wrong = on & expert_on & (learner != target)
    result = dict(expert_lateral_on_fraction=float(expert_on.mean()),
                  learner_lateral_on_fraction=float(on.mean()))
    for name, values, denominator in (("missed", missed, expert_on),
                                       ("extra", extra, ~expert_on),
                                       ("wrong_direction", wrong, expert_on)):
        result[f"lateral_{name}_count"] = int(values.sum())
        result[f"lateral_{name}_fraction"] = float(values.mean())
        result[f"lateral_{name}_conditional_fraction"] = float(values.sum()/denominator.sum()) if denominator.any() else None
    return result


def diagnose(report_dir):
    from gymnasium.envs.box2d.lunar_lander import heuristic
    protocol = verify_protocol(report_dir / "protocol.json")
    scales = np.asarray(protocol["supervision"]["normalization_statistics"]["action_scale"], np.float32)
    output = dict(kind="Posthoc descriptive expert disagreement; no training or selection",
        protocol_sha256=sha256(report_dir / "protocol.json"),
        source_sha256=sha256(__file__), reused_metrics_source_sha256=sha256(inspect.getfile(action_metrics)),
        heuristic_source_sha256=sha256(inspect.getfile(heuristic)),
        action_scale=scales.tolist(), simulator_steps=0, simulator_resets=0,
        grouping="Current contacts first; otherwise approach y<0.25, otherwise flight. Later means zero-based step >=20. Outcome groups use eventual terminal outcome, retrospectively.",
        denominators="Missed/wrong-direction lateral conditional fractions divide by expert-on commands; extra divides by expert-off commands. Overall metrics weight transitions equally; episode means weight each episode equally.",
        limitations=["Controllers visit different states; this is not a matched-state causal comparison.",
                     "Heuristic recommendations are posthoc reference actions, not demonstrated recovery or optimal actions.",
                     "No labels from this diagnosis were used in training, preprocessing, or checkpoint selection."], controllers={})
    env = make_env()  # Heuristic only reads continuous flag: no reset, no step.
    try:
        for arm in ("probes", "auxiliary"):
            row_path = report_dir / "evaluations" / f"{arm}_selected_test.json"
            row = json.loads(row_path.read_text())
            if row["protocol_sha256"] != output["protocol_sha256"] or sha256(row["traces"]) != row["traces_sha256"]:
                raise RuntimeError("Trace/protocol provenance mismatch")
            with np.load(row["traces"], allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            if sorted(np.unique(arrays["seeds"]).tolist()) != protocol["test_seeds"]:
                raise ValueError("Unexpected test episode identities")
            states, actions = arrays["states"], arrays["actions"]
            expert = np.asarray([heuristic(env.unwrapped, state) for state in states], np.float32)
            phase, early = phases(states), arrays["steps"] < 20
            masks = dict(overall=np.ones(len(states), bool), first20=early, later=~early)
            for i, name in enumerate(("flight", "approach", "contact")):
                masks[name] = phase == i
                masks["later_" + name] = (phase == i) & ~early
            outcomes = {e["seed"]: e["outcome"] for e in row["episodes"]}
            for outcome in sorted(set(outcomes.values())):
                masks["outcome_" + outcome] = np.isin(arrays["seeds"], [s for s,o in outcomes.items() if o == outcome])
            def metrics(mask):
                return {**action_metrics(actions, expert, scales, mask), **lateral_metrics(actions, expert, mask)}
            per_episode = [dict(seed=seed, outcome=outcomes[seed], **metrics(arrays["seeds"] == seed))
                           for seed in protocol["test_seeds"]]
            output["controllers"][arm] = dict(evaluation=str(row_path.resolve()), evaluation_sha256=sha256(row_path),
                checkpoint_sha256=row["checkpoint_sha256"], traces=row["traces"], traces_sha256=row["traces_sha256"],
                control_summary=row["summary"], expert_diagnostics=row["expert_diagnostics"],
                groups={name: {**metrics(mask), "fraction_of_all_records": float(mask.mean())} for name,mask in masks.items()},
                per_episode=per_episode, episode_means={key: float(np.mean([r[key] for r in per_episode]))
                    for key in ("physical_action_mse", "standardized_action_mse", "main_regime_agreement", "lateral_regime_agreement")})
    finally:
        env.close()
    return output


def markdown(value):
    lines = ["# Sparse-controller action disagreement on measured learner states", "",
        "Posthoc heuristic commands on saved selected test traces; zero simulator resets/steps and no model fitting. Overall metrics weight transitions equally. Episode means and outcome groups are in the JSON.", "",
        "| Controller | Expert-state physical MSE | Learner-state physical MSE | Episode-mean physical MSE | Main agreement | Side agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm,row in value["controllers"].items():
        g = row["groups"]["overall"]
        lines.append(f"| {arm} | {row['expert_diagnostics']['physical_action_mse']:.5f} | {g['physical_action_mse']:.5f} | {row['episode_means']['physical_action_mse']:.5f} | {g['main_regime_agreement']:.1%} | {g['lateral_regime_agreement']:.1%} |")
    lines += ["", "| Controller / group | Records | Physical MSE | Side missed / expert on | Side extra / expert off | Side wrong direction / expert on |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def pct(x):
        return "—" if x is None else f"{x:.1%}"
    for arm,row in value["controllers"].items():
        for group in ("overall", "first20", "later_flight", "approach", "contact", "outcome_out_of_bounds"):
            g = row["groups"].get(group, {})
            if not g.get("records"):
                continue
            lines.append(f"| {arm} / {group} | {g['records']} | {g['physical_action_mse']:.5f} | {pct(g['lateral_missed_conditional_fraction'])} | {pct(g['lateral_extra_conditional_fraction'])} | {pct(g['lateral_wrong_direction_conditional_fraction'])} |")
    lines += ["", value["grouping"], "", *value["limitations"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports/gym/lunar_lander_sparse_action")
    args = parser.parse_args()
    path = args.reports / "on_policy_action_diagnostics.json"
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError("Use fresh diagnostic output")
    value = diagnose(args.reports)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    path.with_suffix(".md").write_text(markdown(value))
    print(markdown(value), flush=True)


if __name__ == "__main__":
    main()
