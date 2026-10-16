"""Write the horizon-clock arm idea cards (reproduction source) and print declared clock deps."""
import json
from pathlib import Path

from experiments.forge.api import resolve_public_recipe
from experiments.forge.clockfree import source_audit

base = json.loads(Path("configs/forge/ideas/k3p-hc-progress-v1.json").read_text())
ARMS = {
    "k3p-hc-progress-v1": dict(
        ov={"progress_schedule": "critic_order", "progress_rate": 0.001, "progress_target": 0.75,
            "progress_ema": 0.98, "d_guard_min_steps": 0},
        cf=["K3P's cosine LR, input-noise anneal and output-noise warmup are driven by an internal progress "
            "value p (critic paired-order accuracy gate, at most 0.001 per update, monotone) instead of "
            "completed_steps/total_steps",
            "Critic spike guard active from the first update (d_guard_min_steps=0) to remove the fixed-count release"],
        mr="Horizon-free annealing: progress advances only while the critic cannot order real above generated "
           "samples better than 75%, so schedules follow training state rather than the step label or declared horizon."),
    "k3p-hc-progress-ungated-v1": dict(
        ov={"progress_schedule": "updates", "progress_rate": 0.001, "d_guard_min_steps": 0},
        cf=["Diagnostic: K3P schedules driven by an internal progress counter advancing 0.001 every update (no critic gate)",
            "Critic spike guard active from the first update (d_guard_min_steps=0)"],
        mr="Diagnostic control separating annealing within the 1,600-update cap from progress gating; internal "
           "counter, no step label or horizon dependence."),
    "k3p-hc-constant-v1": dict(
        ov={"lr_floor": 1.0, "network_lr_floor": 1.0, "network_lr_horizon_cap": None, "input_noise_std": 0.0,
            "output_noise_warmup": 0.0, "d_guard_min_steps": 0},
        cf=["Constant K3P learning rates (no cosine anneal), no input-noise anneal, constant output noise, "
            "spike guard from the first update"],
        mr="Constant-rate clock-free K3P control with no declared clock dependency."),
}
for cid, arm in ARMS.items():
    card = dict(base, id=cid, changed_factors=arm["cf"], mechanism_rationale=arm["mr"],
                recipe_overrides={"critic_formulation": "k3p", "name": "k3p", **arm["ov"]})
    Path(f"configs/forge/ideas/{cid}.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
for cid in ["k3p", *ARMS]:
    card = json.loads(Path(f"configs/forge/ideas/{cid}.json").read_text())
    print(cid, source_audit(resolve_public_recipe(card).to_dict(), {})["unexplained_clock_dependencies"])
