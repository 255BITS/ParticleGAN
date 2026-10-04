"""Read retained, certified force tensors; never run or modify training."""
import argparse
import importlib.util
from pathlib import Path
import statistics

import torch

REPORT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("horizon_publication", REPORT / "publish.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)
from benchmarks.locked_shared.two_pole import HostCritic, real_batch


def vector(value):
    return value.detach().double().flatten()


def cosine(left, right):
    left, right = vector(left), vector(right)
    denominator = float(left.norm() * right.norm())
    return float(left.dot(right) / denominator) if denominator else None


def balance(left, right):
    left, right = vector(left), vector(right)
    a, b = float(left.norm()), float(right.norm())
    return {"left_norm": a, "right_norm": b,
            "cosine": cosine(left, right),
            "right_to_left_norm_ratio": b / a if a else None,
            "cancellation_ratio": float((left + right).norm()) / (a + b) if a + b else None}


def point_summary(point):
    g = balance(point["adversarial_gradient"], point["particle_l2_gradient"])
    d = balance(point["critic_payoff_parameter_gradient"], point["critic_penalty_parameter_gradient"])
    return {"step": point["step"], "mean_abs": point["mean_abs"],
            "particle_std": point["particle_std"],
            "positions_min_max": [point["minimum_position"], point["maximum_position"]],
            "particle_sign_counts": {"negative": point["negative_particles"], "zero": point["zero_particles"],
                                     "positive": point["positive_particles"]},
            "nearest_pole_distance": point["nearest_pole_distance"],
            "adversarial_vs_particle_l2": g, "critic_payoff_vs_penalty": d,
            "mean_signed_adversarial_force": point["mean_signed_adversarial_force"],
            "mean_signed_l2_force": point["mean_signed_l2_force"],
            "total_gradient_norm": point["total_gradient_norm"],
            "displacement_norm": point["displacement_norm"],
            "descent_vs_displacement_cosine": cosine(-point["total_gradient"], point["displacement"]),
            "actual_particle_lr": point["actual_generator_lr"], "critic_lr": point["critic_lr"],
            "direct_gain": point["direct_gain"], "input_noise_std": point["input_noise_std"],
            "clean_gradient_median": point["critic_diagnostics"]["clean_gradient_median"],
            "existing_penalty": point["penalty"]}


def window_summary(trace, start, stop):
    points = trace[start - 1:stop]
    g_balances = [balance(p["adversarial_gradient"], p["particle_l2_gradient"]) for p in points]
    d_balances = [balance(p["critic_payoff_parameter_gradient"], p["critic_penalty_parameter_gradient"]) for p in points]
    def median(records, key):
        values = [p[key] for p in records if p[key] is not None]
        return statistics.median(values) if values else None
    inward = sum(int((p["adversarial_gradient"] * p["positions_before"] > 0).sum()) for p in points)
    outward = sum(int((p["adversarial_gradient"] * p["positions_before"] < 0).sum()) for p in points)
    opposite = sum(int((p["displacement"] * p["total_gradient"] > 0).sum()) for p in points)
    return {"updates": [start, stop],
            "mean_abs_change": points[-1]["mean_abs"] - float(points[0]["positions_before"].abs().mean()),
            "mean_abs_range": [min(p["mean_abs"] for p in points), max(p["mean_abs"] for p in points)],
            "median_adversarial_l2_cancellation": median(g_balances, "cancellation_ratio"),
            "median_adversarial_l2_cosine": median(g_balances, "cosine"),
            "median_l2_to_adversarial_norm_ratio": median(g_balances, "right_to_left_norm_ratio"),
            "median_critic_payoff_penalty_cancellation": median(d_balances, "cancellation_ratio"),
            "median_critic_payoff_penalty_cosine": median(d_balances, "cosine"),
            "median_total_gradient_norm": statistics.median(p["total_gradient_norm"] for p in points),
            "median_displacement_norm": statistics.median(p["displacement_norm"] for p in points),
            "coordinate_path_length": float(sum((p["displacement"].abs().sum() for p in points))),
            "coordinate_net_displacement_l1": float((points[-1]["positions"] - points[0]["positions_before"]).abs().sum()),
            "adversarial_force_coordinate_events": {"inward": inward, "outward": outward,
                "zero_or_origin": 12 * len(points) - inward - outward},
            "optimizer_coordinate_updates_opposing_current_descent": opposite}


def clean_pairing_check(point, state):
    """Recompute only the deterministic loss derivative at a retained state."""
    if point["input_noise_std"] != 0:
        return {"checked": False, "reason": "Actual update used input noise; clean scores cannot reconstruct that draw."}
    # HostCritic's nn.Linear construction consumes RNG before loading weights.
    with torch.random.fork_rng(devices=[]):
        critic = HostCritic()
    critic.load_state_dict(state["critic"])
    x = point["positions_before"].detach().clone().requires_grad_(True)
    fake_scores = critic(x)
    real_scores = critic(real_batch(x.numel())).detach()
    derivative = torch.autograd.grad(fake_scores.sum(), x)[0]
    pairing = torch.sigmoid(real_scores - fake_scores).unsqueeze(1).detach()
    expected = -pairing * derivative / x.numel()
    actual = point["adversarial_gradient"]
    assert torch.allclose(expected, actual, atol=2e-9, rtol=2e-4)
    return {"checked": True, "maximum_absolute_gradient_error": float((expected - actual).abs().max()),
            "paired_sigmoid_min_max": [float(pairing.min()), float(pairing.max())],
            "clean_D_prime_at_preupdate_particles_min_max": [float(derivative.min()), float(derivative.max())],
            "scope": "Existing critic state after its update and pre-generator coordinates; no sample draws or optimizer updates."}


def analyze(arm, data, trace, checkpoints):
    durable, _, request, _, row, certificate = data
    guard_updates = sum(not torch.equal(p["critic_total_parameter_gradient"], p["critic_applied_parameter_gradient"]) for p in trace)
    cumulative = checkpoints["checkpoints"][800]["critic_optimizer"]["regularizer"]["guard"]["clipped_tensors"]
    audit = row["evidence"]["guards"]["mechanism_audit"]["mechanisms"]["critic_guard"]
    assert guard_updates == audit["applied"] == cumulative == 0
    first_travel = next((p["step"] for p in trace if p["mean_abs"] >= .3), None)
    peak = max(trace, key=lambda p: p["mean_abs"])
    milestones = [point_summary(trace[step - 1]) for step in (80, 200, 400, 800)]
    for milestone in milestones:
        step = milestone["step"]
        milestone["paired_loss_derivative_validation"] = clean_pairing_check(trace[step - 1], checkpoints["checkpoints"][step])
    descriptor = row["evidence"]["horizon_diagnostic"]
    result = {"arm_id": arm["id"], "recipe_label": arm["recipe_label"], "schedule_horizon": arm["schedule_horizon"],
              "attempt_id": durable.name, "candidate_id": arm["candidate_id"],
              "candidate_revision": arm["candidate_revision"], "source_digest": request["source"]["digest"],
              "source_commit": request["source"]["origin_commit"], "original_result_stable_hash": certificate["result_hash"],
              "applied_numeric_controls": {k: row["applied"]["recipe"][k] for k in (
                  "lr", "d_lr_mult", "prior_lr_mult", "reg_coeff", "reg_kappa", "reg_anchor_weight",
                  "input_noise_std", "output_noise_std", "total_steps", "direct_particle_betas", "direct_particle_gain")},
              "early_real_squared_gradient_prefactor": row["applied"]["recipe"]["reg_coeff"] / 2.,
              "trace_artifacts": descriptor["artifacts"], "diagnostic_gate_status": row["gate_status"],
              "final_metrics": row["metrics"], "stable_terminal_gate_result": row["evaluator_result"],
              "first_mean_abs_threshold_crossing": first_travel,
              "mean_abs_peak": {"step": peak["step"], "value": peak["mean_abs"]},
              "maximum_particle_std": max(p["particle_std"] for p in trace),
              "milestones": milestones,
              "windows": [window_summary(trace, a, b) for a, b in ((1,80),(81,200),(201,400),(401,800),(701,800))],
              "critic_guard": {"eligible_updates": audit["eligible"], "clipped_update_count": guard_updates,
                               "clipped_tensor_count": cumulative, "pre_and_post_gradient_tensors_identical": True},
              "initial_state_sha256": publication.state_hash(checkpoints["initial"])}
    if arm["schedule_horizon"] == 80:
        original = publication.historical_prefix(arm["recipe_label"], {"effective_recipe": row["applied"]["recipe"],
            "prefix80": {"observation_sha256": publication.stable_hash([p for p in row["evidence"]["diagnostic_observations"] if p["step"] <= 80])}})
        result["historical_prefix_control"] = {k: v for k,v in original.items() if k != "reference_artifacts"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=REPORT / "plans.json")
    parser.add_argument("--queue-root", type=Path)
    args = parser.parse_args()
    plan = publication.read(args.plans)
    state = publication.Queue(args.queue_root or publication.ROOT / "runs/forge" / plan["study"] / "queue", on_completion=None).inspect()
    assert len(plan["arms"]) == len(state["jobs"]) == 4
    sources, results = set(), []
    for arm in plan["arms"]:
        data = publication.certified(arm, state, plan, sources)
        _, trace, checkpoints = publication.saved_trace(data[1], data[4], plan)
        results.append(analyze(arm, data, trace, checkpoints))
    paired = {}
    for result in results:
        paired.setdefault(result["recipe_label"], []).append(result["initial_state_sha256"])
    assert all(len(set(hashes)) == 1 for hashes in paired.values())
    output = {"schema_version": 1, "study": plan["study"], "scope": "read_only_existing_training_force_analysis",
        "source_digest": plan["source_digest"], "analysis_source": publication.identity(Path(__file__)),
        "validation_source": publication.identity(REPORT / "publish.py"),
        "kernel_sources": [publication.identity(publication.ROOT / name) for name in (
            "particlegan/gan_loss.py", "particlegan/grad_regularizers.py", "particlegan/k3p.py",
            "benchmarks/locked_shared/two_pole.py", "experiments/forge/two_pole_observer.py")],
        "plans": publication.identity(args.plans), "arms": results,
        "definitions": {
            "force": "Negative loss gradient; scalar signs describe coordinate direction, not raw critic derivative.",
            "cancellation_ratio": "norm(left+right)/(norm(left)+norm(right)); near zero means opposing vectors nearly cancel.",
            "particle_l2_gradient": ".04*x/12, from the unchanged .02*mean(x^2) host term.",
            "clean_RpGAN_adversarial_gradient": "-sigmoid(D(real_i)-D(x_i))*D_prime(x_i)/12; real score is detached.",
            "early_real_squared_gradient_prefactor": "reg_coeff/(2*sample_dimension); sample_dimension=1 here. Later blend/anchor terms differ and are recorded separately.",
            "first_mean_abs_threshold_crossing": "Trace-only travel threshold; does not replace the certified stable terminal grade."},
        "findings": [
            "Neither word-positive arm reaches mean_abs .3 during 800 updates. Schedule80 keeps moving slowly; schedule800 reaches a small-radius adversarial/L2 balance and its radius later decreases.",
            "The word schedule800 adversarial and restoring L2 vectors nearly cancel at retained milestones200/400/800. Critic payoff and penalty gradients also oppose; this is measured vector cancellation, not evidence of a broken update binding.",
            "The 1D host's early squared-real-gradient prefactor is85 for the word recipe versus .5 for the movement control. The recorded small slopes and penalty/payoff cancellation are consistent with a strongly flattened word critic; the multi-control recipe contrast cannot establish coefficient-only causality.",
            "All four arms have zero critic guard clipping despite 600 eligible updates. The guard cannot explain the measured word plateau in these runs.",
            "The word recipe remains one-sided, with a small nonzero coordinate spread. Rowwise relativistic pairing can produce unequal gradients even from identical clean zero coordinates; noise is not its only possible symmetry source.",
            "Both movement-control arms retain six negative and six positive particles at the end and pass their complete declared terminal checks. Diagnostic movement and sign balance do not confer ordinary Tier1 or word-task qualification."],
        "recommendations": [
            "Do not spend on another unchanged word-recipe continuation to claim a global winner: both declared 800-update contrasts fail and one shows near force balance far inside the .3 gate.",
            "Keep task gates and global-recipe scope fixed. A future bounded existing-parameter hypothesis should target critic shape/penalty balance, with a prediction of increased outward adversarial force at the same radius; changing optimizer rates alone cannot change a fixed zero-gradient equilibrium.",
            "These two recipes differ in several existing controls, so their contrast does not isolate coefficient, input noise, or rate as the causal factor. Consult prior whole-family searches before selecting another finite grid."],
        "limits": [
            "Only retained single-seed declared arms are analyzed; no seed study, fresh training, added sample draws or modified gate.",
            "Optimizer normalization can produce finite steps near force cancellation; small instantaneous gradients alone do not prove convergence. Window drift and path length are provided separately.",
            "Schedule horizon changes LR/noise durations and LR-dependent penalty/anchor timing together; it is not an isolated LR intervention.",
            "Both fixed80 prefixes reproduce all24 original observations and effective recipes exactly. Original critic/optimizer checkpoints were not retained, so historical full-state identity cannot be proved."],
        "qualification_input": False, "optimizer_updates_added": 0, "sampling_draws_added": 0}
    destination = REPORT / "force-analysis.json"
    publication.write(destination, output)
    print(publication.identity(destination))


if __name__ == "__main__":
    main()
