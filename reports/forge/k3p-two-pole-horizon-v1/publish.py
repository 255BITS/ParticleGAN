"""Publish four certified horizon diagnostics from actual retained training data."""
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import file_hash, stable_hash
from experiments.forge.decision_contracts import evaluate
from experiments.forge.queue import Queue


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def state_hash(value):
    """Hash tensor bytes and structure without serializing states into reports."""
    import torch
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return stable_hash({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "bytes_sha256": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})
    if isinstance(value, dict):
        return stable_hash({str(k): state_hash(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return stable_hash([state_hash(v) for v in value])
    return stable_hash(value)


def certified(arm, state, plan, verified_sources):
    jobs = [j for j in state["jobs"].values() if j["definition"]["compatibility_key"] == arm["compatibility_key"]]
    assert len(jobs) == 1 and jobs[0]["status"] == "terminal", arm["id"]
    job, result = jobs[0], jobs[0]["result"]
    assert len(job["attempts"]) == 1 and len(result["task_results"]) == 1
    attempt = result["attempt_id"]
    durable = ROOT / "reports/forge/attempts" / attempt
    envelope, saved, certificate = [read(durable / name) for name in ("request.json", "result.json", "evidence.json")]
    request = envelope["request"]
    assert saved == result and certificate["result_hash"] == stable_hash(result)
    assert certificate["source"] == request["source"] and certificate["runtime"] == request["runtime"]
    assert request["campaign_id"] == plan["campaign_id"] and request["view"]["id"] == plan["view"]
    assert request["candidate"]["id"] == arm["candidate_id"]
    assert result["candidate_revision"] == request["candidate_revision"] == arm["candidate_revision"]
    assert request["source"]["digest"] == plan["source_digest"] == stable_hash(request["source"]["files"])
    assert stable_hash(request["protocol"]) == plan["protocol_sha256"]
    assert request["protocol"]["seed"] == plan["seed"] == 0
    assert request["runtime"] == plan["runtime"] and request["compute_profiles"] == plan["compute"]
    declaration = read(ROOT / arm["declaration"])
    assert all(request["candidate"][key] == value for key, value in declaration.items())
    local = Path(certificate["local_artifact_root"]).resolve()
    queue = Path(request["queue_root"]).resolve()
    assert local.is_relative_to(queue) and queue.is_relative_to(ROOT / "runs/forge")
    manifest = request["source"]
    source_key = (str(queue), manifest["digest"])
    if source_key not in verified_sources:
        snapshot = queue / "snapshots" / manifest["digest"]
        for name, expected in manifest["files"].items():
            assert file_hash(snapshot / name) == expected and file_hash(ROOT / name) == expected, name
        verified_sources.add(source_key)
    raw, grading = [read(local / name) for name in ("raw-result.json", "graded-result.json")]
    assert grading["raw_hash"] == stable_hash(raw) and grading["source_digest"] == manifest["digest"]
    assert result["raw"]["grading"] == grading
    assert (durable / "request.json").read_bytes() == (local / "request.json").read_bytes()
    assert (durable / "result.json").read_bytes() == (local / "result.json").read_bytes()
    task = request["tasks"][arm["task_id"]]
    row = result["task_results"][0]
    assert row["task_id"] == arm["task_id"] and row["compatibility_key"] == arm["compatibility_key"]
    assert row["gate_status"] in {"PASS", "FAIL"} and row["raw_status"] == "completed"
    assert task["execution"]["steps"] == plan["execution_updates_per_arm"] == 800
    assert task["execution"]["original_schedule_horizon"] == arm["schedule_horizon"]
    assert task["evaluation"]["thresholds"] == [["mean_abs", ">=", .3], ["grad_med", "<=", 1.0]]
    assert row["evidence"]["guards"]["all_finite"] and row["evidence"]["guards"]["unintended_rng_deviations"] == 0
    assert row["evidence"]["guards"]["optimizer_updates"] == {"prior": 800, "discriminator": 800}
    context = task_formulation_context(request["candidate"], task, request["protocol"], root=ROOT)
    applied = row["applied"]
    assert applied["recipe"] == json.loads(json.dumps(asdict(context.recipe)))
    assert applied["prior"] == context.prior_config and applied["initializer"] == context.initializer
    assert applied["recipe"]["total_steps"] == arm["schedule_horizon"]
    assert applied["recipe"]["direct_particle_betas"] == [0.0, .9]
    assert row["evidence"]["sampling_law"] == "learned_particles_and_critic_gradient"
    assert row["evidence"]["eval_output_noise"] == "not_applied_to_measurement"
    assert len(row["evidence"]["observations"]) == 24
    assert [p["step"] for p in row["evidence"]["observations"]] == [math.ceil(i * 800 / 24) for i in range(1, 25)]
    assert row["evaluator_result"]["convergence"]["complete"] is True
    return durable, local, request, task, row, certificate


def saved_trace(local, row, plan):
    import torch
    descriptor = row["evidence"]["horizon_diagnostic"]
    assert descriptor["execution_updates"] == 800 and descriptor["schedule_horizon"] in (80, 800)
    assert descriptor["known_particle_l2_coefficient"] == .02
    assert descriptor["optimizer_updates_added"] == descriptor["sampling_draws_added"] == 0
    artifacts = descriptor["artifacts"]
    assert {a["path"] for a in artifacts} == {"force-trace.pt", "force-trace.jsonl", "diagnostic-checkpoints.pt"}
    for artifact in artifacts:
        path = local / artifact["path"]
        assert path.resolve().is_relative_to(local) and file_hash(path) == artifact["sha256"]
        assert path.stat().st_size == artifact["bytes"]
    trace = torch.load(local / "force-trace.pt", map_location="cpu", weights_only=True)
    checkpoints = torch.load(local / "diagnostic-checkpoints.pt", map_location="cpu", weights_only=True)
    scalars = [json.loads(line) for line in (local / "force-trace.jsonl").read_text().splitlines()]
    assert len(trace) == len(scalars) == 800 and [p["step"] for p in trace] == list(range(1, 801))
    assert sorted(checkpoints["checkpoints"]) == plan["milestones"] == [80, 200, 400, 800]
    previous = checkpoints["initial"]["positions"]
    assert previous.shape == (12, 1) and torch.count_nonzero(previous) == 0
    for point, scalar in zip(trace, scalars):
        assert scalar == {k: v for k, v in point.items() if not isinstance(v, torch.Tensor) and k != "critic_diagnostics"}
        assert point["positions"].shape == (12, 1) and torch.equal(previous, point["positions_before"])
        for value in point.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
        assert torch.equal(point["positions"] - previous, point["displacement"])
        assert torch.allclose(point["particle_l2_gradient"], .04 * previous / previous.numel(), atol=1e-10, rtol=1e-6)
        assert torch.allclose(point["adversarial_gradient"], point["total_gradient"] - point["particle_l2_gradient"], atol=1e-10, rtol=1e-6)
        assert point["actual_generator_betas"] == [0.0, .9] and 1 <= point["direct_gain"] <= 2
        assert point["actual_generator_lr"] == point["scheduled_generator_lr"] * point["direct_gain"]
        assert point["mean_abs"] == float(point["positions"].abs().mean())
        assert point["output_noise_std"] == 0
        assert torch.equal(point["critic_payoff_parameter_gradient"],
                           point["critic_total_parameter_gradient"] - point["critic_penalty_parameter_gradient"])
        if point["step"] in checkpoints["checkpoints"]:
            assert torch.equal(point["positions"], checkpoints["checkpoints"][point["step"]]["positions"])
        previous = point["positions"]
    assert trace[-1]["mean_abs"] == row["metrics"]["mean_abs"]
    assert trace[-1]["critic_diagnostics"]["clean_gradient_median"] == row["metrics"]["grad_med"]
    diagnostics = row["evidence"]["diagnostic_observations"]
    expected = sorted({math.ceil(i * 80 / 24) for i in range(1, 25)} | set(plan["milestones"]))
    assert [p["step"] for p in diagnostics] == expected
    for p in diagnostics:
        assert p["mean_abs"] == trace[p["step"] - 1]["mean_abs"]
    return descriptor, trace, checkpoints


def phase_summary(trace, start, stop):
    points = trace[start - 1:stop]
    names = ("total_gradient_norm", "adversarial_gradient_norm", "l2_gradient_norm", "displacement_norm",
             "critic_total_gradient_norm", "critic_penalty_gradient_norm", "critic_payoff_gradient_norm",
             "critic_applied_gradient_norm", "critic_displacement_norm", "direct_gain")
    return {"first_update": start, "last_update": stop,
            "ranges": {k: {"first": points[0][k], "last": points[-1][k],
                            "minimum": min(p[k] for p in points), "maximum": max(p[k] for p in points),
                            "mean": sum(p[k] for p in points) / len(points)} for k in names},
            "scheduled_particle_lr_sum": sum(p["scheduled_generator_lr"] for p in points),
            "actual_particle_lr_sum": sum(p["actual_generator_lr"] for p in points),
            "critic_lr_sum": sum(p["critic_lr"] for p in points),
            "nonzero_input_noise_updates": sum(p["input_noise_std"] > 0 for p in points),
            "gain_above_one_updates": sum(p["direct_gain"] > 1 for p in points),
            "mean_signed_adversarial_force": sum(p["mean_signed_adversarial_force"] for p in points) / len(points),
            "mean_signed_l2_force": sum(p["mean_signed_l2_force"] for p in points) / len(points)}


def historical_prefix(label, current):
    """Bind original saved 80-update observations; never execute that control."""
    controls = {
        "word_positive": Path("/home/martyn/dev/ParticleGAN-family-wide/reports/forge/attempts/abb9e24f625d449da70f6641ec1bce0a"),
        "movement_control": Path("/home/martyn/dev/ParticleGAN-k3p-search-more/reports/forge/attempts/1e0e2fbd6d1840bd9271785ac1eaf2ca"),
    }
    durable = controls[label]
    envelope, result, certificate = [read(durable / name) for name in ("request.json", "result.json", "evidence.json")]
    assert certificate["result_hash"] == stable_hash(result)
    request = envelope["request"]
    assert certificate["source"] == request["source"] and certificate["runtime"] == request["runtime"]
    local = Path(certificate["local_artifact_root"])
    raw = read(local / "raw-result.json")
    grading = read(local / "graded-result.json")
    assert grading["raw_hash"] == stable_hash(raw) and grading["source_digest"] == request["source"]["digest"]
    assert result["raw"]["grading"] == grading
    original = raw["evidence"]["observations"]
    assert len(original) == 24 and [p["step"] for p in original] == [math.ceil(i * 80 / 24) for i in range(1,25)]
    original_recipe = raw["applied"]["recipe"]
    assert original_recipe == current["effective_recipe"]
    matching = stable_hash(original) == current["prefix80"]["observation_sha256"]
    assert matching, f"{label}: diagnostic80prefix differs from original saved80 control"
    artifacts = []
    for category, directory, names in (("durable",durable,("request.json","result.json","evidence.json")),
                                        ("raw",local,("raw-result.json","graded-result.json"))):
        for name in names:
            path = directory / name
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path),
                              "archive_member": f"references/{label}/{category}/{name}"})
    return {"recipe_label": label, "original_attempt_id": durable.name,
            "original_candidate_id": request["candidate"]["id"], "original_candidate_revision": request["candidate_revision"],
            "original_source_digest": request["source"]["digest"], "original_source_commit": request["source"]["origin_commit"],
            "original_result_stable_hash": certificate["result_hash"],
            "original_prefix_observations_sha256": stable_hash(original), "all24prefix_observations_exactly_match": matching,
            "final_metrics": original[-1], "raw_effective_recipe_exactly_matches": True,
            "reference_artifacts": artifacts, "qualification_input": False, "charged_wall_seconds_added": 0,
            "limit": "Exact recorded trajectory agreement across source identities supports comparability; original optimizer/critic checkpoints were not retained to prove state identity. No historical qualification is transferred."}


def render(arm, trace, checkpoints, bound, destination):
    """Every point is a retained particle; vertical placement identifies its row."""
    from PIL import Image, ImageDraw
    steps = [0, 40, 80, 160, 200, 320, 400, 600, 800]
    width, height, left, right = 1000, 510, 90, 910
    frames = []
    def x(value):
        return round(left + (value + bound) / (2 * bound) * (right - left))
    for step in steps:
        positions = checkpoints["initial"]["positions"] if step == 0 else trace[step - 1]["positions"]
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 18), f"{arm['recipe_label']} | schedule horizon {arm['schedule_horizon']} | update {step}/800", fill="black")
        draw.text((20, 43), "Actual retained 12-particle coordinates; rows show particle identity", fill="black")
        for pole in (-1., 0., 1.):
            color = "#b0b0b0" if pole else "#dfdfdf"
            draw.line((x(pole), 85, x(pole), 355), fill=color, width=2)
            draw.text((x(pole) - 12, 364), str(pole), fill="#606060")
        draw.line((left, 355, right, 355), fill="black")
        draw.text((left, 383), f"Fixed position axis: {-bound:g} to {bound:g}; gray references are declared target poles +/-1", fill="black")
        for index, value in enumerate(positions.flatten().tolist()):
            y, px = 105 + 20 * index, x(value)
            draw.text((40, y - 5), str(index + 1), fill="#606060")
            draw.ellipse((px - 5, y - 5, px + 5, y + 5), fill="#c43b3b")
        if step:
            p = trace[step - 1]
            draw.text((20, 412), f"mean_abs={p['mean_abs']:.6f}  particle std={p['particle_std']:.6f}  signs (-/0/+)={p['negative_particles']}/{p['zero_particles']}/{p['positive_particles']}", fill="black")
            draw.text((20, 438), f"actual particle LR={p['actual_generator_lr']:.7g}  critic LR={p['critic_lr']:.7g}  gain={p['direct_gain']:.4f}", fill="black")
        draw.text((20, 474), "Movement gate: mean_abs >= .3 and grad_med <= 1; two-mode coverage is diagnostic only", fill="black")
        frames.append(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=550, loop=0, disposal=2, optimize=False)
    with Image.open(destination) as gif:
        assert gif.n_frames == len(steps)
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=REPORT / "plans.json")
    parser.add_argument("--queue-root", type=Path)
    args = parser.parse_args()
    assert not (REPORT / "archive.json").exists(), "Publication frozen in an immutable archive"
    plan = read(args.plans)
    queue_root = args.queue_root or ROOT / "runs/forge" / plan["study"] / "queue"
    state = Queue(queue_root, on_completion=None).inspect()
    assert plan["qualification_input"] is False and len(plan["arms"]) == 4
    assert len(state["jobs"]) == 4 and all(s["status"] in {"completed", "concluded"} for s in state["submissions"].values())
    loaded, sources = [], set()
    for arm in plan["arms"]:
        certified_data = certified(arm, state, plan, sources)
        descriptor, trace, checkpoints = saved_trace(certified_data[1], certified_data[4], plan)
        loaded.append((arm, certified_data, descriptor, trace, checkpoints))
    common_bound = max(1.5, math.ceil(max(float(p["positions"].abs().max()) for _, _, _, trace, _ in loaded for p in trace) * 10) / 10)
    decisions = {}
    for arm, data, _, _, _ in loaded:
        request = data[2]
        if arm["candidate_id"] not in decisions:
            rows = [item[1][4] for item in loaded if item[0]["candidate_id"] == arm["candidate_id"]]
            decisions[arm["candidate_id"]] = evaluate(request, rows)
            assert not decisions[arm["candidate_id"]]["binding_errors"]
            assert decisions[arm["candidate_id"]]["outcome"] != "incomplete"
    arms, labels, charged, statuses = [], {}, 0., Counter()
    declarations = {str(args.plans.resolve().relative_to(ROOT)), f"configs/forge/views/{plan['view']}.json",
                    f"configs/forge/campaigns/{plan['campaign_id']}.json", "configs/forge/tasks/two_pole.json"}
    for arm, (durable, local, request, task, row, certificate), descriptor, trace, checkpoints in loaded:
        milestones = []
        diagnostic_points = {p["step"]: p for p in row["evidence"]["diagnostic_observations"]}
        for step in plan["milestones"]:
            point = trace[step - 1]
            scalar = {k: v for k, v in point.items() if not hasattr(v, "shape") and k != "critic_diagnostics"}
            scalar["grad_med"] = diagnostic_points[step]["grad_med"]
            scalar["behavior_bounds_pass"] = scalar["mean_abs"] >= .3 and scalar["grad_med"] <= 1
            scalar["state_sha256"] = state_hash(checkpoints["checkpoints"][step])
            scalar["critic_diagnostics"] = {k: v for k, v in point["critic_diagnostics"].items() if not hasattr(v, "shape")}
            milestones.append(scalar)
        gif = REPORT / "media" / (arm["id"] + ".gif")
        media_steps = render(arm, trace, checkpoints, common_bound, gif)
        wall = row["cost"]["wall_seconds"]
        assert math.isfinite(wall) and 0 <= wall <= plan["task_ceiling_seconds"]
        charged += wall; statuses[row["gate_status"]] += 1
        receipt = {"schema_version": 1, "scope": "nonqualifying_global_recipe_horizon_diagnostic", "arm_id": arm["id"],
            "recipe_label": arm["recipe_label"], "candidate_id": arm["candidate_id"], "candidate_revision": arm["candidate_revision"],
            "task_id": arm["task_id"], "attempt_id": durable.name, "request_id": request["request_id"],
            "compatibility_key": arm["compatibility_key"], "source_digest": plan["source_digest"],
            "source_commit": request["source"]["origin_commit"], "runtime": request["runtime"],
            "task_execution_binding": request["jobs"][next(i for i,j in enumerate(request["jobs"]) if j["task_id"] == arm["task_id"])]["science"]["execution"],
            "global_recipe_overrides": request["candidate"]["recipe_overrides"], "effective_recipe": row["applied"]["recipe"],
            "prior": row["applied"]["prior"], "initializer": row["applied"]["initializer"],
            "initial_state_sha256": state_hash(checkpoints["initial"]),
            "initial_component_sha256": {k: state_hash(v) for k,v in checkpoints["initial"].items()},
            "rng_manifest_sha256": stable_hash(row["applied"]["rng"]), "sampling_law": row["evidence"]["sampling_law"],
            "execution_updates": 800, "schedule_horizon": arm["schedule_horizon"],
            "diagnostic_gate_status": row["gate_status"], "final_metrics": row["metrics"],
            "evaluator_result": row["evaluator_result"], "guards": row["evidence"]["guards"],
            "training_schedules": row["applied"]["training_schedules"], "optimizer_group_bindings": row["applied"]["optimizer_group_bindings"],
            "milestones": milestones, "phases": [phase_summary(trace, a,b) for a,b in ((1,80),(81,200),(201,400),(401,800))],
            "prefix80": {"observations": 24, "final_metrics": diagnostic_points[80],
                "original_cadence_steps": [math.ceil(i * 80 / 24) for i in range(1,25)],
                "observation_sha256": stable_hash([p for p in row["evidence"]["diagnostic_observations"] if p["step"] <= 80]),
                "qualification_input": False},
            "retained_diagnostic_artifacts": descriptor["artifacts"],
            "durable_certificate": {"result_stable_hash": certificate["result_hash"],
                "artifacts": [identity(durable / name) for name in ("request.json", "result.json", "evidence.json")]},
            "raw_artifacts": [identity(local / name) for name in ("request.json", "raw-result.json", "graded-result.json", "result.json")],
            "media": identity(gif), "media_frames": len(media_steps), "media_steps": media_steps,
            "media_fixed_axis": [-common_bound, common_bound], "decision_review": decisions[arm["candidate_id"]],
            "charged_wall_seconds": wall, "qualification_input": False, "default_adoption": False,
            "publication_optimizer_updates": 0, "publication_sampling_draws": 0,
            "publication_source": identity(Path(__file__))}
        path = REPORT / "receipts" / (arm["id"] + ".json")
        write(path, receipt)
        arms.append({"arm_id": arm["id"], "recipe_label": arm["recipe_label"], "candidate_id": arm["candidate_id"],
                     "schedule_horizon": arm["schedule_horizon"], "diagnostic_gate_status": row["gate_status"],
                     "final_metrics": row["metrics"], "charged_wall_seconds": wall, "receipt": identity(path)})
        labels.setdefault(arm["recipe_label"], {})[arm["schedule_horizon"]] = receipt
        declarations.add(arm["declaration"])
        declarations.add(f"configs/forge/tasks/{arm['task_id']}.json")
    comparisons, prefix_controls = [], []
    for label, pair in labels.items():
        fixed, stretched = pair[80], pair[800]
        assert fixed["global_recipe_overrides"] == stretched["global_recipe_overrides"]
        assert fixed["initial_component_sha256"] == stretched["initial_component_sha256"]
        assert {k:v for k,v in fixed["effective_recipe"].items() if k != "total_steps"} == {k:v for k,v in stretched["effective_recipe"].items() if k != "total_steps"}
        prefix_controls.append(historical_prefix(label, fixed))
        comparisons.append({"recipe_label": label, "global_recipe_unchanged": True,
            "initial_component_states_identical": True,
            "only_effective_recipe_difference": "task-owned total_steps schedule horizon80 versus800",
            "milestone_deltas_stretched_minus_fixed": [{"step": a["step"], **{k:b[k]-a[k] for k in ("mean_abs","grad_med","particle_std","nearest_pole_distance","adversarial_gradient_norm","l2_gradient_norm")}}
                                                       for a,b in zip(fixed["milestones"],stretched["milestones"])],
            "scope": "Declared schedule bundle: LR/noise durations and LR-dependent K3P behavior; no isolated LR causal claim."})
    summary = {"schema_version": 1, "study": plan["study"], "scope": "bounded_two_recipe_two_horizon_diagnostic",
        "source_digest": plan["source_digest"], "source_commits": sorted({data[2]["source"]["origin_commit"] for _,data,_,_,_ in loaded}),
        "arms": arms, "comparisons": comparisons, "historical_prefix_controls": prefix_controls,
        "decisions": decisions, "statuses": dict(statuses),
        "charged_wall_seconds": charged, "campaign_ceiling_seconds": plan["campaign_ceiling_seconds"],
        "candidate_ceiling_seconds": plan["candidate_ceiling_seconds"], "task_ceiling_seconds": plan["task_ceiling_seconds"],
        "declarations": sorted(declarations), "qualification_input": False, "default_adoption": False,
        "ordinary_gate_changes": 0, "ordinary_qualification_reuse": 0,
        "publication_optimizer_updates": 0, "publication_sampling_draws": 0,
        "interpretation_limits": ["Movement and bounded slope do not establish two-pole mass balance or coverage.",
            "Clean zero-origin coordinates can separate through the existing paired relativistic payoff; report measured sign balance and spread rather than infer two-mode acquisition from movement.",
            "Horizon800 stretches input-noise duration8 to80 for the noisy recipe and changes LR-dependent penalty/anchor timing. Continuing800updates may activate the existing fixed200-step critic guard.",
            "Neither diagnostic horizon supplies ordinary Tier1 qualification, word-task evidence, calibrated screening or default adoption."],
        "validation": {"four_durable_certificates_and_raw_grades": True, "full_source_snapshots": True,
            "force_trace_arithmetic_and_actual_milestone_states": True, "paired_initial_states_identical": True,
            "unchanged_global_recipes_and_behavior_thresholds": True, "formal_decisions_complete": True,
            "two_historical80prefixes_exact_all24_observation_agreement": True}}
    assert charged <= plan["campaign_ceiling_seconds"]
    write(REPORT / "summary.json", summary)
    print(json.dumps({"event": "horizon_diagnostic_published", "summary": identity(REPORT / "summary.json"),
                      "arms": 4, "statuses": dict(statuses), "charged_wall_seconds": charged}, sort_keys=True))


if __name__ == "__main__":
    main()
