"""Audit archived Tier 1 samples and numerical controls; never train or sample a model.

Oracle panels use one isolated diagnostic RNG, not training-seed experiments.
Historical grading and the single current leaderboard are never modified.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from scipy.special import ndtr
from scipy.stats import beta
import torch

from benchmarks.toy_audit.gaussian1d_experiment import controls as gaussian_controls
from benchmarks.toy_audit.gaussian1d_quality import score_samples as gaussian_score
from benchmarks.toy_audit.ring16_controls import run_controls as ring_controls
from benchmarks.toy_audit.ring16_quality import score_samples as ring_score


ROOT = Path(__file__).resolve().parents[3]
ATTEMPTS = {
    "bcap_gaussian": "d0638ad5ce5a47e5b2fcb00b369768b2",
    "bcap_ring": "d8185ca486b54af79ef422395eac8065",
    "k3p_gaussian": "7ccd8797a0ab4592b312a86354518228",
    "k3p_ring": "d6d907c833524c1ea23374d90c548cf6",
}
RECIPE_KEYS = ("lr", "d_lr_mult", "prior_lr_mult", "reg_arm", "reg_coeff", "reg_kappa",
               "input_noise_std", "output_noise_std", "total_steps", "network_lr_horizon_cap")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def event(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def failed_bounds(metrics, thresholds):
    compare = {"<=": lambda x, y: x <= y, ">=": lambda x, y: x >= y, "==": lambda x, y: x == y}
    return [key for key, op, bound in thresholds
            if metrics.get(key) is None or not math.isfinite(metrics[key]) or not compare[op](metrics[key], bound)]


def load_archive(card, archive_path):
    """Bind byte-exact originals; the published archive works after local queue cleanup."""
    archive_bytes = archive_path.read_bytes()
    assert sha(archive_bytes) == card["archive"]["sha256"], "archive hash differs"
    names = []
    for attempt in ATTEMPTS.values():
        names.extend(f"durable/{attempt}/{name}" for name in ("request.json", "result.json", "evidence.json"))
        names.extend(f"attempts/{attempt}/{name}" for name in
                     ("raw-result.json", "graded-result.json", "observed-samples.pt"))
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        all_names = archive.getnames()
        assert len(all_names) == len(set(all_names)), "duplicate archive members"
        data = {name: archive.extractfile(name).read() for name in names}
    for name, value in data.items():
        assert sha(value) == card["manifest"]["original_receipts_sha256"][name], name
    return data


def bind_attempt(data, attempt):
    def read(prefix, name):
        return json.loads(data[f"{prefix}/{attempt}/{name}"])
    envelope = read("durable", "request.json")
    request = envelope["request"]
    result, certificate = read("durable", "result.json"), read("durable", "evidence.json")
    raw, grading = read("attempts", "raw-result.json"), read("attempts", "graded-result.json")
    assert stable(result) == certificate["result_hash"]
    assert request["source"] == certificate["source"] and request["runtime"] == certificate["runtime"]
    assert stable(request["source"]["files"]) == request["source"]["digest"]
    assert result["attempt_id"] == attempt and result["candidate_revision"] == request["candidate_revision"]
    assert stable(raw) == grading["raw_hash"] and grading["source_digest"] == request["source"]["digest"]
    assert result["raw"]["grading"] == grading
    row, = result["task_results"]
    task = request["tasks"][row["task_id"]]
    assert row["compatibility_key"] == envelope["job"]["compatibility_key"]
    assert row["gate_status"] == grading["grades"][row["task_id"]]["gate_status"]
    steps = task["execution"]["steps"]
    assert row["raw_status"] == "completed" and row["cost"]["completed_steps"] == steps
    assert row["cost"]["optimizer_updates"] == dict(generator=steps, discriminator=steps, prior=steps)
    assert row["evidence"]["guards"]["all_finite"]
    assert row["evidence"]["guards"]["unintended_rng_deviations"] == 0
    for key in ("recipe", "prior", "initializer", "initialization", "rng", "field_ownership", "evidence"):
        assert row[key] == raw[key], key
    assert raw["prior"] == task["execution"]["prior"]
    assert raw["initializer"] == task["execution"]["initializer"]
    assert raw["evidence"]["host"]["definition"] == task["execution"]["host_definition"]
    for key, value in request["candidate"]["recipe_overrides"].items():
        if key in raw["recipe"]:
            assert value == raw["recipe"][key], key
    desc = raw["evidence"]["saved_observer_outputs"]
    tensor_bytes = data[f"attempts/{attempt}/{desc['path']}"]
    assert sha(tensor_bytes) == desc["sha256"] and len(tensor_bytes) == desc["bytes"]
    assert desc["optimizer_updates_added"] == desc["sampling_draws_added"] == 0
    records = torch.load(io.BytesIO(tensor_bytes), weights_only=True, map_location="cpu")
    assert len(records) == desc["observation_count"] == len(raw["evidence"]["observations"]) == 24
    return request, task, raw, row, records


def ring_details(points, spec):
    means = points.new_tensor(spec["means"])
    cov = points.new_tensor(spec["covariances"])
    labels = torch.cdist(points, means).argmin(1)
    delta = points - means[labels]
    mahal = torch.einsum("ni,nij,nj->n", delta, torch.linalg.inv(cov)[labels], delta)
    centered = torch.empty_like(points)
    components = []
    for k in range(len(means)):
        selected = labels == k
        member = points[selected]
        local = member - member.mean(0)
        centered[selected] = local
        empirical = local.T @ local / len(member)
        whiten = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
        eigen = torch.linalg.eigvalsh(whiten @ empirical @ whiten.T)
        radial = means[k] / means[k].norm()
        tangent = torch.stack((-radial[1], radial[0]))
        components.append(dict(component=k, count=len(member),
            covariance_error=float((empirical-cov[k]).norm()/cov[k].norm()),
            min_eigen_ratio=float(eigen.min()), max_eigen_ratio=float(eigen.max()),
            centroid_offset_sigma=float((member.mean(0)-means[k]).norm()/cov[k, 0, 0].sqrt()),
            radial_variance_ratio=float(radial @ empirical @ radial/cov[k, 0, 0]),
            tangential_variance_ratio=float(tangent @ empirical @ tangent/cov[k, 0, 0]),
            beyond_four_sigma_fraction=float((mahal[selected] > 16).float().mean())))
    energy = centered.square().sum(1)
    return dict(beyond_four_sigma_fraction=float((mahal > 16).float().mean()),
        beyond_four_sigma_covariance_energy_share=float(energy[mahal > 16].sum()/energy.sum()),
        components_below_min_eigen_gate=sum(c["min_eigen_ratio"] < .15 for c in components),
        component_count_range=[min(c["count"] for c in components), max(c["count"] for c in components)],
        mean_centroid_offset_sigma=float(np.mean([c["centroid_offset_sigma"] for c in components])),
        mean_radial_variance_ratio=float(np.mean([c["radial_variance_ratio"] for c in components])),
        mean_tangential_variance_ratio=float(np.mean([c["tangential_variance_ratio"] for c in components])),
        distance_sigma_quantiles=dict(zip(("p50", "p85", "p95", "p99"),
            torch.quantile(mahal.sqrt(), points.new_tensor((.5, .85, .95, .99))).tolist())),
        components=components)


def bootstrap_ring(points, spec, rng, draws):
    keys = ("component_covariance_error", "component_core_covariance_error",
            "component_min_eigen_ratio", "component_core_min_eigen_ratio", "hq", "mass_tv")
    values = {key: [] for key in keys}
    for _ in range(draws):
        sampled = points[torch.from_numpy(rng.integers(len(points), size=len(points)))]
        scored = ring_score(sampled, spec, 400)
        for key in keys:
            values[key].append(scored[key])
    return dict(method="iid percentile bootstrap of retained samples, exploratory, not qualification",
                resamples=draws, percentile_95_intervals={key: np.quantile(v, [.025, .975]).tolist()
                                                        for key, v in values.items()})


def audit_attempt(name, data, rng, bootstrap_draws):
    attempt = ATTEMPTS[name]
    request, task, raw, row, records = bind_attempt(data, attempt)
    gaussian = "gaussian" in name
    score = gaussian_score if gaussian else ring_score
    spec = raw["evidence"]["host"]["definition"]
    evaluator_paths = ["benchmarks/toy_audit/gaussian1d_quality.py"] if gaussian else [
        "benchmarks/toy_audit/ring16_quality.py", "benchmarks/transfer_suite/vector_tasks.py"]
    for path in evaluator_paths:
        assert sha((ROOT/path).read_bytes()) == task["evaluation"]["sources"][path], path
    max_error = 0.
    for record, observation in zip(records, raw["evidence"]["observations"]):
        assert record["step"] == observation["step"] and record["samples"].shape == (4096, 1 if gaussian else 2)
        scored = score(record["samples"], spec, record["step"])
        for key, _, _ in task["evaluation"]["thresholds"]:
            delta = abs(scored[key]-observation[key])
            max_error = max(max_error, delta)
            assert math.isclose(scored[key], observation[key], abs_tol=2e-6, rel_tol=2e-6), (name, key)
    terminal = []
    for record, observation in zip(records[-5:], raw["evidence"]["observations"][-5:]):
        metrics = {key: observation[key] for key, _, _ in task["evaluation"]["thresholds"]}
        item = dict(step=record["step"], metrics=metrics, failed_bounds=failed_bounds(metrics, task["evaluation"]["thresholds"]))
        if gaussian:
            points = record["samples"].numpy().astype(np.float64)
            epsilon = math.sqrt(math.log(2*5/.01)/(2*len(points)))
            item["population_cdf_ks_99pct_simultaneous_dkw_interval"] = [
                max(0., metrics["cdf_ks"]-epsilon), min(1., metrics["cdf_ks"]+epsilon)]
            recentered = points-points.mean()+spec["means"][0][0]
            normalized = (points-points.mean())/points.std()*math.sqrt(spec["covariances"][0][0][0])+spec["means"][0][0]
            item["recentered_cdf_ks_diagnostic_only"] = score(recentered, spec)["cdf_ks"]
            item["matched_moments_cdf_ks_diagnostic_only"] = score(normalized, spec)["cdf_ks"]
            item["mean_standard_error_in_target_sigma"] = float(points.std()/math.sqrt(len(points))/math.sqrt(spec["covariances"][0][0][0]))
        terminal.append(item)
    entry = dict(attempt_id=attempt, candidate_id=request["candidate"]["id"],
        candidate_revision=request["candidate_revision"], source_commit=request["source"]["origin_commit"],
        source_digest=request["source"]["digest"], original_gate_status=row["gate_status"],
        compatibility_key=row["compatibility_key"], task_id=task["id"],
        task_identity=stable({key: value for key, value in task.items() if key != "field_ownership"}),
        runtime_cohort=stable(request["runtime"]), recipe={key: raw["recipe"].get(key) for key in RECIPE_KEYS},
        recipe_fields_absent_in_original=[key for key in RECIPE_KEYS if key not in raw["recipe"]],
        nominal_prior_base_lr=raw["recipe"]["lr"]*raw["recipe"]["prior_lr_mult"], prior=raw["prior"],
        initializer=raw["initializer"], initialization_digest=stable(raw["initialization"]),
        sampling_law=raw["evidence"]["sampling_law"], scoring_weights=task["evaluation"]["scoring_weights"],
        completed_steps=row["cost"]["completed_steps"], thresholds=task["evaluation"]["thresholds"],
        convergence=row["evaluator_result"]["convergence"], saved_samples=raw["evidence"]["saved_observer_outputs"],
        all_24_gate_observations_match=True, max_gate_recomputation_absolute_error=max_error,
        final_metrics=row["metrics"], terminal_checks=terminal)
    if not gaussian:
        entry["final_diagnostics"] = ring_details(records[-1]["samples"], spec)
        entry["final_sampling_uncertainty"] = bootstrap_ring(records[-1]["samples"], spec, rng, bootstrap_draws)
    return entry, task, raw


def gaussian_oracle_panels(task, rng, panels):
    """Assess repeated-check rejection on exact declared laws, with no training."""
    spec = task["execution"]["host_definition"]
    n = 4096
    rows = []
    for shift_sigma in (0., .075, .10, .15, .25):
        checks_failed = panels_failed = 0
        for _ in range(panels):
            failed = 0
            for _ in range(5):
                points = 2.+.5*(rng.standard_normal((n, 1))+shift_sigma)
                failed += bool(failed_bounds(gaussian_score(points, spec), task["evaluation"]["thresholds"]))
            checks_failed += failed
            panels_failed += failed > 0
        rows.append(dict(control=f"Gaussian_shift_{shift_sigma:g}_sigma", population_cdf_ks=float(2*ndtr(shift_sigma/2)-1),
            true_population_pass=shift_sigma <= .1, observations=5*panels, rejected_observations=checks_failed,
            panels=panels, rejected_five_check_panels=panels_failed,
            observation_rejection_rate=checks_failed/(5*panels), panel_rejection_rate=panels_failed/panels,
            panel_rejection_rate_exact_binomial_95_interval=[
                float(beta.ppf(.025, panels_failed, panels-panels_failed+1)) if panels_failed else 0.,
                float(beta.ppf(.975, panels_failed+1, panels-panels_failed)) if panels_failed < panels else 1.]))
    return dict(scope="analytic sampling-control diagnostic; not training or tier calibration",
        iid_panel_checks=5, sample_count=n, oracle_panel_false_reject_dkw_union_upper_bound=10*math.exp(-2*n*.05**2),
        controls=rows)


def expanded_ring_controls(task):
    spec = task["execution"]["host_definition"]
    means = torch.tensor(spec["means"])
    n = 4096
    labels = torch.arange(n) % 16
    offsets = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])*(math.sqrt(2)*.1)
    atom_points = means[labels]+offsets[(torch.arange(n)//16) % 4]
    points = means[labels]+.1*torch.randn(n, 2, generator=torch.Generator().manual_seed(98271))
    # Five of 256 samples per component at nine sigma; near-perfect core with bad tails.
    tail = torch.arange(n)//16 < 5
    points[tail] = means[labels[tail]]*1.3
    core_policy = [[key.replace("component_covariance_error", "component_core_covariance_error")
                        .replace("component_min_eigen_ratio", "component_core_min_eigen_ratio"), op, bound]
                   for key, op, bound in task["evaluation"]["thresholds"]]+[["max_component_spill", "<=", .05]]
    controls = []
    for name, values in (("same_covariance_four_atoms_per_cluster", atom_points), ("two_percent_nine_sigma_tails", points)):
        metrics = ring_score(values, spec, 400)
        original = failed_bounds(metrics, task["evaluation"]["thresholds"])
        alternate = failed_bounds(metrics, core_policy)
        controls.append(dict(control=name, full_covariance_gate_passed=not original, full_covariance_failed_bounds=original,
            core_plus_spill_gate_passed=not alternate, core_plus_spill_failed_bounds=alternate,
            metrics={key: value for key, value in metrics.items() if key in {
                "sample_count", "modes", "mass_tv", "hq", "component_covariance_error", "component_min_eigen_ratio",
                "component_core_covariance_error", "component_core_min_eigen_ratio", "max_component_spill"}}))
    assert controls[0]["full_covariance_gate_passed"] and controls[0]["core_plus_spill_gate_passed"]
    assert not controls[1]["full_covariance_gate_passed"] and controls[1]["core_plus_spill_gate_passed"]
    return dict(scope="counterexamples, not new qualification gates", core_plus_spill_policy=core_policy, controls=controls)


def two_pole_budget_audit():
    task_path = ROOT/"configs/forge/tasks/two_pole.json"
    task = json.loads(task_path.read_text())
    summary_path = ROOT/"reports/forge/k3p-two-pole-horizon-v1/summary.json"
    summary = json.loads(summary_path.read_text())
    assert task["execution"]["steps"] == 80
    assert summary["ordinary_gate_changes"] == summary["ordinary_qualification_reuse"] == 0
    return dict(scope="read-only existing budget and optimizer-role audit; no new training",
        task_sha256=sha(task_path.read_bytes()), budget_updates=80,
        thresholds=task["evaluation"]["thresholds"],
        numerical_question="Mean absolute direct-particle travel >=.3 with median critic slope <=1, not two-pole distribution fidelity",
        optimizer_role_binding=dict(direct_coordinates="Recipe.lr; direct_particle_gain can raise it up to 2x",
            prior_lr_mult="not consumed by direct-coordinate parameter group", schedule="prior schedule over frozen host horizon"),
        source_hashes={path: sha((ROOT/path).read_bytes()) for path in (
            "benchmarks/locked_shared/two_pole.py", "experiments/forge/behavior_adapters.py", "particlegan/recipes.py")},
        existing_horizon_diagnostic=dict(path=str(summary_path.relative_to(ROOT)), sha256=sha(summary_path.read_bytes()),
            source_digest=summary["source_digest"], arms=summary["arms"]),
        recommendation="Keep the 80-update gate in the first search. Slower global LR may conflict with this movement budget; a failure requires force/trajectory evidence before a task-budget revision.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("task-audit.json"))
    parser.add_argument("--panels", type=int, default=400)
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    args = parser.parse_args(argv)
    if args.panels < 1 or args.bootstrap_resamples < 1:
        parser.error("diagnostic counts must be positive")
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    rng = np.random.default_rng(9152026)
    card_path = ROOT/"reports/forge/tier1-completion/artifact-inventory.json"
    card = json.loads(card_path.read_text())
    archive_path = args.archive or Path(card["archive"]["path"])
    event("bind_archive", archive=str(archive_path))
    data = load_archive(card, archive_path)
    entries, bound = {}, {}
    for name in ATTEMPTS:
        event("audit_saved_samples", arm=name)
        entry, task, raw = audit_attempt(name, data, rng, args.bootstrap_resamples)
        entries[name], bound[name] = entry, (task, raw)
    for kind in ("gaussian", "ring"):
        a, b = bound[f"bcap_{kind}"], bound[f"k3p_{kind}"]
        assert a[0]["execution"] == b[0]["execution"]
        assert a[0]["evaluation"] == b[0]["evaluation"]
        for field in ("prior", "initializer", "initialization", "rng"):
            assert a[1][field] == b[1][field], field
    event("existing_scorer_controls")
    existing = dict(gaussian=gaussian_controls(), ring=ring_controls())
    assert existing["gaussian"]["passed"] and existing["ring"]["passed"]
    for kind, control_report in existing.items():
        gate_keys = {key for key, _, _ in bound[f"bcap_{kind}"][0]["evaluation"]["thresholds"]}
        for control in control_report["controls"]:
            control["metrics"] = {key: value for key, value in control["metrics"].items()
                                  if key in gate_keys or key == "nonfinite_output_values"}
    event("oracle_sampling_panels", panels_per_law=args.panels)
    panels = gaussian_oracle_panels(bound["bcap_gaussian"][0], rng, args.panels)
    expanded = expanded_ring_controls(bound["bcap_ring"][0])
    assert torch.equal(rng_before, torch.get_rng_state()), "global Torch RNG changed"
    report = dict(schema_version=1, id="bcap-tier1-task-audit-v1", qualification_input=False,
        training_updates_added=0, model_sampling_draws_added=0,
        diagnostic_rng=dict(numpy_generator_seed=9152026, ring_control_seed=98271),
        archive=card["archive"], archive_card_sha256=sha(card_path.read_bytes()),
        original_evidence_identities_preserved=True, cpu_global_rng_unchanged=True,
        comparable_task_prior_initializer_rng_bindings=True,
        interpretation_limits=["BCAP/K3P recipes differ; this is not a penalty-only causal comparison.",
            "DKW intervals concern iid evaluation draw uncertainty at five different trained states; no states are pooled.",
            "Bootstrap intervals are exploratory conditional on retained samples and do not measure training reliability.",
            "Oracle scorer controls do not establish training solvability or calibrate Tier 1 placement.",
            "Core or recentered metrics are diagnostics and never replace the recorded grading."],
        retained_outputs=entries, existing_controls=existing,
        gaussian_five_check_sampling_diagnostic=panels, expanded_ring_counterexamples=expanded,
        two_pole_budget_audit=two_pole_budget_audit())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n")
    event("complete", output=str(args.output), saved_observations_verified=96, training_updates_added=0)


if __name__ == "__main__":
    main()
