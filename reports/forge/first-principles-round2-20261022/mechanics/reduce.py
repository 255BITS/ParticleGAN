#!/usr/bin/env python3
"""Reduce final certified mechanics/energy readouts and saved scalar traces.

Read-only: no model construction, forwards, training, sampling or regrading.
The parent study owns certification, the leaderboard and scientific selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


TASKS = ("gaussian1d_smoke", "gaussian1d_stability", "vector_unequal_mass",
         "vector_unequal_width", "vector_anisotropic", "vector_overlap", "img_bars4",
         "img_blobs4", "img_intensity2", "grid100", "rotated100", "staggered100",
         "ring16_acquisition", "mode_hold", "img_stripes2")
CONTROL = "bispf-output-energy-v1"
CANDIDATE = "bispf-secant-extragradient-v1"
METRICS = ("cdf_ks", "mean_error_sigma", "std_ratio", "mass_tv", "hq", "modes",
           "component_counts", "component_mass", "component_covariance_error",
           "component_core_covariance_error", "component_covariance_errors",
           "component_core_covariance_errors", "component_min_eigen_ratio",
           "min_mass_ratio", "max_component_spill", "sw1_normalized", "mean_rmse",
           "distribution_tv", "mode_fractions", "quality_mode_fractions", "mean_error", "covariance_error",
           "precision", "center_rms_sigma", "cov_frob_rms", "radial_ks")


def read(path, candidate):
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["candidate_id"] != candidate:
        raise ValueError(f"Wrong candidate: {path}")
    if data.get("complete") is not True or set(data["expected_tasks"]) != set(TASKS):
        raise ValueError("Readout is not a complete declared panel")
    rows = {row["task_id"]: row for row in data["task_results"]}
    if len(rows) != len(data["task_results"]) or set(rows) != set(TASKS):
        raise ValueError("Final analysis requires exactly the complete 15-task panel")
    if any(row["status"] not in ("PASS", "FAIL", "BLOCKED", "ERROR", "INCOMPLETE", "INVALID") for row in rows.values()):
        raise ValueError("Nonterminal task in readout")
    return data, rows, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                           candidate_id=candidate, candidate_revision=data["candidate_revision"],
                           source_digest=data["source"]["digest"])


def training_streams(row):
    packet = (row.get("evidence") or {}).get("provenance_checkpoint")
    if packet is None:
        return None
    result = {}
    for name, digest in packet["named_stream_state_sha256"].items():
        key = json.loads(name)
        if key[0] == "eval":
            continue
        if key[-1].startswith("cuda:"):
            key[-1] = "cuda"
        normalized = json.dumps(key, separators=(",", ":"))
        if normalized in result and result[normalized] != digest:
            raise ValueError("Conflicting normalized stream names")
        result[normalized] = digest
    return result


def distribution(values):
    if not values:
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite trace statistic")
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.fmean(values), max=max(values))


def trace_section(rows):
    stats = [row["generator_replay"] for row in rows]
    if not stats:
        return None
    accepted = [s for s in stats if s["accepted"]]
    nonzero = [s for s in stats if s["applied_scale"] > 0 and s["corrected_proposal_l2"] > 0]
    ratios = [s["applied_l2"] / (s["applied_scale"] * s["corrected_proposal_l2"]) for s in nonzero]
    thermal = [(r.get("penalty_stats") or {}).get("s_thermal") for r in rows]
    thermal = [value for value in thermal if type(value) in (int, float)]
    return dict(rows=len(rows), first_step=rows[0]["step"], last_step=rows[-1]["step"],
        accepted=len(accepted), rejected=len(stats)-len(accepted),
        zero_actual_movements=sum(s["applied_l2"] == 0 for s in stats),
        alpha_counts=dict(sorted(Counter(str(s["applied_scale"]) for s in stats).items())),
        trial_counts=dict(sorted(Counter(str(s["trials"]) for s in stats).items())),
        mean_trials=statistics.fmean(s["trials"] for s in stats),
        thermal_observations=len(thermal), thermal_scale=distribution(thermal),
        nonfinite_trials=sum(s["nonfinite_trials"] for s in stats),
        protection_changed=sum(s["protection_delta_l2"] not in (None, 0) for s in stats),
        base_raw_proposal_l2=distribution([s["base_proposal_l2"] for s in stats]),
        raw_corrected_proposal_l2=distribution([s["raw_corrected_proposal_l2"] for s in accepted]),
        protected_corrected_proposal_l2=distribution([s["corrected_proposal_l2"] for s in stats]),
        protection_delta_l2=distribution([s["protection_delta_l2"] for s in accepted]),
        actual_applied_l2=distribution([s["applied_l2"] for s in stats]),
        raw_corrected_over_base_l2=distribution([s["raw_corrected_proposal_l2"] / s["base_proposal_l2"]
            for s in accepted if s["base_proposal_l2"] > 0]),
        protected_over_raw_corrected_l2=distribution([s["corrected_proposal_l2"] / s["raw_corrected_proposal_l2"]
            for s in accepted if s["raw_corrected_proposal_l2"] > 0]),
        actual_over_base_l2=distribution([s["applied_l2"] / s["base_proposal_l2"]
            for s in stats if s["base_proposal_l2"] > 0]),
        actual_over_scaled_protected_l2=distribution(ratios),
        optimizer_clocks=sum(s["optimizer_clocks"] for s in stats),
        additional_sampling_draws=sum(s["additional_sampling_draws"] for s in stats),
        field_evaluations=sum(s["field_evaluations"] for s in stats),
        accepted_bound_violations=sum(s["consistency_lhs"] is not None and
            s["consistency_lhs"] > s["consistency_rhs"] for s in accepted))


def trace_summary(row):
    evidence = row.get("evidence") or {}
    receipt = evidence.get("public_update_trace")
    if receipt is None:
        return dict(status="UNAVAILABLE", task_status=row["status"])
    path = Path(receipt["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError(f"Trace hash differs: {path}")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != receipt["rows"] or not rows:
        raise ValueError("Trace row count differs or empty")
    expected = list(range(rows[0]["step"], rows[-1]["step"]+1))
    if [row["step"] for row in rows] != expected:
        raise ValueError("Trace steps are missing/repeated")
    for item in rows:
        s = item["generator_replay"]
        if (s["certificate_scope"] != "pre_protection" or s["optimizer_clocks"] != 1
                or s["additional_sampling_draws"] != 0 or not 0 <= s["trials"] <= 9
                or s["field_evaluations"] != s["trials"]+1):
            raise ValueError("Replay invariant differs")
    guards = evidence["guards"]["optimizer_updates"]
    if rows[-1]["step"] != guards["generator"]:
        raise ValueError("Last trace step differs from actual generator update clock")
    all_rows = trace_section(rows)
    quarter = max(1, len(rows)//4)
    result = dict(status="VERIFIED", receipt=receipt, all=all_rows,
                  first_quarter=trace_section(rows[:quarter]),
                  last_quarter=trace_section(rows[-quarter:]),
                  actual_optimizer_updates=guards)
    if row["task_id"] == "gaussian1d_stability":
        result["stationary_continuation"] = trace_section([r for r in rows if r["step"] <= 4000])
        result["shifted"] = trace_section([r for r in rows if r["step"] > 4000])
        result["shift_recovery"] = trace_section([r for r in rows if 4000 < r["step"] <= 5000])
        result["shifted_hold"] = trace_section([r for r in rows if r["step"] > 5000])
    return result


def native_outcome(row):
    evidence, grade = row["evidence"], row["evaluator_result"]
    evaluation = row["field_ownership"]["task_contract"]["evaluation"]["value"]
    name = row["task_id"]+"/events.jsonl"
    packet = evidence["artifact_manifest"]["files"][name]
    path = Path(evidence["artifact_root"])/name
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != packet["sha256"]:
        raise ValueError("Native recorded metric stream differs")
    events = [json.loads(line) for line in raw.splitlines()]
    observations = [dict(step=e["step"], **e["accuracy"]) for e in events
                    if e.get("event") == "eval" and e.get("model") == "live"]
    if len({o["step"] for o in observations}) != len(observations):
        raise ValueError("Duplicate native observation")
    postinitial = [o for o in observations if o["step"] > 0]
    bounds = []
    for name, limit in evaluation["accuracy_limits"].items():
        failed = [o["step"] for o in postinitial if o[name] is None or o[name] > limit]
        value = grade["holdout_metrics"][name]
        bounds.append(dict(metric=name, op="<=", threshold=limit,
            failed=len(failed), total=len(postinitial), failing_steps=failed,
            holdout_value=value, holdout_failed=value is None or value > limit))
    terminal = [dict(step=c["step"], passed=c["passed"], metrics=c["metrics"])
                for c in grade["terminal_checks"]]
    return dict(metric_stream=dict(path=str(path), **packet), initial_observation=observations[0],
        postinitial_observations=len(postinitial), postinitial_passing=sum(o["passed"] for o in postinitial),
        terminal_checks=terminal, terminal_passing=sum(c["passed"] for c in terminal),
        terminal_required=len(terminal), coverage_status=grade["coverage_status"],
        final_20k_metrics=grade["final_metrics"], independent_100k_holdout=grade["holdout_metrics"],
        accuracy_limits=evaluation["accuracy_limits"], coverage_limits=evaluation["coverage_thresholds"],
        accuracy_bound_failures=bounds,
        covariance_frobenius_scope="diagnostic; no independent threshold in declared accuracy_limits")


def outcome(row):
    evidence, grade = row.get("evidence") or {}, row.get("evaluator_result") or {}
    observations = evidence.get("observations", [])
    result = dict(attempt_id=row.get("_attempt_id"), status=row["status"],
        gate_status=row["gate_status"], endpoint=row.get("metrics") or {},
        convergence=grade.get("convergence"),
        failing_endpoint_bounds=[b for b in grade.get("metrics", []) if b["status"] != "PASS"],
        bound_failures_across_observations=[])
    bounds = grade.get("metrics", [])
    if not bounds:
        thresholds = row.get("field_ownership", {}).get("task_contract", {}).get("evaluation", {}).get("value", {}).get("thresholds", [])
        bounds = [dict(metric=name, op=op, threshold=value) for name, op, value in thresholds]
    for bound in bounds:
        name, op, threshold = bound["metric"], bound["op"], bound["threshold"]
        failed = []
        for obs in observations:
            value = obs.get(name)
            if op not in (">=", "<=", "=="):
                raise ValueError(f"Unsupported bound operator {op}")
            ok = type(value) in (int, float) and math.isfinite(value)
            if ok:
                ok = (value >= threshold if op == ">=" else value <= threshold if op == "<=" else value == threshold)
            if not ok:
                failed.append(obs["step"])
        result["bound_failures_across_observations"].append(dict(metric=name, op=op,
            threshold=threshold, failed=len(failed), total=len(observations), failing_steps=failed))
    if row["task_id"] == "gaussian1d_stability":
        result["phase_grade"] = grade
        def passed(observation):
            for bound in bounds:
                value, threshold, op = observation.get(bound["metric"]), bound["threshold"], bound["op"]
                if type(value) not in (int, float) or not math.isfinite(value):
                    return False
                if not (value >= threshold if op == ">=" else value <= threshold if op == "<=" else value == threshold):
                    return False
            return True
        phases = {}
        for phase, selected in (("stationary_continuation", [o for o in observations if o["step"] <= 4000]),
            ("shift_recovery", [o for o in observations if 4000 < o["step"] <= 5000]),
            ("shift_hold", [o for o in observations if o["step"] > 5000])):
            flags = [passed(o) for o in selected]
            suffix = 0
            for flag in reversed(flags):
                if not flag:
                    break
                suffix += 1
            phases[phase] = dict(observations=len(flags), passing=sum(flags), passing_suffix=suffix,
                                 final_five_passing=sum(flags[-5:]))
        if (phases["stationary_continuation"]["passing"] != grade["stationary_passes"]
                or phases["shift_hold"]["passing"] != grade["shift_hold_passes"]):
            raise ValueError("Declared Gaussian bounds do not reproduce phase counts")
        result["scheduled_phase_bound_counts"] = phases
    if row["task_id"] == "gaussian1d_smoke":
        result["acquisition_grade"] = grade
        result["gate_semantics"] = "any_independently_confirmed_acquisition; endpoint_is_separate"
    if row["task_id"] == "vector_unequal_mass" and observations:
        rare = min(range(len(observations[-1]["target_mass"])), key=observations[-1]["target_mass"].__getitem__)
        result["rare_support"] = dict(index=rare, occupied_checks=sum(o["component_counts"][rare]>0 for o in observations),
            scheduled_checks=len(observations), endpoint_count=observations[-1]["component_counts"][rare])
    if row["task_id"] in ("vector_unequal_mass", "vector_unequal_width", "vector_overlap"):
        names = dict.fromkeys(("step", *METRICS, *(b["metric"] for b in bounds)))
        result["last_five_observations"] = [{key: observation[key] for key in names
            if key in observation} for observation in observations[-5:]]
    if row["task_id"] in ("grid100", "rotated100", "staggered100") and grade.get("terminal_checks"):
        result["native"] = native_outcome(row)
    return result


def component_axes(row):
    """Reconstruct covariance axes from the exact saved scored sample tensor.

    This repeats only deterministic evaluator algebra, not its target sampler,
    model evaluation or overall gate decision. Axis signs are arbitrary.
    """
    import torch
    evidence = row["evidence"]
    saved = evidence["saved_observer_outputs"]
    path = Path(evidence["public_update_trace"]["path"]).parent / saved["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
        raise ValueError("Saved scored samples differ")
    observations = torch.load(path, map_location="cpu", weights_only=True)
    if len(observations) != saved["observation_count"]:
        raise ValueError("Saved observation count differs")
    observed = observations[-1]
    spec = evidence["host"]["definition"]
    step = observed["step"]
    if step != row["metrics"]["step"]:
        raise ValueError("Saved sample endpoint step differs")
    start, end = spec.get("scale_start", 1.), spec.get("scale_end", 1.)
    units = start+(end-start)*min(1., step/max(1., spec.get("scale_ramp_end", .6)*spec["steps"]))
    samples = observed["samples"]
    means = torch.tensor(spec["means"])*units
    covariances = torch.tensor(spec["covariances"])*units**2
    assignments = torch.cdist(samples, means).argmin(1)
    components = []
    for index in range(len(means)):
        points = samples[assignments == index]
        centered = points-points.mean(0)
        covariance = centered.T@centered/len(points)
        inverse = torch.linalg.inv(torch.linalg.cholesky(covariances[index]))
        ratios, axes = torch.linalg.eigh(inverse@covariance@inverse.T)
        axis = axes[:, 0]
        if axis[0] < 0:
            axis = -axis
        components.append(dict(index=index, target_mass=spec["masses"][index], count=len(points),
            sample_mean=points.mean(0).tolist(), target_mean=means[index].tolist(),
            covariance=covariance.tolist(), whitened_eigen_ratios=ratios.tolist(),
            minimum_whitened_axis=axis.tolist(), target_covariance=covariances[index].tolist()))
    actual = min(c["whitened_eigen_ratios"][0] for c in components)
    if [c["count"] for c in components] != row["metrics"]["component_counts"]:
        raise ValueError("Saved sample component counts differ")
    expected = row["metrics"]["component_min_eigen_ratio"]
    if abs(actual-expected) > 1e-6:
        raise ValueError("Saved samples do not reproduce the reported component minimum eigenvalue")
    return dict(saved_samples=saved, step=step, components=components,
                minimum_component=min(components, key=lambda c:c["whitened_eigen_ratios"][0])["index"],
                additional_draws=0, additional_forwards=0)


def compare(control, candidate):
    result = {}
    for task in TASKS:
        left, right = control[task], candidate[task]
        if left["status"] == "BLOCKED" or right["status"] == "BLOCKED":
            result[task] = dict(pairing=None, pairing_status="UNAVAILABLE_PREREQUISITE",
                energy=outcome(left), mechanics=outcome(right), replay=trace_summary(right))
            continue
        pairing = {key: left.get(key) == right.get(key)
                   for key in ("initializer", "initialization", "initial_prior", "prior_contract")}
        pairing["sampling_law"] = (left.get("evidence") or {}).get("sampling_law") == (right.get("evidence") or {}).get("sampling_law")
        pairing["optimizer_updates"] = ((left.get("evidence") or {}).get("guards") or {}).get("optimizer_updates") == ((right.get("evidence") or {}).get("guards") or {}).get("optimizer_updates")
        streams_left, streams_right = training_streams(left), training_streams(right)
        pairing["consumed_non_eval_streams"] = None if streams_left is None or streams_right is None else streams_left == streams_right
        a, b = left.get("recipe") or {}, right.get("recipe") or {}
        changed = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        pairing["only_declared_recipe_delta"] = (changed == {"bispf_generator_step"}
            and a.get("bispf_generator_step", "ordinary") == "ordinary"
            and b.get("bispf_generator_step") == "secant_extragradient")
        result[task] = dict(pairing=pairing, pairing_status="OBSERVED", energy=outcome(left), mechanics=outcome(right),
                           replay=trace_summary(right))
        if task == "vector_unequal_mass":
            result[task]["energy"]["saved_component_axes"] = component_axes(left)
            result[task]["mechanics"]["saved_component_axes"] = component_axes(right)
    return result


def compact_receipt(raw, path):
    """Keep aggregate/final evidence in Git; all observation values stay RAW."""
    result = json.loads(raw)
    result["raw_analysis"] = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    result["compact_scope"] = "final metrics, phase/terminal counts, bound summaries and artifact hashes; no per-observation metric dictionaries"
    for task in result["tasks"].values():
        for arm in ("energy", "mechanics"):
            outcome = task[arm]
            outcome.pop("last_five_observations", None)
            summaries = list(outcome["bound_failures_across_observations"])
            if "native" in outcome:
                native = outcome["native"]
                native.pop("initial_observation")
                checks = native.pop("terminal_checks")
                native["terminal_passing_steps"] = [c["step"] for c in checks if c["passed"]]
                summaries += native["accuracy_bound_failures"]
            for bound in summaries:
                steps = bound.pop("failing_steps")
                bound["first_failing_step"] = min(steps, default=None)
                bound["last_failing_step"] = max(steps, default=None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-output", type=Path,
                        help="optional Git-sized receipt without per-observation metric values")
    args = parser.parse_args()
    started = time.process_time()
    a, control, provenance_a = read(args.control, CONTROL)
    b, candidate, provenance_b = read(args.candidate, CANDIDATE)
    if a["source"] != b["source"]:
        raise ValueError("Control and candidate do not share the exact frozen source")
    result = dict(schema=1, qualification_input=False, scope="read_only_mechanics_analysis",
                  additional_updates=0, additional_draws=0, sources=[provenance_a, provenance_b],
                  tasks=compare(control, candidate))
    observed = [r["replay"]["all"] for r in result["tasks"].values() if r["replay"]["status"] == "VERIFIED"]
    alphas = Counter()
    for item in observed:
        alphas.update(item["alpha_counts"])
    n = sum(s["rows"] for s in observed)
    fields = sum(s["field_evaluations"] for s in observed)
    result["summary"] = dict(energy_pass=sum(r["energy"]["gate_status"] == "PASS" for r in result["tasks"].values()),
        mechanics_pass=sum(r["mechanics"]["gate_status"] == "PASS" for r in result["tasks"].values()),
        gained=[t for t,r in result["tasks"].items() if r["energy"]["gate_status"] != "PASS" and r["mechanics"]["gate_status"] == "PASS"],
        lost=[t for t,r in result["tasks"].items() if r["energy"]["gate_status"] == "PASS" and r["mechanics"]["gate_status"] != "PASS"],
        verified_traces=len(observed), observed_updates=n, field_evaluations=fields,
        mean_trials=(fields-n)/n if n else None, alpha_counts=dict(sorted(alphas.items())),
        rejected=sum(s["rejected"] for s in observed), zero_actual_movements=sum(s["zero_actual_movements"] for s in observed),
        protection_changed=sum(s["protection_changed"] for s in observed),
        optimizer_clocks=sum(s["optimizer_clocks"] for s in observed),
        additional_sampling_draws=sum(s["additional_sampling_draws"] for s in observed),
        accepted_bound_violations=sum(s["accepted_bound_violations"] for s in observed))
    result["process_cpu_seconds"] = time.process_time()-started
    errors = [(task, key) for task, row in result["tasks"].items()
              for key, value in (row["pairing"] or {}).items() if value is False]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    args.output.write_bytes(raw)
    if args.compact_output is not None:
        if args.compact_output.resolve() == args.output.resolve():
            raise ValueError("Raw and compact analysis paths must differ")
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(json.dumps(compact_receipt(raw, args.output),
            sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), tasks=len(result["tasks"]), pairing_errors=errors,
                          process_cpu_seconds=result["process_cpu_seconds"])), flush=True)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
