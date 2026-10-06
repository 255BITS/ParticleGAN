"""Read the two certified 10x trajectories; never train or change qualification.

Raw curves, scored tensors and checkpoints remain in the ignored queue. The
compact readout keeps four prefix summaries and their five terminal checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.locked_shared.observation import threshold_margin
from experiments.forge.budget_diagnostics import checkpoint_steps, test_verdict
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.tier1_media import export_attempt
from particlegan.recipes import Recipe, learning_rate_scales
import torch

HERE = Path(__file__).resolve().parent
CAMPAIGN = "bcap-pure-budget10x-v1"
CANDIDATE = "bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9"
OLD_CAMPAIGN = "bcap-dualnorm-pacing-v2"
OLD_COMMIT = "a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be"
OLD_DIGEST = "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226"
OLD_ANALYSIS_SHA = "c5f758adaf8da45aafe547a31ad54cf2bfe96ad662c007b17cb7194bd4b3ed86"
TASKS = {"gaussian1d_acquisition_budget10x_v1": ("gaussian1d_acquisition", 1000,
         "2f3251483c684806a8f54c30aa360eca"),
         "ring16_acquisition_budget10x_v1": ("ring16_acquisition", 400,
         "e7a839160d7a4095990c034557ec1d24")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class Inputs:
    def __init__(self):
        self.refs = {}

    def bind(self, path):
        path = Path(path).resolve()
        self.refs[str(path)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        return path

    def json(self, path):
        return read_json(self.bind(path))

    def tensor(self, path):
        return torch.load(self.bind(path), map_location="cpu", weights_only=True)


def certified(inputs, durable, campaign, candidate, source):
    wrapper = inputs.json(durable / "request.json")
    request, job = wrapper["request"], wrapper["job"]
    result = inputs.json(durable / "result.json")
    certificate = inputs.json(durable / "evidence.json")
    actual_source = request["source"]
    require(certificate["result_hash"] == stable_hash(result), "certificate result hash differs")
    require(certificate["source"] == actual_source and
            certificate["runtime"] == request["runtime"], "source/runtime certificate differs")
    require({k: actual_source[k] for k in ("origin_commit", "digest")} == source and
            stable_hash(actual_source["files"]) == actual_source["digest"], "wrong frozen source")
    require(request["campaign_id"] == campaign and request["protocol"]["seed"] == 0,
            "wrong campaign/seed")
    require(request["candidate"]["id"] == candidate and
            request["candidate_revision"] == result["candidate_revision"], "candidate identity differs")
    require(result["attempt_id"] == durable.name and not result.get("retry_of") and
            not wrapper.get("retry_of"), "attempt identity/retry differs")
    require(len(result["task_results"]) == len(job["task_ids"]) == 1, "unexpected grouped job")
    row = result["task_results"][0]
    require(job["task_ids"] == [row["task_id"]] and stable_hash(job["science"]) ==
            job["compatibility_key"] == row["compatibility_key"], "scientific task binding differs")
    local = Path(certificate["local_artifact_root"])
    require(local.name == durable.name and inputs.json(local / "result.json") == result,
            "local result differs from durable certificate")
    require(inputs.json(local / "request.json") == wrapper, "local request differs from durable envelope")
    terminal = inputs.json(local / "terminal.json")
    require(terminal["attempt_status"] == "completed", "attempt is not completed")
    return request, row, certificate, local


def saved_samples(inputs, row, local):
    descriptor = row["evidence"]["saved_observer_outputs"]
    path = (local / descriptor["path"]).resolve()
    require(path.is_relative_to(local.resolve()), "scored-output path escaped attempt")
    require(file_hash(path) == descriptor["sha256"] and path.stat().st_size == descriptor["bytes"],
            "scored-output bytes differ")
    require(descriptor["sampling_draws_added"] == descriptor["optimizer_updates_added"] == 0,
            "saved-output writer changed training/evaluation")
    records = inputs.tensor(path)
    require(len(records) == descriptor["observation_count"] and
            [p["step"] for p in records] == [p["step"] for p in row["evidence"]["observations"]],
            "scored-output cadence differs")
    require(all(finite_tree(p["samples"]) for p in records), "nonfinite scored tensor")
    return records


def joint_pass(point, requirements):
    return all(type(point.get(key)) in (int, float) and math.isfinite(point[key]) and
               threshold_margin(point[key], op, bound) >= 0 for key, op, bound in requirements)


def stability(curve, requirements):
    """Separate earliest successful windows from the eventual passing suffix."""
    flags = [joint_pass(p, requirements) for p in curve]
    first = next((p["step"] for p, ok in zip(curve, flags) if ok), None)
    first_window = next((i for i in range(4, len(flags)) if all(flags[i - 4:i + 1])), None)
    suffix = len(flags)
    while suffix and flags[suffix - 1]:
        suffix -= 1
    losses = [curve[i]["step"] for i in range(1, len(flags)) if flags[i - 1] and not flags[i]]
    after_confirmation = [p["step"] for i, p in enumerate(curve)
                          if first_window is not None and i > first_window and not flags[i]]
    return {"first_joint_pass_step": first,
            "first_five_pass_window_start_step": curve[first_window - 4]["step"] if first_window is not None else None,
            "first_five_pass_confirmation_step": curve[first_window]["step"] if first_window is not None else None,
            "terminal_passing_suffix": len(flags) - suffix,
            "terminal_suffix_start_step": curve[suffix]["step"] if suffix < len(curve) else None,
            "losses_of_joint_pass": len(losses), "last_loss_of_joint_pass_step": losses[-1] if losses else None,
            "failed_checks_after_first_five_pass_confirmation": len(after_confirmation),
            "last_failure_after_first_five_pass_confirmation_step": after_confirmation[-1] if after_confirmation else None,
            "scope": "Recorded observations only; a first five-pass window need not persist to the endpoint."}


def canonical_rng(value):
    bindings = []
    for binding in value["bindings"].values():
        item = dict(binding)
        item["device"] = "cuda" if item["device"].startswith("cuda:") else item["device"]
        bindings.append(item)
    return {"seed": value["seed"], "version": value["version"],
            "bindings": sorted(bindings, key=stable_hash)}


def compare_prefix(old, new, old_samples, new_samples):
    old_curve, new_curve = old["evidence"]["observations"], new["evidence"]["observations"][:24]
    require(len(old_curve) == len(new_curve) == len(old_samples) == 24, "original prefix length differs")
    require([p["step"] for p in old_curve] == [p["step"] for p in new_curve], "original spacing differs")
    require(all(set(a) == set(b) for a, b in zip(old_curve, new_curve)), "original prefix metric keys differ")
    require(all(a["samples"].shape == b["samples"].shape and a["samples"].dtype == b["samples"].dtype
                for a, b in zip(old_samples, new_samples)), "original sample shape/dtype differs")
    tensor_diffs = [float((a["samples"].double() - b["samples"].double()).abs().max())
                    for a, b in zip(old_samples, new_samples)]
    tensor_equal = [torch.equal(a["samples"].contiguous().view(torch.uint8),
                               b["samples"].contiguous().view(torch.uint8))
                    for a, b in zip(old_samples, new_samples)]
    metric_diffs = [abs(float(a[key]) - float(b[key])) for a, b in zip(old_curve, new_curve)
                    for key in a if type(a[key]) in (int, float) and type(b.get(key)) in (int, float)]
    initial = {name: {"old_sha256": item.get("initial_state_sha256"),
               "new_sha256": new["initialization"].get(name, {}).get("initial_state_sha256"),
               "exact_equal": item.get("initial_state_sha256") is not None and
                   item.get("initial_state_sha256") == new["initialization"].get(name, {}).get("initial_state_sha256")}
               for name, item in old["initialization"].items()}
    old_rng, new_rng = canonical_rng(old["rng"]), canonical_rng(new["rng"])
    return {"observations_compared": 24, "original_saved_metrics_exact_equal": old_curve == new_curve,
            "maximum_absolute_scalar_metric_difference": max(metric_diffs, default=0.),
            "saved_sample_tensors_bitwise_equal": all(tensor_equal),
            "maximum_absolute_sample_tensor_difference": max(tensor_diffs),
            "first_sample_difference_step": next((p["step"] for p, same in zip(old_curve, tensor_equal) if not same), None),
            "initial_component_state_hashes": initial,
            "named_stream_starts_equal_ignoring_cuda_index": old_rng == new_rng,
            "old_rng_start_digest": stable_hash(old_rng), "new_rng_start_digest": stable_hash(new_rng),
            "unavailable_comparisons": ["old final model/optimizer/consumed-stream tensors",
                                        "direct consumed-batch digest"],
            "scope": "Exact retained scored outputs and emitted initialization/stream starts; no old resumable state exists."}


def same_task_law(original, new):
    a, b = original["execution"], new["execution"]
    keys = ("adapter", "requires_capabilities")
    require(all(original.get(k) == new.get(k) for k in keys), "adapter/capabilities changed")
    for key in ("host", "host_source", "initializer", "prior", "protocol"):
        require(a.get(key) == b.get(key), f"task execution.{key} changed")
    require({k: v for k, v in a["host_definition"].items() if k != "steps"} ==
            {k: v for k, v in b["host_definition"].items() if k != "steps"}, "architecture/target law changed")
    for key in ("thresholds", "sample_evaluator", "sampling_law", "sampling_contract_version",
                "scoring_weights", "eval_output_noise", "minimum_stable_checks", "gate_policy"):
        require(original["evaluation"].get(key) == new["evaluation"].get(key), f"evaluation.{key} changed")
    scorer = new["evaluation"]["sample_evaluator"].split(":")[0].replace(".", "/") + ".py"
    require(original["evaluation"]["sources"][scorer] == new["evaluation"]["sources"][scorer],
            "scorer source changed")


def verify_state_rates(inputs, row, local, steps):
    state = inputs.tensor(local / "state.pt")
    require(finite_tree(state), "nonfinite final checkpoint")
    trainer = state["trainer"]
    # Torch retains tuple-valued recipe fields; JSON stores the same fields as lists.
    require(trainer["completed_steps"] == steps and
            Recipe(**state["recipe"]).to_dict() == Recipe(**row["recipe"]).to_dict(),
            "final state recipe/consumed updates differ")
    require(state["initialization"] == row["initialization"] and state["prior"] == row["prior"],
            "final state initialization/prior binding differs")
    require(state["streams"]["manifest"] == row["rng"] and
            set(state["streams"]["states"]) == set(row["rng"]["bindings"]), "consumed named streams missing")
    recipe = Recipe(**row["recipe"])
    require((recipe.optimizer_family, recipe.optimizer_momentum, recipe.lr, recipe.d_lr_mult,
             recipe.prior_lr_mult, recipe.eps) == ("dualnorm", 0., .012, 1.5, 2.5, 1e-8), "winning optimizer changed")
    expected = {"generator": .012, "encoder": .012, "critic": .018, "prior": .03}
    groups = []
    for optimizer in trainer["optimizers"]:
        for group in optimizer["param_groups"]:
            role = group["role"]
            require(role in expected and math.isclose(group["lr"], expected[role], rel_tol=0, abs_tol=1e-15),
                    "observed final step size changed")
            groups.append({"role": role, "lr": group["lr"]})
    require({g["role"] for g in groups} == {"generator", "critic", "prior"}, "unexpected optimizer player set")
    require(all(learning_rate_scales(step, recipe) == (1., 1.) for step in range(steps + 1)),
            "effective schedule is not constant")
    return {"final_observed_optimizer_groups": groups, "constant_schedule_steps_checked": steps + 1,
            "constant_schedule_scales": [1., 1.], "checkpoint_sha256": file_hash(local / "state.pt"),
            "consumed_named_streams_retained": len(state["streams"]["states"]),
            "scope": "Actual final optimizer group rates plus deterministic schedule law at every consumed step; per-step rate values were not logged. The reader records the retained checkpoint SHA; the original adapter emitted no separate checkpoint descriptor."}


def read_trace(inputs, row, local, expected):
    descriptor = row["evidence"].get("optimizer_diagnostics")
    if descriptor is None:
        return {"available": False, "reason": "optional optimizer diagnostics were not enabled"}
    path = (local / descriptor["path"]).resolve()
    require(path.is_relative_to(local.resolve()) and file_hash(path) == descriptor["sha256"], "trace bytes differ")
    require(descriptor["qualification_input"] is False and descriptor["sampling_draws_added"] ==
            descriptor["optimizer_updates_added"] == 0, "observer changed training/scoring")
    rows = [json.loads(line) for line in inputs.bind(path).read_text().splitlines() if line.strip()]
    require(len(rows) == descriptor["rows"] and finite_tree(rows), "trace length/nonfinite values")
    require(all([r["step"] for r in rows if r["optimizer"] == role] == expected for role in ("G", "D")),
            "optimizer trace cadence differs")
    priors = [r["prior"] for r in rows if "prior" in r]
    require(len(priors) == len(expected) and all(p["row_support_kind"] == "sampled_indices" and
            p["max_outside_support_row_displacement"] == 0 for p in priors), "unsampled particle rows moved")
    return {"available": True, "rows": len(rows), "all_finite": True,
            "maximum_unsampled_row_displacement": 0.,
            "input_gradient_scope": "Vector GANTrainer D training phase only, data x real/fake pair.",
            "spectral_proxy_scope": "Matrix-weight product; excludes Fourier/input maps and nonlinearities."}


def prefixes(task, row, verdict):
    curve, requirements = row["evidence"]["observations"], task["evaluation"]["thresholds"]
    output = {}
    for label, snapshot in verdict["prefix_snapshots"].items():
        prefix = [p for p in curve if p["step"] <= snapshot["step_budget"]]
        output[label] = {"step_budget": snapshot["step_budget"], "status": snapshot["status"],
            "convergence": snapshot["convergence"], "endpoint_gate_cells": snapshot["metrics"],
            "stability": stability(prefix, requirements),
            "last_five": [{"step": p["step"], "joint_pass": joint_pass(p, requirements),
                "metrics": {key: p[key] for key, _, _ in requirements}} for p in prefix[-5:]]}
    return output


def late_window(task, row):
    points = row["evidence"]["observations"][-60:]
    requirements = task["evaluation"]["thresholds"]
    return {"observations": len(points), "start_step": points[0]["step"],
            "end_step": points[-1]["step"],
            "joint_passing_observations": sum(joint_pass(p, requirements) for p in points),
            "metrics": {key: {
                "failing_observations": sum(any(
                    threshold_margin(p[key], op, bound) < 0
                    for name, op, bound in requirements if name == key) for p in points),
                "minimum": min(p[key] for p in points),
                "maximum": max(p[key] for p in points)}
                for key in dict.fromkeys(name for name, _, _ in requirements)}}


def draw_curves(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = (("cdf_ks", "mean_error_sigma", "std_ratio"),
               ("component_covariance_error", "hq", "component_min_eigen_ratio"))
    for index, (task_id, (_, base, _)) in enumerate(TASKS.items()):
        task, row = rows[task_id]
        curve = row["evidence"]["observations"]
        for ax, metric in zip(axes[index], metrics[index]):
            ax.plot([p["step"] / base for p in curve], [p[metric] for p in curve], lw=1.2)
            for key, _, bound in task["evaluation"]["thresholds"]:
                if key == metric:
                    ax.axhline(bound, color="black", ls="--", lw=.8)
            for factor in (1, 2, 4, 10):
                ax.axvline(factor, color="gray", lw=.6, alpha=.4)
            ax.set(title=metric, xlabel="Original update budgets", xlim=(0, 10))
            if metric == "component_covariance_error":
                ax.set_yscale("log")
        axes[index, 0].set_ylabel("Gaussian" if index == 0 else "Ring")
    figure.suptitle("Same whole BCAP-pure recipe · seed 0 · unchanged metric bounds\nIndividual curves are diagnostics; joint last-five gates determine each prefix")
    figure.savefig(output, dpi=150, metadata={"Software": "ParticleGAN budget readout"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=ROOT / "runs/forge" / (CAMPAIGN + "-queue"))
    parser.add_argument("--original-queue-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    inputs = Inputs()
    freeze = inputs.json(HERE / "freeze.json")
    require(freeze["input_digest"] == stable_hash({k: v for k, v in freeze.items() if k != "input_digest"}), "freeze digest differs")
    require(freeze["study"] == CAMPAIGN and freeze["candidate_id"] == CANDIDATE and
            freeze["tasks"] == {k: v[1] * 10 for k, v in TASKS.items()} and
            freeze["automatic_retries"] == 0 and freeze["qualification_input"] is False,
            "unexpected frozen experiment scope")
    analysis_path = args.original_root / "reports/forge/dualnorm-pacing-v2/analysis.json"
    require(file_hash(analysis_path) == OLD_ANALYSIS_SHA, "original selected analysis changed")
    original_analysis = inputs.json(analysis_path)
    original_configs = [c for c in original_analysis["configurations"] if c["candidate_id"] == CANDIDATE]
    require(len(original_configs) == 1 and original_configs[0]["required_passes"] == 4 and
            original_analysis["selection"]["required_pass_count"] == 4, "original winner identity/count differs")
    original_config = original_configs[0]
    attempts = sorted(p.parent for p in (args.queue_root / CAMPAIGN).glob("*/result.json"))
    require(len(attempts) == 2, "exactly two completed diagnostic attempts are required")
    readout, curves, media, seen = {}, {}, [], set()
    for local in attempts:
        request, row, cert, retained = certified(inputs, ROOT / "reports/forge/attempts" / local.name,
                                                CAMPAIGN, CANDIDATE, freeze["source"])
        task_id = row["task_id"]
        require(task_id in TASKS and task_id not in seen and retained.resolve() == local.resolve(), "wrong/duplicate task")
        seen.add(task_id)
        original_id, base, old_attempt = TASKS[task_id]
        require(original_config["tasks"][original_id]["attempt_id"] == old_attempt,
                "original selected attempt differs")
        task = request["tasks"][task_id]
        require(request["request_id"] == freeze["request_id"] and task["execution"]["steps"] == base * 10 and
                task["execution"]["original_schedule_horizon"] == base and
                task["execution"]["produces_state"] is True, "frozen request/task allowance differs")
        snapshot_root = Path(request["source"]["snapshot_path"])
        require(all(file_hash(snapshot_root / name) == digest for name, digest in request["source"]["files"].items()),
                "frozen source snapshot bytes differ")
        expected = checkpoint_steps(task)
        verdict = test_verdict(task, row["evidence"])
        require(verdict == row["evaluator_result"] and verdict["status"] == row["gate_status"], "certified verdict differs from recomputation")
        require(verdict["convergence"]["complete"] and row["cost"]["completed_steps"] == base * 10,
                "incomplete consumed update/observation allowance")
        guards = row["evidence"]["guards"]
        require(guards["all_finite"] is True and guards["hooks_exercised"] is True and
                guards["unintended_rng_deviations"] == 0 and
                set(guards["optimizer_updates"]) == {"generator", "discriminator", "prior"} and
                all(n == base * 10 for n in guards["optimizer_updates"].values()), "finite/update/RNG guard failed")
        new_samples = saved_samples(inputs, row, local)
        old_request, old_row, old_cert, old_local = certified(inputs,
            args.original_root / "reports/forge/attempts" / old_attempt, OLD_CAMPAIGN, CANDIDATE,
            {"origin_commit": OLD_COMMIT, "digest": OLD_DIGEST})
        original_binding = original_config["tasks"][original_id]
        require(old_cert["result_hash"] == original_binding["result_hash"] and
                old_row["compatibility_key"] == original_binding["compatibility_key"] and
                old_row["gate_status"] == original_binding["gate_status"], "original analysis receipt binding differs")
        require(old_local.resolve() == (args.original_queue_root / OLD_CAMPAIGN / old_attempt).resolve(), "wrong original queue")
        same_task_law(old_request["tasks"][original_id], task)
        require(Recipe(**old_row["recipe"]).to_dict() == Recipe(**row["recipe"]).to_dict(), "whole effective recipe changed")
        parity = compare_prefix(old_row, row, saved_samples(inputs, old_row, old_local), new_samples)
        summary = {"attempt_id": local.name, "result_hash": cert["result_hash"], "gate_status": row["gate_status"],
            "completed_steps": base * 10, "thresholds": task["evaluation"]["thresholds"],
            "prefixes": prefixes(task, row, verdict), "last_60_observations": late_window(task, row),
            "guards": {k: guards[k] for k in
                ("all_finite", "hooks_exercised", "optimizer_updates", "unintended_rng_deviations")},
            "constant_rate_and_final_state_audit": verify_state_rates(inputs, row, local, base * 10),
            "optimizer_diagnostics": read_trace(inputs, row, local, expected),
            "original_prefix_comparison": {"attempt_id": old_attempt, "result_hash": old_cert["result_hash"], **parity},
            "recipe_sha256": stable_hash(row["recipe"]), "cost": row["cost"]}
        readout[task_id], curves[task_id] = summary, (task, row)
        media.extend(export_attempt(ROOT / "reports/forge/attempts" / local.name, args.output_dir / "media"))
    require(seen == set(TASKS), "required diagnostic tasks missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_curves(curves, args.output_dir / "curves.png")
    output = {"schema_version": 1, "campaign": CAMPAIGN, "qualification_input": False,
        "evidence_scope": "research_diagnostic", "candidate_id": CANDIDATE,
        "source": freeze["source"], "protocol_seed": 0, "tasks": readout,
        "diagnostic_counts": dict(Counter(r["gate_status"] for r in readout.values())),
        "ordinary_tier1": {"unchanged": True, "selected_passes": 4, "required_tasks": 6,
                           "source_commit": OLD_COMMIT, "source_digest": OLD_DIGEST,
                           "analysis_sha256": OLD_ANALYSIS_SHA},
        "media": media, "curve_plot": {"path": "curves.png", "sha256": file_hash(args.output_dir / "curves.png")},
        "input_artifacts": inputs.refs, "analyzer_sha256": file_hash(Path(__file__)),
        "software_sources": {name: file_hash(ROOT / name) for name in
            ("experiments/forge/budget_diagnostics.py", "experiments/forge/tier1_media.py",
             "benchmarks/locked_shared/observation.py", "particlegan/recipes.py")},
        "limits": ["Two uninterrupted seed-0 trajectories, not independent prefix attempts or cross-seed evidence.",
                   "Original qualification receipts and the single ordinary leaderboard are not rewritten.",
                   "No old resumable model/optimizer/end-stream tensors exist; retained-prefix comparisons cannot prove unavailable-state equality."]}
    atomic_json(args.output_dir / "results.json", output)
    print(json.dumps({"results": str(args.output_dir / "results.json"), "diagnostic_counts": output["diagnostic_counts"],
                      "prefix_sample_parity": {k: v["original_prefix_comparison"]["saved_sample_tensors_bitwise_equal"]
                                               for k, v in readout.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
