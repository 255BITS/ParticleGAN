"""Verify and summarize the finished, frozen BCAP pacing study without training.

Read certified original attempts and saved diagnostics. This reader writes no
selection, task grade, recipe, or leaderboard. An optional immutable archive can
supply exact durable/attempts and attempts members after local logs are moved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from benchmarks.locked_shared.observation import sustained, threshold_margin
from benchmarks.transfer_suite.protocol import test_verdict
from experiments.forge.clockfree import verify_probe

CAMPAIGN = "bcap-dualnorm-pacing-v2"
COMMIT = "a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be"
DIGEST = "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226"
REQUIRED = ("gaussian1d_acquisition", "two_pole", "unused_token_hold", "ae_gan_hold",
            "ring16_acquisition", "five_word_joint_acquisition")
CLOCK = "clockfree_audit_measurement_v1"
TRACED = (REQUIRED[0], REQUIRED[4], REQUIRED[5])
OPT_FIELDS = {"lr", "d_lr_mult", "prior_lr_mult", "optimizer_momentum"}
REPORT = Path("reports/forge/dualnorm-pacing-v2")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# These pure readers already distinguish missing aggregate initialization hashes
# from initializer/seed metadata. Bind their bytes in the output provenance.
old = module(ROOT / "reports/forge/dualnorm-tier1/analyze.py", "pacing_previous_reader")
driver = module(ROOT / REPORT / "run.py", "pacing_frozen_driver")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_digest(value, field="input_digest"):
    require(value.get(field) == stable_hash({k: v for k, v in value.items() if k != field}),
            f"invalid {field}")


def requirements_pass(point, requirements):
    return all(isinstance(point.get(key), (int, float)) and not isinstance(point.get(key), bool)
               and math.isfinite(point[key]) and threshold_margin(point[key], op, bound) >= 0
               for key, op, bound in requirements)


def input_gradient_scope(task):
    if task in REQUIRED[:1] + REQUIRED[4:5]:
        return {"usable": True, "coordinates": "data x", "source": "particlegan/training.py:_step",
                "reason": "GANTrainer sets D.eval() throughout the generator phase; the observer captures the current D real/fake pair in train mode."}
    return {"usable": False, "coordinates": "joint word-probability/latent input" if task == REQUIRED[5] else None,
            "source": "benchmarks/toy_audit/api_images.py:WordFixture.step; experiments/forge/optimizer_diagnostics.py:_capture_inputs",
            "reason": "Word D stays in train mode during G. Before a checkpoint, previous-G fake/real forwards fill the next-D input cache; real/fake labels cannot be trusted. Other behavioral hosts have no optimizer diagnostic trace."}


def clean_trace(rows, task, expected_steps):
    retained, steps = [], defaultdict(list)
    for original in rows:
        role = original.get("optimizer")
        require(role in {"G", "D"}, "unknown diagnostic optimizer")
        steps[role].append(original["step"])
        row = deepcopy(original)
        if not input_gradient_scope(task)["usable"]:
            for key in list(row):
                if key.startswith("critic_input_gradient_"):
                    del row[key]
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        require(finite(row), "nonfinite retained diagnostic value")
        for player, aggregate in row["players"].items():
            layers = [x for x in row["layers"] if x["player"] == player]
            require(math.isclose(aggregate["relative_update_sum"], sum(x["relative_update"] for x in layers), rel_tol=1e-12, abs_tol=1e-12), "relative-speed reduction mismatch")
        if "critic_log_spectral_product" in row:
            require(math.isclose(row["critic_log_spectral_product"], sum(math.log(max(x, 1e-30)) for x in row["critic_weight_spectral_norms"]), rel_tol=1e-12, abs_tol=1e-12), "spectral log reduction mismatch")
        if "prior" in row:
            require(row["prior"]["row_support_kind"] == "sampled_indices", "full dualnorm requires actual sampled-row support")
            require(row["prior"]["max_outside_support_row_displacement"] == 0., "unsampled prior rows moved")
        retained.append(row)
    require(dict(steps) == {"D": expected_steps, "G": expected_steps}, "diagnostic cadence differs from task")
    return retained


class Artifacts:
    def __init__(self, root, queue, archive=None):
        self.root, self.queue = root, queue
        self.archive = tarfile.open(archive, "r:gz") if archive else None
        self.refs = {}

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, attempt, name, durable=False):
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "artifact path must stay within its attempt")
        local = (self.root / "reports/forge/attempts" / attempt if durable else self.queue / CAMPAIGN / attempt) / name
        member = ("durable/" if durable else "campaign/" + CAMPAIGN + "/") + attempt + "/" + name
        if local.is_file():
            payload = local.read_bytes()
        else:
            require(self.archive is not None, f"original artifact missing: {member}")
            with self.archive.extractfile(member) as stream:
                payload = stream.read()
        self.refs[member] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        return payload

    def json(self, attempt, name, durable=False):
        return json.loads(self.read(attempt, name, durable))

    def receipt(self, trial, task, cohort):
        attempt = task["attempt_id"]
        wrapped = self.json(attempt, "request.json", True)
        request, job = wrapped["request"], wrapped["job"]
        result, cert = self.json(attempt, "result.json", True), self.json(attempt, "evidence.json", True)
        source = request["source"]
        require(cert["result_hash"] == stable_hash(result), f"certificate hash mismatch: {attempt}")
        require(cert["source"] == source and cert["runtime"] == request["runtime"], "certificate source/runtime mismatch")
        require(source["origin_commit"] == COMMIT and source["digest"] == DIGEST and stable_hash(source["files"]) == DIGEST, "wrong scientific source")
        require(request["campaign_id"] == CAMPAIGN and request["protocol"]["seed"] == 0, "wrong campaign/seed")
        require(request["candidate"]["id"] == trial["candidate_id"] and request["candidate_revision"] == result["candidate_revision"] == trial["candidate_revision"], "wrong candidate identity")
        require(result["attempt_id"] == attempt and request["request_id"] == trial["request_id"], "wrong attempt/request identity")
        require(not result.get("retry_of") and not wrapped.get("retry_of"), "unexpected retry")
        actual_cohort = {"source_digest": source["digest"], "runtime_cohort": {k: request[k] for k in ("runtime", "execution_backend", "compute_profiles")},
                         "protocol_hash": stable_hash(request["protocol"]), "policy_fingerprint": request["policy_fingerprint"]}
        require(actual_cohort == cohort, "mixed execution cohort")
        require(job["task_ids"] == [task["task"]] and job["compatibility_key"] == task["compatibility_key"], "wrong job task/key")
        require(job["budget_seconds"] == task["budget_seconds"], "changed timeout allowance")
        require(stable_hash(job["science"]) == task["compatibility_key"], "science compatibility hash mismatch")
        require(len(result["task_results"]) == 1, "unexpected combined task result")
        row = result["task_results"][0]
        require(row["task_id"] == task["task"] and row["compatibility_key"] == task["compatibility_key"] and row["gate_status"] == task["gate_status"], "compact task differs from original result")
        scalar_metrics = {k:v for k,v in row["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        require(scalar_metrics == task["metrics"], "compact scalar metrics differ from certified original")
        terminal = self.json(attempt, "terminal.json")
        require(terminal["attempt_status"] == "completed", "non-completed execution")
        require(self.json(attempt, "result.json") == result, "durable/local results differ")
        # For task caps use reported completed updates, not the public recipe's
        # unrelated default total_steps. Fixed direct fixtures have task-owned priors.
        applied = row.get("applied", {})
        recipe = row.get("recipe", applied.get("recipe", {}))
        for field in OPT_FIELDS | {"optimizer_family"}:
            default = 0. if field == "optimizer_momentum" else None
            require(recipe.get(field, default) == trial["resolved_recipe"].get(field, default), f"applied optimizer delta mismatch: {field}")
        return request, result, cert, row, terminal

    def trace(self, attempt, receipt, task, steps):
        require(receipt["qualification_input"] is False and receipt["sampling_draws_added"] == receipt["optimizer_updates_added"] == 0, "diagnostics modified execution/grading")
        payload = self.read(attempt, receipt["path"])
        require(hashlib.sha256(payload).hexdigest() == receipt["sha256"], "trace SHA mismatch")
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        require(len(rows) == receipt["rows"], "trace row-count mismatch")
        return clean_trace(rows, task, steps)

    def clock_probe(self, attempt, declaration, evidence):
        # Stage only the two certified proof artifacts, never extract a tar's
        # arbitrary paths. Their manifest hashes are checked again by verify_probe.
        original = Path(evidence["artifact_root"])
        require(original.name == "clockfree-proof" and original.parent.name == attempt, "clock proof belongs to a different attempt")
        with TemporaryDirectory(prefix="bcap-pacing-clock-proof-") as directory:
            for name in evidence["artifact_manifest"]["files"]:
                require(Path(name).name == name, "clock manifest path must be a basename")
                Path(directory, name).write_bytes(self.read(attempt, "clockfree-proof/" + name))
            return verify_probe(declaration, {**evidence, "artifact_root":directory})


def task_summary(request, row, task, clock_proof=None):
    declaration = request["tasks"][task["task"]]
    evidence = row["evidence"]
    if task["task"] == CLOCK:
        comparisons, audit = clock_proof if clock_proof is not None else verify_probe(declaration, evidence)
        require(len(comparisons) == 4 and all(c["reference_sha256"] == c["changed_sha256"] for c in comparisons), "clock parity failure")
        require(audit["unexplained_clock_dependencies"] == [], "unexplained clock source dependency")
        return {"gate_status": task["gate_status"], "parity_comparisons": len(comparisons),
                "saved_state_comparisons_recomputed": True, "source_audit": audit}
    spec = {"steps": declaration["execution"]["steps"], "thresholds": declaration["evaluation"]["thresholds"]}
    measured = {"observations": evidence["observations"], "live": evidence["live"]}
    verdict = test_verdict(spec, measured)
    require(verdict == row["evaluator_result"], "recomputed sustained verdict differs from original")
    require(verdict["status"] == task["gate_status"], "recomputed gate differs from compact table")
    guard = evidence["guards"]
    if "completed_steps" in row["cost"]:
        require(row["cost"]["completed_steps"] == spec["steps"], "task update cap mismatch")
    require(guard["optimizer_updates"] and all(count == spec["steps"] for count in guard["optimizer_updates"].values()), "role update cap mismatch")
    require(guard["all_finite"] is True and guard["hooks_exercised"] is True and guard["unintended_rng_deviations"] == 0, "training guard failure")
    for audit in evidence.get("rng_audits", row.get("applied", {}).get("rng_audits", [])):
        require(audit["unintended_rng_deviations"] == 0, "consumed stream audit failed")
    cells = verdict["metrics"]
    suffix = verdict["convergence"]["passing_suffix"]
    last = [{"step": point["step"], "joint_pass": requirements_pass(point, spec["thresholds"]),
             "metrics": {key: point[key] for key, _, _ in spec["thresholds"]}} for point in evidence["observations"][-5:]]
    output = {"gate_status": task["gate_status"], "metrics": deepcopy(task["metrics"]), "gate_cells": cells,
              "convergence": verdict["convergence"], "terminal_conjuncts_pass": all(c["status"] == "PASS" for c in cells),
              "failed_terminal_conjuncts": [c["metric"] for c in cells if c["status"] != "PASS"],
              "terminal_suffix_required": 5, "suffix_shortfall": max(0, 5 - suffix), "last_five": last}
    if task["task"] == REQUIRED[4]:
        live = evidence["live"]
        output["ring_components"] = {key: live[key] for key in ("component_covariance_errors", "component_core_covariance_errors", "component_spill", "component_counts", "hq_component_counts", "component_core_eigen_ratios")}
        output["ring_covariance_scope"] = "Mean per-component relative Frobenius covariance error using all nearest-assigned samples; core-only errors are separate diagnostics, never substitute gate inputs."
    return output


def trace_summary(rows, task):
    summary = old.trace_summary(rows)
    summary["input_gradient_scope"] = input_gradient_scope(task)
    if summary["input_gradient_scope"]["usable"]:
        summary["critic_input_gradient"] = {label: old.curve_stats([r["critic_input_gradient_mean_" + label] for r in rows if r["optimizer"] == "D"])
                                             for label in ("real", "fake")}
    # Zero biases make endpoint/first ratios a poor growth test. Matrix growth is
    # still descriptive, relative to the first logged checkpoint, never infinity.
    summary["matrix_growth_flags"] = [dict(optimizer=x["optimizer"], index=x["index"], final_over_first=x["final_over_first"])
                                      for x in summary["layer_weight_norms"] if len(x["shape"]) == 2 and x["growth_ge_10x"]]
    sampled = [r for r in rows if "prior" in r]
    summary["particle_support"] = {
        "unique_sampled_rows":old.curve_stats([r["prior"]["support_rows"] for r in sampled]),
        "checks_with_unsampled_rows":sum(r["prior"]["support_rows"] < next(x["shape"][0] for x in r["layers"] if x["player"] == "prior") for r in sampled),
        "scope":"Word table has five rows and usually every row is sampled; outside-support displacement is then vacuously zero. Gaussian/ring retain unsampled rows at observed steps."}
    return summary


def settings(trial):
    r = trial["resolved_recipe"]
    return {"eta_G_E": r["lr"], "eta_D": r["lr"] * r["d_lr_mult"], "D_G_ratio": r["d_lr_mult"],
            "eta_prior": r["lr"] * r["prior_lr_mult"], "prior_lr_mult": r["prior_lr_mult"], "momentum_networks": r["optimizer_momentum"], "momentum_prior": 0.}


def word_stream_end(artifacts, attempt, row, trial):
    """Read the host's retained state; label its post-run digest separately.

    This optional raw checkpoint was not an evaluator input or an artifact hash
    in the original certificate. Its bytes become bound by this readout/archive.
    """
    import torch
    payload = artifacts.read(attempt, "state.pt")
    state = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    require(state["streams"]["manifest"] == row["rng"], "retained word stream manifest differs from receipt")
    recipe = state["fixture"]["recipe"]
    for key in OPT_FIELDS | {"optimizer_family"}:
        require(recipe.get(key, 0. if key == "optimizer_momentum" else None) == trial["resolved_recipe"][key], "retained word optimizer differs from recipe")
    hashes = {}
    for key, value in state["streams"]["states"].items():
        binding = state["streams"]["manifest"]["bindings"][key]
        device = "cuda" if binding["device"].startswith("cuda:") else binding["device"]
        canonical = json.dumps([binding[k] for k in ("family", "component", "purpose")] + [device])
        hashes[canonical] = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
    return {"candidate_id":trial["candidate_id"], "attempt_id":attempt,
            "state_file_sha256":hashlib.sha256(payload).hexdigest(), "streams":hashes}


def plot(trials, traces, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": CAMPAIGN})
    colors = ["#0072b2", "#d55e00", "#009e73"]
    figures = []
    for task in TRACED:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        axspeed, axspectrum, axgrad, axd, axg, axprior = axes.flat
        for trial, color in zip(trials, colors):
            rows = traces[(trial["candidate_id"], task)]
            pace = settings(trial)
            label = f"ηG={pace['eta_G_E']:g}, μ={pace['momentum_networks']:g} ({trial['candidate_id'].split('--')[1][:8]})"
            drows = [r for r in rows if r["optimizer"] == "D"]
            grows = [r for r in rows if r["optimizer"] == "G"]
            for player, role, linestyle in (("D", drows, "-"), ("G", grows, "--")):
                axspeed.plot([r["step"] for r in role], [r["players"][player]["relative_update_sum"] for r in role], color=color, linestyle=linestyle, label=label + " " + player)
            axspectrum.plot([r["step"] for r in drows], [r["critic_log_spectral_product"] for r in drows], color=color, label=label)
            if input_gradient_scope(task)["usable"]:
                for kind, style in (("real", "-"), ("fake", "--")):
                    axgrad.plot([r["step"] for r in drows], [r["critic_input_gradient_mean_" + kind] for r in drows], color=color, linestyle=style, label=label + " " + kind)
            for axis, role in ((axd, drows), (axg, grows)):
                for layer in role[0]["layers"]:
                    if layer["player"] == "prior":
                        continue
                    idx = layer["index"]
                    style = "-" if len(layer["shape"]) == 2 else ":"
                    axis.plot([r["step"] for r in role], [next(x["weight_norm"] for x in r["layers"] if x["index"] == idx) for r in role], color=color, linestyle=style, alpha=.65)
            axprior.plot([r["step"] for r in grows], [r["prior"]["mean_support_row_displacement"] for r in grows], color=color, label=label)
            axprior.plot([r["step"] for r in grows], [r["prior"]["max_outside_support_row_displacement"] for r in grows], color=color, linestyle=":")
        axspeed.set_title("Relative update speed: sum over parameter tensors")
        axspeed.set_yscale("log")
        axspeed.legend(fontsize=7)
        axspectrum.set_title("D log spectral product (matrix weights only)")
        axspectrum.legend(fontsize=7)
        axgrad.set_title("Post-D input gradient norm: real / fake")
        if input_gradient_scope(task)["usable"]:
            axgrad.legend(fontsize=7)
        else:
            axgrad.text(.5, .5, "EXCLUDED: word phase/label ambiguity\nSaved weight/update/row curves remain usable", ha="center", va="center", transform=axgrad.transAxes)
        axd.set_title("Every D tensor: weight norm (vectors dotted)")
        axg.set_title("Every G/E tensor: weight norm (vectors dotted)")
        axprior.set_title("Prior sampled displacement; outside rows dotted")
        axprior.legend(fontsize=7)
        for axis in axes.flat:
            axis.set_xlabel("Completed updates")
            axis.grid(alpha=.2)
        fig.suptitle(task + " — three complete recipes chosen by required PASS / configuration hash")
        path = output / f"diagnostics-{task}.svg"
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)
        figures.append({"path": str(REPORT / path.name), "sha256": file_hash(path), "bytes": path.stat().st_size})
    return figures


def analyze(root, queue, output, archive=None, figures=True):
    results = read_json(root / REPORT / "results.json")
    frozen = read_json(queue / "frozen-study.json")
    state = read_json(queue / "driver-state.json")
    terminal = read_json(queue / "terminal-summary.json")
    clock = read_json(queue / "execution-clock.json")
    checked_digest(frozen)
    checked_digest(clock)
    require(results["source"] == {"commit": COMMIT, "digest": DIGEST} and frozen["source"]["origin_commit"] == COMMIT and frozen["source"]["digest"] == DIGEST, "unexpected frozen source")
    require(stable_hash(frozen["source"]["files"]) == DIGEST, "source manifest hash mismatch")
    require(results["study_binding_sha256"] == state["study_binding_sha256"] == clock["study_binding_sha256"] == frozen["input_digest"], "study binding mismatch")
    require(state["execution_clock_sha256"] == clock["input_digest"], "execution clock binding mismatch")
    for name, digest in frozen["inputs_sha256"].items():
        require(file_hash(root / name) == digest, f"frozen study input changed: {name}")
    snapshot = queue / "snapshots" / DIGEST
    for name, digest in frozen["source"]["files"].items():
        require(file_hash(snapshot / name) == digest, f"executed snapshot file changed: {name}")
    # The evaluator used below must be the exact executed evaluator; broader
    # reporting edits may change HEAD without changing these source bytes.
    for name in ("benchmarks/transfer_suite/protocol.py", "benchmarks/locked_shared/observation.py", "benchmarks/locked_shared/baseline.py", "experiments/forge/clockfree.py"):
        require(file_hash(root / name) == frozen["source"]["files"][name], "current verifier differs from frozen evaluator")
    require(results["status"] == state["status"] == terminal["status"] == "completed" and results["exit_code"] == terminal["exit_code"] == 0, "study not successfully terminal")
    require(results["running_workers"] == terminal["running_workers"] == 0 and results["all_attempts_supervised_finished"] is True and terminal["all_supervised_attempts_finished"] is True, "workers not quiescent")
    require(results["automatic_retries"] == 0 and results["admitted_recipes"] == 25 and len(results["trials"]) == 25, "wrong admitted roster")
    require(results["elapsed_seconds"] <= 43200 and results["accounting"]["spent_seconds"] <= 63000 and results["accounting"]["reserved_seconds"] == 0, "budget or reservation failure")
    contract, stage_a = driver.load_inputs(root)
    trials = results["trials"]
    require(driver.whole_selection(trials) == results["selection"], "global PASS/hash selection differs")
    control = next(t for t in trials if t["candidate_id"] == results["control_candidate_id"])
    stages, pool, stage_by_id = [], [], {}
    for index, (record, count) in enumerate(zip(results["stages"], (20, 3, 2))):
        study_id = record["spec"]["id"]
        searchpath = root / "reports/forge/configuration-search" / (study_id + ".json")
        search = read_json(searchpath)
        checked_digest(search)
        current = [t for t in trials if t["candidate_id"] in {x["candidate_id"] for x in search["trials"]}]
        require(len(current) == count and record["admitted"] is True and all(driver.execution_complete(t) for t in current), "stage completion/roster mismatch")
        selection = driver.whole_selection(current)
        require(selection == record["selection"] and {k:v for k,v in selection.items() if k != "selected_evidence_valid"} == search["selection"], "stage selection differs")
        if index:
            require(driver.activation(record["stage"], pool, control, contract) == record["decision"], "conditional activation changed")
        frozen_branch = next(x for x in frozen["branches"] if x["study_id"] == study_id)
        for t in current:
            declared = next(x for x in frozen_branch["trials"] if x["candidate_id"] == t["candidate_id"])
            require(declared["candidate_revision"] == t["candidate_revision"] and declared["configuration_id"] == t["configuration_id"], "candidate changed after freeze")
            require({x["task"]:x["compatibility_key"] for x in declared["task_bindings"]} == {x["task"]:x["compatibility_key"] for x in t["tasks"]}, "frozen task compatibility binding changed")
            stage_by_id[t["candidate_id"]] = record["stage"]
        stages.append({"stage": record["stage"], "study_id": study_id, "recipes": count,
                       "selected_candidate_id": record["selection"]["selected_candidate_id"], "required_pass_count": record["selection"]["required_pass_count"],
                       "activation_recomputed": index > 0, "search_input_digest": search["input_digest"], "search_sha256": file_hash(searchpath)})
        pool.extend(current)
    require(len(stage_by_id) == 25, "missing or overlapping stage membership")
    baseline = {k:v for k,v in control["resolved_recipe"].items() if k not in OPT_FIELDS}
    require(all({k:v for k,v in t["resolved_recipe"].items() if k not in OPT_FIELDS} == baseline for t in trials), "nonoptimizer recipe fields changed")
    top = sorted(trials, key=lambda t:(-sum(x["gate_status"] == "PASS" and x["importance"] == "required" for x in t["tasks"]), t["configuration_id"]))[:3]
    artifacts = Artifacts(root, queue, archive)
    receipts, observed, traces, compact, attempts, ends = [], defaultdict(list), {}, [], set(), []
    try:
        for trial in trials:
            tasks = {}
            for task in trial["tasks"]:
                name, attempt = task["task"], task["attempt_id"]
                require(attempt not in attempts, "unexpected paid attempt sharing")
                attempts.add(attempt)
                request, result, cert, row, attemptterminal = artifacts.receipt(trial, task, frozen["cohort"])
                proof = artifacts.clock_probe(attempt, request["tasks"][name], row["evidence"]) if name == CLOCK else None
                tasks[name] = task_summary(request, row, task, proof)
                tasks[name].update(attempt_id=attempt, result_hash=cert["result_hash"], compatibility_key=task["compatibility_key"])
                if name != CLOCK:
                    observed[name].append((trial["candidate_id"], task, (request, result, cert, row)))
                    steps = sorted({math.ceil(i * request["tasks"][name]["execution"]["steps"] / 24) for i in range(1,25)})
                    if name in TRACED:
                        trace = artifacts.trace(attempt, row["evidence"]["optimizer_diagnostics"], name, steps)
                        traces[(trial["candidate_id"], name)] = trace
                    if name == REQUIRED[5]:
                        ends.append(word_stream_end(artifacts, attempt, row, trial))
                    saved = row["evidence"].get("saved_observer_outputs")
                    if saved:
                        payload = artifacts.read(attempt, saved["path"])
                        require(hashlib.sha256(payload).hexdigest() == saved["sha256"] and len(payload) == saved["bytes"], "saved observation artifact mismatch")
                receipts.append({"candidate_id":trial["candidate_id"], "task":name, "attempt_id":attempt,
                                 "result_hash":cert["result_hash"], "terminal_sha256":artifacts.refs[f"campaign/{CAMPAIGN}/{attempt}/terminal.json"]["sha256"], "budget_seconds": task["budget_seconds"]})
            compact.append({"candidate_id":trial["candidate_id"], "configuration_id":trial["configuration_id"], "stage":stage_by_id[trial["candidate_id"]], "settings":settings(trial),
                            "required_passes": sum(t["gate_status"] == "PASS" for name,t in tasks.items() if name != CLOCK), "tasks":tasks})
            if len(compact) % 5 == 0:
                print(json.dumps({"event":"certificates_verified", "configurations":len(compact), "attempts":len(attempts)}, sort_keys=True), flush=True)
        require(len(attempts) == 175, "not exactly175 distinct certified attempts")
        qstate = read_json(queue / "queue/state.json")
        paid = [j for j in qstate["jobs"].values() if (j.get("cost_owner") or {}).get("campaign") == CAMPAIGN]
        require(len(paid) == 175 and all(j["status"] == "terminal" and len(j["attempts"]) == 1 and j["reserved_seconds"] == 0 for j in paid), "queue terminal/history mismatch")
        require({a["attempt_id"] for j in paid for a in j["attempts"]} == attempts, "queue/receipt roster mismatch")
        task_audits = [old.audit_task(name, observed[name], 25) for name in REQUIRED]
        require(all(all(x["matched_all_component_receipts"] for x in audit["initialization"].values()) for audit in task_audits), "emitted aggregate initial component states differ or are missing")
        require(all(all(x["matched_present"] for x in audit["stream_starts"].values()) for audit in task_audits), "named stream starts differ")
        require(all(all(x["matched_present"] for x in audit["task_contracts"].values()) for audit in task_audits), "task geometry/evaluation changed")
        end_keys = set(ends[0]["streams"])
        require(all(set(e["streams"]) == end_keys for e in ends), "word stream checkpoint roster differs")
        end_summary = {key:{"receipts":len(ends), "distinct_end_hashes":sorted({e["streams"][key] for e in ends}),
                           "matched_all":len({e["streams"][key] for e in ends}) == 1} for key in sorted(end_keys)}
        summaries = [{"candidate_id":trial["candidate_id"], "settings":settings(trial), "tasks":{name:trace_summary(traces[(trial["candidate_id"], name)], name) for name in TRACED}} for trial in top]
        output.mkdir(parents=True, exist_ok=True)
        figrefs = plot(top, traces, output) if figures else []
        source_helpers = [REPORT / "analyze.py", REPORT / "run.py", Path("reports/forge/dualnorm-tier1/analyze.py")]
        inputs = {str(REPORT / "results.json"):file_hash(root / REPORT / "results.json"),
                  **{str(queue / name):file_hash(queue / name) for name in ("frozen-study.json", "terminal-summary.json", "driver-state.json", "execution-clock.json", "queue/state.json")}}
        result = {"schema_version":1, "study":CAMPAIGN, "qualification_input":False, "source":results["source"], "seed":0,
                  "verification":{"certified_attempts":175,"required_gates_recomputed":150,"clock_saved_state_comparisons_recomputed":100,
                    "optimizer_traces_verified":75,"valid_scalar_ring_input_gradient_checkpoints":1200,"word_input_gradient_checkpoints_excluded":600,
                    "terminal_workers":0,"no_retries":True,"source_snapshot_files_verified":len(frozen["source"]["files"]),
                    "matched_source_runtime_protocol_policy":frozen["cohort"], "admitted_stage_contracts_recomputed":True},
                  "execution":{"elapsed_seconds":results["elapsed_seconds"],"paid_seconds":results["accounting"]["spent_seconds"],"maximum_elapsed_seconds":43200,"maximum_paid_reserved_seconds":63000},
                  "selection":results["selection"], "control_candidate_id":control["candidate_id"], "stages":stages,
                  "configuration_order":"Frozen stage order, then configuration hash; no separate ranked leaderboard.",
                  "configurations":sorted(compact,key=lambda t:(t["stage"],t["configuration_id"])),
                  "training_contract_audits":task_audits, "plot_selection":{"objective":results["selection"]["objective"],"whole_candidates":[t["candidate_id"] for t in top],"control_included":control["candidate_id"] in {t["candidate_id"] for t in top}},
                  "retained_word_stream_end_audit":{"scope":"Post-run bytes of retained state.pt, not an original evaluator/certificate artifact digest; gates use independently verified certified observations.",
                      "streams":end_summary,"checkpoints":[{k:v for k,v in e.items() if k != "streams"} for e in ends],
                      "other_required_tasks":"Receipts bind named initial states and observation isolation audits; no full final-stream checkpoint was emitted for these task attempts.",
                      "batch_sequence_limit":"Equal named starts and word final states with unchanged task draw laws support matched consumption. No direct consumed-batch digest was recorded."},
                  "diagnostics":summaries, "figures":figrefs, "certificates":receipts,
                  "input_artifacts_sha256":inputs, "consumed_original_artifacts":artifacts.refs,
                  "reader_source_sha256":{str(p):file_hash(root/p) for p in source_helpers},
                  "limitations":["Fixed seed0, provisional finite Tier1 screen; no independent seeds, native7k runs, scale transfer, R1R2 or P1–P6 score.",
                    "4/6 bestObserved is an experimental improvement, not full6/6 qualification or default adoption. No Adam comparison was rerun on this source.",
                    "Only optimizer/rate/momentum fields vary; task-owned priors/initialization fixtures remain explicit, and all task laws/caps/gates stay fixed.",
                    "Spectral product includes matrix weights only, excludes Fourier maps and nonlinearities, and is not a whole-network Lipschitz bound.",
                    "Word input-gradient real/fake labels are excluded because previous-G forwards can fill the next-D cache; scalar/ring labels are phase-proven.",
                    "Weight norms use first recorded checkpoint as reference; finite growth flags do not prove unbounded asymptotic growth.",
                    "Initialization claims use actually emitted aggregate state hashes; missing aggregate hashes and absent direct consumed-batch hashes remain explicit."]}
        result["input_digest"] = stable_hash(result)
        atomic_json(output / "analysis.json", result)
        return result
    finally:
        artifacts.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.queue_root.resolve(), args.output or args.root / REPORT, args.archive, not args.no_figures)
    print(json.dumps({"event":"analysis_verified", "attempts":result["verification"]["certified_attempts"], "required_gates":result["verification"]["required_gates_recomputed"],
                      "selected_candidate_id":result["selection"]["selected_candidate_id"], "required_passes":result["selection"]["required_pass_count"], "input_digest":result["input_digest"]}, sort_keys=True))


if __name__ == "__main__":
    main()
