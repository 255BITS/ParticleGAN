"""Read saved BCAP evidence; never train, sample, or change qualification.

Run from the repository root. Bulk inputs must exist in the original archive.
--check verifies the committed projection and generated Markdown in place.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REPORT = Path(__file__).resolve().parent
BASE = "0337bbbc2300fd351e77e9a222b92401137fd421"
SOURCE = "d378734f40b09ce223a389e8f54a9783ec6a0c75"
BEGIN = "<!-- BEGIN GENERATED MEASUREMENTS -->"
END = "<!-- END GENERATED MEASUREMENTS -->"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def compare(value, op, target):
    if value is None:
        return False
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return False
    return {">=": lambda: value >= target, "<=": lambda: value <= target,
            "==": lambda: value == target}[op]()


def fmt(value):
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def row(metric, value, limits=(), target=None):
    limits = list(limits)
    status = ("UNAVAILABLE" if value is None else
              "PASS" if all(compare(value, op, bound) for op, bound in limits) else "FAIL")
    if not limits:
        status = "DIAGNOSTIC" if value is not None else "UNAVAILABLE"
    return {"metric": metric, "score": value,
            "target": target or (" and ".join(f"{op} {fmt(bound)}" for op, bound in limits)
                                 if limits else "No declared gate"),
            "limits": [[op, bound] for op, bound in limits], "status": status}


def metric_rows(metrics, thresholds):
    limits = {}
    for key, op, bound in thresholds:
        limits.setdefault(key, []).append((op, bound))
    keys = list(limits) + sorted(set(metrics) - set(limits))
    return [row(k, metrics.get(k), limits.get(k, ())) for k in keys
            if k in limits or not isinstance(metrics[k], str)]


def coverage_thresholds(card, n):
    c = card["evaluation"]["coverage_thresholds"]
    return [("n", ">=", c["min_samples"]), ("valid_n", "==", n),
            ("modes", "==", c["min_modes"]), ("precision", ">=", c["min_precision"]),
            ("min_hq_mode_mass", ">=", c["min_hq_mode_mass"]),
            ("mass_tv", "<=", c["max_mass_tv"]),
            ("max_mode_mass", "<=", c["max_mode_mass"]),
            ("min_cov_eig_ratio", ">=", c["min_cov_eig_ratio"]),
            ("max_cov_eig_ratio", "<=", c["max_cov_eig_ratio"]),
            ("min_radial_median_ratio", ">=", c["min_radial_median_ratio"]),
            ("max_radial_median_ratio", "<=", c["max_radial_median_ratio"]),
            ("min_hq_count", ">=", math.ceil(c["min_hq_mode_mass"] * n - 1e-12)),
            ("passed", "==", True)]


def accuracy_thresholds(card, n):
    return [("n", ">=", card["evaluation"]["coverage_thresholds"]["min_samples"]),
            ("valid_n", "==", n), ("all_finite", "==", True),
            ("precision", ">=", card["evaluation"]["coverage_thresholds"]["min_precision"]),
            *[(k, "<=", v) for k, v in card["evaluation"]["accuracy_limits"].items()],
            *[(k, "==", True) for k in ("accuracy_pass", "frozen_pass", "passed")]]


def extract_prior(state):
    if "fixture" in state:
        state = state["fixture"]["api_state"]
    state = state.get("trainer", state)
    models = state.get("models", {})
    tables = [(name, value["z"]) for name, value in models.items()
              if isinstance(value, dict) and "z" in value and not name.startswith("ema")]
    return tables


def build():
    import numpy as np
    import torch
    from benchmarks.toy100.metrics import evaluate_samples
    from benchmarks.toy100.problems import evaluation_geometry
    # Load the public loss at its original Git pin, before the ownership refactor.
    loss_source = git_bytes(SOURCE, "particlegan/vicreg_loss.py")
    loss_namespace = {}
    exec(compile(loss_source, "git-pinned-vicreg_loss.py", "exec"), loss_namespace)
    VICRegLikeLoss = loss_namespace["VICRegLikeLoss"]

    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    results_path = ROOT / "reports/forge/bcap-default-baseline/results.json"
    spec_path = ROOT / "reports/forge/bcap-default-baseline/spec.json"
    results = json.loads(results_path.read_text())
    spec = json.loads(spec_path.read_text())
    assert results["source_commit"] == SOURCE and results["protocol_seed"] == 0
    required = [r["task"] for r in spec["requirements"] if r["qualification_tier"] == 2]
    selected = {r["task_id"]: r for r in results["task_results"]
                if r["role"] == "candidate" and r["tier"] == 2}
    controls = {r["task_id"]: r for r in results["task_results"]
                if r["role"] == "control" and r["tier"] == 2}
    assert set(required) == set(selected) == set(controls) and len(required) == 21
    assert Counter(r["gate_status"] for r in selected.values()) == {"PASS": 9, "FAIL": 12}
    code_paths = ["benchmarks/toy100/metrics.py", "benchmarks/toy100/problems.py"]
    for name in code_paths:
        assert git_bytes(SOURCE, name) == (ROOT / name).read_bytes(), name
    inputs = [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
              for p in (results_path, spec_path)]
    for item in inputs:
        assert git_bytes(BASE, item["path"]) == (ROOT / item["path"]).read_bytes()
    source_proofs = [{"path": name, "sha256": sha(ROOT / name),
                      "matches_executed_source": True} for name in code_paths]
    source_proofs.append({"path": "particlegan/vicreg_loss.py",
                          "sha256": hashlib.sha256(loss_source).hexdigest(),
                          "loaded_from_git_commit": SOURCE})
    tasks, probes, proofs = [], [], []
    for task_id in required:
        print(f"Inspecting {task_id}", flush=True)
        record, control = selected[task_id], controls[task_id]
        card_bytes = git_bytes(SOURCE, f"configs/forge/tasks/{task_id}.json")
        card = json.loads(card_bytes)
        cp = record["provenance_checkpoint"]
        attempt_dir = Path(cp["artifact_root"]).parent
        request_path = attempt_dir / "request.json"
        request = json.loads(request_path.read_text())
        assert request["request"]["source"]["origin_commit"] == SOURCE
        frozen_card = request["request"]["tasks"][task_id]
        assert set(frozen_card) - set(card) <= {"field_ownership", "preflight_blockers"}
        assert {k: frozen_card[k] for k in card} == card
        raw_path = attempt_dir / "raw-result.json"
        raw = json.loads(raw_path.read_text())
        published_original = json.loads((attempt_dir / "result.json").read_text())
        assert published_original["raw"]["result"] == raw
        applied = raw.get("applied", raw)
        assert applied["recipe"]["prior_reg"] == 0
        assert applied["recipe"]["constraint_geometry_mode"] == "direction_blend"
        e = record["evaluator_summary"]
        thresholds = card["evaluation"].get("thresholds", [])
        # Confirm that metric_checks are the recorded bounds, not today's defaults.
        for key, check in e.get("metric_checks", {}).items():
            assert [key, check["op"], check["threshold"]] in thresholds
            assert check["value"] == record["metrics"][key]
        sections = [{"name": "Recorded endpoint", "rows": metric_rows(record["metrics"], thresholds)}]
        host_diagnostics = {k: v for k, v in raw.get("raw", {}).items()
                            if k not in record["metrics"] and not isinstance(v, (dict, list, str))}
        if host_diagnostics:
            sections.append({"name": "Additional saved host diagnostics", "rows": metric_rows(host_diagnostics, [])})
        temporal = []
        if "convergence" in e:
            c = e["convergence"]
            temporal = [row("terminal passing suffix", c["passing_suffix"],
                            [(">=", c["minimum_stable_checks"])]),
                        row("passing scheduled observations", c["passing_observations"],
                            target=f"Diagnostic count out of {c['observations']}; suffix is the gate"),
                        row("complete declared execution", c["complete"], [("==", True)])]
        elif task_id == "gaussian1d_stability":
            ev = card["evaluation"]
            temporal = [row("stationary passing checks", e["stationary_passes"],
                            [("==", ev["stationary_checks"])]),
                        row("deadline reacquisition", e["reacquisition_status"] == "PASS", [("==", True)]),
                        row("shifted hold passing checks", e["shift_hold_passes"],
                            [("==", ev["shift_hold_checks"])]),
                        row("frozen-copy passing checks", e["frozen_passes"],
                            target=f"Diagnostic control out of {e['frozen_checks']}; no qualification gate")]
        elif task_id == "five_word_joint_hold":
            count = card["evaluation"]["observations"] + 1
            temporal = [row("passing primary checks", e["passing_observations"], [("==", count)]),
                        row("confirmed checks", e["confirmed_checks"], [("==", count)])]
        elif card["adapter"] == "native100":
            ev = card["evaluation"]
            temporal = [row("coverage decision", e["coverage_status"] == "PASS", [("==", True)]),
                        row("terminal live accuracy passing checks", e["terminal_summary"]["passing_checks"],
                            [("==", ev["minimum_stable_checks"])])]
            manifest = raw["evidence"]["artifact_manifest"]["files"]
            art_root = Path(raw["evidence"]["artifact_root"])
            for name in (f"{task_id}/summary.json", f"{task_id}/holdout_samples.npz"):
                p = art_root / name
                assert sha(p) == manifest[name]["sha256"] and p.stat().st_size == manifest[name]["size"]
                proofs.append({"path": str(p), **manifest[name]})
            summary = json.loads((art_root / task_id / "summary.json").read_text())
            assert summary["completed_steps"] == card["execution"]["steps"] == 7000
            assert summary["holdout"]["live"] == record["metrics"]
            assert summary["config"]["eval_output_noise"] == "clean"
            assert summary["config"]["forge_recipe"]["ema_decay"] == 0
            sections = [{"name": "Live endpoint coverage, 20000 saved draws",
                         "rows": metric_rows(summary["final"]["live"], coverage_thresholds(card, 20000))},
                        {"name": "Live endpoint accuracy, 20000 saved draws",
                         "rows": metric_rows(e["final_metrics"], accuracy_thresholds(card, 20000))},
                        {"name": "Independent live holdout accuracy, 100000 saved draws",
                         "rows": metric_rows(record["metrics"], accuracy_thresholds(card, 100000))}]
            with np.load(art_root / task_id / "holdout_samples.npz") as sample_file:
                points = sample_file["live"]
            coverage = evaluate_samples(torch.from_numpy(points), task_id)
            assert coverage["passed"] == record["metrics"]["frozen_pass"]
            sections.append({"name": "Independent holdout coverage recomputed on those same saved draws",
                             "rows": metric_rows(coverage, coverage_thresholds(card, 100000))})
            centers, sigma = evaluation_geometry(task_id)
            distances, assignments = torch.cdist(torch.from_numpy(points).float(), centers).min(dim=1)
            native_diagnostics = [row("nearest cells occupied", assignments.unique().numel(),
                                      target="Diagnostic reference 100; genuine modes use the coverage gate"),
                                  row("median nearest-center distance in target sigmas",
                                      distances.double().div(sigma).median().item())]
            sections.append({"name": "Saved holdout placement diagnostics", "rows": native_diagnostics})
        sections.insert(0, {"name": "Required temporal gate", "rows": temporal})
        checkpoint_path = Path(cp["artifact_root"]) / cp["path"]
        assert sha(checkpoint_path) == cp["sha256"] and checkpoint_path.stat().st_size == cp["bytes"]
        proofs.extend([{"path": str(checkpoint_path), "sha256": cp["sha256"], "size": cp["bytes"]},
                       {"path": str(request_path), "sha256": sha(request_path)},
                       {"path": str(raw_path), "sha256": sha(raw_path)}])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for name, table in extract_prior(state):
            z = table.detach().double().clone().requires_grad_(True)
            centered = z - z.mean(0)
            cov = centered.T @ centered / (len(z) - 1)
            std = torch.sqrt(z.var(0) + 1e-4)
            floor = torch.relu(1 - std).mean()
            off = cov - torch.diag(torch.diag(cov))
            decorrelation = off.square().sum() / z.shape[1]
            loss = VICRegLikeLoss()(z)
            assert torch.allclose(loss, floor + decorrelation, atol=1e-12)
            gradient, = torch.autograd.grad(loss, z)
            probes.append({"task_id": task_id, "table": name, "task_status": record["gate_status"],
                           "rows": len(z), "dimensions": z.shape[1], "std": std.detach().tolist(),
                           "minimum_std": std.min().item(), "maximum_std": std.max().item(),
                           "variance_hinge": floor.item(), "covariance_loss": decorrelation.item(),
                           "unweighted_vicreg_loss": loss.item(), "gradient_norm": gradient.norm().item(),
                           "maximum_row_gradient_norm": gradient.norm(dim=1).max().item(),
                           "target_std": 1, "eps": 1e-4,
                           "cohort": "deterministic full saved center table, CPU float64; not served samples"})
        same_metrics = record["metrics"] == control["metrics"]
        if record["gate_status"] == "FAIL":
            assert same_metrics and e == control["evaluator_summary"]
        tasks.append({"task_id": task_id, "status": record["gate_status"],
                      "control_status": control["gate_status"], "same_final_metrics_as_control": same_metrics,
                      "attempt_id": record["attempt_id"], "candidate_revision": record["candidate_revision"],
                      "task_contract_sha256": record["task_contract_sha256"],
                      "frozen_task_file_sha256": hashlib.sha256(card_bytes).hexdigest(),
                      "prior": applied["prior"], "sampling": record["sampling"],
                      "budget_steps": card["execution"].get("steps"),
                      "timeout_seconds": card["resources"]["timeout_seconds"],
                      "actual_training_gif": f"../bcap-default-baseline/media/candidate-{task_id}.gif",
                      "sections": sections})
    assert torch.equal(torch.get_rng_state(), rng_before)
    return {"schema_version": 1, "scope": "read_only_saved_evidence_investigation",
            "qualification_input": False, "base_commit": BASE, "executed_source_commit": SOURCE,
            "executed_source_digest": results["source_digest"], "protocol_seed": 0,
            "optimizer_updates_added": 0, "random_sampling_draws_added": 0,
            "torch_global_rng_unchanged": True, "inputs": inputs,
            "code_proofs": source_proofs, "artifact_proofs": proofs,
            "tier2_status_counts": dict(Counter(t["status"] for t in tasks)),
            "tasks": tasks, "prior_diagnostics": probes}


def render(data):
    lines = [BEGIN, "", "## Measurements for all 21 Tier 2 tasks", "",
             "Scores use six significant digits; [the numerical receipt](measurements.json) keeps full precision. "
             "PASS/FAIL in a metric row checks that row only. The task result retains its complete recorded gate. "
             "UNAVAILABLE fails a required numerical check. DIAGNOSTIC means no declared gate; it grants no PASS. "
             "Each task links to its existing actual-training GIF; images do not determine these grades.", ""]
    for task in data["tasks"]:
        lines.extend([f"### {task['task_id']} {task['status']}", "",
                      f"[Actual training GIF]({task['actual_training_gif']}) · "
                      f"Prior `{task['prior'].get('kind')}`, sigma `{task['prior'].get('sigma')}` · "
                      f"Declared updates `{task['budget_steps']}` · allowance `{task['timeout_seconds']} s`.", ""])
        for section in task["sections"]:
            lines.extend([f"**{section['name']}**", "", "| Measurement | Score | Target | Result |",
                          "| --- | ---: | --- | --- |"])
            for r in section["rows"]:
                status = r["status"]
                if status in {"FAIL", "UNAVAILABLE"}:
                    status = f"**{status}**"
                lines.append(f"| `{r['metric']}` | {fmt(r['score'])} | {r['target']} | {status} |")
            lines.append("")
    lines.extend(["## Saved prior measurements", "",
                  "Full-table diagnostics use the existing `target_std=1`, `eps=1e-4` regularizer "
                  "on copied endpoint centers. They do not change the trained recipe, whose coefficient is zero. "
                  "Standard deviations include epsilon, matching the implementation. The numerical receipt also "
                  "retains each dimension, table shape and unweighted gradient norms.", "",
                  "| Task | Center std min / max, target ≥1 | Variance hinge, diagnostic ideal 0 | "
                  "Covariance loss, diagnostic ideal 0 | Maximum row gradient norm, no declared gate |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for p in data["prior_diagnostics"]:
        lines.append(f"| `{p['task_id']}` ({p['table']}) | {fmt(p['minimum_std'])} / {fmt(p['maximum_std'])} | "
                     f"{fmt(p['variance_hinge'])} | {fmt(p['covariance_loss'])} | {fmt(p['maximum_row_gradient_norm'])} |")
    lines.extend(["", END])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = build()
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    receipt = REPORT / "measurements.json"
    readme = REPORT / "README.md"
    original = readme.read_text()
    start, finish = original.index(BEGIN), original.index(END) + len(END)
    revised = original[:start] + render(data) + original[finish:]
    if args.check:
        assert receipt.read_text() == encoded, "Numerical receipt is stale"
        assert original == revised, "Generated measurement tables are stale"
        print("Verified 21 tasks, every recorded endpoint metric, original bounds and saved artifact hashes.")
    else:
        receipt.write_text(encoded)
        readme.write_text(revised)
        print("Wrote compact measurements and report tables; zero updates and zero random draws.")


if __name__ == "__main__":
    main()
