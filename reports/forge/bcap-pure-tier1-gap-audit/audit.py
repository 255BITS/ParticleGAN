"""Inspect preserved BCAP evidence; never train, sample a model, or rewrite grades."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import tarfile

import numpy as np
import scipy
from scipy.special import ndtr, ndtri
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import stable_hash
from particlegan import get_recipe
from particlegan.recipes import Recipe, learning_rate_scales

HERE = Path(__file__).resolve().parent
STUDY = ROOT / "reports/forge/dualnorm-pacing-v2"
CAMPAIGN = "bcap-dualnorm-pacing-v2"
SOURCE = "a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be"
DIGEST = "f1755b1b5538901ffd4882f196bfd475030b06df16fd940c9b839eff86dc8226"
ANALYSIS_SHA = "c5f758adaf8da45aafe547a31ad54cf2bfe96ad662c007b17cb7194bd4b3ed86"
GAUSSIAN, RING = "gaussian1d_acquisition", "ring16_acquisition"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def archive_inputs(path, needed, refs, inventory):
    require(Path(path).stat().st_size == inventory["bytes"], "archive size changed")
    require(sha_file(path) == inventory["sha256"], "archive SHA changed")
    data = {}
    # One streaming pass avoids repeated seeks through a compressed archive.
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.name not in needed:
                continue
            require(member.isfile() and member.name not in data, "invalid/duplicate member")
            with archive.extractfile(member) as stream:
                payload = stream.read()
            ref = refs[member.name]
            require(len(payload) == ref["bytes"], f"size mismatch: {member.name}")
            require(hashlib.sha256(payload).hexdigest() == ref["sha256"],
                    f"SHA mismatch: {member.name}")
            data[member.name] = payload
    require(set(data) == needed, "required original artifacts missing")
    return data


def original_row(config, task, payloads):
    attempt = config["tasks"][task]["attempt_id"]
    prefix = f"durable/{attempt}/"
    wrapper = json.loads(payloads[prefix + "request.json"])
    request, job = wrapper["request"], wrapper["job"]
    result = json.loads(payloads[prefix + "result.json"])
    certificate = json.loads(payloads[prefix + "evidence.json"])
    require(stable_hash(result) == certificate["result_hash"] ==
            config["tasks"][task]["result_hash"], "result/certificate mismatch")
    require(request["source"] == certificate["source"], "certificate source mismatch")
    require(request["runtime"] == certificate["runtime"], "certificate runtime mismatch")
    require(request["source"]["origin_commit"] == SOURCE and
            request["source"]["digest"] == DIGEST and
            stable_hash(request["source"]["files"]) == DIGEST, "wrong source cohort")
    require(request["protocol"]["seed"] == 0 and
            request["candidate"]["id"] == config["candidate_id"], "wrong seed/candidate")
    require(request["candidate_revision"] == result["candidate_revision"], "candidate revision changed")
    require(len(result["task_results"]) == 1, "unexpected combined result")
    row = result["task_results"][0]
    require(row["task_id"] == task and row["gate_status"] ==
            config["tasks"][task]["gate_status"], "task/status mismatch")
    require(row["compatibility_key"] == config["tasks"][task]["compatibility_key"],
            "task binding mismatch")
    require(job["task_ids"] == [task] and stable_hash(job["science"]) ==
            job["compatibility_key"] == row["compatibility_key"], "science binding changed")
    require(not result.get("retry_of"), "unexpected retry")
    return request, row


def constant_rates(recipe):
    parsed = Recipe(**recipe)
    probes = sorted({0, 1, parsed.total_steps, parsed.total_steps * 2,
                     parsed.total_steps * 4, 100_000})
    scales = [list(learning_rate_scales(step, parsed)) for step in probes]
    require(all(s == [1., 1.] for s in scales), "nonconstant step schedule")
    require(parsed.lr_floor == parsed.resolved_network_lr_floor == 1., "nonunit floor")
    require(parsed.optimizer_family == "dualnorm" and parsed.optimizer_momentum == 0.,
            "unexpected optimizer")
    require((parsed.lr, parsed.d_lr_mult, parsed.prior_lr_mult) == (.012, 1.5, 2.5),
            "winning step sizes changed")
    require(parsed.eps == 1e-8 and parsed.d_eps is None and parsed.prior_eps is None and
            parsed.optimizer_adam_lr is None and parsed.loss == "relativistic",
            "optimizer epsilon/control or loss changed")
    require(parsed.reg_arm == "b_cap" and parsed.reg_coeff == parsed.reg_kappa == 1.,
            "BCAP coefficient/cap changed")
    require(parsed.beta2_end is None and parsed.reg_coeff_end is None and
            parsed.continuous_policy is None and parsed.network_lr_horizon_cap is None,
            "active optional schedule/controller")
    require(parsed.d_guard_ratio == parsed.reg_anchor_weight == parsed.latent_damping_max_rate ==
            parsed.input_noise_std == parsed.output_noise_std == parsed.output_noise_warmup ==
            parsed.ema_decay == parsed.serve_average == parsed.prior_reg == 0.,
            "unexpected stabilizer/noise/serving override")
    require(not parsed.direct_particle_gain and not parsed.particle_birth_death and
            not parsed.row_evidence_gate, "unexpected optimizer intervention")
    return {"steps_checked": probes, "network_prior_scales": scales,
            "G_E": parsed.lr, "D": parsed.lr * parsed.d_lr_mult,
            "prior": parsed.lr * parsed.prior_lr_mult}


def gaussian_shape(points):
    values = np.sort(points[:, 0].numpy().astype(np.float64))
    n = len(values)
    ranks = np.arange(n) / n
    cdf = ndtr((values - 2.) / .5)
    dplus, dminus = ranks + 1 / n - cdf, cdf - ranks
    up, down = int(dplus.argmax()), int(dminus.argmax())
    z = (values - values.mean()) / values.std()
    corrected = ndtr(z)
    return {"cdf_ks_recomputed": float(max(dplus.max(), dminus.max())),
            "largest_target_minus_empirical": {
                "gap": float(dminus[down]), "x": float(values[down]),
                "target_cdf": float(cdf[down]), "empirical_left_cdf": float(ranks[down])},
            "largest_empirical_minus_target": {
                "gap": float(dplus[up]), "x": float(values[up])},
            "moment_corrected_ks_diagnostic_only": float(max(
                np.max(corrected - ranks), np.max(ranks + 1 / n - corrected))),
            "quantiles": [{"probability": p, "generated": float(np.quantile(values, p)),
                           "target": float(2 + .5 * ndtri(p))}
                          for p in [.01, .05, .1, .25, .5, .75, .9, .95, .99]]}


def ring_shape(points, spec, live):
    # Preserve the float32 assignment used by the original scorer; use float64
    # only for new tail diagnostics. These numbers do not enter the gate.
    means32 = points.new_tensor(spec["means"])
    labels = torch.cdist(points, means32).argmin(1)
    points64, means = points.double(), torch.tensor(spec["means"], dtype=torch.float64)
    radius2 = ((points64 - means[labels]) / .1).square().sum(1)
    components = []
    for k in range(16):
        member = labels == k
        group, distance2 = points64[member], radius2[member]
        require(len(group) == live["component_counts"][k], "ring assignment changed")
        require(int((distance2 <= 9).sum()) == live["hq_component_counts"][k],
                "ring HQ count changed")
        centered_energy = (group - group.mean(0)).square().sum(1)
        components.append({"component": k, "assigned_count": len(group),
                           "hq_count": live["hq_component_counts"][k],
                           "full_covariance_error": live["component_covariance_errors"][k],
                           "core_4sigma_covariance_error": live["component_core_covariance_errors"][k],
                           "spill_3sigma": live["component_spill"][k],
                           "tail_4sigma_share_of_centered_energy": float(
                               centered_energy[distance2 > 16].sum() / centered_energy.sum())})
    return {"tail_counts_by_radius_sigma": {
                str(s): int((radius2 > s * s).sum()) for s in [3, 4, 6, 10]},
            "maximum_radius_sigma": float(radius2.max().sqrt()), "components": components}


def draw_figure(rows, samples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, task_rows in rows.items():
        for task, metric, ax in [(GAUSSIAN, "cdf_ks", axes[0, 0]),
                                 (RING, "component_covariance_error", axes[0, 1])]:
            obs = task_rows[task]["evidence"]["observations"]
            ax.plot([o["step"] for o in obs], [o[metric] for o in obs], label=label)
    for ax, threshold, title in [(axes[0, 0], .05, "Gaussian: CDF KS (lower is better)"),
                                  (axes[0, 1], .85, "Ring: full component covariance error")]:
        ax.axhline(threshold, color="black", linestyle="--", label="gate threshold")
        ax.set(title=title, xlabel="Completed updates")
        ax.legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    vals = np.sort(samples[GAUSSIAN][-1]["samples"][:, 0].numpy())
    axes[1, 0].plot(vals, np.arange(1, len(vals) + 1) / len(vals), label="Winner empirical CDF")
    axes[1, 0].plot(vals, ndtr((vals - 2) / .5), label="Exact target CDF")
    axes[1, 0].set(title="Gaussian at 1,000 updates", xlabel="Generated scalar", ylabel="CDF")
    axes[1, 0].legend(fontsize=8)
    live = rows["Winner .012"][RING]["evidence"]["live"]
    x = np.arange(16)
    axes[1, 1].bar(x - .18, live["component_covariance_errors"], .36, label="All assigned samples")
    axes[1, 1].bar(x + .18, live["component_core_covariance_errors"], .36, label="4σ core diagnostic")
    axes[1, 1].axhline(.85, color="black", linestyle="--")
    axes[1, 1].set(title="Ring at 400 updates: component shape", xlabel="Component index",
                   ylabel="Relative covariance error", yscale="log", xticks=x)
    axes[1, 1].legend(fontsize=8)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require(sha_file(STUDY / "analysis.json") == ANALYSIS_SHA, "frozen analysis changed")
    analysis = read_json(STUDY / "analysis.json")
    inventory = read_json(STUDY / "artifact-inventory.json")["archive"]
    winner = next(c for c in analysis["configurations"] if c["candidate_id"] ==
                  analysis["selection"]["selected_candidate_id"])
    control = next(c for c in analysis["configurations"] if c["candidate_id"] ==
                   analysis["control_candidate_id"])
    faster = next(c for c in analysis["configurations"] if c["settings"]["eta_G_E"] == .016)
    selected = [(winner, task) for task in winner["tasks"]]
    selected += [(c, task) for c in (control, faster) for task in (GAUSSIAN, RING)]
    needed = {f"durable/{c['tasks'][task]['attempt_id']}/{name}"
              for c, task in selected for name in ("request.json", "result.json", "evidence.json")}
    for task in (GAUSSIAN, RING):
        needed.add(f"campaign/{CAMPAIGN}/{winner['tasks'][task]['attempt_id']}/observed-samples.pt")
    payloads = archive_inputs(args.archive, needed, analysis["consumed_original_artifacts"], inventory)
    tasks, rows, saved = {}, {}, {}
    for c, task in selected:
        request, row = original_row(c, task, payloads)
        label = "Winner .012" if c is winner else "Control .010" if c is control else "Faster .016"
        rows.setdefault(label, {})[task] = row
        if c is not winner:
            continue
        declaration = request["tasks"][task]
        summary = winner["tasks"][task]
        recipe = row.get("recipe", row.get("applied", {}).get("recipe"))
        require(recipe is not None, "applied recipe missing")
        tasks[task] = {"attempt_id": summary["attempt_id"], "result_hash": summary["result_hash"],
                       "gate_status": summary["gate_status"], "prior": row.get("prior"),
                       "declared_steps": declaration["execution"]["steps"],
                       "timeout_seconds": declaration["resources"]["timeout_seconds"],
                       "constant_schedule": constant_rates(recipe),
                       "applied_recipe": recipe, "gate_cells": summary.get("gate_cells"),
                       "convergence": summary.get("convergence"),
                       "last_five": summary.get("last_five")}
        if task in (GAUSSIAN, RING):
            member = f"campaign/{CAMPAIGN}/{summary['attempt_id']}/observed-samples.pt"
            receipt = row["evidence"]["saved_observer_outputs"]
            require(hashlib.sha256(payloads[member]).hexdigest() == receipt["sha256"],
                    "sample receipt mismatch")
            require(receipt["sampling_draws_added"] == receipt["optimizer_updates_added"] == 0,
                    "observer changed training")
            saved[task] = torch.load(BytesIO(payloads[member]), map_location="cpu", weights_only=True)
            require([s["step"] for s in saved[task]] ==
                    [o["step"] for o in row["evidence"]["observations"]], "sample cadence changed")
            require(len(saved[task]) == receipt["observation_count"] == 24, "wrong sample count")
            tasks[task]["sample_diagnostics"] = gaussian_shape(saved[task][-1]["samples"]) if task == GAUSSIAN else ring_shape(
                saved[task][-1]["samples"], declaration["execution"]["host_definition"], row["evidence"]["live"])
    require(abs(tasks[GAUSSIAN]["sample_diagnostics"]["cdf_ks_recomputed"] -
                rows["Winner .012"][GAUSSIAN]["metrics"]["cdf_ks"]) < 1e-12, "KS parity failed")
    public = asdict(get_recipe("bcap"))
    import matplotlib
    report = {"schema_version": 1, "qualification_input": False, "training_runs_added": 0,
              "model_sampling_draws_added": 0, "archived_grades_rewritten": False,
              "audited_develop_commit": "9cdfd70a49ef4e81143d304c550dfaea72ae51ff",
              "executed_source_commit": SOURCE, "executed_source_digest": DIGEST,
              "selected_candidate_id": winner["candidate_id"], "required_passes": 4,
              "required_total": 6, "public_default": public,
              "public_default_schedule": constant_rates(public), "tasks": tasks,
              "prior_study_sustained_passes": {task: sum(c["tasks"][task]["gate_status"] == "PASS"
                    for c in analysis["configurations"]) for task in (GAUSSIAN, RING)},
              "prior_study_configs_with_any_joint_pass": {task: sum(c["tasks"][task]["convergence"]["passing_observations"] > 0
                    for c in analysis["configurations"]) for task in (GAUSSIAN, RING)},
              "reader_runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                                 "numpy": np.__version__, "scipy": scipy.__version__,
                                 "matplotlib": matplotlib.__version__},
              "archive": {k: inventory[k] for k in ("bytes", "sha256", "path")},
              "reader_sha256": sha_file(__file__),
              "input_artifacts": {name: analysis["consumed_original_artifacts"][name] for name in sorted(needed)},
              "publication_inputs": {str(p.relative_to(ROOT)): sha_file(p) for p in
                    (STUDY / "analysis.json", STUDY / "DEFAULT_SELECTION.md", ROOT / "particlegan/recipes.py")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "audit-diagnostics.png"
    draw_figure(rows, saved, figure)
    report["figure_sha256"] = sha_file(figure)
    report["input_digest"] = stable_hash(report)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "verified", "original_artifacts": len(needed),
                      "required_passes": "4/6", "training_runs_added": 0,
                      "input_digest": report["input_digest"]}), flush=True)


if __name__ == "__main__":
    main()
