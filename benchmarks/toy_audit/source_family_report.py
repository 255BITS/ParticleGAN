"""Publish a compact join of fresh source fixtures without regrading history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .source_routed_ring_training import FIXTURES, ROOT, VERSION, sha, write


def identity(path):
    return dict(path=str(path), sha256=sha(path))


def model_manifest(checkpoint):
    return {name: dict(shapes={key: list(value.shape) for key, value in state.items()},
                       tensor_elements=sum(value.numel() for value in state.values()))
            for name, state in checkpoint.items() if name in ("generator", "critic", "prior")}


def assemble(artifacts):
    artifacts = Path(artifacts)
    fixtures = []
    for name, spec in FIXTURES.items():
        receipt_path = artifacts / (name + "-receipt.json")
        receipt = json.loads(receipt_path.read_text())
        record = {**receipt, "catalog_id": spec["catalog_id"], "raw_training_receipt": identity(receipt_path),
                  "original_scientific_status": receipt["status"] if name == "ring" else "NO_FROZEN_GATE",
                  "added_correspondence_gate_status": "NOT_APPLICABLE" if name == "ring" else
                      "PASS" if receipt.get("final", {}).get("correspondence", {}).get("passed") else "FAIL",
                  "declared_budget_complete": receipt.get("completed_updates") == spec["updates"]}
        metrics_path = artifacts / name / "captured-metrics.json"
        record["captured_metrics"] = identity(metrics_path)
        metrics = json.loads(metrics_path.read_text())
        record["media"] = json.loads((artifacts / (name + "-media.json")).read_text())
        record["media_path"] = record["media"]["media"]["path"]
        record["gif_relative_to_report"] = "media/" + Path(record["media_path"]).name
        record["binding"]["quality_evaluator"]["declared_quality_version"] = "toy-definition-quality-v1"
        record["binding"]["quality_evaluator"]["version_field_scope"] = "The raw runner receipt's version field names the audit runner; declared_quality_version names the separate quality evaluator"
        record["executed_audit_source"] = {
            "runner": identity(artifacts / "executed-audit-source/source_family_training.py"),
            "executed_runner_path": "benchmarks/toy_audit/source_family_training.py",
            "published_runner_path": "benchmarks/toy_audit/source_routed_ring_training.py",
            "published_runner_content_identical": sha(ROOT / "benchmarks/toy_audit/source_routed_ring_training.py") ==
                sha(artifacts / "executed-audit-source/source_family_training.py"),
            "rename_reason": "Avoid a path collision with an independent agent's sign/landing/native audit module; source binding retains the exact executed bytes",
            "renderer": identity(artifacts / "executed-audit-source/source_family_media.py"),
            "quality_evaluator": identity(artifacts / "executed-audit-source/definition_quality.py"),
        }
        if record["executed_audit_source"]["quality_evaluator"]["sha256"] != record["binding"]["quality_evaluator"]["sha256"]:
            raise ValueError("archived quality evaluator differs from executed source")
        if name == "ring":
            checkpoint = torch.load(receipt["checkpoint"]["path"], map_location="cpu", weights_only=True)
            record["actual_model_manifest"] = model_manifest(checkpoint)
            record["actual_host_contract"] = dict(
                prior_type="ParticlePrior", prior_shape=list(checkpoint["prior"]["z"].shape),
                clean_law="Uniform 12 deterministic generator outputs; no latent perturbation",
                output_noisy_law="Uniform 12 isotropic Gaussian kernels around clean outputs; sigma0.029 at terminal evaluation",
                target="Eight equal-weight isotropic Gaussian modes on radius3 ring, sigma0.07",
                z_dim=4, particles=12, batch_size=128, generator_width=96, generator_hidden_layers=3,
                critic_width=96, critic_hidden_layers=3, critic_fourier_frequencies=3,
                configured_native_particles=20000, configured_native_batch=2048,
                resource_rule="The source bridge transfers global recipe/noise settings while retaining frozen mode_hold architecture, prior support, batch and initialization. Source_recipe resource defaults do not describe the executed host.")
            record["structural_witness"] = dict(
                clean_mass_tv_minimum_if_all_eight_components_resolved=1 / 6,
                witness_allocation=[2, 2, 2, 2, 1, 1, 1, 1],
                single_kernel_covariance_ratio=(.029 / .07) ** 2,
                evaluator_covariance_floor=.5,
                scope="Clean 12-row uniform-bank arithmetic and one terminal isotropic output-noise kernel. These exact witnesses explain a density/resource mismatch; they do not prove every approximate noisy gate is unattainable or determine the optimizer path.")
            record["added_gaussian_gate_status"] = "PASS" if receipt["final_gaussian_law"]["passed"] else "FAIL"
            record["full_added_gate_status"] = record["added_gaussian_gate_status"]
            best = max(metrics, key=lambda row: row["original"]["hq"])
            record["best_observed_original"] = best
            record["failed_terminal_gates"] = dict(modes=dict(value=receipt["final"]["modes"], expected=8),
                hq=dict(value=receipt["final"]["hq"], minimum=.90),
                mass_tv=dict(value=receipt["final_gaussian_law"]["mass_tv"], maximum=.075),
                max_radial_ks=dict(value=receipt["final_gaussian_law"]["max_radial_ks"], maximum=.10))
            record["scientific_failure_class"] = ["observed acquisition collapse", "separate full-density evaluator/resource mismatch"]
            record["optimization_mechanism"] = "UNRESOLVED: saved curves show transient six-mode quality at1000 then collapse; no controlled training intervention establishes the cause"
            record["next_action"] = "Version a representable density contract separately from the historical mode-retention task; inspect saved endpoint G/D/Adam state with a bounded diagnostic before proposing an optimization repair. Failed acquisition leaves hold/shift unknown."
        else:
            ablation_path = artifacts / (name + "-ablation.json")
            ablation = json.loads(ablation_path.read_text())
            record["code_ablation"] = {**ablation, "raw_ablation_receipt": identity(ablation_path)}
            record["executed_audit_source"]["ablation"] = identity(artifacts / "executed-audit-source/source_family_ablation.py")
            record["added_useful_code_gate_status"] = ablation["status"]
            if not record["declared_budget_complete"]:
                record["full_added_gate_status"] = "INCOMPLETE"
            else:
                record["full_added_gate_status"] = "PASS" if ablation.get("full_added_endpoint_gate") and receipt["stronger_gate_window"]["passed"] else "FAIL"
            record["convergence_qualified"] = bool(name != "replay" and record["full_added_gate_status"] == "PASS")
            if name == "moving":
                record["period_endpoints"] = [r for r in metrics if r["step"] in (500, 1000, 1500) and r["event"] == "check"]
                record["target_turns"] = []
                for turn in [r for r in metrics if r["event"] == "target_turn"]:
                    recovered = next((r for r in metrics if turn["step"] < r["step"] <= turn["step"] + 500 and
                                      r["event"] == "check" and r["correspondence"]["passed"]), None)
                    record["target_turns"].append(dict(instantaneous=turn,
                        first_passing_observation=None if recovered is None else recovered,
                        observed_recovery_updates=None if recovered is None else recovered["step"] - turn["step"]))
                record["next_action"] = "Retain this pre-R1 paired-transition result as a component fixture; separate R1 and independent-row native claims require their own contracts."
            elif name == "paired":
                record["next_action"] = "Use this fixture to check clean correspondence and useful mixed code; row moves, architecture transfer and independent-row Gaussian density require separate controls."
            elif name == "support":
                record["scientific_failure_class"] = ["wall-cap incomplete", "missing ready-boundary policy checkpoint"]
                record["optimization_mechanism"] = "UNRESOLVED: last complete captures at100/200 show remaining transition error, but the1200-update endpoint is absent"
                record["next_action"] = "Add checkpoint observation at completed capture boundaries before any separately authorized same-default audit; no ready checkpoint exists for continuing this attempt, and no full-budget code/quality result can be inferred."
            else:
                record["scientific_failure_class"] = ["software replay smoke passed", "short-budget paired quality failed", "code removal improves clean paired MSE"]
                record["optimization_mechanism"] = "UNRESOLVED: eightupdates are insufficient to diagnose eventual convergence; the frozen game judge favors live codes while clean held-out MSE favors zero codes"
                record["next_action"] = "Keep replay as a software protocol fixture. Declare a separate justified trained-quality budget/gate before seeking convergence; do not extend this eight-update smoke merely to obtain a pass."
        record["added_gate_status"] = record["full_added_gate_status"]
        fixtures.append(record)
    return dict(version=VERSION, coverage_catalog_ids=["source-family-10", "source-family-14"],
                fixtures=fixtures, source_parent_commit="54604961f1dcf6ffaa9c0807afb384bc590debe9",
                immutable_original_diagnosis=dict(commit="54604961f1dcf6ffaa9c0807afb384bc590debe9",
                    report_sha256=sha(ROOT / "reports/toy_audit/failure-diagnosis.json")),
                runner_sha256=sha(ROOT / "benchmarks/toy_audit/source_routed_ring_training.py"),
                assembler_sha256=sha(__file__),
                scope="Fresh source-only default paths, independent of historical73-arm diagnosis. No default/config/library repair, seed study or failed-acquisition extension.",
                limitation="Five GIFs verify captured states, not five convergence passes. Original routed scalars have no absolute trained binary gate; replay is software only; support is capped. Added audit gates do not replace historical oracles.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write(args.output, assemble(args.artifacts))


if __name__ == "__main__":
    main()
