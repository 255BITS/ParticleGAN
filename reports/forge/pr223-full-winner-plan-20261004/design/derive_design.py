#!/usr/bin/env python3
"""Derive a source/metadata-only PR223 retest design. Never import scientific code."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess

CONFIG_SHA = "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4"
REFERENCE_SHA = "1e1b536bfd66e20a240436b417d429059d22ea55c2b6ecfff31e796b96c470fa"
PR_HEAD = "bc9d9aec23618e84b6b3cc8f5169638b0925671b"
PR_MERGE = "437c7554235a6de0c6777aca9d8993b2c3e62c67"
PR_SCIENCE = "bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f"
ROOT = pathlib.Path("/ml2/hypergan/ParticleGAN-atlas-forge-unblock-20261003")
PACKET = pathlib.Path("/ml2/hypergan/pg-pr223-primary-packet-20261004")
HARNESS = pathlib.Path("/ml2/hypergan/lrfree-20260926/harness")
OUT = pathlib.Path(__file__).resolve().parent
ADAPTER = "reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/validation-ra15"
REPORT = "reports/forge/continuous-baseline-20261003"
RESOURCE_KEYS = {"num_particles", "z_dim", "batch_size"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stable(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--primary-packet", type=pathlib.Path, default=PACKET)
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    args = parser.parse_args()
    root, packet, output = args.root.resolve(), args.primary_packet.resolve(), args.output.resolve()
    consumed = {}

    def read(path, expected=None):
        path = pathlib.Path(path).resolve()
        data = path.read_bytes()
        digest = sha(data)
        if expected is not None and digest != expected:
            raise ValueError(f"input digest changed: {path}")
        consumed[str(path)] = {"path": str(path), "sha256": digest, "bytes": len(data)}
        return data

    def load(path, expected=None):
        return json.loads(read(path, expected))

    primary_manifest = load(packet / "manifest.json")
    for filename, pin in primary_manifest["files"].items():
        data = read(packet / filename, pin["sha256"])
        assert len(data) == pin["bytes"]
    pr = load(packet / "pr223.json")
    assert pr["number"] == 223 and pr["merged"] is True
    assert pr["head"]["sha"] == PR_HEAD and pr["merge_commit_sha"] == PR_MERGE
    config = load(root / "configs/100gaussians/atlas.json", CONFIG_SHA)
    local_git_config_checks = []
    for revision in (PR_SCIENCE, PR_HEAD, PR_MERGE):
        blob = subprocess.check_output(
            ["git", "show", f"{revision}:configs/100gaussians/atlas.json"], cwd=root
        )
        assert sha(blob) == CONFIG_SHA
        local_git_config_checks.append({"commit": revision, "path": "configs/100gaussians/atlas.json", "sha256": sha(blob), "bytes": len(blob)})
    selected_source_diff = subprocess.check_output(
        ["git", "diff", "--name-only", PR_SCIENCE, PR_HEAD, "--", "particlegan",
         "configs/100gaussians/atlas.json", f"{ADAPTER}/screen_current.py", f"{ADAPTER}/current_api_fixtures.py"], cwd=root, text=True
    ).splitlines()
    assert selected_source_diff == []

    publication = load(root / REPORT / "results.json")
    baseline = publication["baseline"]
    assert baseline["required_questions"] == baseline["completed"] == 19
    assert baseline["required_evidence_complete"] is True
    assert baseline["declared_updates"] == 48800
    assert baseline["scientific_counts"]["PASS"] == 19
    assert all(value == 0 for key, value in baseline["scientific_counts"].items() if key != "PASS")
    reference = load(root / "reports/develop-gates-20261001/atlas19-replay.json", REFERENCE_SHA)
    load(baseline["study"]["path"], baseline["study"]["sha256"])
    read(root / REPORT / "README.md")
    read(root / REPORT / "PLAN.md")
    read(root / REPORT / "ORIGINAL_POSITIVE_CHECK.md")
    load(root / REPORT / "ORIGINAL_POSITIVE_CHECK.json")
    archive = load(root / REPORT / "archive.json")
    read(root / REPORT / "ARCHIVE.md")
    vector_specs = load(HARNESS / "tasks/vector_task_specs.json")
    image_specs = load(HARNESS / "tasks/image_task_specs.json")
    native_fixture = load(HARNESS / "tasks/native100_fixture.json")
    inspection_source_paths = [
        root / ADAPTER / "screen_current.py", root / ADAPTER / "current_api_fixtures.py",
        root / REPORT / "run_atlas_baseline.py", root / REPORT / "publish_results.py",
        root / REPORT / "export_moving_goal.py", root / REPORT / "export_native_goal.py",
        root / "particlegan/recipes.py", root / "particlegan/policy.py",
        root / "particlegan/training.py", root / "particlegan/particle_prior.py",
        root / "particlegan/continuous.py", root / "particlegan/feature_policy.py",
        root / "particlegan/feature_cells.py", root / "particlegan/feature_reference.py",
        root / "particlegan/recipe_schedules.py", root / "particlegan/ka2.py",
        root / "particlegan/k3p.py", root / "experiments/forge/policy_execution.py",
        root / "experiments/forge/policy_publication.py",
        HARNESS / "hosts/image_host.py", HARNESS / "hosts/vector_host.py",
        HARNESS / "hosts/mode_hold_host.py", HARNESS / "hosts/ring_host.py",
        HARNESS / "native100_score.py",
        pathlib.Path("/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py"),
        pathlib.Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan/initialization.py"),
        pathlib.Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan/qr_bz_pq_init.py"),
    ]
    inspected = {}
    for path in inspection_source_paths:
        read(path)
        inspected[str(path)] = consumed[str(path.resolve())]

    shared_recipe = None
    matrix = []
    new_allowances = 0
    for ordinal, row in enumerate(baseline["rows"], 1):
        definition = row["definition"]
        assert row["scientific_status"] == "PASS" and row["full_protocol_complete"] is True
        assert stable(definition) == row["case_sha256"]
        assert definition["config_sha256"] == CONFIG_SHA and definition["reference_sha256"] == REFERENCE_SHA
        raw = load(row["result"]["path"], row["result"]["sha256"])
        resource = {key: definition["original_host"][key] for key in sorted(RESOURCE_KEYS)}
        if row["group"] != "moving":
            assert raw["completed_steps"] == definition["original_host"]["steps"]
            assert raw["observations"] == len(definition["observation_steps"])
            assert raw["eval_output_noise"] is True and raw["stream_deviations"] == 0
            recipe = raw["recipe"]
            assert all(recipe[key] == value for key, value in resource.items())
            common = {key: value for key, value in recipe.items() if key not in RESOURCE_KEYS}
            if shared_recipe is None:
                shared_recipe = common
            else:
                assert common == shared_recipe
            recipe_basis = "recorded full result.recipe; no tensor loading"
        else:
            # The bound executed wrapper is the source authority; checkpoint tensors are not read.
            runner_pin = row["artifacts"]["runner.py"]
            wrapper = read(runner_pin["path"], runner_pin["sha256"]).decode()
            assert "recipe = get_recipe(**{**options, 'num_particles': n_rows, 'z_dim': 2, 'batch_size': batch})" in wrapper
            recipe = None
            recipe_basis = "expected full original config plus N20000/z2/b2048, bound executed runner.py; checkpoint values not independently deserialized"
        allowance = math.ceil((1.5 * row["cost"]["paid_seconds"] + 90.0) / 30.0) * 30
        new_allowances += allowance
        group, task = row["group"], row["task"]
        if task.startswith("img_"):
            host = {"generator": "original residual_upsample width16 8x8 grayscale host", "discriminator": "original image_host.Discriminator(spec)", "spec": image_specs[task]}
            observer = {"kind": "ordered all32 selected rows, public indexed _generate", "samples": 32, "latent_extra_prior_perturb": False, "public_policy_latent_perturbation": "retained as actually selected", "evaluation_seed": "2303 + completed_steps", "output_noise": "current learned positive selected output kernel", "diagnostic_weights": "forced trainer.ema_G/ema_prior only; no qualification", "classification": image_specs[task]["thresholds"], "tv_gate": False}
            gate = "complete24 observations and original final5 passing suffix; modes/HQ only, using original template classification cutoffs"
        elif task.startswith("vector_"):
            host = {"generator": "SimpleMLPGenerator(z4,64,2,2)", "discriminator_card": vector_specs[task]["discriminator_card"], "spec": vector_specs[task]["spec"]}
            observer = {"samples": 4096, "prior_indices_seed": 990, "policy_output_noise_seed": 2303, "output_noise": "current learned kernel added by original host measure", "primary_weights": "trainer.G/prior as selected by original policy", "diagnostic_weights": "forced trainer.ema_G/ema_prior only; no qualification", "public_policy_latent_perturbation": "retained"}
            gate = "complete24 observations and original final5 passing suffix; exact distribution requirements"
        elif task == "mode_hold":
            host = {"generator": "SimpleMLPGenerator(z4,96,3,2)", "critic": "SimpleMLPDiscriminator(2,96,3,3)", "reference": "original mode_hold_host and frozen fit batches"}
            observer = {"samples": 4096, "seed": "original measure 402+completed_steps (global),2303+completed_steps (isolated policy)", "output_noise": "original noisy primary", "public_policy_latent_perturbation": "retained"}
            gate = "complete24 observations and original final5 passing suffix, modes>=8 and HQ>=.9"
        elif group == "portability":
            host = {"generator": "SimpleMLPGenerator(z2,96,3,2)", "critic": "SimpleMLPDiscriminator(2,96,3,3)", "reference": "original ring_host; shift after update2400 for ring_shift"}
            observer = {"samples": 4096, "evaluation_seed": 9, "cadence": "every10 updates; no initial0 metric", "primary_sampler": "trainer.sample(4096,ema=False,output_noise=True)", "diagnostic_sampler": "same selected law output_noise=False plus forced EMA diagnostics", "public_policy_latent_perturbation": "retained"}
            gate = "complete full horizon; exact8 modes/HQ>=.9; original5-check final suffix in each required pre/post-shift segment"
        elif group == "moving":
            host = {"generator": "identity Linear(2,2)", "critic": "SimpleMLPDiscriminator(2,128,3,3)", "initializer": "original native CUDA ordering plus RA11 deterministic critic", "target": "original geometry rotates30 degrees after updates500 and1000"}
            observer = {"gate_samples": 20000, "retained_movie_samples": 4096, "movie_steps": [0,500,1000,1500], "target_degrees": [0,0,30,60], "gate_latent_seed": 1637, "gate_noise_seed": 1636, "movie_latent_seed": 77, "movie_noise_seed": 78, "primary_weights": "trainer.G/prior actual selected public policy", "output_noise": "current learned kernel", "public_policy_latent_perturbation": "retained"}
            gate = "both post-turn gates PASS; modes>=95 and HQ>=.9 times this run's observed update500 HQ; not an absolute .9 HQ gate"
        else:
            host = {"generator": "affine_square_v1 identity Linear(2,2)", "critic": "SimpleMLPDiscriminator(2,128,3,3)", "initializer": "source-defined native CUDA draw order plus RA11 deterministic critic", "fixture": native_fixture}
            observer = {"samples": 20000, "observations": 34, "terminal_steps": [6000,6250,6500,6750,7000], "independent_holdout_samples": 100000, "primary_law": "state-selected weights/current learned output kernel", "clean_diagnostic": "output-noise-off same selected law", "forced_ema_diagnostic": "separate original EMA branch", "evaluation_prior_seed": 1637, "evaluation_noise_seed": 1636, "evaluation_target_seed": 1635, "holdout_seed_offsets": {"target":1601,"noise":1602,"latent":1603}, "public_policy_latent_perturbation": "retained"}
            gate = "joint coverage AND accuracy PASS: original>=5 terminal coverage checks plus each final5 20k fidelity checks AND independent100k fidelity holdout"
        matrix.append({"ordinal": ordinal, "id": f"pr223-faithful19-retest-v1-{group}-{task}", "group": group, "task": task, "prospective_status": "NOT_RUN", "historical_original_status": row["scientific_status"], "historical_clean_diagnostic_status": row["clean_diagnostic_status"], "historical_result": row["result"], "historical_case_sha256": row["case_sha256"], "original_definition": definition, "host": host, "observer": observer, "resolved_resources": resource, "resolved_recipe": recipe, "recipe_evidence_scope": recipe_basis, "original_gate_semantics": gate, "historical_paid_seconds": row["cost"]["paid_seconds"], "proposed_inclusive_allowance_seconds": allowance, "export_grace_seconds": 0})
    assert len(matrix) == 19 and sum(row["original_definition"]["original_host"]["steps"] for row in matrix) == 48800
    assert new_allowances == 9810
    assert shared_recipe["lr"] == .00425 and shared_recipe["prior_lr_mult"] == 2 and shared_recipe["d_lr_mult"] == 1
    assert shared_recipe["output_noise_std"] == .029 and shared_recipe["total_steps"] is None
    for row in matrix:
        if row["resolved_recipe"] is None:
            row["expected_resolved_recipe_from_bound_source"] = {**shared_recipe, **row["resolved_resources"]}
    metadata_snapshot_drift = []
    for path, past_sha in baseline["source"]["protected_files_sha256"].items():
        current = root / path
        if path.startswith("particlegan/") and current.is_file():
            data = read(current)
            if sha(data) != past_sha:
                metadata_snapshot_drift.append({"path": path, "historical_sha256": past_sha, "inspected_current_sha256": sha(data)})

    design = {
        "schema": "pg_pr223_faithful_winner_retest_design_v1", "status": "DESIGN_ONLY_NOT_PREPARED_NOT_RUN",
        "claim": "One full original19 Atlas winner replay; fresh source-scoped evidence, no C6 or ordinaryMoG pooling",
        "primary_source": {"pr": 223, "url": "https://github.com/255BITS/ParticleGAN/pull/223", "merged_at": pr["merged_at"], "head": PR_HEAD, "merge": PR_MERGE, "body_scientific_source": PR_SCIENCE, "config": {"path":"configs/100gaussians/atlas.json","sha256":CONFIG_SHA,"full_json":config}, "local_git_config_checks": local_git_config_checks, "selected_package_config_RA15_diff_bdf_to_head": selected_source_diff, "files_endpoint_scope": "first100 of4379 changedfiles, incomplete; never a closure certificate"},
        "verified_historical_evidence": {"baseline_source": baseline["source"], "counts": {"PASS":19}, "required":19, "updates":48800, "cost":baseline["cost"], "runtime":baseline["runtime"], "study":baseline["study"], "reference_pin":consumed[str((root / "reports/develop-gates-20261001/atlas19-replay.json").resolve())], "archive_card":archive, "raw_archive_availability":"LOCAL_ONLY; remote upload not performed; archive bytes not reread by this design"},
        "winner_resolved_common_recipe_excluding_only_three_resource_fields": shared_recipe,
        "family_label_caveat": "Original frozen get_recipe(**full Atlas config) records Recipe.name=ka2. Full controls, actual Atlas backend and config digest identify Atlas; this is not ordinaryKA2 learning credit.",
        "actual_law": {"prior":"ParticlePrior raw learnable row table; uniform row selection at source-defined sampling sites; standardize=True Recipe metadata does not standardize ParticlePrior reads", "row_policy":"independent", "continuous_policy":"actual public DV12 retained", "output_kernel":"learned positive kernel initially .029; primary uses current selected kernel; no forced clean scoring", "serving":"original policy-selected state, which can be fast or averaged; serve_average4/ema_decay.995 retained; do not force EMA", "small_host_backend":"automatic reference kNN on N12/32/256; preserve factor1", "large_host_backend":"automatic featurecells when source admits N20000; preserve .25 G/noise nominal calibration, D/prior unchanged", "nominal_rates":{"reference":{"G":.00425,"noise":.00425,"D":.00425,"prior":.0085},"admitted_feature_cells":{"G":.0010625,"noise":.0010625,"D":.00425,"prior":.0085}}, "applied_rates":"endogenous policy/stationarity schedules; record full actual rates rather than infer displacement", "horizon":"external full original steps; total_steps=None; retain configured anneal fields without injecting a new cosine horizon", "prose_correction":"ORIGINAL_POSITIVE_CHECK.md's latent-perturbation-disabled sentence is overbroad. image_prior_perturb=False skips only extra prior.perturb; _generate still executes public policy perturbation. Original evidence/files untouched."},
        "matrix":matrix,
        "proposed_budget": {"currency":"new separately supervised wall seconds; all construction/training/observation/score/export/attestation included", "per_case_formula":"ceil((1.5 * prior paid seconds +90)/30)*30", "all19_full_allowance_seconds":new_allowances, "bounded_shared_metadata_and_final_publication_seconds":180, "initial_full_reservation_seconds":9990, "proposed_campaign_ceiling_seconds":10800, "unallocated_margin_seconds":810, "old_baseline_reference_paid_seconds":baseline["cost"]["paid_seconds"], "old_cost_recharged_as_new":False, "named10500_ledger_scope":"unchanged separate ledger; no reset or pooling", "interrupt":"non-completed/error/cancelled/missing terminal charges max(full case allowance,measured), reserve=max(0,allowance-paid); completed numericFAIL charges measured only", "overrun":"record actual paid overrun, halt; no zero-fill or cap clipping", "retries":0, "export_grace_seconds":0},
        "runtime_contract": {"physical_gpu":"1", "cuda_visible_devices":"1", "logical_device":"cuda:0", "cpu_threads":1, "torch_memory_fraction":.2, "minimum_free_mib":12288, "maximum_temperature_c":82, "shared_queue":"/ml2/hypergan/ParticleGAN-single-recipe/runs/forge", "python":"/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python", "cublas_workspace_config":":4096:8", "tf32":False, "deterministic_algorithms":True, "cudnn_deterministic":True, "float32_matmul_precision":"highest", "serial_backward":True, "optimizer_foreach":False, "optimizer_fused":False, "one_scientific_job_per_physical_device":True, "other_owner_jobs":"never modified/interrupted", "telemetry":"must be freshly verified by root at admission; read-only earlier telemetry is not admission"},
        "implementation_path": {"existing_driver":f"{REPORT}/run_atlas_baseline.py", "existing_operations":["original_inputs","task_definition","plan","prepare","child","certify","run"], "public_training_owner":"particlegan.GANTrainer.step; no copied training loop/new scheduler", "required_narrow_extension":["new explicit protocol/case/source/runtime identities and proposed per-case inclusive caps, zero grace, fresh output sibling preparation", "bind full current public package and immutable relocated external RA15/harness/RA11/native fixture/scorer/rotate sources and all definition data bytes", "real copied-source CUDA-hidden metadata preflight with exact canonical namespaces/config/JSON-wire/full19 definitions, zero models/draws/scorers/CUDA and RNG purity", "inherited study+attempt leases and one case deadline through original child plus grading/actual goal media/final source attestation", "capture only arrays already computed at original observation sites when adding image/vector/ring goal illustrations; no extra sampling/evaluation or official gate changes", "paired original gate/status and full-completeness/media/source proof; no exit0-only qualification"], "no_blind_C6_wrapper_reuse":True, "future_clean_source_origin":"ROOT_TO_FREEZE_AFTER_PARITY_CONTROLS", "runtime_source_certificate":"fresh complete current/imported source+config+external closure; old60 inspection pins/partialPR files are not a new execution certificate", "inspected_current_package_drift_against_a0d_reference":metadata_snapshot_drift, "scorer_portability":"snapshot relocates actual absolute native consumers into immutable external native_root; hash-check every executed consumer, not just unused copied files"},
        "acceptance": {"denominator":19,"gate_changes":False,"complete_updates":48800,"required":["each full original resource/seed/cadence/target/init/Recipe/primary sampler exact","pure isolated observations and complete owner/checkpoint/RNG state","actual status JSON numeric gate, original native coverage AND accuracy incl100k","source/import/durable deadline/lease/cost proof","source-bound actual goal GIF and final attestation within case allowance"],"numericFAIL":"preserve and continue other independent declared cases","source_or_harness_or_model_fault":"INVALID; halt, retain prefix and UNKNOWN remainder; no automatic retry","timeout_or_export_or_missing_final_attestation":"INCOMPLETE/BUDGET_EXCEEDED; accepted numeric result unavailable; retained evidence only with explicit inseparable status","qualification":"fresh original19 cohort only; no current26/default/speed credit without separately compatible required gates"},
        "additional_gates":{"owner":"/root/merge_readiness","design_directory":"/ml2/hypergan/pg-pr223-additional-gates-mapping-20261004","scope":"separate original26/API domain mapping, retain parent targets/init/seeds/observers/full horizons and explicit ownership/resource adaptations; same full winning controls when compatible","no_automatic_credit":"historical19 does not fill changed26/API gates; API native24+initial lacks ordinary native34+100k; C6 rates/noise-off/transpose12/ring12 are distinct","ordinary_mog":"KA2/K3P4/5 results are separate sharedRecipe learned studies, not Atlas policy eligibility","excluded_unapproved_work":["Toy/MNIST new sweep","NLL objective variant","seed-only retest","new candidate search"]},
        "model_free_preflight_controls":["all19 definitions/config/seeds/full Recipes equality and input relocation; altered image RMSE/TV/native threshold/cadence/holdout/resource rejects","full JSON roundtrip exact identity; no tuple/list false mismatch or ignored nested mutation","protected namespace alias dedup and exact imported file closure; pseudo Torch namespaces do not weaken actual repository file checks","original initialization and observer-present/absent structural checkpoint/RNG parity before costly acquisition","native coverageFAIL+accuracyPASS classifies numericFAIL, not PASS/ERROR; missing final5/100k is incomplete","deadline covers media/final attestation; completedFAIL versus interrupted max-allowance recovery and dropped-supervisor proof","old result cannot attach under changed source/case/runtime/sampler; duplicates never silently rerun","retained image/grid/vector/ring media clocks and arrays exact; no fresh draws or fabricated early convergence"],
        "inspection_sources":inspected,
        "work_performed":{"metadata_json_source_reads":True,"read_only_git_show_diff":True,"network_calls_by_subagent":"one failed transport, no retry; root primary packet consumed","models_constructed":0,"states_restored":0,"forwards":0,"draws":0,"scorer_calls":0,"training_updates":0,"gpu_calls":0,"science_preparations":0,"queue_mutations":0,"repository_mutations":0},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "design.json").write_text(json.dumps(design, sort_keys=True, indent=2, allow_nan=False)+"\n")
    (output / "input-index.json").write_text(json.dumps({"schema":"pg_pr223_retest_design_input_index_v1","scope":"read-only inspection metadata/source; not a new runtime certificate","files":consumed},sort_keys=True,indent=2)+"\n")
    print(json.dumps({"cases":len(matrix),"updates":48800,"all_case_allowance":new_allowances,"full_reservation":9990,"inputs":len(consumed),"design_sha256":sha((output/"design.json").read_bytes()),"input_index_sha256":sha((output/"input-index.json").read_bytes()),"package_drift_paths":[p["path"] for p in metadata_snapshot_drift]},sort_keys=True))


if __name__ == "__main__":
    main()
