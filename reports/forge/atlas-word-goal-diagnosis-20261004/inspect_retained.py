"""Read saved tensor dictionaries and arrays; never construct or execute models."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA must be explicitly invisible")
import numpy as np
import torch

torch.set_num_threads(1)

TASK = "five_word_joint_acquisition_word_joint_policy_min11_v1"
COMMIT = "fb7acc775b3a1a6184d36b55e035b9da04531492"
DIGEST = "f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed"
RAW_PINS = {
    "raw-result.json": "e26ec506c81e8cdd74447c466d516ef30785de501da28dd50e698c115ebe7515",
    "graded-result.json": "5eb697cf6678ef764204ca5cd9e755a9d2f6f1fe0530c58a608b18ee9e3d766a",
    "resolved.json": "52a639183e9ed3c541b90311a16d0968ed85cd7cb039c64d0df693f4cc43bbc1",
    "word-joint-policy/state.pt": "16388c4e59a657a154a4730e285191acfd18fbc23d41102d0c3ba82227efb8a7",
}
CHARS = "abcdefghijklmnopqrstuvwxyz_ "
SOURCES = (
    "experiments/forge/word_joint_policy_adapters.py",
    "experiments/forge/word_joint_policy_contracts.py",
    "experiments/forge/policy_adapters.py",
    "experiments/forge/views.py",
    "benchmarks/toy_audit/api_images.py",
    "benchmarks/toy_audit/definition_quality.py",
    "particlegan/policy.py", "particlegan/continuous.py",
    "particlegan/recipes.py", "particlegan/gan_loss.py",
    "particlegan/particle_prior.py", "particlegan/birth_death.py",
    "configs/forge/task-variants/word_joint_policy_min11_v1/" + TASK + ".json",
)


def pin(path):
    b = path.read_bytes()
    return {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def strings(value):
    return ["".join(CHARS[int(i)] for i in row) for row in value.argmax(1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new evidence output required")
    if torch.cuda.is_initialized():
        raise RuntimeError("no CUDA context allowed")
    rng_before = torch.get_rng_state().clone()
    raw_path, grade_path = args.run / "raw-result.json", args.run / "graded-result.json"
    resolved_path, checkpoint = args.run / "resolved.json", args.run / "word-joint-policy/state.pt"
    steps = [math.ceil(i * 20001 / 24) for i in range(1, 25)]
    arrays = [(step, args.run / "word-joint-policy/observations" / f"step_{step:06d}.npz") for step in steps]
    files = [raw_path, grade_path, resolved_path, checkpoint, args.run.parent.parent / "study.json",
             args.source / "forge-source.json"]
    files += [args.source / p for p in SOURCES]
    files += [p for _, p in arrays]
    before = {str(p): pin(p) for p in files}
    manifest = json.loads((args.source / "forge-source.json").read_text())
    assert manifest["origin_commit"] == COMMIT and manifest["digest"] == DIGEST
    assert all(before[str(args.source/p)]["sha256"] == manifest["files"][p] for p in SOURCES)
    assert all(before[str(args.run/p)]["sha256"] == expected for p, expected in RAW_PINS.items())
    raw, grade, resolved = [json.loads(p.read_text()) for p in (raw_path, grade_path, resolved_path)]
    assert grade["source_digest"] == DIGEST and raw["task_id"] == TASK
    task = resolved["request"]["tasks"][TASK]
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    policy = state["policy"]
    assert policy["completed_steps"] == state["caller_cursor"] == 20001
    assert state["recipe"] == policy["recipe"]
    assert policy["roles"] == [["generator", "encoder", "table", "noise"], ["critic"]]
    evidence = raw["evidence"]
    for _, path in arrays:
        relative = path.relative_to(args.run / "word-joint-policy").as_posix()
        assert before[str(path)]["sha256"] == evidence["artifact_manifest"]["files"][relative]["sha256"]
    observations = evidence["observations"]
    assert len(observations) == 24
    assert [o["step"] for o in observations] == steps
    assert all(math.isfinite(v) for row in observations for v in row.values())
    summary = []
    for step, path in arrays:
        with np.load(path, allow_pickle=False) as a:
            assert all(np.isfinite(a[k]).all() for k in a.files)
            t, r, e, effective = [a[k] for k in ("target", "reconstruction", "encoded_code", "reconstruction_effective_code")]
            assert a["generated"].shape == (1024, 28, 6) and r.shape == (5, 28, 6)
            assert a["prior"].shape == (11, 2) and e.shape == effective.shape == (5, 2)
            target, reconstruction = strings(t), strings(r)
            correct = [x == y for x, y in zip(target, reconstruction)]
            true_probs = (t * r).sum(1)
            summary.append(dict(step=step, target_words=target, reconstruction_argmax=reconstruction,
                individually_correct_argmax=sum(correct), paired_argmax_correct=correct,
                raw_encoder_codes=e.tolist(), reconstruction_effective_codes=effective.tolist(),
                actual_dv12_displacements=(effective-e).tolist(),
                prior_unique_rows=len(np.unique(a["prior"], axis=0)),
                prior_axis_std=a["prior"].std(0).tolist(), encoder_axis_std=e.std(0).tolist(),
                true_token_probabilities=true_probs.tolist() if step in (834,10001,15835,20001) else None,
                argmax_only_display_diagnostic=True, numerical_regrading=False))
    optimizers = []
    for i, optimizer in enumerate(policy["optimizers"]):
        for j, group in enumerate(optimizer["param_groups"]):
            entries = [optimizer["state"][k] for k in group["params"]]
            first_moment_l2 = sum(float(x["exp_avg"].double().square().sum()) for x in entries) ** .5
            optimizers.append(dict(optimizer=i, group=j, role=policy["roles"][i][j],
                initial_lr=policy["initial_lrs"][i][j], terminal_applied_lr=group["lr"],
                betas=list(group["betas"]), amsgrad=group["amsgrad"],
                parameter_steps=[int(x["step"]) for x in entries],
                saved_first_moment_l2=first_moment_l2,
                applied_parameter_delta_history="NOT_RETAINED"))
    stationarity = []
    for i, rows in enumerate(policy["lr_settle"]):
        for j, row in enumerate(rows):
            stationarity.append(dict(optimizer=i, group=j, role=policy["roles"][i][j],
                s=row["s"], b=row["b"], tau=row["tau"], counts=row["counts"],
                last_decision=row["last"].get("decision"), last_decision_step=row["last"].get("step"),
                last_decisive=row["last_decisive"]))
    c = policy["controller"]
    d_ratio = optimizers[-1]["terminal_applied_lr"] / optimizers[-1]["initial_lr"]
    report = dict(schema="pg_word_retained_goals_and_rate_proposals_v1", status="READ_ONLY_METADATA_PROPOSAL",
        source=dict(commit=COMMIT,digest=DIGEST,consumed_files={p:before[str(args.source/p)] for p in SOURCES}),
        task_id=TASK,family="atlas_word_joint_min11",cohort="word_joint_policy_min11_v1",
        inputs=before, original_recipe=state["recipe"], original_task_evaluation=task["evaluation"],
        original_task_execution=task["execution"], initialization=state["initialization"],
        preserved=dict(family_status="INVALID",numeric_gate_status="UNAVAILABLE",old_grade=grade["grades"][TASK],
            paid_seconds=558.5739127129782,reserved_seconds=0,numerical_regrading=False),
        scope=dict(models_constructed=0,public_state_restores=0,forwards=0,sampler_calls=0,
            optimizer_updates=0,official_scorer_calls=0,cuda_initialized=False),
        recorded_metrics=observations, retained_array_descriptors=summary,
        terminal_optimizer_groups=optimizers,terminal_stationarity=stationarity,last_update=state["last_update"],
        terminal_controller=dict(payoff_error=c["payoff_error"],alignment=c["alignment"],
            last_cosine=c["last_cosine"],mobility=c["mobility"],closed=c["closed"],
            recorded_d_lr_fraction=d_ratio, current_post_generator_damping=1/(1+c["payoff_error"]**2),
            temporal_caveat="Applied critic LR used pre-generator payoff state; saved payoff is post-generator."),
        terminal_birth_counters=policy["birth_death"]["counters"],
        terminal_row_evidence=evidence["policy_controls"]["diagnostics"]["row_evidence"],
        terminal_surprise=evidence["policy_controls"]["diagnostics"]["surprise"],
        observations_pure=all(x["pure"] for x in evidence["policy_purity"]),
        all_observed_selected_sources=sorted({x["selected_source"] for x in evidence["policy_observations"]}),
        missing=["Per-update applied G/E/prior/D displacement and loss histories",
                 "Earlier complete model/optimizer/controller checkpoints",
                 "Counterfactual G(E(word)) with DV12 disabled on identical selected state",
                 "Current N11/full-DV12 word capacity certificate"],
        candidates=[dict(id="word-min11-quarter-base-rate-v1",status="PROPOSED_NOT_EXECUTED",
            overrides=dict(lr=.001328125,prior_lr_mult=1.5,d_lr_mult=1.),
            rationale="Reduce nominal motion of every role fourfold at preserved ratios; test whether sharp wrong-token/coverage excursions persist.",
            competing_explanation="No direct inverse objective and the stochastic joint law may dominate; endogenous scales will differ.",
            worst_case_seconds=900),
            dict(id="word-min11-slower-prior-v1",status="PROPOSED_NOT_EXECUTED",
            overrides=dict(lr=.0053125,prior_lr_mult=.15,d_lr_mult=1.),
            rationale="Reduce only nominal row motion tenfold; test a moving-support hypothesis suggested by changing saved code geometry and many table moves.",
            competing_explanation="Birth/death transport and DV12 geometry are endogenous; eleven distinct terminal rows are not collapsed latent support.",
            worst_case_seconds=900)],
        prospective_contract=dict(executed=False,full_updates=20001,observations=24,terminal_checks=5,eval_samples=1024,
            seed=0,actual_rows=11,free_encoder=True,same_effective_joint_code=True,words_only_learned_training_noise=True,
            selected_output_noise=False,actual_dv12=True,mechanism_overrides=False,gate_changes=False,
            max_total_new_gpu1_seconds=1800,old_named_gpu1_paid_seconds=675.413062,
            old_named_total_paid_seconds=910.239143,prior_cost_once=True,
            source_requirement="New exact rate-declaration/resolver and current health/source pins; frozen original C6-only resolver rejects new knobs.",
            qualification_credit=False,passing_case_reruns=False,root_owns_spec_admission_execution=True))
    assert torch.equal(rng_before,torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert all(pin(p)==before[str(p)] for p in files)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"],report=str(args.output),**pin(args.output),
                         inputs=len(files),models=0,scorer_calls=0,cuda=False)))


if __name__ == "__main__":
    main()
