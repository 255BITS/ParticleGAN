"""Run one frozen teacher-span acquisition diagnostic; bulk stays outside Git.

PYTHONPATH=. python -u examples/run_e22_routed_convergence_rotated_teacher.py \
    --out runs/routed-convergence-rotated-v1

This runner reuses PR227's public native loop, update, checkpoint and scoring
functions. It adds campaign orchestration and receipts, not a new optimizer.
The prepared card blocks quality execution until the coordinator authorizes
this single task after the full Supra fixed endpoints fail.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import particlegan
import torch

if __package__:
    from . import e22_routed_convergence as base
    from . import e22_routed_convergence_rotated_teacher as law
    # The held shared runner predates package-relative example imports.
    example_path = str(Path(__file__).resolve().parent)
    sys.path.insert(0, example_path)
    try:
        from . import run_e22_routed_convergence as common
    finally:
        sys.path.remove(example_path)
else:
    import e22_routed_convergence as base
    import e22_routed_convergence_rotated_teacher as law
    import run_e22_routed_convergence as common


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_convergence_rotated_teacher_v1.json"
SOURCES = (CARD, Path(__file__).resolve(), ROOT / "examples/e22_routed_convergence_rotated_teacher.py",
           ROOT / "examples/e22_routed_convergence.py", ROOT / "examples/e22_routed_convergence_neutral.py",
           ROOT / "examples/run_e22_routed_convergence.py", ROOT / "docs/e22_routed_convergence_v1.json",
           ROOT / "docs/e22_routed_convergence_neutral_v1.json",
           ROOT / "tests/test_e22_routed_convergence_rotated_teacher.py")
JUDGES = tuple(f"{arm}@{step}" for arm in law.ARMS[:2] for step in (800, 6400))
ENDPOINTS = (5120, 6400)


def source_hashes():
    return {str(path.relative_to(ROOT)): common.sha(path) for path in SOURCES}


def archive_sources(out, bindings):
    """Preserve exact held files and native Python for later artifact review."""
    archived = {}
    native = {str(path.relative_to(ROOT)): common.sha(path)
              for path in sorted((ROOT / "particlegan").rglob("*.py"))}
    if base.digest(native) != bindings["native_source_hash"]:
        raise RuntimeError("native source changed before archival")
    for prefix, hashes in (("", bindings["source_hashes"]), ("native/", native)):
        for relative, expected in hashes.items():
            destination = out / "source" / prefix / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            if common.sha(destination) != expected or common.sha(ROOT / relative) != expected:
                raise RuntimeError("source snapshot differs from frozen bytes: " + relative)
            archived[prefix + relative] = expected
    return archived


def validate_contract(card):
    execution = card["execution"]
    rates = {"generator": base.BRANCH_LR, "router": base.BRANCH_LR,
             "table": .0085, "critic": .00425, "learned_noise": .00425}
    host = card["host"]
    if (card["task_id"] != law.TASK or card["teacher_geometry"]["rho"] != law.RHO
            or card["teacher_geometry"]["audit_report_sha256"] != law.AUDIT_SHA256
            or [arm["id"] for arm in card["arms"]] != list(law.ARMS)
            or execution["steps_per_arm"] != 6400 or execution["checkpoint_cadence"] != 200
            or execution["device"] != "cpu" or execution["threads"] != 1
            or execution["arm_wall_budget_seconds"] != 900 or execution["total_wall_budget_seconds"] != 2700
            or execution["software_recovery_witness"] != [800, 802]
            or host["width"] != base.WIDTH or host["tokens_per_context"] != base.TOKENS
            or host["adapter_rank"] != base.RANK or host["sites"] != len(law.SITES)
            or card["prior"]["num_particles"] != base.PARTICLES or card["prior"]["z_dim"] != base.Z_DIM
            or card["output_noise_std"] != base.PANEL_SIGMA or card["native_nominal_rates"] != rates
            or tuple(card["evaluation"]["mandatory_common_judges"]) != JUDGES
            or tuple(card["evaluation"]["endpoints"]) != ENDPOINTS):
        raise ValueError("card differs from the single frozen rotated-teacher law")
    if common.native_source_hash() != card["native_python_source_digest"]:
        raise ValueError("native Python source differs from the pinned parent")
    for relative, key in (("docs/e22_routed_convergence_v1.json", "parent_card_sha256"),
                          ("docs/e22_routed_convergence_neutral_v1.json", "parent_neutral_card_sha256")):
        if common.sha(ROOT / relative) != card[key]:
            raise ValueError("pinned parent card changed: " + relative)
    for relative, expected in card["sources"].items():
        if common.sha(ROOT / relative) != expected:
            raise ValueError("declared source changed: " + relative)


def checkpoint_steps():
    return tuple(sorted({0, 802, *range(200, 6401, 200), *ENDPOINTS}))


def curve_steps():
    return tuple(step for step in checkpoint_steps() if step != 802)


def load_judge(state, data, *, expected_step):
    if state["step"] != expected_step or state["training"]["completed_steps"] != expected_step:
        raise ValueError("common critic checkpoint is mislabeled")
    with torch.random.fork_rng(devices=[]):
        judge = base.ConditionalCritic(data["scale"])
    judge.load_state_dict(state["training"]["models"]["critic"], strict=True)
    return judge.eval().requires_grad_(False)


def immutable_scores(loop, judges, panels, *, code_ablation=False):
    """Prove clean scoring does not consume training streams or diagnostics."""
    state = base.digest(base.checkpoint(loop))
    data, private = base.digest(loop.data), base.digest(panels)
    heads = {name: base.digest(judge.state_dict()) for name, judge in judges.items()}
    rng = torch.get_rng_state().clone()
    result = {name: {pool: base.evaluate(loop, judge, pool, panels, code_ablation=code_ablation)
                     for pool in base.SPLITS} for name, judge in judges.items()}
    for pools in result.values():
        for score in pools.values():
            for key, value in score.items():
                values = value if isinstance(value, list) else [value]
                if not all(math.isfinite(item) for item in values):
                    raise FloatingPointError("nonfinite offline score: " + key)
    if (state != base.digest(base.checkpoint(loop)) or data != base.digest(loop.data)
            or private != base.digest(panels)
            or heads != {name: base.digest(judge.state_dict()) for name, judge in judges.items()}
            or not torch.equal(rng, torch.get_rng_state())):
        raise AssertionError("offline scoring changed state, diagnostic counters, data, judges, or RNG")
    return result


def compact_scores(scores):
    return {name: {pool: common.compact(value) for pool, value in pools.items()}
            for name, pools in scores.items()}


def final_gates(endpoint_scores, witnesses):
    """Keep observed improvement, reproduced gaps and reference wins separate."""
    scores = {arm: endpoint_scores[f"{arm}@6400"] for arm in law.ARMS}
    ordinary = {name: scores[law.ARMS[0]][name]["test"]["paired_game"] for name in JUDGES}
    particle = {name: scores[law.ARMS[1]][name]["test"]["paired_game"] for name in JUDGES}
    neutral = {name: scores[law.ARMS[2]][name]["test"]["paired_game"] for name in JUDGES}
    gaps = {name: particle[name] - ordinary[name] for name in JUDGES}
    neutral_gaps = {name: neutral[name] - ordinary[name] for name in JUDGES}
    improvements = {name: particle[name] - neutral[name] for name in JUDGES}
    endpoints = tuple(name for name in JUDGES if name.endswith("@6400"))
    reproduced = all(gaps[name] > 1e-4 for name in endpoints)
    reductions = {name: improvements[name] / gaps[name] for name in endpoints} if reproduced else None
    retained = {}
    for arm in law.ARMS[1:]:
        witness = witnesses[arm]
        retained[arm] = (witness["bridge_still_trainable"]
                         and witness["bank_still_trainable"] and witness["router_still_trainable"]
                         and set(witness["C_norms"]) == set(law.SITES)
                         and all(value > 0 for value in witness["C_norms"].values())
                         and witness["live_bank_updates"] > 0 and witness["live_query_updates"] > 0
                         and set(witness["zero_code_minus_live_test_game"]) == set(JUDGES)
                         and all(abs(value) > 1e-6 for value in witness["zero_code_minus_live_test_game"].values()))
    return {"original_particle_minus_ordinary_game": gaps,
            "neutral_particle_minus_ordinary_game": neutral_gaps,
            "paired_game_H_b_improvement": improvements,
            "original_particle_gap_reproduced": reproduced,
            "endpoint_gap_reduction": reductions,
            "support_gate_applicable": reproduced,
            "retained_particle_gate": retained,
            "H_b_support_gate": (reproduced and all(value > 1e-4 for value in improvements.values())
                                  and all(value >= .5 for value in reductions.values())
                                  and retained[law.ARMS[2]]),
            "neutral_beats_ordinary_all_four": (all(value < -1e-4 for value in neutral_gaps.values())
                                               and retained[law.ARMS[2]]),
            "remaining_neutral_gap_witness": (reproduced and all(neutral_gaps[name] > 1e-4 for name in endpoints)
                                               and retained[law.ARMS[2]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve existing artifacts; choose a fresh output directory")
    card = json.loads(CARD.read_text())
    validate_contract(card)
    if not card["execution"].get("execution_authorized", False):
        parser.error("source-only task is held pending the full Supra fixed-endpoint readout")
    if Path(particlegan.__file__).resolve().parent != ROOT / "particlegan":
        parser.error("set PYTHONPATH to this checkout")
    native_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", "particlegan"], cwd=ROOT, text=True).strip():
        parser.error("native package has uncommitted changes")
    if subprocess.run(["git", "diff", "--quiet", card["native_base_revision"], "--", "particlegan"], cwd=ROOT).returncode:
        parser.error("native package differs from the frozen parent")
    torch.set_num_threads(1)
    execution = card["execution"]
    bindings = {"source_hashes": source_hashes(), "native_revision": native_revision,
                "native_base_revision": card["native_base_revision"], "native_source_hash": common.native_source_hash()}
    args.out.mkdir(parents=True)
    started = time.monotonic()
    log = (args.out / "run.log").open("w", buffering=1)
    receipt = {"schema": "routed_convergence_rotated_execution_v1", "task": law.TASK,
               "status": "running", "qualification_status": "awaiting_independent_review",
               "qualification_credit": "none", "contract": card, "bindings": bindings,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "device": "cpu", "threads": 1, "platform": platform.platform()}, "arms": {}}

    def emit(**row):
        line = json.dumps(common.json_value(row), allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)

    def budget(arm_started=None):
        if source_hashes() != bindings["source_hashes"] or common.native_source_hash() != bindings["native_source_hash"]:
            raise RuntimeError("held source/card/native bytes changed during execution")
        if time.monotonic() - started > execution["total_wall_budget_seconds"]:
            raise TimeoutError("total diagnostic budget exhausted")
        if arm_started is not None and time.monotonic() - arm_started > execution["arm_wall_budget_seconds"]:
            raise TimeoutError("arm diagnostic budget exhausted")

    try:
        receipt["source_archive"] = archive_sources(args.out, bindings)
        data = law.make_rotated_data()
        common.save(args.out / "data.pt", data)
        receipt["data_file_sha256"] = common.sha(args.out / "data.pt")
        initial_data_digest = base.digest(data)
        receipt.update(data_digest=data["digest"], data_hashes={pool: base.digest(data[pool]) for pool in base.SPLITS},
                       raw_coordinate_scale=data["raw_scale"].tolist(),
                       coordinate_scale_floor_count=int((data["raw_scale"] < .04).sum()),
                       reachability=base.reachability_witness(data),
                       geometry_witness=law.feasibility_witness(),
                       initialization_witness=law.neutral.initial_difference_witness(data, law.neutral.make_neutral_data(data)))
        common.write_json(args.out / "provenance.json", receipt)
        emit(event="frozen", data_digest=data["digest"], rho=law.RHO, steps=6400,
             sources=bindings["source_hashes"], reachability=receipt["reachability"])
        shared_draws = []
        for arm in law.ARMS:
            budget()
            arm_started = time.monotonic()
            loop = law.make_rotated_loop(arm, data, bindings=bindings)
            directory = args.out / arm
            directory.mkdir()
            frozen = base.digest(common.frozen_values(loop))
            input_digest = base.digest(loop.data)
            record = {"law": loop.law, "status": "running", "checkpoints": [],
                      "initial_frozen_owner_digest": frozen, "initial_data_digest": input_digest,
                      "initial_model_digest": base.digest(loop.G.state_dict()),
                      "initial_critic_digest": base.digest(loop.policy.D.state_dict()),
                      "initial_table_digest": base.digest(loop.policy.table),
                      "initial_router_digest": None if loop.policy.router is None else base.digest(loop.policy.router.state_dict()),
                      "active_parameter_counts": {role: sum(p.numel() for p in model.parameters() if p.requires_grad)
                                                  for role, model in base.modules(loop).items()},
                      "coverage": {"live_bank_updates": 0, "live_query_updates": 0, "row_control_events": 0,
                                   "proposal_events": 0, "candidate_proposals": 0, "moves": 0}}
            record["active_parameter_counts"].update(table=loop.policy.table.numel() if loop.policy.table.requires_grad else 0, noise=1)
            if arm in law.ARMS[1:]:
                spec = loop.policy.routed_control.spec
                if spec.max_context_harm != 0 or spec.output_error_guard or spec.sites != law.SITES:
                    raise AssertionError("routed game/guard/site law changed")
                record["structural_guard"] = {"max_feature_context_harm": spec.max_context_harm,
                                               "output_error_guard": spec.output_error_guard, "sites": list(spec.sites),
                                               "recipe": loop.policy.recipe.to_dict()}
            receipt["arms"][arm] = record

            def save_state(step):
                common.assert_finite(loop)
                if base.digest(common.frozen_values(loop)) != frozen or base.digest(loop.data) != input_digest:
                    raise AssertionError("frozen FAST/EMA owner or task data changed")
                path = directory / f"step-{step:04d}.pt"
                state = base.checkpoint(loop)
                common.save(path, state)
                record["checkpoints"].append({"step": step, "file": path.name, "sha256": common.sha(path),
                                              "state_digest": base.digest(state)})

            save_state(0)
            recovery_rows = []
            with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                for step in range(1, 6401):
                    if step % 25 == 1:
                        budget(arm_started)
                        common.assert_finite(loop)
                    row = base.update(loop)
                    for key in ("loss_g", "loss_d_game", "penalty", "bank_gradient_norm", "query_gradient_norm"):
                        if not math.isfinite(row[key]):
                            raise FloatingPointError("nonfinite training observation: " + key)
                    draws = {key: row[key] for key in ("batch_indices", "paired_base_digest", "data_rng", "paired_rng")}
                    if arm == law.ARMS[0]:
                        shared_draws.append(draws)
                    elif draws != shared_draws[step - 1]:
                        raise AssertionError("matched native arms consumed different fit or paired-base streams")
                    coverage = record["coverage"]
                    coverage["live_bank_updates"] += int(row["bank_gradient_rows"] > 0)
                    coverage["live_query_updates"] += int(row["query_gradient_norm"] > 0)
                    coverage["row_control_events"] += int(row["move"] is not None)
                    coverage["moves"] += int((row["move"] or {}).get("moves", 0))
                    routing = row["controls"]["routing"]
                    proposals = 0 if routing is None else routing["counters"]["proposals"]
                    coverage["proposal_events"] += int(proposals > coverage["candidate_proposals"])
                    coverage["candidate_proposals"] = proposals
                    trace.write(json.dumps(common.json_value(row), allow_nan=False) + "\n")
                    if step in (801, 802):
                        recovery_rows.append(deepcopy(row))
                    if step in checkpoint_steps():
                        save_state(step)
                        emit(event="progress", arm=arm, step=step, seconds=time.monotonic() - arm_started,
                             loss_g=row["loss_g"], loss_d_game=row["loss_d_game"],
                             bank_gradient_rows=row["bank_gradient_rows"], sigma=row["controls"]["actual_sigma"],
                             rates={v["role"]: {"applied_lr": v["applied_lr"], "raw_scale": v["tester"]["s"]}
                                    for v in row["controls"]["groups"].values()},
                             fires=row["controls"]["surprise"]["fires"], moves=coverage["moves"])
            live = base.digest(base.checkpoint(loop))
            recovery_started = time.monotonic()
            with torch.random.fork_rng(devices=[]):
                replay = law.make_rotated_loop(arm, data, bindings=bindings)
                base.restore(replay, torch.load(directory / "step-0800.pt", weights_only=False))
                rows = [base.update(replay), base.update(replay)]
                target = torch.load(directory / "step-0802.pt", weights_only=False)
                if base.digest(rows) != base.digest(recovery_rows) or base.digest(base.checkpoint(replay)) != base.digest(target):
                    raise AssertionError("own-state 800-to-802 rows/state replay differs")
            if live != base.digest(base.checkpoint(loop)):
                raise AssertionError("recovery witness changed the authoritative final state")
            record["recovery_witness"] = {"from": 800, "to": 802, "rows_exact": True, "state_exact": True,
                                          "final_state_unchanged": True, "software_updates": 2,
                                          "wall_seconds": time.monotonic() - recovery_started}
            record.update(status="complete", training_and_recovery_seconds=time.monotonic() - arm_started,
                          final_checkpoint_digest=live, final_frozen_owner_digest=base.digest(common.frozen_values(loop)))
            if arm in law.ARMS[1:]:
                record["particle_witness"] = {
                    "C_norms": {site: float(getattr(loop.G, site).bridge.weight[:, base.RANK:].detach().norm()) for site in law.SITES},
                    "bridge_still_trainable": all(getattr(loop.G, site).bridge.weight.requires_grad
                                                  and getattr(loop.G, site).bridge.bias.requires_grad for site in law.SITES),
                    "bank_still_trainable": loop.policy.table.requires_grad,
                    "router_still_trainable": all(parameter.requires_grad for parameter in loop.policy.router.parameters()),
                    "final_bank_gradient_rows": row["bank_gradient_rows"],
                    "final_query_gradient_norm": row["query_gradient_norm"],
                    "bank_changed": base.digest(loop.policy.table) != record["initial_table_digest"],
                    "router_changed": base.digest(loop.policy.router.state_dict()) != record["initial_router_digest"],
                    **record["coverage"]}
            budget(arm_started)
            common.write_json(args.out / "provenance.json", receipt)
        if base.digest(data) != initial_data_digest:
            raise AssertionError("training changed the common teacher/input data")
        panels = base.evaluation_panels(data)
        receipt["private_panel_digest"] = base.digest(panels)
        judges = {}
        for name in JUDGES:
            arm, step = name.split("@")
            state = torch.load(args.out / arm / f"step-{int(step):04d}.pt", weights_only=False)
            judges[name] = load_judge(state, data, expected_step=int(step))
        receipt["judges"] = {name: base.digest(judge.state_dict()) for name, judge in judges.items()}
        receipt["judge_reference_path"] = {
            name: {pool: {str(fraction): common.compact(base.score_residual(judge, data[pool]["context"],
                         fraction * (data[pool]["base"] - data[pool]["targets"]) / data["scale"], panels[pool]))
                         for fraction in (0, .25, .5, 1)} for pool in base.SPLITS} for name, judge in judges.items()}
        receipt["endpoint_scores"], receipt["initial_scores"], receipt["code_ablation"] = {}, {}, {}
        with (args.out / "common-judge-curves.jsonl").open("w", buffering=1) as curves:
            for arm in law.ARMS:
                evaluation_started = time.monotonic()
                loop = law.make_rotated_loop(arm, data, bindings=bindings)
                for step in curve_steps():
                    budget()
                    if receipt["arms"][arm]["training_and_recovery_seconds"] + time.monotonic() - evaluation_started > execution["arm_wall_budget_seconds"]:
                        raise TimeoutError("arm budget exhausted during offline scoring")
                    base.restore(loop, torch.load(args.out / arm / f"step-{step:04d}.pt", weights_only=False))
                    scores = immutable_scores(loop, judges, panels)
                    curves.write(json.dumps({"arm": arm, "step": step, "scores": scores}, allow_nan=False) + "\n")
                    small = compact_scores(scores)
                    if step == 0:
                        receipt["initial_scores"][arm] = small
                    if step in ENDPOINTS:
                        receipt["endpoint_scores"][f"{arm}@{step}"] = small
                    emit(event="scored", arm=arm, step=step,
                         test_game={name: pools["test"]["paired_game"] for name, pools in small.items()})
                if arm in law.ARMS[1:]:
                    ablation = compact_scores(immutable_scores(loop, judges, panels, code_ablation=True))
                    receipt["code_ablation"][arm] = ablation
                    receipt["arms"][arm]["particle_witness"]["zero_code_minus_live_test_game"] = {
                        name: ablation[name]["test"]["paired_game"]
                              - receipt["endpoint_scores"][f"{arm}@6400"][name]["test"]["paired_game"] for name in JUDGES}
                record = receipt["arms"][arm]
                record["evaluation_seconds"] = time.monotonic() - evaluation_started
                record["cumulative_seconds"] = record["training_and_recovery_seconds"] + record["evaluation_seconds"]
                if record["cumulative_seconds"] > execution["arm_wall_budget_seconds"]:
                    raise TimeoutError("arm budget exhausted during final evaluation")
        receipt["gates"] = final_gates(receipt["endpoint_scores"],
                                     {arm: receipt["arms"][arm]["particle_witness"] for arm in law.ARMS[1:]})
        budget()
        receipt.update(status="complete", wall_seconds=time.monotonic() - started,
                       independent_review_budget_remaining_seconds=execution["total_wall_budget_seconds"] - (time.monotonic() - started))
        common.write_json(args.out / "receipt.json", receipt)
        compact_report = {key: receipt[key] for key in ("schema", "task", "status", "qualification_status", "qualification_credit",
                                                       "bindings", "data_digest", "reachability", "geometry_witness",
                                                       "private_panel_digest", "judges", "judge_reference_path",
                                                       "endpoint_scores", "gates", "wall_seconds")}
        compact_report["arm_summary"] = {
            arm: {key: record[key] for key in ("active_parameter_counts", "recovery_witness",
                                               "initial_frozen_owner_digest", "final_frozen_owner_digest",
                                               "final_checkpoint_digest", "training_and_recovery_seconds", "evaluation_seconds")}
            for arm, record in receipt["arms"].items()}
        compact_report["particle_witnesses"] = {arm: receipt["arms"][arm]["particle_witness"] for arm in law.ARMS[1:]}
        compact_report["checkpoint_counts"] = {arm: len(record["checkpoints"]) for arm, record in receipt["arms"].items()}
        compact_report["checkpoint_manifest_digest"] = base.digest({arm: record["checkpoints"] for arm, record in receipt["arms"].items()})
        compact_report["execution_receipt_sha256"] = common.sha(args.out / "receipt.json")
        compact_report["limits"] = card["limits"]
        common.write_json(args.out / "compact-report.json", compact_report)
        emit(event="complete", seconds=receipt["wall_seconds"], gates=receipt["gates"], qualification=receipt["qualification_status"])
    except Exception as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "error",
                       error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
        common.write_json(args.out / "receipt.json", receipt)
        emit(event=receipt["status"], error=receipt["error"])
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
