"""Independently qualify one fixed rotated-teacher diagnostic, never train it.

The only optimizer work is six software updates replaying each saved 800->802
state. All quality scores use actual clean FAST states and four actual baseline
critics; execution aggregates are comparison targets. --source-only checks
readiness without granting scientific qualification.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

if __package__:
    from . import e22_routed_convergence as base
    from . import e22_routed_convergence_rotated_teacher as law
    from . import run_e22_routed_convergence_rotated_teacher as runner
else:
    import e22_routed_convergence as base
    import e22_routed_convergence_rotated_teacher as law
    import run_e22_routed_convergence_rotated_teacher as runner


ROOT = Path(__file__).resolve().parents[1]
REVIEWER = Path(__file__).resolve()
JUDGES = tuple(f"{arm}@{step}" for arm in law.ARMS[:2] for step in (800, 6400))
CHECKPOINT_STEPS = tuple(sorted({0, 802, 5120, *range(200, 6401, 200)}))
CURVE_STEPS = tuple(step for step in CHECKPOINT_STEPS if step != 802)
NOMINAL_RATES = {"generator": 5e-5, "router": 5e-5, "table": .0085,
                 "critic": .00425, "noise": .00425}


class Review:
    def __init__(self):
        self.checks = 0

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def same(self, actual, expected, label):
        self.require(base.digest(actual) == base.digest(expected), label)

    def close(self, actual, expected, label, *, atol=2e-7, rtol=2e-7):
        self.require(math.isfinite(float(actual)) and math.isfinite(float(expected))
                     and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), label)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def native_hash():
    return base.digest({str(path.relative_to(ROOT)): sha(path)
                        for path in sorted((ROOT / "particlegan").rglob("*.py"))})


def finite_tensors(review, value, label):
    """Require finite numerical owners, not intentionally absent diagnostic ratios."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            review.require(torch.isfinite(value).all().item(), label)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tensors(review, item, label + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_tensors(review, item, label + "/" + str(index))


def diagnostic_nonfinite(value, path=""):
    """Preserve native sentinel counts without treating them as parameter failures."""
    if isinstance(value, torch.Tensor):
        return [path] if value.is_floating_point() and not torch.isfinite(value).all() else []
    if isinstance(value, float):
        return [path] if not math.isfinite(value) else []
    if isinstance(value, dict):
        return [name for key, item in value.items() for name in diagnostic_nonfinite(item, path + "/" + str(key))]
    if isinstance(value, (tuple, list)):
        return [name for index, item in enumerate(value) for name in diagnostic_nonfinite(item, path + "/" + str(index))]
    return []


def frozen_owners(state):
    training = state["training"]
    values = {}
    for family in ("models", "averages"):
        for role, tensors in training[family].items():
            for name, tensor in tensors.items():
                if role == "encoder" or name.endswith("base.weight") or name == "scale":
                    values[family + "/" + role + "/" + name] = tensor
    values["critic_optimizer_ema_scale"] = training["optimizers"][1]["regularizer"]["ema"]["scale"]
    if not training["table_requires_grad"]:
        values["ordinary_frozen_prior"] = training["table"]
    return values


def json_native(value):
    if isinstance(value, torch.Tensor):
        return json_native(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": repr(value)}
    return value


def expected_streams(data):
    """Recreate every fit draw and raw paired base using declared streams only."""
    indices_rng = torch.Generator(device="cpu").manual_seed(7)
    paired_rng = torch.Generator(device="cpu").manual_seed(43)
    draws, checkpoints = [], {0: (indices_rng.get_state(), paired_rng.get_state())}
    for step in range(1, 6401):
        indices = torch.randint(len(data["fit"]["context"]), (4,), generator=indices_rng)
        shape = (4, 16, 16)
        critic = torch.randn(shape, generator=paired_rng)
        generator = torch.randn(shape, generator=paired_rng)
        draws.append({"step": step, "batch_indices": indices.tolist(),
                      "paired_base_digest": base.digest((critic, generator)),
                      "data_rng": base.digest(indices_rng.get_state()),
                      "paired_rng": base.digest(paired_rng.get_state())})
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = (indices_rng.get_state(), paired_rng.get_state())
    return draws, checkpoints


def check_trace_row(review, row, expected, arm):
    for key, item in expected.items():
        review.require(row.get(key) == item, arm + "/actual named draw/" + key)
    review.require("loss_reference_mse" not in row, arm + "/no output-MSE training arm")
    for key in ("loss_g", "loss_d_game", "penalty", "bank_gradient_norm", "query_gradient_norm"):
        review.require(isinstance(row[key], (int, float)) and math.isfinite(row[key]), arm + "/finite " + key)
    groups = row["controls"]["groups"]
    expected_roles = {"generator", "critic", "noise"} | ({"router", "table"} if arm != law.ARMS[0] else set())
    review.require({group["role"] for group in groups.values()} == expected_roles, arm + "/all native owners")
    for group in groups.values():
        role = group["role"]
        review.close(group["base_lr"], NOMINAL_RATES[role], arm + "/nominal " + role, atol=1e-15, rtol=0)
        review.require(math.isfinite(group["applied_lr"]) and 0 <= group["applied_lr"] <= group["base_lr"] * (1 + 1e-12),
                       arm + "/bounded applied " + role)
        review.close(group["effective_scale"], group["applied_lr"] / group["base_lr"], arm + "/rate attribution")
    ka2 = row["controls"]["ka2"]
    review.require(ka2["formulation"] == "ka2" and ka2["calls"] == row["step"]
                   and ka2["observed_steps"] == row["step"], arm + "/one native KA2 call per update")
    review.require((row["controls"]["routing"] is None) == (arm == law.ARMS[0]), arm + "/actual routed control")


def check_state(review, state, initial, expected_law, arm, step, stream_state):
    training = state["training"]
    review.require(set(state) == {"law", "step", "training", "data_rng", "paired_rng", "modes", "gradients"}, arm + "/complete checkpoint envelope")
    review.require(state["step"] == step == training["completed_steps"], arm + "/actual native step")
    review.require(state["law"] == expected_law, arm + "/entire recipe/init/source task law")
    for key in ("recipe", "roles", "initial_lrs", "requires_grad"):
        review.same(training[key], initial["training"][key], arm + "/unchanged " + key)
    review.same(frozen_owners(state), frozen_owners(initial), arm + "/all frozen FAST/EMA owners")
    review.same(state["data_rng"], stream_state[0], arm + "/saved CPU fit stream")
    review.same(state["paired_rng"], stream_state[1], arm + "/saved raw paired stream")
    for field in ("models", "averages", "table", "averaged_table", "output_noise"):
        finite_tensors(review, training[field], arm + "/finite " + field)
    finite_tensors(review, state["gradients"], arm + "/finite actual gradients")
    for optimizer in training["optimizers"]:
        finite_tensors(review, optimizer["state"], arm + "/finite optimizer moments")
        if "ema" in optimizer["regularizer"]:
            finite_tensors(review, optimizer["regularizer"]["ema"], arm + "/finite complete EMA critic")
        for group in optimizer["param_groups"]:
            review.require(tuple(group["betas"]) == (0., .999) and group["amsgrad"] and group["weight_decay"] == 0,
                           arm + "/native AMSGrad beta and decay")
    if arm != law.ARMS[0]:
        config = training["routing"]["config"]
        review.require(config["model_forward"] and tuple(config["sites"]) == law.SITES
                       and config["max_context_harm"] == 0 and config["probe_interval"] == 100,
                       arm + "/whole-model sites, per-site controller and zero feature harm")
        review.require(training["table_requires_grad"] and training["recipe"]["particle_birth_death"]
                       and training["recipe"]["row_evidence_gate"]
                       and training["recipe"]["row_policy"] == "routed_paired", arm + "/full native particle policy")
        review.require(training["recipe"]["birth_death_backend"] == "auto"
                       and training["recipe"]["reopen_guard"] == "settled", arm + "/PR223 shared profile")
    else:
        review.require(not training["table_requires_grad"] and training["routing"] is None
                       and not training["recipe"]["particle_birth_death"]
                       and not training["recipe"]["row_evidence_gate"], arm + "/frozen ordinary scaffold")


@torch.no_grad()
def score_residual(judge, context, residual, panels):
    """Independent reductions; stored paired-game means are never scoring inputs."""
    condition = context[:, 0, 16:]
    per_draw = [F.softplus(-(judge(panel + residual, condition) - judge(panel, condition))).flatten()
                for panel in panels]
    per_context = torch.stack(per_draw).mean(0)
    zeros = torch.zeros_like(residual)
    clean = F.softplus(-(judge(residual, condition) - judge(zeros, condition))).mean()
    features = (judge.features(residual, condition) - judge.features(zeros, condition)).square().mean((1, 2))
    return {"paired_game": float(torch.stack([values.mean() for values in per_draw]).mean()),
            "paired_game_by_context": per_context.tolist(),
            "clean_zero_noise_game": float(clean), "feature_proxy": float(features.mean()),
            "teacher_zero_residual_anchor": float(F.softplus(torch.zeros_like(judge(panels[0], condition))).mean())}


@torch.no_grad()
def score_prediction(judge, data, pool, prediction, panels):
    values = data[pool]
    result = score_residual(judge, values["context"], (prediction - values["targets"]) / data["scale"], panels[pool])
    result["output_rmse_diagnostic"] = float((prediction - values["targets"]).square().mean().sqrt())
    return result


def compare_score(review, actual, expected, data, pool, label, *, compact=False):
    required = {"paired_game", "clean_zero_noise_game", "feature_proxy", "teacher_zero_residual_anchor"}
    if "output_rmse_diagnostic" in actual:
        required.add("output_rmse_diagnostic")
    review.require(required <= set(actual), label + "/complete score observations")
    for key in required:
        review.close(actual[key], expected[key], label + "/independent " + key)
    review.close(actual["teacher_zero_residual_anchor"], math.log(2), label + "/zero-residual anchor")
    if not compact:
        items = actual["paired_game_by_context"]
        review.require(len(items) == len(data[pool]["context"]), label + "/entire pool")
        review.require(all(math.isfinite(value) for value in items), label + "/finite paired contexts")
        review.require(torch.allclose(torch.tensor(items, dtype=torch.float64),
                                      torch.tensor(expected["paired_game_by_context"], dtype=torch.float64), rtol=2e-7, atol=2e-7),
                       label + "/actual context score equality")
        review.close(sum(items) / len(items), actual["paired_game"], label + "/context mean reduction")


def by_subject(score, data, pool):
    items = torch.tensor(score["paired_game_by_context"], dtype=torch.float64)
    subjects = data[pool]["subjects"]
    return {str(subject): float(items[subjects == subject].mean()) for subject in range(6)}


@torch.no_grad()
def predictions(loop, data, *, zero_code=False):
    return {pool: torch.cat([base.forward(loop, data[pool]["context"][start:start + 4], code_ablation=zero_code)
                            for start in range(0, len(data[pool]["context"]), 4)])
            for pool in base.SPLITS}


def compact_scores(scores):
    return {judge: {pool: {key: value for key, value in score.items() if key != "paired_game_by_context"}
                    for pool, score in pools.items()} for judge, pools in scores.items()}


def independent_gates(endpoint_scores, witnesses):
    """Recompute signed comparisons and denominator guards independently."""
    ordinary, particle, neutral = [endpoint_scores[f"{arm}@6400"] for arm in law.ARMS]
    gaps = {judge: particle[judge]["test"]["paired_game"] - ordinary[judge]["test"]["paired_game"] for judge in JUDGES}
    neutral_gaps = {judge: neutral[judge]["test"]["paired_game"] - ordinary[judge]["test"]["paired_game"] for judge in JUDGES}
    gains = {judge: particle[judge]["test"]["paired_game"] - neutral[judge]["test"]["paired_game"] for judge in JUDGES}
    endpoints = tuple(judge for judge in JUDGES if judge.endswith("@6400"))
    applicable = all(gaps[judge] > 1e-4 for judge in endpoints)
    reductions = {judge: gains[judge] / gaps[judge] for judge in endpoints} if applicable else None
    retained = {arm: (witness["bridge_still_trainable"] and witness["bank_still_trainable"]
                     and witness["router_still_trainable"] and set(witness["C_norms"]) == set(law.SITES)
                     and all(value > 0 for value in witness["C_norms"].values())
                     and witness["live_bank_updates"] > 0 and witness["live_query_updates"] > 0
                     and set(witness["zero_code_minus_live_test_game"]) == set(JUDGES)
                     and all(abs(value) > 1e-6 for value in witness["zero_code_minus_live_test_game"].values()))
                for arm, witness in witnesses.items()}
    return {"original_particle_minus_ordinary_game": gaps, "neutral_particle_minus_ordinary_game": neutral_gaps,
            "paired_game_H_b_improvement": gains, "original_particle_gap_reproduced": applicable,
            "endpoint_gap_reduction": reductions, "support_gate_applicable": applicable, "retained_particle_gate": retained,
            "H_b_support_gate": bool(applicable and all(value > 1e-4 for value in gains.values())
                                      and all(value >= .5 for value in reductions.values()) and retained[law.ARMS[2]]),
            "neutral_beats_ordinary_all_four": bool(all(value < -1e-4 for value in neutral_gaps.values()) and retained[law.ARMS[2]]),
            "remaining_neutral_gap_witness": bool(applicable and all(neutral_gaps[judge] > 1e-4 for judge in endpoints)
                                                   and retained[law.ARMS[2]])}


def source_readiness():
    review = Review()
    card = read(runner.CARD)
    runner.validate_contract(card)
    review.require(native_hash() == card["native_python_source_digest"], "actual native package digest")
    review.require(len(CHECKPOINT_STEPS) == 35 and len(CURVE_STEPS) == 34, "fixed checkpoint and curve denominator")
    review.require(tuple(card["evaluation"]["mandatory_common_judges"]) == JUDGES, "all four actual common judges")
    review.require(tuple(card["evaluation"]["endpoints"]) == (5120, 6400), "both mandatory endpoints")
    data = law.make_rotated_data()
    witness = law.feasibility_witness()
    review.require(witness["optimizer_updates"] == 0 and witness["data_digest"] == data["digest"], "zero-update reachability and geometry")
    review.require(all(value for pools in witness["reachability"].values() for value in pools.values()), "both family witnesses on every pool")
    return {"status": "source_ready", "qualified": False, "qualification_credit": "none", "optimizer_updates": 0,
            "checks": review.checks, "reviewer_sha256": sha(REVIEWER), "source_hashes": runner.source_hashes(),
            "native_source_digest": native_hash(), "data_digest": data["digest"], "geometry_witness": witness,
            "quality_evidence": "not executed or reviewed; conditional task only",
            "execution_authorized": card["execution"]["execution_authorized"]}


def review_run(directory):
    started = time.monotonic()
    review, artifacts = Review(), {}
    directory = Path(directory).resolve()
    receipt_path = directory / "receipt.json"
    receipt = read(receipt_path)
    artifacts["receipt.json"] = sha(receipt_path)
    card = receipt["contract"]
    runner.validate_contract(card)
    review.require(card == read(runner.CARD), "receipt contract equals the exact held card")
    review.require(receipt["schema"] == "routed_convergence_rotated_execution_v1" and receipt["task"] == law.TASK,
                   "actual standalone task identity")
    review.require(receipt["status"] == "complete" and card["execution"]["execution_authorized"] is True,
                   "complete authorized fixed execution; source-only evidence cannot qualify")
    review.require(receipt["qualification_credit"] == "none" and receipt["qualification_status"] == "awaiting_independent_review",
                   "execution receipt confers no prior qualification")
    review.require(receipt["runtime"]["device"] == "cpu" and receipt["runtime"]["threads"] == 1
                   and receipt["runtime"]["torch"] == torch.__version__, "exact CPU replay runtime")
    bindings = receipt["bindings"]
    review.same(bindings["source_hashes"], runner.source_hashes(), "every held source/card byte")
    review.require(bindings["native_base_revision"] == card["native_base_revision"]
                   and bindings["native_source_hash"] == native_hash(), "held native source identity")
    source_snapshot = directory / "source"
    expected_archive = {**bindings["source_hashes"],
                        **{"native/" + str(path.relative_to(ROOT)): sha(path)
                           for path in sorted((ROOT / "particlegan").rglob("*.py"))}}
    review.same(receipt["source_archive"], expected_archive, "entire actual source archive manifest")
    for relative, expected in expected_archive.items():
        path = source_snapshot / relative
        review.require(path.is_file() and sha(path) == expected, "archived held source/" + relative)
        artifacts[str(path.relative_to(directory))] = expected
    data = law.make_rotated_data()
    data_path = directory / "data.pt"
    review.require(data_path.is_file(), "archived actual fixture tensors")
    artifacts["data.pt"] = sha(data_path)
    review.require(receipt["data_file_sha256"] == artifacts["data.pt"], "actual fixture file SHA")
    review.same(load(data_path), data, "actual archived fixture equals declared reconstruction")
    review.require(receipt["data_digest"] == data["digest"], "actual task data digest")
    review.same(receipt["data_hashes"], {pool: base.digest(data[pool]) for pool in base.SPLITS}, "all pool hashes")
    review.same(receipt["geometry_witness"], law.feasibility_witness(), "actual geometric and reachability witnesses")
    review.same(receipt["reachability"], base.reachability_witness(data), "all six complete family reachability witnesses")
    draws, stream_states = expected_streams(data)
    initial, loops, trace_summaries, saved_states, sentinels = {}, {}, {}, {}, {}

    def budget():
        total = receipt["wall_seconds"] + time.monotonic() - started
        review.require(total <= card["execution"]["total_wall_budget_seconds"],
                       "execution plus independent review within 2700 seconds")

    for arm in law.ARMS:
        budget()
        record = receipt["arms"][arm]
        review.require(record["status"] == "complete" and 0 < record["cumulative_seconds"] <= 900,
                       arm + "/complete fixed arm within declared budget")
        manifest = {entry["step"]: entry for entry in record["checkpoints"]}
        review.require(len(record["checkpoints"]) == 35 and set(manifest) == set(CHECKPOINT_STEPS),
                       arm + "/all 35 actual states")
        directory_arm = directory / arm
        review.require({path.name for path in directory_arm.glob("step-*.pt")}
                       == {f"step-{step:04d}.pt" for step in CHECKPOINT_STEPS},
                       arm + "/exact state files and no selected-only denominator")
        initial[arm] = load(directory_arm / "step-0000.pt")
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial[arm]["training"]["cpu_rng"])
            loop = law.make_rotated_loop(arm, data, bindings=bindings)
            review.same(base.checkpoint(loop), initial[arm], arm + "/entire actual fresh native initialization")
        loops[arm] = loop
        spec = loop.policy.routed_control.spec if arm != law.ARMS[0] else None
        if spec is not None:
            review.require(not spec.output_error_guard and tuple(spec.sites) == law.SITES,
                           arm + "/actual output guards disabled and sequential sites")
        checkpoints = {}
        for step in CHECKPOINT_STEPS:
            path = directory_arm / f"step-{step:04d}.pt"
            state = load(path)
            actual_sha = sha(path)
            entry = manifest[step]
            review.require(entry["file"] == path.name and entry["sha256"] == actual_sha
                           and entry["state_digest"] == base.digest(state),
                           arm + "/actual state hash and full owner digest")
            artifacts[str(path.relative_to(directory))] = actual_sha
            check_state(review, state, initial[arm], loop.law, arm, step, stream_states[step])
            base.restore(loop, state)
            review.same(base.checkpoint(loop), state, arm + "/complete model/optimizer/controller/stream restore")
            diagnostic = base.controls(loop.policy)
            review.require(diagnostic["ka2"]["calls"] == step, arm + "/saved native KA2 call counter")
            checkpoints[step] = {"state_digest": base.digest(state), "controls": json_native(diagnostic),
                                 "dv12_rng": base.digest(state["training"]["streams"]["noise_generator"])}
            if step in (800, 802, 6400):
                saved_states[arm, step] = state
            sentinels[arm + "@" + str(step)] = diagnostic_nonfinite(state["training"]["lr_settle"])
        trace_path = directory_arm / "trace.jsonl"
        artifacts[str(trace_path.relative_to(directory))] = sha(trace_path)
        summary = {"live_bank_updates": 0, "live_query_updates": 0, "row_control_events": 0,
                   "proposal_events": 0, "candidate_proposals": 0, "moves": 0, "recovery_rows": []}
        count = 0
        with trace_path.open() as stream:
            for count, line in enumerate(stream, 1):
                review.require(count <= 6400, arm + "/no undeclared extra quality updates")
                row = json.loads(line)
                check_trace_row(review, row, draws[count - 1], arm)
                summary["live_bank_updates"] += int(row["bank_gradient_rows"] > 0)
                summary["live_query_updates"] += int(row["query_gradient_norm"] > 0)
                summary["row_control_events"] += int(row["move"] is not None)
                summary["moves"] += int((row["move"] or {}).get("moves", 0))
                routed = row["controls"]["routing"]
                proposals = 0 if routed is None else routed["counters"]["proposals"]
                summary["proposal_events"] += int(proposals > summary["candidate_proposals"])
                summary["candidate_proposals"] = proposals
                if count in checkpoints:
                    review.same(row["controls"], checkpoints[count]["controls"], arm + "/native controls match saved state")
                    review.require(row["dv12_rng"] == checkpoints[count]["dv12_rng"], arm + "/private per-site DV12 stream")
                if count in (801, 802):
                    summary["recovery_rows"].append(row)
                if count == 6400:
                    summary["final_bank_gradient_rows"] = row["bank_gradient_rows"]
                    summary["final_query_gradient_norm"] = row["query_gradient_norm"]
        review.require(count == 6400, arm + "/entire 6400-update trace")
        review.same(record["coverage"], {key: summary[key] for key in record["coverage"]},
                    arm + "/actual gradient/control coverage")
        review.require(record["final_checkpoint_digest"] == checkpoints[6400]["state_digest"], arm + "/actual final owner state")
        trace_summaries[arm] = summary
    expected = deepcopy(initial[law.ARMS[1]]["training"])
    for family in ("models", "averages"):
        for name, value in expected[family]["generator"].items():
            if name.endswith("bridge.weight"):
                value[:, :2].zero_()
            elif name.endswith("bridge.bias"):
                value.zero_()
    review.same(expected, initial[law.ARMS[2]]["training"], "actual original/neutral native initial states differ only in H/b")
    for arm in law.ARMS:
        training = initial[arm]["training"]
        for site in law.SITES:
            review.same(training["models"]["generator"][site + ".down.weight"],
                        data["initial_ordinary"][site + ".down.weight"], arm + "/common initial down")
            review.require(training["models"]["generator"][site + ".up.weight"].count_nonzero().item() == 0,
                           arm + "/zero initial up")
        review.same(training["models"]["critic"], initial[law.ARMS[0]]["training"]["models"]["critic"], arm + "/common initial D")
    replay_records = {}
    for arm in law.ARMS:
        budget()
        replay_started = time.monotonic()
        with torch.random.fork_rng(devices=[]):
            replay = law.make_rotated_loop(arm, data, bindings=bindings)
            base.restore(replay, saved_states[arm, 800])
            rows = [base.update(replay), base.update(replay)]
            review.same(json_native(rows), trace_summaries[arm]["recovery_rows"], arm + "/independent actual 801/802 diagnostic rows")
            review.same(base.checkpoint(replay), saved_states[arm, 802], arm + "/independent all-owner 800->802 replay")
        replay_records[arm] = {"from": 800, "to": 802, "software_updates": 2, "rows_exact": True,
                               "state_exact": True, "wall_seconds": time.monotonic() - replay_started}
    panels = base.evaluation_panels(data)
    review.require(base.digest(panels) == receipt["private_panel_digest"], "actual private Gaussian panels")
    panel_binding = base.digest(panels)
    judges = {}
    for name in JUDGES:
        arm, step = name.split("@")
        with torch.random.fork_rng(devices=[]):
            judge = base.ConditionalCritic(data["scale"])
        judge.load_state_dict(saved_states[arm, int(step)]["training"]["models"]["critic"], strict=True)
        judge.eval().requires_grad_(False)
        judges[name] = judge
        review.require(base.digest(judge.state_dict()) == receipt["judges"][name], name + "/actual baseline trained critic weights")
    review.require(set(receipt["judges"]) == set(JUDGES), "exactly all four mandatory judges")
    judge_binding = {name: base.digest(judge.state_dict()) for name, judge in judges.items()}
    reference_path = {}
    for name, judge in judges.items():
        reference_path[name] = {}
        for pool in base.SPLITS:
            reference_path[name][pool] = {}
            for fraction in (0, .25, .5, 1):
                residual = fraction * (data[pool]["base"] - data[pool]["targets"]) / data["scale"]
                score = score_residual(judge, data[pool]["context"], residual, panels[pool])
                compare_score(review, receipt["judge_reference_path"][name][pool][str(fraction)], score,
                              data, pool, name + "/residual fraction " + str(fraction), compact=True)
                reference_path[name][pool][str(fraction)] = {
                    "paired_game": score["paired_game"], "six_subject_game": by_subject(score, data, pool)}
    curve_path = directory / "common-judge-curves.jsonl"
    artifacts[curve_path.name] = sha(curve_path)
    curve_rows = {}
    with curve_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            identity = (row["arm"], row["step"])
            review.require(identity not in curve_rows, "no duplicate fixed curve identity")
            curve_rows[identity] = row["scores"]
    review.require(set(curve_rows) == {(arm, step) for arm in law.ARMS for step in CURVE_STEPS},
                   "all 102 fixed clean FAST curves")
    endpoint_scores, endpoint_subjects, particle_witnesses = {}, {}, {}
    for arm in law.ARMS:
        # An unresolved step-zero backend must enter a fresh policy, never a
        # policy already resolved by a later checkpoint.
        with torch.random.fork_rng(devices=[]):
            loop = law.make_rotated_loop(arm, data, bindings=bindings)
        for step in CURVE_STEPS:
            budget()
            state = load(directory / arm / f"step-{step:04d}.pt")
            base.restore(loop, state)
            before, rng = base.digest(base.checkpoint(loop)), torch.get_rng_state().clone()
            all_predictions = predictions(loop, data)
            scores = {name: {pool: score_prediction(judge, data, pool, all_predictions[pool], panels)
                             for pool in base.SPLITS} for name, judge in judges.items()}
            actual = curve_rows[arm, step]
            review.require(set(actual) == set(JUDGES), arm + "/all four curve heads")
            for name, pools in scores.items():
                review.require(set(actual[name]) == set(base.SPLITS), arm + "/all curve pools")
                for pool, score in pools.items():
                    compare_score(review, actual[name][pool], score, data, pool,
                                  arm + "@" + str(step) + "/" + name + "/" + pool)
            review.require(base.digest(base.checkpoint(loop)) == before and torch.equal(rng, torch.get_rng_state()),
                           arm + "/clean scoring leaves all owners/RNG/diagnostics unchanged")
            if step == 0:
                for name, pools in scores.items():
                    for pool, score in pools.items():
                        compare_score(review, receipt["initial_scores"][arm][name][pool], score, data, pool,
                                      arm + "/initial compact score", compact=True)
            if step in (5120, 6400):
                endpoint_scores[f"{arm}@{step}"] = compact_scores(scores)
                endpoint_subjects[f"{arm}@{step}"] = {
                    name: {pool: by_subject(score, data, pool) for pool, score in pools.items()}
                    for name, pools in scores.items()}
                for name, pools in scores.items():
                    for pool, score in pools.items():
                        compare_score(review, receipt["endpoint_scores"][f"{arm}@{step}"][name][pool], score,
                                      data, pool, arm + "/mandatory endpoint " + str(step), compact=True)
        if arm != law.ARMS[0]:
            before = base.digest(base.checkpoint(loop))
            actual_predictions = predictions(loop, data, zero_code=True)
            ablation = {name: {pool: score_prediction(judge, data, pool, actual_predictions[pool], panels)
                              for pool in base.SPLITS} for name, judge in judges.items()}
            for name, pools in ablation.items():
                for pool, score in pools.items():
                    compare_score(review, receipt["code_ablation"][arm][name][pool], score, data, pool,
                                  arm + "/zero code " + name, compact=True)
            review.require(base.digest(base.checkpoint(loop)) == before, arm + "/code ablation state unchanged")
            p = loop.policy
            witness = {
                "C_norms": {site: float(getattr(loop.G, site).bridge.weight[:, 2:].detach().norm()) for site in law.SITES},
                "bridge_still_trainable": all(getattr(loop.G, site).bridge.weight.requires_grad
                                              and getattr(loop.G, site).bridge.bias.requires_grad for site in law.SITES),
                "bank_still_trainable": p.table.requires_grad,
                "router_still_trainable": all(parameter.requires_grad for parameter in p.router.parameters()),
                "final_bank_gradient_rows": trace_summaries[arm]["final_bank_gradient_rows"],
                "final_query_gradient_norm": trace_summaries[arm]["final_query_gradient_norm"],
                "bank_changed": base.digest(p.table) != base.digest(initial[arm]["training"]["table"]),
                "router_changed": base.digest(p.router.state_dict()) != base.digest(initial[arm]["training"]["models"]["router"]),
                **{key: trace_summaries[arm][key] for key in receipt["arms"][arm]["coverage"]},
                "zero_code_minus_live_test_game": {
                    name: ablation[name]["test"]["paired_game"]
                          - endpoint_scores[f"{arm}@6400"][name]["test"]["paired_game"] for name in JUDGES}}
            compare_nested(review, witness, receipt["arms"][arm]["particle_witness"], arm + "/actual signed live particle witness")
            particle_witnesses[arm] = witness
    review.require(base.digest(panels) == panel_binding
                   and judge_binding == {name: base.digest(judge.state_dict()) for name, judge in judges.items()},
                   "all private panels and judges immutable")
    gates = independent_gates(endpoint_scores, particle_witnesses)
    # Reductions from independent floating-point reduction ordering can differ by
    # a few ulps; compare observations numerically and every decision exactly.
    compare_nested(review, gates, receipt["gates"], "independent all-four gates and denominator-safe reductions")
    for relative, expected in artifacts.items():
        review.require(sha(directory / relative) == expected, "review did not alter evidence/" + relative)
    review.same(bindings["source_hashes"], runner.source_hashes(), "held source/card immutable through review")
    review.require(native_hash() == bindings["native_source_hash"], "native source immutable through review")
    budget()
    wall = time.monotonic() - started
    return {
        "schema": "routed_convergence_rotated_independent_review_v1", "status": "qualified", "qualified": True,
        "qualification_scope": "this standalone policy-aware CPU task only", "qualification_credit": "none",
        "checks": review.checks, "reviewer_sha256": sha(REVIEWER), "execution_receipt_sha256": artifacts["receipt.json"],
        "execution_wall_seconds": receipt["wall_seconds"], "independent_review_wall_seconds": wall,
        "total_execution_and_review_seconds": receipt["wall_seconds"] + wall,
        "quality_updates": dict.fromkeys(law.ARMS, 6400), "independent_replay_software_updates": 6,
        "recovery": replay_records, "state_counts": dict.fromkeys(law.ARMS, 35),
        "curve_counts": dict.fromkeys(law.ARMS, 34), "native_source_digest": bindings["native_source_hash"],
        "source_hashes": bindings["source_hashes"], "artifact_manifest_digest": base.digest(artifacts),
        "data_digest": data["digest"], "judges": judge_binding, "private_panel_digest": panel_binding,
        "endpoint_scores": endpoint_scores, "endpoint_six_subject_scores": endpoint_subjects,
        "judge_reference_path": reference_path, "particle_witnesses": particle_witnesses, "gates": gates,
        "native_diagnostic_nonfinite_sentinels": {name: paths for name, paths in sentinels.items() if paths},
        "finite_scope": "all FAST/EMA model owners, bank/noise, gradients and optimizer moments; native LR diagnostic sentinels separately preserved",
        "limits": card["limits"]}


def compare_nested(review, actual, expected, label):
    if isinstance(expected, dict):
        review.require(isinstance(actual, dict) and set(actual) == set(expected), label + "/keys")
        for key, value in expected.items():
            compare_nested(review, actual[key], value, label + "/" + str(key))
    elif isinstance(expected, float):
        review.close(actual, expected, label)
    else:
        review.require(actual == expected, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.source_only == (args.run is not None):
        parser.error("choose exactly one of --source-only and --run")
    if args.out is not None and args.out.exists():
        parser.error("preserve existing review receipts; choose a fresh output path")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            report = source_readiness() if args.source_only else review_run(args.run)
    except Exception as error:
        report = {"status": "invalid_or_incomplete", "qualified": False, "qualification_credit": "none",
                  "reviewer_sha256": sha(REVIEWER), "error": f"{type(error).__name__}: {error}"}
        if args.out:
            runner.common.write_json(args.out, report)
        raise
    finally:
        torch.set_num_threads(threads)
    if args.out:
        runner.common.write_json(args.out, report)
    print(json.dumps({key: report[key] for key in ("status", "qualified", "checks", "reviewer_sha256")}, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
