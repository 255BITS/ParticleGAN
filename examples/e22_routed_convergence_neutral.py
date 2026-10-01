"""One declared H/b-neutral acquisition intervention, retaining live particles.

The fixed baseline helper remains unchanged. Only initial hidden bridge H and
bias b are zeroed; the sampled particle gate C, bank, queries and native law
remain intact. Policy and EMA construction happen after that initialization.
"""

from copy import deepcopy
import torch

if __package__:
    from . import e22_routed_convergence as baseline
else:
    import e22_routed_convergence as baseline


TASK = "supra_conditional_two_site_hidden_neutral_v1"
ARM = "particle_native_game"
INTERVENTION = {"id": "hidden_bridge_and_bias_zero_v1", "changed": "initial H and b only",
                "preserved": "sampled C; common down/up; bank; router; critic; native controls"}


def make_neutral_data(original=None):
    original = baseline.make_data() if original is None else original
    if original["digest"] != baseline.digest({key: value for key, value in original.items() if key != "digest"}):
        raise ValueError("baseline data changed after its digest was bound")
    data = deepcopy(original)
    for site in ("first", "second"):
        data["initial_particle"][site + ".bridge.weight"][:, :baseline.RANK].zero_()
        data["initial_particle"][site + ".bridge.bias"].zero_()
    data["intervention"] = {**INTERVENTION, "baseline_data_digest": original["digest"]}
    data["digest"] = baseline.digest({key: value for key, value in data.items() if key != "digest"})
    return data


def make_neutral_loop(data, *, bindings=None):
    if data.get("intervention", {}).get("id") != INTERVENTION["id"]:
        raise ValueError("expected the declared H/b-neutral data law")
    for site in ("first", "second"):
        if (data["initial_particle"][site + ".bridge.weight"][:, :baseline.RANK].count_nonzero()
                or data["initial_particle"][site + ".bridge.bias"].count_nonzero()):
            raise ValueError("H and b must be neutral before policy construction")
    loop = baseline.make_loop(ARM, data, bindings=bindings)
    loop.law.update(task=TASK, intervention=deepcopy(data["intervention"]))
    return loop


def initial_difference_witness(original, data):
    """Check the actual tensors, including EMA, before any update."""
    neutral, reference = make_neutral_loop(data), baseline.make_loop(ARM, original)
    changed = []
    for role in ("generator", "average_generator"):
        left, right = baseline.modules(neutral)[role], baseline.modules(reference)[role]
        for key, tensor in left.state_dict().items():
            expected = right.state_dict()[key]
            if ".bridge.weight" in key:
                assert tensor[:, :baseline.RANK].count_nonzero() == 0
                assert torch.equal(tensor[:, baseline.RANK:], expected[:, baseline.RANK:])
                assert tensor[:, baseline.RANK:].count_nonzero() > 0
                changed.append(role + "." + key + "[H]")
            elif ".bridge.bias" in key:
                assert tensor.count_nonzero() == 0
                changed.append(role + "." + key)
            else:
                assert torch.equal(tensor, expected), (role, key)
    for role, model in baseline.modules(neutral).items():
        if role not in ("generator", "average_generator"):
            assert baseline.digest(model.state_dict()) == baseline.digest(baseline.modules(reference)[role].state_dict())
    assert torch.equal(neutral.policy.table, reference.policy.table)
    assert torch.equal(neutral.policy.averaged_table, reference.policy.averaged_table)
    assert torch.equal(neutral.policy.log_output_sigma, reference.policy.log_output_sigma)
    assert neutral.policy.recipe.to_dict() == reference.policy.recipe.to_dict()
    native_states = [deepcopy(loop.policy.state_dict()) for loop in (neutral, reference)]
    for state in native_states:
        state["models"].pop("generator")
        state["averages"].pop("generator")
    assert baseline.digest(native_states[0]) == baseline.digest(native_states[1])
    for key in original:
        if key not in ("digest", "initial_particle"):
            assert baseline.digest(original[key]) == baseline.digest(data[key]), key
    return {"changed_initial_coordinates": changed, "other_initial_owners_equal": True,
            "teacher_data_scales_equal": True, "native_recipe_equal": True,
            "all_non_generator_native_initial_state_equal": True, "sampled_C_nonzero": True,
            "original_data_digest": original["digest"], "neutral_data_digest": data["digest"]}


def main():
    import argparse
    import json
    import math
    from pathlib import Path
    import time
    import particlegan
    if __package__:
        from . import run_e22_routed_convergence as common
    else:
        import run_e22_routed_convergence as common

    root = Path(__file__).resolve().parents[1]
    card_path = root / "docs/e22_routed_convergence_neutral_v1.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=root / "runs/routed-convergence-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve its artifacts")
    if Path(particlegan.__file__).resolve().parent != root / "particlegan":
        parser.error("set PYTHONPATH to this checkout")
    torch.set_num_threads(1)
    card = json.loads(card_path.read_text())
    parent = json.loads((args.parent / "receipt.json").read_text())
    if card["task_id"] != TASK or card["execution"]["steps"] != 6400 or card["execution"]["checkpoint_cadence"] != 200:
        parser.error("runner requires the declared neutral v1 law")
    if parent["status"] != "complete" or parent["task"] != card["parent_task"]:
        parser.error("parent must be the completed fixed convergence diagnostic")
    if common.sha(common.CARD) != card["parent_card_sha256"]:
        parser.error("parent protocol changed")
    for name, value in parent["bindings"]["source_hashes"].items():
        if common.sha(root / name) != value:
            parser.error("frozen parent source changed")
    if common.native_source_hash() != parent["bindings"]["native_source_hash"]:
        parser.error("native source differs from completed parent")
    sources = (*common.SOURCES, card_path, Path(__file__).resolve())
    source_hashes = lambda: {str(path.relative_to(root)): common.sha(path) for path in sources}
    bindings = {**parent["bindings"], "source_hashes": source_hashes(),
                "parent_receipt_sha256": common.sha(args.parent / "receipt.json")}
    args.out.mkdir(parents=True)
    start = time.monotonic()
    receipt = {"task": TASK, "status": "running", "contract": card, "bindings": bindings}
    log = (args.out / "run.log").open("w", buffering=1)

    def emit(**row):
        text = json.dumps(common.json_value(row), allow_nan=False)
        log.write(text + "\n")
        print(text, flush=True)

    def budget():
        if source_hashes() != bindings["source_hashes"]:
            raise RuntimeError("frozen parent/neutral sources changed")
        if common.native_source_hash() != bindings["native_source_hash"]:
            raise RuntimeError("frozen native source changed")
        if time.monotonic() - start > card["execution"]["timeout_seconds"]:
            raise TimeoutError("declared total intervention budget exhausted")

    try:
        original = baseline.make_data()
        if original["digest"] != parent["data_digest"]:
            raise AssertionError("parent teacher/input identity changed")
        data = make_neutral_data(original)
        receipt["initialization_witness"] = initial_difference_witness(original, data)
        loop = make_neutral_loop(data, bindings=bindings)
        receipt["law"], receipt["data_digest"] = loop.law, data["digest"]
        receipt["active_parameter_counts"] = {role: sum(parameter.numel() for parameter in model.parameters()
                                               if parameter.requires_grad) for role, model in baseline.modules(loop).items()}
        receipt["active_parameter_counts"].update(table=loop.policy.table.numel(), noise=1)
        frozen_digest = baseline.digest(common.frozen_values(loop))
        initial_table_digest = baseline.digest(loop.policy.table)
        initial_router_digest = baseline.digest(loop.policy.router.state_dict())
        receipt["initial_frozen_owner_digest"] = frozen_digest
        receipt["checkpoints"] = []
        common.write_json(args.out / "provenance.json", receipt)
        common.save(args.out / "step-0000.pt", baseline.checkpoint(loop))
        recovery_rows, coverage = [], {"live_bank_updates": 0, "live_query_updates": 0, "moves": 0,
                                      "row_control_events": 0, "proposal_events": 0, "candidate_proposals": 0}
        with (args.parent / ARM / "trace.jsonl").open() as parent_trace, (args.out / "trace.jsonl").open("w", buffering=1) as trace:
            for step in range(1, 6401):
                if step % 25 == 1:
                    budget()
                expected = json.loads(next(parent_trace))
                row = baseline.update(loop)
                if row["batch_indices"] != expected["batch_indices"] or row["paired_base_digest"] != expected["paired_base_digest"]:
                    raise AssertionError("parent and intervention input/Gaussian streams differ")
                for key in ("loss_g", "loss_d_game", "penalty", "bank_gradient_norm", "query_gradient_norm"):
                    if not math.isfinite(row[key]):
                        raise FloatingPointError("nonfinite training observation: " + key)
                coverage["live_bank_updates"] += int(row["bank_gradient_rows"] > 0)
                coverage["live_query_updates"] += int(row["query_gradient_norm"] > 0)
                coverage["moves"] += int((row.get("move") or {}).get("moves", 0))
                coverage["row_control_events"] += int(row["move"] is not None)
                proposals = row["controls"]["routing"]["counters"]["proposals"]
                coverage["proposal_events"] += int(proposals > coverage["candidate_proposals"])
                coverage["candidate_proposals"] = proposals
                trace.write(json.dumps(common.json_value(row), allow_nan=False) + "\n")
                if step in (801, 802):
                    recovery_rows.append(deepcopy(row))
                if step % 200 == 0 or step == 802:
                    common.assert_finite(loop)
                    if baseline.digest(common.frozen_values(loop)) != frozen_digest:
                        raise AssertionError("frozen native owners changed")
                    path = args.out / f"step-{step:04d}.pt"
                    common.save(path, baseline.checkpoint(loop))
                    receipt["checkpoints"].append({"step": step, "file": path.name, "sha256": common.sha(path)})
                    emit(event="progress", step=step, seconds=time.monotonic() - start,
                         loss_g=row["loss_g"], rates={value["role"]: {"applied_lr": value["applied_lr"],
                             "raw_scale": value["tester"]["s"]} for value in row["controls"]["groups"].values()},
                         sigma=row["controls"]["actual_sigma"], fires=row["controls"]["surprise"]["fires"],
                         bank_gradient_rows=row["bank_gradient_rows"], moves=coverage["moves"])
        final_state_digest = baseline.digest(baseline.checkpoint(loop))
        recovery_start = time.monotonic()
        with torch.random.fork_rng(devices=[]):
            recovered = make_neutral_loop(data, bindings=bindings)
            baseline.restore(recovered, torch.load(args.out / "step-0800.pt", weights_only=False))
            replay = [baseline.update(recovered) for _ in range(2)]
            target = torch.load(args.out / "step-0802.pt", weights_only=False)
            if baseline.digest(replay) != baseline.digest(recovery_rows) or baseline.digest(baseline.checkpoint(recovered)) != baseline.digest(target):
                raise AssertionError("neutral own-state 800-to-802 recovery changed")
        if baseline.digest(baseline.checkpoint(loop)) != final_state_digest:
            raise AssertionError("private recovery changed authoritative step6400 state")
        receipt["recovery_witness"] = {"from": 800, "to": 802, "rows_exact": True, "state_exact": True,
                                       "final_state_unchanged": True, "software_updates": 2,
                                       "wall_seconds": time.monotonic() - recovery_start}
        receipt["particle_coverage"] = coverage
        if not coverage["live_bank_updates"] or not coverage["live_query_updates"]:
            raise AssertionError("particles or queries never acquired gradients")
        if baseline.digest({key: value for key, value in data.items() if key != "digest"}) != data["digest"]:
            raise AssertionError("training mutated the frozen teacher/data law")
        receipt["final_model_digest"] = baseline.digest(loop.G.state_dict())
        receipt["final_checkpoint_digest"] = final_state_digest
        receipt["final_frozen_owner_digest"] = baseline.digest(common.frozen_values(loop))
        receipt["final_particle_witness"] = {
            "C_norms": {site: float(getattr(loop.G, site).bridge.weight[:, baseline.RANK:].detach().norm())
                        for site in ("first", "second")},
            "bridge_still_trainable": all(getattr(loop.G, site).bridge.weight.requires_grad
                                          and getattr(loop.G, site).bridge.bias.requires_grad for site in ("first", "second")),
            "bank_changed": baseline.digest(loop.policy.table) != initial_table_digest,
            "router_changed": baseline.digest(loop.policy.router.state_dict()) != initial_router_digest}
        panels = baseline.evaluation_panels(data)
        if baseline.digest(panels) != parent["private_panel_digest"]:
            raise AssertionError("private paired evaluation panel changed")
        judges = {}
        for name in card["evaluation"]["common_judges"]:
            arm, step = name.split("@")
            state = torch.load(args.parent / arm / f"step-{int(step):04d}.pt", weights_only=False)
            with torch.random.fork_rng(devices=[]):
                judge = baseline.ConditionalCritic(data["scale"])
            judge.load_state_dict(state["training"]["models"]["critic"], strict=True)
            judge.eval().requires_grad_(False)
            if baseline.digest(judge.state_dict()) != parent["judges"][name]:
                raise AssertionError("mandatory parent judge changed")
            judges[name] = judge
        receipt["judges"] = {name: baseline.digest(judge.state_dict()) for name, judge in judges.items()}
        receipt["private_panel_digest"] = baseline.digest(panels)
        receipt["endpoint_scores"] = {}
        with (args.out / "common-judge-curves.jsonl").open("w", buffering=1) as curves:
            for step in range(0, 6401, 200):
                budget()
                baseline.restore(loop, torch.load(args.out / f"step-{step:04d}.pt", weights_only=False))
                scores = {name: {pool: baseline.evaluate(loop, judge, pool, panels) for pool in baseline.SPLITS}
                          for name, judge in judges.items()}
                curves.write(json.dumps({"step": step, "scores": scores}, allow_nan=False) + "\n")
                small = {name: {pool: common.compact(value) for pool, value in pools.items()} for name, pools in scores.items()}
                if step in (0, 1600, 6400):
                    receipt["endpoint_scores"][str(step)] = small
                emit(event="scored", step=step, test_game={name: pools["test"]["paired_game"] for name, pools in small.items()})
        receipt["code_ablation"] = {name: {pool: common.compact(baseline.evaluate(loop, judge, pool, panels, code_ablation=True))
                                           for pool in baseline.SPLITS} for name, judge in judges.items()}
        receipt["final_particle_witness"]["zero_code_minus_live_test_game"] = {
            name: receipt["code_ablation"][name]["test"]["paired_game"]
                  - receipt["endpoint_scores"]["6400"][name]["test"]["paired_game"] for name in judges}
        improvements, gap_reductions = {}, {}
        for name in judges:
            neutral_game = receipt["endpoint_scores"]["6400"][name]["test"]["paired_game"]
            particle_game = parent["endpoint_scores"][ARM + "@6400"][name]["test"]["paired_game"]
            improvements[name] = particle_game - neutral_game
            if name.endswith("@6400"):
                ordinary_game = parent["endpoint_scores"]["ordinary_native_game@6400"][name]["test"]["paired_game"]
                gap_reductions[name] = improvements[name] / (particle_game - ordinary_game)
        particle_witness = receipt["final_particle_witness"]
        retained_particle_gate = (particle_witness["bridge_still_trainable"]
            and all(value > 0 for value in particle_witness["C_norms"].values())
            and coverage["live_bank_updates"] > 0 and coverage["live_query_updates"] > 0
            and all(abs(value) > 1e-6 for value in particle_witness["zero_code_minus_live_test_game"].values()))
        budget()
        receipt.update(status="complete", wall_seconds=time.monotonic() - start,
                       paired_game_improvements=improvements, endpoint_gap_reduction=gap_reductions,
                       retained_particle_gate=retained_particle_gate,
                       mechanism_supported=all(value > 1e-4 for value in improvements.values())
                                           and all(value >= .5 for value in gap_reductions.values())
                                           and retained_particle_gate,
                       qualification_credit="none")
        common.write_json(args.out / "receipt.json", receipt)
        emit(event="complete", seconds=receipt["wall_seconds"], mechanism_supported=receipt["mechanism_supported"])
    except Exception as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "error",
                       error=type(error).__name__ + ": " + str(error), wall_seconds=time.monotonic() - start)
        common.write_json(args.out / "receipt.json", receipt)
        emit(event=receipt["status"], error=receipt["error"])
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
