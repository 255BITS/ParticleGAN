"""Version2 allocation gates with explicit diagnostic-counter reconciliation.

The context factory is supplied explicitly; this driver makes no critic-fit
assumption. Its recorder must expose the reviewed PR84 three-phase interface
and a mutable ``correction`` switch. The disabled branch is checked against
independently archived original PR84, including its full training-state hash.
No task runs when an earlier gate, source binding, or control is incomplete.
The two raw first200 snapshots are retained; only the independently counted
native evaluation differences are reconciled. Version1 stays frozen.
"""

import argparse
import ast
from contextlib import contextmanager
import gzip
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONFIG = "configs/toy100/constraints_simple_regularization.json"
BRANCHES = ((1324, 1335), (1380, 1395), (1530, 1545))
CAPTURE_SHA = "37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47"
RATES = dict(g=.00425, d=.00425, prior=.0085)
REFERENCES = {
    "warm": "reports/toy100/continuous-evidence/pr84-independent-audit/smooth40-warm/alternating_curvature.json.gz",
    "hold": "reports/toy100/continuous-evidence/pr84-stationary-hold/forks/original.json.gz",
}
FACTORY = "reports.toy100.crossfit_reallocation:crossfit_reallocation"
CLEAN_WARM_REFERENCE = "reports/toy100/continuous-evidence/coverage-projection-round4/warm/forks/original.json.gz"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def untimed(value):
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if "seconds" not in k}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def source_path(name, root=ROOT):
    path = (root / name).resolve()
    if Path(name).is_absolute() or path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError(f"source must be inside the repository: {name}")
    return path


def verify_sources(sources, *, root=ROOT, archive=None):
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("nonempty source binding required")
    for name, digest in sources.items():
        if sha(source_path(name, root).read_bytes()) != digest:
            raise RuntimeError(f"source changed: {name}")
        if archive is not None and sha(source_path(name, archive).read_bytes()) != digest:
            raise RuntimeError(f"archived source changed: {name}")


def load_factory(entrypoint):
    module_name, separator, name = entrypoint.partition(":")
    if not separator or not module_name.startswith("reports.toy100.") or not name.isidentifier():
        raise ValueError("factory must be reports.toy100.module:context_name")
    module = importlib.import_module(module_name)
    function = getattr(module, name)
    filename = str(Path(module.__file__).resolve().relative_to(ROOT))
    if not callable(function) or not isinstance(module.METHOD, str) or not module.METHOD:
        raise TypeError("context factory and nonempty METHOD required")
    return function, module.METHOD, filename


def validate_filter(gate, method):
    """Recompute all 44 live decisions; do not trust a summary PASS alone."""
    declaration = gate["declaration"]
    if (gate["status"] != "PASS" or gate["warm_eligible"] is not True
            or declaration["method"] != method or declaration["states_sha256"] != CAPTURE_SHA
            or declaration["noise_horizon"] != 1200 or declaration["nominal_rates"] != RATES
            or declaration["seed"] != 0):
        raise RuntimeError("saved-state gate or frozen configuration failed")
    rows = gate["branches"]
    if [(r["start"], r["end"]) for r in rows] != list(BRANCHES):
        raise RuntimeError("all three declared saved-state branches are required")
    for branch in rows:
        start, end = branch["start"], branch["end"]
        original, candidate = (branch["variants"][k] for k in ("original", "reallocation"))
        expected = list(range(start, end + 1))
        if (candidate["local_gate"]["pass_all"] is not True
                or candidate["local_gate"]["checks"] != len(expected)
                or candidate["local_gate"]["passing_checks"] != len(expected)
                or [p["step"] for p in candidate["points"]] != expected
                or any(p["grade"]["modes"] != 8 or not p["grade"]["hq"] >= .9
                       for p in candidate["points"])):
            raise RuntimeError(f"saved-state live quality failed in {start}:{end}")
        if (candidate["rng_final_sha256"] != original["rng_final_sha256"]
                or candidate["noise"] != original["noise"]
                or candidate["moment_steps"] != {"d": [end], "g": [end]}):
            raise RuntimeError("saved-state RNG, noise, or once-per-update moments changed")
        rates = [dict(step=step, role=role, rates=value) for step in expected
                 for role, value in (("d", [.00425]), ("g_prior", [.00425, .0085]))]
        if candidate["rates"] != rates or original["rates"] != rates:
            raise RuntimeError("saved-state role rates changed")
        dynamics = candidate["dynamics"]
        if (dynamics["method"] != method or dynamics["outer_steps"] != len(expected)
                or len(dynamics["records"]) != len(expected)
                or len(dynamics["corrections"]) != len(expected)
                or dynamics["correction_rng_checks"] != len(expected)
                or dynamics["correction_owner_checks"] != len(expected)):
            raise RuntimeError("saved-state controller or ownership checks are incomplete")


def require_filter(path, method, factory_source, *, root=ROOT):
    gate = json.loads(path.read_text())
    validate_filter(gate, method)
    sources = gate["declaration"]["sources"]
    if factory_source not in sources:
        raise RuntimeError("saved-state filter did not bind this context factory")
    verify_sources(sources, root=root, archive=path.parent / "source")
    return gate


def dense_warm_runner():
    """Keep prefix checks at 50; observe every update after the same fork.

    Exactly three AST edits extend the dense endpoint, retain cadence50 for
    the unchanged prefix, and account for one late check per update. Neither
    the host loop nor its original 24 observations is rewritten here.
    """
    from benchmarks.toy100 import warm_equilibrium_probe as warm
    tree = ast.parse(inspect.getsource(warm.run_warm_variants))
    cadence = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "cadence" for t in n.targets)]
    ends = [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "dense_until"]
    divisors = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)
                and isinstance(n.op, ast.FloorDiv) and isinstance(n.right, ast.Name)
                and n.right.id == "cadence"]
    if len(cadence) != 1 or len(ends) != 1 or len(divisors) != 1:
        raise RuntimeError("reviewed warm-observer shape changed")
    if (ast.unparse(cadence[0].value) != "50 if steps == FROZEN_STEPS else 10"
            or ast.unparse(ends[0].value) != "FROZEN_STEPS"
            or ast.unparse(divisors[0].left) != "steps - FROZEN_STEPS"):
        raise RuntimeError("reviewed diagnostic cadence semantics changed")
    cadence[0].value = ast.Constant(50)
    ends[0].value = ast.Name(id="steps", ctx=ast.Load())
    divisors[0].right = ast.Constant(1)
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    namespace = {}
    exec(compile(tree, "<allocation-dense-observer>", "exec"), warm.__dict__, namespace)
    return namespace["run_warm_variants"], source


def bound_sources(filter_sources):
    """Bind candidate helpers plus all shared executable host/wrapper code."""
    names = set(filter_sources) | set(REFERENCES.values()) | {
        CLEAN_WARM_REFERENCE,
        CONFIG, str(Path(__file__).resolve().relative_to(ROOT)),
        "reports/toy100/pr84_smoothed_parity.py",
        "reports/toy100/pr84_critic_refinement_capture.py",
        "reports/toy100/allocation_snapshot_reconciliation.py",
        "reports/toy100/allocation_prefix_hash_diagnosis.py",
        "reports/toy100/continuous-evidence/anchor-prefix-hash-diagnosis/reconciliation.json",
        "lib/toy_models.py", "lib/toy_metrics.py", "benchmarks/learned_lr_evaluation.py",
        "benchmarks/transfer_suite/plans/default_comparison.json",
        "benchmarks/transfer_suite/plans/recipe_history.json",
    }
    for pattern in ("particlegan/**/*.py", "benchmarks/locked_shared/**/*.py",
                    "benchmarks/smart_descent/*.py", "benchmarks/transfer_suite/*.py",
                    "benchmarks/toy100/*.py"):
        names.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    return {name: sha(source_path(name).read_bytes()) for name in sorted(names)}


def require_previous(previous, *, phase, method, factory, sources, filter_sha):
    expected_phase = "warm" if phase == "hold" else "hold"
    if (previous["method"] != method or previous["phase"] != expected_phase
            or previous["factory"] != factory or previous["source"] != sources
            or previous["saved_state_filter_sha256"] != filter_sha
            or previous["identity_cold_parity"] is not True
            or previous["original_control_exact_parity"] is not True
            or previous["status"] != "PASS"):
        raise RuntimeError("previous gate, factory, or source binding failed")
    candidate = previous["variants"]["candidate"]
    if (candidate["status"] != "PASS" or candidate["local_stability"]["pass_all"] is not True
            or candidate["local_stability"]["checks"] != 200):
        raise RuntimeError("complete warm200 pass required")
    if phase == "cold" and (previous["first200_training_parity"] is not True
            or previous["first200_snapshot_reconciliation"]["status"] != "EXACT_TRAINING_STATE_WITH_ENUMERATED_EVALUATION_COUNTERS"
            or candidate["long_hold"]["pass_all"] is not True
            or candidate["long_hold"]["checks"] != 1200):
        raise RuntimeError("complete dense1200 hold and exact first200 parity required")


def compare_original(disabled, expected, clean_records):
    from reports.toy100.pr84_smoothed_parity import _compare_records
    for key in ("warm_state_sha256", "final_state_sha256", "observations", "noise",
                "final", "ema", "optimizer_final", "post_checkpoint_rate_ranges"):
        if untimed(disabled[key]) != untimed(expected[key]):
            raise RuntimeError(f"disabled correction changed original PR84 {key}")
    by_step = {p["step"]: p for p in disabled["diagnostic"]}
    for point in expected["diagnostic"]:
        step = point["step"]
        # Old hold archive observed prefix every10; this observer preserves
        # warm's every50 prefix so full noise histories agree through1200.
        # Every archived post-fork check and shared prefix check must match.
        if step <= 1000 and step % 50:
            continue
        if untimed(by_step.get(step)) != untimed(point):
            raise RuntimeError("disabled correction changed original diagnostic")
    _compare_records(disabled["dynamics_receipt"]["records"], expected["dynamics_receipt"]["records"])
    if untimed(disabled["dynamics_receipt"]["records"]) != untimed(clean_records):
        raise RuntimeError("disabled correction changed complete clean PR84 update records")


def compare_first200(old, new, *, old_directory, new_directory):
    if old["warm_state_sha256"] != new["warm_state_sha256"]:
        raise RuntimeError("long hold changed initial warm state")
    old_rows = [p for p in old["diagnostic"] if p["step"] > 1000]
    new_rows = [p for p in new["diagnostic"] if 1000 < p["step"] <= 1200]
    if len(old_rows) != 200 or untimed(old_rows) != untimed(new_rows):
        raise RuntimeError("long hold changed first200 live checks")
    a, b = old["dynamics_receipt"], new["dynamics_receipt"]
    for key in ("records", "corrections"):
        if len(a[key]) != 200 or untimed(a[key]) != untimed(b[key][:200]):
            raise RuntimeError(f"long hold changed first200 {key}")
    from reports.toy100.allocation_snapshot_reconciliation import load_and_verify, reconcile
    return reconcile(load_and_verify(old_directory, a), load_and_verify(new_directory, b))


def internal_state_metrics(local):
    """Read-only finite-state audit; norms do not accept or reject updates."""
    import torch
    def check(value):
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite final model or Adam state")
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
    result = {}
    for role in ("generator", "critic", "prior"):
        module = local[role]
        check(module.state_dict())
        values = [p.detach().double() for p in module.parameters()]
        result[role] = dict(parameter_l2=float(sum(v.square().sum() for v in values).sqrt()),
                            parameter_abs_max=max(float(v.abs().max()) for v in values))
    for key in ("opt_g", "opt_d"):
        check(local[key].state_dict())
    result["model_and_adam_finite"] = True
    return result


def fit_geometry_blocks(corrections, block_size=200):
    """Expose possible internal stiffness without adding a quality gate."""
    rows = []
    for start in range(0, len(corrections), block_size):
        block = corrections[start:start + block_size]
        records = [r for row in block for r in row["fit"]["records"]]
        rows.append(dict(first_update=block[0]["step"], last_update=block[-1]["step"],
            jacobians=len(records), singular_min=min((r["singular_min"] for r in records), default=None),
            singular_max=max((r["singular_max"] for r in records), default=None),
            selections={name: sum(row["selected"] == name for row in block)
                        for name in ("rest", "native_gan", "joint_fit")}))
    return rows


def run_continuation(output, declaration, factory, previous_path):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context
    from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha
    import torch

    @contextmanager
    def activate(name, state, prefix):
        recorder, _ = prefix
        recorder.enabled = name in ("disabled", "candidate")
        recorder.correction = name == "candidate"
        completed, target = state["completed_steps"], state["target_steps"]
        receipt = dict(variant=name, shared_gate_eligible=False, internal_state_blocks=[])

        def accounting(calls, outer):
            state["declare_optimizer_accounting"](calls=calls + target - outer,
                                                   moment_updates=target)
            if outer == 200:
                value = snapshot(recorder._local)
                path = output / f"{name}-update1200-pre-ema.pt"
                torch.save(value, path)
                receipt.update(first200_raw_snapshot_sha256=_sha(value), first200_snapshot_file=path.name,
                               first200_snapshot_file_sha256=sha(path.read_bytes()))
                receipt["first200_state_stage"] = "after update1200 game/correction; before its EMA/checkpoint"
            if outer % 200 == 0:
                receipt["internal_state_blocks"].append(dict(update=completed + outer,
                    **internal_state_metrics(recorder._local)))
            if name == "candidate" and outer % 20 == 0:
                print(json.dumps(dict(event="ALLOCATION_PROGRESS", phase=declaration["phase"],
                    update=completed + outer, active_updates=outer)), flush=True)

        recorder.accounting = accounting
        print(json.dumps(dict(event="VARIANT_START", variant=name)), flush=True)
        if name == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
                receipt["final_internal_state"] = internal_state_metrics(recorder._local)
                receipt["fit_geometry_blocks"] = fit_geometry_blocks(receipt.get("corrections", []))
                state_file = output / f"{name}-final-state.pt"
                value = snapshot(recorder._local)
                torch.save(value, state_file)
                receipt.update(final_state_file=state_file.name,
                    final_snapshot_sha256=_sha(value), final_state_file_sha256=sha(state_file.read_bytes()))
        print(json.dumps(dict(event="VARIANT_DONE", variant=name)), flush=True)

    phase = declaration["phase"]
    names = ("identity", "constant", "disabled", "candidate") if phase == "warm" else (
        "identity", "disabled", "candidate")
    variants = {name: (lambda state, prefix, name=name: activate(name, state, prefix)) for name in names}
    runner, _ = dense_warm_runner()
    result = runner(json.loads((ROOT / CONFIG).read_text()), variants, output_dir=output / "forks",
                    steps=1200 if phase == "warm" else 2400,
                    prefix_context=lambda: factory(task="mode_hold", start_step=1000, correction=True))
    expected = json.loads(gzip.decompress((ROOT / REFERENCES[phase]).read_bytes()))
    disabled = json.loads((output / "forks/disabled.json").read_text())
    clean_name = CLEAN_WARM_REFERENCE if phase == "warm" else REFERENCES["hold"]
    clean_records = json.loads(gzip.decompress((ROOT / clean_name).read_bytes()))["dynamics_receipt"]["records"]
    compare_original(disabled, expected, clean_records)
    candidate = json.loads((output / "forks/candidate.json").read_text())
    reconciliation = None
    if phase == "hold":
        old = json.loads((previous_path.parent / "forks/candidate.json").read_text())
        reconciliation = compare_first200(old, candidate, old_directory=previous_path.parent, new_directory=output)
    result.update(method=declaration["method"], phase=phase, factory=declaration["factory"],
        source=declaration["source"], shared_gate_eligible=False,
        saved_state_filter_sha256=declaration["saved_state_filter_sha256"],
        original_control_exact_parity=True, original_reference=REFERENCES[phase],
        original_reference_sha256=sha((ROOT / REFERENCES[phase]).read_bytes()),
        original_diagnostic_scope="all archived post1000 checks and every50 prefix checkpoints; full host observations/state/records",
        complete_record_reference=clean_name,
        complete_record_reference_sha256=sha((ROOT / clean_name).read_bytes()),
        first200_training_parity=True if phase == "hold" else None,
        first200_snapshot_reconciliation=reconciliation,
        status=result["variants"]["candidate"]["status"])
    write_json(output / "summary.json", result)
    print(json.dumps(dict(event="CONTINUATION_DONE", **result)), flush=True)


def run_cold(output, declaration, factory):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
    from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha

    config = json.loads((ROOT / CONFIG).read_text())
    config.update(name=declaration["method"], lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap")
    config.pop("network_lr_floor")
    recipe, noise, _ = declared_recipe(config)
    write_json(output / "config.json", config)
    stages = []
    for task, steps in (("trajectory", 400), ("mode_hold", 1200)):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        ordinary = torch.optim.Adam.step
        calls = {}

        def audited_step(optimizer, closure=None):
            calls.setdefault(optimizer, []).append(tuple(group["lr"] for group in optimizer.param_groups))
            return ordinary(optimizer, closure=closure)

        with patch.object(torch.optim.Adam, "step", audited_step), factory(task=task) as (recorder, source):
            (output / f"generated-{task}.py").write_text(source)
            def progress(calls, outer):
                if outer % 20 == 0:
                    print(json.dumps(dict(event="COLD_PROGRESS", task=task, update=outer)), flush=True)
            recorder.accounting = progress
            try:
                result, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
            except BaseException as error:
                # A partial phase is diagnostic only; never mark it restartable.
                error_row = dict(status="ERROR", task=task, error=repr(error), shared_gate_eligible=False)
                if recorder._local is not None:
                    error_file = output / f"{task}-error-state.pt"
                    torch.save(snapshot(recorder._local), error_file)
                    error_row.update(state_file=error_file.name, state_file_sha256=sha(error_file.read_bytes()),
                                     state_stage="exception inside host; partial phase, not a resume boundary")
                write_json(output / f"{task}.error.json", error_row)
                raise
        accounting = {}
        for role, optimizer, rates in zip(("d", "g"), recorder.optimizers, ((.00425,), (.00425, .0085))):
            observed = calls[optimizer]
            moments = [int(optimizer.state[p]["step"]) for group in optimizer.param_groups for p in group["params"]]
            if len(observed) != steps or any(row != rates for row in observed) or set(moments) != {steps}:
                raise RuntimeError(f"{task}: actual {role} rates/calls/moments changed")
            accounting[role] = dict(applied_rates=rates, actual_adam_calls=len(observed),
                                    moment_steps_min=min(moments), moment_steps_max=max(moments))
        if recorder.outer_steps != steps or len(recorder.records) != steps:
            raise RuntimeError("cold host omitted active update records")
        if (recorder.rng_replay_verified != 2 * steps
                or any(recorder.rows[opt]["calls"] != 3 * steps for opt in recorder.optimizers)
                or context["noise_receipt"]["step_calls"] != steps
                or context["noise_receipt"]["total_steps"] != steps):
            raise RuntimeError("cold replay, field counts, or original noise horizon changed")
        verdict = test_verdict(spec, result)
        value = snapshot(recorder._local)
        state_file = output / f"{task}-final-state.pt"
        torch.save(value, state_file)
        data = dict(method=declaration["method"], result=result, applied=context["applied"],
            noise=context["noise_receipt"], dynamics=recorder.receipt(), actual_adam_accounting=accounting,
            spec=spec, verdict=verdict, source=declaration["source"], shared_gate_eligible=False,
            final_state_file=state_file.name, final_state_file_sha256=sha(state_file.read_bytes()),
            final_snapshot_sha256=_sha(value),
            final_internal_state=internal_state_metrics(recorder._local),
            fit_geometry_blocks=fit_geometry_blocks(recorder.receipt().get("corrections", [])),
            saved_state_stage="after final host evaluation, live weights restored; before next set_step")
        write_json(output / f"{task}.json", data)
        row = dict(task=task, verdict=verdict, live=result["live"], seconds=result["seconds"])
        stages.append(row)
        print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
        if verdict["passed"] is not True:
            break
    result = dict(method=declaration["method"], phase="cold", factory=declaration["factory"],
        source=declaration["source"], stages=stages, shared_gate_eligible=False,
        status="PASS" if len(stages) == 2 and all(r["verdict"]["passed"] is True for r in stages) else "FAIL",
        own_acquired_continuation_tested=False, distribution_shift_tested=False)
    write_json(output / "summary.json", result)
    print(json.dumps(dict(event="COLD_DONE", **result)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "hold", "cold"), required=True)
    parser.add_argument("--factory", default=FACTORY)
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("forked gates require CPU-only state")
    factory, method, filename = load_factory(args.factory)
    gate = require_filter(args.filter, method, filename)
    sources = bound_sources(gate["declaration"]["sources"])
    filter_sha = sha(args.filter.read_bytes())
    if args.phase != "warm":
        if args.previous is None:
            raise ValueError("hold requires warm; cold requires dense hold")
        previous = json.loads(args.previous.read_text())
        require_previous(previous, phase=args.phase, method=method, factory=args.factory,
                         sources=sources, filter_sha=filter_sha)
        verify_sources(previous["source"], archive=args.previous.parent / "source")
    _, generated = dense_warm_runner()
    from benchmarks.toy100.continuous_probe import _provenance
    declaration = dict(method=method, factory=args.factory, phase=args.phase, source=sources,
        runtime=_provenance()["runtime"], shared_gate_eligible=False, seed=0, nominal_rates=RATES,
        saved_state_filter_sha256=filter_sha, scheduled_prefix_updates=1000,
        noise_horizon=dict(warm=1200, hold=1200, cold_trajectory=400, cold_mode_hold=1200),
        live_checks=dict(warm="1001..1200, every update", hold="1201..2400, every update"),
        original_observations="unchanged host checkpoints and original thresholds",
        control="scheduled identity vs fresh cold; correction=False vs independent archived PR84 full state",
        observer="prefix every50; every update after1000; no target shift",
        generated_observer_sha256=sha(generated.encode()),
        first200_state_comparison="retain both raw sidecars; exact all-field hash after only native evaluation-counter reconciliation",
        cold_order=["trajectory400", "mode_hold1200"], stop_at_first_failed_host=True,
        scope="scratch sequential gates; cold pass still needs own-acquired-state continuous hold")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path(name).read_bytes())
    (args.output / "source/generated_warm_observer.py").write_text(generated)
    (args.output / "saved-state-filter.json").write_bytes(args.filter.read_bytes())
    if args.previous is not None:
        (args.output / "previous-gate.json").write_bytes(args.previous.read_bytes())
    write_json(args.output / "declaration.json", declaration)
    print(json.dumps(dict(event="DECLARED", **declaration)), flush=True)
    if args.phase == "cold":
        run_cold(args.output, declaration, factory)
    else:
        run_continuation(args.output, declaration, factory, args.previous)
    verify_sources(sources)


if __name__ == "__main__":
    main()
