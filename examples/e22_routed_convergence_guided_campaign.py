"""Explicit immutable factory adapter for the held routed campaign/reviewer.

The training, replay and independent scoring code objects come from the held
rotated task. New local globals bind the guided host factory and task law.
Schema translation is an I/O envelope adapter; the actual artifacts always
carry the new task identity. Held modules are never monkey-patched.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
from types import CodeType, FunctionType, SimpleNamespace

import torch

if __package__:
    from . import e22_routed_convergence_guided_pair as factory
    from . import run_e22_routed_convergence_rotated_teacher as held_runner
    from . import review_e22_routed_convergence_rotated_teacher as held_reviewer
else:
    import e22_routed_convergence_guided_pair as factory
    import run_e22_routed_convergence_rotated_teacher as held_runner
    import review_e22_routed_convergence_rotated_teacher as held_reviewer

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_convergence_guided_pair_v1.json"
RUNNER = ROOT / "examples/run_e22_routed_convergence_guided_pair.py"
REVIEWER = ROOT / "examples/review_e22_routed_convergence_guided_pair.py"
SOURCES = tuple(dict.fromkeys((
    CARD, Path(__file__).resolve(), RUNNER, REVIEWER,
    ROOT / "examples/e22_routed_convergence_guided_pair.py",
    ROOT / "tests/test_e22_routed_convergence_guided_pair.py",
    ROOT / "tests/test_e22_routed_convergence_guided_campaign.py",
    Path(held_reviewer.__file__).resolve(),
    *held_runner.SOURCES,
)))
EXECUTION_SCHEMA = "routed_convergence_guided_pair_execution_v1"
REVIEW_SCHEMA = "routed_convergence_guided_pair_independent_review_v1"
HELD_EXECUTION_SCHEMA = "routed_convergence_rotated_execution_v1"
HELD_REVIEW_SCHEMA = "routed_convergence_rotated_independent_review_v1"
COMPLETION_SCHEMA = "routed_convergence_guided_pair_execution_completion_v1"
CFG_BUFFERS = factory.CFG_BUFFERS


def code_sha(function):
    """Hash every public code field, without marshal's refcount-dependent flags."""
    def canonical(value):
        if isinstance(value, CodeType):
            return {name: canonical(getattr(value, name)) for name in (
                "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize",
                "co_flags", "co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_name",
                "co_qualname", "co_firstlineno", "co_linetable", "co_exceptiontable", "co_freevars", "co_cellvars")}
        if isinstance(value, tuple):
            return {"tuple": [canonical(item) for item in value]}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, frozenset):
            return {"frozenset": sorted((canonical(item) for item in value), key=repr)}
        if value is Ellipsis:
            return {"ellipsis": True}
        if isinstance(value, complex):
            return {"complex": [value.real, value.imag]}
        return value
    payload = json.dumps(canonical(function.__code__), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def module_code_manifest(module):
    result = {}
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            result[name] = {"callable_code_sha256": code_sha(value),
                            "unwrapped_code_sha256": code_sha(inspect.unwrap(value))}
    return result


def orchestration_manifest():
    return {
        "id": "isolated_guided_campaign_namespace_v1", "task": factory.TASK,
        "code_object_hash_law": "SHA256(canonical JSON of all public code fields recursively); includes checkout filenames and interpreter bytecode, excludes marshal refcount/intern flags",
        "python_cache_tag": sys.implementation.cache_tag,
        "held_runner": module_code_manifest(held_runner),
        "held_reviewer": module_code_manifest(held_reviewer),
        "held_native_helpers": {name: {"callable_code_sha256": code_sha(getattr(held_runner.base, name)),
                                       "unwrapped_code_sha256": code_sha(inspect.unwrap(getattr(held_runner.base, name)))}
                                for name in ("make_loop", "update", "forward", "checkpoint", "restore", "evaluate", "score_residual")},
        "held_factory": factory.factory_binding_manifest(),
        "guided_factory_callbacks": module_code_manifest(factory),
        "guided_host_methods": {name: code_sha(getattr(factory.GuidedHost, name))
                                for name in ("__init__", "half_inputs", "combine", "forward", "forward_routed")},
        "global_bindings": {
            "law": "new guided factory; local held-name aliases only",
            "base": "held native helpers; reachability_witness bound to guided factory",
            "common": "held I/O/finiteness; extra frozen CFG buffers; new schema writer",
            "reviewer": "new actual source path, new source/card validator, frozen CFG buffer owners",
        },
        "schema_translation": {HELD_EXECUTION_SCHEMA: EXECUTION_SCHEMA,
                               HELD_REVIEW_SCHEMA: REVIEW_SCHEMA},
        "held_module_globals_mutated": False,
        "loop_or_scoring_source_copied": False,
    }


def _cell(value):
    return (lambda: value).__closure__[0]


def clone_namespace(module, overrides):
    """Clone module functions and decorator closures into one private namespace.

    torch.no_grad wrappers retain their torch globals but get the cloned inner
    function in their closure. This prevents a decorated reviewer helper from
    escaping the local bindings. Every code object remains the held object.
    """
    namespace, clones = dict(vars(module)), {}

    def clone(function):
        if id(function) in clones:
            return clones[id(function)]
        closure = function.__closure__
        if closure:
            closure = tuple(_cell(clone(cell.cell_contents))
                            if isinstance(cell.cell_contents, FunctionType)
                            and cell.cell_contents.__module__ == module.__name__ else cell
                            for cell in closure)
        globals_dict = namespace if function.__globals__ is vars(module) else function.__globals__
        result = FunctionType(function.__code__, globals_dict, function.__name__,
                              function.__defaults__, closure)
        result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
        result.__dict__.update(function.__dict__)
        result.__module__, result.__doc__ = function.__module__, function.__doc__
        if hasattr(function, "__wrapped__"):
            result.__wrapped__ = clone(function.__wrapped__)
        clones[id(function)] = result
        return result

    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            namespace[name] = clone(value)
    namespace.update(overrides)
    return SimpleNamespace(**namespace)


def source_hashes():
    return {str(path.relative_to(ROOT)): held_runner.common.sha(path) for path in SOURCES}


def validate_contract(card):
    """Pin both the new scientific law and every concrete namespace binding."""
    execution, host = card["execution"], card["host"]
    base = held_runner.base
    rates = {"generator": base.BRANCH_LR, "router": base.BRANCH_LR,
             "table": .0085, "critic": .00425, "learned_noise": .00425}
    if (card["task_id"] != factory.TASK
            or [arm["id"] for arm in card["arms"]] != list(factory.ARMS)
            or execution["steps_per_arm"] != 6400 or execution["checkpoint_cadence"] != 200
            or execution["device"] != "cpu" or execution["threads"] != 1
            or execution["arm_wall_budget_seconds"] != 900 or execution["total_wall_budget_seconds"] != 2700
            or execution["software_recovery_witness"] != [800, 802]
            or host["width"] != base.WIDTH or host["tokens_per_context"] != base.TOKENS
            or host["adapter_rank"] != base.RANK or host["sites"] != len(factory.SITES)
            or card["prior"]["num_particles"] != base.PARTICLES or card["prior"]["z_dim"] != base.Z_DIM
            or card["output_noise_std"] != base.PANEL_SIGMA or card["native_nominal_rates"] != rates
            or tuple(card["evaluation"]["mandatory_common_judges"]) != held_runner.JUDGES
            or tuple(card["evaluation"]["endpoints"]) != held_runner.ENDPOINTS
            or card["teacher_geometry"]["rho"] != factory.RHO
            or card["teacher_geometry"]["audit_report_sha256"] != factory.AUDIT_SHA256
            or card["guided_pair"]["guidance"] != factory.GUIDANCE
            or card["guided_pair"]["unconditional_source"] != "fixed mean of the unchanged six parent source vectors"
            or card["native_base_revision"] != "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"):
        raise ValueError("card differs from the single fixed guided-pair law")
    if card["orchestration_adapter"] != orchestration_manifest():
        raise ValueError("actual held code objects or namespace bindings differ from the card")
    if held_runner.common.native_source_hash() != card["native_python_source_digest"]:
        raise ValueError("native Python differs from the fixed parent")
    for relative, key in (
        ("docs/e22_routed_convergence_v1.json", "parent_card_sha256"),
        ("docs/e22_routed_convergence_neutral_v1.json", "parent_neutral_card_sha256"),
        ("docs/e22_routed_convergence_rotated_teacher_v1.json", "parent_rotated_card_sha256"),
    ):
        if held_runner.common.sha(ROOT / relative) != card[key]:
            raise ValueError("pinned parent card changed: " + relative)
    required = {str(path.relative_to(ROOT)) for path in SOURCES if path != CARD}
    if set(card["sources"]) != required:
        raise ValueError("new and held source manifest is incomplete or has extra identities")
    for relative, expected in card["sources"].items():
        if held_runner.common.sha(ROOT / relative) != expected:
            raise ValueError("declared source changed: " + relative)


def frozen_values(loop):
    values = held_runner.common.frozen_values(loop)
    for role, model in held_runner.base.modules(loop).items():
        for name, tensor in model.state_dict().items():
            if name in CFG_BUFFERS:
                values[role + "." + name] = tensor
    return values


def frozen_owners(state):
    values = held_reviewer.frozen_owners(state)
    for family in ("models", "averages"):
        for role, tensors in state["training"][family].items():
            for name, tensor in tensors.items():
                if name in CFG_BUFFERS:
                    values[family + "/" + role + "/" + name] = tensor
    return values


def output_envelope(value):
    """Translate only orchestration schemas; preserve all observation values."""
    result = deepcopy(value)
    if isinstance(result, dict):
        schemas = {HELD_EXECUTION_SCHEMA: EXECUTION_SCHEMA, HELD_REVIEW_SCHEMA: REVIEW_SCHEMA}
        if result.get("schema") in schemas:
            result["schema"] = schemas[result["schema"]]
        if result.get("schema") in (EXECUTION_SCHEMA, REVIEW_SCHEMA):
            if result.get("task", factory.TASK) != factory.TASK:
                raise ValueError("old task identity cannot pass the new envelope")
            result["task"] = factory.TASK
            result["orchestration_adapter"] = orchestration_manifest()
    return result


def write_json(path, value):
    return held_runner.common.write_json(path, output_envelope(value))


def reviewer_read(path):
    value = held_reviewer.read(path)
    if Path(path).name == "receipt.json":
        if value.get("schema") != EXECUTION_SCHEMA or value.get("task") != factory.TASK:
            raise ValueError("review requires the actual new guided-pair task envelope")
        if value.get("orchestration_adapter") != orchestration_manifest():
            raise ValueError("execution namespace differs from the fixed reviewer namespace")
        completion = held_reviewer.read(Path(path).with_name("execution-completion.json"))
        wall = completion.get("wall_seconds")
        if (completion.get("schema") != COMPLETION_SCHEMA or completion.get("task") != factory.TASK
                or completion.get("complete") is not True
                or completion.get("receipt_sha256") != held_runner.common.sha(path)
                or completion.get("compact_sha256") != held_runner.common.sha(Path(path).with_name("compact-report.json"))
                or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall)
                or not value["wall_seconds"] <= wall <= value["contract"]["execution"]["total_wall_budget_seconds"]
                or wall <= 0):
            raise ValueError("execution lacks the matching complete bounded-write receipt")
        # The held reviewer expects its generic envelope label. Normalize only
        # this private loaded copy, after enforcing the new on-disk identity.
        value = deepcopy(value)
        value["schema"] = HELD_EXECUTION_SCHEMA
        value["wall_seconds"] = completion["wall_seconds"]
    return value


def adapters():
    """Build isolated campaign/reviewer closures with identical guided factory."""
    base = SimpleNamespace(**vars(held_runner.base))
    base.reachability_witness = factory.reachability_witness
    law = SimpleNamespace(**vars(factory))
    frozen_manifest = orchestration_manifest()
    fixed_make_loop = factory.make_guided_loop

    def bound_loop(arm, data, *, bindings=None):
        if orchestration_manifest() != frozen_manifest:
            raise ValueError("held code objects/factory bindings changed after namespace creation")
        loop = fixed_make_loop(arm, data, bindings=bindings)
        loop.law["campaign_adapter"] = deepcopy(frozen_manifest)
        return loop

    law.make_rotated_data, law.make_rotated_loop = factory.make_guided_data, bound_loop
    law.neutral = SimpleNamespace(make_neutral_data=factory.neutral.make_neutral_data,
                                  initial_difference_witness=factory.initial_difference_witness)
    common = SimpleNamespace(**vars(held_runner.common))
    common.frozen_values, common.write_json = frozen_values, write_json
    runner = clone_namespace(held_runner, {
        "base": base, "law": law, "common": common, "ROOT": ROOT, "CARD": CARD,
        "SOURCES": SOURCES, "source_hashes": source_hashes, "validate_contract": validate_contract,
    })
    reviewer = clone_namespace(held_reviewer, {
        "base": base, "law": law, "runner": runner, "ROOT": ROOT, "REVIEWER": REVIEWER,
        "read": reviewer_read, "frozen_owners": frozen_owners,
    })
    return runner, reviewer


def require_budget(started, budget):
    elapsed = time.monotonic() - started
    if elapsed > budget:
        raise TimeoutError("guided campaign/review wall budget exhausted including serialization")
    return elapsed


def run_main():
    """Run the exact held campaign and bind its completed artifact write cost."""
    started = time.monotonic()
    runner, _ = adapters()
    runner.main()
    parser = held_runner.argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = held_reviewer.read(args.out / "receipt.json")
    budget = receipt["contract"]["execution"]["total_wall_budget_seconds"]
    completion = {"schema": COMPLETION_SCHEMA, "task": factory.TASK, "complete": False,
                  "receipt_sha256": held_runner.common.sha(args.out / "receipt.json"),
                  "compact_sha256": held_runner.common.sha(args.out / "compact-report.json")}
    try:
        completion.update(complete=True, wall_seconds=require_budget(started, budget))
        held_runner.common.write_json(args.out / "execution-completion.json", completion)
        require_budget(started, budget)
    except TimeoutError as error:
        completion.update(complete=False, wall_seconds=time.monotonic() - started, error=str(error))
        held_runner.common.write_json(args.out / "execution-completion.json", completion)
        raise


def review_main():
    """Independent held review, with new identity and a final serialization gate."""
    parser = held_reviewer.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.source_only == (args.run is not None):
        parser.error("choose exactly one of --source-only and --run")
    if args.out is not None and args.out.exists():
        parser.error("preserve existing review receipts; choose a fresh output path")
    if args.run is not None and args.out is None:
        parser.error("qualification requires --out for the bounded-write completion receipt")
    started, threads = time.monotonic(), torch.get_num_threads()
    torch.set_num_threads(1)
    _, reviewer = adapters()
    report, execution_seconds, execution_completion_sha = None, 0., None
    try:
        with torch.random.fork_rng(devices=[]):
            if args.source_only:
                report = reviewer.source_readiness()
            else:
                receipt = reviewer_read(args.run / "receipt.json")
                execution_seconds = receipt["wall_seconds"]
                execution_completion_sha = held_runner.common.sha(args.run / "execution-completion.json")
                report = reviewer.review_run(args.run)
        report = output_envelope(report)
        report.update(task=factory.TASK, orchestration_adapter=orchestration_manifest())
        if not args.source_only:
            if held_runner.common.sha(args.run / "execution-completion.json") != execution_completion_sha:
                raise AssertionError("execution completion changed during independent review")
            report["execution_completion_sha256"] = execution_completion_sha
        if args.out is not None:
            write_json(args.out, report)
        if not args.source_only:
            total = execution_seconds + require_budget(started, 2700 - execution_seconds)
            completion = {"schema": "routed_convergence_guided_pair_review_completion_v1",
                          "task": factory.TASK, "complete": True,
                          "review_sha256": held_runner.common.sha(args.out),
                          "total_execution_and_review_seconds": total}
            held_runner.common.write_json(args.out.with_suffix(args.out.suffix + ".completion.json"), completion)
            require_budget(started, 2700 - execution_seconds)
    except Exception as error:
        report = {"task": factory.TASK, "status": "invalid_or_incomplete", "qualified": False,
                  "qualification_credit": "none", "reviewer_sha256": held_runner.common.sha(REVIEWER),
                  "error": f"{type(error).__name__}: {error}"}
        if args.out is not None:
            write_json(args.out, report)
            held_runner.common.write_json(args.out.with_suffix(args.out.suffix + ".completion.json"),
                                          {"task": factory.TASK, "complete": False,
                                           "review_sha256": held_runner.common.sha(args.out), "error": report["error"]})
        raise
    finally:
        torch.set_num_threads(threads)
    print(json.dumps({key: report[key] for key in ("status", "qualified", "checks", "reviewer_sha256")}, allow_nan=False),
          flush=True)
