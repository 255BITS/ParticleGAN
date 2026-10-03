"""Software prefix proof that source-family observations preserve training.

Compare complete model/optimizer/gradient/RNG state at boundaries 0..3, with
and without the observer. Stop externally before update4; leave all declared
source budgets and recipes intact. These are software checks, not qualification
campaigns or attempts to obtain a passing model.
"""
import argparse
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from . import source_family_training as observer


class PrefixComplete(BaseException):
    pass


def complete_state(local):
    """Extend the observer's snapshot to owners held in containers as well."""
    extra = {}
    seen = set()

    def walk(name, value):
        if isinstance(value, torch.nn.Module):
            if id(value) in seen:
                return
            seen.add(id(value))
            extra[name] = {"state": value.state_dict(),
                           "modes": {k: m.training for k, m in value.named_modules()},
                           "parameters": {k: (p.requires_grad, p.grad) for k, p in value.named_parameters()}}
        elif isinstance(value, torch.optim.Optimizer):
            if id(value) not in seen:
                seen.add(id(value)); extra[name] = value.state_dict()
        elif isinstance(value, dict):
            for index, part in enumerate(value.values()):
                walk(f"{name}/dict_value_{index}", part)
        elif isinstance(value, (tuple, list)):
            for index, part in enumerate(value):
                walk(f"{name}/{index}", part)
    for name, value in local.items():
        walk(name, value)
    return observer.digest({"direct_owners_and_RNG": observer.owned_state(local), "all_nested_owners": extra})


def run_prefix(module, family, quality, output, function, args, *, observe):
    output.mkdir(parents=True, exist_ok=False)
    capture = observer.Capture(module, family, output, quality)
    states, final_locals = {}, {}
    target = function.__code__

    def trace(frame, event, arg):
        return local_trace if event == "call" and frame.f_code == target else None

    def local_trace(frame, event, arg):
        if event == "line" and frame.f_lineno in capture.functions[target][1]:
            local = frame.f_locals
            step = local.get("step", local.get("_", -1) + 1)
            if isinstance(step, int) and step <= 3 and step not in states:
                arm = (function.__name__.removeprefix("train_") if family == "sign" else
                       local["mode"] if family == "landing" else
                       "pretrain" if function.__name__ == "pretrain" else local["name"])
                if observe:
                    capture.observe(arm, step, local)
                states[step] = complete_state(local)
                if step == 3:
                    final_locals.update(local)
                    raise PrefixComplete()
        return local_trace

    sys.settrace(trace)
    try:
        function(*args)
    except PrefixComplete:
        pass
    else:
        raise AssertionError("expected an external stop before the fourth update")
    finally:
        sys.settrace(None)
    assert tuple(states) == (0, 1, 2, 3)
    return states, final_locals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve prior prefix artifacts and choose a new output directory")
    args.out.mkdir(parents=True)
    sys.path.insert(0, str(args.source.resolve()))
    spec = importlib.util.spec_from_file_location("prefix_definition_quality", args.quality)
    quality = importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)
    torch.set_num_threads(1)
    log = lambda message: None
    results = []
    started = time.monotonic()
    for family, path in observer.SOURCES.items():
        module = importlib.import_module(path[:-3].replace("/", "."))
        assert Path(module.__file__).resolve() == args.source.resolve() / path
        if family == "sign":
            cases = [(name, getattr(module, "train_" + name), ()) for name in ("collapsed", "supervised", "paired")]
        elif family == "landing":
            cases = [(name, module.train_arm, (name,)) for name in ("baseline", "combined")]
            # The source requires adv_weight=0 for this software control.
            cases.append(("supervised", module.train_arm, ("supervised", module.STEPS, 0)))
        else:
            cases = [("pretrain", module.pretrain, (log,))]
        native_initial = None
        for name, function, arguments in cases:
            with torch.random.fork_rng(devices=[]):
                if family == "native":
                    torch.manual_seed(0)
                rng = torch.get_rng_state().clone()
                base_states, local = run_prefix(module, family, quality, args.out / family / name / "baseline",
                                                function, arguments, observe=False)
                torch.set_rng_state(rng)
                seen_states, _ = run_prefix(module, family, quality, args.out / family / name / "observed",
                                            function, arguments, observe=True)
                assert base_states == seen_states, (family, name)
                if family == "native":
                    # Make a deterministic three-update software fixture from
                    # the prefix; it is not the qualified 250-update host.
                    control = module.Enc()
                    control.load_state_dict(local["encoder"].state_dict())
                    native_initial = (local["recipe"], module.clone_init(dict(ep=local["encoder"], ec=control,
                                                                           g=local["decoder"], prior=local["prior"])))
                results.append({"family": family, "arm": name, "boundaries": 4,
                                "state_optimizer_gradient_RNG_hashes_exact": True,
                                "baseline_hashes": base_states})
        if family == "native":
            recipe, initial = native_initial
            for name in ("current", "fixed"):
                with torch.random.fork_rng(devices=[]):
                    rng = torch.get_rng_state().clone()
                    arguments = (name, name, recipe, initial, log)
                    baseline, _ = run_prefix(module, family, quality, args.out / family / name / "baseline",
                                              module.run_arm, arguments, observe=False)
                    torch.set_rng_state(rng)
                    observed, _ = run_prefix(module, family, quality, args.out / family / name / "observed",
                                              module.run_arm, arguments, observe=True)
                    assert baseline == observed, (family, name)
                    results.append({"family": family, "arm": name, "boundaries": 4,
                                    "state_optimizer_gradient_RNG_hashes_exact": True,
                                    "baseline_hashes": baseline})
    receipt = {"format": "source_family_observer_prefix_parity_v1", "source_revision": observer.REVISION,
               "source_budget_changes": False, "recipe_changes": False,
               "external_stop": "before update4 at the original loop boundary",
               "native_finetune_fixture": "three-update pretraining prefix; not qualified 250-update host",
               "software_update_count": 54, "qualified_campaign_count": 0, "qualification_credit": "none",
               "comparisons": results, "matched_owner_boundaries": sum(r["boundaries"] for r in results),
               "observer_sha256": observer.sha(observer.__file__), "checker_sha256": observer.sha(__file__),
               "quality_evaluator_sha256": observer.sha(args.quality), "wall_seconds": time.monotonic() - started,
               "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "threads": 1, "device": "cpu"}}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"matched_owner_boundaries": receipt["matched_owner_boundaries"],
                      "cases": len(results), "wall_seconds": receipt["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
