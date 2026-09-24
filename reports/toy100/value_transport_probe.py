"""Fail-fast gates for PR84 plus one critic-value spare-particle pull."""

import argparse
from contextlib import contextmanager
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reports.toy100.value_transport_candidate import METHOD, value_transport_candidate


def _declaration(output, phase):
    names = (
        "reports/toy100/value_transport.py",
        "reports/toy100/value_transport_candidate.py",
        "reports/toy100/value_transport_probe.py",
        "reports/toy100/pr84_smoothed_candidate.py",
        "reports/toy100/coverage_pullback.py",
        "reports/toy100/alternating_curvature_scratch.py",
        "reports/toy100/extra_adam_scratch.py",
        "benchmarks/toy100/warm_equilibrium_probe.py",
        "benchmarks/toy100/continuous_probe.py",
        "benchmarks/locked_shared/mode_hold.py",
        "benchmarks/locked_shared/trajectory.py",
    )
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source"
    archive.mkdir()
    for name in names:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    row = dict(
        phase=phase, seed=0, shared_gate_eligible=False, scratch_optimizer_policy=METHOD,
        method=METHOD, d_curvature_bound=3.0, g_curvature_bound=0.25, smooth_cap=0.15,
        operator="Frozen PR84 GAN update then one critic-value spare-particle prior pull",
        objective_addition="softmax centroid of reals whose critic value exceeds mean fake by 2 real-score stds",
        trust_output=0.1, gap_spreads=2.0,
        conditional_policy="unchanged PR84 path",
        nominal_rates=dict(g=0.00425, d=0.00425, prior=0.0085),
        source=source)
    (output / "declaration.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def warm(output):
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants

    @contextmanager
    def variant(method, state, prefix):
        print(json.dumps(dict(event="VARIANT_START", variant=method,
                              completed_steps=state["completed_steps"])), flush=True)
        recorder, _ = prefix
        recorder.enabled = method in ("original", "value")
        recorder.correction = method == "value"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + (target - completed - outer),
            moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False,
                       scratch_optimizer_policy=METHOD if recorder.enabled else "control")
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=method)), flush=True)

    factories = {name: (lambda state, prefix, name=name: variant(name, state, prefix))
                 for name in ("identity", "constant", "original", "value")}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    result = run_warm_variants(config, factories, output_dir=output / "forks",
        prefix_context=lambda: value_transport_candidate(start_step=1000))
    reference = ROOT / "reports/toy100/continuous-evidence/pr84-independent-audit/smooth40-warm/alternating_curvature.json.gz"
    archive_parity = False
    if reference.exists():
        expected = json.loads(gzip.decompress(reference.read_bytes()))
        observed = json.loads((output / "forks/original.json").read_text())

        def untimed(value):
            if isinstance(value, dict):
                return {key: untimed(item) for key, item in value.items() if "seconds" not in key}
            if isinstance(value, list):
                return [untimed(item) for item in value]
            return value
        archive_parity = all(untimed(observed.get(key)) == untimed(expected.get(key))
                             for key in ("warm_state_sha256", "final_state_sha256", "observations", "diagnostic"))
    compact = dict(method=METHOD,
                   source=json.loads((output / "declaration.json").read_text())["source"],
                   original_control_exact_parity=archive_parity,
                   torch=torch_version(),
                   identity_cold_parity=result["identity_cold_parity"],
                   variants={k: dict(status=v["status"],
                                     local_stability=v["local_stability"], final=v["final"])
                             for k, v in result["variants"].items()})
    (output / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(json.dumps(dict(event="WARM_DONE", **compact)), flush=True)


def torch_version():
    import torch
    return torch.__version__


def cold(output):
    import torch
    from benchmarks.transfer_suite.compare_defaults import plan
    from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy

    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="value_transport_candidate", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stages = []
    began = time.perf_counter()
    for task in ("trajectory", "mode_hold"):
        spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
        recorder = None
        try:
            with value_transport_candidate(task=task) as (recorder, source):
                result, context = run_legacy(spec, recipe, noise,
                    model_policy=declared_model_policy(config))
            verdict = test_verdict(spec, result)
            data = dict(result=result, applied=context["applied"],
                        noise=context["noise_receipt"], dynamics=recorder.receipt(),
                        spec=spec, verdict=verdict, shared_gate_eligible=False,
                        scratch_optimizer_policy=METHOD)
            (output / f"{task}.json").write_text(json.dumps(data, allow_nan=False) + "\n")
            row = dict(task=task, verdict=verdict, live=result["live"], seconds=result["seconds"])
            stages.append(row)
            print(json.dumps(dict(event="STAGE_DONE", **row)), flush=True)
            if not verdict["passed"]:
                break
        except Exception as error:
            import traceback
            (output / f"{task}.error.json").write_text(json.dumps(
                dict(task=task, error=repr(error), traceback=traceback.format_exc(),
                     completed_outer_steps=None if recorder is None else recorder.outer_steps),
                indent=2) + "\n")
            stages.append(dict(task=task, error=repr(error)))
            break
    status = dict(stages=stages, seconds=time.perf_counter() - began)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(dict(event="DONE", **status)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("warm", "cold"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    previous = None
    if args.phase == "cold":
        if args.previous is None:
            raise ValueError("cold gate requires a passing warm summary")
        previous = json.loads(args.previous.read_text())
        value = previous["variants"]["value"]
        if (not previous["identity_cold_parity"] or value["status"] != "PASS"
                or not value["local_stability"]["pass_all"]
                or value["local_stability"]["checks"] != 200):
            raise RuntimeError("warm gate did not pass; cold acquisition forbidden")
    declaration = _declaration(args.output, args.phase)
    print(json.dumps(dict(event="DECLARED", phase=args.phase, method=METHOD,
                          output=str(args.output))), flush=True)
    if args.phase == "cold":
        if previous.get("method") != METHOD or previous.get("source") != declaration["source"]:
            raise RuntimeError("warm gate method/source differs from this frozen candidate")
        (args.output / "previous-gate.json").write_bytes(args.previous.read_bytes())
        cold(args.output)
    else:
        warm(args.output)


if __name__ == "__main__":
    main()
