"""Reproducible loss-budget warm filter and fail-fast cold acquisition."""
import argparse
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100.warm_equilibrium_probe import run_warm_variants, constant_rate_context
from reports.toy100.loss_budget_scratch import loss_budget


@contextmanager
def warm(state, prefix, **options):
    with constant_rate_context(state, lr=.00425) as rate, loss_budget(state=state, **options) as receipt:
        receipt["fixed_rate"] = rate
        yield receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    rows = [dict(name="network", scope="network"), dict(name="joint", scope="joint"),
            dict(name="observed", scope="network", observe_only=True)]
    from benchmarks.transfer_suite import suite
    archive = suite.snapshot(args.output)
    sources = {}
    for name in ("reports/toy100/loss_budget_scratch.py", "reports/toy100/loss_budget_probe.py",
                 "reports/toy100/confidence_dynamics_scratch.py", "benchmarks/toy100/models.py"):
        target = args.output / Path(name).name
        shutil.copyfile(ROOT / name, target)
        sources[name] = dict(file=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    declaration = dict(config=config, rows=rows, sources=sources, source_archive=archive,
                       shared_gate_eligible=False, seed=0, cold_from=str(args.cold_from))
    (args.output / "declaration.json").write_text(json.dumps(declaration, indent=2)+"\n")
    if args.cold_from is None:
        variants = {"identity": lambda state, prefix: nullcontext()}
        for row in rows:
            options = {k:v for k,v in row.items() if k != "name"}
            variants[row["name"]] = lambda state, prefix, opts=options: warm(state, prefix, **opts)
        result = run_warm_variants(config, variants, output_dir=args.output / "forks")
    else:
        from benchmarks.transfer_suite.compare_defaults import plan
        from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
        from benchmarks.transfer_suite.protocol import test_verdict
        from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, declared_model_policy
        summary = json.loads((args.cold_from / "results.json").read_text())
        if not summary["identity_cold_parity"]:
            raise RuntimeError("warm identity parity required")
        config.update(lr_floor=1., lr_anneal_start=0.)
        config.pop("network_lr_horizon_cap"); config.pop("network_lr_floor")
        recipe, noise, _ = declared_recipe(config)
        result = []
        for row in rows[:2]:
            if summary["variants"][row["name"]]["status"] != "PASS":
                continue
            for task in ("trajectory", "mode_hold"):
                spec = next(job["spec"] for job in plan() if job["spec"]["name"] == task)
                with loss_budget(scope=row["scope"]) as receipt:
                    evidence, context = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
                verdict = test_verdict(spec, evidence)
                data = dict(evidence=evidence, applied=context['applied'], noise=context['noise_receipt'], verdict=verdict, receipt=receipt,
                            config=config, spec=spec, shared_gate_eligible=False)
                path = args.output / f"{row['name']}-{task}.json.gz"
                path.write_bytes(gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0))
                status = dict(name=row["name"], task=task, verdict=verdict,
                              live=evidence["live"], seconds=evidence["seconds"])
                result.append(status)
                print(json.dumps(status), flush=True)
                if not verdict["passed"]:
                    break
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
