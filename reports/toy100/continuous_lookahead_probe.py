"""Reproduce the finite cold-start Lookahead-Minmax comparison on live iterates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100.continuous_probe import run_probe
from reports.toy100.continuous_lookahead import lookahead


def worker(root, row, config):
    path = Path(root) / row["tag"]
    path.mkdir()
    config = dict(config, lr=row["lr"], name=row["tag"])
    with lookahead(row["k"], row["alpha"]) as receipt:
        evidence = run_probe(config, mode="constant", steps=1200, diagnostic_every=1)
    dense = [point for point in evidence["diagnostic"] if point["step"] >= 1000]
    passed = sum(point["modes"] == 8 and point["hq"] >= .9 for point in dense)
    evidence.update(lookahead=row, shared_gate_eligible=False,
        source_adapter_sha256=hashlib.sha256((ROOT / "reports/toy100/continuous_lookahead.py").read_bytes()).hexdigest(),
        dense_terminal=dict(passed=passed, checks=len(dense)))
    for name, data in (("evidence", evidence), ("receipt", receipt)):
        (path / f"{name}.json.gz").write_bytes(gzip.compress(
            json.dumps(data, allow_nan=False).encode(), mtime=0))
    summary = dict(row, official=evidence["stationary"]["pass_all"],
                   dense_passed=passed, dense_checks=len(dense),
                   final=evidence["final"], seconds=evidence["seconds"])
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    rows = [dict(tag=f"la_k{k}_lr{str(lr).replace('.', 'p')}", k=k, alpha=.5, lr=lr)
            for k in (2, 5, 10) for lr in (.001, .00425)]
    rows.append(dict(tag="la_control", k=5, alpha=1., lr=.00425))
    (args.output / "declaration.json").write_text(json.dumps(dict(
        rows=rows, config=config, steps=1200, diagnostic_every=1), indent=2) + "\n")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, str(args.output), row, config) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
