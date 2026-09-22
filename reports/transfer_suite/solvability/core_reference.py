"""Check the shared cap-coefficient challenger against all nine required hosts."""
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.smart_descent import evaluate, study
from benchmarks.transfer_suite import suite, vector_tasks
from benchmarks.transfer_suite.protocol import required_tasks, test_verdict


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    driver = Path(__file__).read_bytes()
    (output / "core_reference.py").write_bytes(driver)
    protocol["driver_sha256"] = hashlib.sha256(driver).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    candidate = replace(study.BASE, name="cap10", reg_coeff=10.)
    policy = vector_tasks.fixed_policy()
    rows = []
    for spec in required_tasks():
        name = spec["name"]
        print(f"START cap10 {name}", flush=True)
        suite.verify_source(protocol)
        with patch.object(study, "BASE", candidate):
            result = evaluate.fixed_toy(name, policy)
        row = dict(spec=spec, candidate=asdict(candidate), policy=policy,
                   result=result, verdict=test_verdict(spec, result))
        raw = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        file = f"episodes/cap10__{name}.json.gz"
        (output / file).write_bytes(gzip.compress(raw, mtime=0))
        rows.append({k: v for k, v in row.items() if k != "result"} |
                    dict(artifact=file, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                         live=result.get("live"), ema=result.get("ema"), seconds=result["seconds"]))
        (output / "index.json").write_text(json.dumps(dict(records=rows), indent=2) + "\n")
        print(f"DONE cap10 {name} {row['verdict']['status']} {result.get('live')}", flush=True)
    suite.verify_source(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
