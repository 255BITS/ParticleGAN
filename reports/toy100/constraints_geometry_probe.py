"""Predeclared, fixed-seed geometry/capacity ablation of the shared toy100 recipe.

The first stage runs every declared row on grid100 for the full frozen 7,000
updates, five final 20k-draw checks, and separate 100k holdout. Only a strict
grid PASS may advance to rotated100 and staggered100 with the same config.
No labels, target centers, mode count, or target sigma initialize any model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_gate
from benchmarks.toy100.gate import evaluate_suite as coverage_gate
from benchmarks.toy100.train import _source_provenance
from benchmarks.toy_suite import _check_toy100_policy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/toy100/shared_candidate.json"
CONFIGS = ROOT / "configs/toy100/constraints_geometry_v1"
MANIFEST = ROOT / "reports/toy100/constraints_geometry_v1_manifest.json"
DEFAULT_OUTPUT = Path("/dev/shm/particlegan-constraints-geometry-v1-346b1ed")
BASE_SHA256 = "6de24743336a3ce7e922deceab91dff0688bdb28ae8f5e6ff2c8b88b9f2bec27"
REMOVE = object()
ROWS = (
    ("control", {}),
    ("mlp_normal_z2", {"toy100_model": REMOVE}),
    ("mlp_normal_z4", {"toy100_model": REMOVE, "z_dim": 4}),
    ("affine_normal", {"toy100_model": "affine_normal_v1"}),
    ("affine_normal_random", {"toy100_model": "affine_normal_random_v1"}),
    ("affine_square_random", {"toy100_model": "affine_square_random_v1"}),
    ("affine_empirical_box", {"toy100_model": "affine_empirical_box_v1"}),
    ("affine_empirical_box_random", {"toy100_model": "affine_empirical_box_random_v1"}),
    ("fourier0", {"fourier": 0}),
    ("fourier1", {"fourier": 1}),
    ("fourier2", {"fourier": 2}),
    ("particles10000", {"num_particles": 10_000}),
    ("particles5000", {"num_particles": 5_000}),
    ("batch1024", {"batch_size": 1024}),
    ("batch512", {"batch_size": 512}),
    ("critic64", {"d_hidden": 64}),
    ("critic32", {"d_hidden": 32}),
    ("critic_depth2", {"n_hidden": 2}),
    ("lean_joint", {"num_particles": 10_000, "batch_size": 1024,
                    "d_hidden": 64, "fourier": 2}),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare() -> None:
    if digest(BASE) != BASE_SHA256:
        raise RuntimeError("winning base config changed")
    if MANIFEST.exists() or CONFIGS.exists():
        raise FileExistsError("declared manifest/configs already exist")
    base = json.loads(BASE.read_text())
    entries = []
    for slug, delta in ROWS:
        config = dict(base)
        config["name"] = f"constraints_geometry_{slug}"
        for key, value in delta.items():
            if value is REMOVE:
                config.pop(key)
            else:
                config[key] = value
        path = CONFIGS / f"{slug}.json"
        write_json(path, config)
        entries.append({"slug": slug, "config": str(path.relative_to(ROOT)),
                        "config_sha256": digest(path),
                        "delta": {key: "<absent>" if value is REMOVE else value
                                  for key, value in delta.items()}})
    manifest = {
        "protocol": "constraints-geometry-v1", "base_config_sha256": BASE_SHA256,
        "source_sha256": _source_provenance(include_policy=True)["source_sha256"],
        "orchestrator_sha256": digest(Path(__file__)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(),
        "fixed_seed": 1234, "fixed_budget": 7000, "eval_samples": 20_000,
        "holdout_samples": 100_000, "first_stage": "grid100 all rows",
        "promotion": "strict grid PASS, then same config rotated100 and staggered100",
        "old_control": "/dev/shm/particlegan-constraints-geometry-f0595c4/baseline-old6/episode",
        "rows": entries,
    }
    write_json(MANIFEST, manifest)
    print(f"PREPARED {len(entries)} configs, source {manifest['source_commit']}", flush=True)


def verify(manifest):
    if manifest["protocol"] != "constraints-geometry-v1":
        raise ValueError("wrong protocol")
    if digest(BASE) != manifest["base_config_sha256"]:
        raise RuntimeError("winning base config changed")
    if _source_provenance(include_policy=True)["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("source changed after predeclaration")
    if digest(Path(__file__)) != manifest["orchestrator_sha256"]:
        raise RuntimeError("orchestrator changed after predeclaration")
    for row in manifest["rows"]:
        if digest(ROOT / row["config"]) != row["config_sha256"]:
            raise RuntimeError(f"config changed: {row['slug']}")


def run_one(row, output: Path, problem: str):
    slug = row["slug"]
    directory = output / "rows" / slug / problem
    if directory.exists():
        raise FileExistsError(f"run output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    log = directory.parent / f"{problem}.log"
    command = ["ionice", "-c2", "-n4", sys.executable, "-u", "-m", "benchmarks.toy100",
               "run", "--config", str(ROOT / row["config"]), "--output", str(directory),
               "--problem", problem, "--require-accuracy", "--no-render"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", ATEN_CPU_CAPABILITY="avx2")
    print(f"START {slug} {problem} log={log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    episode = directory / problem
    result = {"slug": slug, "problem": problem, "exit_code": process.returncode,
              "config_sha256": row["config_sha256"], "episode": str(episode)}
    try:
        coverage = coverage_gate(directory, problem=problem, write=False)
        accuracy = accuracy_gate(directory, problem=problem, write=False)
        summary = json.loads((episode / "summary.json").read_text())
        config = json.loads((ROOT / row["config"]).read_text())
        _check_toy100_policy(episode, summary, summary["config"],
                             declared_model_policy(config))
        c = coverage["problems"][problem]
        a = accuracy["problems"][problem]
        result.update(status="PASS" if c["passed"] and a["passed"] else "FAIL",
                      coverage=c["status"], accuracy=a["status"],
                      terminal_accuracy_passes=sum(x["passed"] for x in a["terminal_checks"]),
                      coverage_final=summary.get("final", {}).get("live"),
                      final=a.get("final_metrics"), holdout=a.get("holdout_metrics"),
                      first_full_step=summary.get("first_full_coverage_step", {}).get("live"),
                      model_policy=summary.get("model_policy"),
                      source_sha256=summary["provenance"]["source_sha256"])
    except Exception as error:
        result.update(status="INVALID", error=repr(error))
    write_json(directory.parent / f"{problem}_result.json", result)
    final = result.get("final") or {}
    coverage_final = result.get("coverage_final") or {}
    print(f"DONE {slug} {problem} {result['status']} coverage={result.get('coverage')} "
          f"checks={result.get('terminal_accuracy_passes')} "
          f"modes={coverage_final.get('modes')} HQ={final.get('precision')} "
          f"TV={final.get('mass_tv')} center={final.get('center_rms_sigma')} "
          f"exit={process.returncode}", flush=True)
    return result


def run_grid(output: Path, workers: int):
    manifest = json.loads(MANIFEST.read_text())
    verify(manifest)
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, row, output, "grid100"): row["slug"]
                   for row in manifest["rows"]}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    verify(manifest)
    order = {row["slug"]: i for i, row in enumerate(manifest["rows"])}
    write_json(output / "grid_results.json",
               {"protocol": manifest["protocol"], "source_commit": manifest["source_commit"],
                "rows": sorted(results, key=lambda row: order[row["slug"]])})
    print(f"GRID COMPLETE {sum(row['status']=='PASS' for row in results)}/{len(results)} PASS",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "grid"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if not 1 <= args.workers <= 6:
            raise ValueError("workers must be 1..6")
        run_grid(args.output, args.workers)


if __name__ == "__main__":
    main()
