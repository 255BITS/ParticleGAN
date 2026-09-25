"""Predeclared, fail-fast signal-quality batch on unchanged frozen hosts."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import fcntl
import gzip
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuous_screen import write, sha, source_hashes, mark_scratch, verify_receipt

ORDER = ["trajectory", "mode_hold", "residual_student", "img_stripes2", "img_bars4",
         "vector_overlap", "img_blobs4", "img_intensity2", "vector_unequal_mass", "vector_unequal_width"]


def append(path, value):
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def capture(directory, task):
    """Observe final host state without advancing any training RNG stream."""
    import torch
    from benchmarks.locked_shared import mode_hold, trajectory
    from particlegan import GANTrainer
    module = {"mode_hold": mode_hold, "trajectory": trajectory}.get(task)
    if module is None:
        original_step = GANTrainer.step
        def trainer_step(trainer, *args, **kwargs):
            result = original_step(trainer, *args, **kwargs)
            if trainer.completed_steps == trainer.recipe.total_steps:
                payload = {"trainer": trainer.state_dict(), "task": task}
                caller = inspect.currentframe().f_back.f_locals
                context = caller.get("context", {})
                if "data_rng" in context:
                    payload["data_rng"] = context["data_rng"].get_state()
                with torch.random.fork_rng(devices=[]):
                    payload["samples"] = trainer.sample(4096).detach()
                torch.save(payload, directory / "final-state.pt")
            return result
        with patch.object(GANTrainer, "step", trainer_step):
            yield
        return
    original = module.checkpoint
    budget = 1200 if task == "mode_hold" else 400

    def checkpoint(step, measure):
        original(step, measure)
        if step % 200 == 0:
            print(json.dumps({"event": "UPDATE", "task": task, "step": step}), flush=True)
        if step != budget:
            return
        values = inspect.currentframe().f_back.f_locals
        payload = {"step": step, "task": task, "torch_rng": torch.get_rng_state(),
                   "models": {k: deepcopy(values[k].state_dict()) for k in ("generator", "critic", "prior")},
                   "optimizers": {k: deepcopy(values[k].state_dict()) for k in ("opt_g", "opt_d")}}
        for key in ("stream",):
            if key in values:
                payload[key + "_rng"] = values[key].get_state()
        policy = values.get("noise_policy")
        if policy is not None:
            payload["noise_policy"] = deepcopy(policy.__dict__)
        for key in ("ema_g", "ema_z"):
            if key in values:
                payload[key] = deepcopy(values[key])
        # Eval-only sample draw, with global and noise RNG restored.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            if task == "mode_hold":
                with policy.evaluation(step):
                    latent, _ = values["prior"].sample(mode_hold.EVAL_N, generator=torch.Generator().manual_seed(9))
                    payload["samples"] = values["generator"](latent).detach().clone()
            else:
                with policy.evaluation(step):
                    payload["samples"] = values["generator"](values["slow"], values["prior"].z).detach().clone()
        torch.save(payload, directory / "final-state.pt")

    with patch.object(module, "checkpoint", checkpoint):
        yield


def worker(output, row, sources, ledger, tasks):
    import torch
    from critic_signal import signal_policy
    from benchmarks.transfer_suite.toy100_compatibility import run
    from benchmarks.transfer_suite.protocol import test_verdict
    from benchmarks.toy_suite import _episode_rows
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    directory = Path(output) / row["tag"]
    directory.mkdir()
    write(directory / "config.json", row["config"])
    write(directory / "options.json", row["options"])
    result = {"tag": row["tag"], "stages": [], "status": "PASS", "shared_gate_eligible": False}
    started = time.perf_counter()
    with (directory / "run.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        for task in tasks:
            if result["status"] != "PASS":
                event = dict(candidate=row["tag"], gate=task, status="SKIPPED", seconds=0.,
                             metrics={}, artifact=str(directory / "status.json"), reason="earlier gate did not pass")
                append(ledger, event)
                continue
            stage_started = time.perf_counter()
            stage_dir = directory / task
            print(json.dumps(dict(event="START", candidate=row["tag"], gate=task)), flush=True)
            try:
                changed = [name for name, digest in sources.items() if sha(ROOT / name) != digest]
                if changed:
                    raise RuntimeError(f"sources changed: {changed}")
                with signal_policy(row["options"]) as receipt, capture(directory, task):
                    records = run(directory / "config.json", stage_dir, tasks=[task])
                record = records[0]
                artifact = stage_dir / record["artifact"]
                saved = json.loads(gzip.decompress(artifact.read_bytes()))
                if saved["result"].get("error"):
                    raise RuntimeError(saved["result"]["error"])
                verdict = test_verdict(saved["spec"], saved["result"])
                if verdict != record["verdict"] or verdict["status"] not in ("PASS", "FAIL"):
                    raise RuntimeError("independent saved-episode regrade mismatch or incomplete budget")
                verify_receipt(receipt, row["config"], task=task)
                if len(receipt["updates"]) != 2 * saved["spec"]["steps"]:
                    raise RuntimeError("host did not perform full G/D budget")
                (stage_dir / "signal-policy.json.gz").write_bytes(gzip.compress(
                    json.dumps(receipt, sort_keys=True, allow_nan=False).encode(), mtime=0))
                # The existing scratch envelope prevents accidental production promotion.
                mark_scratch(stage_dir, row["options"])
                audit = _episode_rows(stage_dir, (task,), candidate=True, allow_scratch=True)
                if audit["status"] not in ("PASS", "FAIL") or audit["passed"] != int(verdict["passed"]):
                    raise RuntimeError(f"source/config/noise evidence audit: {audit}")
                state = directory / "final-state.pt"
                if state.exists():
                    state.replace(stage_dir / "final-state.pt")
                event = dict(candidate=row["tag"], gate=task, status=verdict["status"],
                             seconds=time.perf_counter() - stage_started,
                             metrics={"live": saved["result"].get("live"), "verdict": verdict,
                                      "observations": saved["result"].get("observations", []),
                                      "frozen_evidence_audit": audit["status"]}, artifact=str(artifact))
                if (stage_dir / "final-state.pt").exists():
                    event["checkpoint_sha256"] = sha(stage_dir / "final-state.pt")
            except Exception:
                event = dict(candidate=row["tag"], gate=task, status="ERROR",
                             seconds=time.perf_counter() - stage_started, metrics={},
                             artifact=str(directory / "run.log"), error=traceback.format_exc())
                print(event["error"], flush=True)
            append(ledger, event)
            result["stages"].append(event)
            if event["status"] != "PASS":
                result["status"] = event["status"]
            write(directory / "status.json", result)
            print(json.dumps({"event": "DONE", **{k:event[k] for k in ("candidate", "gate", "status", "seconds")},
                              "live": event["metrics"].get("live")}), flush=True)
    result["seconds"] = time.perf_counter() - started
    write(directory / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", default=ORDER)
    parser.add_argument("--stop-after-survivor", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("one to four workers")
    rows = json.loads(args.declaration.read_text())
    if len({r["tag"] for r in rows}) != len(rows):
        raise ValueError("duplicate candidate tags")
    for row in rows:
        cfg = row["config"]
        if cfg["lr_floor"] != 1 or "network_lr_horizon_cap" in cfg or cfg["prior_reg"] != 0:
            raise ValueError("fixed LRs and exclusively adversarial generator required")
    args.output.mkdir(parents=True, exist_ok=False)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    for name in ("critic_signal.py", "critic_signal_screen.py"):
        rel = "reports/toy100/" + name
        sources[rel] = sha(ROOT / rel)
    manifest = dict(rows=rows, tasks=args.tasks, source_sha256=sources, workers=args.workers,
                    fixed_seeds=True, host_budgets_unchanged=True, scoring_unchanged=True,
                    coverage_auxiliary_disabled=True, shared_gate_eligible=False)
    write(args.output / "manifest.json", manifest)
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for name in sources:
            archive.add(ROOT / name, arcname=name)
    print(json.dumps(dict(event="PREDECLARED", rows=len(rows), output=str(args.output))), flush=True)
    started = time.perf_counter()
    results = []
    # One-row harness canary before dispatching further rows; ERROR halts expansion.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        pending = {}
        row_iter = iter(rows)
        def submit(row):
            future = pool.submit(worker, str(args.output.resolve()), row, sources,
                                 str(args.ledger.resolve()), args.tasks)
            pending[future] = row["tag"]
        submit(next(row_iter))
        halt = False
        halted_for_error = False
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                tag = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    event = dict(candidate=tag, gate=args.tasks[0], status="ERROR", seconds=0.,
                                 metrics={}, artifact=str(args.output / "manifest.json"),
                                 error=traceback.format_exc())
                    append(args.ledger, event)
                    for task in args.tasks[1:]:
                        append(args.ledger, dict(candidate=tag, gate=task, status="SKIPPED", seconds=0.,
                               metrics={}, artifact=event["artifact"], reason="worker setup error"))
                    result = dict(tag=tag, status="ERROR", stages=[event], seconds=0.)
                results.append(result)
                write(args.output / "results.json", results)
                print(json.dumps(dict(event="CANDIDATE_DONE", tag=result["tag"], status=result["status"],
                                      seconds=result["seconds"], stages=[{k:s[k] for k in ("gate", "status")} for s in result["stages"]])), flush=True)
                halted_for_error = halted_for_error or result["status"] == "ERROR"
                halt = halted_for_error or halt or (args.stop_after_survivor and result["status"] == "PASS")
            if not halt:
                while len(pending) < args.workers:
                    row = next(row_iter, None)
                    if row is None:
                        break
                    submit(row)
    write(args.output / "summary.json", dict(executed_candidates=len(results),
          elapsed_seconds=time.perf_counter()-started, halted_for_error=halted_for_error,
          unstarted_candidates=len(rows)-len(results),
          survivors=[r["tag"] for r in results if r["status"] == "PASS"]))


if __name__ == "__main__":
    main()
