"""Run the fixed-seed 2D extraction and optionally audit historical model-glue artifacts."""

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from .task import ARMS, CLOUDS, PROTOCOL, TASKS, Game, data, metrics, state_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def provenance():
    root = Path(__file__).resolve().parents[2]
    names = ["benchmarks/paired_error_2d/task.py", "benchmarks/paired_error_2d/run.py",
             "benchmarks/paired_error_2d/__main__.py",
             "particlegan/gan_loss.py", "particlegan/grad_regularizers.py",
             "particlegan/vicreg_loss.py", "particlegan/recipes.py"]
    return dict(protocol=PROTOCOL, arms=ARMS, torch=str(torch.__version__),
                python=platform.python_version(), device="cpu", threads=1,
                sources={p: digest(root / p) for p in names})


def identifier(task, arm, cloud):
    return f"{task}-{arm}-{cloud}-s0"


def train(output, task, arm, cloud):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(output) / "runs" / identifier(task, arm, cloud)
    root.mkdir(parents=True, exist_ok=True)
    x, y = data(task, "train")
    vx, vy = data(task, "validation")
    game = Game(y, arm, cloud)
    averaged = copy.deepcopy(game.model).requires_grad_(False)
    initial = dict(generator=state_hash(game.model), critic=state_hash(game.critic))
    start_cloud = game.model.particles.detach().clone()
    history, first, elapsed = [], 0, 0.
    stamp = provenance()
    latest = root / "latest.pt"
    if latest.exists():
        state = torch.load(latest, weights_only=True, map_location="cpu")
        if state["provenance"] != stamp:
            raise ValueError("Source, protocol or runtime changed on resume")
        game.load_state_dict(state)
        history, first, elapsed = state["history"], state["step"], state["elapsed"]
    if first == PROTOCOL["steps"]:
        return json.loads((root / "metrics.json").read_text())
    started = time.monotonic()

    def record(step, losses):
        averaged.load_state_dict(game.ema)
        with torch.no_grad():
            ep, lp = averaged(vx), game.model(vx)
            displacement = float((game.model.particles - start_cloud).norm(dim=1).mean())
            if cloud == "fixed" and not torch.equal(game.model.particles, start_cloud):
                raise AssertionError("Fixed particles moved")
            history.append(dict(step=step, validation=metrics(ep, vy),
                live_validation=metrics(lp, vy), training=metrics(averaged(x), y),
                particle_displacement=displacement, losses=losses))
            best = min(history, key=lambda row: row["validation"]["nmse"])
            write(root / f"samples-{step:06d}.json", dict(source=vx[:128].tolist(),
                target=vy[:128].tolist(), ema=ep[:128].tolist(), live=lp[:128].tolist()))
        state = dict(**game.state_dict(), provenance=stamp, history=history, step=step,
                     elapsed=elapsed + time.monotonic() - started, initial=initial)
        save(latest, state)
        if step == best["step"]:
            save(root / "best.pt", state)
        row = dict(id=identifier(task, arm, cloud), task=task, arm=arm, cloud=cloud,
                   seed=0, history=history, initial=initial, best_step=best["step"],
                   best_validation=best["validation"], elapsed=state["elapsed"],
                   complete=step == PROTOCOL["steps"])
        write(root / "metrics.json", row)
        print(json.dumps(dict(id=row["id"], step=step, ema_nmse=history[-1]["validation"]["nmse"],
                              live_nmse=history[-1]["live_validation"]["nmse"])), flush=True)
        return row

    if first == 0:
        record(0, {})
    for step in range(first + 1, PROTOCOL["steps"] + 1):
        losses = game.update(x, y, step)
        if step % PROTOCOL["record_every"] == 0:
            record(step, losses)
    return json.loads((root / "metrics.json").read_text())


def stable(row, kind):
    tail = row["history"][-PROTOCOL["stable_tail"]:]
    return row["complete"] and len(tail) == PROTOCOL["stable_tail"] and all(
        p[kind]["nmse"] <= PROTOCOL["validation_nmse_bound"]
        and p[kind]["p95_distance"] <= PROTOCOL["validation_p95_bound"] for p in tail)


def freeze_and_evaluate(output, runs):
    root = Path(output)
    if len(runs) != len(TASKS) * len(ARMS) * len(CLOUDS) or not all(r["complete"] for r in runs):
        raise ValueError("All twelve runs must finish before test evaluation")
    choices = [dict(id=r["id"], task=r["task"], arm=r["arm"], cloud=r["cloud"], step=r["best_step"],
                    validation=r["best_validation"], sha256=digest(root / "runs" / r["id"] / "best.pt"))
               for r in sorted(runs, key=lambda row: row["id"])]
    selection = dict(rule="minimum validation EMA NMSE, first step on ties", choices=choices,
                     provenance=provenance(), test_role="historical replication; not a new holdout")
    path = root / "selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Frozen selection changed")
    write(path, selection)
    # The historical test set is intentionally reused to audit exact reproduction.
    # Checkpoint choices are still frozen before reading it in this runner.
    rows = []
    torch.set_num_threads(1)
    for choice in choices:
        checkpoint = root / "runs" / choice["id"] / "best.pt"
        if digest(checkpoint) != choice["sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        _, ty = data(choice["task"], "train")
        game = Game(ty, choice["arm"], choice["cloud"])
        tx, target = data(choice["task"], "test")
        values = {}
        with torch.no_grad():
            for label, key in [("ema", "ema"), ("live", "model")]:
                game.model.load_state_dict(state[key])
                values[label] = metrics(game.model(tx), target)
        run = next(r for r in runs if r["id"] == choice["id"])
        rows.append(dict(**choice, test=values, initial=run["initial"],
                         stable_ema=stable(run, "validation"),
                         stable_live=stable(run, "live_validation"), history=run["history"]))
    result = dict(selection_sha256=digest(path), results=rows)
    write(root / "results.json", result)
    return result


def compare_states(left, right):
    """Compare every nested tensor and scalar; report real discrepancies without tolerance masking."""
    differences = []

    def visit(a, b, path):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                differences.append(dict(path=path, reason="tensor shape/dtype"))
            elif not torch.equal(a, b):
                delta = (a.double() - b.double()).abs()
                differences.append(dict(path=path, max_absolute_difference=float(delta.max())))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                differences.append(dict(path=path, reason="dictionary keys"))
            else:
                for k in a:
                    visit(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                differences.append(dict(path=path, reason="sequence length"))
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{path}/{i}")
        elif a != b:
            differences.append(dict(path=path, left=a, right=b))

    visit(left, right, "")
    return dict(exact=not differences, differences=differences)


def audit_reference(output, reference, result):
    """Read artifacts only. The application code is not imported or executed."""
    root, reference = Path(output), Path(reference)
    original = json.loads((reference / "test-results.json").read_text())
    prior = {r["id"]: r for r in original["results"]}
    rows = []
    for row in result["results"]:
        old_metrics = json.loads((reference / "runs" / row["id"] / "metrics.json").read_text())
        ours = torch.load(root / "runs" / row["id"] / "latest.pt", weights_only=True, map_location="cpu")
        old = torch.load(reference / "runs" / row["id"] / "latest.pt", weights_only=True, map_location="cpu")
        checks = {name: compare_states(ours[name], old[name])
                  for name in ("model", "critic", "ema", "g", "d", "sampler", "torch_rng")}
        curves = [{k: p[k] for k in ("step", "validation", "live_validation", "training", "particle_displacement")}
                  for p in row["history"]]
        old_curves = [{k: p[k] for k in curves[0]} for p in old_metrics["history"]]
        rows.append(dict(id=row["id"], states=checks,
            curves_exact=curves == old_curves, test_exact=row["test"] == prior[row["id"]]["test"],
            initial_exact=row["initial"] == old_metrics["initial"],
            selection_exact=row["step"] == prior[row["id"]]["step"]))
    audit = dict(reference_results_sha256=digest(reference / "test-results.json"),
                 reference_config_sha256=digest(reference / "config.json"), rows=rows,
                 all_exact=all(all(c["exact"] for c in r["states"].values()) and r["curves_exact"]
                    and r["test_exact"] and r["initial_exact"] and r["selection_exact"] for r in rows))
    write(root / "reference-audit.json", audit)
    return audit


def markdown(result, audit=None):
    lines = ["# Paired 2D transport: MSE-free cap/schedule transfer", "",
        "Seed 0, 6,000 updates, batch 64, 1,024 training points, 1,024 validation points. "
        "Each row uses its validation-selected EMA checkpoint; live is scored at that same step. "
        "Historical test points (4,096) are reused for reproduction, not claimed as a fresh holdout.", "",
        "| Task | Recipe | Cloud | EMA test NMSE ↓ | Live test NMSE ↓ | EMA p95 ↓ | Stable live |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for r in result["results"]:
        lines.append(f'| {r["task"]} | {r["arm"]} | {r["cloud"]} | '
            f'{r["test"]["ema"]["nmse"]:.9g} | {r["test"]["live"]["nmse"]:.9g} | '
            f'{r["test"]["ema"]["p95_distance"]:.7g} | {r["stable_live"]} |')
    lines += ["", "## Matched recipe comparisons", "",
              "Geometric mean candidate/baseline EMA error ratio over the two tasks. "
              "One seed, descriptive only; do not use this to claim a universal winner.", ""]
    for cloud in CLOUDS:
        for arm in list(ARMS)[1:]:
            ratios = []
            for task in TASKS:
                selected = {r["arm"]: r for r in result["results"]
                            if r["task"] == task and r["cloud"] == cloud}
                ratios.append(selected[arm]["test"]["ema"]["nmse"]
                              / selected["baseline"]["test"]["ema"]["nmse"])
            ratio = math.exp(statistics.mean(math.log(v) for v in ratios))
            lines.append(f"- {cloud}, {arm}: **{ratio:.6f}×**; task ratios {ratios}.")
    if audit:
        lines += ["", "## Application-artifact reproduction", "",
            f'All {len(audit["rows"])} extracted runs exact: **{audit["all_exact"]}**. '
            "Comparison includes full final G/D/EMA/Adam/sampler/global-RNG state, all recorded "
            "validation metrics, checkpoint selections, and historical live/EMA test scores. "
            "See `reference-audit.json` for every mismatch; no tolerance turns a mismatch into PASS."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/paired_error_2d"))
    parser.add_argument("--reference-artifacts", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = provenance()
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != stamp:
        raise ValueError("Existing protocol/source/runtime differs")
    write(path, stamp)
    planned = [(t, a, c) for t in TASKS for a in ARMS for c in CLOUDS]
    write(args.output / "plan.json", dict(jobs=planned, seed=0, protocol_sha256=digest(path)))
    runs, errors = [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(train, args.output, *job): job for job in planned}
        for future in as_completed(pending):
            try:
                runs.append(future.result())
            except Exception as error:
                errors.append(dict(job=pending[future], error=repr(error)))
            write(args.output / "progress.json", dict(completed=len(runs), planned=len(planned), errors=errors))
    if errors:
        raise RuntimeError(errors)
    result = freeze_and_evaluate(args.output, runs)
    audit = audit_reference(args.output, args.reference_artifacts, result) if args.reference_artifacts else None
    (args.output / "README.md").write_text(markdown(result, audit))
    write(args.output / "completion.json", dict(wall_seconds=time.monotonic() - started,
        completed=len(runs), updates=len(runs) * PROTOCOL["steps"],
        audit_exact=None if audit is None else audit["all_exact"]))
    print(markdown(result, audit), flush=True)
    return 0 if audit is None or audit["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
