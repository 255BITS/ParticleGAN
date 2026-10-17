"""Run the frozen three-arm CPU diagnostic; bulk artifacts stay outside Git.

PYTHONPATH=. python -u examples/run_e22_routed_convergence.py --out runs/routed-convergence-v1
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import torch
import particlegan

from e22_routed_convergence import (
    ARMS, SPLITS, checkpoint, digest, evaluate,
    evaluation_panels, make_data, make_loop, modules, parameters, reachability_witness, restore, score_residual, update,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_convergence_v1.json"
SOURCES = (CARD, Path(__file__).resolve(), ROOT / "examples/e22_routed_convergence.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for path in SOURCES}


def native_source_hash():
    return digest({str(path.relative_to(ROOT)): sha(path)
                   for path in sorted((ROOT / "particlegan").rglob("*.py"))})


def json_value(value):
    """Preserve native diagnostic sentinels without changing checkpoint state."""
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_diagnostic": repr(value)}
    return value


def frozen_values(loop):
    values = {role + "." + name: tensor for role, model in modules(loop).items()
              for name, tensor in model.state_dict().items()
              if name.endswith("base.weight") or name == "scale" or role in ("encoder", "average_encoder")}
    if loop.policy is not None and not loop.policy.table.requires_grad:
        values["frozen_prior_scaffold"] = loop.policy.table
    return values


def assert_finite(loop):
    for name, parameter in parameters(loop).items():
        if not torch.isfinite(parameter).all() or (parameter.grad is not None and not torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"nonfinite owned tensor or gradient: {name}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def compact(result):
    return {key: value for key, value in result.items() if key != "paired_game_by_context"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; preserve it and choose a new output directory")
    torch.set_num_threads(1)
    contract = json.loads(CARD.read_text())
    execution = contract["execution"]
    steps, cadence = execution["steps_per_arm"], execution["checkpoint_cadence"]
    if steps != 6400 or cadence != 200:
        parser.error("this runner implements the frozen v1 6400/200 law")
    package_path = Path(particlegan.__file__).resolve()
    if package_path.parent != ROOT / "particlegan":
        parser.error("set PYTHONPATH to this checkout; imported a different ParticleGAN")
    native_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--", "particlegan"], cwd=ROOT, text=True):
        parser.error("native package has uncommitted changes")
    if subprocess.run(["git", "diff", "--quiet", contract["native_base_revision"], "--", "particlegan"], cwd=ROOT).returncode:
        parser.error("native package differs from the frozen v1 base; declare a new diagnostic revision")
    bindings = {"source_hashes": source_hashes(), "native_revision": native_revision,
                "native_base_revision": contract["native_base_revision"], "native_source_hash": native_source_hash()}
    args.out.mkdir(parents=True)
    start = time.monotonic()
    log = (args.out / "run.log").open("w", buffering=1)

    def emit(**row):
        line = json.dumps(json_value(row), allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)

    def budget(arm_start=None):
        if source_hashes() != bindings["source_hashes"] or native_source_hash() != bindings["native_source_hash"]:
            raise RuntimeError("frozen diagnostic source/card changed during execution")
        elapsed = time.monotonic() - start
        if elapsed > execution["timeout_seconds_total"]:
            raise TimeoutError("declared total diagnostic budget exhausted")
        if arm_start is not None and time.monotonic() - arm_start > execution["timeout_seconds_per_arm"]:
            raise TimeoutError("declared arm diagnostic budget exhausted")

    receipt = {"status": "running", "task": contract["task_id"], "contract": contract,
               "bindings": bindings, "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                                  "cpu": platform.processor(), "platform": platform.platform(),
                                                  "device": "cpu", "threads": 1}, "arms": {}}
    try:
        data = make_data()
        receipt.update(data_digest=data["digest"], reachability=reachability_witness(data),
                       data_hashes={pool: digest(data[pool]) for pool in SPLITS},
                       raw_coordinate_scale=data["raw_scale"].tolist(),
                       coordinate_scale_floor_count=int((data["raw_scale"] < .04).sum()))
        write_json(args.out / "provenance.json", receipt)
        emit(event="frozen", data_digest=data["digest"], sources=bindings["source_hashes"],
             steps=steps, cadence=cadence, reachability=receipt["reachability"])
        order_reference, paired_reference = [], []
        for arm in ARMS:
            budget()
            arm_start = time.monotonic()
            loop = make_loop(arm, data, bindings=bindings)
            directory = args.out / arm
            directory.mkdir()
            counts = {"generator": sum(parameter.numel() for parameter in loop.G.parameters() if parameter.requires_grad)}
            if loop.policy is not None:
                counts.update(critic=sum(parameter.numel() for parameter in loop.policy.D.parameters() if parameter.requires_grad),
                              router=0 if loop.policy.router is None else sum(parameter.numel() for parameter in loop.policy.router.parameters()),
                              active_bank=loop.policy.table.numel() if loop.policy.table.requires_grad else 0, noise=1)
            record = {"law": loop.law, "active_parameter_counts": counts,
                      "initial_model_digest": digest(loop.G.state_dict()), "checkpoints": [], "moves": 0, "proposal_events": 0,
                      "initial_critic_digest": None if loop.policy is None else digest(loop.policy.D.state_dict())}
            frozen_digest = digest(frozen_values(loop))
            record["initial_frozen_owner_digest"] = frozen_digest
            receipt["arms"][arm] = record
            save(directory / "step-0000.pt", checkpoint(loop))
            recovery_rows = []
            with (directory / "trace.jsonl").open("w", buffering=1) as trace:
                for index in range(steps):
                    if index % 25 == 0:
                        budget(arm_start)
                        assert_finite(loop)
                    row = update(loop)
                    if row["step"] in (801, 802):
                        recovery_rows.append(deepcopy(row))
                    for key in ("loss_g", "loss_d_game", "penalty", "loss_reference_mse"):
                        if key in row and not math.isfinite(row[key]):
                            raise AssertionError(f"nonfinite training observation: {key}")
                    trace.write(json.dumps(json_value(row), allow_nan=False) + "\n")
                    if arm == ARMS[0]:
                        order_reference.append(row["batch_indices"])
                        paired_reference.append(row["paired_base_digest"])
                    elif row["batch_indices"] != order_reference[index]:
                        raise AssertionError("arms consumed different fit-context batches")
                    if arm == ARMS[1] and row["paired_base_digest"] != paired_reference[index]:
                        raise AssertionError("native arms consumed different paired Gaussian bases")
                    record["proposal_events"] += int(row.get("move") is not None)
                    record["moves"] += int((row.get("move") or {}).get("moves", 0))
                    if (index + 1) % cadence == 0 or index + 1 == 802:
                        step = index + 1
                        path = directory / f"step-{step:04d}.pt"
                        state = checkpoint(loop)
                        assert_finite(loop)
                        if digest(frozen_values(loop)) != frozen_digest:
                            raise AssertionError("frozen FAST/EMA host, contexts, scale or prior changed")
                        save(path, state)
                        record["checkpoints"].append({"step": step, "file": path.name, "sha256": sha(path)})
                        emit(event="progress" if step % cadence == 0 else "recovery_witness", arm=arm, step=step, seconds=round(time.monotonic() - arm_start, 3),
                             loss_g=row.get("loss_g"), loss_d_game=row.get("loss_d_game"),
                             loss_reference_mse=row.get("loss_reference_mse"),
                             bank_gradient_rows=row.get("bank_gradient_rows"), controls=row.get("controls"))
            budget(arm_start)
            recovery_start = time.monotonic()
            live_digest = digest(checkpoint(loop))
            with torch.random.fork_rng(devices=[]):
                recovered = make_loop(arm, data, bindings=bindings)
                restore(recovered, torch.load(directory / "step-0800.pt", weights_only=False))
                replay_rows = [update(recovered) for _ in range(2)]
                expected_recovery = torch.load(directory / "step-0802.pt", weights_only=False)
                if digest(replay_rows) != digest(recovery_rows) or digest(checkpoint(recovered)) != digest(expected_recovery):
                    raise AssertionError("own-state 800-to-802 recovery changed native rows or authoritative state")
            if digest(checkpoint(loop)) != live_digest:
                raise AssertionError("private recovery validation changed live training state")
            record["recovery_witness"] = {"from_step": 800, "to_step": 802, "rows_exact": True, "state_exact": True,
                                           "software_updates": 2, "wall_seconds": time.monotonic() - recovery_start}
            budget(arm_start)
            record.update(status="complete", wall_seconds=time.monotonic() - arm_start,
                          final_checkpoint_digest=digest(checkpoint(loop)), final_frozen_owner_digest=digest(frozen_values(loop)))
            write_json(args.out / "provenance.json", receipt)
        panels = evaluation_panels(data)
        if digest({key: value for key, value in data.items() if key != "digest"}) != data["digest"]:
            raise AssertionError("training mutated frozen task inputs/teacher")
        receipt["private_panel_digest"] = digest(panels)
        judges = {}
        for native_arm in ARMS[:2]:
            for step in (800, 6400):
                loop = make_loop(native_arm, data, bindings=bindings)
                restore(loop, torch.load(args.out / native_arm / f"step-{step:04d}.pt", weights_only=False))
                judges[f"{native_arm}@{step}"] = deepcopy(loop.policy.D).eval().requires_grad_(False)
        receipt["judges"] = {name: digest(judge.state_dict()) for name, judge in judges.items()}
        receipt["judge_reference_path"] = {
            name: {pool: {str(fraction): compact(score_residual(judge, data[pool]["context"],
                              fraction * (data[pool]["base"] - data[pool]["targets"]) / data["scale"], panels[pool]))
                          for fraction in (0, .25, .5, 1)} for pool in SPLITS}
            for name, judge in judges.items()}
        receipt["endpoint_scores"], receipt["initial_scores"] = {}, {}
        with (args.out / "common-judge-curves.jsonl").open("w", buffering=1) as curves:
            for arm in ARMS:
                arm_eval_start = time.monotonic()
                loop = make_loop(arm, data, bindings=bindings)
                for step in range(0, steps + 1, cadence):
                    budget()
                    if receipt["arms"][arm]["wall_seconds"] + time.monotonic() - arm_eval_start > execution["timeout_seconds_per_arm"]:
                        raise TimeoutError("declared arm budget exhausted during common-judge scoring")
                    restore(loop, torch.load(args.out / arm / f"step-{step:04d}.pt", weights_only=False))
                    scores = {name: {pool: evaluate(loop, judge, pool, panels) for pool in SPLITS}
                              for name, judge in judges.items()}
                    curves.write(json.dumps({"arm": arm, "step": step, "scores": scores}, allow_nan=False) + "\n")
                    small = {name: {pool: compact(value) for pool, value in pools.items()} for name, pools in scores.items()}
                    if step == 0:
                        receipt["initial_scores"][arm] = small
                    if step in (1600, 6400):
                        receipt["endpoint_scores"][f"{arm}@{step}"] = small
                    if arm == ARMS[1] and step == steps:
                        receipt["particle_code_ablation"] = {name: {pool: compact(evaluate(loop, judge, pool, panels, code_ablation=True))
                                                                    for pool in SPLITS} for name, judge in judges.items()}
                    emit(event="scored", arm=arm, step=step,
                         test_game={name: pools["test"]["paired_game"] for name, pools in small.items()})
                receipt["arms"][arm]["evaluation_seconds"] = time.monotonic() - arm_eval_start
                receipt["arms"][arm]["cumulative_seconds"] = receipt["arms"][arm]["wall_seconds"] + receipt["arms"][arm]["evaluation_seconds"]
                if receipt["arms"][arm]["cumulative_seconds"] > execution["timeout_seconds_per_arm"]:
                    raise TimeoutError("declared arm budget exhausted during final common-judge score")
        gaps = {name: receipt["endpoint_scores"][f"{ARMS[1]}@6400"][name]["test"]["paired_game"]
                      - receipt["endpoint_scores"][f"{ARMS[0]}@6400"][name]["test"]["paired_game"]
                for name in judges if name.endswith("@6400")}
        receipt.update(status="complete", wall_seconds=time.monotonic() - start,
                       endpoint_particle_minus_ordinary_game=gaps,
                       particle_gap_reproduced=all(value > 1e-4 for value in gaps.values()),
                       qualification_credit="none")
        budget()
        write_json(args.out / "receipt.json", receipt)
        emit(event="complete", seconds=receipt["wall_seconds"], particle_gap_reproduced=receipt["particle_gap_reproduced"], gaps=gaps)
    except Exception as error:
        receipt.update(status="incomplete" if isinstance(error, TimeoutError) else "error",
                       error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - start)
        write_json(args.out / "receipt.json", receipt)
        emit(event=receipt["status"], error=receipt["error"])
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
