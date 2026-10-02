"""Observe exact source-family training functions; do not replace their loops.

One CPU1 cohort per family, preserving source inputs, budgets and original
gates. The Python line hook observes real completed updates. Bulk observations
stay outside Git; only compact receipts and actual checkpoint GIFs are copied
into the audit. It does not interpolate states or rerun different seeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

REVISION = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
SOURCES = {"sign": "lib/yue2_particle_toy.py",
           "landing": "lib/safe_fast_landing.py",
           "native": "experiments/toy_particle_native_2d.py"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def digest(value):
    out = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            out.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            out.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(str(key)); add(item[key])
        elif isinstance(item, (tuple, list)):
            out.update(type(item).__name__.encode())
            for part in item:
                add(part)
        else:
            out.update(repr(item).encode())
        out.update(b"\0")
    add(value)
    return out.hexdigest()


def owned_state(local):
    """Models, gradients, modes, optimizers and local/global named RNG states."""
    values = {"global_rng": torch.get_rng_state(), "threads": torch.get_num_threads()}
    for name, value in local.items():
        if isinstance(value, torch.nn.Module):
            values[name] = {"state": value.state_dict(),
                            "modes": {k: m.training for k, m in value.named_modules()},
                            "parameters": {k: (p.requires_grad, p.grad) for k, p in value.named_parameters()}}
        elif isinstance(value, torch.optim.Optimizer):
            values[name] = value.state_dict()
        elif isinstance(value, torch.Generator):
            values[name] = value.get_state()
        elif isinstance(value, torch.Tensor):
            values[name] = (value, value.requires_grad, value.grad if value.is_leaf else None)
    return digest(values)


class Capture:
    def __init__(self, module, family, output, quality):
        import ast
        self.module, self.family, self.output, self.quality = module, family, output, quality
        self.rows, self.arrays, self.protocols = [], {}, {}
        self.parity = {"all_observations_state_and_rng_pure": True,
                       "native_original_series_exact": None, "final_original_metrics_exact": False}
        self.completed = {}
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        wanted = ({"train_collapsed", "train_supervised", "train_paired"} if family == "sign" else
                  {"train_arm"} if family == "landing" else {"pretrain", "run_arm"})
        self.functions = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                # Select completed-update loops rather than parameter/mode
                # traversal loops that run before optimizer construction.
                lines = {n.lineno for n in ast.walk(node) if isinstance(n, ast.For)
                         and isinstance(n.target, ast.Name) and n.target.id in ("step", "_")}
                self.functions[getattr(module, node.name).__code__] = (node.name, lines)

    def trace(self, frame, event, arg):
        if event == "call":
            return self.local_trace if frame.f_code in self.functions else None
        return None

    def local_trace(self, frame, event, arg):
        name, loop_lines = self.functions[frame.f_code]
        local = frame.f_locals
        arm = (name.removeprefix("train_") if self.family == "sign" else
               local["mode"] if self.family == "landing" else
               "pretrain" if name == "pretrain" else local["name"])
        if event == "line" and frame.f_lineno in loop_lines:
            step = local.get("step", local.get("_", -1) + 1)
            maximum = local.get("steps", self.module.PRE if arm == "pretrain" else self.module.FINE) if self.family == "native" else local["steps"]
            cadence = 50 if self.family == "native" else 25
            if step in (0, 1, 10) or step % cadence == 0 or step == maximum:
                if self.completed.get(arm) != step:
                    self.observe(arm, step, local)
        elif event == "return" and isinstance(arg, dict):
            self.final_parity(arm, arg, local)
        return self.local_trace

    def observe(self, arm, step, local):
        before = owned_state(local)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            metrics, arrays = (self.sign(local) if self.family == "sign" else
                               self.landing(local) if self.family == "landing" else self.native(arm, local))
        assert before == owned_state(local), (arm, step, "observation mutated a training owner")
        prefix = f"{arm}_{step:04d}"
        for key, value in arrays.items():
            self.arrays[prefix + "__" + key] = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        row = {"arm": arm, "step": step, "metrics": metrics, "training_state_sha256": before,
               "observation_state_and_rng_pure": True, "array_prefix": prefix}
        self.rows.append(row)
        self.completed[arm] = step
        if arm not in self.protocols and "recipe" in local:
            self.protocols[arm] = {"recipe": local["recipe"].to_dict(),
                                   "optimizer_class": type(local.get("opt", local.get("opt_g"))).__name__}
        with (self.output / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
        print(json.dumps({"event": "observed", "family": self.family, "arm": arm, "step": step,
                          "metrics": metrics}, allow_nan=False), flush=True)

    def sign(self, local):
        m, alpha = self.module, local["alpha"].detach()
        policy = lambda state: (alpha * m.expert_action(state)).clamp(-1, 1)
        gen = torch.Generator().manual_seed(123)
        position = torch.rand(80, 2, generator=gen) * 1.2 - .6
        velocity = torch.rand(80, 2, generator=gen) * .4 - .2
        trajectory = [position.clone()]
        for _ in range(m.HORIZON):
            action = policy(torch.cat([position, velocity], 1))
            velocity = velocity + m.GAIN * action
            position = position + m.GAIN * velocity
            trajectory.append(position.clone())
        landed = (position.norm(dim=1) < m.POSITION_LIMIT) & (velocity.norm(dim=1) < m.VELOCITY_LIMIT)
        states = torch.rand(256, 4, generator=torch.Generator().manual_seed(9)) * 2 - 1
        target, prediction = m.expert_action(states), policy(states)
        metrics = m._policy_metrics(alpha)
        assert metrics["landings"] == float(landed.float().mean())
        metrics["strict_paired"] = self.quality.paired_edit_metrics(prediction.numpy(), target.numpy(), np.zeros_like(target.numpy()))
        if "beta" in local:
            metrics["beta"] = float(local["beta"].detach())
        return metrics, {"prediction": prediction, "target": target, "position": torch.stack(trajectory),
                         "final_velocity": velocity, "landed": landed}

    def landing(self, local):
        m, beta = self.module, local["beta"].detach()
        sink = float(m.sink_of(beta))
        policy = m._sink_policy(sink)
        states = m.initial_states(m.EVAL_ROWS, 1000)
        trajectory = [states.clone()]
        original_step = m.kinematic_step

        def observe_step(state, action):
            result = original_step(state, action)
            trajectory.append(result.clone())
            return result
        try:
            m.kinematic_step = observe_step
            landed, crashed, steps = m._hard_rollout(policy, states)
        finally:
            m.kinematic_step = original_step
        metrics = m.evaluate_policy(policy, states)
        metrics.update(beta=float(beta), sink=sink,
                       strict_landing=self.quality.landing_metrics(landed.numpy(), crashed.numpy(), steps.numpy(), horizon=m.HORIZON))
        assert metrics["landings"] == float(landed.float().mean())
        assert metrics["crash_rate"] == float(crashed.float().mean())
        return metrics, {"states": torch.stack(trajectory), "landed": landed, "crashed": crashed, "steps": steps}

    def native(self, arm, local):
        m = self.module
        state, previous, action, _ = m.batch(2048, torch.Generator().manual_seed(99))
        encoder = local["encoder"] if arm == "pretrain" else local["control"]
        decoder, prior = local["decoder"], local["prior"]
        context = torch.cat([state, previous], 1)
        prediction = decoder(encoder(context, prior).codes[:, 0])[:, 1:]
        live = float(torch.nn.functional.mse_loss(prediction, action))
        assert live == m.action_mse(encoder, decoder, prior)
        metrics = {"live_action_mse": live,
                   "strict_paired_live": self.quality.paired_edit_metrics(prediction.numpy(), action.numpy(), np.zeros_like(action.numpy()))}
        arrays = {"previous": previous, "target": action, "live_prediction": prediction}
        if arm == "pretrain":
            paired = decoder(encoder(torch.cat([state, action], 1), prior).codes[:, 0])[:, 1:]
            metrics["paired_reconstruction_mse"] = float(torch.nn.functional.mse_loss(paired, action))
            arrays["paired_prediction"] = paired
        else:
            ema = local["ema_g"](local["ema_c"](context, prior).codes[:, 0])[:, 1:]
            metrics["ema_action_mse"] = float(torch.nn.functional.mse_loss(ema, action))
            assert metrics["ema_action_mse"] == m.action_mse(local["ema_c"], local["ema_g"], prior)
            metrics["strict_paired_ema"] = self.quality.paired_edit_metrics(ema.numpy(), action.numpy(), np.zeros_like(action.numpy()))
            arrays["ema_prediction"] = ema
        return metrics, arrays

    def final_parity(self, arm, result, local):
        row = next(r for r in reversed(self.rows) if r["arm"] == arm)["metrics"]
        if self.family == "sign":
            for key in ("alpha", "landings", "rel_l2"):
                assert row[key] == result[key], (arm, key)
        elif self.family == "landing":
            for key in ("beta", "sink", "landings", "mean_steps", "crash_rate", "score"):
                assert row[key] == result[key], (arm, key)
        elif arm != "pretrain":
            by_step = {r["step"]: r["metrics"] for r in self.rows if r["arm"] == arm}
            for entry in local["series"]:
                assert by_step[entry["step"]]["live_action_mse"] == entry["live"]
                assert by_step[entry["step"]]["ema_action_mse"] == entry["ema"]
            assert row["live_action_mse"] == result["live"] and row["ema_action_mse"] == result["ema"]
            self.parity["native_original_series_exact"] = True
        self.parity["final_original_metrics_exact"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--family", choices=SOURCES, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve the existing cohort; select a fresh output directory")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second family wall cap exhausted")))
    signal.setitimer(signal.ITIMER_REAL, 120)
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import particlegan
    assert Path(particlegan.__file__).resolve().parent == source / "particlegan"
    import importlib
    module = importlib.import_module(SOURCES[args.family][:-3].replace("/", "."))
    assert Path(module.__file__).resolve() == source / SOURCES[args.family]
    spec = importlib.util.spec_from_file_location("source_family_definition_quality", args.quality)
    quality = importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)
    paths = [SOURCES[args.family], "lib/vendor/concept_slider_core/reference.py"]
    if args.family == "native":
        paths.remove("lib/vendor/concept_slider_core/reference.py")
    hashes = {name: sha(source / name) for name in paths}
    audit_root = Path(__file__).resolve().parents[2]
    for name, expected in hashes.items():
        content = subprocess.check_output(["git", "show", REVISION + ":" + name], cwd=audit_root)
        assert hashlib.sha256(content).hexdigest() == expected
    package = {str(p.relative_to(source)): sha(p) for p in sorted((source / "particlegan").rglob("*.py"))}
    receipt = {"format": "source_family_training_coverage_v1", "family": args.family,
               "source_revision": REVISION, "source_sha256": hashes,
               "native_package_sha256": digest(package), "quality_evaluator_sha256": sha(args.quality),
               "quality_evaluator_version": quality.VERSION, "observer_sha256": sha(__file__),
               "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "threads": 1},
               "wall_cap_seconds": 120, "sampling": "clean deterministic held-out evaluation of actual live state; native EMA separately labelled",
               "fresh_training_campaigns": 1, "seed_tuning": False,
               "status": "RUNNING", "qualification_credit": "none"}
    capture = Capture(module, args.family, args.out, quality)
    try:
        torch.set_num_threads(1)
        sys.settrace(capture.trace)
        if args.family in ("sign", "landing"):
            result = module.run_gate()
        else:
            # Preserve run_gate's original function order, seed and budgets;
            # CPU1 is an explicitly distinct runtime from its four-thread CLI.
            torch.manual_seed(0)
            log = lambda message: print(message, flush=True)
            recipe, initial, init_mse = module.pretrain(log)
            if recipe.reg_coeff <= 0:
                raise RuntimeError("Fixed arm requires the recipe critic penalty (reg_coeff > 0)")
            current = module.run_arm("current", "current", recipe, initial, log)
            fixed = module.run_arm("fixed", "fixed", recipe, initial, log)
            collapse, passed = module.gate_status(current["ema"], fixed["ema"])
            result = dict(ok=bool(collapse and passed), init=init_mse, current=current, fixed=fixed,
                          collapse=collapse, fixed_pass=passed, adversarial_weight=1., l2_weight=0.,
                          penalty_arm="recipe", penalty_coeff=recipe.reg_coeff, supervised_only=False)
            receipt["thread_cohort"] = "CPU1 wrapper calls original functions; the source CLI run_gate requests CPU4"
        sys.settrace(None)
        assert hashes == {name: sha(source / name) for name in paths}
        assert package == {str(p.relative_to(source)): sha(p) for p in sorted((source / "particlegan").rglob("*.py"))}
        receipt.update(status="COMPLETE", original_gate="PASS" if result.get("passed", result.get("ok")) else "FAIL",
                       original_result=result, observation_count=len(capture.rows), protocols=capture.protocols,
                       observation_parity=capture.parity, training_and_observation_seconds=time.monotonic() - started)
    except Exception as error:
        sys.settrace(None)
        receipt.update(status="BLOCKED" if isinstance(error, (ImportError, AttributeError)) else "ERROR",
                       error=type(error).__name__ + ": " + str(error),
                       observation_count=len(capture.rows), observation_parity=capture.parity,
                       training_and_observation_seconds=time.monotonic() - started)
        raise
    finally:
        sys.settrace(None)
        signal.setitimer(signal.ITIMER_REAL, 0)
        np.savez_compressed(args.out / "observations.npz", **capture.arrays)
        receipt["observations_sha256"] = sha(args.out / "observations.npz")
        receipt["source_unchanged"] = hashes == {name: sha(source / name) for name in paths}
        receipt["final_observed_arms"] = {arm: next(r for r in reversed(capture.rows) if r["arm"] == arm)
                                          for arm in capture.completed}
        (args.out / "receipt.json").write_text(json.dumps(json_safe(receipt), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "finished", "family": args.family, "status": receipt["status"],
                          "original_gate": receipt.get("original_gate"), "seconds": receipt["training_and_observation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
