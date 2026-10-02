"""One capped CUDA observation of an unchanged shipped conditional source.

The source trainer, initializer, config, schedule and samplers are executed as
written. Observations preserve every visible CUDA RNG and the source's named
streams. All traces/checkpoints belong in --output outside Git. CPU failures
remain separate evidence; this module never repairs a source API or recipe.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from . import source_conditional_scoring as scoring

SOURCE_SHA = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
VERSION = "source-cuda-observation-v1"
WALL_CAP = 120
PREFIX_UPDATES = 3
ENTRIES = {
    "source-family-02": ("denoising", "configs/denoising/diagnostics/ddgan_class_free_28k_s24002.yaml", 28000),
    "source-family-03": ("denoising", "configs/denoising/default.toml", 7000),
    "source-family-04": ("trajectory", "configs/trajectory/default.yaml", 10000),
    "source-family-05": ("trajectory", "configs/trajectory/diversity/confirm_10k/mlp_continuous.yaml", 10000),
}
LABELS = {
    "source-family-02": "Analytic denoising grid, one class",
    "source-family-03": "Analytic denoising grid, four classes",
    "source-family-04": "Conditional two-route trajectories, discrete geometry",
    "source-family-05": "Conditional two-route trajectories, continuous geometry",
}
OBSERVATION_STEPS = (0, 1, 10, 25, 50, 100)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(value):
    result = hashlib.sha256()

    def add(part):
        if isinstance(part, torch.Tensor):
            tensor = part.detach().contiguous().cpu()
            result.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(part, np.ndarray):
            result.update(str((str(part.dtype), part.shape)).encode())
            result.update(part.tobytes())
        elif isinstance(part, dict):
            for key in sorted(part, key=str):
                add(str(key)); add(part[key])
        elif isinstance(part, (tuple, list)):
            for item in part:
                add(item)
        else:
            result.update(repr(part).encode())
        result.update(b"\0")
    add(value)
    return result.hexdigest()


def global_rng():
    # Avoid initializing CUDA from a CPU-only software test. Actual source
    # construction initializes the sole visible GPU before an observation.
    return dict(cpu=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_global_rng(state):
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated(modules=()):
    """Preserve all visible CUDA/CPU RNGs and each exact module mode."""
    rng = global_rng()
    modes = [(child, child.training) for module in modules for child in module.modules()]
    try:
        # Do not impose eval mode: original endpoint samplers use no_grad with
        # the source module modes. Neither held model has running BN/dropout.
        with torch.no_grad():
            yield
    finally:
        restore_global_rng(rng)
        for child, mode in modes:
            child.training = mode


def owners(local):
    """Read complete modules, optimizers, gradients and nested named streams."""
    states, seen = {}, set()

    def walk(name, value):
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer, torch.Generator, torch.Tensor)):
            if id(value) in seen:
                return
            seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            states[name] = dict(state=value.state_dict(),
                                modes={k: m.training for k, m in value.named_modules()},
                                gradients={k: (p.requires_grad, p.grad) for k, p in value.named_parameters()})
        elif isinstance(value, torch.optim.Optimizer):
            states[name] = value.state_dict()
        elif isinstance(value, torch.Generator):
            states[name] = dict(device=str(value.device), state=value.get_state())
        elif isinstance(value, torch.Tensor):
            states[name] = (value, value.requires_grad, value.grad if value.is_leaf else None)
        elif isinstance(value, dict):
            for key, part in value.items():
                walk(f"{name}/{key}", part)
        elif isinstance(value, (tuple, list)):
            for key, part in enumerate(value):
                walk(f"{name}/{key}", part)
    for name, value in local.items():
        walk(name, value)
    if "toy" in local and hasattr(local["toy"], "__dict__"):
        walk("toy", local["toy"].__dict__)
    return dict(owners=states, global_rng=global_rng(), threads=torch.get_num_threads())


def pure_read(local, callback):
    before = digest(owners(local))
    modules = [v for v in local.values() if isinstance(v, torch.nn.Module)]
    with isolated(modules):
        result = callback()
    assert digest(owners(local)) == before, "observer changed a model, optimizer, gradient or RNG owner"
    return result, before


def streams(device, count, seed):
    return [torch.Generator(device=device).manual_seed(seed + k) for k in range(count)]


def denoising_measure(module, local):
    cfg, toy, schedule = local["cfg"], local["toy"], local["schedule"]
    device = toy.means.device
    requested = torch.arange(cfg["eval_samples"], device=device) % cfg["classes"]
    reference = toy.sample(requested, streams(device, 1, 99003)[0])
    arrays = dict(requested=requested, reference=reference, means=toy.means)
    values = {}
    # The additional frozen posterior gate is deliberately the same narrow
    # class0/observation0/abar .5 law used by the existing analytic controls.
    count, step = scoring.COUNT, 2
    cls = torch.zeros(count, dtype=torch.long, device=device)
    xt = torch.zeros(count, 2, device=device)
    t = torch.full_like(cls, step)
    cpu_toy = module.GaussianGrid("cpu", cfg["std"], cfg["classes"])
    cpu_cls = torch.zeros(count, dtype=torch.long)
    arrays["posterior_reference"] = cpu_toy.oracle_clean(torch.zeros(count, 2), cpu_cls, float(schedule.ab[step]),
                                                          torch.Generator().manual_seed(scoring.EVAL_SEED + 1))
    for label, prefix in (("live", ""), ("ema", "ema_")):
        g, prior, noise = (local[prefix + key] for key in ("g", "prior", "noise"))
        sample = module.generate(g, prior, noise, schedule, requested, *streams(device, 3, 99000))
        if not bool(torch.isfinite(sample).all()):
            raise FloatingPointError("nonfinite source denoising samples")
        original = module.grid_metrics(sample, requested, toy, reference)
        probe, _ = module.conditional_probe(g, prior, noise, schedule, toy, cfg["probe_samples"])
        rng = streams(device, 1, scoring.EVAL_SEED)[0]
        clean = g(prior.sample(count, rng)[0], cls, xt, t)
        if not bool(torch.isfinite(clean).all()):
            raise FloatingPointError("nonfinite source clean-posterior samples")
        posterior = scoring.posterior_metrics(cpu_toy, clean.cpu(), torch.zeros(2), 0, float(schedule.ab[step]))
        values[label] = dict(original_marginal=original, original_reverse_probe=probe,
                             added_clean_posterior=posterior, passed=posterior["passed"])
        arrays[label + "_sample"], arrays[label + "_posterior"] = sample, clean
    values["panel"] = dict(marginal_count=cfg["eval_samples"], marginal_seeds=[99000, 99001, 99002, 99003],
                           source_probe_samples=cfg["probe_samples"], source_probe_seed=8128,
                           source_probe_law="All original four observations, diffusion times and requested classes; Gaussian reverse transitions",
                           added_clean_posterior=dict(class_id=0, observation=[0., 0.], diffusion_step=step,
                                                     alpha_bar=float(schedule.ab[step]), count=count, seed=scoring.EVAL_SEED),
                           added_scope_limit="One fixed clean posterior; not a full class/time/observation posterior gate.")
    return arrays, values


def trajectory_measure(module, local):
    cfg, toy, schedule = local["cfg"], local["toy"], local["schedule"]
    device = local["device"]
    arrays, values = {}, {"live": {}, "ema": {}}
    for split in ("train", "test"):
        cc, gg = toy.contexts(split)
        group = torch.arange(len(cc), device=device).repeat_interleave(cfg["eval_per_context"])
        c, geom = cc[group], gg[group]
        context = toy.condition(geom)
        # This is the exact original endpoint draw order/chunk size, with a
        # private panel per cohort. It includes all reverse Gaussian noise.
        reference = toy.sample(c, geom, streams(device, 1, 99003)[0])[0]
        reference2 = toy.sample(c, geom, streams(device, 1, 99004)[0])[0]
        values.setdefault("reference_floor", {})[split] = module.metrics(toy, reference2, reference, c, geom, group)
        arrays.update({split + "_c": c, split + "_geom": geom, split + "_group": group,
                       split + "_reference": reference})
        for label, prefix in (("live", ""), ("ema", "ema_")):
            g, prior, noise = (local[prefix + key] for key in ("g", "prior", "noise"))
            rngs = streams(device, 3, 99000)
            chunks = [module.generate(g, prior, noise, schedule, c[j:j + 256], context[j:j + 256], rngs)
                      for j in range(0, len(c), 256)]
            prediction = torch.cat(chunks)
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("nonfinite source trajectories")
            values[label][split] = module.metrics(toy, prediction, reference, c, geom, group)
            arrays[label + "_" + split + "_sample"] = prediction
    values["panel"] = dict(splits=["train", "test"], contexts_per_split=len(cc),
                           rows_per_context=cfg["eval_per_context"], chunk_size=256,
                           source_seeds=[99000, 99001, 99002, 99003],
                           law="Original DDGAN sampler: fresh prior particle and configured reverse noise at each reverse step",
                           aggregate_acceptance_gate="NO_FROZEN_GATE")
    return arrays, values


def effective(local):
    prior = local["prior"]
    return dict(recipe=local["recipe"].to_dict(),
                actual_prior=dict(class_name=type(prior).__qualname__, kind=prior.kind,
                                  parameter_shapes={k: list(v.shape) for k, v in prior.named_parameters()},
                                  buffers={k: list(v.shape) for k, v in prior.named_buffers()}),
                optimizers={key: dict(class_name=type(local[key]).__qualname__,
                       groups=[{k: safe(v) for k, v in group.items() if k != "params" and isinstance(v, (float, int, str, bool, tuple, list, type(None)))}
                               for group in local[key].param_groups]) for key in ("opt_g", "opt_d")},
                actual_model_parameters={k: sum(p.numel() for p in local[k].parameters()) for k in ("g", "d")},
                named_training_streams={k: str(v.device) for k, v in local["rngs"].items()}
                    if isinstance(local["rngs"], dict) else [str(v.device) for v in local["rngs"]])


class Capture:
    def __init__(self, module, family, output):
        self.module, self.family, self.output = module, family, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows, self.seconds = [], 0.

    def observe(self, step, local):
        started = time.monotonic()
        callback = denoising_measure if self.family == "denoising" else trajectory_measure
        (arrays, metrics), before = pure_read(local, lambda: callback(self.module, local))
        index = len(self.rows)
        cloud, checkpoint = self.output / f"cloud-{index:03d}.npz", self.output / f"state-{index:03d}.pt"
        np.savez_compressed(cloud, **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
        torch.save(owners(local), checkpoint)
        row = dict(step=step, metrics=safe(metrics), owner_state_sha256=before,
                   complete_owner_and_all_visible_cuda_rng_pure=True,
                   cloud_sha256=sha(cloud), state_sha256=sha(checkpoint))
        with (self.output / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.rows.append(row)
        if index == 0:
            write(self.output / "effective.json", effective(local))
        self.seconds += time.monotonic() - started
        print(json.dumps(dict(event="CUDA_SOURCE_OBSERVATION", family=self.family, completed_updates=step,
                              scored_states=len(self.rows))), flush=True)


class PrefixComplete(BaseException):
    pass


class EntryTimeout(BaseException):
    pass


@contextmanager
def working_directory(directory):
    old = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old)


def execute(module, cfg, family, directory, *, observe, prefix, expected_initial_hash=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    capture = Capture(module, family, directory / "raw") if observe else None
    tree = ast.parse(Path(module.__file__).read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    loop = next(node.lineno for node in ast.walk(train) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "step")
    completed, boundaries, result, error, error_type = 0, {}, None, None, None

    def trace(frame, event, arg):
        return local_trace if event == "call" and frame.f_code is module.train.__code__ else None

    def local_trace(frame, event, arg):
        nonlocal completed
        if event == "line" and frame.f_lineno == loop:
            local, completed = frame.f_locals, frame.f_locals.get("step", 0)
            if expected_initial_hash is not None and completed == 0:
                assert digest(owners(local)) == expected_initial_hash, "full source initialization differs from baseline prefix"
            if prefix and completed not in boundaries:
                # One full-strength observer read tests purity cheaply; owner
                # hashes at all four boundaries still verify update parity.
                if capture and completed == 1:
                    capture.observe(completed, local)
                boundaries[completed] = digest(owners(local))
            write(directory / "progress.json", dict(completed_updates=completed,
                  last_scored_step=capture.rows[-1]["step"] if capture and capture.rows else None,
                  observations=len(capture.rows) if capture else 0))
            if prefix and completed >= PREFIX_UPDATES:
                raise PrefixComplete()
            if not prefix and capture and (completed in OBSERVATION_STEPS or completed % 250 == 0 or completed == cfg["steps"]):
                if not capture.rows or capture.rows[-1]["step"] != completed:
                    capture.observe(completed, local)
        return local_trace

    started, previous = time.monotonic(), sys.gettrace()
    try:
        with working_directory(directory):
            sys.settrace(trace)
            result = module.train(copy.deepcopy(cfg))
        status = "COMPLETE" if completed == cfg["steps"] else "INCOMPLETE"
    except PrefixComplete:
        status = "PREFIX_COMPLETE"
    except EntryTimeout:
        status, error, error_type = "INCOMPLETE", traceback.format_exc(), "EntryTimeout"
    except FloatingPointError:
        status, error, error_type = "ERROR", traceback.format_exc(), "FloatingPointError"
    except Exception as exc:
        status, error, error_type = "BLOCKED", traceback.format_exc(), type(exc).__name__
    finally:
        sys.settrace(previous)
    if result is not None:
        write(directory / "original-summary.json", result)
    if error:
        print(error, flush=True)
    receipt = dict(status=status, completed_updates=completed, original_budget=cfg["steps"],
                   boundaries=boundaries, observed_states=len(capture.rows) if capture else 0,
                   observer_seconds=capture.seconds if capture else 0.,
                   elapsed_seconds=time.monotonic() - started, error=error, error_type=error_type,
                   complete_original_endpoint=bool(status == "COMPLETE" and result is not None),
                   original_summary_sha256=sha(directory / "original-summary.json") if result is not None else None)
    write(directory / "receipt.json", receipt)
    return receipt


def manifest(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".toml", ".yaml")}


def imported_source_bindings(source, expected, *, modules=None):
    """Prove source imports resolved to the held export, not this worktree."""
    result = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        if name.split(".")[0] not in ("particlegan", "experiments", "lib"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        assert path.is_relative_to(source), f"scientific import escaped held source: {name}: {path}"
        rel = str(path.relative_to(source))
        assert rel in expected and sha(path) == expected[rel], f"scientific imported bytes changed: {name}"
        result[name] = dict(path=rel, sha256=expected[rel])
    return result


def worker(args):
    output = args.output.resolve()
    source = output / "source"
    receipt = dict(catalog_id=args.entry, version=VERSION, full_campaign_count=0,
                   source_revision=SOURCE_SHA, source_root=str(source), source_attempts=1,
                   qualification_credit="none", scientific_retries=0, configuration_repairs=0,
                   wall_cap_seconds=WALL_CAP, error=None, original_scientific_status="NO_FROZEN_GATE")
    started = time.monotonic()
    original_sources = manifest(source)
    # Deadline was frozen by the supervisor before Python/CUDA startup. Stop
    # updates before it, leaving three seconds for compact receipt cleanup.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(EntryTimeout("120-second CUDA entry wall cap exhausted")))
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.monotonic() - 3))
    try:
        assert torch.cuda.is_available(), "CUDA prerequisite remains unavailable"
        assert torch.cuda.device_count() == 1, "exactly one GPU must be visible"
        torch.set_num_threads(1)
        sys.path.insert(0, str(source))
        family, config, budget = ENTRIES[args.entry]
        script = source / f"experiments/train_{family}.py"
        spec = importlib.util.spec_from_file_location("_held_cuda_source", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = {**module.DEFAULTS, **module.read_config(str(source / config))}
        assert cfg["steps"] == budget and cfg["seed"] == 24002
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.reset_peak_memory_stats(0)
        protocol = dict(catalog_id=args.entry, family=family, label=LABELS[args.entry],
            source_revision=SOURCE_SHA, source_manifest_sha256=digest(original_sources),
            script=str(script.relative_to(source)), script_sha256=sha(script), config=config,
            config_sha256=sha(source / config), resolved_config=cfg,
            resolved_recipe=module.training_recipe(cfg).to_dict(), original_update_budget=budget,
            original_cli=[sys.executable, str(script), "--config", str(source / config)],
            imported_scientific_sources=imported_source_bindings(source, original_sources),
            exact_train_function_executed=True, config_mutations=[], output_policy="Unchanged relative out_dir, phase-specific working directory outside Git",
            runtime=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         device="cuda:0", visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         visible_gpu_count=1, gpu_name=properties.name, total_memory_bytes=properties.total_memory,
                         cpu_threads=1), wall_cap_seconds=WALL_CAP,
            observation_steps=dict(initial=list(OBSERVATION_STEPS), thereafter_every=250, final=budget),
            prefix=dict(updates_per_control=PREFIX_UPDATES, observed_read_step=1, compared_boundaries=[0, 1, 2, 3], full_budget_not_shortened=True),
            original_gate="NO_FROZEN_GATE; original endpoint diagnostics preserved",
            added_gate=dict(version=scoring.VERSION, thresholds=scoring.GATES,
                            scope="class0/xt0/t2 clean posterior only") if family == "denoising" else "NO_FROZEN_GATE; source train/test metrics observed without invented binary acceptance",
            gate_policy="Held-out values never enter optimization, updates, source guards or budget selection; full endpoint and five terminal passing observations required for added denoising PASS",
            selection="One original capped acquisition, no seed/profile/optimizer change, retries or extension")
        write(output / "protocol.json", protocol)
        receipt.update(protocol_sha256=sha(output / "protocol.json"), family=family, runtime=protocol["runtime"])
        entry_rng = global_rng()
        print(json.dumps(dict(event="CUDA_SOURCE_PREFIX_BASELINE", entry=args.entry, steps=budget, batch=cfg["batch_size"])), flush=True)
        baseline = execute(module, cfg, family, output / "prefix-baseline", observe=False, prefix=True)
        receipt["prefix_baseline"] = baseline
        if baseline["status"] != "PREFIX_COMPLETE":
            receipt.update(fresh_execution_status=baseline["status"], blocked_phase="original-source baseline prerequisite", error=baseline["error"])
        else:
            restore_global_rng(entry_rng)
            print(json.dumps(dict(event="CUDA_SOURCE_PREFIX_OBSERVED", entry=args.entry)), flush=True)
            observed = execute(module, cfg, family, output / "prefix-observed", observe=True, prefix=True)
            receipt["prefix_observed"] = observed
            same = observed["status"] == "PREFIX_COMPLETE" and baseline["boundaries"] == observed["boundaries"]
            equal = [step for step, value in baseline["boundaries"].items() if observed["boundaries"].get(step) == value]
            receipt["prefix_parity"] = dict(passed=same, compared_boundaries=[0, 1, 2, 3],
                                           exact_matches=len(equal), matched_boundaries=equal,
                                           software_updates=baseline["completed_updates"] + observed["completed_updates"],
                                           requested_software_updates=2 * PREFIX_UPDATES,
                                           all_visible_cuda_global_and_named_rngs_included=True)
            if not same:
                receipt.update(fresh_execution_status="BLOCKED", blocked_phase="observer parity prerequisite",
                               error=observed["error"] or "Exact source prefix owner hashes differ")
            else:
                restore_global_rng(entry_rng)
                print(json.dumps(dict(event="CUDA_SOURCE_FULL_ORIGINAL_BUDGET", entry=args.entry)), flush=True)
                receipt["full_campaign_count"] = 1
                receipt["training"] = execute(module, cfg, family, output / "training", observe=True, prefix=False,
                                                expected_initial_hash=baseline["boundaries"][0])
                receipt.update(fresh_execution_status=receipt["training"]["status"], error=receipt["training"]["error"])
        receipt["allocator_peak_bytes"] = dict(allocated=torch.cuda.max_memory_allocated(0), reserved=torch.cuda.max_memory_reserved(0))
    except EntryTimeout:
        receipt.update(fresh_execution_status="INCOMPLETE", error=traceback.format_exc())
    except Exception:
        receipt.update(fresh_execution_status="BLOCKED", error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    receipt["source_unchanged"] = original_sources == manifest(source)
    try:
        receipt["imported_scientific_sources"] = imported_source_bindings(source, original_sources)
    except Exception:
        receipt["import_resolution_error"] = traceback.format_exc()
        receipt.update(fresh_execution_status="BLOCKED", error=receipt["error"] or receipt["import_resolution_error"])
    receipt["worker_wall_seconds"] = time.monotonic() - started
    write(output / "execution-summary.json", receipt)
    print(json.dumps(dict(event="CUDA_SOURCE_FINISHED", entry=args.entry,
                          status=receipt["fresh_execution_status"], full_campaigns=receipt["full_campaign_count"])), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=ENTRIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    data = subprocess.check_output(["git", "archive", SOURCE_SHA, "particlegan", "lib", "experiments", "configs"], cwd=repo)
    archive = args.output / "source.tar"
    archive.write_bytes(data)
    snapshot = args.output / "source"
    snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data)) as contents:
        contents.extractall(snapshot, filter="data")
    own_sources = {str(p.relative_to(repo)): sha(p) for p in (Path(__file__), Path(scoring.__file__))}
    write(args.output / "source-receipt.json", dict(source_revision=SOURCE_SHA, archive_sha256=sha(archive),
          source_sha256=manifest(snapshot), observer_source_sha256=own_sources))
    # Preserve root's single-GPU selection. Never unhide a second device.
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/toy-source-cuda-mpl")
    launched = time.monotonic()
    command = [sys.executable, "-m", "benchmarks.toy_audit.source_cuda_observation", "--worker", "--entry", args.entry,
               "--output", str(args.output), "--deadline", str(launched + WALL_CAP)]
    killed = False
    with (args.output / "execution.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
        next_progress = 0.
        while process.poll() is None:
            elapsed = time.monotonic() - launched
            if elapsed >= WALL_CAP:
                process.kill(); process.wait(); killed = True
                write(args.output / "hard-timeout.json", dict(wall_cap_seconds=WALL_CAP, killed_child_pid=process.pid))
                break
            if elapsed >= next_progress:
                path = args.output / "training/progress.json"
                progress = json.loads(path.read_text()) if path.exists() else {}
                print(json.dumps(dict(event="CUDA_SOURCE_PROGRESS", entry=args.entry, elapsed_seconds=elapsed, **progress)), flush=True)
                next_progress = elapsed + 10
            time.sleep(.1)
    write(args.output / "launcher.json", dict(child_pid=process.pid, supervisor_pid=os.getpid(), command=command,
          hard_wall_cap_seconds=WALL_CAP, child_ownership="Only this Popen child PID is signaled", hard_killed=killed,
          elapsed_seconds=time.monotonic() - launched, returncode=process.returncode))
    return 0 if (args.output / "execution-summary.json").exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
