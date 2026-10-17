"""Bounded public-trainer DCGAN throughput measurements on reproducible mock data.

This is an execution benchmark, not a distribution-quality or promotion gate.
The timed region contains only ``GANTrainer.step`` on already resident paired
batches. Data generation, numerical checks, media and checkpoints are untimed.
Raw traces/checkpoints belong in local Forge artifacts, never in Git.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from benchmarks.toy_audit.reproducibility import reproducible_execution
from .api import CapabilityError, FormulationContext
from .contracts import atomic_json, canonical, file_hash, stable_hash
from .priors import prior_policy_receipt
from .state import state_digest


BENCHMARK = "dcgan_performance_v1"
DEFAULT_SPEC = dict(image_size=64, batch_size=16, base_channels=64,
                    max_channels=512, latent_dim=128, n_particles=256,
                    warmup_updates=3, measured_updates=10, seed=0,
                    prior=dict(kind="mog", sigma=.1, standardize=False),
                    max_seconds=600., profile_updates=0)
RESOURCE_FIELDS = frozenset({"batch_size", "num_particles", "z_dim", "total_steps"})


def validate_spec(spec):
    """Return a normalized, bounded task declaration, rejecting silent variants."""
    if not isinstance(spec, dict) or set(spec) - set(DEFAULT_SPEC):
        raise ValueError("unknown DCGAN performance specification fields")
    value = {**DEFAULT_SPEC, **spec}
    value["prior"] = dict(value["prior"])
    limits = dict(image_size=(32, 256), batch_size=(2, 256), base_channels=(1, 128),
                  max_channels=(1, 1024), latent_dim=(2, 512), n_particles=(2, 4096),
                  warmup_updates=(1, 20), measured_updates=(1, 100), seed=(0, 0),
                  profile_updates=(0, 0))
    for key, (low, high) in limits.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if value["image_size"] not in (32, 64, 128, 256):
        raise ValueError("image_size must be 32, 64, 128 or 256")
    if value["max_channels"] < value["base_channels"]:
        raise ValueError("max_channels must be at least base_channels")
    seconds = value["max_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError("max_seconds must be finite and in (0, 3600]")
    value["max_seconds"] = float(seconds)
    p = value["prior"]
    if (not {"kind", "sigma", "standardize"} <= set(p)
            or set(p) - {"kind", "sigma", "standardize", "exception_reason"}
            or p["kind"] not in ("mog", "particle_cloud")
            or type(p["standardize"]) is not bool or p["standardize"]
            or type(p["sigma"]) not in (int, float)
            or not math.isfinite(p["sigma"])):
        raise ValueError("performance tasks require an explicit unstandardized prior")
    if p["kind"] == "mog" and (p["sigma"] <= 0 or "exception_reason" in p):
        raise ValueError("MoG performance tasks require positive sigma and no cloud exception")
    if p["kind"] == "particle_cloud" and (p["sigma"] != 0 or not isinstance(p.get("exception_reason"), str)
                                           or not p["exception_reason"].strip()):
        raise ValueError("particle-cloud performance tasks require sigma zero and an explicit exception_reason")
    return value


def recipe_blockers(recipe, spec):
    """Cheap compatibility checks shared by preflight and execution."""
    spec = validate_spec(spec)
    blockers = []
    required = dict(batch_size=spec["batch_size"], num_particles=spec["n_particles"],
                    z_dim=spec["latent_dim"],
                    prior_kind="mog" if spec["prior"]["kind"] == "mog" else "particles", standardize=False)
    for field, value in required.items():
        if getattr(recipe, field) != value:
            blockers.append(f"resolved recipe {field} differs from performance task ({value})")
    if recipe.model != "gan" or recipe.conditioning != "scalar" or recipe.encoder_mode != "none":
        blockers.append("DCGAN performance requires the unconditional scalar GANTrainer API")
    if recipe.row_policy != "independent":
        blockers.append("DCGAN performance does not implement routed contexts")
    if recipe.total_steps not in (None, spec["warmup_updates"] + spec["measured_updates"]):
        blockers.append("scheduled recipe total_steps differs from the performance task; schedule-free recipes retain None")
    if spec["prior"]["kind"] == "mog" and (recipe.row_evidence_gate or recipe.particle_birth_death):
        blockers.append("recipe row controls require a separately declared particle-cloud performance cohort; MoG is unsupported")
    if recipe.constraint_geometry_mode == "strict_progress":
        blockers.append("strict_progress cannot execute standard DCGAN BatchNorm running buffers")
    return blockers


class DCGANGenerator(nn.Module):
    """DCGAN ConvTranspose/BatchNorm/ReLU stack with a Tanh RGB output.

    Resolutions above 64 add stages while retaining the declared channel cap;
    this is a DCGAN-family extension, with a standard 64px DCGAN at base 64.
    The public deterministic initializer replaces constructor weights, rather
    than using DCGAN's conventional random Normal(0, .02) initialization.
    """
    def __init__(self, image_size, latent_dim, base_channels, max_channels):
        super().__init__()
        stages = int(math.log2(image_size)) - 2
        channels = min(base_channels * 2 ** (stages - 1), max_channels)
        layers = [nn.ConvTranspose2d(latent_dim, channels, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(channels), nn.ReLU(inplace=False)]
        for stage in range(stages - 1, 0, -1):
            output = min(base_channels * 2 ** (stage - 1), max_channels)
            layers.extend([nn.ConvTranspose2d(channels, output, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(output), nn.ReLU(inplace=False)])
            channels = output
        layers.extend([nn.ConvTranspose2d(channels, 3, 4, 2, 1, bias=False), nn.Tanh()])
        self.main = nn.Sequential(*layers)

    def forward(self, latent):
        return self.main(latent[:, :, None, None])


class DCGANCritic(nn.Module):
    """DCGAN Conv/BatchNorm/LeakyReLU stack returning an unbounded scalar.

    BatchNorm includes batch-coupled input derivatives in training-mode critic
    penalties. The public trainer switches D to eval mode for the G step. This
    is a declared performance cohort, not a per-example gradient-penalty claim.
    The final 4x4 convolution is expressed as an equivalent bias-free Linear
    head on flattened 4x4 features, enabling the public policy feature contract.
    Its parameter layout is dense: optimizers that treat convolutions per
    kernel offset will update this head differently from a Conv2d module.
    """
    def __init__(self, image_size, base_channels, max_channels):
        super().__init__()
        stages = int(math.log2(image_size)) - 2
        channels = min(base_channels, max_channels)
        layers = [nn.Conv2d(3, channels, 4, 2, 1, bias=False), nn.LeakyReLU(.2, inplace=False)]
        for stage in range(1, stages):
            output = min(base_channels * 2 ** stage, max_channels)
            layers.extend([nn.Conv2d(channels, output, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(output), nn.LeakyReLU(.2, inplace=False)])
            channels = output
        layers.extend([nn.Flatten(), nn.Linear(channels * 4 * 4, 1, bias=False)])
        self.main = nn.Sequential(*layers)

    def forward(self, images):
        return self.main(images).reshape(-1, 1)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _contention(device):
    if device.type != "cuda":
        return {"available": False, "reason": "CPU benchmark; GPU contention not applicable"}
    query = _command(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    processes = _command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    gpu_rows = [] if query is None else query.splitlines()
    process_rows = [] if processes is None else processes.splitlines()
    gpu_fields = [[part.strip() for part in line.split(",")] for line in gpu_rows]
    process_fields = [[part.strip() for part in line.split(",")] for line in process_rows]
    # Some hosts provide an nvidia-smi shim that silently returns a GPU listing
    # for unsupported queries with exit code zero. That is unknown contention.
    valid_gpus = bool(gpu_fields) and all(len(parts) == 5 and parts[0].isdigit()
                                         and parts[1].startswith("GPU-") for parts in gpu_fields)
    valid_processes = all(len(parts) == 3 and parts[0].startswith("GPU-") and parts[1].isdigit()
                          for parts in process_fields)
    return {"available": query is not None and processes is not None and valid_gpus and valid_processes,
            "gpu_rows": gpu_rows, "compute_process_rows": process_rows,
            "benchmark_pid": os.getpid(), "scope": "point-in-time before/after; does not prove exclusive occupancy"}


def _gpu_uuid(value):
    """Torch exposes bare UUIDs; nvidia-smi exposes the same UUID with GPU-."""
    value = str(value or "").strip()
    return value if not value or value.startswith("GPU-") else "GPU-" + value


def _hardware(device):
    result = dict(device=str(device), type=device.type, platform=platform.platform(),
                  cpu=platform.processor() or platform.machine(), name=platform.processor() or platform.machine(),
                  physical_uuid=None, pci_bus_id=None, total_memory_mib=None,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  contention_before=_contention(device))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(name=properties.name, total_memory_mib=properties.total_memory / 2**20,
                      physical_uuid=_gpu_uuid(getattr(properties, "uuid", "")) or None)
        rows = _command(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"])
        for row in (rows or "").splitlines():
            uuid, bus = [item.strip() for item in row.split(",")]
            if result["physical_uuid"] and _gpu_uuid(uuid) == result["physical_uuid"]:
                result["pci_bus_id"] = bus or None
    return result


def _source():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(set((root / "particlegan").rglob("*.py")) | {
        Path(__file__).resolve(), root / "experiments/forge/api.py", root / "experiments/forge/rng.py",
        root / "experiments/forge/priors.py", root / "experiments/forge/state.py",
        root / "experiments/forge/contracts.py", root / "benchmarks/toy_audit/reproducibility.py"})
    snapshot = root / "forge-source.json"
    if snapshot.exists():
        import json
        commit = json.loads(snapshot.read_text())["origin_commit"]
    else:
        commit = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    source = dict(commit=commit, files_sha256={str(path.relative_to(root)): file_hash(path) for path in paths})
    return {**source, "sha256": stable_hash(source)}


def _finite_model(*models):
    # One reduction per component; no optimizer sentinels are mistaken for model NaNs.
    return all(bool(torch.stack([torch.isfinite(value).all() for value in model.state_dict().values()
                                if isinstance(value, torch.Tensor) and value.is_floating_point()]).all())
               for model in models)


def _render_frame(samples, targets, step, finite_fraction):
    from PIL import Image, ImageDraw
    def strip(values):
        pixels = values.detach().cpu().float().clamp(-1, 1).add(1).mul(127.5).byte()
        return np.concatenate([image.permute(1, 2, 0).numpy() for image in pixels], axis=1)
    upper, lower = strip(targets), strip(samples)
    pixels = np.concatenate([upper, lower], axis=0)
    image = Image.fromarray(pixels).resize((512, 256))
    panel = Image.new("RGB", (512, 310), "white")
    panel.paste(image, (0, 54))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), f"Actual training: update {step}; finite updates {finite_fraction:.0%}", fill="black")
    draw.text((8, 20), "Top: U[-1,1] RGB mock law; bottom: clean API samples", fill="black")
    draw.text((8, 36), "Performance/finite-execution diagnostic; no quality gate", fill="black")
    return panel


def _percentile(values, quantile):
    return float(np.percentile(values, quantile)) if values else None


def run_benchmark(recipe, spec: dict, output: Path, device: str = "cpu") -> dict:
    """Execute one exclusive artifact directory and return its compact receipt.

    Bounds are checked before construction. A soft wall limit is checked before
    and after every update; Forge's worker supplies the external hard timeout.
    OOM/error/incomplete runs retain measured evidence without a PASS verdict.
    """
    spec = validate_spec(spec)
    blockers = recipe_blockers(recipe, spec)
    if blockers:
        raise CapabilityError(blockers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("result.json", "updates.jsonl", "state.pt")):
        raise FileExistsError("DCGAN performance artifacts already exist; use an exclusive directory")
    return _run(recipe=recipe, spec=spec, output=output, device=str(device), seed=0)


@reproducible_execution
def _run(*, recipe, spec, output, device, seed):
    started = time.perf_counter()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise CapabilityError(["DCGAN performance supports CPU and CUDA devices"])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    runtime = dict(python=platform.python_version(), torch=str(torch.__version__),
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), dtype="torch.float32",
                   deterministic_algorithms=True, cudnn_benchmark=False, tf32=False, threads=1)
    result = dict(schema_version=1, benchmark=BENCHMARK, spec=spec, recipe=recipe.to_dict(),
                  status="INCOMPLETE", benchmark_status="NOT_STARTED", runtime=runtime,
                  hardware=_hardware(device),
                  identity=dict(source=_source(), task_sha256=stable_hash(spec),
                                recipe_sha256=stable_hash(recipe.to_dict()),
                                global_recipe_sha256=stable_hash({key: value for key, value in recipe.to_dict().items()
                                                                 if key not in RESOURCE_FIELDS}),
                                runtime_sha256=stable_hash(runtime)),
                  measurement=dict(timed_region="synchronized GANTrainer.step only; one current critic and generator batch resident on device",
                                   excluded="mock generation, hashing, finite checks, evaluation, media, checkpoints",
                                   data_law="independent RGB pixels Uniform[-1,1]; fresh critic and generator batches per update",
                                   architecture="DCGAN ConvTranspose G / Conv D, RGB, capped stage width, G Tanh; equivalent flattened4x4 Linear scalar D head without Sigmoid",
                                   head_layout="shared dense Linear head; equivalent forward class/count to final4x4Conv, but dense optimizer geometry",
                                   normalization="standard BatchNorm G and D; critic penalties have batch-coupled training-mode derivatives",
                                   sampling="clean public API samples, recipe-selected serving, untimed; no quality claim",
                                   throughput_units="batch_size per public D+G/prior update divided by mean measured update duration",
                                   percentile_note="linear empirical percentile of the declared measured updates; ten updates give a short-window estimate",
                                   phase_profile=None, phase_profile_note="unavailable in v1; end-to-end update timing is uninstrumented",
                                   wall_limit="soft check between updates; external Forge worker enforces hard reservation timeout"),
                  artifacts=dict(trace="updates.jsonl", checkpoint=None, gif=None))
    context = trainer = None
    frames, rows = [], []
    digests = {name: hashlib.sha256() for name in ("critic", "generator")}
    initial, initial_parameters = {}, {}
    peak_allocated = peak_reserved = None
    finite_model = changed_g = changed_d = False
    total = spec["warmup_updates"] + spec["measured_updates"]
    try:
        # Fail before training if the required actual-training media cannot be produced.
        from PIL import Image  # noqa: F401
        context = FormulationContext(recipe_overrides=recipe.to_dict(), prior=spec["prior"],
                                     prior_contract="recipe_owned_v1", seed=0, device=device)
        if context.recipe.to_dict() != recipe.to_dict():
            raise CapabilityError(["performance context resolution changed the supplied recipe"])
        g = context.construct(lambda: DCGANGenerator(spec["image_size"], spec["latent_dim"],
                              spec["base_channels"], spec["max_channels"]), component="generator").to(device)
        d = context.construct(lambda: DCGANCritic(spec["image_size"], spec["base_channels"],
                              spec["max_channels"]), component="discriminator").to(device)
        trainer = context.build_trainer(g, d, max_steps=total)
        initial = {name: state_digest(model.state_dict()) for name, model in
                   (("generator", g), ("discriminator", d), ("prior", trainer.prior))}
        initial_parameters = {name: state_digest(dict(model.named_parameters()))
                              for name, model in (("generator", g), ("discriminator", d))}
        result["identity"].update(initial_model_sha256=initial,
                                  initial_parameter_sha256=initial_parameters,
                                  initialization_sha256=stable_hash(context.initialization))
        shape = (spec["batch_size"], 3, spec["image_size"], spec["image_size"])
        critic_stream = context.streams.generator("data", component="critic", purpose="training")
        generator_stream = context.streams.generator("data", component="generator", purpose="training")
        target_stream = context.streams.generator("eval", component="target", purpose="performance_panel")
        sample_stream = context.streams.generator("eval", component="sampler", purpose="performance_panel")
        target = torch.rand((4, *shape[1:]), generator=target_stream, device=device).mul_(2).sub_(1).cpu()
        def observe(step):
            with torch.no_grad():
                samples = trainer.sample(4, generator=sample_stream, ema=False, output_noise=False)
            fraction = sum(row["finite_losses"] for row in rows) / len(rows) if rows else 1.
            frames.append(_render_frame(samples, target, step, fraction))
        observe(0)
        with (output / "updates.jsonl").open("x", buffering=1) as trace:
            for update in range(total):
                if time.perf_counter() - started >= spec["max_seconds"]:
                    result["benchmark_status"] = "WALL_LIMIT"
                    break
                phase = "warmup" if update < spec["warmup_updates"] else "measured"
                if update == spec["warmup_updates"] and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                real = torch.rand(shape, generator=critic_stream, device=device).mul_(2).sub_(1)
                real_g = torch.rand(shape, generator=generator_stream, device=device).mul_(2).sub_(1)
                for name, batch in (("critic", real), ("generator", real_g)):
                    digests[name].update(batch.detach().cpu().numpy().tobytes())
                _sync(device)
                tick = time.perf_counter()
                losses = trainer.step(real, generator_real=real_g, collect_stats=False)
                _sync(device)
                milliseconds = (time.perf_counter() - tick) * 1000.
                numbers = {key: float(value) for key, value in losses.items()
                           if isinstance(value, torch.Tensor) and value.numel() == 1}
                finite = all(math.isfinite(value) for value in numbers.values())
                row = dict(step=trainer.completed_steps, phase=phase, duration_ms=milliseconds,
                           finite_losses=finite,
                           losses={key: value if math.isfinite(value) else str(value) for key, value in numbers.items()})
                rows.append(row)
                line = canonical(row)
                trace.write(line + "\n")
                print(canonical(dict(benchmark=BENCHMARK, **{key: row[key] for key in
                                ("step", "phase", "duration_ms", "finite_losses")})), flush=True)
                if not finite:
                    result["benchmark_status"] = "NONFINITE"
                    break
                if trainer.completed_steps == spec["warmup_updates"]:
                    observe(trainer.completed_steps)
                if time.perf_counter() - started >= spec["max_seconds"]:
                    result["benchmark_status"] = "WALL_LIMIT"
                    break
                del real, real_g, losses
        # Read allocator peaks before numerical state checks, evaluation or checkpoint copies.
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        finite_model = _finite_model(g, d, trainer.prior)
        # BatchNorm buffers alone must not satisfy the parameter-update gate.
        final_parameter_hashes = {name: state_digest(dict(model.named_parameters()))
                                  for name, model in (("generator", g), ("discriminator", d))}
        changed_g = initial_parameters["generator"] != final_parameter_hashes["generator"]
        changed_d = initial_parameters["discriminator"] != final_parameter_hashes["discriminator"]
        result["identity"]["final_parameter_sha256"] = final_parameter_hashes
        if not frames or trainer.completed_steps != spec["warmup_updates"]:
            observe(trainer.completed_steps)
        frames[0].save(output / "training.gif", save_all=True, append_images=frames[1:],
                       duration=700, loop=0, optimize=False)
        result["artifacts"]["gif"] = "training.gif"
        result["api"] = context.receipt()
        result["prior_policy"] = prior_policy_receipt(recipe, trainer.prior)
        result["identity"]["batch_sequence_sha256"] = {name: digest.hexdigest() for name, digest in digests.items()}
        checkpoint = dict(schema_version=1, benchmark=BENCHMARK, spec=spec, context=context.state_dict(),
                          data_cursor=dict(completed_updates=trainer.completed_steps, paired_batches_per_update=2,
                                           batch_sequence_sha256=result["identity"]["batch_sequence_sha256"]))
        torch.save(checkpoint, output / "state.pt")
        result["artifacts"]["checkpoint"] = "state.pt"
        if result["benchmark_status"] == "NOT_STARTED":
            result["benchmark_status"] = "COMPLETED"
    except Exception as error:
        oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        result["benchmark_status"] = "OOM" if oom else ("UNSUPPORTED" if isinstance(error, CapabilityError) else "ERROR")
        result["error"] = dict(type=type(error).__name__, message=str(error)[:2000])
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        # A failed/OOM update is not a resumable state. Preserve consumed streams
        # separately so failures still have their actual random-stream identity.
        if context is not None:
            torch.save(dict(schema_version=1, benchmark=BENCHMARK, completed_updates=len(rows),
                            streams=context.streams.state_dict()), output / "failed-streams.pt")
            result["artifacts"]["failed_streams"] = "failed-streams.pt"
        if frames:
            frames[0].save(output / "training.gif", save_all=True, append_images=frames[1:],
                           duration=700, loop=0, optimize=False)
            result["artifacts"]["gif"] = "training.gif"
    warmup = [row["duration_ms"] for row in rows if row["phase"] == "warmup"]
    measured = [row["duration_ms"] for row in rows if row["phase"] == "measured"]
    fraction = sum(row["finite_losses"] for row in rows) / len(rows) if rows else 0.
    full = len(rows) == total and len(measured) == spec["measured_updates"]
    passed = full and fraction == 1. and finite_model and changed_g and changed_d and result["benchmark_status"] == "COMPLETED"
    result["validity"] = dict(finite_update_fraction=fraction, finite_model_state=finite_model,
                             changed_generator=changed_g, changed_discriminator=changed_d,
                             full_budget=full, passed=passed,
                             scope="finite update execution only; no distribution-quality or promotion claim")
    result["status"] = "PASS" if passed else ("INCOMPLETE" if result["benchmark_status"] == "WALL_LIMIT" else "FAIL")
    result["metrics"] = dict(finite_update_fraction=fraction, completed_updates=len(rows), measured_updates=len(measured),
                             update_ms_median=_percentile(measured, 50), update_ms_p95=_percentile(measured, 95),
                             images_per_second=(spec["batch_size"] * len(measured) * 1000 / sum(measured)) if measured else None,
                             warmup_ms_median=_percentile(warmup, 50), first_update_ms=rows[0]["duration_ms"] if rows else None,
                             peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved,
                             parameter_count_generator=sum(p.numel() for p in trainer.G.parameters()) if trainer else None,
                             parameter_count_discriminator=sum(p.numel() for p in trainer.D.parameters()) if trainer else None,
                             total_wall_seconds=time.perf_counter() - started)
    result["hardware"]["contention_after"] = _contention(device)
    snapshots = [result["hardware"][name] for name in ("contention_before", "contention_after")]
    uuid = result["hardware"]["physical_uuid"]
    if device.type == "cuda" and uuid and all(snapshot["available"] for snapshot in snapshots):
        other = [line for snapshot in snapshots for line in snapshot["compute_process_rows"]
                 if line.split(",")[0].strip() == uuid and line.split(",")[1].strip() != str(os.getpid())]
        result["hardware"]["contention_status"] = "contended" if other else "clear"
    else:
        result["hardware"]["contention_status"] = "unknown"
    result["artifact_sha256"] = {key: file_hash(output / name) for key, name in result["artifacts"].items()
                                 if name is not None and (output / name).is_file()}
    atomic_json(output / "result.json", result)
    print(canonical(dict(benchmark=BENCHMARK, status=result["status"], benchmark_status=result["benchmark_status"],
                         metrics=result["metrics"], output=str(output))), flush=True)
    return result
