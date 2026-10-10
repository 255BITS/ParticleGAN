#!/usr/bin/env python3
"""Bounded public word software profiling; never produces Forge qualification."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import subprocess
import time

import torch

from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge.api import task_formulation_context


SOURCES = ("particlegan/recipes.py", "particlegan/recipe_compat.py",
           "particlegan/optim/dualnorm.py", "experiments/forge/boundaries.py")


def training_gif(initial, final, path, completed):
    """Actual observed probabilities, including original targets/reconstructions."""
    import numpy as np
    from PIL import Image, ImageDraw
    frames = []
    for observation, step in ((initial, 0), (final, completed)):
        canvas = Image.new("RGB", (420, 360), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"Software diagnostic: actual word host, update {step}", fill="black")
        panels = (observation["views"][1]["target"],
                  observation["views"][1]["samples"],
                  observation["views"][3]["samples"][:5])
        for row, (label, batch) in enumerate(zip(("Target", "Reconstruction", "Generated"), panels)):
            draw.text((8, 34 + row * 106), label, fill="black")
            for column, word in enumerate(batch):
                value = word.squeeze().cpu().numpy()
                image = Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255)).convert("RGB")
                image = image.resize((54, 84), Image.Resampling.NEAREST)
                canvas.paste(image, (110 + column * 60, 30 + row * 106))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1400, loop=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("native", "cpu"), default="native")
    parser.add_argument("--diagnostic-driver", choices=("gesvd",))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.updates <= 1024 or (args.diagnostic_driver and args.backend != "native"):
        parser.error("software timing requires 1..1024 updates; diagnostic driver uses native only")
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    candidate = json.loads((root / "configs/forge/ideas/bcap-develop-integration-combined-v1.json").read_text())
    candidate["recipe_overrides"]["optimizer_svd_backend"] = args.backend
    task = json.loads((root / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=args.device, root=root)
    fixture = WordFixture(device=args.device, seed=0, recipe_name=None, max_steps=20001, components=context)
    initial = fixture.observe()
    if args.diagnostic_driver:
        original_svd = torch.linalg.svd
        def selected_svd(value, *positional, **kwargs):
            if value.is_cuda:
                kwargs["driver"] = args.diagnostic_driver
            return original_svd(value, *positional, **kwargs)
        torch.linalg.svd = selected_svd
    def synchronize():
        if fixture.device.type == "cuda":
            torch.cuda.synchronize(fixture.device)
    for _ in range(16):
        fixture.step()
    synchronize()
    start = time.monotonic()
    for update in range(args.updates):
        fixture.step()
        if (update + 1) % 50 == 0:
            print(json.dumps({"event": "timing_progress", "updates": update + 1}), flush=True)
    synchronize()
    seconds = time.monotonic() - start
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(32):
        fixture.step()
    synchronize()
    profile.disable()
    summary = io.StringIO()
    pstats.Stats(profile, stream=summary).sort_stats("cumtime").print_stats(35)
    print(summary.getvalue(), flush=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if fixture.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        for _ in range(8):
            fixture.step()
        synchronize()
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
    profiler.export_chrome_trace(str(args.output / "profiler-trace.json"))
    torch.save({"fixture": fixture.state_dict(), "streams": context.streams.state_dict()}, args.output / "state.pt")
    observation = fixture.observe()
    training_gif(initial, observation, args.output / "actual-training.gif", fixture.completed_steps)
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    result = dict(scope="software_diagnostic_only", qualification_input=False,
        backend=args.backend, diagnostic_driver=args.diagnostic_driver,
        numerical_trainer_delta=args.backend == "cpu" and fixture.device.type != "cpu",
        original_execution_cap=20001, original_schedule_horizon=fixture.recipe.total_steps,
        warmup=16, timed_updates=args.updates, cprofile_updates=32, kineto_updates=8,
        completed_steps=fixture.completed_steps, timing_seconds=seconds,
        milliseconds_per_update=seconds * 1000 / args.updates,
        projected_20001_training_seconds=seconds * 20001 / args.updates,
        projection=fixture.opt_g.direction_blend_stats, transport=fixture.transport.state_dict(),
        observation_metrics=observation["metrics"], recipe=fixture.recipe.to_dict(),
        seed=0, gpu=torch.cuda.get_device_name(fixture.device) if fixture.device.type == "cuda" else None,
        torch_version=torch.__version__, sources=source_hashes,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        tree_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "--", *SOURCES], cwd=root)).hexdigest(),
        limitation="Training-only extrapolation excludes ordinary evaluation, checkpoint time and contention; no full-budget claim.")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "software_profile_complete", **result}), flush=True)


if __name__ == "__main__":
    main()
