"""Record actual training through the current public default; keep raw observations local."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from particlegan import GANTrainer, get_recipe, init
from lib.denoising_toy import GaussianGrid, grid_metrics
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator, sample_100gaussians

SEED, EVAL_SAMPLES, PLOT_SAMPLES = 0, 20_000, 4096
STREAM_SEEDS = dict(constructor=11, data=12, latent_generator=13,
                    penalty_generator=14, noise_generator=15,
                    input_noise_generator=16, eval_generator=17,
                    model_generator=18, reference=19)


def frame_steps(total, frames, power):
    """Strictly increasing observed updates, dense near the start, including the endpoint."""
    previous, steps = 0, []
    for i in range(1, frames + 1):
        previous = min(total, max(previous + 1, round(total * (i / frames) ** power)))
        steps.append(previous)
    return sorted(set(steps))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def score(samples, centers):
    """20k draws; a covered mode has >=10 samples within Euclidean radius 3 sigma."""
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("nonfinite model outputs")
    distance, nearest = torch.cdist(samples, centers).min(1)
    inside = distance <= .09
    covered = torch.bincount(nearest[inside], minlength=100)
    counts = torch.bincount(nearest, minlength=100).float()
    modes, hq = int((covered >= 10).sum()), float(inside.float().mean())
    return dict(modes=modes, hq=hq, mode_tv=float((counts / len(samples) - .01).abs().sum() / 2),
                passed=modes == 100 and hq >= .95)


def make_trainer(recipe, device):
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(STREAM_SEEDS["constructor"])
        generator = init.deterministic_orthogonal_(SimpleMLPGenerator(z_dim=recipe.z_dim), seed=0)
        critic = init.deterministic_orthogonal_(SimpleMLPDiscriminator(fourier=2), seed=1)
        prior = init.deterministic_orthogonal_(recipe.make_prior(), seed=2)
        constructor_rng = torch.get_rng_state().clone()
    streams = {name: torch.Generator(device=device).manual_seed(seed)
               for name, seed in STREAM_SEEDS.items() if name.endswith("_generator")}
    trainer = GANTrainer(recipe, generator.to(device), critic.to(device), prior=prior.to(device),
                         seed=SEED, **streams)
    trainer.readme_constructor_rng = constructor_rng
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="Explicit demo budget; default: get_recipe().total_steps.")
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--frame-power", type=float, default=2.)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--hold-seconds", type=float, default=2.5)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.device_count() > 1 else
                        ("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/100gaussians")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()
    default = get_recipe()  # Follow the public default; never select a different favorite for the demo.
    recipe = default if args.steps is None else get_recipe(total_steps=args.steps)
    total = recipe.total_steps
    if total is None or total <= 0 or args.frames <= 0 or not np.isfinite(args.frame_power) or args.frame_power <= 0:
        parser.error("a positive budget, --frames and finite --frame-power are required")
    if args.fps <= 0 or 100 % args.fps or not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        parser.error("--fps must divide 100 and --hold-seconds must be finite and nonnegative")
    if recipe.model != "gan" or recipe.conditioning != "scalar" or recipe.encoder_mode != "none":
        parser.error("the current default requires a different host; update this example explicitly")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(SEED)
    trainer = make_trainer(recipe, device)
    data = torch.Generator(device=device).manual_seed(STREAM_SEEDS["data"])
    evaluation_initial = trainer.eval_generator.get_state().clone()
    reference = torch.Generator(device=device).manual_seed(STREAM_SEEDS["reference"])
    target = sample_100gaussians(EVAL_SAMPLES, device, generator=reference)
    centers = torch.cartesian_prod(torch.arange(10, device=device), torch.arange(10, device=device)).float() - 4.5
    oracle, collapsed = score(target, centers), score(torch.zeros_like(target), centers)
    if not oracle["passed"] or collapsed["passed"]:
        raise ValueError("coverage scorer failed oracle/collapse controls")
    media = [0, *frame_steps(total, args.frames, args.frame_power)]
    scoring = sorted(set(round(total * i / min(24, total)) for i in range(1, min(24, total) + 1)))
    sources = [*sorted((ROOT / "particlegan").rglob("*.py")),
               ROOT / "lib/toy_models.py", ROOT / "lib/denoising_toy.py", ROOT / "lib/toy_metrics.py",
               Path(__file__), Path(__file__).with_name("generate.py")]
    contract = dict(
        schema=1, scope="README mode-acquisition illustration; not Forge qualification",
        seed=SEED, default_recipe=default.name, recipe=recipe.to_dict(),
        version=tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"],
        task=dict(target="uniform 10x10 Gaussian grid", sigma=.03,
                  generator="SimpleMLPGenerator(width=128, depth=3)",
                  critic="SimpleMLPDiscriminator(width=128, depth=3, fourier=2)",
                  prior="recipe.make_prior(); public deterministic initialization",
                  initialization=dict(generator=0, critic=1, prior=2),
                  sampling="live, clean, uniform prior draws; fixed first 4096 rows for plotting"),
        gate=dict(modes=100, minimum_nearby_samples=10, radius=.09, hq_min=.95,
                  terminal_checks=5, evaluation_samples=EVAL_SAMPLES, full_default_budget=default.total_steps),
        scoring_steps=scoring, media_steps=media, stream_seeds=STREAM_SEEDS,
        controls=dict(oracle=oracle, collapsed=collapsed),
        initial_sha256={name: module_digest(module) for name, module in
                        [("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior)]},
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    def checkpoint(name):
        torch.save(dict(trainer=trainer.state_dict(), data_rng=data.get_state(),
                        constructor_rng=trainer.readme_constructor_rng,
                        reference_rng=reference.get_state(), evaluation_initial=evaluation_initial,
                        contract=contract), out / name)

    checkpoint("initial.pt")
    print(json.dumps(dict(event="start", output_dir=str(out), recipe=recipe.name,
                          total_steps=total, device=str(device), controls=contract["controls"])), flush=True)
    observations, coordinates = [], []
    final_samples = None

    @torch.no_grad()
    def observe(step):
        nonlocal final_samples
        before = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS
                  if name != "eval_generator"}
        trainer.eval_generator.set_state(evaluation_initial)
        samples = trainer.sample(EVAL_SAMPLES)
        plotted = trainer.sample(min(PLOT_SAMPLES, recipe.num_particles), fixed_first_n=True)
        row = dict(step=step, **score(samples, centers), scoring=step in scoring, frame=step in media)
        for name, state in before.items():
            if not torch.equal(state, getattr(trainer, name).get_state()):
                raise RuntimeError(f"observation consumed training stream {name}")
        observations.append(row)
        coordinates.append(plotted.cpu().numpy())
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if step == total:
            final_samples = samples

    observe(0)
    wanted = set(media + scoring)
    train_seconds = 0.
    for step in range(1, total + 1):
        tick = time.perf_counter()
        stats = trainer.step(sample_100gaussians(recipe.batch_size, device, generator=data),
                             generator_real=lambda: sample_100gaussians(recipe.batch_size, device, generator=data))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - tick
        if not all(bool(torch.isfinite(stats[key])) for key in ("loss_d", "loss_g", "penalty")):
            checkpoint("nonfinite.pt")
            raise ValueError(f"nonfinite training at step {step}")
        if step in wanted:
            observe(step)
        if step % 100 == 0 or step == total:
            print(json.dumps(dict(event="progress", step=step, total_steps=total,
                                  loss_d=float(stats["loss_d"]), loss_g=float(stats["loss_g"]),
                                  train_seconds=round(train_seconds, 2), observation=observations[-1])), flush=True)
    checkpoint("final.pt")
    distribution = grid_metrics(final_samples, torch.zeros(EVAL_SAMPLES, device=device, dtype=torch.long),
                                GaussianGrid(device, .03, 1), target)
    checked = [row for row in observations if row["scoring"]]
    passed = total == default.total_steps and len(checked) >= 5 and all(row["passed"] for row in checked[-5:])
    np.savez_compressed(out / "observations.npz", samples=np.stack(coordinates),
                        steps=np.array([row["step"] for row in observations]),
                        hq=np.array([row["hq"] for row in observations]),
                        modes=np.array([row["modes"] for row in observations]),
                        mode_tv=np.array([row["mode_tv"] for row in observations]),
                        scoring=np.array([row["scoring"] for row in observations]),
                        frame=np.array([row["frame"] for row in observations]), centers=centers.cpu().numpy(),
                        target=target.cpu().numpy(), final_evaluation=final_samples.cpu().numpy())
    summary = dict(**contract, completed_steps=trainer.completed_steps, status="PASS" if passed else "FAIL",
                   final=observations[-1], terminal_checks=checked[-5:], distribution_metrics=distribution,
                   train_seconds=train_seconds, record_seconds=time.perf_counter() - started,
                   observations_sha256=digest(out / "observations.npz"),
                   final_checkpoint_sha256=digest(out / "final.pt"),
                   final_data_rng_sha256=hashlib.sha256(data.get_state().numpy().tobytes()).hexdigest(),
                   observation_training_rng_unchanged=True,
                   limitations="One fixed-seed illustration of mode acquisition; clean particle outputs; "
                                "coverage does not establish Gaussian width/covariance fidelity or a trainer comparison.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(event="record_complete", status=summary["status"], final=summary["final"],
                          train_seconds=train_seconds, output_dir=str(out))), flush=True)
    if not args.record_only:
        from render import render
        render(out, fps=args.fps, hold=args.hold_seconds)
    return 0 if passed else 1
