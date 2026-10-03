"""Fresh Atlas native-100 replay with isolated clean/noisy observations.

The affine host, 20k-row square prior, critic and budgets follow PR223's
native fixture. Sampling laws and current-source hashes are explicit; this
run does not overwrite historical qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from benchmarks.toy100.metrics import evaluate_samples, passes
from benchmarks.toy100.accuracy import fidelity_metrics, passes_accuracy
from benchmarks.toy100.problems import sample_real, evaluation_geometry
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GANTrainer, ParticlePrior, Recipe, init
from .capture import write


def run(problem, output, *, device="cuda:0", steps=7000, shift=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    root = Path(__file__).resolve().parents[2]
    options = json.loads((root / "configs/100gaussians/atlas.json").read_text())
    recipe = Recipe(**{**options, "num_particles": 20000, "z_dim": 2, "batch_size": 2048})
    with torch.device(device):
        prior = ParticlePrior(20000, 2)
        with torch.no_grad():
            prior.z.uniform_(-5, 5)
        generator = nn.Linear(2, 2)
        with torch.no_grad():
            generator.weight.copy_(torch.eye(2))
            generator.bias.zero_()
        critic = SimpleMLPDiscriminator(2, 128, 3, 3)
        for layer in critic.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    init.deterministic_orthogonal_(critic, seed=1)
    trainer = GANTrainer(recipe, generator, critic, prior=prior, seed=1234,
                         max_steps=steps, serial_backward=True,
                         optimizer_options={"foreach": False, "fused": False})
    stream = torch.Generator(device=device).manual_seed(1234)
    started = time.perf_counter()
    base, _ = evaluation_geometry(problem, device=device)
    rows, frames, clean_frames, frame_steps, angles, gate_rows = [], [], [], [], [], []
    def angle(step):
        return math.radians(30) * ((max(1, step)-1)//500) if shift else 0.
    def rotation(theta):
        c, s = math.cos(theta), math.sin(theta)
        return torch.tensor([[c, -s], [s, c]], device=device)
    def real(step):
        points = sample_real(problem, recipe.batch_size, device=device, generator=stream)
        return points @ rotation(angle(step)).T
    @torch.no_grad()
    def draw(n, latent_seed, noise_seed):
        clean = trainer.sample(n, generator=torch.Generator(device=device).manual_seed(latent_seed), output_noise=False)
        noisy = clean + trainer.output_sigma() * torch.randn(clean.shape, device=device,
                    generator=torch.Generator(device=device).manual_seed(noise_seed))
        return clean, noisy
    def capture(step, *, event="observation", theta=None):
        theta = angle(step) if theta is None else theta
        clean, noisy = draw(4096, 77, 78)
        centers = base @ rotation(theta).T
        distance, nearest = torch.cdist(noisy, centers).min(1)
        hq = distance <= .09
        row = dict(step=step, event=event, target_degrees=math.degrees(theta),
                   hq=float(hq.float().mean()), modes=int((torch.bincount(nearest[hq], minlength=100)>=10).sum()),
                   sigma=float(trainer.output_sigma()), seconds=time.perf_counter()-started)
        rows.append(row)
        frames.append(noisy.cpu().numpy())
        clean_frames.append(clean.cpu().numpy())
        frame_steps.append(step)
        angles.append(theta)
        print(json.dumps({**row, "event": "NATIVE_OBSERVATION", "observation_kind": event, "task": problem}), flush=True)
        if step and (step % 250 == 0 or shift and step % 500 == 0) and event == "observation":
            clean20, noisy20 = draw(20000, 1637, 1636)
            r = rotation(theta)
            clean_metric = evaluate_samples(clean20 @ r, problem)
            noisy_metric = evaluate_samples(noisy20 @ r, problem)
            accuracy = fidelity_metrics(noisy20 @ r, problem)
            gate_rows.append(dict(step=step, clean=clean_metric, noisy=noisy_metric,
                                  clean_pass=passes(problem, clean_metric), noisy_pass=passes(problem, noisy_metric),
                                  accuracy=accuracy, accuracy_pass=passes_accuracy(accuracy)))
            write(output / "gates.json", gate_rows)
    capture(0)
    for step in range(1, steps+1):
        batch_d = real(step)
        cached = []
        def generator_real():
            if not cached:
                cached.append(real(step))
            return cached[0]
        with torch.autograd.set_multithreading_enabled(False):
            trainer.step(batch_d, generator_real=generator_real)
        if step % 100 == 0 or step % 250 == 0 or step == steps:
            capture(step)
        if shift and step in (500, 1000):
            capture(step, event="target_jump", theta=angle(step+1))
    np.savez_compressed(output / "observations.npz", live=np.asarray(frames), clean=np.asarray(clean_frames),
                        steps=np.asarray(frame_steps), angles=np.asarray(angles), centers=base.cpu().numpy())
    write(output / "observations.json", rows)
    torch.save(trainer.state_dict(), output / "checkpoint.pt")
    def suffix(key):
        count = 0
        for row in reversed(gate_rows):
            if not row[key]:
                break
            count += 1
        return count
    summary = dict(name=problem+("_moving" if shift else ""), kind="native", recipe=recipe.to_dict(),
                   steps=steps, seed=1234, device=device, frames=len(rows),
                   sampling="served weights; independent fixed latent seed 77 and output-noise seed 78; 4k diagnostic clouds; independent 20k gate draws",
                   native_noisy_status="PASS" if suffix("noisy_pass")>=5 else "FAIL",
                   native_clean_status="PASS" if suffix("clean_pass")>=5 else "FAIL",
                   accuracy_status="PASS" if suffix("accuracy_pass")>=5 else "FAIL",
                   noisy_passing_suffix=suffix("noisy_pass"), final=gate_rows[-1],
                   seconds=time.perf_counter()-started, torch=torch.__version__,
                   base_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                   source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((root / "particlegan").glob("*.py"))},
                   capture_sha256=hashlib.sha256((output / "observations.npz").read_bytes()).hexdigest(),
                   artifact=str(output))
    if shift:
        phases=[x for x in gate_rows if x["step"]%500==0]
        baseline=phases[0]["noisy"]["hq"]
        summary["moving_original_status"]="PASS" if all(x["noisy"]["hq"]>=.9*baseline and x["noisy"]["modes"]>=95 for x in phases[1:]) else "FAIL"
    write(output / "summary.json", summary)
    print(json.dumps(dict(event="NATIVE_DONE", name=summary["name"], native=summary["native_noisy_status"],
                          accuracy=summary["accuracy_status"], seconds=summary["seconds"])), flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("problem",choices=["grid100","rotated100","staggered100"])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--steps",type=int,default=7000)
    ap.add_argument("--shift",action="store_true")
    args=ap.parse_args()
    run(args.problem,args.output,device=args.device,steps=args.steps,shift=args.shift)


if __name__ == "__main__":
    main()
