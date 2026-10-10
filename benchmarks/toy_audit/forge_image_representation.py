"""Bounded reference fitting of the exact Forge image hosts, not GAN qualification.

This diagnostic supplies retained parameter witnesses through the public clean,
enumerated sampler. Supervised template labels are used only by this reference
fit; its results cannot fill any ordinary training PASS cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from experiments.forge.api import FormulationContext
from experiments.forge.contracts import file_hash, read_json, stable_hash
from experiments.forge.imageprofiles import build_image_models, resolve_image_spec
from benchmarks.transfer_suite.image_tasks import image_metrics, templates


TASKS = ("img_stripes2", "img_bars4", "img_blobs4", "img_intensity2")


def run(root, output, *, device="cpu", iterations=1000, timeout=300, tasks=TASKS, loss_kind="mse"):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    declaration = {"schema_version": 1, "kind": "supervised_representation_diagnostic",
                   "tasks": list(tasks), "iterations_per_host": iterations,
                   "timeout_seconds_per_host": timeout, "seed": 0,
                   "reference_lr": .01 if loss_kind == "mse" else .003,
                   "reference_loss": loss_kind,
                   "reference_optimizer": "torch.optim.Adam", "device": device,
                   "ordinary_training_updates": 0, "ordinary_qualification_credit": False,
                   "sampling_law": "enumerated_prior_without_output_noise",
                   "scoring_weights": "live", "scope": "Exact frozen image host and declared numerical tolerances",
                   "reproducer_sha256": file_hash(__file__)}
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    results = []
    for name in tasks:
        start = time.monotonic()
        task = read_json(root / "configs/forge/tasks" / f"{name}.json")
        spec = resolve_image_spec(task)
        context = FormulationContext(seed=0, device=device, prior=task["execution"]["prior"],
            recipe_overrides={"num_particles": spec["particles"], "z_dim": spec["z_dim"],
                              "batch_size": spec["batch_size"], "total_steps": spec["steps"]})
        g, d = build_image_models(context, spec)
        trainer = context.build_trainer(g, d)
        targets = templates(spec).to(device)
        labels = torch.arange(spec["particles"], device=device) % len(targets)
        with torch.no_grad():
            trainer.prior.z.zero_()
            trainer.prior.z[:, :len(targets)] = torch.nn.functional.one_hot(labels, len(targets)).float()
        latent = trainer.prior.z.detach()
        optimizer = torch.optim.Adam(g.parameters(), lr=declaration["reference_lr"])
        updates = 0
        for updates in range(1, iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            predicted = g(latent)
            expected = targets[labels]
            if loss_kind == "balanced_bce":
                weights = torch.where(expected > 0, 15., 1.)
                loss = (torch.nn.functional.binary_cross_entropy(predicted, expected, reduction="none") * weights).mean()
            else:
                loss = (predicted - expected).square().mean()
            loss.backward()
            optimizer.step()
            if time.monotonic() - start >= timeout:
                break
        with torch.no_grad():
            generated = trainer.sample(spec["particles"], fixed_first_n=True,
                generator=context.streams.generator("eval", component="witness", purpose="enumeration"))
            metrics = image_metrics(generated, targets, task["evaluation"]["measurement"])
        passed = metrics["modes"] >= spec["modes"] and metrics["hq"] >= task["evaluation"]["measurement"]["hq_min"]
        state = output / f"{name}-witness.pt"
        torch.save({"generator": g.state_dict(), "prior": trainer.prior.state_dict(),
                    "resolved_recipe": trainer.recipe.to_dict(), "task": task}, state)
        result = {"task": name, "representation_status": "SUPPORTED" if passed else "UNRESOLVED",
                  "task_sha256": stable_hash(task), "host_definition": spec, "metrics": metrics,
                  "prior": task["execution"]["prior"], "diagnostic_fit_updates": updates,
                  "ordinary_training_updates": trainer.completed_steps,
                  "ordinary_qualification_credit": False,
                  "elapsed_seconds": time.monotonic() - start,
                  "artifact": {"path": str(state), "sha256": file_hash(state)},
                  "source_sha256": {p: file_hash(root / p) for p in
                     ("benchmarks/transfer_suite/image_tasks.py", "experiments/forge/imageprofiles.py",
                      "particlegan/training.py", "particlegan/recipes.py")}}
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "result.json").write_text(json.dumps({"declaration": declaration, "results": results}, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--loss", choices=("mse", "balanced_bce"), default="mse")
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[2], args.output, device=args.device,
        tasks=tuple(args.task) if args.task else TASKS, loss_kind=args.loss)


if __name__ == "__main__":
    main()
