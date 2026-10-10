"""Reproduce configuration/analytic checks; never train or submit Forge work."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from particlegan import get_recipe
from experiments.forge.configuration_search import _grid, _load_spec
from experiments.forge.techniques import validate_same_technique


def rejection(call):
    try:
        call()
    except (ValueError, TypeError) as exc:
        return str(exc)
    raise AssertionError("Expected the audited interface to reject this input")


def fixture():
    # Explicit constant software fixture: no model construction RNG or training.
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(torch.ones(1, dtype=torch.float64)))
    return module


def group(optimizer):
    values = optimizer.param_groups[0]
    return {key: values[key] for key in ("lr", "betas", "eps", "amsgrad")}


def audit(extract):
    trainer, loss = extract["trainer"], extract["loss"]
    rate = trainer["g_learn_rate"]
    ratio = trainer["d_learn_rate"] / rate
    recipe = get_recipe("bcap", lr=rate, d_lr_mult=ratio,
                        betas=(trainer["g_beta1"], trainer["g_beta2"]),
                        eps=trainer["g_epsilon"], loss="least_squares")
    g, d = fixture(), fixture()
    og, od = recipe.make_optimizers(g, d)
    direct_d = recipe.make_critic_optimizer(
        d, betas=(trainer["d_beta1"], trainer["d_beta2"]), eps=trainer["d_epsilon"])
    assert not og.state and not od.state and not direct_d.state
    assert math.isclose(group(od)["lr"], trainer["d_learn_rate"], rel_tol=1e-15)
    assert group(od)["betas"] != group(direct_d)["betas"]
    assert group(od)["eps"] != group(direct_d)["eps"]

    grids = {
        "cartesian": {"lr": [.001, .002], "eps": [1e-8, .01]},
        "coupled": {"rates": [{"lr": .001, "d_lr_mult": 1.},
                                 {"lr": .002, "d_lr_mult": .5}]},
        "union": [{"lr": [.001]}, {"lr": [.002], "eps": [1e-8, .01]}],
        "pair_as_literal_choice": {"betas": [[.1, .72], [.61, .56]]},
    }
    grid_counts = {key: len(_grid(value)) for key, value in grids.items()}
    assert grid_counts == {"cartesian": 4, "coupled": 2, "union": 3,
                           "pair_as_literal_choice": 2}
    rejected = {key: rejection(lambda key=key, value=value: _grid({key: value}))
                for key, value in {
                    "loss": ["least_squares", "hinge"],
                    "optimizer_family": ["adam", "formulation"],
                    "g_betas": [[.61, .56]], "d_betas": [[.1, .72]],
                    "g_eps": [.3], "d_eps": [.0075], "seed": [0],
                }.items()}
    rejected["over_256"] = rejection(lambda: _grid({"lr": list(range(257))}))
    rejected["duplicate"] = rejection(lambda: _grid({"lr": [.001, .001]}))
    rejected["enable_beta1"] = rejection(lambda: validate_same_technique(
        get_recipe("bcap"), get_recipe("bcap", betas=recipe.betas)))
    rejected["random_strategy"] = rejection(lambda: _load_spec(ROOT, {"strategy": "random"}))

    real = torch.tensor([0., 1., 2.], dtype=torch.float64)
    fake = torch.tensor([-1., 0., 1.], dtype=torch.float64, requires_grad=True)
    a, b, c = loss["labels"]
    current_d = recipe.make_loss().d_loss(real, fake)
    legacy_d = .5 * ((real - b).square().mean() + (fake - a).square().mean())
    current_g = recipe.make_loss().g_loss(fake)
    legacy_g = .5 * (fake - c).square().mean()
    current_grad, = torch.autograd.grad(current_d, fake, retain_graph=True)
    legacy_grad, = torch.autograd.grad(legacy_d, fake)
    value_gap = abs((current_d - legacy_d).item())
    gradient_gap = (current_grad - legacy_grad).abs().max().item()
    assert math.isclose(value_gap, .5, abs_tol=1e-14)
    assert math.isclose(gradient_gap, 1/3, abs_tol=1e-14)
    assert abs((current_g - legacy_g).item()) < 1e-14

    adam = {}
    for role in ("g", "d"):
        beta2, eps = trainer[f"{role}_beta2"], trainer[f"{role}_epsilon"]
        lr, gradient = trainer[f"{role}_learn_rate"], .001
        # First dense step with zero initial moments; beta1 cancels here.
        tf_step = lr * gradient / (abs(gradient) + eps / math.sqrt(1 - beta2))
        pt_step = lr * gradient / (abs(gradient) + eps)
        assert abs(tf_step - pt_step) > 1e-10
        adam[role] = {"gradient": gradient, "tf_dense_first_step_magnitude": tf_step,
                      "pytorch_first_step_magnitude_same_numeric_eps": pt_step,
                      "absolute_gap": abs(tf_step - pt_step),
                      "equivalent_pytorch_eps_at_step_1": eps / math.sqrt(1 - beta2),
                      "equivalent_pytorch_eps_at_step_100": eps / math.sqrt(1 - beta2**100)}

    source_files = ["experiments/forge/configuration_search.py", "experiments/forge/boundaries.py",
                    "experiments/forge/techniques.py", "particlegan/recipes.py",
                    "particlegan/gan_loss.py", "particlegan/recipe_schedules.py",
                    "configs/forge/protocols/screening.json",
                    "configs/forge/views/discriminator_stability.json"]
    adam_source = Path(inspect.getfile(torch.optim.Adam))
    return {"schema": "forge_hypergan_search_audit_v1", "software_checks": "PASS",
            "scope": "configuration and analytic software audit; no training qualification",
            "training_updates": 0, "torch_version": torch.__version__,
            "installed_adam_source_sha256": hashlib.sha256(adam_source.read_bytes()).hexdigest(),
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                              for path in source_files},
            "historical_file_sha256": extract["source"]["sha256"],
            "grid_counts": grid_counts, "expected_rejections": rejected,
            "rate_mapping": {"lr": rate, "d_lr_mult": ratio},
            "current_recipe_generator_group": group(og),
            "current_recipe_critic_group": group(od),
            "direct_factory_critic_override_group": group(direct_d),
            "loss_probe": {"real_scores": real.tolist(), "fake_scores": fake.tolist(),
                           "current_d": current_d.item(), "legacy_d": legacy_d.item(),
                           "d_absolute_gap": value_gap, "d_fake_gradient_max_gap": gradient_gap,
                           "g_absolute_gap": abs((current_g - legacy_g).item())},
            "dense_adam_analytic_probe": adam}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optionally verify the full original Halloween JSON")
    parser.add_argument("--output", type=Path, help="Write a compact receipt (default: stdout)")
    args = parser.parse_args()
    extract = json.loads(Path(__file__).with_name("halloween-optimizer-loss.json").read_text())
    if args.source:
        raw = args.source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == extract["source"]["sha256"]
        original = json.loads(raw)
        assert original["trainer"] == extract["trainer"] and original["loss"] == extract["loss"]
    result = audit(extract)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"PASS: configuration/analytic audit; 0 training updates; receipt {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
