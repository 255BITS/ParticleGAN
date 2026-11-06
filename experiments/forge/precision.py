"""Task-owned low-precision execution conditions, shared by every candidate."""
from copy import deepcopy
from pathlib import Path

import torch

from particlegan.low_precision import EmulatedLinear, emulate_linear_training_

from .contracts import file_hash
from .state import state_digest


def declaration(format):
    if format not in ("fp32", "nvfp4", "mxfp8"):
        raise ValueError("unsupported toy linear precision")
    return {"schema_version": 1, "format": format, "backend": "software_reference",
                "gemms": ["fprop", "dgrad", "wgrad"], "master_parameters": "fp32",
                "accumulator": "fp32", "gemm_output": "fp32" if format == "fp32" else "bf16",
                "bias_storage_nonlinearity_loss_prior_optimizer": "fp32",
                "higher_order_autograd": "native_autograd" if format == "fp32" else "straight_through_quantizers",
                "stochastic_gradient_rounding": format == "nvfp4",
                "wgrad_random_hadamard": format == "nvfp4",
                "native_parity": "unverified"}


def validate(task):
    value = task["execution"].get("linear_precision")
    if value is None:
        return None
    if value != declaration(value.get("format")):
        raise ValueError("linear precision differs from the supported complete reference contract")
    if task["adapter"] != "transfer_vector":
        raise ValueError("low-precision toys require the shared public vector trainer")
    return value


def prepare(context, task, generator, discriminator):
    policy = validate(task)
    if policy is None:
        return
    signs_rng = context.streams.generator("init", component="precision", purpose="hadamard", device="cpu")
    signs = 2 * torch.randint(0, 2, (16,), generator=signs_rng).float() - 1
    if policy["format"] != "fp32":
        for role, model in (("generator", generator), ("discriminator", discriminator)):
            emulate_linear_training_(model, policy["format"], signs=signs,
                seed_for=lambda name, role=role: context.streams.seed_for("noise", component=role, purpose="rounding:" + name))


def receipt(task, generator, discriminator):
    policy = validate(task)
    if policy is None:
        return None
    models = {"generator": generator, "discriminator": discriminator}
    layers = {role: {name: {"format": layer.format, **layer.calls}
                     for name, layer in model.named_modules() if isinstance(layer, EmulatedLinear)}
              for role, model in models.items()}
    return {"contract": deepcopy(policy), "layers": layers,
            "native_speed_memory_or_kernel_parity_claim": False,
            "rounding_rng_owner": "each EmulatedLinear._extra_state tensor (format tag and RNG bytes)",
            "higher_order_scope": "software straight-through reference; native higher-order parity unverified"}


def initial_parameters(generator, discriminator):
    return {role: state_digest(dict(model.named_parameters()))
            for role, model in (("generator", generator), ("discriminator", discriminator))}


def grade(task, evidence):
    policy = validate(task)
    if policy is None:
        return None
    observed = evidence.get("linear_precision")
    if not isinstance(observed, dict):
        return "INCOMPLETE", "missing observed linear-precision execution"
    if observed.get("contract") != policy:
        return "INVALID", "observed linear precision differs from task"
    if policy["format"] != "fp32":
        for role in ("generator", "discriminator"):
            layers = observed.get("layers", {}).get(role, {})
            if not layers or any(layer.get("format") != policy["format"] for layer in layers.values()):
                return "INVALID", "missing or contradictory quantized linear layers"
            for phase in ("forward", "dgrad", "wgrad"):
                if not any(type(layer.get(phase)) is int and layer[phase] > 0 for layer in layers.values()):
                    return "INCOMPLETE", f"{role} did not exercise quantized {phase}"
    if not evidence.get("actual_training_gif", {}).get("sha256"):
        return "INCOMPLETE", "missing actual-training GIF"
    return None


def render(task, records, observations, output):
    """Render the already-scored samples; no new model calls or RNG draws."""
    import numpy as np

    from benchmarks.toy_audit.api_run import render_gif
    from benchmarks.toy_audit.gaussian1d_quality import sample_target
    spec = task["execution"]["host_definition"]
    target = sample_target(spec, 4096, torch.Generator().manual_seed(80311)).numpy()[:, 0]
    edges = np.linspace(-1., 5., 61)
    width = float(edges[1] - edges[0])
    def density(values):
        return np.histogram(values, edges)[0] / len(values) / width
    frames = []
    for record, row in zip(records, observations):
        failed = [f"{name} {op} {bound}" for name, op, bound in task["evaluation"]["thresholds"]
                  if not {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}[op](row[name], bound)]
        frames.append({"step": record["step"], "passed": not failed, "failed_bounds": failed,
            "metrics": {k: v for k, v in row.items() if k != "step"}, "views": [{
                "kind": "bar", "title": "Target and actual low-precision generator samples",
                "target": density(target), "samples": density(record["samples"].numpy()[:, 0]),
                "bin_centers": (edges[1:] + edges[:-1]) / 2, "bin_width": width,
                "xlim": [-1., 5.], "ylim": [0., 8.5], "xlabel": "x", "ylabel": "density",
                "caption": "Native-format software reference; exact CDF gate includes all tails. Native kernel parity unverified."}]})
    passed = len(frames) == task["evaluation"]["observations"] and all(
        frame["passed"] for frame in frames[-task["evaluation"]["minimum_stable_checks"]:])
    path = Path(output) / "goal.gif"
    render_gif({"id": task["id"], "goal": task["description"], "default_steps": task["execution"]["steps"]},
               frames, path, full_budget=True, requested_steps=task["execution"]["steps"],
               final_verdict="PASS" if passed else "FAIL")
    return {"path": str(path.resolve()), "sha256": file_hash(path)}
