"""Fixed-state minibatch diagnostic for the delayed PR84 ring failures.

This never continues training.  At a captured post-D state it replays the
actual next G batch, then draws sixteen successive held-out G batches while
restoring G/prior weights and Adam moments before every hypothetical update.
The D state and Adam metric start identically for every batch.  The ring
centers appear only in the offline directional analysis.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from benchmarks.transfer_suite.legacy_noise_adapters import (
    NoisePolicy, wrap_input, wrap_output,
)
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe


STEPS = (1324, 1325, 1389, 1539, 1540)
HELDOUT_BATCHES = 16


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vectors(parameters, optimizer):
    result = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            result.append(parameter)
    assert parameters == len(result)
    return result


def _state_equal(actual, expected) -> bool:
    if isinstance(actual, torch.Tensor):
        return isinstance(expected, torch.Tensor) and torch.equal(actual, expected)
    if isinstance(actual, dict):
        return isinstance(expected, dict) and actual.keys() == expected.keys() and all(
            _state_equal(actual[key], expected[key]) for key in actual)
    if isinstance(actual, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(
            _state_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _construct(snapshot: dict, config: dict):
    recipe, noise, _ = declared_recipe(config)
    policy = NoisePolicy(noise["output_noise_std"], noise["input_noise_std"],
                         noise["input_noise_anneal_end"], 1200, seed=0,
                         output_noise_warmup=noise.get("output_noise_warmup", 0.0),
                         output_noise_learnable=noise.get("output_noise_learnable", False),
                         output_noise_rng=noise.get("output_noise_rng"))
    prior = recipe.make_prior(num_particles=12, z_dim=mode_hold.Z_DIM,
                              init_std=.5, generator=torch.Generator().manual_seed(0))
    generator = wrap_output(SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                                mode_hold.N_HIDDEN, 2), policy)
    critic = wrap_input(SimpleMLPDiscriminator(2, mode_hold.HIDDEN,
                                               mode_hold.N_HIDDEN, mode_hold.FOURIER), policy)
    optimizer_g, optimizer_d = recipe.make_optimizers(generator, critic, prior)
    generator.load_state_dict(snapshot["generator"])
    critic.load_state_dict(snapshot["critic"])
    prior.load_state_dict(snapshot["prior"])
    optimizer_g.load_state_dict(deepcopy(snapshot["optimizer_g"]))
    optimizer_d.load_state_dict(deepcopy(snapshot["optimizer_d"]))
    policy.input_sigma = snapshot["noise"]["input_sigma"]
    policy.output_sigma = snapshot["noise"]["output_sigma"]
    policy.input_stream.set_state(snapshot["rng"]["input"])
    if policy.output_stream is not None:
        policy.output_stream.set_state(snapshot["rng"]["output"])
    data = torch.Generator().set_state(snapshot["rng"]["data"])
    torch.set_rng_state(snapshot["rng"]["torch"])
    assert policy.input_sigma == 0.0 and policy.output_sigma == .029
    assert [g["lr"] for g in optimizer_g.param_groups] == [.00425, .0085]
    return recipe, policy, prior, generator, critic, optimizer_g, optimizer_d, data


def _stencil(critic, x: torch.Tensor, width: float) -> torch.Tensor:
    # Input noise is exactly zero at these saved states, so the wrapped
    # critic's base forward is its inner SimpleMLPDiscriminator forward.
    model = critic.model
    values = [model(x)]
    for dim in range(x.shape[-1]):
        shift = torch.zeros_like(x)
        shift[..., dim] = width
        values.extend((model(x + shift), model(x - shift)))
    return torch.stack(values, 0).mean(0)


def _clean(generator, prior) -> torch.Tensor:
    with torch.no_grad():
        return generator.model(prior.z).detach().clone()


def _energy_fraction(vectors: list[torch.Tensor]) -> dict:
    stacked = torch.stack([value.double().flatten() for value in vectors])
    mean = stacked.mean(0)
    energy = float(stacked.square().sum(1).mean())
    variance = float(stacked.var(0, unbiased=True).sum())
    signal = max(0., float(mean.square().sum()) - variance / len(vectors))
    return dict(batches=len(vectors), rms=math.sqrt(energy / stacked.shape[1]),
                energy=energy, variance_trace=variance,
                bias_corrected_coherent_fraction=signal / energy if energy else 0.)


def _restore_for_trial(snapshot, generator, prior, optimizer_g):
    generator.load_state_dict(snapshot["generator"])
    prior.load_state_dict(snapshot["prior"])
    optimizer_g.load_state_dict(deepcopy(snapshot["optimizer_g"]))


def _metric_and_scaled_gradient(optimizer_g) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    metric, scaled = [], []
    for group in optimizer_g.param_groups:
        for p in group["params"]:
            saved = optimizer_g.state[p]
            bias = 1. - group["betas"][1] ** float(saved["step"])
            denom = (saved["exp_avg_sq"] / bias).sqrt() + group["eps"]
            m = (group["lr"] / denom).double().detach().clone()
            metric.append(m)
            scaled.append((p.grad.detach().double() * m.sqrt()).clone())
    return metric, scaled


def _rho(before, after, g0, g1, metric):
    numerator = denominator = 0.
    for old, new, first, second, p in zip(before, after, g0, g1, metric):
        numerator += float((p * (second - first).double().square()).sum())
        denominator += float(((new - old).double().square() / p).sum())
    value = math.sqrt(numerator / denominator) if denominator else 0.
    if not math.isfinite(value):
        raise RuntimeError("nonfinite replay curvature")
    return value


def _directional(row: torch.Tensor, displacement: torch.Tensor, means: torch.Tensor):
    distances = torch.cdist(row, means)
    radius, nearest = distances.min(dim=1)
    center = means[nearest]
    outward = (row - center) / radius.clamp_min(1e-12).unsqueeze(1)
    radial = (displacement * outward).sum(1)
    # A positive derivative of distance squared is outward and locally harmful
    # only for particles already assigned to a mode; this is an offline probe.
    derivative = 2 * ((row - center) * displacement).sum(1)
    return dict(radial=radial.tolist(), mean_radius=float(radius.mean()),
                max_radius=float(radius.max()),
                outward_rows=int((radial > 0).sum()),
                mean_derivative=float(derivative.mean()),
                nearest=nearest.tolist(), radius=radius.tolist())


def _quality_gradient(generator, prior, means, parameters):
    """Offline nearest-center squared-distance gradient; never trains G."""
    points = generator.model(prior.z)
    with torch.no_grad():
        nearest = torch.cdist(points.detach(), means).argmin(dim=1)
    quality = .5 * (points - means[nearest]).square().sum() / len(points)
    return [value.detach().clone() for value in torch.autograd.grad(quality, parameters)]


def _quality_direction(quality_gradient, game_gradient, metric, roles):
    result = {}
    for role in ("network", "prior"):
        indices = [index for index, name in enumerate(roles) if name == role]
        quality_norm = sum(float(quality_gradient[i].double().square().sum())
                           for i in indices)
        raw_norm = sum(float(game_gradient[i].double().square().sum())
                       for i in indices)
        scaled_norm = sum(float((metric[i] * game_gradient[i].double()).square().sum())
                          for i in indices)
        raw_dot = -sum(float((quality_gradient[i].double()
                              * game_gradient[i].double()).sum()) for i in indices)
        scaled_dot = -sum(float((quality_gradient[i].double() * metric[i]
                                 * game_gradient[i].double()).sum()) for i in indices)
        result[role] = dict(
            raw_direction_cosine=(raw_dot / math.sqrt(quality_norm * raw_norm)
                                  if quality_norm * raw_norm else 0.),
            adam_metric_direction_cosine=(scaled_dot / math.sqrt(
                quality_norm * scaled_norm) if quality_norm * scaled_norm else 0.),
            raw_directional_derivative=raw_dot,
            adam_metric_directional_derivative=scaled_dot,
        )
    return result


def analyze_step(step: int, saved: dict, capture: dict, config: dict) -> dict:
    snapshot = saved[step]["post_accepted_d"]
    recipe, policy, prior, generator, critic, optimizer_g, _, data = _construct(snapshot, config)
    loss = recipe.make_loss()
    means = mode_hold.ring_means()
    width = capture[step]["record"]["critic_width"]
    base = _clean(generator, prior)
    roles = ["prior" if group.get("_comparison_prior") else "network"
             for group in optimizer_g.param_groups for _ in group["params"]]
    quality_gradient = _quality_gradient(generator, prior, means,
                                         _vectors(len(roles), optimizer_g))
    raw = {key: [] for key in ("network", "prior")}
    scaled = {key: [] for key in ("network", "prior")}
    output = {key: [] for key in ("joint", "network", "prior")}
    rows = []
    for batch_index in range(HELDOUT_BATCHES + 1):
        _restore_for_trial(snapshot, generator, prior, optimizer_g)
        before_params = [p.detach().clone() for p in _vectors(len(roles), optimizer_g)]
        before_rng = (torch.get_rng_state().clone(), data.get_state().clone(),
                      policy.input_stream.get_state().clone())
        latent, indices = prior.sample(mode_hold.BATCH, generator=data)
        fake_logits = _stencil(critic, generator(latent), width)
        real = mode_hold.sample_ring(means, mode_hold.BATCH, mode_hold.SIGMA, data)
        real_logits = _stencil(critic, real, width)
        after_rng = (torch.get_rng_state().clone(), data.get_state().clone(),
                     policy.input_stream.get_state().clone())
        optimizer_g.zero_grad()
        loss.g_loss(fake_logits, real_logits).backward()
        first = [p.grad.detach().clone() for p in _vectors(len(roles), optimizer_g)]
        old_metric, scaled_first = _metric_and_scaled_gradient(optimizer_g)
        quality_direction = _quality_direction(quality_gradient, first, old_metric, roles)
        for role in ("network", "prior"):
            raw[role].append(torch.cat([g.flatten() for g, r in zip(first, roles) if r == role]))
            scaled[role].append(torch.cat([g.flatten() for g, r in zip(scaled_first, roles) if r == role]))
        optimizer_g.step()
        after_params = [p.detach().clone() for p in _vectors(len(roles), optimizer_g)]
        metric, _ = _metric_and_scaled_gradient(optimizer_g)
        unbounded = _clean(generator, prior)
        with torch.no_grad():
            base_net = {name.removeprefix("model."): value for name, value in
                        snapshot["generator"].items() if name.startswith("model.")}
        if batch_index == 0:
            # There are two phase-1 prior samples: D and G.  The G sample is
            # the second, after the D fake draw.
            actual_prior = [v for v in capture[step]["batches"]
                            if v["kind"] == "prior" and v["phase"] == 1][-1]
            actual_real = [v for v in capture[step]["batches"]
                           if v["kind"] == "real" and v["phase"] == 1][-1]
            if indices.tolist() != actual_prior["indices"] or not torch.equal(
                    real, torch.tensor(actual_real["values"])):
                raise RuntimeError(f"step {step}: actual batch replay differs")
            expected_unbounded = saved[step].get("post_unbounded_g")
            if expected_unbounded is not None and not (
                    _state_equal(generator.state_dict(), expected_unbounded["generator"])
                    and _state_equal(prior.state_dict(), expected_unbounded["prior"])
                    and _state_equal(optimizer_g.state_dict(), expected_unbounded["optimizer_g"])):
                raise RuntimeError(f"step {step}: ordinary G proposal differs")
            if not torch.equal(unbounded, torch.tensor(capture[step]["unbounded_joint"])):
                raise RuntimeError(f"step {step}: clean unbounded output differs")
        # Re-evaluate the same G field at the proposed point, without a second
        # optimizer update.  Restore the post-first-pass streams afterward.
        torch.set_rng_state(before_rng[0]); data.set_state(before_rng[1]);
        policy.input_stream.set_state(before_rng[2])
        latent2, _ = prior.sample(mode_hold.BATCH, generator=data)
        fake2 = _stencil(critic, generator(latent2), width)
        real2 = mode_hold.sample_ring(means, mode_hold.BATCH, mode_hold.SIGMA, data)
        optimizer_g.zero_grad()
        loss.g_loss(fake2, _stencil(critic, real2, width)).backward()
        second = [p.grad.detach().clone() for p in _vectors(len(roles), optimizer_g)]
        rho = _rho(before_params, after_params, first, second, metric)
        factor = min(1., .25 / rho) if rho else 1.
        with torch.no_grad():
            for parameter, old, new in zip(_vectors(len(roles), optimizer_g),
                                            before_params, after_params):
                parameter.copy_(torch.lerp(old, new, factor) if factor < 1 else new)
        bounded = _clean(generator, prior)
        with torch.no_grad():
            bounded_network_only = generator.model(snapshot["prior"]["z"]).detach().clone()
            bounded_prior_only = functional_call(generator.model, base_net,
                                                 (prior.z,)).detach().clone()
        if batch_index == 0:
            expected_bounded = saved[step].get("post_bounded_g")
            expected_rho = capture[step]["record"]["g"]["rho"]
            if abs(rho - expected_rho) > 1e-6:
                raise RuntimeError(f"step {step}: replay rho differs: {rho} vs {expected_rho}")
            parameter_error = (None if expected_bounded is None else max(
                [float((v - expected_bounded["generator"][k]).abs().max())
                 for k, v in generator.state_dict().items()]
                + [float((v - expected_bounded["prior"][k]).abs().max())
                   for k, v in prior.state_dict().items()]))
            output_error = float((bounded - torch.tensor(
                capture[step]["bounded_joint"])).abs().max())
            if (parameter_error is not None and parameter_error > 1e-6) or output_error > 1e-6:
                raise RuntimeError(f"step {step}: bounded G proposal differs: "
                                   f"parameter={parameter_error}, output={output_error}")
        # The next held-out draw continues from the exact post-first-pass RNG,
        # never from an extra curvature replay.
        torch.set_rng_state(after_rng[0]); data.set_state(after_rng[1]);
        policy.input_stream.set_state(after_rng[2])
        output["joint"].append((bounded - base).flatten())
        output["network"].append((bounded_network_only - base).flatten())
        output["prior"].append((bounded_prior_only - base).flatten())
        rows.append(dict(batch="actual" if batch_index == 0 else "heldout",
                         rho=rho, factor=factor,
                         quality_direction=quality_direction,
                         **(dict(actual_parameter_error=parameter_error,
                                 actual_output_error=output_error,
                                 actual_rho_error=abs(rho - expected_rho))
                            if batch_index == 0 else {}),
                         unbounded_output_rms=float((unbounded - base).square().sum(1).mean().sqrt()),
                         bounded_output_rms=float((bounded - base).square().sum(1).mean().sqrt()),
                         **_directional(base, bounded - base, means)))
    return dict(step=step, actual_unbounded_output_exact=True,
                actual_full_state_parity_available=expected_unbounded is not None,
                actual_bounded_replay_tolerance=1e-6, width=width,
                sample_count=HELDOUT_BATCHES,
                clean_support=base.tolist(), actual=rows[0], heldout=rows[1:],
                raw_gradients={role: _energy_fraction(values[1:]) for role, values in raw.items()},
                adam_scaled_gradients={role: _energy_fraction(values[1:])
                                       for role, values in scaled.items()},
                clean_output_displacement={role: _energy_fraction(values[1:])
                                           for role, values in output.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    full_source = args.capture / "selected-states.pt"
    compact_source = args.capture / "compact-states.pt.gz"
    full_mode = full_source.exists()
    source = full_source if full_mode else compact_source
    raw_path = args.capture / "captured-raw.json"
    diagnosis_path = args.capture / ("diagnosis.json" if full_mode else "diagnosis.json.gz")
    config_path = ROOT / "configs/toy100/constraints_simple_regularization.json"
    with gzip.open(diagnosis_path, "rt") if diagnosis_path.suffix == ".gz" else \
            diagnosis_path.open() as file:
        diagnosis = json.load(file)
    if diagnosis["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("capture has no exact-source parity receipt")
    if full_mode:
        if diagnosis["selected_states_sha256"] != _digest(source):
            raise RuntimeError("complete saved states changed")
        saved = torch.load(source, weights_only=True)
        captured = {row["step"]: row for row in json.loads(raw_path.read_text())}
    else:
        manifest = json.loads((args.capture / "manifest.json").read_text())
        if (_digest(source) != manifest["files"]["compact-states.pt.gz"]["sha256"]
                or _digest(diagnosis_path) != manifest["files"]["diagnosis.json.gz"]["sha256"]
                or diagnosis["selected_states_sha256"] !=
                manifest["original_replay_sha256"]["selected-states.pt"]):
            raise RuntimeError("portable capture archive has a source/hash mismatch")
        with gzip.open(source, "rb") as file:
            saved = torch.load(file, weights_only=True)
        captured = {row["step"]: row["stages"] for row in diagnosis["rows"]}
    config = json.loads(config_path.read_text())
    with torch.random.fork_rng(devices=[]):
        rows = [analyze_step(step, saved, captured, config) for step in STEPS]
    result = dict(scope="fixed_post_d_saved_state_diagnostic_only",
                  no_training_updates=True, actual_unbounded_output_exact=True,
                  actual_full_state_parity_available=full_mode,
                  actual_bounded_replay_tolerance=1e-6,
                  fixed_d_and_initial_adam_metric=True,
                  heldout_batches_per_state=HELDOUT_BATCHES,
                  steps=list(STEPS), rows=rows,
                  source_sha256=_digest(Path(__file__)),
                  config_sha256=_digest(config_path),
                  captured_states_sha256=_digest(source),
                  original_full_states_sha256=diagnosis["selected_states_sha256"],
                  capture_diagnosis_sha256=_digest(diagnosis_path),
                  capture_source_sha256=diagnosis["observer_source_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with gzip.open(args.output, "wt") as file:
        json.dump(result, file, allow_nan=False)
    print(json.dumps(dict(scope=result["scope"], steps=result["steps"],
                          output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
