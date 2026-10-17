"""Versioned evaluator controls for the audit's weaker non-image definitions.

No training occurs here. These stricter gates have their own identity and do
not replace historical verdicts. An oracle PASS calibrates an evaluator, not a
model. All arrays describe the explicitly declared clean sampling law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION = "toy-definition-quality-v1"
WORDS = ("apple", "grape", "lemon", "melon", "berry")
CHARS = "abcdefghijklmnopqrstuvwxyz_ "
WORD_TOKENS = np.array([[CHARS.index(c) for c in word + "_"] for word in WORDS])


def _array(value, *, ndim=None):
    result = np.asarray(value, dtype=np.float64)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("evaluation arrays must be nonempty and finite")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions")
    return result


def _ks(values, cdf):
    ordered = np.sort(values)
    expected = cdf(ordered)
    n = len(ordered)
    return float(max(np.max(np.arange(1, n + 1) / n - expected),
                     np.max(expected - np.arange(n) / n)))


def two_pole_metrics(points):
    """Clean enumeration against the actual 12-row, six-per-pole target grid.

    The source's deterministic linspace sampler depends on batch size. This
    contract uses its actual training batch/table size, not a continuous law
    inferred from a larger real_batch call. Sorted quantiles compare discrete
    atoms directly and avoid a continuous-CDF KS formula at target jumps.
    """
    points = _array(points).reshape(-1)
    ids = (points >= 0).astype(int)
    centers = np.array([-1., 1.])
    distances = np.abs(points - centers[ids])
    fractions = np.bincount(ids, minlength=2) / len(points)
    offsets = np.linspace(-.05, .05, 6)
    target = np.concatenate([-1 + offsets, 1 + offsets])
    quantile_errors = np.abs(np.sort(points) - target) / .05 if len(points) == 12 else None
    metrics = dict(sample_count=len(points), pole_fractions=fractions.tolist(),
                   mass_tv=float(np.abs(fractions - .5).sum() / 2),
                   support_fraction=float((distances <= .050001).mean()),
                   expected_sample_count=12, sampling="exact clean 12-atom enumeration",
                   quantile_normalizer="target offset half-width .05, not standard deviation",
                   max_grid_quantile_error_halfwidth=float(quantile_errors.max()) if quantile_errors is not None else None,
                   mean_grid_quantile_error_halfwidth=float(quantile_errors.mean()) if quantile_errors is not None else None)
    metrics["passed"] = bool(len(points) == 12 and metrics["mass_tv"] <= .05
                             and metrics["support_fraction"] >= .95
                             and quantile_errors.max() <= .10)
    return metrics


def gaussian_metrics(points, *, mean=1., sigma=.2):
    """N((1,1), .04 I): moments, radial law and 16 fixed projections."""
    points = _array(points, ndim=2)
    if points.shape[1] != 2 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("the quickstart law needs two coordinates and sigma > 0")
    z = (points - mean) / sigma
    covariance = np.cov(z, rowvar=False, bias=True) if len(z) > 1 else np.zeros((2, 2))
    eigenvalues = np.linalg.eigvalsh(covariance)
    normal_cdf = lambda x: np.array([.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in x])
    projected_ks = []
    for theta in np.arange(16) * math.pi / 16:
        projected_ks.append(_ks(z @ [math.cos(theta), math.sin(theta)], normal_cdf))
    radial_ks = _ks((z * z).sum(1), lambda x: 1 - np.exp(-x / 2))
    metrics = dict(sample_count=len(z), mean_error_sigma=float(np.linalg.norm(z.mean(0))),
                   covariance_eigenvalues=eigenvalues.tolist(), radial_ks=radial_ks,
                   max_projection_ks=max(projected_ks))
    metrics["passed"] = bool(len(z) >= 1024 and metrics["mean_error_sigma"] <= .10
                             and eigenvalues.min() >= .85 and eigenvalues.max() <= 1.15
                             and radial_ks <= .075 and max(projected_ks) <= .06)
    return metrics


def word_templates():
    return np.eye(len(CHARS))[WORD_TOKENS].transpose(0, 2, 1)


def word_probabilities(logits):
    """Explicit logits-to-probabilities adapter; hard decoding is insufficient."""
    logits = _array(logits, ndim=3)
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def five_word_metrics(probabilities, reconstructions):
    """Uniform fixed-word mass plus correctly paired full six-token recovery."""
    probabilities, reconstructions = (_array(x, ndim=3) for x in (probabilities, reconstructions))
    for value in (probabilities, reconstructions):
        if value.shape[1:] != (len(CHARS), 6) or (value < 0).any() or not np.allclose(value.sum(1), 1):
            raise ValueError("expected normalized probabilities over all characters, including padding")
    if len(reconstructions) != len(WORDS):
        raise ValueError("reconstruction rows must correspond to the five canonical inputs")
    decoded = probabilities.argmax(1)
    matches = (decoded[:, None] == WORD_TOKENS[None]).all(2)
    ids = matches.argmax(1)
    confidence = probabilities[np.arange(len(probabilities))[:, None], WORD_TOKENS[ids], np.arange(6)]
    quality = matches.any(1) & (confidence.min(1) >= .90)
    masses = np.bincount(ids[quality], minlength=5) / len(probabilities)
    rejected = float(1 - quality.mean())
    mass_tv = float((np.abs(masses - .2).sum() + rejected) / 2)
    reconstruction_confidence = reconstructions[np.arange(5)[:, None], WORD_TOKENS, np.arange(6)]
    recon_exact = bool(np.array_equal(reconstructions.argmax(1), WORD_TOKENS))
    metrics = dict(sample_count=len(probabilities), quality_fraction=float(quality.mean()),
                   word_masses=masses.tolist(), mass_tv=mass_tv,
                   modes=int((masses > 0).sum()), reconstruction_exact=recon_exact,
                   minimum_reconstruction_token_probability=float(reconstruction_confidence.min()),
                   reconstruction_nll=float(-np.log(np.clip(reconstruction_confidence, 1e-12, 1)).mean()))
    metrics["passed"] = bool(len(probabilities) >= 100 and metrics["quality_fraction"] >= .95
                             and metrics["modes"] == 5 and mass_tv <= .10 and recon_exact
                             and reconstruction_confidence.min() >= .90)
    return metrics


def paired_edit_metrics(prediction, target, neutral):
    """A correspondence gate: output marginals cannot substitute for row pairs."""
    prediction, target, neutral = (_array(x) for x in (prediction, target, neutral))
    if prediction.shape != target.shape or neutral.shape != target.shape:
        raise ValueError("paired arrays must have identical shape")
    baseline = float(np.mean((neutral - target) ** 2))
    if baseline <= 1e-12:
        raise ValueError("the task must contain a nonzero edit relative to the neutral witness")
    mse = float(np.mean((prediction - target) ** 2))
    return dict(mse=mse, neutral_mse=baseline, relative_mse=mse / baseline,
                passed=bool(mse / baseline <= .10))


def useful_code_metrics(live_losses, zero_code_losses):
    """Every fixed judge must become worse when the learned code is removed."""
    live, zero = (_array(x, ndim=1) for x in (live_losses, zero_code_losses))
    if live.shape != zero.shape:
        raise ValueError("ablations must use the same judges and held-out panels")
    deltas = zero - live
    return dict(zero_minus_live=deltas.tolist(), passed=bool((deltas > 1e-6).all()))


def landing_metrics(landed, crashed, steps, *, horizon=48):
    """Count every initial state, including crashes and censored timeouts."""
    landed, crashed = np.asarray(landed), np.asarray(crashed)
    steps = _array(steps, ndim=1)
    if (landed.dtype != np.bool_ or crashed.dtype != np.bool_ or landed.shape != steps.shape
            or crashed.shape != steps.shape or (landed & crashed).any()
            or horizon < 1 or (steps < 1).any() or (steps > horizon).any()):
        raise ValueError("landing events must be disjoint, aligned and within the horizon")
    timeout = ~(landed | crashed)
    restricted_mean = float(np.where(landed, steps, horizon).mean())
    metrics = dict(landings=float(landed.mean()), crash_rate=float(crashed.mean()),
                   timeout_rate=float(timeout.mean()), restricted_mean_steps=restricted_mean)
    metrics["passed"] = bool(metrics["landings"] >= .95 and metrics["crash_rate"] <= .02
                             and metrics["timeout_rate"] <= .03 and restricted_mean <= 28)
    return metrics


def ring_metrics(points, *, shift=(0., 0.)):
    """All eight equal-weight Gaussian ring modes, including within-mode width."""
    points = _array(points, ndim=2)
    if points.shape[1] != 2:
        raise ValueError("ring samples need two coordinates")
    angles = np.arange(8) * 2 * math.pi / 8
    centers = 3 * np.stack([np.cos(angles), np.sin(angles)], 1) + np.asarray(shift)
    distances = np.linalg.norm(points[:, None] - centers[None], axis=2)
    ids = distances.argmin(1)
    quality = distances[np.arange(len(points)), ids] <= 3 * .07
    masses = np.bincount(ids, minlength=8) / len(points)
    eigenvalues, radial_ks = [], []
    for k in range(8):
        z = (points[ids == k] - centers[k]) / .07
        if len(z) >= 32:
            eigenvalues.append(np.linalg.eigvalsh(np.cov(z, rowvar=False, bias=True)).tolist())
            radial_ks.append(_ks((z * z).sum(1), lambda x: 1 - np.exp(-x / 2)))
        else:
            eigenvalues.append([0., 0.])
            radial_ks.append(1.)
    metrics = dict(sample_count=len(points), hq=float(quality.mean()),
                   modes=int((np.bincount(ids[quality], minlength=8) > 0).sum()),
                   mass_tv=float(np.abs(masses - 1 / 8).sum() / 2),
                   covariance_eigenvalues=eigenvalues, max_radial_ks=max(radial_ks))
    eig = np.array(eigenvalues)
    metrics["passed"] = bool(len(points) >= 4096 and metrics["modes"] == 8 and metrics["hq"] >= .90
                             and metrics["mass_tv"] <= .075 and eig.min() >= .5 and eig.max() <= 1.5
                             and metrics["max_radial_ks"] <= .10)
    return metrics


def terminal_window(passed, minimum=5):
    if minimum < 1 or any(type(value) is not bool for value in passed):
        raise ValueError("use boolean observations and a positive terminal window")
    suffix = 0
    for value in reversed(passed):
        if not value:
            break
        suffix += 1
    return dict(passing_suffix=suffix, passed=suffix >= minimum)


def build_controls(root):
    """Positive witnesses and discriminating counterexamples; no optimizer."""
    import torch
    from lib import safe_fast_landing as landing, yue2_particle_toy as sign
    from benchmarks.locked_shared import two_pole

    rng = np.random.default_rng(713)
    templates = word_templates()
    words = np.repeat(templates, 40, axis=0)
    weak_words = .04 * words + .96 / len(CHARS)
    gaussian = 1 + .2 * rng.standard_normal((4096, 2))
    theta = np.arange(4096) * 2 * math.pi / 4096
    circle = 1 + .2 * math.sqrt(2) * np.stack([np.cos(theta), np.sin(theta)], 1)
    from benchmarks.legacy.locked_shared import LOCKED_SHARED
    poles = two_pole.real_batch(LOCKED_SHARED.n_particles).numpy()
    wrong_padding = words.copy()
    wrong_padding[:, :, -1] = 0
    wrong_padding[:, CHARS.index(" "), -1] = 1
    controls = {
        "develop-two_pole": {
            "exact_target": two_pole_metrics(poles),
            "one_pole_travel": two_pole_metrics(np.full_like(poles, .6)),
            "centers_only": two_pole_metrics(np.repeat([-1., 1.], 6)),
            "wrong_256_atom_cohort": two_pole_metrics(two_pole.real_batch(256).numpy()),
            "law": "Twelve equally weighted atoms: six linspace offsets from -.05 to .05 per pole. The source sampler is batch-size dependent.",
            "old_travel_accepts_one_pole": two_pole.cell_wins(.6, .5),
        },
        "source-family-16": {
            "independent_normal_draw": gaussian_metrics(gaussian),
            "center_only": gaussian_metrics(np.ones_like(gaussian)),
            "wrong_width": gaussian_metrics(1 + (gaussian - 1) / 5),
            "shifted": gaussian_metrics(gaussian + .2),
            "isotropic_circle_same_covariance": gaussian_metrics(circle),
        },
        "source-family-15": {
            "exact_words_and_reconstruction": five_word_metrics(words, templates),
            "correct_argmax_low_confidence": five_word_metrics(weak_words, templates),
            "one_word_only": five_word_metrics(np.repeat(templates[:1], 200, axis=0), templates),
            "correct_marginal_wrong_reconstruction": five_word_metrics(words, np.roll(templates, 1, axis=0)),
            "display_equivalent_wrong_padding": five_word_metrics(wrong_padding, templates),
        },
    }
    catalog = json.loads((root / "reports/toy_audit/catalog.json").read_text())
    retained = next(row for row in catalog["cases"] if row["id"] == "develop-two_pole")
    mean_abs = retained["final_live"]["mean_abs"]
    # Every in-support row has |x| >= .949999. The sum of absolute values
    # therefore bounds how many rows could possibly lie on either pole.
    maximum_supported = min(12, math.floor(12 * mean_abs / .949999 + 1e-6))
    controls["develop-two_pole"]["retained_80_update_support_bound"] = dict(
        mean_abs=mean_abs, actual_particles=12, original_travel_status=retained["status"],
        source_result_sha256=retained["source_result_sha256"],
        maximum_supported_rows=maximum_supported, support_fraction_upper_bound=maximum_supported / 12,
        added_support_gate="FAIL" if maximum_supported / 12 < .95 else "UNRESOLVED",
        claim="An original travel PASS cannot be a density PASS when even the best possible support count allowed by its recorded mean is below the support gate.")
    states = torch.tensor(np.concatenate([rng.uniform(-1, 1, (256, 4)), rng.uniform(-1, 1, (256, 4))]))
    # Make exact reflection pairs, so both policies have *identical* empirical
    # output marginals. Only the paired objective can distinguish the sign.
    states[len(states) // 2:] = -states[:len(states) // 2]
    target = sign.expert_action(states).numpy()
    neutral = np.zeros_like(target)
    joint = np.concatenate([states.numpy(), target], 1)
    reflected_joint = -joint
    sorted_joint = joint[np.lexsort(joint.T[::-1])]
    sorted_reflected = reflected_joint[np.lexsort(reflected_joint.T[::-1])]
    controls["source-family-11"] = {
        "expert_correspondence": paired_edit_metrics(target, target, neutral),
        "wrong_sign_identical_output_marginal": paired_edit_metrics(-target, target, neutral),
        "same_empirical_marginal": bool(np.array_equal(np.sort(target, axis=0), np.sort(-target, axis=0))),
        "simultaneous_state_action_reflection_preserves_joint_law": bool(np.array_equal(sorted_joint, sorted_reflected)),
        "joint_law_limit": "A decoder allowed to reflect both state and action can match the joint data law while reversing the caller's physical action.",
    }
    previous = np.linspace(-1, 1, 512).reshape(-1, 1)
    action = np.tanh(-2.2 * previous)
    controls["source-family-13"] = {
        "analytic_paired_action": paired_edit_metrics(action, action, np.zeros_like(action)),
        "reversed_previous_command": paired_edit_metrics(action[::-1], action, np.zeros_like(action)),
        "same_empirical_action_marginal": bool(np.allclose(np.sort(action, axis=0), np.sort(action[::-1], axis=0))),
    }
    states = landing.initial_states(200, 1000)
    landing_controls = {}
    for name, sink in (("quick_soft", landing.QUICK_SINK), ("hover", 0.), ("crash_sink", landing.CRASH_SINK)):
        landed, crashed, steps = landing._hard_rollout(lambda x: landing.pd_action(x, sink), states)
        landing_controls[name] = landing_metrics(landed.numpy(), crashed.numpy(), steps.numpy())
    controls["source-family-12"] = landing_controls
    centers = 3 * np.stack([np.cos(np.arange(8) * math.pi / 4), np.sin(np.arange(8) * math.pi / 4)], 1)
    ring = np.repeat(centers, 512, axis=0) + .07 * rng.standard_normal((4096, 2))
    controls["source-family-10"] = {
        "balanced_ring": ring_metrics(ring),
        "eight_centers_no_width": ring_metrics(np.repeat(centers, 512, axis=0)),
        "frozen_before_target_shift": ring_metrics(ring, shift=(1., 0.)),
        "adapted_to_target_shift": ring_metrics(ring + [1., 0.], shift=(1., 0.)),
        "temporary_pass_late_failure": terminal_window([True] * 8 + [False]),
        "five_terminal_passes": terminal_window([False] * 4 + [True] * 5),
    }
    # The grouped routed entry needs both paired output and useful bank
    # intervention, plus its existing separate own-state/replay tests. These
    # controls freeze the missing direction contract; they are not trained data.
    controls["source-family-14"] = {
        "paired_target": paired_edit_metrics(target, target, neutral),
        "row_permutation": paired_edit_metrics(-target, target, neutral),
        "helpful_code_all_judges": useful_code_metrics([.70, .71, .72, .73], [.80, .81, .82, .83]),
        "harmful_code_all_judges": useful_code_metrics([.80, .81, .82, .83], [.70, .71, .72, .73]),
        "one_judge_disagrees": useful_code_metrics([.70] * 4, [.80, .80, .80, .69]),
    }
    paths = ["reports/toy_audit/catalog.json", "examples/five_modes.py", "examples/quickstart_gan.py", "benchmarks/locked_shared/two_pole.py",
             "benchmarks/locked_shared/mode_hold.py", "benchmarks/toy100/continuous_probe.py",
             "benchmarks/legacy/locked_shared.py", "lib/yue2_particle_toy.py", "lib/safe_fast_landing.py", "experiments/toy_particle_native_2d.py",
             "examples/e22_routed_paired.py", "examples/e22_routed_support.py",
             "examples/e22_routed_moving.py", "examples/e22_routed_replay.py", "benchmarks/toy_audit/definition_quality.py"]
    return dict(version=VERSION, scope="evaluator calibration controls; no training or convergence qualification",
                historical_verdicts_unchanged=True, controls=controls,
                source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
                evaluator_seeds=dict(numpy_controls=713, two_pole="deterministic source sampler; no RNG"),
                runtime=dict(numpy=np.__version__, torch=torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_controls(Path(__file__).resolve().parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"version": VERSION, "cases": len(result["controls"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
