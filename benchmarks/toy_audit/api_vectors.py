"""Public-API callers for the vector audit's mathematical target laws.

These are new callers, not replays of historical optimizer campaigns. Legacy
helpers supply data or fixed mathematical evaluators only. No legacy training
function, imported-source replacement, configuration mutation or metric-driven
optimizer decision is used. ``observe`` reads independent held-out draws.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

from particlegan import (BatchDistanceDiscriminator, GANTrainer, Recipe,
                        get_recipe, init, scale_learning_rates)
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from benchmarks.transfer_suite import stress_tasks, vector_tasks
from benchmarks.toy100 import accuracy, metrics as native_metrics, problems
from .definition_quality import gaussian_metrics, ring_metrics, two_pole_metrics
from .vector_quality import projection_ks

VERSION = "toy-public-api-vectors-v1"
DEFAULT_EVAL = 4096
PROJECTION_KS_MAX = .06

# Frozen mathematical proposal definitions and actual arm initializers are
# expanded below. These dictionaries contain no executable training code.
PROPOSAL_BASE_SPEC = {'kind': 'gaussian_mixture',
 'identifiable': True,
 'particles': 256,
 'z_dim': 4,
 'batch': 128,
 'steps': 1200,
 'hidden': 64,
 'layers': 2,
 'lr': 0.00425,
 'd_lr_mult': 1.0,
 'prior_lr_mult': 2.0,
 'prior_reg': 0.05,
 'betas': [0.0, 0.99],
 'ema_decay': 0.995,
 'thresholds': [['sw1_normalized', '<=', 0.18], ['mass_tv', '<=', 0.15], ['hq', '>=', 0.85],
                ['component_covariance_error', '<=', 0.85], ['component_min_eigen_ratio', '>=', 0.15]]}

PROPOSAL_DEFINITIONS = {45: {'spec': {'name': 'vector_two_broad_gauge_x32',
               'means': [[-32.0, 0.0], [32.0, 0.0]],
               'covariances': [[[64.0, 0.0], [0.0, 64.0]], [[64.0, 0.0], [0.0, 64.0]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_gauge_x32',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'batch_distance',
                               'width': 96,
                               'layers': 3,
                               'scales': [3.2, 8.0, 16.0, 32.0],
                               'init_std': 16.0}},
      'comparison_factors': {'kernel_scales_changed': True,
                             'prior_initialization_changed': True,
                             'individual_mechanism_identified': False}},
 152: {'spec': {'name': 'vector_binary_cluster_pairs_d24_pair4.0_local1.0',
                'means': [[-12.0, -2.0], [-12.0, 2.0], [12.0, -2.0], [12.0, 2.0]],
                'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                                [[1.0, 0.0], [0.0, 1.0]]],
                'masses': [0.25, 0.25, 0.25, 0.25]},
       'title': 'vector_binary_cluster_pairs_d24_pair4.0_local1.0',
       'profiles': {'published': {'kind': 'batch_distance',
                                  'width': 96,
                                  'layers': 3,
                                  'scales': [0.1, 0.25, 0.5, 1.0],
                                  'init_std': 0.5},
                    'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 6.0}},
       'comparison_factors': {'critic_architecture_changed': True,
                              'prior_initialization_changed': True,
                              'prior_init_std': [0.5, 6.0],
                              'individual_mechanism_identified': False}},
 57: {'spec': {'name': 'vector_boomerang_wide_r16',
               'means': [[-16.0, 8.0], [0.0, 16.0], [16.0, -2.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_boomerang_wide_r16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 53: {'spec': {'name': 'vector_chevron_wide_d16_local1.0',
               'means': [[-8.0, 8.0], [8.0, 8.0], [0.0, -16.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_chevron_wide_d16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 50: {'spec': {'name': 'vector_four_corner_wide_g8_local1.0',
               'means': [[-8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [8.0, 8.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_four_corner_wide_g8_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 5.656854249492381}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 5.656854249492381],
                             'individual_mechanism_identified': False}},
 51: {'spec': {'name': 'vector_hex_wide_side16_local1.0',
               'means': [[16.0, 0.0], [8.000000000000002, 13.856406460551018], [-7.9999999999999964, 13.85640646055102],
                         [-16.0, 1.959434878635765e-15], [-8.000000000000007, -13.856406460551014],
                         [7.999999999999989, -13.856406460551025]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
                          0.16666666666666666, 0.16666666666666666]},
      'title': 'vector_hex_wide_side16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 55: {'spec': {'name': 'vector_kite_wide_h16',
               'means': [[0.0, 16.0], [-10.0, 0.0], [10.0, 0.0], [0.0, -5.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_kite_wide_h16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 52: {'spec': {'name': 'vector_pentagon_wide_r16_local1.0',
               'means': [[9.797174393178826e-16, -16.0], [15.216904260722456, -4.944271909999158],
                         [9.40456403667957, 12.94427190999916], [-9.404564036679568, 12.94427190999916],
                         [-15.216904260722458, -4.9442719099991566]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.2, 0.2, 0.2, 0.2, 0.2]},
      'title': 'vector_pentagon_wide_r16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 54: {'spec': {'name': 'vector_tall_spike_wide_h16_local1.0',
               'means': [[-6.0, 0.0], [6.0, 0.0], [0.0, 16.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_tall_spike_wide_h16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 49: {'spec': {'name': 'vector_three_island_wide_d16_local1.0',
               'means': [[-8.0, -4.618802153517006], [8.0, -4.618802153517006], [0.0, 9.237604307034012]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]},
      'title': 'vector_three_island_wide_d16_local1.0',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 4.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 4.0],
                             'individual_mechanism_identified': False}},
 56: {'spec': {'name': 'vector_trapezoid_wide_h16',
               'means': [[-8.0, 16.0], [8.0, 16.0], [-16.0, 0.0], [16.0, 0.0]],
               'covariances': [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                               [[1.0, 0.0], [0.0, 1.0]]],
               'masses': [0.25, 0.25, 0.25, 0.25]},
      'title': 'vector_trapezoid_wide_h16',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 8.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 8.0],
                             'individual_mechanism_identified': False}},
 47: {'spec': {'name': 'vector_two_broad_gauge_x24_mlp_control',
               'means': [[-24.0, 0.0], [24.0, 0.0]],
               'covariances': [[[36.0, 0.0], [0.0, 36.0]], [[36.0, 0.0], [0.0, 36.0]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_gauge_x24_mlp_control',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 12.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 12.0],
                             'individual_mechanism_identified': False}},
 48: {'spec': {'name': 'vector_two_broad_wide_gap_x8',
               'means': [[-8.0, 0.0], [8.0, 0.0]],
               'covariances': [[[0.0625, 0.0], [0.0, 0.0625]], [[0.0625, 0.0], [0.0, 0.0625]]],
               'masses': [0.5, 0.5]},
      'title': 'vector_two_broad_wide_gap_x8',
      'profiles': {'published': {'kind': 'batch_distance',
                                 'width': 96,
                                 'layers': 3,
                                 'scales': [0.1, 0.25, 0.5, 1.0],
                                 'init_std': 0.5},
                   'control': {'kind': 'mlp', 'width': 64, 'layers': 2, 'fourier': 2, 'init_std': 4.0}},
      'comparison_factors': {'critic_architecture_changed': True,
                             'prior_initialization_changed': True,
                             'prior_init_std': [0.5, 4.0],
                             'individual_mechanism_identified': False}}}
for _proposal in PROPOSAL_DEFINITIONS.values():
    _proposal["spec"] = deepcopy(PROPOSAL_BASE_SPEC) | _proposal["spec"]



def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _law(spec):
    law = {key: deepcopy(spec[key]) for key in
            ("kind", "means", "covariances", "masses", "turns", "radius_min", "radius_max",
             "noise", "scale_start", "scale_end", "scale_ramp_end")
            if key in spec}
    if "scale_start" in spec:
        law["schedule_budget"] = spec["steps"]
    return law


def _case(identifier, legacy, title, goal, *, kind="vector", steps=1200,
          batch=128, evaluation=DEFAULT_EVAL, default_recipe="atlas", **extra):
    return dict(id=identifier, legacy_ids=list(legacy), title=title, goal=goal,
                kind=kind, default_steps=steps, batch_size=batch,
                eval_samples=evaluation, default_recipe=default_recipe,
                sampling="public served latent law; independent evaluation RNG; output_noise=False removes additive output noise only; DV12/feature-cell latent perturbation remains when enabled",
                scope="A per-observation gate is not full-budget or sustained convergence; no historical verdict is replaced.") | extra


def _registry():
    cases = {}
    for name in problems.PROBLEM_NAMES:
        identifier = "api-" + name
        cases[identifier] = _case(
            identifier, ["atlas-" + name], name,
            "Recover all 100 equal-weight Gaussian modes, their mass and local width, including independent density-fidelity bounds.",
            kind="native100", steps=7000, batch=2048, evaluation=20000,
            problem=name, particles=20000, z_dim=2,
            sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus the recipe's output noise; diagnostics remove additive output noise only",
            thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS)},
            law=dict(kind="equal100_gaussian", problem=name, sigma=.03))
    name = "api-rotated100-moving"
    cases[name] = _case(
        name, ["atlas-rotated100_moving"], "Rotating 100 modes",
        "Fit the full 100-mode law after two 30-degree target jumps, rather than preserve a low relative coverage baseline.",
        kind="native100", steps=1500, batch=2048, evaluation=20000,
        problem="rotated100", moving=True, particles=20000, z_dim=2,
        sampling="public served latent law, including enabled DV12/feature-cell perturbation, plus output noise; full gates use the current target frame",
        thresholds={"native": deepcopy(native_metrics.REQUIREMENTS), "accuracy": deepcopy(accuracy.LIMITS)},
        law=dict(kind="equal100_gaussian", problem="rotated100", sigma=.03,
                 extra_degrees=[0, 30, 60], update_intervals=[[1, 500], [501, 1000], [1001, 1500]]))
    for spec in vector_tasks.TASKS + vector_tasks.RESERVED + stress_tasks.TASKS + stress_tasks.RESERVED_TASKS:
        spec = vector_tasks.resolve(deepcopy(spec), allow_reserved=True)
        identifier = "api-" + spec["name"].replace("_", "-")
        legacy = "develop-" + spec["name"]
        alternating = spec.get("d_every", 1) == 2
        fixed_penalty = spec["name"] == "stress_r1_r2"
        goal = spec["importance_reason"]
        if spec["name"] == "stress_overlapping_data":
            goal = "Fit the observable broad eight-component law; analytic projected CDFs must reject a centers-only zero-width cloud. Latent labels are not recoverable observations."
        if spec["name"] == "reserved_alternating_critic_updates":
            goal = "Fit the same narrow eight-mode law when D updates on outer steps 1,3,5,... and G/prior update every step; preserve the unequal work budget."
        thresholds = deepcopy(spec["thresholds"]) + [["sample_count", ">=", DEFAULT_EVAL], ["projection_ks", "<=", PROJECTION_KS_MAX]]
        cases[identifier] = _case(
            identifier, [legacy], spec["name"], goal,
            steps=spec["steps"], batch=spec["batch"], default_recipe="ka2" if alternating or fixed_penalty else "atlas",
            spec=spec, particles=spec["particles"], z_dim=spec["z_dim"], thresholds=thresholds,
            law=_law(spec), law_sha256=_digest(_law(spec)),
            original_split=spec["split"], split="already-observed-audit-fixture",
            cadence=dict(d_every=spec.get("d_every", 1), g_every=spec.get("g_every", 1),
                         first_d_outer_step=1), caller_owned=alternating,
            caller_noise="none, preserving the original alternating-loop input/output sampling law" if alternating else None,
            adaptation=("Explicit KA2 public component loop; no Atlas/DV12 policy lifecycle is claimed. Original half-frequency D cadence is preserved."
                        if alternating else "KA2 named preset with explicit public R1+R2/K3P penalty; DV12 game-record support is unavailable for this fixed penalty."
                        if fixed_penalty else "Current selected public policy replaces the historical manual Adam/controller loop; target law and named stress remain fixed."),
            penalty_arm=spec["reg_arm"] if fixed_penalty else None,
            limitations=spec["limitations"])
    for pr, record in sorted(PROPOSAL_DEFINITIONS.items()):
        spec = deepcopy(record["spec"])
        for arm in ("published", "control"):
            identifier = f"api-pr{pr}-{arm}"
            profile = record["profiles"][arm]
            composite = "Joint kernel-scale/prior-initialization contrast" if pr == 45 else "Composite critic-architecture/prior-initialization contrast"
            cases[identifier] = _case(
                identifier, [f"pr{pr}-adapted"], f"PR{pr}: {record['title']} ({arm})",
                f"{composite} on this exact geometry: test each arm's mass, width and observable projected law. A passing arm alone does not identify an individual causal mechanism.",
                steps=spec["steps"], batch=spec["batch"], spec=spec,
                particles=spec["particles"], z_dim=spec["z_dim"], profile=deepcopy(profile),
                thresholds=deepcopy(spec["thresholds"]) + [["sample_count", ">=", DEFAULT_EVAL], ["projection_ks", "<=", PROJECTION_KS_MAX]],
                law=_law(spec), law_sha256=_digest(_law(spec)),
                comparison_group=f"api-pr{pr}", comparison_factors=deepcopy(record["comparison_factors"]),
                adaptation="Current public trainer/preset replaces historical source execution; both original composite arms remain registered, without a critic-only attribution.")
    cases["api-two-pole-grid12"] = _case(
        "api-two-pole-grid12", ["develop-two_pole"], "Twelve atoms on two poles",
        "Check six target offsets per pole in one full row-ID realization, beyond mere travel; full served output-law fidelity remains unmeasured when latent perturbation is active.",
        kind="two_pole", steps=80, batch=12, evaluation=12, particles=12, z_dim=1,
        thresholds=[["sample_count", "==", 12], ["mass_tv", "<=", .05], ["support_fraction", ">=", .95], ["max_grid_quantile_error_halfwidth", "<=", .10]],
        sampling="one realization enumerating all 12 prior row IDs; output_noise=False; enabled DV12 latent perturbation remains; this is not exact enumeration of the served output law",
        law=dict(kind="uniform12_atoms", offsets=np.linspace(-.05, .05, 6).tolist(), poles=[-1., 1.]),
        adaptation="Public GANTrainer needs a trainable generator: an identity-initialized affine host is added; initial prior remains twelve zeros. The fixed host critic weights and data law are retained.")
    cases["api-gaussian2d"] = _case(
        "api-gaussian2d", ["source-family-16"], "Gaussian and checkpoint continuation",
        "Fit the mean, full covariance and radial/projected law of N((1,1),.04I); checkpoint continuation must preserve trainer and data-stream state.",
        kind="gaussian", steps=1000, batch=2048, evaluation=4096, particles=20000, z_dim=2,
        thresholds=[["sample_count", ">=", 1024], ["mean_error_sigma", "<=", .10], ["min_cov_eigen", ">=", .85], ["max_cov_eigen", "<=", 1.15], ["radial_ks", "<=", .075], ["max_projection_ks", "<=", .06]],
        law=dict(kind="normal2d", mean=[1., 1.], covariance=[[.04, 0.], [0., .04]]),
        profile=dict(kind="batch_distance", width=96, layers=3, scales=[.1, .25, .5, 1.], init_std=1.),
        adaptation="Same quickstart data/host/resources and 1000-update default budget; current selected preset is explicit. Software continuation equality is separate from density convergence.")
    ring_bounds = [["sample_count", ">=", 4096], ["modes", "==", 8], ["hq", ">=", .90], ["mass_tv", "<=", .075], ["min_cov_eigen", ">=", .5], ["max_cov_eigen", "<=", 1.5], ["max_radial_ks", "<=", .10]]
    for suffix, steps, particles in (("acquire", 1200, 256), ("hold", 2400, 256), ("shift", 3600, 256), ("resolution12", 1200, 12)):
        identifier = "api-ring8-" + suffix
        cases[identifier] = _case(
            identifier, ["source-family-10", "develop-mode_hold"], "Narrow ring: " + suffix,
            {"acquire": "Acquire all eight equal-weight radius-three, sigma-.07 Gaussian modes, including their within-mode law.",
             "hold": "Acquire by update1200, then retain the same full eight-mode law without resetting optimizer state through update2400.",
             "shift": "After qualified acquisition/hold, adapt to a +1 x translation at update2401; require recovery by update2800 and retained width/mass through3600.",
             "resolution12": "Retain the original 12-row resource as a low-resource public-API control; assess the actual perturbed served law. A separate unperturbed twelve-equal-atom witness has a mass/width obstruction, which does not prove this stochastic served law impossible."}[suffix],
            kind="ring", steps=steps, batch=128, evaluation=4096, particles=particles, z_dim=4,
            phase=suffix, thresholds=deepcopy(ring_bounds), law=dict(kind="ring8", radius=3., sigma=.07, masses=[.125]*8, shift_update=2401 if suffix == "shift" else None, shift=[1., 0.] if suffix == "shift" else [0., 0.]),
            resource_change=dict(original_actual_rows=12, current_rows=particles, latent_dim=4, configured_20000_was_not_actual_resource=True),
            phase_prerequisites={"acquisition_step": 1200, "hold_step": 2400, "shift_step": 2401, "recovery_deadline": 2800},
            adaptation="Public GANTrainer host;256 rows are an explicit resource reform, not a rerating of the source's12-row run. Warm/cold restarts use identical bound checkpoints; causal shift qualification also needs the matched frozen checkpoint control.")
    return cases


def list_cases():
    """Independent metadata copies; default budgets do not follow short overrides."""
    return [deepcopy(record) for record in _registry().values()]


def _bounds(metrics, bounds):
    failed = []
    for name, operator, bound in bounds:
        value = metrics.get(name)
        valid = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(float(value))
        passed = valid and ((value <= bound) if operator == "<=" else (value >= bound) if operator == ">=" else value == bound)
        if not passed:
            failed.append(f"{name} {operator} {bound}")
    return failed


def _scalar_metrics(result):
    # Optional undefined moments stay absent and fail their explicit bound;
    # do not fabricate a numerical value to make a malformed cloud look valid.
    return {key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in result.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))}


def _empirical_projection_ks(points, reference):
    """32 predeclared two-sample CDF tests for non-Gaussian continuous laws."""
    points, reference = (np.asarray(value, dtype=np.float64) for value in (points, reference))
    values = []
    for angle in np.arange(32) * math.pi / 32:
        direction = np.array([math.cos(angle), math.sin(angle)])
        left, right = np.sort(points @ direction), np.sort(reference @ direction)
        at = np.concatenate([left, right])
        values.append(float(np.max(np.abs(np.searchsorted(left, at, side="right") / len(left)
                                         - np.searchsorted(right, at, side="right") / len(right)))))
    return max(values)


def score_case(metadata, samples, completed_steps=0):
    """Pure evaluator for supplied held-out samples; no trainer or optimizer."""
    points = torch.as_tensor(samples).detach().to(device="cpu", dtype=torch.float32)
    if points.ndim != 2 or len(points) == 0 or not torch.isfinite(points).all():
        raise ValueError("the vector gate needs nonempty, finite row-major points")
    kind = metadata["kind"]
    if kind == "native100":
        if points.shape[1] != 2:
            raise ValueError("100-mode points need two coordinates")
        theta = _native_angle(metadata, completed_steps)
        points = points @ _rotation(theta)
        original = native_metrics.evaluate_samples(points, metadata["problem"])
        exact = accuracy.fidelity_metrics(points, metadata["problem"])
        metrics = {**_scalar_metrics(original), **{"accuracy_" + key: value for key, value in _scalar_metrics(exact).items()}}
        failed = []
        if not native_metrics.passes(metadata["problem"], original):
            failed.extend(_bounds(metrics, _native_bounds()))
            if not failed:
                failed.append("native metric consistency")
        failed.extend(_bounds(metrics, [["accuracy_" + key, "<=", value] for key, value in accuracy.LIMITS.items()]))
        if not accuracy.passes_accuracy(exact) and not any(x.startswith("accuracy_") for x in failed):
            failed.append("accuracy sample validity")
    elif kind == "vector":
        if points.shape[1] != 2:
            raise ValueError("vector points need two coordinates")
        spec = metadata["spec"]
        metrics = _scalar_metrics(vector_tasks.score_samples(points, spec, completed_steps))
        if spec["kind"] == "gaussian_mixture":
            metrics["projection_ks"] = projection_ks(points, spec, completed_steps)
        else:
            reference = vector_tasks.sample_target(spec, 8192, torch.Generator().manual_seed(89213), completed_steps)
            metrics["projection_ks"] = _empirical_projection_ks(points.numpy(), reference.numpy())
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "two_pole":
        if points.shape[1] != 1:
            raise ValueError("two-pole samples need one coordinate")
        metrics = _scalar_metrics(two_pole_metrics(points.numpy()))
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "gaussian":
        result = gaussian_metrics(points.numpy())
        eig = np.asarray(result["covariance_eigenvalues"])
        metrics = _scalar_metrics(result) | dict(min_cov_eigen=float(eig.min()), max_cov_eigen=float(eig.max()))
        failed = _bounds(metrics, metadata["thresholds"])
    elif kind == "ring":
        shift = (1., 0.) if metadata["phase"] == "shift" and completed_steps >= 2401 else (0., 0.)
        result = ring_metrics(points.numpy(), shift=shift)
        eig = np.asarray(result["covariance_eigenvalues"])
        metrics = _scalar_metrics(result) | dict(min_cov_eigen=float(eig.min()), max_cov_eigen=float(eig.max()), target_shift_x=shift[0])
        failed = _bounds(metrics, metadata["thresholds"])
    else:
        raise ValueError("unsupported mathematical vector kind")
    return dict(metrics=metrics, passed=not failed, failed_bounds=failed)


def _native_bounds():
    return [["n", ">=", 20000], ["modes", ">=", 100], ["precision", ">=", .97],
            ["min_hq_mode_mass", ">=", .005], ["mass_tv", "<=", .10], ["max_mode_mass", "<=", .02],
            ["min_cov_eig_ratio", ">=", .4], ["max_cov_eig_ratio", "<=", 1.7],
            ["min_radial_median_ratio", ">=", .65], ["max_radial_median_ratio", "<=", 1.4]]


def _native_angle(case, step):
    # Completed update500 used old target; update501 is the first changed batch.
    return math.radians(30) * ((max(1, step) - 1) // 500) if case.get("moving") else 0.


def _rotation(theta, *, device="cpu"):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)


def _target(case, n, rng, step, *, device="cpu"):
    kind = case["kind"]
    if kind == "native100":
        return problems.sample_real(case["problem"], n, device=device, generator=rng) @ _rotation(_native_angle(case, step), device=device).T
    if kind == "vector":
        # Mathematical helper accepts CPU generators; the caller moves only data.
        return vector_tasks.sample_target(case["spec"], n, rng, step).to(device)
    if kind == "gaussian":
        return 1 + .2 * torch.randn(n, 2, generator=rng, device=device)
    if kind == "two_pole":
        target = torch.cat([torch.linspace(-1.05, -.95, 6), torch.linspace(.95, 1.05, 6)])[:, None]
        # The actual12-row training batch is deterministic, not a larger
        # real_batch(n) law. Plot-only requests repeat this same finite table.
        return target[torch.arange(n) % 12].to(device)
    angles = torch.arange(8, device=device) * math.pi / 4
    centers = 3 * torch.stack([angles.cos(), angles.sin()], 1)
    points = centers[torch.randint(8, (n,), generator=rng, device=device)] + .07 * torch.randn(n, 2, generator=rng, device=device)
    if case["phase"] == "shift" and step >= 2401:
        points = points + points.new_tensor([1., 0.])
    return points


class VectorFixture:
    """Caller-owned data/phase state around one current public trainer."""
    api_components = ("particlegan.Recipe", "particlegan.GANTrainer", "particlegan.ParticlePrior")

    def __init__(self, case, *, device="cpu", seed=24002, recipe_name="atlas", max_steps=None, recipe_overrides=None):
        self.metadata = deepcopy(case)
        self.case_id = case["id"]
        self.device = torch.device(device)
        self.seed = seed
        self.completed_steps = 0
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        if type(self.execution_steps) is not int or not 1 <= self.execution_steps <= case["default_steps"]:
            raise ValueError("max_steps must be a positive prefix of the declared full budget")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if recipe_name == "auto":
            recipe_name = case["default_recipe"]
        if case.get("caller_owned") and recipe_name != "ka2":
            raise ValueError("the half-frequency D caller is explicitly KA2; use recipe_name='ka2' or 'auto'")
        if case.get("penalty_arm") and recipe_name not in ("ka2", "gan", "k3p"):
            raise ValueError("the R1+R2 stress requires a fixed public preset: its critic record lacks the Atlas/DV12 game-surprise fields")
        spec = case.get("spec", {})
        options = dict(num_particles=case["particles"], z_dim=case["z_dim"], batch_size=case["batch_size"])
        for source, destination in (("lr", "lr"), ("d_lr_mult", "d_lr_mult"), ("prior_lr_mult", "prior_lr_mult"),
                                    ("prior_reg", "prior_reg"), ("betas", "betas"), ("ema_decay", "ema_decay")):
            if source in spec:
                options[destination] = tuple(spec[source]) if source == "betas" else spec[source]
        # Explicit named penalty stress stays its actual R1+R2 law. Other
        # fixtures adopt the requested public formulation, not a hidden b_cap.
        if case.get("penalty_arm"):
            options.update(reg_arm=case["penalty_arm"], reg_coeff=spec["reg_coeff"])
        if recipe_name in ("ka2", "k3p", "gan"):
            options["total_steps"] = case["default_steps"]
        if case.get("caller_owned"):
            options.update(input_noise_std=0., output_noise_std=0., output_noise_mode="fixed")
        from .api_contract import validate_recipe_overrides
        self.recipe_overrides = validate_recipe_overrides({**case, "provider": "api_vectors"}, recipe_name, recipe_overrides)
        options.update(self.recipe_overrides)
        self.recipe = get_recipe(recipe_name, **options)
        data_device = "cpu" if case["kind"] in ("vector", "two_pole") else self.device
        self.data_rng = torch.Generator(device=data_device).manual_seed(seed)
        cuda = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed)
            with torch.device(self.device):
                prior = self.recipe.make_prior(generator=torch.Generator(device=self.device).manual_seed(seed),
                                               init_std=case.get("profile", {}).get("init_std", .5))
                if case["kind"] == "native100":
                    with torch.no_grad():
                        prior.z.uniform_(-5, 5, generator=torch.Generator(device=self.device).manual_seed(seed))
                    generator = nn.Linear(2, 2)
                    with torch.no_grad():
                        generator.weight.copy_(torch.eye(2, device=self.device)); generator.bias.zero_()
                    critic = SimpleMLPDiscriminator(2, 128, 3, 3)
                elif case["kind"] == "two_pole":
                    from benchmarks.locked_shared.two_pole import HostCritic
                    generator = nn.Linear(1, 1)
                    with torch.no_grad():
                        generator.weight.fill_(1); generator.bias.zero_(); prior.z.zero_()
                    critic = HostCritic()
                else:
                    width = 96 if case["kind"] == "ring" else spec.get("hidden", 64)
                    depth = 3 if case["kind"] == "ring" else spec.get("layers", 2)
                    generator = SimpleMLPGenerator(self.recipe.z_dim, width, depth, 2)
                    profile = case.get("profile", {})
                    if profile.get("kind") == "batch_distance":
                        critic = BatchDistanceDiscriminator(2, profile["width"], profile["layers"], scales=tuple(profile["scales"]), beta=6.)
                    else:
                        critic = SimpleMLPDiscriminator(2, profile.get("width", spec.get("d_hidden", width)),
                                                        profile.get("layers", spec.get("d_layers", depth)),
                                                        profile.get("fourier", 3 if case["kind"] == "ring" else spec.get("fourier", 2)))
                    init.deterministic_orthogonal_(generator, seed=seed)
                if case["kind"] != "two_pole":
                    init.deterministic_orthogonal_(critic, seed=seed + 1)
        if case.get("caller_owned"):
            self._make_caller_loop(generator, critic, prior)
        else:
            self.trainer = GANTrainer(self.recipe, generator, critic, prior=prior, seed=seed,
                                      max_steps=self.execution_steps, serial_backward=True,
                                      optimizer_options={"foreach": False, "fused": False})
        self.phase_receipts = {}

    def _make_caller_loop(self, generator, critic, prior):
        self.api_components = ("particlegan.Recipe.make_optimizers", "particlegan.GANLoss", "particlegan.Recipe.make_critic_penalty", "particlegan.ParticlePrior", "particlegan.scale_learning_rates")
        self.generator, self.critic, self.prior = generator, critic, prior
        self.ema_critic = deepcopy(critic)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(generator, critic, prior, ema_critic=self.ema_critic, foreach=False, fused=False)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d)
        self.regularizer = self.recipe.make_prior_regularizer()
        self.latent_rng = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        self.base_rates = [[group["lr"] for group in optimizer.param_groups] for optimizer in (self.opt_g, self.opt_d)]
        self.update_counts = dict(g=0, d=0)

    def step(self):
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared execution prefix is complete")
        next_step = self.completed_steps + 1
        # Original continuation prerequisite: no failed acquisition is extended
        # to hunt for a pass. These reads cannot alter recipe/optimizer values.
        prerequisite_steps = (1200, 2400, 2800) if self.metadata.get("phase") == "shift" else (1200,)
        if self.metadata["kind"] == "ring" and self.metadata["phase"] in ("hold", "shift") and self.completed_steps in prerequisite_steps:
            result = self.observe(n=4096, seed=92431 + self.completed_steps)
            self.phase_receipts[str(self.completed_steps)] = {key: deepcopy(result[key]) for key in ("metrics", "passed", "failed_bounds")}
            if not result["passed"]:
                raise RuntimeError(f"scientific prerequisite failed at update{self.completed_steps}: {result['failed_bounds']}")
        real = _target(self.metadata, self.recipe.batch_size, self.data_rng, next_step, device=self.device)
        real_g = lambda: _target(self.metadata, self.recipe.batch_size, self.data_rng, next_step, device=self.device)
        if self.metadata.get("caller_owned"):
            scale_learning_rates(next_step - 1, self.recipe, (self.opt_g, self.opt_d), self.base_rates, self.prior)
            if (next_step - 1) % 2 == 0:
                fake = self.generator(self.prior.sample(self.recipe.batch_size, generator=self.latent_rng)[0]).detach()
                self.opt_d.zero_grad(set_to_none=True)
                value = self.loss.d_loss(self.critic(real), self.critic(fake)) + self.penalty(self.critic, real, fake)
                value.backward(); self.opt_d.step(); self.update_counts["d"] += 1
            self.critic.requires_grad_(False)
            try:
                self.opt_g.zero_grad(set_to_none=True)
                fake = self.generator(self.prior.sample(self.recipe.batch_size, generator=self.latent_rng)[0])
                value = self.loss.g_loss(self.critic(fake), self.critic(real_g())) + self.regularizer(self.prior.z)
                value.backward(); self.opt_g.step(); self.update_counts["g"] += 1
            finally:
                self.critic.requires_grad_(True)
            statistics = dict(step=next_step, **self.update_counts)
        else:
            with torch.autograd.set_multithreading_enabled(False):
                statistics = self.trainer.step(real, generator_real=real_g)
        self.completed_steps = next_step
        return statistics

    @torch.no_grad()
    def observe(self, n=1024, seed=713):
        if type(n) is not int or n <= 0 or type(seed) is not int or seed < 0:
            raise ValueError("held-out count and seed must be positive/nonnegative integers")
        # Caller must request the registered evaluation count to satisfy a
        # minimum-size gate. A smaller diagnostic cannot earn a full gate PASS.
        actual_n = 12 if self.metadata["kind"] == "two_pole" else n
        sample_rng = torch.Generator(device=self.device).manual_seed(seed)
        output_noise = self.metadata["kind"] == "native100"
        if self.metadata.get("caller_owned"):
            prior = self.prior
            samples = self.generator(prior.sample(actual_n, generator=sample_rng)[0])
        else:
            samples = self.trainer.sample(actual_n, generator=sample_rng, output_noise=output_noise,
                                          fixed_first_n=self.metadata["kind"] == "two_pole")
        samples = samples.detach().cpu()
        scored = score_case(self.metadata, samples, self.completed_steps)
        rng_device = "cpu" if self.metadata["kind"] in ("vector", "two_pole") else self.device
        target_rng = torch.Generator(device=rng_device).manual_seed(seed + 100003)
        target = _target(self.metadata, actual_n, target_rng, self.completed_steps, device=self.device).detach().cpu()
        views = [_view("Whole target law", target, samples, self.metadata)]
        if self.metadata["kind"] == "native100":
            clean = self.trainer.sample(actual_n, generator=torch.Generator(device=self.device).manual_seed(seed), output_noise=False).cpu()
            clean_score = score_case(self.metadata, clean, self.completed_steps)
            scored["metrics"].update({"clean_" + key: value for key, value in clean_score["metrics"].items()})
            scored["metrics"]["clean_gate_passed"] = int(clean_score["passed"])
            scored["metrics"]["output_sigma"] = float(self.trainer.output_sigma())
            views.append(_view("Output noise off; served latent law retained", target, clean, self.metadata))
        centers = self._centers()
        if centers is not None and samples.shape[1] == 2:
            labels = torch.cdist(samples, centers).argmin(1)
            target_labels = torch.cdist(target, centers).argmin(1)
            masses = torch.bincount(labels, minlength=len(centers)).float() / len(samples)
            reference_masses = torch.bincount(target_labels, minlength=len(centers)).float() / len(target)
            views.append(dict(kind="bar", title="All mode masses (nearest-center diagnostic)", target=reference_masses, samples=masses,
                              xlabel="mode index", ylabel="probability", caption="Target centers are used only by the observer; overlapping mixtures do not identify hidden component labels."))
            sigma = self._zoom_sigma()
            # Fixed mode0, not an adaptively selected good-looking component.
            views.append(dict(kind="scatter", title="Fixed mode0: local width", target=target, samples=samples,
                              xlim=[float(centers[0, 0] - 4 * sigma), float(centers[0, 0] + 4 * sigma)],
                              ylim=[float(centers[0, 1] - 4 * sigma), float(centers[0, 1] + 4 * sigma)],
                              caption="This local panel is illustrative; the binary gate evaluates all specified modes."))
        scored["views"] = views
        return scored

    def _centers(self):
        if self.metadata["kind"] == "native100":
            centers = problems.evaluation_geometry(self.metadata["problem"])[0]
            return centers @ _rotation(_native_angle(self.metadata, self.completed_steps)).T
        if self.metadata["kind"] == "vector" and self.metadata["spec"]["kind"] == "gaussian_mixture":
            return torch.tensor(self.metadata["spec"]["means"]) * vector_tasks.target_scale(self.metadata["spec"], self.completed_steps)
        if self.metadata["kind"] == "ring":
            angles = torch.arange(8) * math.pi / 4
            points = 3 * torch.stack([angles.cos(), angles.sin()], 1)
            return points + points.new_tensor([1., 0.]) if self.metadata["phase"] == "shift" and self.completed_steps >= 2401 else points
        return None

    def _zoom_sigma(self):
        if self.metadata["kind"] == "native100":
            return .03
        if self.metadata["kind"] == "ring":
            return .07
        spec = self.metadata["spec"]
        covariance = np.asarray(spec["covariances"])[0]
        return float(np.sqrt(np.linalg.eigvalsh(covariance).max()) * vector_tasks.target_scale(spec, self.completed_steps))

    def state_dict(self):
        state = dict(version=VERSION, case_id=self.case_id, recipe=self.recipe.to_dict(),
                     completed_steps=self.completed_steps, data_rng=self.data_rng.get_state().clone(),
                     phase_receipts=deepcopy(self.phase_receipts))
        if self.metadata.get("caller_owned"):
            state["caller"] = deepcopy(dict(generator=self.generator.state_dict(), critic=self.critic.state_dict(), prior=self.prior.state_dict(),
                                              ema_critic=self.ema_critic.state_dict(), opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(),
                                              latent_rng=self.latent_rng.get_state(), update_counts=self.update_counts))
        else:
            state["trainer"] = self.trainer.state_dict()
        return state

    def load_state_dict(self, state):
        if state["version"] != VERSION or state["case_id"] != self.case_id or state["recipe"] != self.recipe.to_dict():
            raise ValueError("checkpoint identity/recipe differs")
        steps = state["completed_steps"]
        if type(steps) is not int or steps < 0 or steps > self.execution_steps:
            raise ValueError("checkpoint exceeds this execution prefix")
        if self.metadata.get("caller_owned"):
            if state["caller"]["update_counts"] != dict(g=steps, d=(steps + 1) // 2):
                raise ValueError("checkpoint cadence counters differ from its phase clock")
        elif state["trainer"]["completed_steps"] != steps:
            raise ValueError("checkpoint trainer differs from its phase clock")
        if self.metadata.get("caller_owned"):
            for key in ("generator", "critic", "prior", "ema_critic", "opt_g", "opt_d"):
                getattr(self, key).load_state_dict(state["caller"][key])
            self.latent_rng.set_state(state["caller"]["latent_rng"])
            self.update_counts = deepcopy(state["caller"]["update_counts"])
        else:
            self.trainer.load_state_dict(state["trainer"])
        self.data_rng.set_state(state["data_rng"])
        self.phase_receipts = deepcopy(state["phase_receipts"])
        self.completed_steps = state["completed_steps"]


def _view(title, target, samples, metadata):
    if samples.shape[1] == 1:
        target = torch.cat([target, torch.zeros_like(target)], 1)
        samples = torch.cat([samples, torch.zeros_like(samples)], 1)
    return dict(kind="scatter", title=title, target=target, samples=samples,
                xlabel="x", ylabel="y", caption=metadata["sampling"])


def build_case(id, *, device="cpu", seed=24002, recipe_name="atlas", max_steps=None, recipe_overrides=None):
    cases = _registry()
    if id not in cases:
        raise ValueError(f"unknown vector case {id!r}")
    return VectorFixture(cases[id], device=device, seed=seed, recipe_name=recipe_name,
                         max_steps=max_steps, recipe_overrides=recipe_overrides)
