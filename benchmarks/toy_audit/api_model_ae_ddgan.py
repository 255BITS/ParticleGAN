"""Learned particle AE plus actual public DDGAN transitions and reverse chains.

Paired reconstruction holds the encoder's code through all reverse steps.
Independent generation redraws prior/kernel/reverse noise at every step, exactly
as in the separately registered DDGAN task. Neither cohort stands in for the other.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn

from experiments.forge.rng import NamedStreams
from .api_model_ddgan import DDGANFixture, list_cases as ddgan_cases, task_recipe
from .reproducibility import DEFAULT_SEED

VERSION = "api-model-ae-ddgan-v1"
PAIRED_BOUNDS = [["paired_nmse", "<=", .10],
    ["paired_worst_axis_nrmse", "<=", .25],
    ["paired_shuffled_gain", ">=", 3.], ["paired_reverse_steps", "==", 4]]
_CASES = []
for _parent in ddgan_cases():
    _case = deepcopy(_parent)
    _case.update(id=_parent["id"].replace("api-model-ddgan-", "api-model-ae-ddgan-"),
        legacy_ids=["ae_ddgan-public-composition", "denoising-toy-full-chain"],
        title="AE-DDGAN " + _parent["target"] + " paired reconstruction and independent chain",
        goal="Reconstruct held-out inputs through a learned particle AE and four public DDGAN reverse steps while preserving independent generation and conditional-transition quality",
        kind="ae_ddgan_chain", encoder_mode="ae",
        scope="New scoped public ae_ddgan composition: same analytic target, schedule, G/D architecture and prior as the DDGAN task, plus a learned 2x32 encoder. Clean paired reconstruction and independent-prior generation are separate cohorts. Historical denoising results retain their identities.")
    _case["thresholds"]["paired"] = deepcopy(PAIRED_BOUNDS)
    _case["forge_thresholds"] += deepcopy(PAIRED_BOUNDS)
    _case["sampling"] += "; separate paired cohort: encode held-out clean input once, hold its deterministic AE code through all four reverse steps from independent N(0,I), fresh Gaussian reverse noise each step; shuffled-code control shares identical start and reverse noise"
    _CASES.append(_case)
CASES = {case["id"]: case for case in _CASES}


def list_cases():
    return deepcopy(_CASES)


class ParticleEncoder(nn.Module):
    def __init__(self, z_dim=4, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.LeakyReLU(.2),
            nn.Linear(width, width), nn.LeakyReLU(.2), nn.Linear(width, 2 * z_dim))

    def forward(self, clean):
        return self.net(clean).chunk(2, dim=1)


def paired_chain(generator, schedule, codes, *, start_rng, reverse_rng, return_states=False):
    """Actual four-step DDGAN reconstruction; only codes carry input identity."""
    xt = torch.randn((len(codes), 2), generator=start_rng, device=codes.device, dtype=codes.dtype)
    states = [xt.clone()]
    for step in range(schedule.steps, 0, -1):
        t = torch.full((len(codes),), step, dtype=torch.long, device=codes.device)
        clean = generator(xt, codes, t)
        eta = torch.randn(xt.shape, generator=reverse_rng, device=xt.device, dtype=xt.dtype)
        xt = schedule.reverse(clean, xt, t, eta)
        states.append(xt.clone())
    return (xt, states) if return_states else xt


def paired_metrics(clean, reconstructed, shuffled, target):
    """Whiten by exact target covariance; test sample pairing and code use."""
    if clean.shape != reconstructed.shape or clean.shape != shuffled.shape or clean.ndim != 2 or clean.shape[1] != 2:
        raise ValueError("paired observations must be matching [N,2] arrays")
    points = [value.detach().cpu().double() for value in (clean, reconstructed, shuffled)]
    if not all(torch.isfinite(value).all() for value in points):
        return dict(paired_nmse=math.inf, paired_worst_axis_nrmse=math.inf,
                    paired_shuffled_gain=0.), ["paired_nonfinite"]
    means, var, weights = (value.double() for value in (target.means, target.variances, target.weights))
    mean = (weights[:, None] * means).sum(0)
    centered = means - mean
    covariance = torch.diag((weights[:, None] * var).sum(0)) + (weights[:, None] * centered).T @ centered
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    whitening = vectors @ torch.diag(eigenvalues.rsqrt()) @ vectors.T
    error = (points[1] - points[0]) @ whitening
    shuffled_error = (points[2] - points[0]) @ whitening
    nmse = float(error.square().mean())
    metrics = dict(paired_nmse=nmse,
        paired_worst_axis_nrmse=float(error.square().mean(0).sqrt().max()),
        paired_shuffled_gain=float(shuffled_error.square().mean()) / max(nmse, 1e-8))
    failures = [name for name, op, bound in PAIRED_BOUNDS[:3]
        if not math.isfinite(metrics[name]) or not (metrics[name] <= bound if op == "<=" else metrics[name] >= bound)]
    return metrics, failures


class AEDDGANFixture(DDGANFixture):
    api_components = DDGANFixture.api_components + ("Recipe.encode", "ParticleEncoding.reconstruction_loss")

    def make_encoder(self):
        return ParticleEncoder(self.recipe.z_dim)

    def encode(self, clean):
        query, offset = self.E(clean)
        return self.recipe.encode(query, self.prior, offset=offset)

    def extra_generator_objective(self, clean, xt, t):
        self.E.train()
        encoding = self.encode(clean)
        self.bind_prior_rows(encoding.indices)
        predicted = self.G(xt, encoding.codes[:, 0], t)
        reconstruction = encoding.reconstruction_loss(predicted[:, None], clean)
        weighted = self.recipe.reconstruction_weight * reconstruction
        return weighted, (weighted,), dict(reconstruction_mse=float(reconstruction.detach()))

    @torch.no_grad()
    def observe(self, n=2048, seed=10000):
        modes = {module: module.training for model in self._models().values() for module in model.modules()}
        try:
            self.E.eval()
            record = super().observe(n=n, seed=seed)
            self.G.eval()
            evaluation = NamedStreams(int(seed), device=self.device)
            clean = self.target.sample(n, evaluation.generator("eval", component="paired_target", device="cpu")).to(self.device)
            encoding = self.encode(clean)
            start = evaluation.generator("eval", component="paired_start", device=self.device)
            reverse = evaluation.generator("eval", component="paired_reverse", device=self.device)
            start_state, reverse_state = start.get_state(), reverse.get_state()
            reconstructed, states = paired_chain(self.G, self.schedule, encoding.codes[:, 0],
                start_rng=start, reverse_rng=reverse, return_states=True)
            start.set_state(start_state); reverse.set_state(reverse_state)
            shuffled = paired_chain(self.G, self.schedule, encoding.codes[:, 0].roll(1, 0),
                start_rng=start, reverse_rng=reverse)
            metrics, failures = paired_metrics(clean, reconstructed, shuffled, self.target)
            metrics["paired_reverse_steps"] = len(states) - 1
            if metrics["paired_reverse_steps"] != 4:
                failures.append("paired_reverse_steps")
            record["metrics"].update(metrics)
            record["failed_bounds"] = sorted(set(record["failed_bounds"] + failures))
            record["passed"] = not record["failed_bounds"]
            record["views"].append(dict(kind="scatter", title="Paired four-step reconstruction",
                target=clean.cpu(), samples=reconstructed.cpu(),
                caption="Held-out input -> learned AE code -> four actual reverse steps. Independent-prior chain is scored in the first panel."))
            # The pairing plot prevents a correct marginal from concealing wrong
            # input/output pairs in the actual-training GIF.
            desired = torch.stack((clean[:, 0], clean[:, 0]), 1).cpu()
            actual = torch.stack((clean[:, 0], reconstructed[:, 0]), 1).cpu()
            record["views"].append(dict(kind="scatter", title="Input x versus reconstructed x",
                target=desired, samples=actual,
                caption=f"Desired diagonal; code shuffle uses identical chain noise. NMSE={metrics['paired_nmse']:.3g}; shuffle gain={metrics['paired_shuffled_gain']:.3g}."))
            return record
        finally:
            for module, mode in modes.items():
                module.training = mode


def build_case(case_id, *, device="cpu", seed=DEFAULT_SEED, recipe_name=None, max_steps=None, recipe=None):
    if case_id not in CASES:
        raise ValueError("unknown AE-DDGAN API toy: " + case_id)
    if type(seed) is not int or seed != 0:
        raise ValueError("this prospective protocol is bound to seed 0")
    return AEDDGANFixture(CASES[case_id], device=device, seed=seed, recipe_name=recipe_name,
                         max_steps=max_steps, recipe=recipe)
