"""A two-coordinate CPU reproducer for an unsafe native generator LR release.

    python -u examples/e22_stiff_game_reopen.py

The fixed nonlinear critic is a CONSTRUCTED feature-score snapshot, not a
trained critic. The generator, Adam/AMSGrad, RpGAN loss and SettleTest are the
public/native implementations. The explicit settled optimizer snapshot is
specified in local spectral units; it was not extracted from Supra.

This isolates a mechanism, not all of Supra: a weak coherent direction can
dominate displacement cosines after a stiff direction has settled. Raising
the common LR makes the stiff direction unstable. Every-two-step blocks can
then alias its alternating updates as positive coherence again.

No particles, routing, KA2, structural proposals or stochastic inputs are
needed for this native-generator-controller unit reproducer. No output-MSE
metric supplies training, guarding, checkpoint choice or its stability oracle.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math

import torch
from torch import nn

from particlegan import get_recipe, init
from particlegan.continuous import SettleTest


HORIZON = 96
CANCEL_STEP = 48
BASE_LR = .00425
SETTLED_SCALE = .5
PAST_ADAM_STEPS = 1000
CONTRACTED_STIFF_FACTOR = 1.6
FEATURE_SCALES = (1000., 1e-6)
INITIAL_RESIDUAL = (1e-8, 1.)
GAME_TOLERANCE = 1e-6


class ConstructedFeatureCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 2, bias=False, dtype=torch.float64, device="cpu")

    def forward(self, residual):
        # RpGAN compares these scalar scores; the generator never optimizes an
        # independently computed output-MSE metric.
        return -self.features(residual).square().sum(-1, keepdim=True)


@dataclass
class Fixture:
    generator: nn.Linear
    critic: ConstructedFeatureCritic
    optimizer: torch.optim.Optimizer
    tester: SettleTest
    loss: object
    contracted_stiff_factor: float

    def game(self):
        residual = self.generator(self.generator.bias.new_zeros((1, 1)))
        fake = self.critic(residual)
        real = self.critic(torch.zeros_like(residual))
        return self.loss.g_loss(fake, real)

    def state_dict(self):
        return deepcopy({"fixture": {"schema": 1, "contracted_stiff_factor": self.contracted_stiff_factor},
                         "generator": self.generator.state_dict(),
                         "critic": self.critic.state_dict(),
                         "optimizer": self.optimizer.state_dict(),
                         "tester": self.tester.state_dict()})

    def load_state_dict(self, state):
        if state["fixture"] != {"schema": 1, "contracted_stiff_factor": self.contracted_stiff_factor}:
            raise ValueError("the constructed game/memory geometry must match the checkpoint")
        self.generator.load_state_dict(state["generator"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.tester.load_state_dict(state["tester"], self.generator.bias.numel())


def make_fixture(*, contracted_stiff_factor=CONTRACTED_STIFF_FACTOR):
    if contracted_stiff_factor not in (.8, 1.6):
        raise ValueError("the two specified geometries use spectral factors .8 and 1.6")
    # Constructor RNG is isolated; public initialization itself consumes none.
    # The known parameter/memory snapshot is loaded AFTER API initialization.
    with torch.random.fork_rng(devices=[]):
        generator = nn.Linear(1, 2, dtype=torch.float64, device="cpu")
        critic = ConstructedFeatureCritic()
    init.deterministic_orthogonal_(generator, seed=0)
    init.deterministic_orthogonal_(critic, seed=1)
    generator.load_state_dict({"weight": torch.zeros((2, 1), dtype=torch.float64, device="cpu"),
                               "bias": torch.tensor(INITIAL_RESIDUAL, dtype=torch.float64, device="cpu")})
    critic.features.load_state_dict({"weight": torch.diag(torch.tensor(FEATURE_SCALES,
                                                                      dtype=torch.float64, device="cpu"))})
    generator.weight.requires_grad_(False)
    critic.requires_grad_(False)
    recipe = get_recipe("e22", lr=BASE_LR)
    optimizer = recipe.make_generator_optimizer([generator.bias], foreach=False)
    group = optimizer.param_groups[0]
    assert group["amsgrad"] and group["betas"] == (0., .999)
    correction = 1. - group["betas"][1] ** PAST_ADAM_STEPS
    stiff_curvature = FEATURE_SCALES[0] ** 2
    # Near the settled point, RpGAN softplus has sigmoid(0)=1/2, so the
    # negative squared-feature score has local Hessian diag(feature_scale²).
    # Choose the existing Adam memory in dimensionless spectral units:
    #   contracted LR * stiff curvature / corrected Adam denominator = 1.6
    #   released LR   * stiff curvature / corrected Adam denominator = 3.2
    # Euler's local stability interval is (0,2); these straddle it. This is a
    # valid specified optimizer snapshot, not a claim about its past dataset.
    denominator = BASE_LR * SETTLED_SCALE * stiff_curvature / contracted_stiff_factor
    moment = generator.bias.new_tensor([denominator ** 2 * correction, 0.])
    optimizer.state[generator.bias] = {
        "step": torch.tensor(float(PAST_ADAM_STEPS), dtype=torch.float32, device="cpu"),
        "exp_avg": torch.zeros_like(generator.bias),
        # A completed zero-gradient step under beta1=0 has m=0 and decays v.
        # These moments are reachable from 999 equal-gradient updates followed
        # by one zero-gradient update: their gradient² is max_v/(1-beta2**999).
        "exp_avg_sq": moment * group["betas"][1], "max_exp_avg_sq": moment.clone()}
    tester = SettleTest()
    tester.s, tester.b = SETTLED_SCALE, 1.
    tester.last_decisive, tester.last_decisive_scale = -1, 1.
    tester.counts["stationary"], tester.windows = 1, 1
    tester.log = [[PAST_ADAM_STEPS, "stationary", SETTLED_SCALE, 1.]]
    tester.begin(group["params"])
    fixture = Fixture(generator, critic, optimizer, tester, recipe.make_loss(), contracted_stiff_factor)
    validate_snapshot(fixture)
    return fixture


def validate_snapshot(fixture):
    """Validate ownership, closed-form memory units and the settled snapshot."""
    generator, optimizer, tester = fixture.generator, fixture.optimizer, fixture.tester
    group = optimizer.param_groups[0]
    assert len(optimizer.param_groups) == 1 and group["params"] == [generator.bias]
    assert generator.bias.dtype == torch.float64 and generator.bias.device.type == "cpu"
    assert not generator.weight.requires_grad
    assert not any(parameter.requires_grad for parameter in fixture.critic.parameters())
    torch.testing.assert_close(generator.bias, generator.bias.new_tensor(INITIAL_RESIDUAL), rtol=0, atol=0)
    torch.testing.assert_close(fixture.critic.features.weight,
                               torch.diag(generator.bias.new_tensor(FEATURE_SCALES)), rtol=0, atol=0)
    state = optimizer.state[generator.bias]
    assert float(state["step"]) == PAST_ADAM_STEPS
    beta2 = group["betas"][1]
    assert torch.equal(state["exp_avg_sq"], beta2 * state["max_exp_avg_sq"])
    assert not state["exp_avg"].any() and state["exp_avg_sq"][1] == 0
    assert torch.isfinite(state["exp_avg_sq"]).all() and (state["exp_avg_sq"] >= 0).all()
    historic_gradient_squared = state["max_exp_avg_sq"] / (1. - beta2 ** (PAST_ADAM_STEPS - 1))
    historic_second_moment = historic_gradient_squared * (1. - beta2 ** (PAST_ADAM_STEPS - 1))
    torch.testing.assert_close(state["max_exp_avg_sq"], historic_second_moment, rtol=2e-16, atol=0)
    correction = 1. - group["betas"][1] ** PAST_ADAM_STEPS
    denominator = float((state["max_exp_avg_sq"][0] / correction).sqrt()) + group["eps"]
    factor = BASE_LR * SETTLED_SCALE * FEATURE_SCALES[0] ** 2 / denominator
    assert math.isclose(factor, fixture.contracted_stiff_factor, rel_tol=1e-10)
    assert 0 < factor < 2
    if fixture.contracted_stiff_factor == 1.6:
        assert factor / SETTLED_SCALE > 2
    else:
        assert 0 < factor / SETTLED_SCALE < 2
    assert tester.s == SETTLED_SCALE and tester.b == 1. and tester.tau == 0
    assert not tester.blocks and not tester.r_b and not tester.r_2b
    assert tester.last_decisive == -1 and tester.last_decisive_scale == 1.
    torch.testing.assert_close(tester.anchor, generator.bias, rtol=0, atol=0)


def advance(fixture, step, *, cancel_first_release=False):
    group, tester = fixture.optimizer.param_groups[0], fixture.tester
    previous_scale = tester.s
    group["lr"] = BASE_LR * previous_scale
    before = float(fixture.game().detach())
    fixture.optimizer.zero_grad()
    fixture.game().backward()
    fixture.optimizer.step()
    decision = tester.observe(group["params"], previous_scale, PAST_ADAM_STEPS + step)
    proposed_scale = tester.s
    cancelled = bool(cancel_first_release and step == CANCEL_STEP
                     and decision == "drift" and proposed_scale > previous_scale)
    if cancelled:
        # Match the Supra causal intervention: retain the full native window,
        # decision/log/history, changing only the one proposed raw G scale.
        tester.s = previous_scale
    return {"step": step, "adam_step": float(fixture.optimizer.state[fixture.generator.bias]["step"]),
            "game_before": before, "game_after": float(fixture.game().detach()),
            "applied_lr": group["lr"], "previous_scale": previous_scale,
            "proposed_scale": proposed_scale, "next_scale": tester.s,
            "cancelled": cancelled, "decision": decision,
            "decision_details": deepcopy(tester.last) if decision else None}


def run(*, cancel_first_release=False, fixture=None, start=1, stop=HORIZON):
    fixture = make_fixture() if fixture is None else fixture
    rows = [advance(fixture, step, cancel_first_release=cancel_first_release)
            for step in range(start, stop + 1)]
    return fixture, rows


def assert_game_stable(rows):
    """Fixture-specific game stability; safe learning-rate increases are allowed."""
    baseline = rows[0]["game_before"]
    assert all(math.isfinite(row["game_after"]) for row in rows)
    maximum = max(row["game_after"] for row in rows)
    assert maximum <= baseline + GAME_TOLERANCE, (
        f"stationary paired RpGAN game escaped its settled neighborhood: "
        f"initial={baseline:.12g}, peak={maximum:.12g}, final={rows[-1]['game_after']:.12g}; "
        "this stability oracle permits any learning-rate changes that remain safe")


def main():
    torch.set_num_threads(1)
    results = {}
    for cancel in (False, True):
        _, rows = run(cancel_first_release=cancel)
        results["cancel_one_release" if cancel else "native"] = {
            "initial_game": rows[0]["game_before"],
            "peak_game": max(row["game_after"] for row in rows),
            "final_game": rows[-1]["game_after"],
            "decisions": [row for row in rows if row["decision"]]}
    _, safe_rows = run(fixture=make_fixture(contracted_stiff_factor=.8))
    assert_game_stable(safe_rows)
    results["safe_geometry_native_release"] = {
        "initial_game": safe_rows[0]["game_before"],
        "peak_game": max(row["game_after"] for row in safe_rows),
        "final_game": safe_rows[-1]["game_after"],
        "decisions": [row for row in safe_rows if row["decision"]]}
    print(json.dumps({"scope": "constructed stationary-score native G-controller unit reproducer",
                      "horizon": HORIZON, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
