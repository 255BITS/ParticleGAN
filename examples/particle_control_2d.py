#!/usr/bin/env python
"""CPU 2D repro of the collapsed Lunar Lander particle finetune.

This is a collapse toy, not a fix. The parent is imitation. The baseline then
drops that paired loss and trains with the particle arm's game: Rp logistic
GAN, sample-point b_cap, a shared action head, and a particle cloud at 100x
the head learning rate. That baseline is expected to FAIL (action MSE stays
high and landings collapse).

Closed PR #15 continued from here with adversarial weight 0, which configures
the GAN and does not apply it. That continuation is not in this file. The
baseline adversarial weight stays 1.

Knob map (toy -> lunar particle finetune):

| Toy | Lunar particle finetune |
| --- | --- |
| ``Ctrl.query`` | ``E_control`` stem / router |
| ``Ctrl.particles`` | MoG particle cloud (``prior``) |
| ``Ctrl.head`` | G2 action head |
| ``Disc`` on actions | unconditional action critic |
| ``recipe.lr`` 6e-4, ``prior_lr_mult`` 100 | G/E group and prior group |
| Rp logistic + sample ``b_cap`` | ``require_classic_particle_gan`` |
| adversarial weight 1, anchor 0 | removed imitation / reconstruction L2 |

No GPU and no Lunar weights. Lines are prefixed ``[particle-2d]``.
"""
import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.gym_particle_finetune import require_classic_particle_gan
from particlegan import get_recipe

HORIZON = 25
LAND_RADIUS = 0.2
CRASH_ABS = 1.6
DT = 0.2
PRETRAIN_STEPS = 250
FINETUNE_STEPS = 80
BATCH = 256
HELD_OUT = 400
PRETRAIN_BETAS = (0.0, 0.999)
# Pure particle game. Closed PR #15 set this to 0 and called that a fix.
BASELINE_ADV_WEIGHT = 1.0
# First selectable update is 10, matching a particle run whose first checkpoint
# is already after action MSE has left the parent. Step 1 is not a candidate.
CHECKPOINTS = (10, 40, 80)
# Margins around the seed-0 split: late baseline action MSE stays above 0.30
# with landings at or below 0.25. The parent stays near action MSE 0.09.
BASELINE_MSE_MIN = 0.30
BASELINE_LAND_MAX = 0.25


class Ctrl(nn.Module):
    def __init__(self, particles=32):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 4))
        self.particles = nn.Parameter(torch.randn(particles, 4) * 0.3)
        self.head = nn.Linear(4, 2)

    def codes(self, state):
        query = self.query(state)
        distance = (query[:, None, :] - self.particles[None]).pow(2).sum(-1)
        weights = torch.softmax(-distance / 0.25, dim=1)
        return weights @ self.particles

    def act(self, state):
        return self.head(self.codes(state)).tanh()

    def prior_act(self, count):
        index = torch.randint(self.particles.shape[0], (count,))
        return self.head(self.particles[index]).tanh()


class ActionDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, action):
        return self.net(action).squeeze(-1)


def kinematic_response(state, action):
    """One double-integrator step. State is [x, y, vx, vy]; action is [ax, ay].

    Scoring only. It is not a Lunar Lander simulator.
    """
    velocity = state[:, 2:4] + DT * action
    position = state[:, :2] + DT * velocity
    return torch.cat([position, velocity], 1)


def expert_action(state):
    return (-2.2 * state[:, :2] - 1.6 * state[:, 2:]).clamp(-1, 1)


def sample_states(count, generator=None):
    state = torch.zeros(count, 4)
    state[:, :2] = torch.rand(count, 2, generator=generator) * 1.6 - 0.8
    state[:, 2:] = torch.rand(count, 2, generator=generator) * 0.8 - 0.4
    return state


def rollout_stats(ctrl, generator):
    state = sample_states(200, generator)
    state[:, :2] = state[:, :2] * 0.75
    state[:, 2:] = state[:, 2:] * 0.5
    for _ in range(HORIZON):
        state = kinematic_response(state, ctrl.act(state))
    final = state[:, :2].norm(dim=1)
    crashed = state[:, :2].abs().amax(dim=1) > CRASH_ABS
    landings = float(((final < LAND_RADIUS) & ~crashed).float().mean())
    held = sample_states(HELD_OUT, generator)
    mse = float(F.mse_loss(ctrl.act(held), expert_action(held)))
    return dict(landings=landings, action_mse=mse)


def evaluate(ctrl, seed):
    ctrl.eval()
    with torch.no_grad():
        stats = rollout_stats(ctrl, torch.Generator().manual_seed(seed))
    ctrl.train()
    return stats


def pretrain(ctrl):
    opt = torch.optim.Adam(list(ctrl.query.parameters()) + list(ctrl.head.parameters()),
                           lr=2e-3, betas=PRETRAIN_BETAS)
    ctrl.particles.requires_grad_(False)
    for _ in range(PRETRAIN_STEPS):
        state = sample_states(BATCH)
        loss = F.mse_loss(ctrl.act(state), expert_action(state))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ctrl.particles.requires_grad_(True)


def baseline_collapsed(stats):
    return stats["action_mse"] >= BASELINE_MSE_MIN and stats["landings"] <= BASELINE_LAND_MAX


def run_baseline(ctrl, recipe):
    """Pure RpGAN + sample b_cap on the action head, stem, and particle cloud."""
    if BASELINE_ADV_WEIGHT == 0:
        raise RuntimeError("The collapse baseline must apply a nonzero adversarial weight")
    gan, reg = recipe.make_loss(), recipe.make_gradient_penalty()
    require_classic_particle_gan(gan, reg)
    disc = ActionDisc()
    opt_g = torch.optim.Adam([
        {"params": list(ctrl.query.parameters()) + list(ctrl.head.parameters()), "lr": recipe.lr},
        {"params": [ctrl.particles], "lr": recipe.lr * recipe.prior_lr_mult,
         "betas": recipe.prior_betas},
    ], betas=recipe.betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=recipe.lr * recipe.d_lr_mult, betas=recipe.betas)
    best, history = None, []
    for step in range(1, FINETUNE_STEPS + 1):
        state = sample_states(BATCH)
        real = expert_action(state)
        with torch.no_grad():
            fake = ctrl.act(state)
            prior = ctrl.prior_act(BATCH)
        d_loss = gan.d_loss(disc(real), disc(fake)) + reg(disc, real, fake, step)
        d_loss = d_loss + gan.d_loss(disc(real), disc(prior)) + reg(disc, real, prior, step)
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()
        state = sample_states(BATCH)
        real = expert_action(state)
        fake = ctrl.act(state)
        prior = ctrl.prior_act(BATCH)
        g_loss = BASELINE_ADV_WEIGHT * gan.g_loss(disc(fake), disc(real).detach())
        g_loss = g_loss + BASELINE_ADV_WEIGHT * gan.g_loss(disc(prior), disc(real).detach())
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        if step in CHECKPOINTS:
            stats = evaluate(ctrl, 1000 + step)
            stats.update(step=step, d_loss=float(d_loss.detach()), g_loss=float(g_loss.detach()))
            history.append({key: value for key, value in stats.items()})
            if best is None or stats["action_mse"] < best["action_mse"]:
                best = {**stats, "weights": copy.deepcopy(ctrl.state_dict())}
    ctrl.load_state_dict(best["weights"])
    selected = evaluate(ctrl, 4242)
    selected.update(step=best["step"], history=history,
                    rule="min action MSE among finetune checkpoints, excluding the parent")
    return selected


def run_gate(seed=0):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    recipe = get_recipe("mog", z_dim=4, num_particles=32, total_steps=FINETUNE_STEPS, batch_size=BATCH)
    parent = Ctrl()
    pretrain(parent)
    parent_stats = evaluate(parent, 7)
    baseline = run_baseline(copy.deepcopy(parent), recipe)
    collapsed = baseline_collapsed(baseline)
    return dict(
        verdict="COLLAPSE" if collapsed else "NO_COLLAPSE",
        parent=parent_stats,
        baseline=baseline,
        adv_weight=BASELINE_ADV_WEIGHT,
        baseline_lr=recipe.lr,
        prior_lr_mult=recipe.prior_lr_mult,
        gan_mode=recipe.gan_mode,
        reg_arm=recipe.reg_arm,
    )


def _line(name, stats, mark):
    return (f"[particle-2d] {name:28} step={stats.get('step', 0):<4} "
            f"landings={stats['landings']:.2f} action_mse={stats['action_mse']:.4f} {mark}")


def main():
    result = run_gate()
    print(_line("parent (imitation pretrain)", result["parent"], "REF"))
    print(_line("baseline pure RpGAN+b_cap", result["baseline"],
                "FAIL" if baseline_collapsed(result["baseline"]) else "UNEXPECTED"))
    base_hist = " ".join(f"{row['step']}:{row['action_mse']:.3f}/{row['landings']:.2f}"
                         for row in result["baseline"]["history"])
    print(f"[particle-2d] baseline checkpoints (step:action_mse/landings) {base_hist}")
    print(f"[particle-2d] GATE {result['verdict']} "
          f"adv={result['adv_weight']} baseline_lr={result['baseline_lr']} "
          f"prior_lr_mult={result['prior_lr_mult']} gan={result['gan_mode']} "
          f"reg={result['reg_arm']}")
    print("[particle-2d] leaderboard (held-out landings, higher is better; action MSE, lower is better)")
    print("[particle-2d] collapse repro only; adv=0 is not a fix and is not an arm here")
    print(f"[particle-2d] {'arm':28} {'landings':>10} {'action_mse':>12}")
    for name, stats in (("parent", result["parent"]), ("baseline", result["baseline"])):
        print(f"[particle-2d] {name:28} {stats['landings']:10.2f} {stats['action_mse']:12.4f}")
    if result["verdict"] != "COLLAPSE":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
