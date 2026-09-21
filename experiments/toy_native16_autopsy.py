#!/usr/bin/env python
"""CPU probes for the PR #16 toy gate versus Lunar landings.

The merged gate scores EMA action MSE on a record whose expert action is
``tanh(-2.2 * previous)``. State is stored and does not set the action, and
there is no closed-loop task. ``diag_action_mse`` in the gym finetune is the
same kind of number: one shuffled minibatch, expert previous command, live
weights, standardized action slice.

Not the gym default. The default is YuE2 paired-error RpGAN at adv_weight 1.
This script does not train Lunar Lander and does not set ``adv_weight=0``.
Old gate PASS with honest gate FAIL is the expected #16 readout.

python -u experiments/toy_native16_autopsy.py
tail -F results/gym/native16_autopsy/live.log
"""
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy

from experiments.toy_particle_native_2d import (B, FINE, FIXED_MAX, PRE, Critic, Enc, Gen,
    Logger, clone_init, restore)
from particlegan import get_recipe, learning_rate_scale

# Same numeric band the merged gate calls a pass, and the band the post-merge
# GPU log occupied (about 0.07 to 0.18) while landings stayed near the floor.
HEALTHY_MIN = 0.07
LAND_ABS = 0.08
HORIZON = 24
POOL_HORIZON = 32
GAIN = 0.5
BIAS = 0.4
HONEST_LAND = 0.8


def expert_action(state):
    return torch.tanh(-2.2 * state)


def step_state(state, action):
    return (state + GAIN * action).clamp(-1.5, 1.5)


def expert_pool(n, horizon, generator):
    state = torch.rand(n, 1, generator=generator) * 2 - 1
    previous = torch.zeros_like(state)
    states, prevs, actions = [], [], []
    for _ in range(horizon):
        action = expert_action(state)
        states.append(state)
        prevs.append(previous)
        actions.append(action)
        state = step_state(state, action)
        previous = action
    return torch.cat(states), torch.cat(prevs), torch.cat(actions)


def teacher_forced_mse(policy, pool, n=None, early_steps=4):
    """Full-pool MSE, plus the first few steps of each episode.

    ``expert_pool`` concatenates time first, so the early window is the transient.
    Shuffled-record diagnostics average both, and settled steps dominate.
    """
    state, previous, action = pool
    full = float((policy(state, previous) - action).pow(2).mean())
    if n is None:
        return full
    early = slice(0, n * early_steps)
    return full, float((policy(state[early], previous[early]) - action[early]).pow(2).mean())


def closed_loop(policy, n=257, horizon=HORIZON):
    """Learner previous command, state moved by the action. Landing is |s| small."""
    state = torch.linspace(-1, 1, n).unsqueeze(1)
    previous = torch.zeros_like(state)
    error = 0.
    for _ in range(horizon):
        action = policy(state, previous)
        error += float((action - expert_action(state)).pow(2).mean())
        state = step_state(state, action)
        previous = action
    return dict(land_rate=float((state.abs() < LAND_ABS).float().mean()),
                on_policy_mse=error / horizon,
                final_abs=float(state.abs().mean()))


def score_policy(name, policy, pool, n=512):
    loop = closed_loop(policy)
    tf, early = teacher_forced_mse(policy, pool, n=n)
    return dict(name=name, tf_mse=tf, early_tf_mse=early, on_policy_mse=loop["on_policy_mse"],
                land_rate=loop["land_rate"], final_abs=loop["final_abs"],
                old_pass=tf <= FIXED_MAX, honest_pass=loop["land_rate"] >= HONEST_LAND)


def prev_plant_dead_controller():
    """On the PR #16 target, closing the loop makes a zero action look accurate.

    The target is ``tanh(-2.2 * previous)``. After a zero action, previous is 0,
    so the target becomes 0. There is no state to land.
    """
    generator = torch.Generator().manual_seed(5)
    previous = torch.rand(8192, 1, generator=generator) * 2 - 1
    target = expert_action(previous)
    zero_tf = float(target.pow(2).mean())
    perfect_tf = float((target - target).pow(2).mean())
    zero_err = perfect_err = 0.
    prev_z, prev_p = previous.clone(), previous.clone()
    steps = 8
    for _ in range(steps):
        zero_err += float(expert_action(prev_z).pow(2).mean())
        prev_z = torch.zeros_like(prev_z)
        pred = expert_action(prev_p)
        perfect_err += float((pred - expert_action(prev_p)).pow(2).mean())
        prev_p = pred
    return dict(zero_tf=zero_tf, zero_closed=zero_err / steps,
                perfect_tf=perfect_tf, perfect_closed=perfect_err / steps)


def state_plant_counterexamples():
    """Policies inside the merged PASS band that miss a state-feedback pad."""
    pool = expert_pool(512, POOL_HORIZON, torch.Generator().manual_seed(7))
    rows = {
        "expert": score_policy("expert", lambda s, p: expert_action(s), pool),
        "bias": score_policy("bias", lambda s, p: expert_action(s) + BIAS, pool),
        "copy_prev": score_policy("copy_prev", lambda s, p: p, pool),
    }
    # Smallest leak of the expert previous command that still sits in the
    # merged pass band and fails the pad. One deterministic scan, not a reseed.
    leak = None
    for alpha in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        row = score_policy(f"prev_leak_{alpha:.2f}",
                           lambda s, p, alpha=alpha: expert_action(s) + alpha * p, pool)
        row["alpha"] = alpha
        if row["old_pass"] and not row["honest_pass"]:
            leak = row
            break
    if leak is None:
        raise RuntimeError("No previous-command leak landed in the pass band")
    rows["prev_leak"] = leak
    rows["ok"] = (rows["expert"]["honest_pass"] and rows["expert"]["tf_mse"] < 1e-8
                  and rows["bias"]["old_pass"] and not rows["bias"]["honest_pass"]
                  and HEALTHY_MIN <= rows["bias"]["tf_mse"] <= FIXED_MAX
                  and leak["on_policy_mse"] > leak["tf_mse"])
    return rows


@torch.no_grad()
def ema_tf(encoder, generator, prior, pool, n, early_steps=4):
    state, previous, action = pool

    def mse(slc):
        code = encoder(torch.cat([state[slc], previous[slc]], 1), prior).codes[:, 0]
        return float(F.mse_loss(generator(code)[:, 1:], action[slc]))

    pick = torch.randperm(len(state))[:2048]
    return mse(pick), mse(slice(0, n * early_steps))


@torch.no_grad()
def ema_loop(encoder, generator, prior):
    def policy(state, previous):
        code = encoder(torch.cat([state, previous], 1), prior).codes[:, 0]
        return generator(code)[:, 1:]

    return closed_loop(policy, n=129)


def pretrain_state(pool, log):
    recipe = get_recipe("mog", z_dim=2, num_particles=32, total_steps=FINE, batch_size=B)
    spread = recipe.make_prior_regularizer()
    prior = recipe.make_prior(generator=torch.Generator().manual_seed(3))
    encoder, decoder = Enc(), Gen()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters())
                           + list(prior.parameters()), lr=1e-3)
    generator = torch.Generator().manual_seed(11)
    state, _, action = pool
    record = torch.cat([state, action], 1)
    for step in range(1, PRE + 1):
        idx = torch.randint(len(state), (B,), generator=generator)
        loss = F.mse_loss(decoder(encoder(record[idx], prior).codes[:, 0]), record[idx])
        loss = loss + spread(prior.z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            log(f"[pretrain] step={step}/{PRE} recon={float(loss.detach()):.4f}")
    control = Enc()
    control.load_state_dict(encoder.state_dict())
    return recipe, clone_init(dict(ep=encoder, ec=control, g=decoder, prior=prior))


def run_state_arm(name, kind, recipe, init, pool, log):
    """kind: fixed full record, action_pair (action, z) only, or paired L2."""
    modules = restore(init)
    encoder, control, decoder, prior = (modules[key] for key in ("ep", "ec", "g", "prior"))
    gan = recipe.make_loss()
    reg = recipe.make_gradient_penalty()
    critics = []
    if kind == "fixed":
        critics = [Critic(2 + 2)]
    elif kind == "action_pair":
        critics = [Critic(1 + 2)]
    elif kind != "l2":
        raise ValueError(kind)
    trainable = [control, decoder.action]
    chosen, seen = [], set()
    for module in trainable:
        for param in module.parameters():
            if id(param) not in seen:
                chosen.append(param)
                seen.add(id(param))
    for module in (encoder, control, decoder, prior):
        for param in module.parameters():
            param.requires_grad_(id(param) in seen)
    opt_g = torch.optim.Adam(chosen, lr=recipe.lr, betas=recipe.betas)
    opt_d = None
    if critics:
        opt_d = torch.optim.Adam([p for critic in critics for p in critic.parameters()],
                                 lr=recipe.lr * recipe.d_lr_mult, betas=recipe.betas)
    ema_c, ema_g = copy.deepcopy(control).eval(), copy.deepcopy(decoder).eval()
    for param in list(ema_c.parameters()) + list(ema_g.parameters()):
        param.requires_grad_(False)
    rng = torch.Generator().manual_seed(21)
    state, previous, action = pool
    series = []
    for step in range(1, FINE + 1):
        scale = learning_rate_scale(step - 1, FINE, recipe.lr_anneal_start, recipe.lr_floor)
        for group in opt_g.param_groups:
            group["lr"] = recipe.lr * scale
        if opt_d is not None:
            for group in opt_d.param_groups:
                group["lr"] = recipe.lr * recipe.d_lr_mult * scale
        idx = torch.randint(len(state), (B,), generator=rng)
        s, prev, act = state[idx], previous[idx], action[idx]
        context = torch.cat([s, prev], 1)
        record = torch.cat([s, act], 1)
        if kind != "l2":
            with torch.no_grad():
                code = control(context, prior).codes[:, 0]
                z_prior, _ = prior.sample(B, rng)
                fake_c, fake_p = decoder(code), decoder(z_prior)
            critic = critics[0]
            if kind == "fixed":
                real_in = torch.cat([record, code], 1)
                fake_in = torch.cat([fake_c, code], 1)
                prior_in = torch.cat([fake_p, z_prior], 1)
            else:
                real_in = torch.cat([act, code], 1)
                fake_in = torch.cat([fake_c[:, 1:], code], 1)
                prior_in = torch.cat([fake_p[:, 1:], z_prior], 1)
            ld = gan.d_loss(critic(real_in), critic(fake_in)) + gan.d_loss(critic(real_in), critic(prior_in))
            ld = ld + reg(critic, real_in, fake_in, step)
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            critic.requires_grad_(False)
        code = control(context, prior).codes[:, 0]
        if kind == "l2":
            loss = F.mse_loss(decoder(code)[:, 1:], act)
        else:
            z_prior, _ = prior.sample(B, rng)
            fake_c, fake_p = decoder(code), decoder(z_prior)
            if kind == "fixed":
                real_live = torch.cat([record, code], 1)
                fake_live = torch.cat([fake_c, code], 1)
                prior_live = torch.cat([fake_p, z_prior], 1)
            else:
                real_live = torch.cat([act, code], 1)
                fake_live = torch.cat([fake_c[:, 1:], code], 1)
                prior_live = torch.cat([fake_p[:, 1:], z_prior], 1)
            # Same split as particle_game: prior term keeps the real code live;
            # the control term detaches the real score.
            loss = gan.g_loss(critic(prior_live), critic(real_live))
            loss = loss + gan.g_loss(critic(fake_live), critic(real_live).detach())
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        opt_g.step()
        if critics:
            critics[0].requires_grad_(True)
        with torch.no_grad():
            for source, target in ((control, ema_c), (decoder, ema_g)):
                for dest, src in zip(target.parameters(), source.parameters()):
                    dest.lerp_(src, 1 - recipe.ema_decay)
        if step == 1 or step % 50 == 0 or step == FINE:
            tf, early = ema_tf(ema_c, ema_g, prior, pool, n=len(state) // POOL_HORIZON)
            loop = ema_loop(ema_c, ema_g, prior)
            series.append(dict(step=step, tf_mse=tf, early_tf_mse=early, **loop))
            log(f"[{name}] step={step}/{FINE} ema_tf_mse={tf:.4f} early_tf_mse={early:.4f} "
                f"on_policy_mse={loop['on_policy_mse']:.4f} land_rate={loop['land_rate']:.3f} "
                f"final_abs={loop['final_abs']:.3f}")
    final = series[-1]
    final.update(name=name, kind=kind, adv_weight=0. if kind == "l2" else 1.,
                 l2_weight=1. if kind == "l2" else 0.,
                 old_pass=final["tf_mse"] <= FIXED_MAX,
                 honest_pass=final["land_rate"] >= HONEST_LAND)
    log(f"[{name}] DONE tf={final['tf_mse']:.4f} on_policy={final['on_policy_mse']:.4f} "
        f"land={final['land_rate']:.3f} old_gate={'PASS' if final['old_pass'] else 'FAIL'} "
        f"honest_gate={'PASS' if final['honest_pass'] else 'FAIL'}")
    return final


def _log_policy(log, row):
    log(f"  {row['name']:16} tf={row['tf_mse']:.4f} early_tf={row['early_tf_mse']:.4f} "
        f"on_policy={row['on_policy_mse']:.4f} land={row['land_rate']:.3f} "
        f"old={'PASS' if row['old_pass'] else 'FAIL'} honest={'PASS' if row['honest_pass'] else 'FAIL'}")


def run_autopsy(log_path=None):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    log = Logger(log_path)
    started = time.perf_counter()
    log("NOT THE GYM DEFAULT. This is the #16 autopsy: teacher-forced MSE vs an on-policy pad.")
    log("Default gym recipe is YuE2 paired-error RpGAN, adv_weight=1. Honest gate is landing rate>=0.80.")
    dead = prev_plant_dead_controller()
    log("PROBE previous-only plant (PR #16 target). Closed-loop MSE uses the learner previous command.")
    log(f"  zero action   teacher_forced_mse={dead['zero_tf']:.4f} closed_loop_mse={dead['zero_closed']:.4f}")
    log(f"  perfect map   teacher_forced_mse={dead['perfect_tf']:.4f} closed_loop_mse={dead['perfect_closed']:.4f}")
    rows = state_plant_counterexamples()
    if not rows["ok"] or dead["zero_closed"] >= dead["zero_tf"] * 0.3:
        raise RuntimeError("Analytic probes did not separate the gates")
    log("PROBE state-feedback pad. Old gate is teacher-forced MSE<=0.18. Honest gate is landing rate>=0.80.")
    log("LEADERBOARD analytic policies")
    for key in ("expert", "bias", "copy_prev", "prev_leak"):
        _log_policy(log, rows[key])
    log(f"  prev_leak alpha={rows['prev_leak']['alpha']}")
    pool = expert_pool(256, POOL_HORIZON, torch.Generator().manual_seed(11))
    recipe, init = pretrain_state(pool, log)
    if recipe.gan_mode != "rp" or recipe.loss_type != "logistic" or recipe.reg_arm != "b_cap":
        raise RuntimeError("State-plant arms must use the locked Rp logistic b_cap recipe")
    trained = []
    for name, kind in (("fixed", "fixed"), ("action_pair", "action_pair"), ("l2_probe", "l2")):
        trained.append(run_state_arm(name, kind, recipe, init, pool, log))
    log("LEADERBOARD trained state plant, EMA, in-pool teacher forcing vs closed loop")
    for row in trained:
        log(f"  {row['name']:16} tf={row['tf_mse']:.4f} early_tf={row['early_tf_mse']:.4f} "
            f"on_policy={row['on_policy_mse']:.4f} land={row['land_rate']:.3f} "
            f"adv_weight={row['adv_weight']} l2_weight={row['l2_weight']} "
            f"old={'PASS' if row['old_pass'] else 'FAIL'} honest={'PASS' if row['honest_pass'] else 'FAIL'}")
    log("L2 probe is a contrast on this plant only. It is not a Lunar change and adv_weight stays 1 on the GAN arms.")
    elapsed = time.perf_counter() - started
    log(f"GATE analytic_ok=True elapsed_s={elapsed:.1f}")
    log.close()
    return dict(ok=True, dead=dead,
                analytic={k: rows[k] for k in ("expert", "bias", "copy_prev", "prev_leak")},
                trained=trained, elapsed_s=elapsed)


def main():
    parser_path = Path("results/gym/native16_autopsy/live.log")
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=str(parser_path))
    parser.add_argument("--summary", default="reports/gym/native16_autopsy/summary.json")
    args = parser.parse_args()
    result = run_autopsy(Path(args.log))
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {summary}", flush=True)


if __name__ == "__main__":
    main()
