#!/usr/bin/env python
"""CPU 2D gate for the particle control finetune after L2 is removed.

The record is (state, action). The expert action is tanh(-2.2 * previous
command); state is in the record and does not set the action. A short paired
reconstruction fits G and E_pair. E_control starts as a copy of E_pair, so it
sees the previous command where E_pair saw the current action.

`current` is the collapsed recipe: observation critics, detached reals, every
module trained, no paired L2. `fixed` is Rp logistic on the live pair
(record, z), sample-point b_cap on that pair, and only E_control plus the
action head trained. This file is that CPU example. The Lunar Lander particle
trainer uses YuE2 paired-error RpGAN (`controller_objective`).

python -u experiments/toy_particle_native_2d.py
tail -F results/gym/particle_native_2d/live.log
"""
import argparse
import copy
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from particlegan import get_recipe, learning_rate_scale, particle_ae

Z, K, H, B = 2, 32, 64, 128
PRE, FINE = 250, 400
# Untrained paired action MSE is about 2. The observation game stays there.
# Historical latent-joint acceptance threshold; a run must earn this result.
COLLAPSE_MIN = 1.0
FIXED_MAX = 0.18


def mlp(din, dout):
    net = nn.Sequential(nn.Linear(din, H), nn.LeakyReLU(0.2), nn.Linear(H, H),
                        nn.LeakyReLU(0.2), nn.Linear(H, dout))
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return net


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.query, self.offset = mlp(2, Z), mlp(2, Z)
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    def forward(self, inputs, prior):
        return particle_ae(F.layer_norm(self.query(inputs), (Z,)), self.offset(inputs), prior,
                           temperature=0.25, distance_reduction="sum", offset_bound=3.)


class Gen(nn.Module):
    """Separate heads, matching G1 and G2. There is no successor in this toy."""

    def __init__(self):
        super().__init__()
        self.state, self.action = mlp(Z, 1), mlp(Z, 1)

    def forward(self, z):
        return torch.cat([self.state(z), self.action(z)], 1)


class Critic(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = mlp(din, 1)

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def batch(n, generator):
    state = torch.rand(n, 1, generator=generator) * 2 - 1
    previous = torch.rand(n, 1, generator=generator) * 2 - 1
    # Previous command flips the action. State is in the record but does not
    # determine the action, so matching p(state, action) leaves the pair free.
    action = torch.tanh(-2.2 * previous)
    return state, previous, action, torch.cat([state, action], 1)


@torch.no_grad()
def action_mse(encoder, generator, prior):
    state, previous, action, _ = batch(2048, torch.Generator().manual_seed(99))
    code = encoder(torch.cat([state, previous], 1), prior).codes[:, 0]
    return float(F.mse_loss(generator(code)[:, 1:], action))


def clone_init(modules):
    return {name: copy.deepcopy(module) for name, module in modules.items()}


def restore(saved):
    return {name: copy.deepcopy(module) for name, module in saved.items()}


class Logger:
    def __init__(self, path):
        self.path = path
        self.file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = path.open("w", buffering=1)

    def __call__(self, message):
        print(message, flush=True)
        if self.file is not None:
            self.file.write(message + "\n")

    def close(self):
        if self.file is not None:
            self.file.close()


def pretrain(log):
    # This frozen regression compares its original arms, not changing API defaults.
    recipe = get_recipe(prior_kind='mog', sigma_rel=.025, z_dim=Z, num_particles=K,
                        total_steps=FINE, batch_size=B, lr=.0006, d_lr_mult=1.5,
                        prior_lr_mult=100., betas=(0., .999), prior_betas=(.5, .999),
                        reg_arm="b_cap", reg_coeff=1., reg_kappa=1., prior_reg=1.)
    spread = recipe.make_prior_regularizer()
    generator = torch.Generator().manual_seed(11)
    prior = recipe.make_prior(generator=torch.Generator().manual_seed(3))
    encoder, decoder = Enc(), Gen()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(prior.parameters()), lr=1e-3)
    for step in range(1, PRE + 1):
        state, _, action, record = batch(B, generator)
        loss = F.mse_loss(decoder(encoder(torch.cat([state, action], 1), prior).codes[:, 0]), record)
        loss = loss + spread(prior.z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            log(f"[pretrain] step={step}/{PRE} recon={float(loss.detach()):.4f}")
    control = Enc()
    control.load_state_dict(encoder.state_dict())
    init_mse = action_mse(control, decoder, prior)
    log(f"[pretrain] DONE init_action_mse={init_mse:.4f}")
    return recipe, clone_init(dict(ep=encoder, ec=control, g=decoder, prior=prior)), init_mse


def run_arm(name, kind, recipe, init, log):
    """kind is 'current' (observation D, all trainable) or 'fixed' (live latent joint)."""
    modules = restore(init)
    encoder, control, decoder, prior = (modules[key] for key in ("ep", "ec", "g", "prior"))
    gan = recipe.make_loss()
    reg = recipe.make_gradient_penalty()
    spread = recipe.make_prior_regularizer()
    d_obs, d_act, d_joint = Critic(2), Critic(1), Critic(2 + Z)
    critics = [d_obs, d_act] if kind == "current" else [d_joint]
    trainable = [encoder, control, decoder, prior] if kind == "current" else [control, decoder.action]
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
    opt_d = torch.optim.Adam([p for critic in critics for p in critic.parameters()],
                             lr=recipe.lr * recipe.d_lr_mult, betas=recipe.betas)
    ema_c, ema_g = copy.deepcopy(control).eval(), copy.deepcopy(decoder).eval()
    for param in list(ema_c.parameters()) + list(ema_g.parameters()):
        param.requires_grad_(False)
    rng = torch.Generator().manual_seed(21)
    series = []
    for step in range(1, FINE + 1):
        scale = learning_rate_scale(step - 1, FINE, recipe.lr_anneal_start, recipe.lr_floor)
        for group in opt_g.param_groups:
            group["lr"] = recipe.lr * scale
        for group in opt_d.param_groups:
            group["lr"] = recipe.lr * recipe.d_lr_mult * scale
        state, previous, action, record = batch(B, rng)
        context = torch.cat([state, previous], 1)
        with torch.no_grad():
            code = control(context, prior).codes[:, 0]
            z_prior, _ = prior.sample(B, rng)
            fake_c, fake_p = decoder(code), decoder(z_prior)
        if kind == "current":
            ld = gan.d_loss(d_obs(record), d_obs(fake_c)) + gan.d_loss(d_obs(record), d_obs(fake_p))
            ld = ld + gan.d_loss(d_act(action), d_act(fake_c[:, 1:])) + gan.d_loss(d_act(action), d_act(fake_p[:, 1:]))
            ld = ld + reg(d_obs, record, fake_c, step) + reg(d_act, action, fake_c[:, 1:], step)
        else:
            real_in = torch.cat([record, code], 1)
            fake_in = torch.cat([fake_c, code], 1)
            prior_in = torch.cat([fake_p, z_prior], 1)
            ld = gan.d_loss(d_joint(real_in), d_joint(fake_in)) + gan.d_loss(d_joint(real_in), d_joint(prior_in))
            ld = ld + reg(d_joint, real_in, fake_in, step)
        opt_d.zero_grad(set_to_none=True)
        ld.backward()
        opt_d.step()
        for critic in critics:
            critic.requires_grad_(False)
        code = control(context, prior).codes[:, 0]
        z_prior, _ = prior.sample(B, rng)
        fake_c, fake_p = decoder(code), decoder(z_prior)
        if kind == "current":
            with torch.no_grad():
                real_score, real_act = d_obs(record), d_act(action)
            encoded = encoder(torch.cat([state, action], 1), prior).codes[:, 0]
            fake_e = decoder(encoded)
            lg = (gan.g_loss(d_obs(fake_c), real_score) + gan.g_loss(d_obs(fake_p), real_score)
                  + gan.g_loss(d_obs(fake_e), real_score))
            lg = lg + (gan.g_loss(d_act(fake_c[:, 1:]), real_act) + gan.g_loss(d_act(fake_p[:, 1:]), real_act)
                       + gan.g_loss(d_act(fake_e[:, 1:]), real_act))
            loss = lg + spread(prior.z)
        else:
            # Example-only live (record, z) pair. The gym trainer uses paired-error RpGAN.
            real_live = torch.cat([record, code], 1)
            lg = gan.g_loss(d_joint(torch.cat([fake_p, z_prior], 1)), d_joint(real_live))
            lg = lg + gan.g_loss(d_joint(torch.cat([fake_c, code], 1)), d_joint(real_live).detach())
            loss = lg
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        opt_g.step()
        for critic in critics:
            critic.requires_grad_(True)
        with torch.no_grad():
            for source, target in ((control, ema_c), (decoder, ema_g)):
                for dest, src in zip(target.parameters(), source.parameters()):
                    dest.lerp_(src, 1 - recipe.ema_decay)
        if step == 1 or step % 50 == 0 or step == FINE:
            live, ema = action_mse(control, decoder, prior), action_mse(ema_c, ema_g, prior)
            series.append(dict(step=step, live=live, ema=ema))
            log(f"[{name}] step={step}/{FINE} live_action_mse={live:.4f} ema_action_mse={ema:.4f}")
    final = series[-1]
    log(f"[{name}] DONE live={final['live']:.4f} ema={final['ema']:.4f}")
    return dict(name=name, live=final["live"], ema=final["ema"], max_live=max(row["live"] for row in series),
                max_ema=max(row["ema"] for row in series))


def gate_status(current_ema, fixed_ema):
    """Keep the historical research thresholds separate from test expectations."""
    finite = math.isfinite(current_ema) and math.isfinite(fixed_ema)
    collapse = math.isfinite(current_ema) and current_ema >= COLLAPSE_MIN
    passed = finite and fixed_ema <= FIXED_MAX and fixed_ema < current_ema * 0.25
    return collapse, passed


def run_gate(log_path=None):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    log = Logger(log_path)
    log(f"ENV torch={torch.__version__} dtype={torch.get_default_dtype()} threads={torch.get_num_threads()}")
    started = time.perf_counter()
    recipe, init, init_mse = pretrain(log)
    if recipe.reg_arm != "b_cap" or recipe.reg_coeff != 1. or recipe.gan_mode != "rp" or recipe.loss_type != "logistic":
        raise RuntimeError("Fixed arm requires nonzero Rp logistic GANLoss and sample-point b_cap")
    current = run_arm("current", "current", recipe, init, log)
    fixed = run_arm("fixed", "fixed", recipe, init, log)
    collapse, passed = gate_status(current["ema"], fixed["ema"])
    log("LEADERBOARD ema_action_mse lower is better")
    log(f"  {fixed['ema']:.4f}  fixed latent-joint  threshold<={FIXED_MAX:.2f}  {'PASS' if passed else 'FAIL'}")
    log(f"  {current['ema']:.4f}  current observation  threshold>={COLLAPSE_MIN:.2f}  {'COLLAPSE' if collapse else 'FAIL'}")
    log("FIXED adv_weight=1 l2_weight=0 b_cap_coeff=1 b_cap_arm=b_cap supervised_only=false")
    log(f"GATE init={init_mse:.4f} collapse={collapse} fixed_pass={passed} elapsed_s={time.perf_counter()-started:.1f}")
    log.close()
    return dict(ok=bool(collapse and passed), init=init_mse, current=current, fixed=fixed,
                collapse=collapse, fixed_pass=passed, adversarial_weight=1., l2_weight=0.,
                b_cap_coeff=1., supervised_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="results/gym/particle_native_2d/live.log")
    args = parser.parse_args()
    result = run_gate(Path(args.log))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
