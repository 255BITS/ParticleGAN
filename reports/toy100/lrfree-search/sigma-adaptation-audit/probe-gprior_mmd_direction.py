"""Read-only fixed-context generator/prior MMD versus GAN gradient probe."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "runs/h2-prior-couple-sigma/grid100"
PACKAGE = ROOT / "candidate-prior-couple-sigma/package"
HOST = Path("/ml2/hypergan/lrfree-20260926/harness/hosts")
sys.path[:0] = [str(PACKAGE), str(HOST)]

from native100.problems import sample_real  # noqa: E402
from native100.toy_models import SimpleMLPDiscriminator  # noqa: E402
from particlegan.continuous import DataDriftController  # noqa: E402
from particlegan.gan_loss import GANLoss  # noqa: E402
from particlegan.particle_prior import ParticlePrior  # noqa: E402

B = 1024
GEN_SEEDS = (8481, 8483)
REAL_SEEDS = (8482, 8484)


def _mmd(fake, real, h):
    def kernel(a, b):
        da = a.square().sum(dim=1, keepdim=True)
        db = b.square().sum(dim=1).unsqueeze(0)
        d2 = (da + db - 2*a@b.T).clamp_min(0)
        return torch.exp(-d2 / (2*h*h))
    ff = kernel(fake, fake)
    fr = kernel(fake, real)
    rr = kernel(real, real)
    return ((ff.sum()-ff.diagonal().sum()) / (B*(B-1))
            + (rr.sum()-rr.diagonal().sum()) / (B*(B-1))
            - 2*fr.mean())


def _draw(g, prior, controller, seed, sigma):
    stream = torch.Generator(device=prior.z.device).manual_seed(seed)
    latent, _ = prior.sample(B, generator=stream)
    latent = controller.perturb_latent(latent, stream, prior)
    noise = torch.randn((B, 2), device=prior.z.device, generator=stream)
    return g(latent) + sigma*noise


def _grad_stats(a, b):
    flat_a = torch.cat([v.detach().flatten() for v in a])
    flat_b = torch.cat([v.detach().flatten() for v in b])
    dot = float(torch.dot(flat_a, flat_b))
    an = float(flat_a.norm())
    bn = float(flat_b.norm())
    return dict(mmd_norm=an, gan_norm=bn, norm_ratio=an/bn if bn else None,
                dot=dot, cosine=dot/(an*bn) if an*bn else None)


def main():
    torch.set_num_threads(2)
    start = time.monotonic()
    state = torch.load(RUN / "final-state.pt", map_location="cpu", weights_only=False)["trainer"]
    h = json.loads((HERE / "result.json").read_text())["bandwidth"]["value"]
    sigma = float(state["output_noise"]["log_sigma"].exp())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = nn.Linear(2, 2).to(device)
    g.load_state_dict(state["models"]["G"])
    prior = ParticlePrior(20000, 2, device=device)
    prior.load_state_dict(state["models"]["prior"])
    critic = SimpleMLPDiscriminator(in_dim=2, hidden_dim=128, n_hidden=3, fourier=3).to(device)
    critic.load_state_dict(state["models"]["D"])
    critic.eval().requires_grad_(False)
    controller = DataDriftController("dv12")
    controller.load_state_dict(state["controller"])
    controller.latent_bandwidth = controller.latent_bandwidth.to(device)
    loss_fn = GANLoss()
    params = (g.weight, g.bias, prior.z)
    contexts = []
    for kind, gs, rs in zip(("fit", "heldout"), GEN_SEEDS, REAL_SEEDS):
        real = sample_real("grid100", B, device=device,
                           generator=torch.Generator(device=device).manual_seed(rs))
        fake = _draw(g, prior, controller, gs, sigma)
        mmd = _mmd(fake, real, h)
        gan = loss_fn.g_loss(critic(fake), critic(real))
        mg = torch.autograd.grad(mmd, params, retain_graph=True)
        gg = torch.autograd.grad(gan, params)
        contexts.append(dict(kind=kind, real=real, seed=gs, mmd=float(mmd.detach()),
                             gan=float(gan.detach()), mmd_grads=mg, gan_grads=gg))
    lr_g = float(state["optimizers"][0]["param_groups"][0]["lr"])
    lr_prior = float(state["optimizers"][0]["param_groups"][1]["lr"])
    fit, held = contexts
    delta = tuple(-lr * grad.sign() for lr, grad in zip((lr_g, lr_g, lr_prior), fit["mmd_grads"]))
    held_directional = float(sum((d*g).sum() for d, g in zip(delta, held["mmd_grads"])))
    fit_directional = float(sum((d*g).sum() for d, g in zip(delta, fit["mmd_grads"])))
    held_directional_g = float(sum((d*g).sum() for d, g in zip(delta[:2], held["mmd_grads"][:2])))
    held_directional_prior = float((delta[2]*held["mmd_grads"][2]).sum())
    touched_fit = fit["mmd_grads"][2].norm(dim=1) > 0
    touched_held = held["mmd_grads"][2].norm(dim=1) > 0
    with torch.no_grad():
        for p, d in zip(params, delta):
            p.add_(d)
        after_fake = _draw(g, prior, controller, held["seed"], sigma)
        after_mmd = float(_mmd(after_fake, held["real"], h))
        after_gan = float(loss_fn.g_loss(critic(after_fake), critic(held["real"])))
        for p, d in zip(params, delta):
            p.sub_(d)
    report = {
        "method": "same frozen GAN checkpoint; independent fixed-context train/heldout draws, B=1024; exact DV12 latent jitter",
        "mmd": "unbiased Gaussian-kernel MMD²; h fixed from independent real calibration, no evaluator geometry",
        "bandwidth": h, "sigma": sigma, "step": state["completed_steps"],
        "batch_size": B, "gen_seeds": GEN_SEEDS, "real_seeds": REAL_SEEDS,
        "fit_losses": {k: fit[k] for k in ("mmd", "gan")},
        "heldout_losses": {k: held[k] for k in ("mmd", "gan")},
        "fit_gradient": {"generator": _grad_stats(fit["mmd_grads"][:2], fit["gan_grads"][:2]),
                         "prior": _grad_stats(fit["mmd_grads"][2:], fit["gan_grads"][2:])},
        "heldout_gradient": {"generator": _grad_stats(held["mmd_grads"][:2], held["gan_grads"][:2]),
                             "prior": _grad_stats(held["mmd_grads"][2:], held["gan_grads"][2:])},
        "fit_heldout_mmd_gradient_cosine": {
            "generator": _grad_stats(fit["mmd_grads"][:2], held["mmd_grads"][:2])["cosine"],
            "prior": _grad_stats(fit["mmd_grads"][2:], held["mmd_grads"][2:])["cosine"]},
        "prior_rows_with_gradient": {"fit": int(touched_fit.sum()),
                                     "heldout": int(touched_held.sum()),
                                     "overlap": int((touched_fit & touched_held).sum())},
        "one_sign_step": {"effective_lr_generator": lr_g, "effective_lr_prior": lr_prior,
                          "fit_predicted_mmd_delta": fit_directional,
                          "heldout_predicted_mmd_delta": held_directional,
                          "heldout_predicted_mmd_delta_generator": held_directional_g,
                          "heldout_predicted_mmd_delta_prior": held_directional_prior,
                          "heldout_observed_mmd_delta": after_mmd-held["mmd"],
                          "heldout_observed_gan_delta": after_gan-held["gan"],
                          "max_abs_delta_generator": max(float(d.abs().max()) for d in delta[:2]),
                          "max_abs_delta_prior": float(delta[2].abs().max())},
        "seconds": time.monotonic()-start, "diagnostic_only": True,
    }
    (HERE / "gprior_mmd_direction.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
