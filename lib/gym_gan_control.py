"""Masked-observation GAN control: E(st) -> z; G1 -> st, G2 -> at, G3 -> st+1."""
import json
from pathlib import Path

import torch
from torch import nn

from lib.gym_state_control import (build_state_models, parameter_hashes,
    predict_state_control, control_action_details, state_control_action)
from lib.gym_transition import GymTransitionDiscriminator, GymTransitionScaler, contact_record

MODULE_KEYS = ("G", "E", "prior", "D")


class MaskedJointCritic(GymTransitionDiscriminator):
    """Reapply observation mask inside the critic, including bcap perturbations."""
    def forward(self, observation, context):
        action = torch.where(context[:, -1:].bool(), observation[:, 8:10],
                             torch.zeros_like(observation[:, 8:10]))
        observation = torch.cat([observation[:, :8], action, observation[:, 10:]], 1)
        return super().forward(observation, context)


class MaskedControlCritics(nn.Module):
    """Mask is observed context. Hidden normalized actions are replaced with zero."""
    def __init__(self, cfg):
        super().__init__()
        self.arm = cfg["arm"]
        self.critics = nn.ModuleDict({"joint": MaskedJointCritic(18, 12, cfg["d_width"])})
        if self.arm == "marginals":
            self.critics["action"] = GymTransitionDiscriminator(2, 11, cfg["marginal_width"])
            self.critics["state"] = GymTransitionDiscriminator(8, 12, cfg["marginal_width"])

    def roles(self):
        return ("joint", "action", "state", "next_state") if self.arm == "marginals" else ("joint",)

    def critic_for(self, role):
        return self.critics["state" if role == "next_state" else role]

    def inputs(self, role, record, terrain, observed):
        if role == "joint":
            if observed.shape != (len(record), 1) or not ((observed == 0) | (observed == 1)).all():
                raise ValueError("Observed-action mask must be binary [batch,1]")
            # where also blocks nonfinite hidden values and their gradients.
            actions = torch.where(observed.bool(), record[:, 8:10], torch.zeros_like(record[:, 8:10]))
            return torch.cat([record[:, :8], actions, record[:, 10:]], 1), torch.cat([terrain, observed], 1)
        if role == "action":
            return record[:, 8:10], terrain
        if role in ("state", "next_state"):
            sl = slice(0, 8) if role == "state" else slice(10, 18)
            context = torch.cat([terrain, terrain.new_full((len(record), 1), float(role == "next_state"))], 1)
            return record[:, sl], context
        raise ValueError(f"Unknown critic role {role}")


def build_gan_models(cfg, scaler, device="cpu"):
    bundle = build_state_models(cfg, scaler, device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"] + 103)
        bundle["D"] = MaskedControlCritics(cfg).to(device)
    return bundle


def initial_hashes(bundle):
    hashes = parameter_hashes(bundle)
    hashes["D"] = parameter_hashes(dict(G=bundle["D"], E=bundle["D"], prior=bundle["D"]))["G"]
    # Reuse the canonical tensor/non-tensor hashing implementation per critic.
    for key, critic in bundle["D"].critics.items():
        hashes[f"D.{key}"] = parameter_hashes(dict(G=critic, E=critic, prior=critic))["G"]
    return hashes


def real_views(bundle, batch):
    """Two observation views; no hidden action array is accepted by this API."""
    scaler = bundle["scaler"]
    complete = torch.cat([scaler.state(batch["labeled_states"]), scaler.action(batch["labeled_actions"]),
                          scaler.state(batch["labeled_next_states"])], 1)
    incomplete = torch.cat([scaler.state(batch["states"]), batch["states"].new_zeros((len(batch["states"]), 2)),
                            scaler.state(batch["next_states"])], 1)
    return dict(labeled=dict(real=complete, terrain=batch["labeled_terrain"],
                            observed=complete.new_ones((len(complete), 1)), states=batch["labeled_states"]),
                all=dict(real=incomplete, terrain=batch["terrain"],
                         observed=incomplete.new_zeros((len(incomplete), 1)), states=batch["states"]))


def fake_views(bundle, views, latent_rng, contact_rng, *, straight_through):
    """Prior and encoded-state GAN paths, sharing the real views' observation masks."""
    result = {}
    for name, view in views.items():
        terrain, states = view["terrain"], view["states"]
        z, _ = bundle["prior"].sample(len(states), latent_rng)
        encoded = bundle["E"](bundle["scaler"].state(states), terrain, bundle["prior"])
        result[name] = {path: contact_record(bundle["G"](code, terrain), rng=contact_rng,
                                            straight_through=straight_through)
                        for path, code in (("prior", z), ("encoded", encoded.codes[:, 0]))}
    return result


def _comparisons(d, role, views, fakes):
    names = ("labeled", "all") if role == "joint" else ("labeled",) if role == "action" else ("all",)
    for name in names:
        view = views[name]
        xr, context = d.inputs(role, view["real"], view["terrain"], view["observed"])
        for path, record in fakes[name].items():
            xf, _ = d.inputs(role, record, view["terrain"], view["observed"])
            yield f"{name}/{path}", xr, xf, context


def discriminator_loss(d, views, fakes, gan, reg, step, rngs):
    """Mean views/paths per role, sum critic roles; generated records detached here.

    ``reg`` is the shared critic module's ``recipe.make_critic_regularizer(...)``.
    """
    terms, role_losses = {}, []
    for role in d.roles():
        critic = d.critic_for(role)
        adversarial_losses, penalties = [], []
        for _, xr, xf, context in _comparisons(d, role, views, fakes):
            xr, xf, context = xr.detach(), xf.detach(), context.detach()
            dr, df = critic(xr, context)[0], critic(xf, context)[0]
            penalty, _ = reg.penalty(
                lambda x: critic(x, context)[0], xr, xf, step, generator=rngs[role],
                ema_critic=reg.ema_critic(lambda m, x: m.critic_for(role)(x, context)[0]))
            adversarial_losses.append(gan.d_loss(dr, df))
            penalties.append(penalty)
        terms[f"{role}_gan"] = torch.stack(adversarial_losses).mean()
        terms[f"{role}_penalty"] = torch.stack(penalties).mean()
        terms[role] = terms[f"{role}_gan"] + terms[f"{role}_penalty"]
        role_losses.append(terms[role])
    return sum(role_losses), terms


def generator_loss(d, views, fakes, gan, marginal_weight=1.):
    """Live generated inputs, fixed critic weights managed by caller; no cycle loss."""
    terms = {}
    for role in d.roles():
        critic = d.critic_for(role)
        losses = []
        for _, xr, xf, context in _comparisons(d, role, views, fakes):
            with torch.no_grad():
                real_logits = critic(xr, context)[0]
            losses.append(gan.g_loss(critic(xf, context)[0], real_logits))
        terms[role] = torch.stack(losses).mean()
    result = terms["joint"]
    if len(terms) > 1:
        result = result + marginal_weight * sum(v for k,v in terms.items() if k != "joint") / 3
    return result, terms


def load_gan_control_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "gym_gan_control_v1" or saved.get("gan_training") is not True:
        raise ValueError("Expected a GAN-trained gym_gan_control_v1 checkpoint")
    if saved.get("gan_steps") != saved["step"] or saved["gan_steps"] < 1:
        raise ValueError("GAN updates must run throughout training")
    bundle = build_gan_models(saved["config"], GymTransitionScaler(**saved["scaler"]), device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(step=saved["step"], provenance=saved["provenance"], validation=saved["validation"],
                  gan_training=True, gan_steps=saved["gan_steps"])
    summary = Path(path).parent / "summary.json"
    if summary.exists():
        bundle["training_summary"] = json.loads(summary.read_text())
    return bundle
