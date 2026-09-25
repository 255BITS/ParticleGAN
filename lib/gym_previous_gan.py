"""Scratch GAN: E(st, at-1, terrain) -> z -> G1 / G2 / G3."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from lib.gym_state_control import build_state_models, parameter_hashes
from lib.gym_transition import (GymTransitionEncoder, GymTransitionCritics,
    GymTransitionScaler, contact_record, state_reconstruction)
from lib.gym_control import control_action_details as legacy_action_details

MODULE_KEYS = ('G', 'E', 'prior', 'D')


def build_models(cfg, scaler, device='cpu'):
    bundle = build_state_models(cfg, scaler, device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'] + 102)
        # Same network type as the old control encoder; trained for previous actions.
        bundle['E'] = GymTransitionEncoder(cfg['z_dim'], cfg['encoder_width'], cfg['context_dim']).to(device)
        torch.random.default_generator.manual_seed(cfg['seed'] + 103)
        bundle['D'] = GymTransitionCritics(cfg['d_width'], cfg['marginal_width'], cfg['context_dim']).to(device)
    return bundle


def hashes(bundle):
    result = parameter_hashes(bundle)
    for name, critic in bundle['D'].critics.items():
        result['D.' + name] = parameter_hashes(dict(G=critic, E=critic, prior=critic))['G']
    return result


def encode(bundle, states, previous_actions, terrain):
    scaler = bundle['scaler']
    return bundle['E'](torch.cat([scaler.state(states), scaler.action(previous_actions)], 1),
                       terrain, bundle['prior'])


def decode(bundle, states, previous_actions, terrain):
    encoding = encode(bundle, states, previous_actions, terrain)
    return bundle['G'](encoding.codes[:, 0], terrain), encoding


def fake_paths(bundle, batch, latent_rng, contact_rng, straight_through=False):
    z, _ = bundle['prior'].sample(len(batch['states']), latent_rng)
    decoded, _ = decode(bundle, batch['states'], batch['previous_actions'], batch['terrain'])
    outputs = dict(prior=bundle['G'](z, batch['terrain']), encoded=decoded)
    return {k: contact_record(v, rng=contact_rng, straight_through=straight_through)
            for k, v in outputs.items()}, decoded


def real_record(bundle, batch):
    return bundle['scaler'](torch.cat([batch[k] for k in ('states', 'actions', 'next_states')], 1))


def task_loss(decoded, real, cfg):
    state, _ = state_reconstruction(decoded[:, :8], real[:, :8], cfg['continuous_weight'], cfg['contact_weight'])
    successor, _ = state_reconstruction(decoded[:, 10:], real[:, 10:], cfg['continuous_weight'], cfg['contact_weight'])
    action = F.mse_loss(decoded[:, 8:10], real[:, 8:10])
    return (cfg['action_weight'] * action + cfg['lambda_state'] * state + cfg['lambda_next'] * successor,
            dict(action_loss=action, state_loss=state, next_loss=successor))


def adversarial_loss(d, real, fakes, terrain, gan, *, reg=None, step=1, rngs=None, marginal_weight=1.):
    """Average prior/encoded paths per role; detach all fake graphs for D updates."""
    terms, roles = {}, {}
    for role in d.roles():
        critic = d.critic_for(role)
        xr, context = d.inputs(role, real.detach(), terrain.detach())
        losses, penalties = [], []
        for fake in fakes.values():
            xf, _ = d.inputs(role, fake.detach() if reg is not None else fake, terrain.detach())
            if reg is not None:
                losses.append(gan.d_loss(critic(xr, context)[0], critic(xf, context)[0]))
                # A critic regularizer gets this role's EMA view as its anchor; a
                # plain stateless penalty (e.g. a pinned b_cap) needs none.
                anchor = ({"ema_critic": reg.ema_critic(lambda m, x: m.critic_for(role)(x, context)[0])}
                          if hasattr(reg, "ema_critic") else {})
                penalty, _ = reg.penalty(lambda x: critic(x, context)[0], xr, xf, step,
                                         generator=rngs[role], collect_stats=False, **anchor)
                penalties.append(penalty)
            else:
                with torch.no_grad():
                    dr = critic(xr, context)[0]
                losses.append(gan.g_loss(critic(xf, context)[0], dr))
        roles[role] = torch.stack(losses).mean()
        terms[role + '_gan'] = roles[role]
        if penalties:
            terms[role + '_penalty'] = torch.stack(penalties).mean()
            roles[role] = roles[role] + terms[role + '_penalty']
    total = (sum(roles.values()) if reg is not None else
             roles['joint'] + marginal_weight * sum(v for k, v in roles.items() if k != 'joint') / 3)
    return total, terms


def control_action_details(bundle, state, previous_action, terrain):
    # Alias only for the established inference helper; there is one encoder network.
    return legacy_action_details({**bundle, 'E_control': bundle['E']}, state, previous_action, terrain)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') != 'gym_previous_gan_v1' or saved.get('gan_steps') != saved['step'] or saved['step'] < 1:
        raise ValueError('Expected previous-action GAN trained at every update')
    cfg = saved['config']
    if any(cfg[k] <= 0 for k in ('action_weight', 'adversarial_weight', 'marginal_weight')):
        raise ValueError('Action L2, joint GAN, and marginal GAN must remain active')
    bundle = build_models(cfg, GymTransitionScaler(**saved['scaler']), device)
    for key in MODULE_KEYS:
        bundle[key].load_state_dict(saved[key])
        bundle[key].eval().requires_grad_(False)
    bundle.update(step=saved['step'], gan_steps=saved['gan_steps'], provenance=saved['provenance'])
    summary = Path(path).parent / 'summary.json'
    if summary.exists():
        bundle['training_summary'] = json.loads(summary.read_text())
    return bundle
