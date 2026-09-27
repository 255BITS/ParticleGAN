"""Host released K3P on the 9 custom-loop transfer hosts (benchmarks.transfer_suite 'legacy' runner).

Why: on develop these hosts build ``GradientPenalty(arm='k3p')`` themselves and never feed its LR record,
so the legacy K3P rule raises ("after_critic_step(critic_optimizer) was never called"), and their optimizers
are plain Adam, so K3P's optimizer-side work (spike guard, A2 latent damping, direct-particle response) is
absent. The frozen gap-fill qualification grafted these (its regularizer/response/latent receipts); this
module does the same with particlegan.k3p's own classes, configured from the arm's recipe.

  graft = legacy_k3p_graft.install(recipe)   # recipe = particlegan.get_recipe(**arm_overrides); before the host runs
  ...run the host (toy100_compatibility.run(..., tasks=(name,)))...
  receipt = graft()                          # guard clips, direct-gain stats, latent state, record steps

Pieces:
  * critic penalty: on a GradientPenalty(arm='k3p')'s first call, an EMA-critic anchor (CriticAnchor, the
    recipe's reg_anchor_decay) + CriticStepRecord, blend floor = resolved network LR floor (<.5) else 0;
    after_critic_step after every step of the optimizer holding that critic's params. Hosts that pass a
    closure critic (``lambda z: critic.score(z, s)``: unipolar, mid_scale_identity) get the same closure
    rebuilt over the EMA module as ``ema_critic``.
  * optimizers: roles from the hosts' own local names (opt_d = critic; opt_g/opt_p/opt = generator side,
    the benchmark bridge's rule); group kinds from the harness's '_comparison_prior' marks. Critic ->
    CriticSpikeGuard before the step. Generator side -> LatentRowDamping on a ParticlePrior table group
    (alone, beta1 0), DirectParticleResponse on another prior-kind group (direct particles).
Smoke (k3p_stock, new init hook): two_pole FAIL (mean_abs .12) without the optimizer graft -> PASS .625
(frozen K3P .645); unipolar/mid_scale_identity ERROR -> PASS with the closure handling.
Device: run these hosts on CPU. On develop unused_token_hold errors on CUDA (host buffer left on CPU) even
with the harness's own default recipe.
"""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import torch
from torch import nn

_HERE = str(Path(__file__).resolve())


def install(recipe):
    from particlegan import ParticlePrior
    from particlegan.k3p import CriticAnchor, CriticSpikeGuard, DirectParticleResponse, LatentRowDamping
    from benchmarks.legacy.grad_regularizers import CriticStepRecord, GradientPenalty
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    floor = recipe.resolved_network_lr_floor
    blend_floor = floor if floor < 0.5 else 0.0
    regs, state = [], {}
    info = {"blend_lr_floor": blend_floor, "anchor_decay": recipe.reg_anchor_decay, "penalties": 0,
            "record_steps": 0, "guard_ratio": recipe.d_guard_ratio,
            "latent_max_rate": recipe.latent_damping_max_rate,
            "direct_betas": list(recipe.direct_particle_betas), "direct_gain": recipe.direct_particle_gain,
            "optimizers": []}

    # ---------------------------------------------------------------- critic penalty: anchor + LR record
    inner = GradientPenalty.penalty

    def owner(D):
        if isinstance(D, nn.Module):
            return D, None
        cells = [c for c in (getattr(D, "__closure__", None) or ()) if isinstance(c.cell_contents, nn.Module)]
        if len(cells) != 1:
            raise TypeError("k3p graft: critic callable must close over exactly one nn.Module")
        return cells[0].cell_contents, cells[0]

    def penalty(self, D, *args, **kwargs):
        if self.arm != "k3p" or (self.anchor is not None and not getattr(self, "_graft", False)):
            return inner(self, D, *args, **kwargs)  # recipe-built penalties (GANTrainer hosts) untouched
        module, cell = owner(D)
        if not getattr(self, "_graft", False):
            anchor = CriticAnchor(module, copy.deepcopy(module).requires_grad_(False), decay=recipe.reg_anchor_decay)
            self.record, self.anchor, self._graft = CriticStepRecord(anchor), anchor, True
            self.lr_floor = blend_floor
            regs.append((self, {id(p) for p in module.parameters()}))
            info["penalties"] += 1
            info["critic_callable"] = cell is not None
        if cell is not None:
            ema = self.anchor.ema_critic
            closure = tuple(types.CellType(ema) if c is cell else c for c in D.__closure__)
            kwargs["ema_critic"] = types.FunctionType(D.__code__, D.__globals__, D.__name__, D.__defaults__, closure)
        return inner(self, D, *args, **kwargs)
    GradientPenalty.penalty = penalty

    # ---------------------------------------------------------------- optimizer side
    tables = set()
    prior_init = ParticlePrior.__init__

    def tracked_prior_init(self, *a, **k):
        prior_init(self, *a, **k)
        tables.add(id(self.z))
    ParticlePrior.__init__ = tracked_prior_init

    def role_of(opt):
        frame = sys._getframe(1)
        while frame is not None:
            name = frame.f_code.co_filename
            if name == _HERE or "/torch/" in name:
                frame = frame.f_back
                continue
            loc = frame.f_locals
            if loc.get("opt_d") is opt:
                return "d"
            if any(loc.get(n) is opt for n in ("opt_g", "opt_p", "opt")):
                return "g"
            frame = frame.f_back
        return None

    def setup(opt):
        role = role_of(opt)
        st = {"role": role, "guard": None, "latent": None, "direct": None, "tokens": None, "gains": []}
        desc = {"role": role, "groups": len(opt.param_groups)}
        if role == "d" and recipe.d_guard_ratio > 0:
            st["guard"] = CriticSpikeGuard(ratio=recipe.d_guard_ratio, min_steps=recipe.d_guard_min_steps)
            desc["guard"] = True
        elif role == "g":
            for group in opt.param_groups:
                if not group.get("_comparison_prior", False):
                    continue
                params = group["params"]
                if len(params) == 1 and id(params[0]) in tables:
                    table = params[0]
                    if (recipe.latent_damping_max_rate > 0 and group["betas"][0] == 0.0 and table.dim() == 2
                            and st["latent"] is None):
                        st["latent"] = LatentRowDamping(table, torch.zeros_like(table, requires_grad=False),
                                                        max_rate=recipe.latent_damping_max_rate)
                        desc["latent_rows"] = table.shape[0]
                elif st["direct"] is None:
                    history = torch.zeros(sum(p.numel() for p in params), dtype=params[0].dtype,
                                          device=params[0].device)
                    st["direct"] = DirectParticleResponse(params, history, betas=recipe.direct_particle_betas,
                                                          gain=recipe.direct_particle_gain)
                    desc["direct_numel"] = history.numel()
        info["optimizers"].append(desc)
        st["desc"] = desc
        return st

    def pre(opt, args, kwargs):
        st = state.get(id(opt))
        if st is None:
            st = state[id(opt)] = setup(opt)
        if st["guard"] is not None:
            st["guard"].apply_(opt)
        latent = None if st["latent"] is None else st["latent"].begin(opt)
        direct = None if st["direct"] is None else st["direct"].begin(opt)
        st["tokens"] = (latent, direct)

    def post(opt, args, kwargs):
        st = state.get(id(opt))
        if st is not None and st["tokens"] is not None:
            latent, direct = st["tokens"]
            if st["direct"] is not None:
                st["direct"].end(direct)
                st["gains"].append(float(st["direct"].last_gain))
            if st["latent"] is not None:
                st["latent"].end(latent)
            st["tokens"] = None
        first = id(opt.param_groups[0]["params"][0])
        for reg, ids in regs:
            if first in ids:
                reg.after_critic_step(opt)
                info["record_steps"] += 1
    register_optimizer_step_pre_hook(pre)
    register_optimizer_step_post_hook(post)

    def receipt():
        for st in state.values():
            desc = st["desc"]
            if st["guard"] is not None:
                desc["guard_clipped_tensors"] = st["guard"].clipped_tensors
            if st["latent"] is not None:
                desc["latent_state"] = st["latent"].state_dict()
            if st["gains"]:
                g = st["gains"]
                desc.update(direct_gain_mean=sum(g) / len(g), direct_gain_max=max(g), direct_calls=len(g))
        return info
    return receipt
