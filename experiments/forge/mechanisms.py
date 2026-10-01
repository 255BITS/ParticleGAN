"""Observe actual public component branches and exercise delayed hooks cheaply.

Synthetic checks use explicit tiny tensors and declared optimizer state. They
prove a hook can execute, never earn quality/continuation evidence, and do not
consume a campaign's data, initialization, prior, or evaluation RNG streams.
"""
from copy import deepcopy
import math

import torch
from torch import nn


NAMES = ("critic_penalty", "critic_anchor", "critic_guard", "a2", "direct_particle_gain")


class _ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, dtype=torch.float64))

    def forward(self, values):
        return values @ self.weight


def _count(value):
    return int(value.detach().item()) if isinstance(value, torch.Tensor) else int(value)


def _probe(recipe, name):
    """Run the declared recipe's real hook against a known activating state."""
    if name in {"critic_guard", "critic_anchor"}:
        model = _ScalarCritic()
        optimizer = recipe.make_critic_optimizer(model, ema_critic=deepcopy(model))
        parameter = model.weight
        if name == "critic_guard":
            step = max(1, recipe.d_guard_min_steps)
            beta2 = optimizer.param_groups[0]["betas"][1]
            optimizer.state[parameter] = {"step": torch.tensor(float(step)),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.full_like(parameter, 1. - beta2 ** step)}
            parameter.grad = torch.full_like(parameter, recipe.d_guard_ratio * 2)
            optimizer.step()
            applied = optimizer.guard.clipped_tensors
            passed = applied > 0 and bool(torch.isfinite(parameter).all())
            measurements = {"clipped_tensors": applied, "optimizer_updates": 1,
                            "initial_adam_step": step}
        else:
            penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
            optimizer.record.lr_max, optimizer.record.lr_last = 1., 0.
            optimizer.record.observed_steps = recipe.reg_every - 1
            values = torch.tensor([[1.], [-1.]], dtype=torch.float64)
            penalty(model, values, -values)  # Start the public anchor.
            with torch.no_grad():
                parameter.add_(1.)
            value = penalty(model, values, -values)
            stats = penalty.last_stats
            passed = stats.get("applied") is True and stats.get("prox", 0) > 0 and bool(torch.isfinite(value))
            measurements = {"prox": stats.get("prox"), "phase": stats.get("phase"),
                            "penalty_calls": 2, "optimizer_updates": 0,
                            "declared_lr_ratio": 0.}
    else:
        parameter = nn.Parameter(torch.zeros(4, 1, dtype=torch.float64))
        options = ({"latent_table": parameter, "betas": recipe.prior_betas or recipe.betas}
                   if name == "a2" else {"direct_particles": [parameter]})
        optimizer = recipe.make_generator_optimizer([parameter], **options)
        parameter.grad = torch.zeros_like(parameter)
        parameter.grad[0] = 1.
        optimizer.step()
        if name == "a2":
            damping = optimizer.latent_damping
            # Simulate long sparse history without running that many updates.
            damping.total = max(damping.total, math.ceil(4 / recipe.latent_damping_max_rate))
            parameter.grad[0] = -1.
            applied = []
            original = damping.begin
            def begin(actual):
                token = original(actual)
                applied.append(token is not None)
                return token
            damping.begin = begin
            optimizer.step()
            passed = applied == [True] and bool(torch.isfinite(parameter).all())
            measurements = {"damping_branch_applied": applied == [True], "optimizer_updates": 2,
                            "initial_sparse_history_total": damping.total - 4}
        else:
            optimizer.step()
            gain = optimizer.direct_response.last_gain
            passed = gain > 1. and bool(torch.isfinite(parameter).all())
            measurements = {"gain": gain, "optimizer_updates": 2}
    return {"status": "PASS" if passed else "FAIL", "kind": "synthetic_public_component",
            "synthetic_state": True, "training_evidence": False, "measurements": measurements}


class MechanismAudit:
    """Instance-local observation; wrapped methods return the original result."""
    def __init__(self, recipe, critic_optimizer, generator_optimizers):
        self.recipe = recipe
        self.rows = {name: dict(requested=False, enabled=False, calls=0, eligible=0, applied=0)
                     for name in NAMES}
        penalty = self.rows["critic_penalty"]
        penalty.update(requested=recipe.reg_coeff > 0, enabled=recipe.reg_coeff > 0)
        anchor_requested = (recipe.reg_arm == "k3p" and recipe.reg_coeff > 0
                            and recipe.reg_anchor_weight > 0)
        self.rows["critic_anchor"].update(requested=anchor_requested,
                                          enabled=critic_optimizer.anchor is not None and anchor_requested)
        self.rows["critic_guard"].update(requested=recipe.d_guard_ratio > 0,
                                         enabled=critic_optimizer.guard is not None)
        self.rows["a2"]["requested"] = recipe.latent_damping_max_rate > 0
        self.rows["direct_particle_gain"]["requested"] = recipe.direct_particle_gain
        guard = critic_optimizer.guard
        if guard is not None:
            original = guard.apply_
            def apply(optimizer):
                row = self.rows["critic_guard"]
                row["calls"] += 1
                eligible = any(p.grad is not None and "exp_avg_sq" in optimizer.state.get(p, {})
                               and _count(optimizer.state[p]["step"]) >= guard.min_steps
                               for group in optimizer.param_groups for p in group["params"])
                row["eligible"] += int(eligible)
                result = original(optimizer)
                row["applied"] += int(_count(result) > 0)
                return result
            guard.apply_ = apply
        for optimizer in generator_optimizers:
            for name, mechanism in (("a2", optimizer.latent_damping),
                                     ("direct_particle_gain", optimizer.direct_response)):
                if mechanism is None:
                    continue
                self.rows[name]["enabled"] = self.rows[name]["requested"]
                self._observe_begin(name, mechanism)

    def _observe_begin(self, name, mechanism):
        original = mechanism.begin
        def begin(optimizer):
            row = self.rows[name]
            row["calls"] += 1
            result = original(optimizer)
            row["eligible"] += int(result is not None)
            row["applied"] += int(result is not None and (name != "direct_particle_gain" or mechanism.last_gain > 1.))
            return result
        mechanism.begin = begin

    def observe_penalty(self, stats):
        penalty = self.rows["critic_penalty"]
        penalty["calls"] += 1
        applied = stats.get("applied") is True
        penalty["eligible"] += int(applied)
        penalty["applied"] += int(applied and penalty["requested"])
        anchor = self.rows["critic_anchor"]
        anchor["calls"] += 1
        anchor["eligible"] += int(applied and stats.get("phase") in {"blend", "b"})
        anchor["applied"] += int(applied and anchor["requested"] and stats.get("prox", 0.) > 0)

    def receipt(self):
        rows = deepcopy(self.rows)
        for name, row in rows.items():
            if not row["requested"]:
                row["reason"] = "disabled by the declared formulation; no activation claimed"
            elif row["applied"]:
                row["reason"] = "actual host mechanism branch applied"
            elif name == "critic_penalty":
                row["reason"] = "requested penalty did not apply within the frozen host budget"
            else:
                row["reason"] = ("host has no applicable component" if not row["enabled"] else
                                 "host conditions did not activate the branch; targeted component check required")
                try:
                    row["probe"] = _probe(self.recipe, name)
                except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as error:
                    row["probe"] = {"status": "BLOCKED", "kind": "synthetic_public_component",
                                    "training_evidence": False, "error": str(error)}
        return {"schema_version": 1, "mechanisms": rows}


def mechanism_blockers(audit):
    """Validate activation receipts without trusting a single PASS boolean."""
    if not isinstance(audit, dict) or not isinstance(audit.get("mechanisms"), dict):
        return ["missing per-mechanism activation evidence"]
    rows = audit["mechanisms"]
    if set(rows) != set(NAMES):
        return ["incomplete per-mechanism activation evidence"]
    blockers = []
    for name, row in rows.items():
        if not isinstance(row, dict) or any(type(row.get(k)) is not bool for k in ("requested", "enabled")):
            blockers.append(f"{name}: invalid activation flags")
            continue
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("calls", "eligible", "applied")):
            blockers.append(f"{name}: invalid activation counters")
            continue
        if row["applied"] > row["eligible"] or row["eligible"] > row["calls"]:
            blockers.append(f"{name}: inconsistent activation counters")
            continue
        if not row["requested"]:
            if row["applied"] or row["enabled"]:
                blockers.append(f"{name}: disabled mechanism claimed activation")
            continue
        if row["enabled"] and row["applied"]:
            continue
        probe = row.get("probe", {})
        probe = probe if isinstance(probe, dict) else {}
        measured = probe.get("measurements", {})
        measured = measured if isinstance(measured, dict) else {}
        actual = (name == "critic_guard" and type(measured.get("clipped_tensors")) is int and measured["clipped_tensors"] > 0
                  or name == "critic_anchor" and type(measured.get("prox")) in (int, float) and math.isfinite(measured["prox"]) and measured["prox"] > 0
                  or name == "a2" and measured.get("damping_branch_applied") is True
                  or name == "direct_particle_gain" and type(measured.get("gain")) in (int, float) and math.isfinite(measured["gain"]) and measured["gain"] > 1.)
        if not (probe.get("status") == "PASS" and probe.get("kind") == "synthetic_public_component"
                and probe.get("synthetic_state") is True and probe.get("training_evidence") is False and actual):
            blockers.append(f"{name}: requested mechanism activation has not been demonstrated")
    return blockers
