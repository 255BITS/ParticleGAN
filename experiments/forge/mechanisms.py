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
            if optimizer.param_groups[0]["amsgrad"]:
                optimizer.state[parameter]["max_exp_avg_sq"] = optimizer.state[parameter]["exp_avg_sq"].clone()
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
            if recipe.effective_critic_formulation == "ka2":
                from particlegan.ka2 import WARMUP_CALLS
                optimizer.record.calls = WARMUP_CALLS - 1
                optimizer.record.observed_steps = max(1, optimizer.record.observed_steps)
            values = torch.tensor([[1.], [-1.]], dtype=torch.float64)
            penalty(model, values, -values)  # Start the public anchor.
            with torch.no_grad():
                parameter.add_(1.)
            value = penalty(model, values, -values)
            stats = penalty.last_stats
            passed = stats.get("applied") is True and stats.get("prox", 0) > 0 and bool(torch.isfinite(value))
            measurements = {"prox": stats.get("prox"), "phase": stats.get("phase"),
                            "penalty_calls": 2, "optimizer_updates": 0,
                            "declared_lr_ratio": 0., "critic_formulation": recipe.effective_critic_formulation}
            if recipe.effective_critic_formulation == "ka2":
                measurements.update(initial_penalty_calls=WARMUP_CALLS - 1,
                                    initial_observed_optimizer_steps=max(1, recipe.reg_every - 1))
    else:
        parameter = nn.Parameter(torch.zeros(4, 1, dtype=torch.float64))
        options = ({"latent_table": parameter, "betas": recipe.prior_betas or recipe.betas,
                    "eps": recipe.eps if recipe.prior_eps is None else recipe.prior_eps}
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


class _PenaltyCadenceAudit:
    """Compact observations of existing scalar state; never evaluate a model.

    Hosts observe penalties both before and after the optimizer step. A passive
    post-step hook records which applied calls that step completed, so an
    application can be labelled with its actual one-based critic update in both
    cases. Counters from the controller remain separate from host observations:
    a host may report only the last of several penalty calls on a shared critic.
    """
    _RANGES = {"s_thermal": "_thermal_scale", "ema_residual": "ema_residual",
               "ema_data_scale": "ema_data_scale"}

    def __init__(self, recipe, optimizer):
        self.recipe, self.optimizer = recipe, optimizer
        self.resets = -1
        self.reset("attachment")
        self._step_hook = optimizer.register_step_post_hook(self._after_step)
        record = getattr(optimizer, "record", None)
        if record is not None:
            self._load_record = record.load_state_dict
            record.load_state_dict = self._after_record_load

    def _counter(self, name):
        value = getattr(getattr(self.optimizer, "record", None), name, None)
        # Do not call item(), coerce tensors, or add device synchronization.
        return value if type(value) is int and value >= 0 else None

    def reset(self, reason):
        """Start a local span; restored lifetime counters are never new calls."""
        self.resets += 1
        self.start_reason = reason
        self.start_steps = self._counter("observed_steps")
        self.start_calls = self._counter("calls")
        self.completed_calls = self.start_calls
        self.observations = dict(calls=0, applied=0, skipped=0, unavailable=0)
        self.first_application = self.last_application = None
        self.unlabelled_applications = 0
        self.phases = {}
        self.ranges = {name: dict(count=0, min=None, max=None, last=None,
                                  unavailable=0, nonfinite=0)
                       for name in self._RANGES}

    def _after_step(self, optimizer, args, kwargs):
        self.completed_calls = self._counter("calls")

    def _after_record_load(self, state):
        # The public optimizers restore their controller after Torch's own
        # optimizer post-load hooks. Observe the actual completed restoration.
        result = self._load_record(state)
        self.reset("controller_state_restored")
        return result

    def observe(self, stats):
        self.observations["calls"] += 1
        applied = stats.get("applied")
        status = "applied" if applied is True else "skipped" if applied is False else "unavailable"
        self.observations[status] += 1
        if applied is True:
            steps, calls = self._counter("observed_steps"), self._counter("calls")
            update = None
            if steps is not None and calls is not None and self.completed_calls is not None:
                if calls > self.completed_calls:  # Observed before its optimizer step.
                    update = steps + 1
                elif calls == self.completed_calls and steps > 0:
                    update = steps  # Observed after its optimizer step.
            if update is None:
                self.unlabelled_applications += 1
            else:
                if self.first_application is None:
                    self.first_application = update
                self.last_application = update
        phase = stats.get("phase")
        if isinstance(phase, str):
            self.phases[phase] = self.phases.get(phase, 0) + 1
        record = getattr(self.optimizer, "record", None)
        for name, attribute in self._RANGES.items():
            value = stats.get(name, getattr(record, attribute, None))
            row = self.ranges[name]
            if type(value) not in (int, float):
                row["unavailable"] += 1
            elif not math.isfinite(value):
                row["nonfinite"] += 1
            else:
                row["count"] += 1
                row["min"] = value if row["min"] is None else min(row["min"], value)
                row["max"] = value if row["max"] is None else max(row["max"], value)
                row["last"] = value

    def receipt(self):
        steps, calls = self._counter("observed_steps"), self._counter("calls")
        def delta(start, end):
            return end - start if start is not None and end is not None and end >= start else None
        return dict(schema_version=1, scope="audit_window", start_reason=self.start_reason,
            resets=self.resets, update_clock="one_based_critic_optimizer_update",
            configured=dict(k=self.recipe.reg_every, base_coefficient=self.recipe.reg_coeff,
                effective_coefficient=self.recipe.reg_coeff * self.recipe.reg_every,
                coefficient_scope="initial_recipe_before_any_schedule",
                scheduled_end_base_coefficient=self.recipe.reg_coeff_end),
            observations=deepcopy(self.observations),
            optimizer=dict(observed_steps_start=self.start_steps, observed_steps_end=steps,
                observed_steps_in_window=delta(self.start_steps, steps),
                applied_record_calls_start=self.start_calls, applied_record_calls_end=calls,
                applied_record_calls_in_window=delta(self.start_calls, calls)),
            first_application_update=self.first_application,
            last_application_update=self.last_application,
            application_updates_unavailable=self.unlabelled_applications,
            phase_counts=deepcopy(self.phases), ranges=deepcopy(self.ranges))


class MechanismAudit:
    """Instance-local observation; wrapped methods return the original result."""
    def __init__(self, recipe, critic_optimizer, generator_optimizers):
        self.recipe = recipe
        self.critic_optimizer = critic_optimizer
        self.penalty_cadence = _PenaltyCadenceAudit(recipe, critic_optimizer)
        self.rows = {name: dict(requested=False, enabled=False, calls=0, eligible=0, applied=0)
                     for name in NAMES}
        penalty = self.rows["critic_penalty"]
        penalty_requested = max(recipe.reg_coeff, recipe.reg_coeff_end or 0.) > 0
        penalty.update(requested=penalty_requested, enabled=penalty_requested)
        anchor_requested = (recipe.reg_arm in (None, "k3p") and recipe.reg_coeff > 0
                            and recipe.reg_anchor_weight > 0)
        self.rows["critic_anchor"].update(requested=anchor_requested,
                                          enabled=critic_optimizer.anchor is not None and anchor_requested)
        self.rows["critic_guard"].update(requested=recipe.d_guard_ratio > 0,
                                         enabled=critic_optimizer.guard is not None)
        self.rows["a2"]["requested"] = recipe.latent_damping_max_rate > 0 and recipe.prior_update == "learned"
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
        stats = stats if isinstance(stats, dict) else {}
        self.penalty_cadence.observe(stats)
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
        if self.recipe.critic_step_mode == 'finite_cap':
            stats = getattr(self.critic_optimizer, 'critic_cap_stats', {})
            steps = stats.get('steps', 0)
            rows['critic_step_damping'] = dict(requested=True, enabled=bool(stats),
                calls=steps, eligible=steps, applied=steps,
                reason='Actual original-training-panel finite cap checked on every critic update',
                stats=deepcopy(stats))
        return {"schema_version": 1, "mechanisms": rows,
                "penalty_cadence": self.penalty_cadence.receipt()}


def mechanism_blockers(audit):
    """Validate activation receipts without trusting a single PASS boolean."""
    if not isinstance(audit, dict) or not isinstance(audit.get("mechanisms"), dict):
        return ["missing per-mechanism activation evidence"]
    rows = audit["mechanisms"]
    if set(rows) not in (set(NAMES), set(NAMES)|{'critic_step_damping'}):
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
