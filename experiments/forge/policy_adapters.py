"""Observed public-policy execution, health and selected-state identity.

This module does not implement an optimizer or decide a quality gate.  Its
instance-local observers delegate to the original public lifecycle.  Explicit
policy task variants use these receipts; historical live-weight tasks do not.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import struct

import torch

from particlegan import UpdatePolicy


DIGEST_KIND = "typed_policy_state_v1"
HOOKS = ("begin_step", "after_critic_step", "after_generator_backward",
         "after_generator_step", "finish_step")


def typed_state_digest(value):
    """Hash complete typed state, retaining unavailable diagnostic sentinels.

    Unlike JSON, IEEE float bytes can retain a source-defined NaN or -Inf.
    Hashability supplies identity only; ``finite_policy_state`` separately
    rejects nonfinite learned parameters, optimizers and unknown control fields.
    """
    digest = hashlib.sha256(b"forge-typed-policy-state-v1\0")

    def write(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"T")
            tensor = item.detach().cpu().contiguous()
            write((str(tensor.dtype), tuple(tensor.shape)))
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
        elif isinstance(item, dict):
            digest.update(b"D" + len(item).to_bytes(8, "big"))
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                write(key); write(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update((b"L" if isinstance(item, list) else b"U") + len(item).to_bytes(8, "big"))
            for child in item:
                write(child)
        elif type(item) is float:
            digest.update(b"F" + struct.pack("!d", item))
        elif item is None or type(item) in (str, int, bool):
            payload = repr((type(item).__name__, item)).encode()
            digest.update(b"S" + len(payload).to_bytes(8, "big") + payload)
        else:
            raise TypeError(f"unsupported policy checkpoint value: {type(item).__name__}")
    write(value)
    return digest.hexdigest()


def finite_policy_state(value, path=()):
    """Strict learned-state health with narrow public LR diagnostic exceptions.

    SettleTest stores NaN for masked/no-evidence displacement/cosine values;
    unavailable last/last_look statistics also use NaN.  A zero-variance t may
    be infinite, and the early log-BF is -Inf before sufficient evidence.
    These named leaves remain untouched and never become a quality PASS.
    The public policy loader still owns schema validation on restoration.
    """
    # Both GANTrainer and caller-owned UpdatePolicy checkpoint envelopes are
    # supported.  Normalize only those exact public envelope prefixes.
    normalized = path
    if path and path[0] in {"trainer", "policy"}:
        normalized = path[1:]
    settle = (len(normalized) >= 4 and normalized[0] == "lr_settle"
              and type(normalized[1]) is int and type(normalized[2]) is int)
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point() and not value.is_complex():
            return True
        masked = settle and normalized[3] in {"r_b", "r_2b", "blocks", "last_block"}
        return bool((~torch.isinf(value)).all()) if masked else bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_policy_state(child, path + (key,)) for key, child in value.items())
    if isinstance(value, (tuple, list)):
        return all(finite_policy_state(child, path + (index,)) for index, child in enumerate(value))
    if not isinstance(value, float) or math.isfinite(value):
        return True
    diagnostic = (settle and len(normalized) >= 5
                  and normalized[3] in {"last", "last_look", "log"}
                  and normalized[-1] in {"t_b", "t_2b", "mean_r_b", "mean_r_2b", "log_bf_b", "log_bf_2b"})
    return diagnostic and (math.isnan(value) or normalized[-1] in {"t_b", "t_2b"}
                           or value == -math.inf and normalized[-1] in {"log_bf_b", "log_bf_2b"})


def evaluation_state(state):
    """Retain all state except explicitly independent evaluation RNG streams.

    The named-stream manifest may acquire new evaluation purposes during a
    read.  Training bindings/states, global RNGs, clocks, models, averages,
    optimizer history, controllers, birth/death RNG and serving stay covered.
    """
    result = deepcopy(state)
    named = result.get("streams")
    if isinstance(named, dict) and "manifest" in named and "states" in named:
        bindings = named["manifest"]["bindings"]
        allowed = {key for key, row in bindings.items() if row["family"] == "eval"}
        named["manifest"]["bindings"] = {key: row for key, row in bindings.items() if key not in allowed}
        named["states"] = {key: row for key, row in named["states"].items() if key not in allowed}
    for owner in (result.get("trainer"), result.get("policy")):
        if isinstance(owner, dict) and isinstance(owner.get("streams"), dict):
            owner["streams"].pop("eval_generator", None)
    return result


class PolicyLifecycleAudit:
    """Count successful real public hooks without changing their return values."""
    def __init__(self, policy):
        if not isinstance(policy, UpdatePolicy):
            raise TypeError("a policy audit requires a real particlegan.UpdatePolicy")
        if getattr(policy, "_forge_lifecycle_audit", None) is not None:
            raise ValueError("one observer owns this policy lifecycle")
        originals = {name: getattr(policy, name) for name in HOOKS}
        if any(getattr(method, "__func__", None) is not getattr(UpdatePolicy, name)
               for name, method in originals.items()):
            raise ValueError("policy audit must delegate the unmodified public lifecycle methods")
        self.policy = policy
        self.start_steps = policy.completed_steps
        self.calls = {name: 0 for name in HOOKS}
        self.pending = []
        self.order_errors = 0
        self.last_order = []
        for name in HOOKS:
            original = originals[name]
            def observed(*args, _name=name, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                self.calls[_name] += 1
                self.pending.append(_name)
                if _name == "finish_step":
                    self.last_order = list(self.pending)
                    self.order_errors += int(tuple(self.pending) != HOOKS)
                    self.pending.clear()
                return result
            setattr(policy, name, observed)
        policy._forge_lifecycle_audit = self

    def reset_after_restore(self):
        """Start a separately attested continuation interval before any step."""
        if any(self.calls.values()) or self.pending:
            raise ValueError("cannot relabel already observed lifecycle calls")
        self.start_steps = self.policy.completed_steps

    def receipt(self, completed_steps):
        expected = completed_steps - self.start_steps
        return {"owner": "particlegan.UpdatePolicy", "kind": "instance_local_successful_public_hooks",
                "start_completed_steps": self.start_steps, "end_completed_steps": completed_steps,
                "observed_updates": expected, "calls": dict(self.calls),
                "last_order": list(self.last_order), "order_errors": self.order_errors,
                "pending": list(self.pending),
                "complete": (type(expected) is int and expected >= 0 and not self.pending
                             and not self.order_errors and all(v == expected for v in self.calls.values()))}


def _json_diagnostics(value):
    """Display unavailable diagnostics as null; raw typed state is preserved."""
    if isinstance(value, torch.Tensor):
        return _json_diagnostics(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _json_diagnostics(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_diagnostics(child) for child in value]
    if type(value) is float and not math.isfinite(value):
        return None
    return value


def routed_owner_receipt(policy):
    """Read the actual guarded-forward owner without calling it or drawing.

    Declaration metadata alone cannot establish that protected contexts and
    mass-aware routing are attached to the executing public policy.
    """
    routed = policy.routed_control
    if routed is None:
        return None
    state = routed.state_dict()
    callback = routed.spec.model_forward
    return {
        "schema_version": 1,
        "owner": "particlegan.routing.RoutedRowControl",
        "config": _json_diagnostics(routed.spec.to_dict()),
        "model_forward": {
            "callable": callable(callback),
            "module": getattr(callback, "__module__", None),
            "qualname": getattr(callback, "__qualname__", None),
        },
        "model_roles": sorted(routed.models),
        "table_matches_policy": routed.table is policy.table,
        "averaged_table_matches_policy": routed.averaged_table is policy.averaged_table,
        "table_shape": list(routed.table.shape),
        "row_ownership": _json_diagnostics(state["row_ownership"]),
        "fit_fill": routed.fit_fill,
        "guard_fill": routed.guard_fill,
        "probe_clock": dict(routed.probe_clock),
        "counters": dict(routed.counters),
        "state_sha256": typed_state_digest(state),
        "state_digest_kind": DIGEST_KIND,
        "counter_credit": "observed_only_no_claim_that_a_proposal_or_move_occurred",
    }


def controls_receipt(policy, completed_steps):
    """Read actual enabled owners/counters; inactive branches earn no firing."""
    if not isinstance(policy, UpdatePolicy):
        raise TypeError("controls require an actual particlegan.UpdatePolicy")
    if type(completed_steps) is not int or completed_steps != policy.completed_steps:
        raise ValueError("public policy and observed execution cursor disagree")
    recipe = policy.recipe
    audit = getattr(policy, "_forge_lifecycle_audit", None)
    lifecycle = (audit.receipt(completed_steps) if isinstance(audit, PolicyLifecycleAudit) else
                 {"owner": "particlegan.UpdatePolicy", "complete": False,
                  "reason": "no instance-local observation of actual public hooks"})
    row_updates = None if policy.row_evidence is None else policy.row_evidence.counters.get("updates", 0)
    birth = None if policy.birth_death is None else policy.birth_death.diagnostics()
    selection = None if policy._feature_selection is None else policy._feature_selection.state_dict()
    requested = {"continuous_controller": recipe.continuous_policy is not None,
                 "stationarity_lr": recipe.lr_control == "stationarity",
                 "row_evidence": recipe.row_evidence_gate,
                 "birth_death": recipe.particle_birth_death,
                 "learned_output_noise": recipe.output_noise_mode == "learnable",
                 "selected_averaging": recipe.serve_average > 0,
                 "optimizer_surprise": recipe.reopen_signal == "optimizer" and recipe.lr_control == "stationarity",
                 "reopen_guard": recipe.reopen_guard is not None}
    enabled = {"continuous_controller": policy.controller is not None,
               "stationarity_lr": policy.lr_settle is not None,
               "row_evidence": policy.row_evidence is not None,
               "birth_death": policy.birth_death is not None,
               "learned_output_noise": policy.log_output_sigma is not None,
               "selected_averaging": recipe.serve_average > 0,
               "optimizer_surprise": policy.surprise is not None,
               "reopen_guard": policy.reopen_guard is not None}
    satisfied = all(not flag or enabled[name] for name, flag in requested.items())
    # This is implementation evidence, not an assertion that a move, reopen or
    # stationary decision occurred.  Their original counters remain explicit.
    diagnostics = {"controller": None if policy.controller is None else policy.controller.diagnostics(),
                   "stationarity_lr": None if policy.lr_settle is None else policy.lr_settle.diagnostics(),
                   "row_evidence": None if policy.row_evidence is None else dict(policy.row_evidence.counters),
                   "birth_death": birth,
                   "surprise": None if policy.surprise is None else policy.surprise.diagnostics(),
                   "reopen_guard": None if policy.reopen_guard is None else policy.reopen_guard.state_dict(),
                   "backend_selection": selection}
    parameters = [value for module in policy._training_modules().values()
                  for value in module.parameters()]
    devices = sorted({str(value.device) for value in parameters})
    floating_dtypes = sorted({str(value.dtype) for value in parameters if value.is_floating_point()})
    device_types = {value.device.type for value in parameters}
    execution = {"model_devices": devices, "floating_dtypes": floating_dtypes,
                 "autocast_enabled": any(torch.is_autocast_enabled(kind) for kind in device_types)}
    receipt = {"schema_version": 1, "cohort": "policy_selected_cloud_v1", "lifecycle": lifecycle,
            "requested": requested, "enabled": enabled, "requested_owners_bound": satisfied,
            "row_evidence_observations": row_updates,
            "row_semantics": policy.row_semantics, "roles": deepcopy(policy.roles),
            "completed_steps": completed_steps,
            "execution": execution,
            "served_source": policy.served_snapshot()["source"],
            "output_sigma": policy.output_sigma(),
            "effective_group_lrs": [[float(group["lr"]) for group in optimizer.param_groups]
                                    for optimizer in policy.optimizers],
            "diagnostics": _json_diagnostics(diagnostics),
            "diagnostic_nulls": "source-defined unavailable diagnostics; exact sentinels retained in checkpoint",
            "implementation_observed": bool(lifecycle["complete"] and satisfied),
            "quality_qualification": False}
    if policy.routed_control is not None:
        receipt["routed_owner"] = routed_owner_receipt(policy)
    return receipt


def observation_receipt(task, policy):
    declaration = task["evaluation"]["policy_observation"]
    selected = policy.served_snapshot()
    return {**deepcopy(declaration), "observed": True, "selected_source": selected["source"],
            "completed_steps": policy.completed_steps, "policy_owner": "particlegan.UpdatePolicy",
            "controller": None if policy.controller is None else policy.controller.variant,
            "backend_selection": _json_diagnostics(selected.get("backend_selection")),
            "snapshot_sha256": typed_state_digest(selected)}
