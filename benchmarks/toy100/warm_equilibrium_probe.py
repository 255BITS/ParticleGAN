"""Matched passing-state local stability filter for the frozen mode-hold host.

One single-thread CPU process trains the ordinary scheduled host to update
1,000. At its live checkpoint, Linux ``fork`` gives every candidate the exact
same in-memory generator, critic, prior, Adam moments, EMA, noise streams, and
RNG states. Each child runs updates 1,001–1,200 in the original Python host
frame. This tests stability near one passing state; it is not a from-scratch
acquisition gate or a proof that the game gradient vanishes there.

``variant_contexts`` maps a name to a factory accepting ``(state, prefix)``
and returning a context manager. The context activates only in its child
after the shared checkpoint. ``state`` contains the live host modules,
optimizers, EMA tensors, noise policy, streams, control, and
``set_step_delegate(fn)`` to replace the underlying Adam update while keeping
the probe's rate and update receipts. ``prefix_context`` can install an
optional source adapter before training; its yielded value is passed as
``prefix`` to each variant factory. Include the identity context as a parity
control. Every branch records each of updates 1,001–1,200, then every ten
updates through a requested longer horizon.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import traceback
from typing import Callable
from unittest.mock import patch

import torch

from benchmarks.transfer_suite import compare_defaults
from . import schedule as schedule_module
from .continuous_probe import FROZEN_STEPS, _window, run_probe


WARM_STEP = 1000


class _PrefixForked(Exception):
    """Unwind only the parent after children inherit the training frame."""


@contextmanager
def constant_rate_context(state: dict, *, lr: float | None = None,
                          d_multiplier: float = 1.0,
                          prior_multiplier: float = 2.0):
    """Switch the inherited policy to fixed G/D/prior rates after the fork."""
    control = state["control"]
    opt_g, opt_d = state["opt_g"], state["opt_d"]
    if opt_g not in control.base_rates or opt_d not in control.base_rates:
        raise RuntimeError("warm control did not capture both optimizer base rates")
    saved = {opt: list(rates) for opt, rates in control.base_rates.items()}
    if lr is not None:
        if not 0 < lr < 1:
            raise ValueError("constant learning rate must be positive and below one")
        control.base_rates[opt_g][:] = [lr, lr * prior_multiplier]
        control.base_rates[opt_d][:] = [lr * d_multiplier]
    with patch.object(schedule_module, "policy_multipliers",
                      lambda *args, **kwargs: (1.0, 1.0)), \
         patch.object(compare_defaults, "learning_rate_scale",
                      lambda *args, **kwargs: 1.0):
        try:
            yield dict(policy="warm_constant_rate_v1",
                       g=control.base_rates[opt_g][0],
                       prior=control.base_rates[opt_g][1],
                       d=control.base_rates[opt_d][0],
                       shared_gate_eligible=False)
        finally:
            for opt, rates in saved.items():
                control.base_rates[opt][:] = rates


def _feed_hash(digest, value) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        digest.update(b"T")
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"D")
        for key in sorted(value, key=lambda item: str(item)):
            _feed_hash(digest, key)
            _feed_hash(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"L")
        digest.update(str(len(value)).encode())
        for item in value:
            _feed_hash(digest, item)
    else:
        digest.update(b"S")
        digest.update(repr(value).encode())


def training_state_sha256(state: dict) -> str:
    """Hash live weights, Adam/EMA state, and every training random stream."""
    policy = state["noise_policy"]
    values = dict(
        generator=state["generator"].state_dict(),
        critic=state["critic"].state_dict(),
        prior=state["prior"].state_dict(),
        opt_g=state["opt_g"].state_dict(),
        opt_d=state["opt_d"].state_dict(),
        ema_g=state["ema_g"], ema_z=state["ema_z"],
        data_stream=state["stream"].get_state(),
        torch_rng=torch.get_rng_state(),
        input_stream=policy.input_stream.get_state(),
        output_stream=(None if policy.output_stream is None else
                       policy.output_stream.get_state()),
        input_sigma=policy.input_sigma,
        output_sigma=policy.output_sigma,
    )
    digest = hashlib.sha256()
    _feed_hash(digest, values)
    return digest.hexdigest()


def _metrics_without_time(evidence: dict) -> list[dict]:
    return [{key: value for key, value in point.items() if key != "seconds"}
            for point in evidence["observations"]]


def run_warm_variants(
    config: dict,
    variant_contexts: dict[str, Callable[[dict, object], object]],
    *,
    output_dir: Path,
    prefix_context: Callable[[], object] = nullcontext,
    mode: str = "scheduled",
    warm_step: int = WARM_STEP,
    steps: int = FROZEN_STEPS,
) -> dict:
    """Fork one shared prefix, run local/extended variants, verify identity.

    An identity variant must be included. A separate cold full-budget control
    supplies numerical and full-state parity. All variants share the exact
    initial warm-state hash. The output directory must be new so stale files
    cannot be mistaken for child results.
    """
    if os.name != "posix" or not hasattr(os, "fork"):
        raise RuntimeError("warm-state fork requires Linux/POSIX CPU")
    if torch.cuda.is_initialized():
        raise RuntimeError("warm-state fork requires CPU-only torch state")
    if threading.active_count() != 1:
        raise RuntimeError("warm-state fork requires one Python thread")
    if type(warm_step) is not int or not 0 < warm_step < FROZEN_STEPS:
        raise ValueError("warm_step must be inside the frozen 1,200-update host")
    if (type(steps) is not int or steps < FROZEN_STEPS
            or steps % 10):
        raise ValueError("steps must be at least 1,200 and divisible by ten")
    if not variant_contexts or "identity" not in variant_contexts:
        raise ValueError("an identity variant is required for exact parity")
    if any(re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None
           for name in variant_contexts):
        raise ValueError("variant names must be simple path-safe labels")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    cold_state: dict = {}
    def cold_hook(state):
        cold_state["final_sha256"] = training_state_sha256(state)
    cadence = 50 if steps == FROZEN_STEPS else 10
    cold = run_probe(config, mode=mode, steps=steps,
                     diagnostic_every=cadence,
                     checkpoint_hook_step=steps,
                     checkpoint_hook=cold_hook)
    (output_dir / "cold.json").write_text(json.dumps(cold, allow_nan=False) + "\n")

    child_name: str | None = None
    child_context = None
    child_receipt = None
    warm_state: dict | None = None
    warm_hash: str | None = None
    pids: dict[str, int] = {}
    prefix_receipt = None

    def fork_at_checkpoint(state):
        nonlocal child_name, child_context, child_receipt, warm_state, warm_hash
        if state["completed_steps"] != warm_step or torch.get_num_threads() != 1:
            raise RuntimeError("warm checkpoint has unexpected step or torch threads")
        warm_hash = training_state_sha256(state)
        warm_state = state
        for name, factory in variant_contexts.items():
            pid = os.fork()
            if pid == 0:
                child_name = name
                try:
                    child_context = factory(state, prefix_receipt)
                    child_receipt = child_context.__enter__()
                except BaseException as error:
                    (output_dir / f"{name}.error.json").write_text(json.dumps(
                        dict(error=repr(error), traceback=traceback.format_exc())) + "\n")
                    traceback.print_exc()
                    os._exit(1)
                return
            pids[name] = pid
        raise _PrefixForked()

    with prefix_context() as receipt:
        prefix_receipt = receipt
        try:
            evidence = run_probe(
                config, mode=mode, steps=steps,
                diagnostic_every=cadence, dense_after=warm_step,
                dense_until=FROZEN_STEPS,
                checkpoint_hook_step=warm_step,
                checkpoint_hook=fork_at_checkpoint,
            )
        except _PrefixForked:
            errors = {}
            for name, pid in pids.items():
                _, status = os.waitpid(pid, 0)
                if not os.WIFEXITED(status) or os.WEXITSTATUS(status):
                    errors[name] = status
            if errors:
                raise RuntimeError(f"warm-state child failures: {errors}")
        except BaseException as error:
            if child_name is not None:
                (output_dir / f"{child_name}.error.json").write_text(json.dumps(
                    dict(error=repr(error), traceback=traceback.format_exc())) + "\n")
                traceback.print_exc()
                os._exit(1)
            raise
        else:
            # Only a child returns normally: the parent raises _PrefixForked.
            if child_name is None or warm_state is None or warm_hash is None:
                raise RuntimeError("warm fork returned without a child identity")
            try:
                final_hash = training_state_sha256(warm_state)
                if child_context is not None:
                    child_context.__exit__(None, None, None)
                local = _window([point for point in evidence["diagnostic"]
                                 if warm_step < point["step"] <= FROZEN_STEPS])
                if local["checks"] != FROZEN_STEPS - warm_step:
                    raise RuntimeError("a per-update continuation check is missing")
                long_hold = (_window([point for point in evidence["diagnostic"]
                                      if FROZEN_STEPS < point["step"] <= steps])
                             if steps > FROZEN_STEPS else None)
                if long_hold is not None and long_hold["checks"] != (
                        steps - FROZEN_STEPS) // cadence:
                    raise RuntimeError("a long-horizon diagnostic check is missing")
                local_pass = local["pass_all"] and (
                    long_hold is None or long_hold["pass_all"])
                evidence.update(
                    scope="passing_state_local_stability_only",
                    warm_step=warm_step, continuation_updates=steps - warm_step,
                    variant=child_name, warm_state_sha256=warm_hash,
                    final_state_sha256=final_hash,
                    local_stability=local,
                    long_hold=long_hold,
                    cold_host_status=evidence["status"],
                    status="PASS" if local_pass else "FAIL",
                )
                if isinstance(child_receipt, dict):
                    evidence["dynamics_receipt"] = child_receipt
                (output_dir / f"{child_name}.json").write_text(
                    json.dumps(evidence, allow_nan=False) + "\n")
            except BaseException:
                (output_dir / f"{child_name}.error.json").write_text(json.dumps(
                    dict(error="child finalization failed",
                         traceback=traceback.format_exc())) + "\n")
                traceback.print_exc()
                os._exit(1)
            os._exit(0)

    variants = {name: json.loads((output_dir / f"{name}.json").read_text())
                for name in variant_contexts}
    identity = variants["identity"]
    cold_rates = cold["rate_ranges"]
    if (_metrics_without_time(identity) != _metrics_without_time(cold)
            or identity["final"] != cold["final"]
            or identity["ema"] != cold["ema"]
            or identity["rate_ranges"] != cold_rates
            or identity["final_state_sha256"] != cold_state["final_sha256"]):
        raise RuntimeError("identity warm continuation differs from cold host")
    warm_hashes = {row["warm_state_sha256"] for row in variants.values()}
    if len(warm_hashes) != 1:
        raise RuntimeError("variants did not inherit the same warm state")
    for name, row in variants.items():
        receipt = row.get("dynamics_receipt", {})
        if receipt.get("policy") == "warm_constant_rate_v1":
            for role in ("g", "d", "prior"):
                observed = row["post_checkpoint_rate_ranges"][role]
                expected = receipt[role]
                if (observed["min"] != expected or observed["max"] != expected
                        or observed["observations"] != row["optimizer_final"][0]["calls"]
                        - warm_step):
                    raise RuntimeError(f"{name} {role} continuation rate differs")
    summary = dict(scope="passing_state_local_stability_only", warm_step=warm_step,
                   steps=steps, continuation_updates=steps - warm_step,
                   identity_cold_parity=True,
                   cold_final_state_sha256=cold_state["final_sha256"],
                   warm_state_sha256=warm_hashes.pop(),
                   variants={name: dict(status=row["status"],
                                        local_stability=row["local_stability"],
                                        long_hold=row["long_hold"],
                                        stationary=row["stationary"],
                                        final=row["final"],
                                        final_state_sha256=row["final_state_sha256"])
                             for name, row in variants.items()})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                         sort_keys=True) + "\n")
    return summary
