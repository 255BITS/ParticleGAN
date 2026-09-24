"""Independent, scratch-only regrade for Optimistic Adam transfer episodes.

The production common gate rejects these episodes. This checker first applies
the frozen host/source regrade, then binds the optimizer implementation and
every actually observed Adam group rate to the declared recipe. It never makes
an Optimistic Adam result eligible for the production common gate.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy_suite import _episode_rows
from benchmarks.toy100.schedule import policy_multipliers
from particlegan.recipes import Recipe, learning_rate_scale
from reports.toy100.optimistic_adam_scratch import ALGORITHM, PAPER


POLICY = "optimistic_adam_algorithm1_damped_v1"
SOURCE_FILE = "optimistic_adam_source.py"
DRIVER_FILE = "optimistic_transfer_driver.py"
REGRADER_FILE = "optimistic_regrade_source.py"
RECEIPT_FILE = "optimizer_receipt.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def _must(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _bind_optimizer_groups(record: dict, receipt: dict, protocol: dict) -> None:
    """Match every captured group to one declared role, size, and full LR curve."""
    applied = record["applied"]
    _must(applied and all(row["role"] in ("g", "d", "prior") for row in applied),
          "missing or unsupported applied optimizer roles")
    _must({"g", "d"} <= {row["role"] for row in applied}
          or record["spec"]["name"] == "two_pole", "G/D optimizer roles are incomplete")
    _must(receipt["applied_roles"] == sorted({row["role"] for row in applied}),
          "optimizer receipt roles differ from applied roles")

    recipe = Recipe(**protocol["global_recipe"])
    policy = protocol.get("model_policy", {})
    cap = policy.get("network_lr_horizon_cap")
    network_floor = policy.get("network_lr_floor")
    steps = record["spec"]["steps"]
    multipliers = []
    for completed in range(steps):
        if cap is None:
            scale = learning_rate_scale(completed, steps, recipe.lr_anneal_start,
                                        recipe.lr_floor)
            multipliers.append((scale, scale))
        else:
            multipliers.append(policy_multipliers(
                completed, steps, recipe.lr_anneal_start, recipe.lr_floor, cap,
                network_lr_floor=network_floor,
            ))

    groups = []
    for optimizer in receipt["optimizers"]:
        _must(optimizer["class"] == "Adam", "a non-Adam optimizer escaped the scratch adapter")
        _must(optimizer["step_calls"] == steps, "optimizer did not update at every host step")
        counts = optimizer["group_parameter_counts"]
        flags = optimizer["group_prior_markers"]
        updates = optimizer["group_parameter_updates"]
        rates = optimizer["group_lrs"]
        _must(len(counts) == len(flags) == len(updates) > 0
              and len(rates) == steps and all(len(row) == len(counts) for row in rates),
              "optimizer group receipt is incomplete")
        _must(all(isinstance(value, int) and value > 0 for value in counts + updates),
              "optimizer group has no parameters or updates")
        _must(optimizer["lr_trace_sha256"] == _sha(json.dumps(
            rates, separators=(",", ":"),
        ).encode()), "optimizer LR trace hash differs")
        _must(optimizer["previous_direction_state_count"] > 0
              and optimizer["optimistic_parameter_update_count"] == sum(updates),
              "optimistic direction state or update count is incomplete")
        for index, count in enumerate(counts):
            groups.append((count, flags[index], [row[index] for row in rates]))

    _must(len(groups) == len(applied), "actual optimizer groups differ from declared roles")
    # A few hosts have equal-sized groups. Search for a one-to-one assignment
    # instead of inferring a role from group order or parameter shape alone.
    def match(position: int, unused: tuple[int, ...]) -> bool:
        if position == len(groups):
            return not unused
        count, prior_flag, rates = groups[position]
        for index in unused:
            role = applied[index]["role"]
            if (count != applied[index]["parameters"]
                    or prior_flag is not None and bool(prior_flag) != (role == "prior")):
                continue
            base_rate = applied[index]["lr"]
            if all(_same(rate, base_rate * (prior if role == "prior" else network))
                   for rate, (network, prior) in zip(rates, multipliers)):
                if match(position + 1, tuple(i for i in unused if i != index)):
                    return True
        return False

    _must(match(0, tuple(range(len(applied)))),
          "actual optimizer group sizes or scheduled rates differ from declared roles")


def regrade_episode(directory: Path, *, task: str, alpha: float,
                    config_sha256: str, optimizer_source_sha256: str,
                    driver_source_sha256: str, regrader_source_sha256: str,
                    manifest_sha256: str,
                    source_commit: str | None = None) -> dict:
    directory = Path(directory)
    _must(0 < alpha <= 1 and math.isfinite(alpha), "invalid declared optimism alpha")
    protocol = _read(directory / "protocol.json")
    summary = _read(directory / "summary.json")
    index = _read(directory / "index.json")
    _must(len(index["records"]) == 1 and index["records"][0]["name"] == task,
          "scratch index task differs")
    indexed = index["records"][0]
    raw = gzip.decompress((directory / indexed["artifact"]).read_bytes())
    record = json.loads(raw)
    receipt_bytes = (directory / RECEIPT_FILE).read_bytes()
    receipt = json.loads(receipt_bytes)
    optimizer_bytes = (directory / SOURCE_FILE).read_bytes()
    driver_bytes = (directory / DRIVER_FILE).read_bytes()
    regrader_bytes = (directory / REGRADER_FILE).read_bytes()
    _must((_sha(optimizer_bytes), _sha(driver_bytes), _sha(regrader_bytes)) ==
          (optimizer_source_sha256, driver_source_sha256, regrader_source_sha256),
          "archived scratch source differs from predeclared source hashes")
    binding = protocol["scratch_optimizer_policy"]
    _must(all(saved == binding for saved in (
        summary["scratch_optimizer_policy"], summary["cases"][0]["scratch_optimizer_policy"],
        indexed["scratch_optimizer_policy"], record["scratch_optimizer_policy"],
    )), "scratch optimizer bindings differ across evidence layers")
    _must(all(saved.get("shared_gate_eligible") is False
              for saved in (protocol, summary, indexed, record)),
          "scratch episode is not marked common-gate ineligible")
    _must(binding == dict(
        shared_gate_eligible=False, scratch_optimizer=POLICY, alpha=float(alpha),
        optimizer_source_sha256=_sha(optimizer_bytes),
        driver_source_sha256=_sha(driver_bytes),
        regrader_source_sha256=_sha(regrader_bytes),
        optimizer_receipt_file=RECEIPT_FILE,
        optimizer_receipt_sha256=_sha(receipt_bytes),
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
    ), "scratch binding or source/receipt hash differs")
    _must(receipt["algorithm"] == ALGORITHM and receipt["paper"] == PAPER
          and receipt["alpha"] == alpha and receipt["previous_direction_unscaled"] is True
          and receipt["both_directions_use_current_scheduled_lr"] is True
          and receipt["additional_gradient_evaluations"] == 0
          and receipt["scratch_common_gate_eligible"] is False,
          "optimizer algorithm receipt differs")
    _must(receipt["task"] == task and receipt["config_sha256"] == config_sha256
          and receipt["source_sha256"] == _sha(optimizer_bytes)
          and receipt["driver_sha256"] == _sha(driver_bytes)
          and receipt["manifest_sha256"] == manifest_sha256
          and protocol["config_sha256"] == config_sha256,
          "optimizer receipt or declared config hash differs")
    if source_commit is not None:
        _must(receipt["source_commit"] == source_commit,
              "optimizer source commit differs from predeclared epoch")
    _must(receipt["optimizer_count"] == len(receipt["optimizers"]) >= 2
          and receipt["optimizer_step_calls"] == sum(
              row["step_calls"] for row in receipt["optimizers"])
          and receipt["parameter_updates"] == sum(
              sum(row["group_parameter_updates"]) for row in receipt["optimizers"])
          and receipt["parameter_updates"] == sum(
              row["optimistic_parameter_update_count"] for row in receipt["optimizers"]),
          "optimizer application counts differ")
    _bind_optimizer_groups(record, receipt, protocol)

    # This checks the complete archived executable suite, frozen task card,
    # 24-point curve, optimizer declarations, common recipe, and saved verdict.
    frozen = _episode_rows(directory, (task,), candidate=True, allow_scratch=True)
    _must(frozen["status"] in ("PASS", "FAIL") and task in frozen["cases"],
          f"frozen host/source regrade failed: {frozen.get('reason')}")
    _must(_episode_rows(directory, (task,), candidate=True)["status"] == "INVALID",
          "production common gate accepted scratch optimizer evidence")
    result = _read(directory / "scratch_result.json")
    _must(result == dict(task=task, alpha=float(alpha),
                         status=frozen["cases"][task]["status"],
                         config_sha256=config_sha256,
                         optimizer_receipt_sha256=_sha(receipt_bytes),
                         optimizer_step_calls=receipt["optimizer_step_calls"],
                         parameter_updates=receipt["parameter_updates"],
                         shared_gate_eligible=False),
          "scratch result stamp differs from independent regrade")
    return dict(task=task, alpha=float(alpha), status=result["status"],
                optimizer_count=receipt["optimizer_count"],
                optimizer_step_calls=receipt["optimizer_step_calls"],
                parameter_updates=receipt["parameter_updates"],
                source_archive_verified=True, common_gate_eligible=False,
                frozen_host_regrade=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--optimizer-source-sha256", required=True)
    parser.add_argument("--driver-source-sha256", required=True)
    parser.add_argument("--regrader-source-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    print(json.dumps(regrade_episode(args.directory, task=args.task,
                                      alpha=args.alpha,
                                      config_sha256=args.config_sha256,
                                      optimizer_source_sha256=args.optimizer_source_sha256,
                                      driver_source_sha256=args.driver_source_sha256,
                                      regrader_source_sha256=args.regrader_source_sha256,
                                      manifest_sha256=args.manifest_sha256,
                                      source_commit=args.source_commit)))
