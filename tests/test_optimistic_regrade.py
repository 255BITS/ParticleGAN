"""Scratch optimizer evidence must bind to every declared group and rate."""

import hashlib
import json

import pytest

from particlegan.recipes import get_recipe, learning_rate_scale
from reports.toy100.optimistic_regrade import _bind_optimizer_groups


def _fixture():
    recipe = get_recipe()
    steps = 4
    applied = [dict(role="d", parameters=5, lr=recipe.lr * recipe.d_lr_mult),
               dict(role="g", parameters=7, lr=recipe.lr),
               dict(role="prior", parameters=2, lr=recipe.lr * recipe.prior_lr_mult)]
    scales = [learning_rate_scale(i, steps, recipe.lr_anneal_start, recipe.lr_floor)
              for i in range(steps)]

    def optimizer(ordinal, rows, flags):
        rates = [[row["lr"] * scale for row in rows] for scale in scales]
        return dict(ordinal=ordinal, step_calls=steps,
                    group_parameter_counts=[row["parameters"] for row in rows],
                    group_prior_markers=flags,
                    group_parameter_updates=[steps] * len(rows),
                    group_lrs=rates,
                    lr_trace_sha256=hashlib.sha256(json.dumps(
                        rates, separators=(",", ":"),
                    ).encode()).hexdigest(),
                    previous_direction_state_count=len(rows),
                    optimistic_parameter_update_count=steps * len(rows),
                    **{"class": "Adam"})

    record = {"applied": applied, "spec": {"name": "trajectory", "steps": steps}}
    receipt = {"applied_roles": ["d", "g", "prior"], "optimizers": [
        optimizer(0, [applied[0]], [False]),
        optimizer(1, applied[1:], [False, True]),
    ]}
    protocol = {"global_recipe": recipe.to_dict()}
    return record, receipt, protocol


def test_scratch_group_regrade_binds_all_roles_and_rates():
    record, receipt, protocol = _fixture()
    _bind_optimizer_groups(record, receipt, protocol)


def test_scratch_group_regrade_rejects_midcourse_rate_even_with_rehashed_trace():
    record, receipt, protocol = _fixture()
    optimizer = receipt["optimizers"][1]
    optimizer["group_lrs"][2][0] *= 0.8
    optimizer["lr_trace_sha256"] = hashlib.sha256(json.dumps(
        optimizer["group_lrs"], separators=(",", ":"),
    ).encode()).hexdigest()
    with pytest.raises(ValueError, match="scheduled rates differ"):
        _bind_optimizer_groups(record, receipt, protocol)


def test_scratch_group_regrade_rejects_missing_prior_group():
    record, receipt, protocol = _fixture()
    optimizer = receipt["optimizers"][1]
    for key in ("group_parameter_counts", "group_prior_markers", "group_parameter_updates"):
        optimizer[key].pop()
    for row in optimizer["group_lrs"]:
        row.pop()
    optimizer["optimistic_parameter_update_count"] = record["spec"]["steps"]
    optimizer["lr_trace_sha256"] = hashlib.sha256(json.dumps(
        optimizer["group_lrs"], separators=(",", ":"),
    ).encode()).hexdigest()
    with pytest.raises(ValueError, match="groups differ"):
        _bind_optimizer_groups(record, receipt, protocol)
