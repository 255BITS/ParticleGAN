"""Bounded bandwidth waves preserve the frozen shared cores."""

import json

from reports.toy100.bandwidth_wave import (
    BASE_CONFIG, F5_BASE_CONFIG, ORDER, _name, _rows,
)


def test_bandwidth_waves_have_unique_declared_rows_and_canonical_hard_hosts():
    assert len(_rows("isolated_v1")) == 56
    assert len(_rows("global_v2")) == 62
    assert len({_name(row) for row in _rows("global_v2")}) == 62
    assert ORDER[:4] == ("trajectory", "residual_student", "img_stripes2", "mode_hold")


def test_f5_bridge_changes_only_output_noise_from_archived_public_core():
    base = json.loads(F5_BASE_CONFIG.read_text())
    assert base["input_noise_std"] == 0 and base["betas"] == [0.0, .99]
    assert base["reg_kappa"] == 1.25 and base["fourier"] == 5
    for row in _rows("global_v2"):
        if row["base_kind"] != "noiseless_f5":
            continue
        config = dict(base)
        config.update(name="bandwidth_" + _name(row),
                      output_noise_std=row["output_std"],
                      output_noise_warmup=row["warmup"])
        if row["learnable"]:
            config["output_noise_learnable"] = True
        changed = {key for key in config if config[key] != base.get(key)}
        assert changed <= {"name", "output_noise_std", "output_noise_warmup",
                           "output_noise_learnable"}
        assert "output_noise_rng" not in config


def test_kappa1176_wave_keeps_optimizer_and_architecture_fields():
    base = json.loads(BASE_CONFIG.read_text())
    for row in _rows("global_v2"):
        if row["base_kind"] != "kappa1176":
            continue
        assert base["reg_kappa"] == 1.176
        assert base["network_lr_horizon_cap"] == 1600
        assert base["toy100_model"] == "affine_square_v1"
        assert row["output_std"] in (0.005, .01, .029, .05, .1, .2)
