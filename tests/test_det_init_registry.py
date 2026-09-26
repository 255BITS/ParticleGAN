"""Every registered deterministic init builds the same CPU tensors twice."""
import os
import subprocess
import sys

import pytest

SCORED = (
    "hid_q",
    "had_tm_frob_bias",
    "halton",
    "qr_pb_pq",
    "mix_pb_weyl",
    "mix_zb_pq",
    "cay_pb_pq",
    "rft_bz_pq",
    "qr_pb_pq_halton_s",
    "qr_pb_pq_lhs_s",
    "qr_pb_pq_weyl",
    "qr_pb_pq_r2_s",
    "qr_pb_pq_sobol_b",
    "qr_pb_pq_strat",
    "qr_pb_pq_r2_x",
    "qr_pb_pq_sobol_x",
    "qr_pb_pq_strat_s",
    "qr_pb_pq_strat_b",
    "qr_pb_pq_strat_x",
    "qr_pb_pq_halton_b",
    "qr_pb_pq_lhs_x",
    "qr_pb_pq_lhs_b",
    "hid_q_halton_s",
    "hid_q_lhs",
    "hid_q_r1_s",
    "hid_q_r2_x",
    "hid_q_weyl_b",
    "qr_pb_pq_weyl_s",
    "hid_q_halton_b",
    "hid_q_r2_s",
    "hid_q_sobol",
    "hid_q_sobol_b",
    "hid_q_lhs_b",
    "hid_q_strat_x",
)


def test_scored_names_are_registered_and_default_is_absent():
    from particlegan.init_registry import NAMES

    missing = [name for name in SCORED if name not in NAMES]
    assert not missing
    assert "qr_bz_pq" not in NAMES
    assert len(NAMES) == len(set(NAMES))
    assert len(NAMES) > len(SCORED)


def test_import_does_not_change_the_default_init():
    script = r"""
import torch
from torch import nn
import particlegan.init_registry  # noqa: F401
torch.manual_seed(0)
first = nn.Linear(4, 8).weight.detach().clone()
torch.manual_seed(1)
second = nn.Linear(4, 8).weight.detach().clone()
assert not torch.equal(first, second)
print("ok")
"""
    done = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    assert done.stdout.strip().endswith("ok")


def _hash_process(name: str) -> str:
    script = r"""
import sys
import torch
torch.set_default_device("cpu")
torch.set_num_threads(1)
from particlegan.init_registry import install, witness_sha256
install(sys.argv[1])
print(witness_sha256())
"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", script, name],
        check=False, capture_output=True, text=True, env=env,
    )
    if done.returncode != 0:
        raise AssertionError(f"{name} failed to build\n{done.stderr[-2000:]}\n{done.stdout[-500:]}")
    lines = [line.strip() for line in done.stdout.splitlines() if len(line.strip()) == 64]
    assert lines, name
    return lines[-1]


def test_every_registered_init_is_deterministic():
    from concurrent.futures import ThreadPoolExecutor

    from particlegan.init_registry import NAMES

    def both(name):
        return name, _hash_process(name), _hash_process(name)

    mismatches = []
    with ThreadPoolExecutor(8) as pool:
        for name, first, second in pool.map(both, NAMES):
            if first != second:
                mismatches.append(name)
    assert not mismatches
