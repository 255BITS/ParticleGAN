"""Every registered deterministic init builds the same valid CPU tensors twice.

Installers patch process-global construction hooks and some refuse a second
install, so each initializer runs in its own child. One helper interpreter
imports the registry once and forks a child per initializer and pass; that
replaces the earlier 2 x 175 fresh-interpreter design. A small sample, one
name per family, is rebuilt in fresh interpreters to show the forked hashes
equal what an unrelated process builds.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

SCORED = (
    "hid_q",
    "had_tm_frob_bias",
    "halton",
    "qr_pb_pq",
    "qr_bz_pq",
    "batch_feature_zero",
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

# Shapes of the parameters init_registry.witness_sha256 hashes, in order:
# generator (2 Linear), critic (2 Linear), wide Linear, Conv2d, ConvTranspose2d
# and the 5 x 2 particle table.
WITNESS_SHAPES = [
    [8, 8], [8], [2, 8], [2], [8, 2], [8], [1, 8], [1],
    [11, 7], [11], [3, 3, 3, 3], [3], [4, 2, 4, 4], [2], [5, 2],
]
CRITIC_HEAD = 6  # Identity-family inits legitimately give its one row equal entries.
PARTICLES = 14

_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}

# Runs in one helper interpreter. Every child records the exact byte stream the
# production witness hashes, so the validity checks see the hashed tensors.
_FORKED = r"""
import contextlib, hashlib, io, json, multiprocessing, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import torch
torch.set_default_device("cpu")
torch.set_num_threads(1)
from benchmarks.init_research import init_registry as registry
default = registry.witness_sha256()  # default PyTorch init; also warms lazy imports

class Recorder:
    def __init__(self):
        self.digest, self.parts = hashlib.sha256(), []
    def update(self, data):
        self.digest.update(data)
        self.parts.append(data)
    def hexdigest(self):
        return self.digest.hexdigest()

def build(name):
    # Pool workers may be forked from a helper thread; torch's default-device
    # mode is thread-local and changes what some installers rewrite, so set it
    # in the child exactly as a fresh interpreter does.
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    recorder = Recorder()
    registry.hashlib = type("RecordingHashlib", (), {"sha256": staticmethod(lambda: recorder)})
    with contextlib.redirect_stdout(io.StringIO()):
        registry.install(name)
        digest = registry.witness_sha256()
    stats = []
    for index in range(0, len(recorder.parts), 3):
        shape = list(eval(recorder.parts[index].decode()))
        assert recorder.parts[index + 1] == b"torch.float32", recorder.parts[index + 1]
        value = np.frombuffer(recorder.parts[index + 2], dtype=np.float32)
        finite = bool(np.isfinite(value).all())
        stats.append({"shape": shape, "finite": finite,
                      "std": float(value.std()) if finite else None,
                      "max_abs": float(np.abs(value).max()) if finite else None})
    return name, digest, stats

if __name__ == "__main__":
    context = multiprocessing.get_context("fork")
    passes = []
    for _ in range(2):
        with context.Pool(8, maxtasksperchild=1) as pool:
            passes.append({name: [digest, stats] for name, digest, stats in
                           pool.map(build, registry.NAMES, chunksize=1)})
    families = {name: registry.family_of(name) for name in registry.NAMES}
    print(json.dumps({"default": default, "passes": passes, "families": families}))
"""

_FRESH = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
torch.set_default_device("cpu")
torch.set_num_threads(1)
from benchmarks.init_research.init_registry import install, witness_sha256
install(sys.argv[2])
print("\n" + witness_sha256())
"""


@pytest.fixture(scope="module")
def built():
    done = subprocess.run([sys.executable, "-c", _FORKED, str(ROOT)], capture_output=True,
                          text=True, env=_ENV, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_scored_names_are_registered():
    from benchmarks.init_research.init_registry import NAMES

    missing = [name for name in SCORED if name not in NAMES]
    assert not missing
    assert "qr_bz_pq" in NAMES
    assert len(NAMES) == len(set(NAMES))
    assert len(NAMES) > len(SCORED)


def test_import_does_not_change_the_default_init():
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from torch import nn
import benchmarks.init_research.init_registry  # noqa: F401
torch.manual_seed(0)
first = nn.Linear(4, 8).weight.detach().clone()
torch.manual_seed(1)
second = nn.Linear(4, 8).weight.detach().clone()
assert not torch.equal(first, second)
print("ok")
"""
    done = subprocess.run([sys.executable, "-c", script, str(ROOT)], check=True,
                          capture_output=True, text=True, env=_ENV)
    assert done.stdout.strip().endswith("ok")


def test_every_registered_init_is_deterministic(built):
    from benchmarks.init_research.init_registry import NAMES

    first, second = built["passes"]
    assert set(first) == set(second) == set(NAMES)
    mismatches = [name for name in NAMES if first[name][0] != second[name][0]]
    assert not mismatches


def test_every_registered_init_builds_valid_finite_scaled_tensors(built):
    """A constant, zero, nonfinite or exploding initializer fails here."""
    problems = []
    for name, (digest, stats) in built["passes"][0].items():
        if [entry["shape"] for entry in stats] != WITNESS_SHAPES:
            problems.append((name, "shapes", [entry["shape"] for entry in stats]))
            continue
        for index, entry in enumerate(stats):
            if not entry["finite"]:
                problems.append((name, index, "nonfinite"))
            elif entry["max_abs"] > 2.0:
                problems.append((name, index, "max_abs", entry["max_abs"]))
            elif len(entry["shape"]) >= 2 and index != CRITIC_HEAD and entry["std"] < 0.05:
                problems.append((name, index, "constant weight", entry["std"]))
        particles = stats[PARTICLES]
        if particles["finite"] and not (particles["std"] >= 0.1 and particles["max_abs"] <= 1.0):
            problems.append((name, "particles", particles))
        if digest == built["default"]:
            problems.append((name, "matches the default PyTorch init"))
    assert not problems, problems[:20]


def test_forked_hashes_match_fresh_interpreters_on_one_name_per_family(built):
    sample = {}
    for name, family in built["families"].items():
        sample.setdefault(family, name)
    assert len(sample) >= 8

    def fresh(name):
        return name, subprocess.run([sys.executable, "-c", _FRESH, str(ROOT), name], capture_output=True,
                                    text=True, env=_ENV, timeout=300)

    with ThreadPoolExecutor(4) as pool:
        for name, done in pool.map(fresh, sorted(sample.values())):
            assert done.returncode == 0, f"{name}\n{done.stderr[-2000:]}"
            assert done.stdout.strip().splitlines()[-1] == built["passes"][0][name][0], name
