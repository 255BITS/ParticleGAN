"""Run original, hash-pinned conceptmod modules without diffusion dependencies."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch

COMMIT = "5571213f5e8e129cfda45c785c3f30aad9c1d8c9"
SOURCE_HASHES = {
    "leaderboard_honesty.py": "e0e6d04f405c2d3b101ef31b99d8bdc8f69cbb12dccd8bdd03eda80f3e4097f3",
    "shared_trajectory.py": "a1f76626d43c3ef96933bedac6809c8b11a44c1f02c30399488a3b3e49f5476c",
    "mode_hold.py": "c6b2fcb43267c1d6b435cde0f555ac3cb811f25c12213dc64a1a0597009654a2",
    "mlp.py": "928bd363b37b91ec557b82bed5c4b23cfa08eca499d6187ae3da071fcb3b2b1a",
}


def run_reference(root: Path):
    folder = root.resolve() / "conceptmod" / "toys"
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"reference source changed: {name}; checkout conceptmod {COMMIT}")
    # Avoid conceptmod.toys.__init__, which eagerly loads unrelated DSL/backend
    # modules. The four selected files themselves are executed unchanged.
    packages = {}
    for name, path in (("conceptmod", folder.parent), ("conceptmod.toys", folder)):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        packages[name] = package
    modules = [f"conceptmod.toys.{Path(name).stem}" for name in SOURCE_HASHES]
    with patch.dict(sys.modules, packages):
        for name in modules:
            sys.modules.pop(name, None)
        two = importlib.import_module("conceptmod.toys.leaderboard_honesty")
        traj = importlib.import_module("conceptmod.toys.shared_trajectory")
        ring = importlib.import_module("conceptmod.toys.mode_hold")
        torch.set_num_threads(1)
        rows = []
        for arm, name in zip(two.demo_arms(), ("locked_shared", "stranger", "thinned_cap")):
            print(f"reference toy=two_pole arm={name}", flush=True)
            result = two.run_cell(arm)
            rows.append({"toy": "two_pole", "arm": name,
                         **{k: getattr(result, k) for k in ("mean_abs", "grad_med", "nearest", "cover_score")},
                         "verdict": "PASS" if result.won else "FAIL"})
        for pairing, name in (("shared", "locked_shared"), ("stranger", "stranger"), ("nearest_stranger", "nearest_stranger")):
            print(f"reference toy=trajectory arm={name}", flush=True)
            result = traj.train(pairing=pairing, locked=pairing == "shared")
            rows.append({"toy": "trajectory", "arm": name,
                         **{k: result[k] for k in ("identity_mse", "paired_target_mse")},
                         "verdict": "PASS" if result["pass"] else "FAIL"})
        for name, overrides in (("locked_shared", {}), ("cap_off", {"reg_arm": "f_none", "reg_coeff": 0.0}),
                                ("vanilla", {"gan_mode": "vanilla"}), ("fm_on", {"fm_weight": 0.1})):
            print(f"reference toy=ring arm={name}", flush=True)
            result = ring.train_mode_hold(ring.locked_recipe().replace(**overrides), drift=overrides, log=False)
            rows.append({"toy": "ring", "arm": name,
                         **{k: result[k] for k in ("modes", "n_modes", "hq", "cover", "effective_modes", "step", "seed", "verdict")}})
    return rows
