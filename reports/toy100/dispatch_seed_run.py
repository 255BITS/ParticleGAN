"""Run one Codex audit command with an explicit host seed.

Seed 0 is the published runner: mode-hold's seed argument and trajectory's
PROTOCOL seed are 0, and NoisePolicy is constructed with seed 0. AUDIT_SEED
overrides those three integers and nothing else. Training math is unchanged.
"""

import os
import runpy
import sys
from pathlib import Path

SEED = int(os.environ.get("AUDIT_SEED", "0"))
REPO = Path(os.environ["AUDIT_REPO"]).resolve()
sys.path.insert(0, str(REPO))

import benchmarks.locked_shared.baseline as baseline
import benchmarks.locked_shared.mode_hold as mode_hold
import benchmarks.locked_shared.trajectory as trajectory
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

_run_toy = baseline.run_toy
_noise_init = NoisePolicy.__init__


def _noise_init_seeded(self, *args, **kwargs):
    kwargs["seed"] = SEED
    return _noise_init(self, *args, **kwargs)


def _run_toy_seeded(toy, cfg, *, noise_policy=None):
    hold = mode_hold.train_mode_hold
    train = trajectory.train

    def train_mode_hold(*args, **kwargs):
        kwargs["seed"] = SEED
        return hold(*args, **kwargs)

    def train_trajectory(*args, **kwargs):
        protocol = trajectory.PROTOCOL
        previous = protocol["seed"]
        protocol["seed"] = SEED
        try:
            return train(*args, **kwargs)
        finally:
            protocol["seed"] = previous

    mode_hold.train_mode_hold = train_mode_hold
    trajectory.train = train_trajectory
    try:
        return _run_toy(toy, cfg, noise_policy=noise_policy)
    finally:
        mode_hold.train_mode_hold = hold
        trajectory.train = train


NoisePolicy.__init__ = _noise_init_seeded
baseline.run_toy = _run_toy_seeded

def main():
    probe = REPO / "reports/toy100/gan_followup_probe.py"
    sys.argv = [str(probe), *sys.argv[1:]]
    runpy.run_path(str(probe), run_name="__main__")


if __name__ == "__main__":
    main()
