"""Grow the paired minibatch by 1/f instead of decaying the learning rate.

Diagnostic only. This is not a leaderboard candidate and must not be read
as a recipe that acquires modes and stays.

K3P's cosine multiplies the learning rate by f(t). The multiplier is 1
through update 720 and first drops at update 721 (Smith et al. 2018,
"Don't Decay the Learning Rate, Increase the Batch Size"). Their noise
scale is proportional to learning_rate / batch, so a multiplier f is
matched by a batch multiplier 1/f.

The ring, hold, and stay host draws one relativistic pair. The real row,
the fake row, and the particle row are that same minibatch: RpGAN subtracts
logits of equal length, and each fake is one draw from the particle table.
The three sizes are therefore one integer.

The published cosine, the one the controller applies on the 1,200-step
horizon (not the constant-LR screen, which forces every multiplier to 1):

* anneal start 0.6, horizon 1,200 (the cap is 1,600, and min(1200, 1600)
  is 1,200), so the drop starts at update 721;
* network floor 0.01, prior floor 0.05;
* after update 1,200 both stay on those floors, which is what hold and
  stay already do.

f_network is at most f_prior. The shared batch grows by 1/f_network, the
stricter factor, and is capped at base / 0.01. That cap is the Smith match
at the network floor. It fits: the host batch on this screen is 128, so
the cap is 12,800 rows of a 2-d toy. The particle table is smaller than
the host batch (12 on the ring, 256 on unequal) and the sampler already
draws with replacement, so the table size is not a second cap. Using it
would shrink the batch while f is still 1 and change the game before the
anneal starts.

Network noise then matches the network anneal. Prior noise falls at least
as far as the prior anneal (100× versus 20× at the floor). With the flag
unset this function returns the host batch and does not touch the RNG.

The unequal critic's distance feature is a dense kernel. At the cap, keeping
that kernel for both the real and the fake backward does not fit. ``install``
recomputes it on the way back for batches above 4,096. The values and the
gradients match the dense kernel bit for bit; the host batch and eval stay
on the dense path.
"""
from __future__ import annotations

import atexit
import json
import math
import os

from particlegan.recipes import learning_rate_scale

# Published K3P cosine. Not read from the constant-LR config: that config
# sets the executed learning rate to 1, which is the point of the screen.
HORIZON = 1200
ANNEAL_START = 0.6
NETWORK_FLOOR = 0.01
PRIOR_FLOOR = 0.05
# 1 / NETWORK_FLOOR. The cap is `base * FLOOR_MULTIPLE`.
FLOOR_MULTIPLE = 100

receipt = {
    "mechanism": "batch_growth",
    "updates": 0,
    "base": None,
    "max_batch": None,
    "cap": None,
    "first_growth_step": None,
    "floor_step": None,
    "milestones": [],
}
_LOGGED = False
_INSTALLED = False
_MILESTONES = (0, 720, 721, 900, 1199, 1200, 2400, 3600)


# The unequal critic's distance features are a dense [N, N, scales] kernel.
# Real and fake (and the penalty interpolation) each keep that kernel for
# backward. At the Smith cap, N = 12,800, two of those graphs are larger
# than this screen. Recomputing the kernel during backward is the same
# function: features and gradients match the dense path bit for bit, and
# batches at or below the eval size (4,096) stay on that path.
_RECOMPUTE_ABOVE = 4096
_PAIRWISE = None


def _install_pairwise_recompute() -> None:
    global _PAIRWISE
    if _PAIRWISE is not None:
        return
    import torch
    from torch.utils.checkpoint import checkpoint

    from particlegan.discriminators import BatchDistanceDiscriminator

    original = BatchDistanceDiscriminator.pairwise_features
    _PAIRWISE = original

    def pairwise_features(self, x):
        if x.shape[0] <= _RECOMPUTE_ABOVE or not torch.is_grad_enabled():
            return original(self, x)
        bound = original.__get__(self, type(self))
        return checkpoint(lambda batch: bound(batch), x, use_reentrant=False)

    BatchDistanceDiscriminator.pairwise_features = pairwise_features


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    _install_pairwise_recompute()
    atexit.register(_emit)
    print(json.dumps({
        "event": "dynamics",
        "name": "batch_growth",
        "setting": "paired real/fake/particle batch *= 1/f_network, cap = base/0.01, LR constant",
        "horizon": HORIZON,
        "anneal_start": ANNEAL_START,
        "network_floor": NETWORK_FLOOR,
        "prior_floor": PRIOR_FLOOR,
    }), flush=True)


def _emit() -> None:
    if receipt["updates"]:
        print(json.dumps({"event": "dynamics_receipt", **receipt}), flush=True)


def factors(step: int) -> tuple[float, float]:
    """Return (f_network, f_prior) for the update about to run.

    ``step`` is the host's completed-update index: 0 before the first
    update, 720 still at 1, 721 the first drop. Past the horizon both
    factors sit on their floors.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    network = learning_rate_scale(step, HORIZON, ANNEAL_START, NETWORK_FLOOR)
    prior = learning_rate_scale(step, HORIZON, ANNEAL_START, PRIOR_FLOOR)
    return network, prior


def _ceil_ratio(base: int, factor: float) -> int:
    """Smallest integer batch whose 1/batch is at most factor/base.

    Values that land on an integer within a rounding hair stay on that
    integer, so the floor batch is exactly ``base / network_floor``.
    """
    if factor >= 1.0:
        return base
    quot = base / factor
    nearest = round(quot)
    if abs(quot - nearest) <= 1e-6 * max(1.0, abs(quot)):
        return int(nearest)
    return math.ceil(quot)


def floor_cap(base: int) -> int:
    """Batch that matches the network floor. This is what fits on the toy."""
    if type(base) is not int or base <= 0:
        raise ValueError("base batch must be a positive integer")
    return base * FLOOR_MULTIPLE


def paired_batch(base: int, step: int) -> int:
    """Host batch, or that batch grown by 1/f_network and capped.

    Flag unset returns ``base`` with no RNG use and no receipt write.
    """
    if os.environ.get("K3P_DYNAMICS") != "batch_growth":
        return base
    if type(base) is not int or base <= 0:
        raise ValueError("base batch must be a positive integer")
    network, prior = factors(step)
    cap = floor_cap(base)
    size = min(cap, _ceil_ratio(base, network))
    if size < base:
        size = base
    _note(step, base, size, network, prior, cap)
    return size


def _note(step: int, base: int, size: int, network: float, prior: float, cap: int) -> None:
    global _LOGGED
    receipt["updates"] += 1
    receipt["base"] = base
    receipt["cap"] = cap
    receipt["max_batch"] = size if receipt["max_batch"] is None else max(receipt["max_batch"], size)
    if size > base and receipt["first_growth_step"] is None:
        receipt["first_growth_step"] = step
    if size == cap and receipt["floor_step"] is None:
        receipt["floor_step"] = step
    milestone = step in _MILESTONES or step == receipt["first_growth_step"] or step == receipt["floor_step"]
    if milestone:
        row = {
            "step": step,
            "batch": size,
            "f_network": network,
            "f_prior": prior,
            "cap": cap,
        }
        receipt["milestones"].append(row)
        print(json.dumps({"event": "batch_growth", **row}), flush=True)
    if not _LOGGED:
        _LOGGED = True
        print(json.dumps({
            "event": "dynamics_step",
            "name": "batch_growth",
            "step": step,
            "batch": size,
        }), flush=True)
