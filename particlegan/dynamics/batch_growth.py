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

The unequal critic's distance feature is a dense kernel, and its penalty
asks for a second derivative. Above 2,048 rows those graphs do not fit
together on this screen. ``install`` evaluates the kernel in row blocks and
recomputes one block at a time for both derivatives. Forward features match
the dense kernel bit for bit. The host batch and eval stay on the dense path.
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


# The unequal critic's distance feature is a dense [N, N, scales] kernel.
# The penalty differentiates through d(score)/dx, so a checkpoint of that
# kernel is kept alive for the second backward. Above 2,048 rows the real
# and fake copies no longer fit on this screen, and the Smith cap is 12,800.
# The kernel is then applied in row blocks. The first derivative and the
# penalty's second derivative each recompute one block and drop it. Forward
# features match the dense kernel bit for bit. Point gradients and penalty
# parameter gradients differ by the blocked reduction (about one ulp).
# Eval is under no_grad, and every batch at or below 2,048 stays dense.
_RECOMPUTE_ABOVE = 2048
_ROW_BLOCK = 512
_PAIRWISE = None


def _install_pairwise_recompute() -> None:
    global _PAIRWISE
    if _PAIRWISE is not None:
        return
    import torch
    from torch.autograd import Function

    from particlegan.discriminators import BatchDistanceDiscriminator

    original = BatchDistanceDiscriminator.pairwise_features
    _PAIRWISE = original

    def _block(points, start, scales, eps, rows):
        rows_x = points[start:start + rows]
        delta = rows_x[:, None, :] - points[None, :, :]
        distance = delta.square().sum(-1)
        scales2 = scales.square()
        kernels = torch.exp(-distance[..., None] / (2 * scales2))
        diag = torch.arange(rows, device=points.device)
        mask = torch.ones(rows, points.shape[0], 1, device=points.device, dtype=points.dtype)
        mask[diag, start + diag, :] = 0
        kernels = kernels * mask
        weighted = (kernels * distance[..., None]).sum(dim=1)
        return weighted / (kernels.sum(dim=1) + eps) / scales2

    def _forward_features(points, scales, eps):
        n = points.shape[0]
        parts = []
        for start in range(0, n, _ROW_BLOCK):
            rows = min(_ROW_BLOCK, n - start)
            parts.append(_block(points, start, scales, eps, rows))
        return torch.cat(parts, dim=0)

    def _vjp(points, scales, eps, grad_feat):
        acc = torch.zeros_like(points)
        upstream = grad_feat.detach()
        n = points.shape[0]
        with torch.enable_grad():
            for start in range(0, n, _ROW_BLOCK):
                rows = min(_ROW_BLOCK, n - start)
                leaf = points.detach().requires_grad_(True)
                feat = _block(leaf, start, scales, eps, rows)
                grad = torch.autograd.grad(feat, leaf, upstream[start:start + rows])[0]
                acc = acc + grad.detach()
        return acc

    def _jvp(points, scales, eps, direction):
        parts = []
        tangent = direction.detach()
        n = points.shape[0]
        with torch.enable_grad():
            for start in range(0, n, _ROW_BLOCK):
                rows = min(_ROW_BLOCK, n - start)

                def fn(z, start=start, rows=rows):
                    return _block(z, start, scales, eps, rows)

                _, feature_tangent = torch.func.jvp(fn, (points.detach(),), (tangent,))
                parts.append(feature_tangent.detach())
        return torch.cat(parts, dim=0)

    class _LinearVJP(Function):
        @staticmethod
        def forward(ctx, grad_feat, points, scales, eps):
            ctx.save_for_backward(points.detach(), scales.detach())
            ctx.eps = float(eps)
            return _vjp(points, scales, ctx.eps, grad_feat)

        @staticmethod
        def backward(ctx, grad_points):
            points, scales = ctx.saved_tensors
            return _jvp(points, scales, ctx.eps, grad_points), None, None, None

    class _BlockedFeatures(Function):
        @staticmethod
        def forward(ctx, points, scales, eps):
            ctx.save_for_backward(points.detach(), scales.detach())
            ctx.eps = float(eps)
            return _forward_features(points, scales, ctx.eps)

        @staticmethod
        def backward(ctx, grad_feat):
            points, scales = ctx.saved_tensors
            return _LinearVJP.apply(grad_feat, points, scales, ctx.eps), None, None

    def pairwise_features(self, x):
        if x.shape[0] <= _RECOMPUTE_ABOVE or not torch.is_grad_enabled():
            return original(self, x)
        return _BlockedFeatures.apply(x, self.scales, torch.as_tensor(self.eps, dtype=x.dtype, device=x.device))

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
