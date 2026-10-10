"""First-order protection of existing losses after a full DualNorm step.

The Euclidean projection changes the optimizer, never a loss coefficient.
It promises non-ascent derivatives, not finite-step descent or GAN convergence.
"""
from copy import deepcopy
import itertools

import torch

from .dualnorm import NormalizedOptimizer


def constraint_geometry_backward(loss, optimizer, protected_losses, *, protected_evaluator=None):
    """Backpropagate the unchanged scalar loss; bind existing protected losses.

    Ordinary optimizers use exactly the original backward. Enabled optimizers
    fail closed if their host omits this call. No samples or RNG draws are added.
    """
    binder = getattr(optimizer, "bind_protected_losses", None)
    if binder is not None:
        binder(protected_losses, protected_evaluator=protected_evaluator)
    loss.backward()


def project_nonascent(displacement, normals):
    """Nearest displacement satisfying each normal dot displacement <= 0.

    Enumerate the active sets of at most two constraints. Computation is double
    precision; zero gradients impose no constraint. Zero is always feasible.
    """
    d = displacement.double()
    a = normals.double()
    lengths = a.norm(dim=1)
    a = a[lengths > 0] / lengths[lengths > 0, None]
    if len(a) > 2:
        raise ValueError("constraint_geometry supports at most two protected losses")
    if not len(a) or bool((a @ d <= 0).all()):
        return displacement.clone()
    best, best_distance = torch.zeros_like(d), d.square().sum()
    tolerance = 1e-10 * max(1., float(d.norm()))
    for count in range(1, len(a) + 1):
        for indices in itertools.combinations(range(len(a)), count):
            active = a[list(indices)]
            gram = active @ active.T
            multiplier = torch.linalg.pinv(gram, hermitian=True) @ (active @ d)
            candidate = d - active.T @ multiplier
            if bool((multiplier >= -tolerance).all()) and bool((a @ candidate <= tolerance).all()):
                distance = (candidate - d).square().sum()
                if bool(distance < best_distance):
                    best, best_distance = candidate, distance
    return best.to(displacement)


class ConstraintGeometryOptimizer(NormalizedOptimizer):
    """Project actual combined G/prior displacement without re-normalizing it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.family != "dualnorm" or self.momentum:
            raise ValueError("constraint_geometry requires zero-momentum full DualNorm")
        self._protected = None
        self.constraint_geometry_stats = dict(steps=0, projected_steps=0,
                                             max_derivative_before=0., max_derivative_after=0.)

    def _parameters(self):
        return [p for group in self.param_groups for p in group["params"]]

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        if self._protected is not None:
            raise ValueError("constraint_geometry losses already pending")
        losses = tuple(losses)
        if not 1 <= len(losses) <= 2 or any(loss.ndim != 0 or not loss.requires_grad for loss in losses):
            raise ValueError("one or two active scalar protected losses required")
        parameters = self._parameters()
        normals = []
        for loss in losses:
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            pieces = []
            for parameter, gradient in zip(parameters, gradients):
                piece = torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
                group = next(g for g in self.param_groups if any(p is parameter for p in g["params"]))
                if group["algorithm"] == "rownorm" and group["sampled_rows_required"]:
                    rows = self.sampled_rows_for(parameter)
                    if rows is None:
                        raise ValueError("constraint_geometry requires explicit sampled rows")
                    masked = torch.zeros_like(piece)
                    masked[rows] = piece[rows]
                    piece = masked
                pieces.append(piece.flatten())
            normals.append(torch.cat(pieces))
        self._protected = torch.stack(normals)
        if not bool(torch.isfinite(self._protected).all()):
            self._protected = None
            raise ValueError("nonfinite constraint_geometry protected gradient")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or self._protected is None:
            raise ValueError("constraint_geometry step requires explicit protected-loss backward")
        parameters = self._parameters()
        originals = [p.detach().clone() for p in parameters]
        normals = self._protected
        result = super().step()
        displacement = torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])
        projected = project_nonascent(displacement, normals)
        before = normals.double() @ displacement.double()
        stats = self.constraint_geometry_stats
        stats["steps"] += 1
        changed = not torch.equal(projected, displacement)
        stats["projected_steps"] += int(changed)
        stats["max_derivative_before"] = max(stats["max_derivative_before"], float(before.max()))
        if changed:
            offset = 0
            for parameter, original in zip(parameters, originals):
                size = parameter.numel()
                parameter.copy_(original + projected[offset:offset + size].view_as(parameter))
                offset += size
        # Preserve the already-applied tensors bitwise when projection is inactive.
        # old + (new - old) can lose near-zero values in float32.
        applied = torch.cat([(p - old).flatten() for p, old in zip(parameters, originals)])
        after = normals.double() @ applied.double()
        stats["max_derivative_after"] = max(stats["max_derivative_after"], float(after.max()))
        self._protected = None
        return result

    def state_dict(self):
        result = super().state_dict()
        result["constraint_geometry"] = dict(schema=2, mode="nonascent", stats=deepcopy(self.constraint_geometry_stats),
                                               pending=None if self._protected is None else self._protected.clone())
        return result

    def validate_state_dict(self, saved):
        saved = dict(saved)
        meta = saved.pop("constraint_geometry", None)
        if (not isinstance(meta, dict) or set(meta) != {"schema", "mode", "stats", "pending"}
                or meta["schema"] != 2 or meta["mode"] != "nonascent"
                or set(meta["stats"]) != set(self.constraint_geometry_stats)):
            raise ValueError("invalid constraint_geometry checkpoint")
        for key, value in meta["stats"].items():
            if key.endswith("steps"):
                if type(value) is not int or value < 0:
                    raise ValueError("invalid constraint_geometry counter")
            elif type(value) not in (int, float) or not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError("invalid constraint_geometry derivative")
        pending = meta["pending"]
        if pending is not None and (not isinstance(pending, torch.Tensor) or pending.ndim != 2
                or not 1 <= len(pending) <= 2 or pending.shape[1] != sum(p.numel() for p in self._parameters())
                or not bool(torch.isfinite(pending).all())):
            raise ValueError("invalid pending constraint_geometry gradients")
        super().validate_state_dict(saved)

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        meta = saved["constraint_geometry"]
        self.constraint_geometry_stats = deepcopy(meta["stats"])
        self._protected = None if meta["pending"] is None else meta["pending"].to(self._parameters()[0])
