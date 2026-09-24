"""Particle consensus from critic disagreement, on the PR84 host.

After each ordinary PR84 G step, read the sharp critic on the particle
images and on a shell of probes around their own cloud. No real batch, mode
center, Chamfer term, or quota is consulted.

- Several particles in one critic basin and a high-D probe far from every
  particle: move the lowest-scoring particle in that basin toward the probe.
- No empty high-D probe: shrink particles that share a basin (near-idle when
  they already sit together). Distinct singleton basins: do nothing.
"""

import ast
from contextlib import contextmanager

import torch

from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100.extra_adam_scratch import HOSTS, sha
from reports.toy100.pr84_smoothed_candidate import (
    SMOOTH_WIDTH_CAP, SmoothedBothBoundRecorder,
)


METHOD = "pr84_particle_consensus_from_critic_disagreement"
LINK_DISTANCE = 0.55
VALLEY = 0.25
SHELL_ANGLES = 24
HOLE_SCORE_MARGIN = 0.15
HOLE_DIST_FRAC = 0.28
HOLE_DIST_MIN = 0.35
SEPARATE_FRACTION = 0.22
SEPARATE_CAP = 0.35
SHRINK = 0.05
SHRINK_CAP = 0.02


def _basins(points, scores, mid_scores):
    n = points.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dist = torch.cdist(points, points)
    for i in range(n):
        for j in range(i + 1, n):
            if float(dist[i, j]) > LINK_DISTANCE:
                continue
            floor = min(float(scores[i]), float(scores[j])) - VALLEY
            if float(mid_scores[i, j]) >= floor:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def consensus_delta(points, scores, mid_scores, probes, probe_scores):
    """Return an output-space particle move and a compact action record.

    ``points`` and ``probes`` are ``(N, 2)`` / ``(M, 2)``. Scores are critic
    values at those locations. ``mid_scores[i, j]`` is the critic at the
    midpoint of particles i and j.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("consensus expects 2D particle images")
    dx = torch.zeros_like(points)
    basins = _basins(points, scores, mid_scores)
    radius = float((points - points.mean(0)).norm(dim=1).median())
    min_dist = max(HOLE_DIST_MIN, HOLE_DIST_FRAC * max(radius, 1e-6))
    holes = []
    if probes.numel():
        nearest = torch.cdist(probes, points).min(dim=1).values
        for k in range(probes.shape[0]):
            if float(nearest[k]) < min_dist:
                continue
            near_score = float(scores[torch.cdist(probes[k:k + 1], points).argmin()])
            if float(probe_scores[k]) < near_score + HOLE_SCORE_MARGIN:
                continue
            holes.append(k)
    info = dict(basins=len(basins), crowded=sum(len(g) >= 2 for g in basins),
                holes=len(holes), radius=radius, min_hole_dist=min_dist)
    if holes:
        crowded = [group for group in basins if len(group) >= 2]
        if not crowded:
            info.update(action="idle", reason="holes_but_no_shared_basin")
            return dx, info
        group = max(crowded, key=len)
        donor = min(group, key=lambda i: float(scores[i]))
        target = max(holes, key=lambda k: float(probe_scores[k]))
        step = SEPARATE_FRACTION * (probes[target] - points[donor])
        norm = float(step.norm())
        if norm > SEPARATE_CAP:
            step = step * (SEPARATE_CAP / norm)
        dx[donor] = step
        info.update(action="separate", donor=int(donor), hole=int(target),
                    step_norm=float(step.norm()))
        return dx, info
    moved = False
    for group in basins:
        if len(group) < 2:
            continue
        weight = scores[group].detach().float()
        weight = weight - weight.min() + 1e-3
        center = (points[group] * weight[:, None]).sum(0) / weight.sum()
        for i in group:
            step = SHRINK * (center - points[i])
            norm = float(step.norm())
            if norm > SHRINK_CAP:
                step = step * (SHRINK_CAP / norm)
            if float(step.norm()) > 0:
                moved = True
            dx[i] = step
    info.update(action="shrink" if moved else "idle",
                step_norm=float(dx.norm()))
    return dx, info


def shell_probes(points):
    """Probes on the particles' own radial shell. Not data locations."""
    center = points.mean(0)
    radius = float((points - center).norm(dim=1).median().clamp_min(1e-3))
    angles = torch.linspace(0, 2 * torch.pi, SHELL_ANGLES + 1, device=points.device,
                            dtype=points.dtype)[:-1]
    direction = torch.stack((angles.cos(), angles.sin()), dim=1)
    return center + radius * direction


def _unwrap(module, kind):
    while not isinstance(module, kind) and hasattr(module, "model"):
        module = module.model
    return module if isinstance(module, kind) else None


def _apply_output_delta(generator, prior, dx):
    applied = 0.0
    for index in range(dx.shape[0]):
        step = dx[index]
        if float(step.norm()) < 1e-8:
            continue
        z = prior.z[index].detach().requires_grad_(True)
        image = generator(z.unsqueeze(0)).squeeze(0)
        grads = []
        for coord in range(image.shape[0]):
            grad, = torch.autograd.grad(image[coord], z, retain_graph=coord + 1 < image.shape[0])
            grads.append(grad)
        jacobian = torch.stack(grads, 0)
        gram = jacobian @ jacobian.T + 1e-4 * torch.eye(jacobian.shape[0], dtype=jacobian.dtype)
        if not torch.isfinite(gram).all():
            continue
        coeff = torch.linalg.solve(gram, step.detach())
        dz = jacobian.T @ coeff
        if not torch.isfinite(dz).all():
            continue
        with torch.no_grad():
            prior.z[index].add_(dz)
        applied += float(dz.norm())
    return applied


class ConsensusRecorder(SmoothedBothBoundRecorder):
    def __init__(self, *, start_step=0, correction=True):
        super().__init__(start_step=start_step)
        self.correction = bool(correction)
        self.consensus_records = []

    def phases(self, step, opt_d, opt_g, local):
        yield from super().phases(step, opt_d, opt_g, local)
        if self.correction and self.enabled and step >= self.start_step:
            self._consensus()

    def _consensus(self):
        local = self._local or {}
        generator = _unwrap(local.get("generator"), SimpleMLPGenerator)
        critic = _unwrap(local.get("critic"), SimpleMLPDiscriminator)
        prior = local.get("prior")
        if generator is None or critic is None or prior is None or not hasattr(prior, "z"):
            return
        smooth = self._smooth_on
        self._smooth_on = False
        try:
            with torch.no_grad():
                points = generator(prior.z).detach()
                if points.ndim != 2 or points.shape[-1] != 2:
                    return
                scores = critic(points).detach()
                n = points.shape[0]
                pair_i, pair_j = torch.triu_indices(n, n, offset=1)
                mids = 0.5 * (points[pair_i] + points[pair_j])
                mid_values = critic(mids).detach() if mids.shape[0] else scores[:0]
                mid_scores = points.new_zeros((n, n))
                if mids.shape[0]:
                    mid_scores[pair_i, pair_j] = mid_values
                    mid_scores[pair_j, pair_i] = mid_values
                probes = shell_probes(points)
                probe_scores = critic(probes).detach()
            dx, info = consensus_delta(points, scores, mid_scores, probes, probe_scores)
            if info["action"] == "idle":
                info.update(latent_norm=0.0, outer_step=self.outer_steps)
            else:
                info["latent_norm"] = _apply_output_delta(generator, prior, dx)
                info["outer_step"] = self.outer_steps
            self.consensus_records.append(info)
            if self.records:
                self.records[-1]["consensus"] = info
            if self.outer_steps % 50 == 0 or info["action"] == "separate":
                print({"event": "CONSENSUS", **{k: info[k] for k in
                      ("outer_step", "action", "basins", "holes", "step_norm", "latent_norm")
                      if k in info}}, flush=True)
        finally:
            self._smooth_on = smooth

    def receipt(self):
        value = super().receipt()
        actions = {}
        for row in self.consensus_records:
            actions[row["action"]] = actions.get(row["action"], 0) + 1
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     shared_gate_eligible=False, particle_consensus=True,
                     consensus_actions=actions,
                     consensus_signal="sharp critic on particle images and their own radial shell",
                     data_assignment=False, coverage_objective=False,
                     smooth_width_cap=SMOOTH_WIDTH_CAP)
        return value


@contextmanager
def gan6_particle_consensus(*, task="mode_hold", start_step=0, correction=True):
    """PR84 smoothed candidate plus the critic-only particle move."""
    import math
    from contextlib import ExitStack
    from unittest.mock import patch

    from particlegan.gan_loss import GANLoss
    from benchmarks.locked_shared import mode_hold, trajectory
    from reports.toy100.extra_adam_scratch import transformed_function

    module = {"mode_hold": mode_hold, "trajectory": trajectory}[task]

    tree, _, original_sha = transformed_function(module, task)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "phases"]
    if len(calls) != 1:
        raise RuntimeError("expected one phase iterator")
    calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    recorder = ConsensusRecorder(start_step=start_step, correction=correction)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss
    original_forward = SimpleMLPDiscriminator.forward

    def observed_d_loss(gan, real_logits, fake_logits):
        value = original_d_loss(gan, real_logits, fake_logits)
        if recorder.phase == 0 and recorder.advantage is None:
            recorder.advantage = math.log(2) - float(value.detach())
        return value

    def smoothed_forward(self, x):
        if (recorder._smooth_on and recorder.enabled and not recorder.passthrough
                and x.ndim >= 2 and x.shape[-1] == 2 and recorder._smooth_width > 0):
            width = recorder._smooth_width
            vals = [original_forward(self, x)]
            for dim in range(x.shape[-1]):
                shift = torch.zeros_like(x)
                shift[..., dim] = width
                vals.append(original_forward(self, x + shift))
                vals.append(original_forward(self, x - shift))
            return torch.stack(vals, 0).mean(0)
        return original_forward(self, x)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<gan6-consensus-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(
            torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(SimpleMLPDiscriminator, "forward", smoothed_forward))
        yield recorder, source
