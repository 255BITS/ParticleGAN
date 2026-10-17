"""Bounded empirical mean-copy phase with fixed score and exact paired packets.

Witness geometry/directions precede count-driven actions. Action means/counts
are refreshed after them. All ranking and sequential objectives are CPU
float64, using bounded bulk chart transfers rather than per-candidate CUDA
scalar reads. This is empirical negative evidence, not population/equality or
emitted-quality certification. Charts and packets are ephemeral.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import math
from dataclasses import dataclass, replace

import torch

Q = .05
POLICY = "even_group_radial_clip_unit_residual_EB_3K_plus_3_v1"


def group_means(values, groups, count):
    counts = torch.bincount(groups, minlength=count)
    sums = torch.zeros(count, values.shape[1], dtype=torch.float64, device=values.device)
    sums.index_add_(0, groups, values.double())
    return sums / counts.clamp_min(1)[:, None], counts


@dataclass(frozen=True)
class FixedMoment:
    centers: torch.Tensor
    scales: torch.Tensor
    even_means: torch.Tensor
    ema_means: torch.Tensor
    ema_counts: torch.Tensor
    weights: torch.Tensor
    directions: torch.Tensor
    radius: float
    rank: int
    cells: int

    def psi(self, metric, groups):
        z = (metric.double() - self.centers[groups]) / self.scales[groups, None]
        fraction = (self.radius / z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)
        return z * fraction[:, None]

    def energy(self, means=None):
        means = self.ema_means if means is None else means
        return (self.weights * (self.even_means - means).square().sum(1)).sum()


@torch.no_grad()
def freeze_moment(snapshot, even_features, ema_features):
    """This function has no odd witness input; no group can be score-selected."""
    if (not snapshot.valid_metric or snapshot.rank <= 0
            or snapshot.duplicate_fraction > Q):
        return None, "invalid_or_duplicate_chart"
    if (not bool(torch.isfinite(even_features).all())
            or not bool(torch.isfinite(ema_features).all())):
        return None, "nonfinite_fit_or_EMA"
    topology = snapshot._mass_topology()
    metric_device = snapshot.transform(even_features)
    even_groups = topology[snapshot._assign_metric(metric_device)[0]].cpu()
    metric = metric_device.cpu()
    count = snapshot.mass_groups
    centers, even_counts = group_means(metric, even_groups, count)
    if bool((even_counts < 2).any()):
        return None, "missing_or_insufficient_even_group"
    squared = (metric - centers[even_groups]).square().sum(1)
    ss = torch.zeros(count, dtype=torch.float64, device=metric.device)
    ss.index_add_(0, even_groups, squared)
    scales = (ss / even_counts / snapshot.rank).sqrt()
    if not bool(torch.isfinite(scales).all() & (scales > 0).all()):
        return None, "nonfinite_or_zero_even_scale"
    radius = math.sqrt(snapshot.rank / Q)
    # The temporary object uses only even fitted reference geometry.
    z = (metric - centers[even_groups]) / scales[even_groups, None]
    psi_even = z * (radius / z.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None]
    even_means, _ = group_means(psi_even, even_groups, count)
    ema_metric_device = snapshot.transform(ema_features)
    ema_groups = topology[snapshot._assign_metric(ema_metric_device)[0]].cpu()
    ema_metric = ema_metric_device.cpu()
    ze = (ema_metric - centers[ema_groups]) / scales[ema_groups, None]
    psi_ema = ze * (radius / ze.norm(dim=1).clamp_min(1e-30)).clamp_max(1.)[:, None]
    ema_means, ema_counts = group_means(psi_ema, ema_groups, count)
    if bool((ema_counts == 0).any()):
        return None, "missing_EMA_group"
    residual = even_means - ema_means
    lengths = residual.norm(dim=1)
    directions = torch.where(lengths[:, None] > 0,
                             residual / lengths.clamp_min(1e-30)[:, None],
                             torch.zeros_like(residual))
    weights = even_counts.double() / int(even_counts.sum())
    if not bool(torch.isfinite(directions).all() & torch.isfinite(ema_means).all()):
        return None, "nonfinite_frozen_direction"
    return FixedMoment(centers, scales, even_means, ema_means, ema_counts,
                       weights, directions, radius, snapshot.rank, snapshot.cells), None


@torch.no_grad()
def odd_witness(snapshot, fixed, odd_features):
    """All odd observations, including zero directions, use the known 4R range."""
    alpha = Q / (3 * snapshot.cells + 3)
    base = dict(policy=POLICY, alpha=alpha, multiplicity=3 * snapshot.cells + 3,
                observations=len(odd_features), authoritative=False,
                limit="conditional iid bounded-score algebra; trained D/shared FIFO empirical evidence only")
    if fixed is None:
        return dict(base, valid=False, fires=False, reason="no_valid_frozen_moment")
    if len(odd_features) <= 1 or not bool(torch.isfinite(odd_features).all()):
        return dict(base, valid=False, fires=False, reason="insufficient_or_nonfinite_odd_rows")
    metric_device = snapshot.transform(odd_features)
    groups = snapshot._mass_topology()[snapshot._assign_metric(metric_device)[0]].cpu()
    metric = metric_device.cpu()
    if (bool((fixed.scales[groups] <= 0).any())
            or bool((fixed.ema_counts[groups] == 0).any())):
        return dict(base, valid=False, fires=False, reason="missing_odd_group_definition")
    psi = fixed.psi(metric, groups)
    values = (fixed.directions[groups] * (psi - fixed.ema_means[groups])).sum(1)
    if not bool(torch.isfinite(values).all()):
        return dict(base, valid=False, fires=False, reason="nonfinite_odd_scalar")
    known_range = 4 * fixed.radius
    # Multiplying the theorem for [0,1] by the known range gives this bound.
    variance = values.var(unbiased=True)
    mean = values.mean()
    t = math.log(2 / alpha)
    variance_penalty = (2 * variance * t / len(values)).sqrt()
    range_penalty = (7 / 3) * known_range * t / (len(values) - 1)
    lcb = mean - variance_penalty - range_penalty
    assert bool((values.abs() <= 2 * fixed.radius + 1e-10).all())
    return dict(base, valid=True, fires=bool(lcb > 0), reason=None,
                mean=float(mean), variance_ddof1=float(variance), radius=fixed.radius,
                known_range=known_range, variance_penalty=float(variance_penalty),
                range_penalty=range_penalty, lower_bound=float(lcb),
                zero_direction_observations=int((fixed.directions[groups].norm(dim=1) == 0).sum()),
                scalar_min=float(values.min()), scalar_max=float(values.max()),
                even_mass_objective=float(fixed.energy()))


POOL = 64


@contextmanager
def neutral_observation(roots, streams=(), devices=()):
    """Preserve the supported owned state; the intended jitter draw is outside."""
    modules = list({id(m): m for root in roots for m in root.modules()}.values())
    parameters = list({id(p): p for root in roots for p in root.parameters()}.values())
    modes = [(m, m.training) for m in modules]
    buffers = [(m, dict(m._buffers),
                {k: None if v is None else v.detach().clone() for k, v in m._buffers.items()},
                set(m._non_persistent_buffers_set)) for m in modules]
    gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters]
    stream_states = [(s, s.get_state()) for s in {id(s): s for s in streams}.values()]
    try:
        with torch.random.fork_rng(devices=list(devices)):
            for root in roots:
                root.eval()
            yield
    finally:
        with torch.no_grad():
            for stream, old in stream_states:
                stream.set_state(old)
            for parameter, original, old in gradients:
                parameter.grad = original
                if original is not None:
                    original.copy_(old)
            for module, original, old, nonpersistent in buffers:
                module._buffers.clear()
                module._buffers.update(original)
                module._non_persistent_buffers_set.clear()
                module._non_persistent_buffers_set.update(nonpersistent)
                for name, value in original.items():
                    if value is not None:
                        value.copy_(old[name])
            for module, mode in modes:
                module.training = mode


@dataclass(frozen=True)
class View:
    features: torch.Tensor
    metric: torch.Tensor
    cells: torch.Tensor
    categories: torch.Tensor
    groups: torch.Tensor
    eligible: torch.Tensor
    pvalues: torch.Tensor


@torch.no_grad()
def observe_view(snapshot, features, *, coordinates, cached_fast=False):
    """Actual device chart predicates; one packed bulk transfer to CPU planning."""
    metric = snapshot.row_features if cached_fast else snapshot.transform(features)
    cells = snapshot.query_cell_ids if cached_fast else snapshot._assign_metric(metric)[0]
    categories, scores = snapshot._count_categories_metric(metric, cells)
    pvalues = (1.+len(snapshot.null_scores)-torch.searchsorted(snapshot.null_scores,scores))/(1.+len(snapshot.null_scores))
    groups = snapshot._mass_topology()[cells]
    finite = torch.isfinite(coordinates).all(1) & torch.isfinite(metric).all(1)
    eligible = finite & (pvalues>Q) & (categories.remainder(2)==0)
    # p>Q implies no BH support flag because every possible BH cutoff<=Q.
    packed = torch.cat((metric, cells[:,None].double(),categories[:,None].double(),
                        groups[:,None].double(),eligible[:,None].double(),pvalues[:,None].double()),1).cpu()
    r=snapshot.rank
    return View(features,packed[:,:r],packed[:,r].long(),packed[:,r+1].long(),
                packed[:,r+2].long(),packed[:,r+3].bool(),packed[:,r+4])


@dataclass(frozen=True)
class CandidatePairs:
    children: torch.Tensor
    parents: torch.Tensor
    pre_gain: torch.Tensor
    eligible_rows: int
    positive_pairs_before_budget: int
    protected_rows: torch.Tensor
    budget: int
    earlier_ordinary: int


@torch.no_grad()
def propose_pairs(snapshot, fixed, fast, ema, *, earlier_ordinary, reserved_rows):
    """Stable bounded pools, no pair matrix, unique disjoint source incarnations."""
    n = len(fast.metric)
    if type(earlier_ordinary) is not int or not 0 <= earlier_ordinary <= math.floor(Q * n):
        raise ValueError("invalid shared ordinary prefix")
    if (reserved_rows.ndim != 1 or reserved_rows.dtype != torch.long
            or bool(((reserved_rows < 0) | (reserved_rows >= n)).any())):
        raise ValueError("invalid complete reaction reservation union")
    budget = math.floor(Q * n) - earlier_ordinary
    empty = torch.empty(0, dtype=torch.long, device=fast.metric.device)
    if fixed is None:
        return CandidatePairs(empty, empty, torch.empty(0, dtype=torch.float64), 0, 0, empty,
                              budget, earlier_ordinary)
    reserved = torch.zeros(n, dtype=torch.bool, device=fast.metric.device)
    reserved[reserved_rows] = True
    legal = fast.eligible & ema.eligible & (fast.groups == ema.groups)
    protected = torch.zeros_like(legal)
    # Preserve an actual jointly eligible survivor in each occupied cell/view.
    for view in (fast, ema):
        for cell in range(snapshot.cells):
            rows = (legal & (view.cells == cell)).nonzero().flatten()
            if len(rows):
                protected[rows[0]] = True
    legal &= ~reserved
    psi = fixed.psi(ema.metric, ema.groups)
    projection = (fixed.directions[ema.groups] * psi).sum(1)
    signature = fast.categories * (2 * snapshot.cells) + ema.categories
    rows = legal.nonzero().flatten()
    order = rows[signature[rows].argsort(stable=True)]
    _, counts = torch.unique_consecutive(signature[order], return_counts=True)
    all_children, all_parents, all_gain = [], [], []
    start = 0
    for size in counts.cpu().tolist():
        bucket = order[start:start + size]
        start += size
        ranked = bucket[projection[bucket].argsort(descending=True, stable=True)]
        parent_count = min(POOL, size // 2)
        parents = ranked[:parent_count]
        candidates = ranked[parent_count:]
        candidates = candidates[~protected[candidates]]
        children = candidates[projection[candidates].argsort(stable=True)[:POOL]]
        count = min(len(children), len(parents))
        if not count:
            continue
        children, parents = children[:count], parents[:count]
        groups = ema.groups[children]
        delta = (psi[parents] - psi[children]) / fixed.ema_counts[groups, None]
        residual = fixed.even_means[groups] - fixed.ema_means[groups]
        gain = fixed.weights[groups] * (2 * (residual * delta).sum(1) - delta.square().sum(1))
        keep = torch.isfinite(gain) & (gain > 0)
        all_children.append(children[keep]); all_parents.append(parents[keep]); all_gain.append(gain[keep])
    children = torch.cat(all_children) if all_children else empty
    parents = torch.cat(all_parents) if all_parents else empty
    gains = torch.cat(all_gain) if all_gain else torch.empty(0, dtype=torch.float64, device=empty.device)
    # Stable ties are parent then child IDs; last stable gain sort is primary.
    ordered = parents.argsort(stable=True)
    ordered = ordered[children[ordered].argsort(stable=True)]
    ordered = ordered[gains[ordered].argsort(descending=True, stable=True)]
    total = len(ordered)
    ordered = ordered[:budget]
    children, parents, gains = children[ordered], parents[ordered], gains[ordered]
    assert len(torch.unique(children)) == len(children)
    assert len(torch.unique(parents)) == len(parents)
    assert not bool(torch.isin(children, parents).any())
    assert not bool(reserved[children].any() | reserved[parents].any())
    return CandidatePairs(children, parents, gains, int(legal.sum()), total,
                          protected.nonzero().flatten(), budget, earlier_ordinary)


def _version(value):
    return id(value), value._version, tuple(value.shape), value.dtype, value.device


def epoch(state, snapshot, *, include_stream=True):
    tensors = [state.prior.z, state.ema_prior.z, state.lineage.neighbors,
               *state.model_tensors, *[v for v in state.row_state.values() if isinstance(v, torch.Tensor)]]
    if state.history is not None:
        tensors.append(state.history)
    tensors += [snapshot.mean, snapshot.scale, snapshot.basis, snapshot.centers,
                snapshot.cell_scale, snapshot.count_boundary, snapshot.null_scores,
                snapshot._mass_topology()]
    # Buffer restoration may increment _version while preserving object and
    # bytes; parameter/table versions and exact buffer bindings/bytes differ.
    buffers = state.buffer_epoch()
    bandwidth = (_version(state.bandwidth), tensor_digest(state.bandwidth))
    stream = (id(state.stream), tensor_digest(state.stream.get_state())) if include_stream else None
    return tuple(map(_version, tensors)), buffers, bandwidth, stream, id(snapshot), snapshot.cache_version


def tensor_digest(*values):
    h = hashlib.sha256()
    for value in values:
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode()); h.update(tensor_digest(value[key]).encode())
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class CopyPacket:
    children: torch.Tensor
    parents: torch.Tensor
    fast_coordinates: torch.Tensor
    ema_coordinates: torch.Tensor
    parent_row_state: dict
    parent_history: object
    expected_epoch: object
    fingerprint: str
    fast_features: torch.Tensor
    ema_features: torch.Tensor

    def content_digest(self):
        return tensor_digest(self.children, self.parents, self.fast_coordinates, self.ema_coordinates,
                             self.parent_row_state, self.parent_history, self.fast_features, self.ema_features)


@torch.no_grad()
def preview_pairs(snapshot, fixed, fast, ema, pairs, state, *, stream, features_fast,
                  features_ema, geometry, roots=(), owned_streams=(), devices=()):
    """Exactly one draw; pure virtual objective; accepted coordinates never redrawn."""
    before_draw = epoch(state, snapshot, include_stream=False)
    if stream is not state.stream:
        raise ValueError("packet stream must be the declared owned reaction stream")
    child_host, parent_host = pairs.children, pairs.parents
    row_pairs = torch.stack((child_host,parent_host),1).to(state.prior.z.device)
    child, parent = row_pairs[:,0],row_pairs[:,1]
    if not len(child):
        return None, dict(attempts=0, accepted=0, reason="no_residual_candidate_slots")
    source_fast, source_ema = state.prior.z[parent], state.ema_prior.z[parent]
    noise = torch.randn(source_fast.shape, dtype=source_fast.dtype, device=source_fast.device, generator=stream)
    proposed_fast = source_fast + geometry.displacement(source_fast, state.prior,
                    state.bandwidth, noise, rows=parent)
    proposed_ema = source_ema + geometry.displacement(source_ema, state.ema_prior,
                    state.bandwidth, noise, rows=parent)
    before = epoch(state, snapshot)
    if epoch(state, snapshot, include_stream=False) != before_draw:
        raise RuntimeError("paired jitter changed semantic pre-action state")
    if not bool(torch.isfinite(proposed_fast).all() & torch.isfinite(proposed_ema).all()):
        return None, dict(attempts=len(child), accepted=0, reason="nonfinite_preview_latents")
    with neutral_observation(roots, (*owned_streams, stream), devices):
        new_fast_features = features_fast(proposed_fast)
        new_ema_features = features_ema(proposed_ema)
    if epoch(state, snapshot) != before:
        raise RuntimeError("observation changed the pre-action packet epoch")
    if (not isinstance(new_fast_features,torch.Tensor) or not isinstance(new_ema_features,torch.Tensor)
            or not bool(torch.isfinite(new_fast_features).all() & torch.isfinite(new_ema_features).all())):
        return None, dict(attempts=len(child), accepted=0, reason="nonfinite_preview_features")
    vf = observe_view(snapshot, new_fast_features, coordinates=proposed_fast)
    ve = observe_view(snapshot, new_ema_features, coordinates=proposed_ema)
    retained = (vf.eligible & ve.eligible
        & (vf.cells == fast.cells[child_host]) & (ve.cells == ema.cells[child_host])
        & (vf.categories == fast.categories[child_host]) & (ve.categories == ema.categories[child_host])
        & (vf.groups == fast.groups[child_host]) & (ve.groups == ema.groups[child_host])
        & (vf.groups == ve.groups))
    working = fixed.ema_means.clone()
    old_psi = fixed.psi(ema.metric[child_host], ema.groups[child_host])
    new_psi = fixed.psi(ve.metric, ve.groups)
    # Every scalar loop below is CPU float64; no per-pair device reads.
    energy = fixed.energy(working)
    initial = float(energy)
    accepted, gains = [], []
    for index in range(len(child)):
        if not bool(retained[index]):
            continue
        group = int(ema.groups[child_host[index]])
        proposed_mean = working[group] + (new_psi[index] - old_psi[index]) / fixed.ema_counts[group]
        old_group = fixed.weights[group] * (fixed.even_means[group] - working[group]).square().sum()
        new_group = fixed.weights[group] * (fixed.even_means[group] - proposed_mean).square().sum()
        next_energy = energy - old_group + new_group
        if bool(torch.isfinite(next_energy) & (next_energy < energy)):
            gains.append(float(energy - next_energy)); accepted.append(index)
            working[group] = proposed_mean
            energy = next_energy
    detail = dict(attempts=len(child), category_retained=int(retained.sum()), accepted=len(accepted),
        objective_before=initial, objective_after_virtual=float(energy), actual_gains=gains,
        paired_noise_draws=1, noise_rows=len(child), no_redraw=True,
        preview_work=dict(geometry.work))
    if not accepted:
        return None, dict(detail, reason="no_actual_retained_progress")
    indices = torch.tensor(accepted, dtype=torch.long).to(child.device)
    accepted_child, accepted_parent = child[indices].clone(), parent[indices].clone()
    row_state = {key: value[accepted_parent].clone() for key, value in state.row_state.items()
                 if isinstance(value, torch.Tensor) and value.shape == state.prior.z.shape}
    history = None if state.history is None else state.history[accepted_parent].clone()
    values = (accepted_child, accepted_parent, proposed_fast[indices].clone(), proposed_ema[indices].clone(),
              row_state, history, new_fast_features[indices].clone(), new_ema_features[indices].clone())
    packet = CopyPacket(values[0], values[1], values[2], values[3], values[4], values[5], before,
                        tensor_digest(*values), values[6], values[7])
    return packet, dict(detail, reason=None, children=packet.children.tolist(), parents=packet.parents.tolist())


@torch.no_grad()
def commit_packet(packet, state, snapshot):
    """Private scratch commit only. One consumed packet, no draw or re-computation."""
    if (packet.fingerprint in state.consumed or packet.content_digest() != packet.fingerprint
            or epoch(state, snapshot) != packet.expected_epoch):
        raise ValueError("stale, mutated or consumed copy packet")
    child, parent = packet.children, packet.parents
    n, width = state.prior.z.shape
    if (state.ema_prior.z.shape != state.prior.z.shape
            or state.ema_prior.z.dtype != state.prior.z.dtype
            or state.ema_prior.z.device != state.prior.z.device
            or state.lineage.n != n or state.lineage.neighbors.device != state.prior.z.device
            or snapshot.device != state.prior.z.device
            or (state.history is not None and len(state.history) != n)):
        raise ValueError("paired tables, history, lineage and chart must share the table layout")
    if (not isinstance(child, torch.Tensor) or not isinstance(parent, torch.Tensor)
            or child.ndim != 1 or parent.ndim != 1 or child.dtype != torch.long
            or parent.dtype != torch.long or child.device != state.prior.z.device
            or parent.device != child.device or child.shape != parent.shape
            or bool(((child < 0) | (child >= n) | (parent < 0) | (parent >= n)).any())):
        raise ValueError("copy packet requires valid paired table row IDs")
    if (len(torch.unique(child)) != len(child) or len(torch.unique(parent)) != len(parent)
            or bool(torch.isin(child, parent).any()) or len(child) != len(parent)):
        raise ValueError("copy packet row incarnations overlap")
    for value, table in ((packet.fast_coordinates, state.prior.z),
                         (packet.ema_coordinates, state.ema_prior.z)):
        if (not isinstance(value, torch.Tensor) or value.shape != (len(child), *table.shape[1:])
                or value.dtype != table.dtype or value.device != table.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("copy packet requires finite matching paired latent coordinates")
    if (snapshot.query_cell_ids is None or snapshot.row_features is None
            or snapshot.query_cell_ids.shape != (n,) or snapshot.query_cell_ids.dtype != torch.long
            or snapshot.query_cell_ids.device != snapshot.device
            or snapshot.row_features.shape != (n, snapshot.rank)
            or snapshot.row_features.device != snapshot.device):
        raise ValueError("copy packet requires the initialized current FAST cache")
    for value in (packet.fast_features, packet.ema_features):
        if (not isinstance(value, torch.Tensor) or value.shape != (len(child), snapshot.width)
                or value.device != snapshot.device or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError("copy packet requires finite matching learned feature rows")
    expected_keys = {key for key, value in state.row_state.items()
                     if isinstance(value, torch.Tensor) and value.shape == state.prior.z.shape}
    if not isinstance(packet.parent_row_state, dict) or set(packet.parent_row_state) != expected_keys:
        raise ValueError("copy packet optimizer-row inheritance schema mismatch")
    for key, value in packet.parent_row_state.items():
        target = state.row_state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != (len(child), *target.shape[1:])
                or value.dtype != target.dtype or value.device != target.device
                or tensor_digest(value) != tensor_digest(target[parent])):
            raise ValueError("copy packet optimizer parent bytes mismatch")
    if state.history is None:
        if packet.parent_history is not None:
            raise ValueError("copy packet has unexpected latent history")
    elif (not isinstance(packet.parent_history, torch.Tensor)
            or packet.parent_history.shape != (len(child), *state.history.shape[1:])
            or packet.parent_history.dtype != state.history.dtype
            or packet.parent_history.device != state.history.device
            or tensor_digest(packet.parent_history) != tensor_digest(state.history[parent])):
        raise ValueError("copy packet latent-history parent bytes mismatch")
    state.lineage._rows(child); state.lineage._rows(parent)
    state.lineage.validate(state.lineage.neighbors)
    # Every shape/type/content/cache check above precedes the first row write.
    state.prior.z[child] = packet.fast_coordinates
    state.ema_prior.z[child] = packet.ema_coordinates
    for key, values in packet.parent_row_state.items():
        state.row_state[key][child] = values
    if state.history is not None:
        state.history[child] = packet.parent_history
    state.lineage.register_copies(child, parent)
    state.lineage.validate(state.lineage.neighbors)
    invalidated, affected = snapshot.refresh_rows(child, packet.fast_features)
    state.consumed.add(packet.fingerprint)
    return dict(moved_rows=child.tolist(), invalidated_rows=int(invalidated.sum()),
                affected_cells=affected.nonzero().flatten().tolist(), lineage_work=dict(state.lineage.work),
                invalidated=invalidated, affected=affected)


MEAN_POLICY = POLICY
PLANNING_POLICY = "bulk_current_device_chart_stable_CPU_float64_fixed_u_v1"
INSIDE_POLICY = "both_views_original_and_actual_supported_inside_own_cell_common_group_v1"
SCALAR_FIELDS = ("alpha", "radius", "known_range", "mean", "variance_ddof1",
                 "variance_penalty", "range_penalty", "lower_bound",
                 "objective_before_mean", "objective_after_mean")
MEAN_KEYS = {"schema", "policy", "status", "reason", "step", "snapshot", "cells", "rank",
             "observations", "attempts", "moves", *SCALAR_FIELDS}


def initial_mean_diagnostics():
    return dict(schema=1, policy=MEAN_POLICY, status="initial", reason="not_evaluated",
                step=0, snapshot=0, cells=0, rank=0, observations=0, attempts=0, moves=0,
                **{key: None for key in SCALAR_FIELDS})


def mean_diagnostics(witness, *, step, snapshot_serial, cells, rank, observations):
    valid = witness["valid"]
    return dict(schema=1, policy=MEAN_POLICY,
        status="firing" if witness["fires"] else ("veto" if valid else "invalid"),
        reason=None if witness["fires"] else ("nonpositive_lower_bound" if valid else witness["reason"]),
        step=step, snapshot=snapshot_serial, cells=cells, rank=rank, observations=observations,
        alpha=Q/(3*cells+3), radius=math.sqrt(rank/Q) if valid else None,
        known_range=4*math.sqrt(rank/Q) if valid else None,
        **{key: witness.get(key) if valid else None
           for key in ("mean","variance_ddof1","variance_penalty","range_penalty","lower_bound")},
        attempts=0, moves=0, objective_before_mean=None, objective_after_mean=None)


@torch.no_grad()
def capture_features(backend, trainer, model, latents):
    """Clean direct G→D-head measurement, without another latent or output draw."""
    roots = (trainer.G, trainer.ema_G, trainer.D)
    streams = (backend.stream, *[getattr(trainer, name) for name in trainer._STREAMS])
    devices = [trainer.device.index] if trainer.device.type == "cuda" else []
    pieces, finite_outputs = [], []
    with neutral_observation(roots, streams, devices):
        for block in latents.detach().split(backend.settings["chunk"]):
            raw = model(block)
            if tuple(raw.shape[1:]) != backend.sample_shape:
                raise ValueError("mean preview output shape disagrees with real FIFO")
            finite_outputs.append(torch.isfinite(raw).flatten(1).all(1))
            features = backend._features(trainer, raw, chunk=backend.settings["chunk"])
            finite_outputs.append(torch.isfinite(features).all(1))
            pieces.append(features)
    # One final finite predicate, rather than a device scalar per feature chunk.
    return torch.cat(pieces) if bool(torch.cat(finite_outputs).all()) else None


@torch.no_grad()
def prepare_mean_witness(backend, trainer, snapshot, real_features):
    """Fixed BEFORE odd-count-driven actions; never retested after the prefix."""
    backend.counters["mean_evals"] += 1
    fixed = None
    reason = None
    if snapshot.rank > 8:
        reason = "mean_rank_work_bound"
    elif not snapshot.valid_metric or snapshot.duplicate_fraction > Q:
        reason = "invalid_or_duplicate_chart"
    elif not bool(torch.isfinite(trainer.ema_prior.z).all()):
        reason = "nonfinite_pre_mean_EMA_coordinates"
    else:
        features = capture_features(backend, trainer, trainer.ema_G, trainer.ema_prior.z)
        backend.counters["mean_forward_rows"] += backend.N
        if features is None:
            reason = "nonfinite_pre_mean_EMA_features"
        else:
            fixed, reason = freeze_moment(snapshot, real_features[0::2], features)
    if fixed is None:
        witness = dict(valid=False, fires=False, reason=reason or "no_valid_frozen_moment")
    else:
        witness = odd_witness(snapshot, fixed, real_features[1::2])
    stamp = mean_diagnostics(witness, step=trainer.completed_steps+1,
        snapshot_serial=backend.snapshot_serial, cells=snapshot.cells, rank=snapshot.rank,
        observations=snapshot.calibration_rows)
    backend.counters["mean_witness_fires"] += int(stamp["status"] == "firing")
    return fixed, stamp


def copy_state(backend, trainer):
    """Ephemeral supported-owned context; parameter versions and exact buffer bytes."""
    roots = (trainer.G, trainer.ema_G, trainer.D)
    parameters = list({id(p):p for root in roots for p in root.parameters()}.values())
    def buffer_epoch():
        return tuple((id(module), tuple((name,id(value),tensor_digest(value))
                      for name,value in module._buffers.items()),
                      tuple(sorted(module._non_persistent_buffers_set)))
                     for root in roots for module in root.modules())
    return SimpleNamespace(prior=trainer.prior, ema_prior=trainer.ema_prior,
        lineage=backend.lineage, row_state=trainer.opt_g.state.get(trainer.prior.z,{}),
        history=trainer.opt_g.latent_history, consumed=set(),
        bandwidth=trainer.controller.latent_bandwidth, stream=backend.stream,
        model_tensors=parameters, buffer_epoch=buffer_epoch)


@torch.no_grad()
def run_mean_phase(backend, trainer, snapshot, fixed, stamp, *, earlier_ordinary, reserved_rows):
    """Post-prefix means/counts, original fixed u, one bounded copy packet."""
    empty = torch.empty(0, dtype=torch.long, device=trainer.prior.z.device)
    result = dict(children=empty, parents=empty, invalidated=None, affected=None,
                  diagnostics=dict(stamp), packet=None)
    diag = result["diagnostics"]
    budget = math.floor(Q*backend.N)-earlier_ordinary
    if budget < 0:
        raise ValueError("earlier ordinary actions exceed the shared budget")
    if diag["status"] != "firing":
        return result
    if backend.dry_run or budget == 0:
        diag["reason"] = "dry_run" if backend.dry_run else "no_remaining_budget"
        return result
    features = capture_features(backend, trainer, trainer.ema_G, trainer.ema_prior.z)
    backend.counters["mean_forward_rows"] += backend.N
    if features is None:
        diag["reason"] = "nonfinite_current_EMA_features"
        return result
    fast = observe_view(snapshot, None, coordinates=trainer.prior.z, cached_fast=True)
    ema = observe_view(snapshot, features, coordinates=trainer.ema_prior.z)
    means, counts = group_means(fixed.psi(ema.metric, ema.groups), ema.groups, snapshot.mass_groups)
    if bool((counts == 0).any()):
        diag["reason"] = "missing_current_EMA_group"
        return result
    # Preserve frozen unit directions; current counts are the actual replacement denominators.
    action = replace(fixed, ema_means=means, ema_counts=counts)
    baseline = float(action.energy())
    diag.update(objective_before_mean=baseline, objective_after_mean=baseline)
    pairs = propose_pairs(snapshot, action, fast, ema, earlier_ordinary=earlier_ordinary,
                          reserved_rows=torch.unique(reserved_rows).cpu())
    if not len(pairs.children):
        diag["reason"] = "no_legal_positive_pairs"
        return result
    state = copy_state(backend, trainer)
    def fast_feature(points):
        return capture_features(backend, trainer, trainer.G, points)
    def ema_feature(points):
        return capture_features(backend, trainer, trainer.ema_G, points)
    packet, preview = preview_pairs(snapshot, action, fast, ema, pairs, state,
        stream=backend.stream, features_fast=fast_feature, features_ema=ema_feature,
        geometry=backend.latent_geometry)
    diag["attempts"] = preview["attempts"]
    backend.counters["mean_preview_rows"] += 2*preview["attempts"]
    backend.counters["mean_forward_rows"] += 2*preview["attempts"]
    if packet is None:
        diag["reason"] = preview["reason"]
        return result
    # Verify the actual device-produced features and final declared CPU objective
    # before writes; no GPU scalar is read inside the sequential decision loop.
    after_metric = snapshot.transform(packet.ema_features).cpu()
    after_groups = ema.groups[packet.children.cpu()]
    psi = action.psi(ema.metric, ema.groups)
    psi[packet.children.cpu()] = action.psi(after_metric, after_groups)
    actual_means, actual_counts = group_means(psi, ema.groups, snapshot.mass_groups)
    actual_energy = float(action.energy(actual_means))
    if (not torch.equal(actual_counts, counts) or not actual_energy < baseline
            or not math.isclose(actual_energy,preview["objective_after_virtual"],rel_tol=1e-10,abs_tol=1e-12)):
        raise RuntimeError("mean packet actual objective disagrees with its virtual ledger")
    applied = commit_packet(packet, state, snapshot)
    diag.update(moves=len(packet.children), reason=None, objective_after_mean=actual_energy)
    result.update(children=packet.children, parents=packet.parents, packet=packet,
                  invalidated=applied["invalidated"], affected=applied["affected"])
    return result


def check_mean_diagnostics(stamp, *, last, paired, n, snapshot_serial, dry_run):
    """Typed scalar receipt only; no chart/packet authority survives reload."""
    if not isinstance(stamp,dict) or set(stamp)!=MEAN_KEYS:
        raise ValueError("invalid mean-transport metadata keys")
    integers=("schema","step","snapshot","cells","rank","observations","attempts","moves")
    if (any(type(stamp[k]) is not int or stamp[k]<0 for k in integers)
            or stamp["schema"]!=1 or type(stamp["policy"]) is not str or stamp["policy"]!=MEAN_POLICY
            or type(stamp["status"]) is not str
            or stamp["status"] not in ("initial","invalid","veto","firing")
            or (stamp["reason"] is not None and type(stamp["reason"]) is not str)
            or any(stamp[k] is not None and (type(stamp[k]) is not float or not math.isfinite(stamp[k]))
                   for k in SCALAR_FIELDS)):
        raise ValueError("invalid mean-transport typed scalars")
    if snapshot_serial==0:
        if stamp!=initial_mean_diagnostics():
            raise ValueError("invalid initial mean-transport receipt")
        return
    if (stamp["snapshot"]!=snapshot_serial or stamp["step"]!=last.get("step")
            or any(stamp[k]!=paired[k] for k in ("snapshot","step","cells","rank"))
            or stamp["observations"]!=paired["calibration_rows"] or stamp["observations"]!=n//2
            or stamp["alpha"]!=Q/(3*stamp["cells"]+3)
            or stamp["moves"]>stamp["attempts"] or stamp["attempts"]>math.floor(Q*n)
            or stamp["moves"]!=last.get("ordinary_mean_moves")
            or (dry_run and (stamp["attempts"] or stamp["moves"]))):
        raise ValueError("mean-transport receipt disagrees with its reaction")
    ordinary,mean_moves=last.get("ordinary_moves"),last.get("ordinary_mean_moves")
    if (type(ordinary) is not int or type(mean_moves) is not int
            or not 0<=mean_moves<=ordinary<=math.floor(Q*n)
            or stamp["attempts"]>math.floor(Q*n)-(ordinary-mean_moves)):
        raise ValueError("mean previews exceed the actual residual ordinary budget")
    stats=("radius","known_range","mean","variance_ddof1","variance_penalty","range_penalty","lower_bound")
    if stamp["status"]=="invalid":
        if (not stamp["reason"] or any(stamp[k] is not None for k in stats)
                or stamp["attempts"] or stamp["moves"]
                or stamp["objective_before_mean"] is not None or stamp["objective_after_mean"] is not None):
            raise ValueError("invalid mean witness cannot authorize actions")
    elif stamp["status"] in ("veto","firing"):
        if (any(stamp[k] is None for k in stats) or not 0<stamp["rank"]<=8
                or stamp["observations"]<=1 or stamp["radius"]!=math.sqrt(stamp["rank"]/Q)
                or stamp["known_range"]!=4*stamp["radius"]
                or any(stamp[k]<0 for k in ("variance_ddof1","variance_penalty","range_penalty"))
                or abs(stamp["mean"])>2*stamp["radius"]+1e-10
                or stamp["variance_ddof1"]>(2*stamp["radius"])**2*stamp["observations"]/(stamp["observations"]-1)+1e-10):
            raise ValueError("invalid fixed mean witness range/variance")
        t=math.log(2/stamp["alpha"])
        expected_var=math.sqrt(2*stamp["variance_ddof1"]*t/stamp["observations"])
        expected_range=(7/3)*stamp["known_range"]*t/(stamp["observations"]-1)
        for actual,expected in ((stamp["variance_penalty"],expected_var),
                                (stamp["range_penalty"],expected_range),
                                (stamp["lower_bound"],stamp["mean"]-expected_var-expected_range)):
            if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("mean witness bound arithmetic mismatch")
        if (stamp["status"]=="firing")!=(stamp["lower_bound"]>0):
            raise ValueError("mean witness status disagrees with its strict sign")
        if stamp["status"]=="veto" and (stamp["attempts"] or stamp["moves"]
                or stamp["reason"]!="nonpositive_lower_bound"
                or stamp["objective_before_mean"] is not None or stamp["objective_after_mean"] is not None):
            raise ValueError("vetoed mean witness cannot preview or move rows")
        if stamp["status"]=="firing" and not stamp["moves"] and not stamp["reason"]:
            raise ValueError("zero mean moves require their actual action-veto reason")
    else:
        raise ValueError("initial mean receipt cannot represent a reaction")
    before,after=stamp["objective_before_mean"],stamp["objective_after_mean"]
    if ((before is None)!=(after is None)
            or (before is not None and (before<0 or after<0 or after>before))
            or (stamp["moves"] and (before is None or not after<before or stamp["reason"] is not None))
            or (not stamp["moves"] and before is not None and after!=before)):
        raise ValueError("mean action objective does not match actual progress")
