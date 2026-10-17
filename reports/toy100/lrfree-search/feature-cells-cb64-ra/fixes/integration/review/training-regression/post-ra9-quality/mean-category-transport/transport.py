"""Scratch bounded within-category copy proposals, exact packets, no trainer step."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch

from witness import Q, group_means

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
def observe_view(snapshot, features, *, coordinates):
    metric = snapshot.transform(features)
    cells, _ = snapshot._assign_metric(metric)
    categories, _ = snapshot._count_categories_metric(metric, cells)
    flags, pvalues, _ = snapshot.support(features)
    groups = snapshot._mass_topology()[cells]
    if len(coordinates) != len(features):
        raise ValueError("feature/latent row mismatch")
    eligible = (~flags & (pvalues > Q) & (categories % 2 == 0)
                & torch.isfinite(features).all(1) & torch.isfinite(coordinates).all(1))
    return View(features, metric, cells, categories, groups, eligible, pvalues)


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
    n = len(fast.features)
    if type(earlier_ordinary) is not int or not 0 <= earlier_ordinary <= math.floor(Q * n):
        raise ValueError("invalid shared ordinary prefix")
    if (reserved_rows.ndim != 1 or reserved_rows.dtype != torch.long
            or bool(((reserved_rows < 0) | (reserved_rows >= n)).any())):
        raise ValueError("invalid complete reaction reservation union")
    budget = math.floor(Q * n) - earlier_ordinary
    empty = torch.empty(0, dtype=torch.long, device=fast.features.device)
    if fixed is None:
        return CandidatePairs(empty, empty, torch.empty(0, dtype=torch.float64), 0, 0, empty,
                              budget, earlier_ordinary)
    reserved = torch.zeros(n, dtype=torch.bool, device=fast.features.device)
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
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
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
    child, parent = pairs.children, pairs.parents
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
    if not bool(torch.isfinite(new_fast_features).all() & torch.isfinite(new_ema_features).all()):
        return None, dict(attempts=len(child), accepted=0, reason="nonfinite_preview_features")
    vf = observe_view(snapshot, new_fast_features, coordinates=proposed_fast)
    ve = observe_view(snapshot, new_ema_features, coordinates=proposed_ema)
    retained = (vf.eligible & ve.eligible
        & (vf.cells == fast.cells[child]) & (ve.cells == ema.cells[child])
        & (vf.categories == fast.categories[child]) & (ve.categories == ema.categories[child])
        & (vf.groups == fast.groups[child]) & (ve.groups == ema.groups[child])
        & (vf.groups == ve.groups))
    working = fixed.ema_means.clone()
    old_psi = fixed.psi(ema.metric[child], ema.groups[child])
    new_psi = fixed.psi(ve.metric, ve.groups)
    energy = fixed.energy(working)
    initial = float(energy)
    accepted, gains = [], []
    for index in range(len(child)):
        if not bool(retained[index]):
            continue
        group = int(ema.groups[child[index]])
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
    indices = torch.tensor(accepted, dtype=torch.long, device=child.device)
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
                affected_cells=affected.nonzero().flatten().tolist(), lineage_work=dict(state.lineage.work))
