"""Bounded learned-feature cells, categorical transport and real-anchor repair.

This is an alternative reaction law, not the kNN/Beta law in birth_death.py.
No data-space distances, evaluator imports or full-table neighbor searches.
"""
import math
import time
import weakref
from fractions import Fraction
import torch
from .birth_death import ParticleBirthDeath

LLOYD_PASSES = 4
POWER_PASSES = 4
SCALE_PRIOR = 4.
PARENT_RESERVOIR = 64
REAL_ANCHORS = 4
PARENT_RANK_CAP = 16
JITTER_STD = .025
JITTER_CAP = .05
Q = .05


def population_policy(n):
    """Whether finite conformal/BH resolution permits the existing guard.

    An N-row FIFO leaves floor(N/2) independent calibration rows. A flagged
    set needs ceil(N / (Q*(M+1))) rows, while the guard permits floor(Q*N).
    Use exact rational arithmetic: this is a feasibility rule, not a cutoff
    fitted to task quality.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("population must be a positive integer")
    q = Fraction(str(Q))
    calibration_rows = n // 2
    denominator = q.numerator * (calibration_rows + 1)
    minimum_flags = (n * q.denominator + denominator - 1) // denominator
    allowed_flags = n * q.numerator // q.denominator
    feasible = minimum_flags <= allowed_flags
    return dict(schema=1, requested_backend="feature_cells", population=n,
                q=Q, calibration_rows=calibration_rows,
                minimum_bh_flags=minimum_flags, maximum_guard_flags=allowed_flags,
                finite_resolution_feasible=feasible,
                actual_backend="feature_cells" if feasible else "knn",
                matching_sampler="feature_cells" if feasible else "controller_reference",
                rule="ceil(N/[Q*(floor(N/2)+1)]) <= floor(Q*N)")


class SmallPopulationReferenceBirthDeath(ParticleBirthDeath):
    """Unchanged reference reaction/sampling law with explicit route metadata."""
    def __init__(self, trainer, seed):
        super().__init__(trainer, seed)
        self.population_policy = population_policy(self.N)
        if self.population_policy["finite_resolution_feasible"]:
            raise ValueError("reference fallback requires impossible feature-cell resolution")

    def diagnostics(self):
        result = super().diagnostics()
        result.update(backend="knn", population_policy=dict(self.population_policy))
        return result

    def state_dict(self):
        result = super().state_dict()
        result["population_policy"] = dict(self.population_policy)
        return result

    def check_state(self, state):
        if (not isinstance(state, dict)
                or state.get("population_policy") != self.population_policy):
            raise ValueError("incompatible small-population backend/sampler metadata")
        super().check_state({key: value for key, value in state.items()
                             if key != "population_policy"})


def make_feature_cell_birth_death(trainer, seed):
    if population_policy(len(trainer.prior.z))["finite_resolution_feasible"]:
        return FeatureCellBirthDeath(trainer, seed)
    return SmallPopulationReferenceBirthDeath(trainer, seed)


class LatentLineage:
    """Bounded symmetric row-copy links; semantic state shared by live/EMA.

    Overwriting a row removes all its old incident edges. New parent/child
    pairs are inserted newest first. A full endpoint evicts its oldest edge
    symmetrically. Links refer to current row incarnations, not old positions.
    """
    def __init__(self, n, degree, device):
        if (type(n) is not int or n <= 0 or type(degree) is not int
                or not 0 < degree <= min(PARENT_RESERVOIR, n-1)):
            raise ValueError("invalid latent lineage size/degree")
        self.n, self.degree = n, degree
        self.neighbors = torch.full((n, degree), -1, dtype=torch.long, device=device)
        self.work = dict(updates=0, copied_rows=0, reciprocal_slots=0,
                         maximum_reciprocal_slots=0, maximum_parent_rows=0)

    def _rows(self, rows):
        if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.dtype != torch.long
                or rows.device != self.neighbors.device
                or bool(((rows < 0) | (rows >= self.n)).any())):
            raise ValueError("invalid latent lineage row indices")

    def validate(self, neighbors):
        if (not isinstance(neighbors, torch.Tensor) or neighbors.shape != self.neighbors.shape
                or neighbors.dtype != torch.long
                or bool(((neighbors < -1) | (neighbors >= self.n)).any())):
            raise ValueError("invalid latent lineage checkpoint shape/indices")
        valid = neighbors >= 0
        rows, slots = valid.nonzero(as_tuple=True)
        partners = neighbors[rows, slots]
        if bool((rows == partners).any()):
            raise ValueError("latent lineage cannot contain self edges")
        ordered = neighbors.masked_fill(~valid, self.n).sort(1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < self.n)).any()):
            raise ValueError("latent lineage contains repeated neighbors")
        if len(rows) and not bool((neighbors[partners] == rows[:, None]).any(1).all()):
            raise ValueError("latent lineage must be symmetric")

    def _compact(self, rows):
        old = self.neighbors[rows]
        slots = torch.arange(self.degree, device=old.device)[None].expand_as(old)
        order = torch.where(old >= 0, slots, slots+self.degree).argsort(dim=1, stable=True)
        return old.gather(1, order)

    @torch.no_grad()
    def _remove_reciprocals(self, sources, adjacent):
        # Only known neighbors of touched endpoints are read. Padding uses a
        # masked row0 read; it never writes row0 merely because a slot is empty.
        targets = adjacent.clamp_min(0)
        remove = (self.neighbors[targets] == sources[:, None, None]) & (adjacent >= 0)[:, :, None]
        slots = torch.arange(self.degree, device=adjacent.device)
        flat = targets[:, :, None]*self.degree+slots
        self.neighbors.view(-1)[flat[remove]] = -1
        amount = len(sources)*self.degree*self.degree
        self.work["reciprocal_slots"] += amount
        self.work["maximum_reciprocal_slots"] = max(self.work["maximum_reciprocal_slots"], amount)

    @torch.no_grad()
    def register_copies(self, children, parents):
        self._rows(children); self._rows(parents)
        if children.shape != parents.shape or len(torch.unique(children)) != len(children):
            raise ValueError("lineage copies require distinct paired child rows")
        if not len(children):
            return
        self.work["updates"] += 1
        self.work["copied_rows"] += len(children)
        self._remove_reciprocals(children,self.neighbors[children].clone())
        self.neighbors[children] = -1
        # A parent overwritten in this same simultaneous batch refers to its
        # previous incarnation, so no edge to its current row is registered.
        ordered_children = children.sort().values
        positions = torch.searchsorted(ordered_children,parents)
        keep = (positions == len(children)) | (ordered_children[positions.clamp_max(max(0,len(children)-1))] != parents)
        children, parents = children[keep], parents[keep]
        if not len(children):
            return
        # Group repeated manual-copy parents in one batch. Later entries in a
        # simultaneous copy batch are newest; keep at most degree children per
        # parent. Internal reaction plans already use distinct parents.
        order = parents.argsort(stable=True)
        unique, counts = torch.unique_consecutive(parents[order],return_counts=True)
        ends = counts.cumsum(0)
        slots = torch.arange(self.degree,device=parents.device)[None]
        locations = ends[:,None]-1-slots
        present = locations >= (ends-counts)[:,None]
        incoming = children[order[locations.clamp_min(0)]].masked_fill(~present,-1)
        room = self.degree-counts.clamp_max(self.degree)
        old = self._compact(unique)
        evicted = old.masked_fill(slots < room[:,None],-1)
        # Truncate both endpoints before re-reading updated rows. This prevents
        # two touched parents from reintroducing an edge one of them evicted.
        self.neighbors[unique] = old.masked_fill(slots >= room[:,None],-1)
        self._remove_reciprocals(unique,evicted)
        old = self._compact(unique)
        used = counts.clamp_max(self.degree)
        old_slots = (slots-used[:,None]).clamp_min(0)
        self.neighbors[unique] = torch.where(slots < used[:,None],incoming,old.gather(1,old_slots))
        selected = incoming >= 0
        selected_children = incoming[selected]
        selected_parents = unique[:,None].expand_as(incoming)[selected]
        self.neighbors[selected_children,0] = selected_parents
        self.work["maximum_parent_rows"] = max(self.work["maximum_parent_rows"],len(unique))


class BoundedLatentGeometry:
    """Prior-derived anisotropic width and bounded local DV12 latent radius.

    At most ``neighbors`` candidates come from coordinate sort neighborhoods
    on the ``rank`` most varying latent coordinates. Distances use the full
    latent vector. The radius can exceed the exact nearest-other radius, so
    this is an approximate DV12 kernel, not an exact nearest-neighbor claim.
    Setup is O(rank*N*log(N)); queries use O(neighbors*latent_width) per row.
    Deterministic caches are derived, versioned and never checkpointed.
    Coordinate width is bounded by the local RMS of the nearest ``rank``
    candidates divided by sqrt(2), the pair-difference variance factor. This
    prevents an isotropic norm ball from ignoring narrow local coordinates.
    """
    def __init__(self, *, rank=8, neighbors=64, chunk=256, lineage=None):
        for name, value in (("rank", rank), ("neighbors", neighbors), ("chunk", chunk)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.rank, self.neighbors, self.chunk = rank, neighbors, chunk
        self.lineage = lineage
        self.clear()

    def clear(self):
        self._entries = {}
        self.work = dict(builds=0, distance_pairs=0, max_query_rows=0, max_candidates=0,
                         lineage_distance_pairs=0)

    @torch.no_grad()
    def _orders(self, points):
        key = id(points)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is points and entry[1] == points._version:
            return entry[2]
        spread = points.detach().std(0, unbiased=False)
        count = min(self.rank, int((spread > 0).sum()))
        # Cache Python indices after one batched transfer. A CUDA scalar axis
        # would otherwise synchronize every block[:, axis] lookup per chunk.
        axes = spread.argsort(descending=True, stable=True)[:count].cpu().tolist()
        orders = []
        for axis in axes:
            values, rows = points.detach()[:, axis].sort(stable=True)
            unique, counts = torch.unique_consecutive(values, return_counts=True)
            # Skip entire tied-coordinate groups. This avoids losing every
            # candidate when many row copies have exactly identical centers.
            first = counts.cumsum(0)-counts
            orders.append((axis, unique, rows[first]))
        if len(self._entries) >= 2:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = (weakref.ref(points), points._version, orders)
        self.work["builds"] += 1
        return orders

    @torch.no_grad()
    def radius(self, latent, prior, *, rows=None):
        return self._local_geometry(latent, prior, rows=rows)[0]

    @torch.no_grad()
    def _local_geometry(self, latent, prior, *, rows=None):
        points = prior.z
        if self.lineage is not None and rows is not None:
            self.lineage._rows(rows)
            if len(rows) != len(latent) or len(points) != self.lineage.n:
                raise ValueError("latent rows and lineage table must match")
        orders = self._orders(points)
        if not orders:
            return latent.new_zeros(len(latent)), torch.zeros_like(latent)
        output, widths = [], []
        for start in range(0, len(latent), self.chunk):
            block = latent.detach()[start:start+self.chunk]
            distances, point_ids = [], []
            for index, (axis, values, axis_rows) in enumerate(orders):
                count = self.neighbors//len(orders)+(index < self.neighbors % len(orders))
                if not count:
                    continue
                offsets = torch.arange(-(count//2), count-count//2, device=block.device)
                positions = torch.searchsorted(values, block[:, axis].contiguous())[:, None]+offsets
                valid = (positions >= 0) & (positions < len(values))
                candidates = axis_rows[positions.clamp(0, len(values)-1)]
                distance = (block[:, None]-points.detach()[candidates]).square().sum(-1)
                distance.masked_fill_(~valid | (distance == 0), float("inf"))
                distances.append(distance); point_ids.append(candidates)
                self.work["distance_pairs"] += len(block)*count
            linked = 0
            if self.lineage is not None and rows is not None:
                candidates = self.lineage.neighbors[rows[start:start+len(block)]]
                distance = (block[:, None]-points.detach()[candidates.clamp_min(0)]).square().sum(-1)
                distance.masked_fill_((candidates < 0) | (distance == 0), float("inf"))
                distances.append(distance); point_ids.append(candidates.clamp_min(0))
                linked = self.lineage.degree
                self.work["distance_pairs"] += len(block)*linked
                self.work["lineage_distance_pairs"] += len(block)*linked
            self.work["max_query_rows"] = max(self.work["max_query_rows"], len(block))
            self.work["max_candidates"] = max(self.work["max_candidates"], self.neighbors+linked)
            distances, point_ids = torch.cat(distances, 1), torch.cat(point_ids, 1)
            nearest = distances.min(1).values.sqrt()
            output.append(torch.where(torch.isfinite(nearest), .5*nearest, torch.zeros_like(nearest)))
            selected = distances.argsort(dim=1, stable=True)[:, :min(self.rank, distances.shape[1])]
            valid = torch.isfinite(distances.gather(1, selected))
            neighbors = points.detach()[point_ids.gather(1, selected)]
            difference2 = (block[:, None]-neighbors).square()*valid[:, :, None]
            # A difference between independent local draws has twice their
            # coordinate variance. Reuse rank as the bounded local sample size.
            widths.append((difference2.sum(1)/(2*valid.sum(1).clamp_min(1)[:, None])).sqrt())
        return torch.cat(output), torch.cat(widths)

    def displacement(self, latent, prior, bandwidth, noise, *, record=None, rows=None):
        radius, local_width = self._local_geometry(latent, prior, rows=rows)
        global_width = torch.as_tensor(bandwidth, device=latent.device, dtype=latent.dtype)
        applied_width = torch.minimum(global_width, local_width)
        delta = applied_width*noise
        fraction = (radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
        delta = delta*fraction[:, None]
        if record is not None:
            record.append(dict(radius_min=float(radius.min()), radius_mean=float(radius.mean()),
                               radius_max=float(radius.max()),
                               perturbation_rms=float(delta.detach().square().mean().sqrt()),
                               clipped_fraction=float((fraction < 1.).float().mean()),
                               global_bandwidth_mean=float(global_width.mean()),
                               local_bandwidth_mean=float(applied_width.mean()),
                               backend="feature_cells", kernel="bounded_local_dv12_lineage" if self.lineage else "bounded_local_dv12"))
            del record[:-2]
        return delta


def bounded_jitter(latent, noise):
    """Historical fixed kernel, retained for explicit comparison diagnostics."""
    delta = JITTER_STD * noise
    fraction = (JITTER_CAP / delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
    return delta * fraction[:, None]


def _features_ok(x, *, minimum_rows=1):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or len(x) < minimum_rows
            or x.shape[1] < 1 or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
        raise ValueError("feature cells require finite floating [rows, learned_features] tensors")


def _integer_allocate(capacity, total):
    """Largest remainders, always <=integer capacity, with stable cell ties."""
    total = min(int(total), int(capacity.sum()))
    if total <= 0:
        return torch.zeros_like(capacity)
    quota = capacity.double() * (total / int(capacity.sum()))
    allocated = quota.floor().long()
    remaining = total - int(allocated.sum())
    if remaining:
        order = (quota - allocated).argsort(descending=True, stable=True)
        allocated[order[:remaining]] += 1
    return allocated


def conditional_count_pvalues(real_counts, fake_counts, real_rows, fake_rows, work=None):
    """Two-sided exact conditional Hypergeom tests, in float64 log arithmetic.

    Validity is conditional on a fixed partition and independent categorical
    draws. Repeated adaptive FIFO/model snapshots have no cumulative guarantee.
    Enumeration is <=K*(min(real_rows,fake_rows)+1), never reference-by-reference.
    """
    if (not isinstance(real_counts, torch.Tensor) or not isinstance(fake_counts, torch.Tensor)
            or real_counts.shape != fake_counts.shape or real_counts.ndim != 1
            or real_counts.dtype != torch.long or fake_counts.dtype != torch.long
            or real_counts.device != fake_counts.device or real_rows <= 0 or fake_rows <= 0):
        raise ValueError("invalid categorical counts")
    invalid = torch.stack(((real_counts < 0).any(), (fake_counts < 0).any(),
                           real_counts.sum() != real_rows, fake_counts.sum() != fake_rows))
    if bool(invalid.any()):
        raise ValueError("invalid categorical counts")
    device = real_counts.device
    factorial = torch.lgamma(torch.arange(real_rows + fake_rows + 1, device=device, dtype=torch.float64) + 1)
    total = real_counts + fake_counts
    low = (total - fake_rows).clamp_min(0)
    high = total.clamp_max(real_rows)
    lengths = high - low + 1
    # One bulk metadata transfer replaces the per-cell CUDA scalar reads.
    max_length, terms = torch.stack((lengths.max(), lengths.sum())).cpu().tolist()
    offsets = torch.arange(max_length, device=device, dtype=torch.long)[None, :]
    valid = offsets < lengths[:, None]
    # Clamp padding to a valid support endpoint before factorial indexing.
    j = torch.minimum(low[:, None] + offsets, high[:, None])
    log_probability = (factorial[real_rows] - factorial[j] - factorial[real_rows-j]
                       + factorial[fake_rows] - factorial[total[:, None]-j] - factorial[fake_rows-total[:, None]+j]
                       - factorial[real_rows+fake_rows] + factorial[total, None] + factorial[real_rows+fake_rows-total, None])
    observed_log_probability = log_probability.gather(1, (real_counts-low)[:, None])
    # Same two-sided conditional test and mathematical equality tolerance.
    selected = valid & (log_probability <= observed_log_probability + 1e-10)
    result = torch.logsumexp(log_probability.masked_fill(~selected, -float("inf")), 1).exp().clamp_max(1.)
    if work is not None:
        work["count_test_terms"] += terms
    return result


def _group_integer_allocate(capacity, groups, totals):
    """Same stable largest remainders for every disjoint group, in one batch."""
    members = groups[None] == torch.arange(len(totals),device=groups.device)[:,None]
    capacities = capacity[None]*members
    sums = capacities.sum(1)
    requested = torch.minimum(totals,sums).clamp_min(0)
    quota = capacities.double()*(requested.double()/sums.clamp_min(1).double())[:,None]
    allocated = quota.floor().long()
    remaining = requested-allocated.sum(1)
    order = (quota-allocated).argsort(dim=1,descending=True,stable=True)
    increments = (torch.arange(len(capacity),device=capacity.device)[None] < remaining[:,None]).long()
    allocated += torch.zeros_like(allocated).scatter(1,order,increments)
    return allocated.sum(0)


class FeatureCellSnapshot:
    """One reference-only fitted critic snapshot; queries are original features.

    Derived caches must be invalidated after copies and rebuilt for new heads.
    The controller rebuilds every FIFO turnover and never uses the old snapshot
    for decisions between evaluations. Current-device Torch only.
    """
    @classmethod
    @torch.no_grad()
    def fit(cls, real_features, *, generator, cells=64, rank=8, chunk=256):
        _features_ok(real_features, minimum_rows=6)
        for name, value in (("cells", cells), ("rank", rank), ("chunk", chunk)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(generator, torch.Generator) or torch.device(generator.device) != real_features.device:
            raise ValueError("feature cells need a generator on the feature device")
        self = cls()
        self.device, self.width = real_features.device, real_features.shape[1]
        self.chunk, self.requested_cells, self.requested_rank = chunk, cells, rank
        self.work = dict(distance_cells=0, projection_products=0, count_test_terms=0,
                         parent_candidate_cells=0, max_distance_rows=0, max_distance_columns=0,
                         retained_array_bytes=0, rebuilds=1)
        real = real_features.double()
        ref = real[0::2]
        self.mean = ref.mean(0)
        spread = ref.std(0, unbiased=True)
        varying = spread > spread.max() * 1e-8
        self.scale = torch.where(varying, spread, torch.full_like(spread, float("inf")))
        self.valid_metric = bool(varying.any())
        self.rank = min(rank, int(varying.sum()), len(ref)-1) if self.valid_metric else 0
        centered = (ref - self.mean) / self.scale
        if self.rank:
            basis, _ = torch.linalg.qr(torch.randn((self.width, self.rank), device=self.device,
                                                    dtype=torch.float64, generator=generator), mode="reduced")
            for _ in range(POWER_PASSES):
                basis, _ = torch.linalg.qr(centered.T @ (centered @ basis), mode="reduced")
                self.work["projection_products"] += 2 * len(ref) * self.width * self.rank
            self.basis = basis
        else:
            self.basis = real.new_zeros((self.width, 0))
        x = self.transform(ref)
        self.cells = min(cells, len(x))
        centers = x.new_empty((self.cells, self.rank))
        selected = (x-x.mean(0)).square().sum(1).argmin()
        nearest = x.new_full((len(x),), float("inf"))
        for cell in range(self.cells):
            centers[cell] = x.index_select(0, selected.reshape(1)).squeeze(0)
            dist2 = (x-centers[cell]).square().sum(1)
            self.work["distance_cells"] += len(x)
            nearest = torch.minimum(nearest, dist2)
            selected = nearest.argmax()
        self.centers = centers
        for _ in range(LLOYD_PASSES):
            ids, _ = self._assign_metric(x)
            # Stable integer grouping keeps the original within-cell row order.
            # One metadata transfer replaces K variable-size boolean gathers;
            # the same per-cell reductions avoid floating scatter atomics.
            order = ids.argsort(stable=True)
            grouped = x.index_select(0, order)
            ends = torch.bincount(ids, minlength=self.cells).cumsum(0).cpu().tolist()
            start = 0
            for cell, end in enumerate(ends):
                if end > start:
                    self.centers[cell] = grouped[start:end].mean(0)
                start = end
        ids, dist2 = self._assign_metric(x)
        self.reference_counts = torch.bincount(ids, minlength=self.cells)
        order = ids.argsort(stable=True)
        ends = self.reference_counts.cumsum(0).cpu().tolist()
        bounds = list(zip([0]+ends[:-1], ends))
        sse = torch.stack([dist2.index_select(0, order[start:end]).sum() for start, end in bounds])
        radius = (sse / self.reference_counts.clamp_min(1)).sqrt()
        positive = radius[(self.reference_counts > 0) & (radius > 0)]
        global_radius = float(torch.quantile(positive, .5)) if len(positive) else 1.
        self.cell_scale = ((sse + SCALE_PRIOR * global_radius**2) /
                           (self.reference_counts + SCALE_PRIOR)).sqrt().clamp_min(global_radius*1e-3)
        self.real_representatives = x.new_empty((self.cells, self.rank))
        self.real_representative_rows = torch.empty(self.cells, device=self.device, dtype=torch.long)
        for cell, (start, end) in enumerate(bounds):
            members = order[start:end]
            if len(members):
                chosen = members.index_select(0, dist2.index_select(0, members).argmin().reshape(1)).squeeze(0)
            else:
                chosen = (x-self.centers[cell]).square().sum(1).argmin()
                self.work["distance_cells"] += len(x)
            self.real_representatives[cell] = x.index_select(0, chosen.reshape(1)).squeeze(0)
            self.real_representative_rows[cell] = 2*chosen
        # Freeze this geometric count partition before consuming odd rows.
        # The existing odd-calibrated support null remains a separate law.
        self._fit_count_partition_metric(x, ids)
        calibration = self.transform(real[1::2])
        calibration_ids, _ = self._assign_metric(calibration)
        self.real_calibration_counts = torch.bincount(calibration_ids, minlength=self.cells)
        self.calibration_rows = len(calibration)
        calibration_categories, calibration_scores = self._count_categories_metric(calibration, calibration_ids)
        self.real_calibration_category_counts = torch.bincount(calibration_categories, minlength=2*self.cells)
        self.null_scores = calibration_scores.sort().values
        self.duplicate_fraction = 1. - len(torch.unique(real, dim=0)) / len(real)
        self.row_features = self.query_cell_ids = self.query_counts = None
        self.cache_version = 0
        self._update_storage()
        return self

    def _update_storage(self):
        tensors = [v for v in vars(self).values() if isinstance(v, torch.Tensor)]
        sizes = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in tensors}
        self.work["retained_array_bytes"] = sum(sizes.values())

    def _input(self, features):
        _features_ok(features)
        if features.device != self.device or features.shape[1] != self.width:
            raise ValueError("features do not match this frozen head snapshot")

    @torch.no_grad()
    def transform(self, features):
        self._input(features)
        self.work["projection_products"] += len(features) * self.width * self.rank
        return ((features.double()-self.mean)/self.scale) @ self.basis

    def _distance(self, query, points):
        self.work["distance_cells"] += len(query)*len(points)
        self.work["max_distance_rows"] = max(self.work["max_distance_rows"], len(query))
        self.work["max_distance_columns"] = max(self.work["max_distance_columns"], len(points))
        return (query.square().sum(1)[:,None]+points.square().sum(1)[None]-2*query@points.T).clamp_min(0.)

    def _assign_metric(self, features):
        ids, dist = [], []
        for block in features.split(self.chunk):
            d, i = self._distance(block,self.centers).min(1)
            ids.append(i)
            dist.append(d)
        return torch.cat(ids), torch.cat(dist)

    def _scores_metric(self, features):
        scores = []
        for block in features.split(self.chunk):
            scores.append((self._distance(block,self.centers)/self.cell_scale.square()[None]).min(1).values.sqrt())
        return torch.cat(scores)

    def _fit_count_partition_metric(self, fitted_features, fitted_ids):
        """Fixed global even-fit score boundary, retaining all 2K categories.

        This empirical fitted boundary is not a conformal p-value threshold.
        Its only role is to define a partition before independent odd counts.
        """
        if hasattr(self, "count_boundary"):
            raise ValueError("count partition is already frozen")
        scores = self._scores_metric(fitted_features)
        q = Fraction(str(Q))
        ordinal = (len(scores)*(q.denominator-q.numerator)+q.denominator-1)//q.denominator
        self.count_boundary = scores.sort().values[max(0, ordinal-1)].clone()
        self.count_partition = dict(rule="even_fit_score_order_statistic", q=Q,
                                    fitted_rows=len(scores), ordinal=ordinal,
                                    ties="inside", categories=2*self.cells)
        categories = 2*fitted_ids+(scores>self.count_boundary).long()
        self.reference_category_counts = torch.bincount(categories, minlength=2*self.cells)

    def _count_categories_metric(self, features, ids=None):
        if not hasattr(self, "count_boundary"):
            raise ValueError("fit count partition before assigning count categories")
        if ids is None:
            ids, _ = self._assign_metric(features)
        scores = self._scores_metric(features)
        return 2*ids+(scores>self.count_boundary).long(), scores

    @torch.no_grad()
    def count_categories(self, features):
        categories, _ = self._count_categories_metric(self.transform(features))
        return categories

    @torch.no_grad()
    def assign(self, features):
        return self._assign_metric(self.transform(features))

    @torch.no_grad()
    def support(self, query_features):
        scores = self._scores_metric(self.transform(query_features))
        p = (1. + len(self.null_scores)-torch.searchsorted(self.null_scores,scores)) / (1.+len(self.null_scores))
        ordered = p.sort().values
        passed = (ordered <= Q*torch.arange(1,len(p)+1,device=self.device,dtype=p.dtype)/len(p)).nonzero().flatten()
        flags = p <= ordered[passed[-1]] if len(passed) else torch.zeros(len(p),device=self.device,dtype=torch.bool)
        if self.duplicate_fraction > Q or not self.valid_metric:
            flags.zero_()
        return flags,p,scores

    @torch.no_grad()
    def cache_queries(self, query_features):
        self.row_features = self.transform(query_features)
        self.query_cell_ids, _ = self._assign_metric(self.row_features)
        self.query_counts = torch.bincount(self.query_cell_ids,minlength=self.cells)
        self._update_storage()
        return self.query_cell_ids

    @torch.no_grad()
    def refresh_rows(self, rows, repaired_features):
        if self.query_cell_ids is None:
            raise ValueError("cache_queries must precede row refresh")
        if rows.ndim != 1 or rows.dtype != torch.long or rows.device != self.device or len(rows) != len(repaired_features):
            raise ValueError("invalid repaired rows")
        if len(torch.unique(rows)) != len(rows) or bool(((rows<0)|(rows>=len(self.query_cell_ids))).any()):
            raise ValueError("invalid repaired row indices")
        moved = self.transform(repaired_features)
        after_ids,_ = self._assign_metric(moved)
        affected = torch.zeros(self.cells,device=self.device,dtype=torch.bool)
        affected[self.query_cell_ids[rows]] = True
        affected[after_ids] = True
        invalidated = affected[self.query_cell_ids]
        invalidated[rows] = True
        self.row_features[rows] = moved
        self.query_cell_ids[rows] = after_ids
        self.query_counts = torch.bincount(self.query_cell_ids,minlength=self.cells)
        self.cache_version += 1
        self._update_storage()
        return invalidated,affected

    def _pool(self, features, eligible, generator):
        ids,_ = self._assign_metric(features)
        priorities = torch.rand(len(features),device=self.device,dtype=torch.float64,generator=generator)
        pool = torch.full((self.cells,PARENT_RESERVOIR),-1,device=self.device,dtype=torch.long)
        counts = torch.zeros(self.cells,device=self.device,dtype=torch.long)
        groups = torch.where(eligible, ids, self.cells)
        eligible_counts = torch.bincount(groups, minlength=self.cells+1)[:self.cells]
        counts = eligible_counts.clamp_max(PARENT_RESERVOIR)
        # Stable priority sort, then stable cell sort: exact original priority
        # and row-index tie order, with the same single RNG draw vector.
        priority_order = priorities.argsort(stable=True)
        order = priority_order[groups[priority_order].argsort(stable=True)]
        starts = eligible_counts.cumsum(0) - eligible_counts
        slots = torch.arange(PARENT_RESERVOIR, device=self.device)[None, :]
        positions = starts[:, None] + slots
        selected = order[positions.clamp_max(len(features)-1)]
        pool = torch.where(slots < counts[:, None], selected, pool)
        return ids,pool,counts,eligible_counts

    def _mass_targets(self, rows):
        """Full reference mass for action quotas; tests still use heldout rows."""
        counts = self.reference_counts + self.real_calibration_counts
        quota = counts.double() * (rows / int(counts.sum()))
        target = quota.floor().long()
        remaining = rows - int(target.sum())
        if remaining:
            order = (quota-target).argsort(descending=True,stable=True)
            target[order[:remaining]] += 1
        return target


    def _mass_topology(self):
        """Real-only coarse support, using only the bounded K centre graph.

        A fine partition responds to within-support width as well as mass.
        Conserve mass within connected centre neighborhoods before allowing
        transfers between them. The deterministic minimum spanning tree is
        cut at the maximum between-class variance split of log edge lengths;
        neither labels nor query/fake rows choose this action partition.
        This derived topology is not used by either original statistical test.
        """
        if hasattr(self,"mass_group_ids"):
            return self.mass_group_ids
        k = self.cells
        device = self.device
        group_ids = torch.arange(k,device=device,dtype=torch.long)
        threshold = None
        if k > 2:
            distance = self._distance(self.centers,self.centers)
            distance.fill_diagonal_(float("inf"))
            used = torch.zeros(k,device=device,dtype=torch.bool); used[0] = True
            nearest = distance[0].clone()
            parent = torch.zeros(k,device=device,dtype=torch.long)
            edge_parent,edge_child,edge_length = [],[],[]
            for _ in range(k-1):
                child = nearest.masked_fill(used,float("inf")).argmin()
                edge_parent.append(parent[child].clone()); edge_child.append(child.clone())
                edge_length.append(nearest[child].clone())
                used[child] = True
                shorter = distance[child] < nearest
                nearest = torch.minimum(nearest,distance[child])
                parent[shorter] = child
            edge_parent,edge_child,edge_length = map(torch.stack,(edge_parent,edge_child,edge_length))
            logs = edge_length.clamp_min(torch.finfo(edge_length.dtype).tiny).log().sort().values
            cumulative = logs.cumsum(0)
            cut = torch.arange(1,len(logs),device=device,dtype=logs.dtype)
            left = cumulative[:-1]/cut
            right = (cumulative[-1]-cumulative[:-1])/(len(logs)-cut)
            separation = cut*(len(logs)-cut)*(left-right).square()
            split = separation.argmax()
            threshold_tensor = ((logs[split]+logs[split+1])*.5).exp()
            threshold = float(threshold_tensor)
            connected = torch.eye(k,device=device,dtype=torch.bool)
            retained = edge_length <= threshold_tensor
            a,b = edge_parent[retained],edge_child[retained]
            connected[a,b] = True; connected[b,a] = True
            for _ in range(k):
                updated = group_ids[None].expand(k,-1).masked_fill(~connected,k).min(1).values
                if torch.equal(updated,group_ids):
                    break
                group_ids = updated
            group_ids = torch.unique(group_ids,sorted=True,return_inverse=True)[1]
        self.mass_group_ids = group_ids
        self.mass_groups = int(group_ids.max())+1
        self.mass_topology = dict(rule="reference_center_mst_log_between_class_variance",
                                  groups=self.mass_groups,threshold_squared=threshold)
        self._update_storage()
        return group_ids


    def _group_counts(self, counts):
        groups = self._mass_topology()
        return torch.stack([counts[groups==group].sum() for group in range(self.mass_groups)])


    @torch.no_grad()
    def select_parents(self, query_features, flags, *, ordinary_children=None, ordinary_parents=None,
                       generator, pvalues=None):
        """Vacancy-limited anchor repair with distinct supported parents.

        Clean supported rows, including planned ordinary moves, reserve their
        reference mass first. Each target cell supplies only its own bounded
        parent ball. A parent can seed at most one copy in the whole reaction;
        unfilled holes wait for later snapshots rather than borrowing full-cell
        mass or multiplying one surviving rare row.
        """
        features = self.transform(query_features)
        if flags.shape != (len(features),) or flags.dtype != torch.bool or flags.device != self.device:
            raise ValueError("invalid isolation flags")
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        dead = flags.nonzero().flatten()
        detail = dict(candidate_ids=empty.reshape(0,1),candidate_mask=torch.empty((0,1),device=self.device,dtype=torch.bool),
                      anchor_reference_rows=empty,anchor_cell_ids=empty,parent_cell_ids=empty,
                      inaccessible_deficit=0.,accessible_cells=0,guard_passed=0<len(dead)<=Q*len(features))
        if not detail["guard_passed"]:
            return empty,empty,detail
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        eligible = ~flags & (pvalues>Q)
        if ordinary_children is not None:
            eligible[ordinary_children] = False
            allowed = torch.ones(len(features),device=self.device,dtype=torch.bool)
            allowed[ordinary_children] = False
            dead = dead[allowed[dead]]
        if ordinary_parents is not None:
            if ordinary_children is None or len(ordinary_parents) != len(ordinary_children):
                raise ValueError("ordinary children/parents must be paired")
            eligible[ordinary_parents] = False
        ids,pool,counts,eligible_counts = self._pool(features,eligible,generator)
        target = self._mass_targets(len(features))
        groups = self._mass_topology()
        kept = torch.bincount(ids[~flags],minlength=self.cells)
        if ordinary_parents is not None:
            supported_children = ordinary_children[~flags[ordinary_children]]
            kept -= torch.bincount(ids[supported_children],minlength=self.cells)
            kept += torch.bincount(ids[ordinary_parents],minlength=self.cells)
        vacancies = (self._group_counts(target)-self._group_counts(kept)).clamp_min(0)
        # Partition membership and the unchanged support p>Q criterion
        # define this own-cell pool. The old cross-cell 2*nearest ball could
        # shrink 64 supported rows to one and repeatedly clone that family;
        # it is redundant once parent and vacancy must share the same cell.
        balls = pool >= 0
        self.work["parent_candidate_cells"] += pool.numel()
        supply = balls.sum(1)
        capacity = supply*(vacancies[groups]>0)
        accessible = capacity > 0
        detail["accessible_cells"] = int(accessible.sum())
        group_supply = self._group_counts(supply)
        detail.update(inaccessible_deficit=float(vacancies[group_supply==0].sum()),target_counts=target,
                      kept_counts=kept,group_vacancies=vacancies,eligible_parent_counts=eligible_counts,
                      unique_parent_capacity=capacity,requested_repairs=len(dead),
                      candidate_policy="supported_same_cell_without_replacement",mass_topology=dict(self.mass_topology))
        if not bool(accessible.any()) or not len(dead):
            return empty,empty,detail
        children,parents,candidates,masks,anchor_rows,anchor_cells = [],[],[],[],[],[]
        distances = torch.cat([self._distance(features[block],self.real_representatives)
                               for block in dead.split(self.chunk)])
        available = torch.ones(len(dead),device=self.device,dtype=torch.bool)
        remaining_capacity = capacity.clone()
        remaining_vacancies = vacancies.clone()
        assignments = torch.full((len(dead),),-1,device=self.device,dtype=torch.long)
        # Child-first proposals preserve learned-feature locality. Each cell
        # accepts its nearest proposals up to its vacancy and parent supply;
        # rejected children try the next nonfull cell. A contested round fills
        # at least one cell, so there are at most K rounds (no table search).
        for _ in range(self.cells):
            accessible = (remaining_capacity>0)&(remaining_vacancies[groups]>0)
            if not bool(available.any()) or not bool(accessible.any()):
                break
            current = distances.masked_fill(~accessible[None],float("inf"))
            nearest_distance,nearest = current.min(1)
            nearest_group = groups[nearest]
            alternative = current.masked_fill(groups[None]==nearest_group[:,None],float("inf")).min(1).values
            regret = alternative-nearest_distance
            for group in range(self.mass_groups):
                rows = (available&(nearest_group==group)).nonzero().flatten()
                if not len(rows) or not int(remaining_vacancies[group]):
                    continue
                # Preserve children with the worst alternative when a local
                # mass quota is contested; absolute closeness alone can use
                # that quota on flexible children and force distant repairs.
                rows = rows[nearest_distance[rows].argsort(stable=True)]
                rows = rows[regret[rows].argsort(descending=True,stable=True)]
                proposals = rows[:int(remaining_vacancies[group])]
                for cell_tensor in (groups==group).nonzero().flatten():
                    cell = int(cell_tensor)
                    local = proposals[nearest[proposals]==cell]
                    take = min(len(local),int(remaining_capacity[cell]))
                    if take:
                        chosen = local[:take]
                        assignments[chosen] = cell
                        available[chosen] = False
                        remaining_capacity[cell] -= take
                        remaining_vacancies[group] -= take
        quota = capacity-remaining_capacity
        detail["birth_allocation"] = quota
        detail["unfilled_repairs"] = int(available.sum())
        detail["assignment_policy"] = "nearest_reference_group_regret_then_capacity"
        for cell in range(self.cells):
            selected_child = (assignments==cell).nonzero().flatten()
            count = len(selected_child)
            if not count:
                continue
            slots = balls[cell].nonzero().flatten()
            priority = torch.rand(len(slots),device=self.device,dtype=torch.float64,generator=generator)
            selected_slot = slots[priority.argsort(stable=True)[:count]]
            candidate = pool[cell].expand(count,-1).clone()
            mask = balls[cell].expand(count,-1).clone()
            previous = (torch.arange(count,device=self.device)[:,None]
                        >torch.arange(count,device=self.device)[None]).nonzero()
            if len(previous):
                mask[previous[:,0],selected_slot[previous[:,1]]] = False
            children.append(dead[selected_child])
            parents.append(pool[cell,selected_slot])
            candidates.append(candidate)
            masks.append(mask)
            anchor_rows.append(self.real_representative_rows[cell].expand(count))
            anchor_cells.append(torch.full((count,),cell,device=self.device,dtype=torch.long))
        child,parent = torch.cat(children),torch.cat(parents)
        detail.update(candidate_ids=torch.cat(candidates),candidate_mask=torch.cat(masks),
                      anchor_reference_rows=torch.cat(anchor_rows),anchor_cell_ids=torch.cat(anchor_cells),
                      parent_cell_ids=ids[parent],parent_cell_deficits=vacancies[groups[ids[parent]]],
                      remaining_group_vacancies=remaining_vacancies)
        return child,parent,detail

    @torch.no_grad()
    def cell_comparison(self, fake_features):
        fake_categories = self.count_categories(fake_features)
        fake_counts = torch.bincount(fake_categories,minlength=2*self.cells)
        real_counts = self.real_calibration_category_counts
        p = conditional_count_pvalues(real_counts,fake_counts,self.calibration_rows,len(fake_features),self.work)
        difference = fake_counts.double()/len(fake_features)-real_counts.double()/self.calibration_rows
        multiplicity = 3*self.cells+2
        significant = p <= Q/multiplicity
        if not self.valid_metric:
            significant.zero_()
        support = dict(real_counts=real_counts,fake_counts=fake_counts,pvalues=p,
                    difference=difference,excess=significant&(difference>0),deficit=significant&(difference<0))
        # K original cells and 2K refinements are overlapping hypotheses,
        # not a single disjoint sample with twice the actual number of rows.
        mass_fake = fake_counts.reshape(self.cells,2).sum(1)
        mass_p = conditional_count_pvalues(self.real_calibration_counts,mass_fake,
                    self.calibration_rows,len(fake_features),self.work)
        mass_difference = mass_fake.double()/len(fake_features)-self.real_calibration_counts.double()/self.calibration_rows
        mass_significant = mass_p <= Q/multiplicity
        if not self.valid_metric:
            mass_significant.zero_()
        mass = dict(real_counts=self.real_calibration_counts,fake_counts=mass_fake,pvalues=mass_p,
                    difference=mass_difference,excess=mass_significant&(mass_difference>0),
                    deficit=mass_significant&(mass_difference<0))
        global_real = real_counts.reshape(self.cells,2).sum(0)
        global_fake = fake_counts.reshape(self.cells,2).sum(0)
        global_p = conditional_count_pvalues(global_real,global_fake,
                    self.calibration_rows,len(fake_features),self.work)
        global_difference = global_fake.double()/len(fake_features)-global_real.double()/self.calibration_rows
        global_significant = global_p <= Q/multiplicity
        if not self.valid_metric:
            global_significant.zero_()
        global_support = dict(real_counts=global_real,fake_counts=global_fake,pvalues=global_p,
                    difference=global_difference,excess=global_significant&(global_difference>0),
                    deficit=global_significant&(global_difference<0))
        return dict(mass=mass,support=support,global_support=global_support,pvalues=torch.cat((mass_p,p,global_p)),
                    categories=2*self.cells,multiplicity=multiplicity,cutoff=Q/multiplicity,
                    family_sizes=(self.cells,2*self.cells,2),
                    count_partition=dict(self.count_partition),count_boundary=self.count_boundary.clone())

    @torch.no_grad()
    def _ordinary_mass_transport(self, query_features, flags, comparison, *, generator, pvalues=None, max_moves=None):
        """One count-certified quota allocation; no inherited sequential evidence."""
        features = self.transform(query_features)
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        ids,pool,pool_counts,eligible_counts = self._pool(features,~flags&(pvalues>Q),generator)
        difference = comparison["difference"]
        death_capacity = torch.floor(len(features)*difference.clamp_min(0.)+1e-10).long()*comparison["excess"]
        birth_capacity = torch.floor(len(features)*(-difference).clamp_min(0.)+1e-10).long()*comparison["deficit"]
        target = self._mass_targets(len(features))
        table_counts = torch.bincount(ids,minlength=self.cells)
        budget = math.floor(Q*len(features)) if max_moves is None else min(int(max_moves),math.floor(Q*len(features)))
        isolation_guard_rejects = int(flags.sum()) > Q*len(features)
        # A small flagged set remains reserved for the unchanged isolation
        # guard. During widespread isolation, ordinary count evidence can
        # certify replacing a flagged row within its separate fixed budget.
        # Supported survivors always reserve their mass and never supply a
        # death in this recovery branch.
        clean_counts = torch.bincount(ids[~flags],minlength=self.cells)
        flagged_counts = table_counts-clean_counts
        excess = (clean_counts-target).clamp_min(0)
        vacancies = (target-clean_counts).clamp_min(0)
        # Emitted iid count evidence certifies direction. Actual clean rows
        # determine how much mass can move, avoiding repeated moves generated
        # solely by emission/pool fluctuation after the table is balanced.
        death_capacity = torch.minimum(death_capacity,flagged_counts if isolation_guard_rejects else excess)
        birth_capacity = torch.minimum(birth_capacity,vacancies)
        # Retain an actual eligible parent, rather than merely leaving an
        # arbitrary nonflagged row that might itself have p<=Q.
        protected = torch.full((self.cells,),-1,device=self.device,dtype=torch.long)
        retain = (target>0)&(pool_counts>0)
        protected[retain] = pool[retain,0]
        death_eligible = flags if isolation_guard_rejects else ~flags&(torch.arange(len(features),device=self.device)!=protected[ids])
        available_deaths = torch.bincount(ids[death_eligible],minlength=self.cells)
        death_capacity = torch.minimum(death_capacity,available_deaths)
        inaccessible = birth_capacity[pool_counts==0].sum()
        birth_capacity = torch.minimum(birth_capacity,pool_counts)
        groups = self._mass_topology()
        group_target,group_clean = self._group_counts(target),self._group_counts(clean_counts)
        within_capacity = torch.minimum(self._group_counts(death_capacity),self._group_counts(birth_capacity))
        if isolation_guard_rejects:
            within_capacity = torch.minimum(within_capacity,(group_target-group_clean).clamp_min(0))
        within = _integer_allocate(within_capacity,max(0,budget))
        within_deaths = _group_integer_allocate(death_capacity,groups,within)
        within_births = _group_integer_allocate(birth_capacity,groups,within)
        remaining_death,remaining_birth = death_capacity-within_deaths,birth_capacity-within_births
        if isolation_guard_rejects:
            group_death = self._group_counts(remaining_death)
            remaining_vacancies = (group_target-group_clean-self._group_counts(within_births)).clamp_min(0)
            group_birth = torch.minimum(self._group_counts(remaining_birth),remaining_vacancies)
        else:
            group_death = torch.minimum(self._group_counts(remaining_death),(group_clean-group_target).clamp_min(0))
            group_birth = torch.minimum(self._group_counts(remaining_birth),(group_target-group_clean).clamp_min(0))
        cross_pairs = min(int(group_death.sum()),int(group_birth.sum()),max(0,budget-int(within.sum())))
        cross_group_death,cross_group_birth = _integer_allocate(group_death,cross_pairs),_integer_allocate(group_birth,cross_pairs)
        cross_deaths = _group_integer_allocate(remaining_death,groups,cross_group_death)
        cross_births = _group_integer_allocate(remaining_birth,groups,cross_group_birth)
        deaths,births = within_deaths+cross_deaths,within_births+cross_births
        # One bounded metadata transfer preserves the original cell/RNG order.
        death_rows,birth_rows,within_death_rows,within_birth_rows,pool_rows,group_rows = torch.stack(
            (deaths,births,within_deaths,within_births,pool_counts,groups)).cpu().tolist()
        group_cells = [[] for _ in range(self.mass_groups)]
        for cell,group in enumerate(group_rows):
            group_cells[group].append(cell)
        priorities = torch.rand(len(features),device=self.device,dtype=torch.float64,generator=generator)
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        children_by_cell,parents_by_cell = [],[]
        for cell in range(self.cells):
            count = death_rows[cell]
            if count:
                rows = ((ids==cell)&death_eligible).nonzero().flatten()
                children_by_cell.append(rows[priorities[rows].argsort(stable=True)[:count]])
            else:
                children_by_cell.append(empty)
            count = birth_rows[cell]
            if count:
                draw = torch.randperm(pool_rows[cell],device=self.device,generator=generator)[:count]
                parents_by_cell.append(pool[cell,draw])
            else:
                parents_by_cell.append(empty)
        children,parents = [],[]
        # Match the conservative within-neighborhood phase inside its group.
        # Only the explicit residual phase can move mass between groups.
        for group in range(self.mass_groups):
            for index in group_cells[group]:
                children.append(children_by_cell[index][:within_death_rows[index]])
                parents.append(parents_by_cell[index][:within_birth_rows[index]])
        for cell in range(self.cells):
            children.append(children_by_cell[cell][within_death_rows[cell]:])
            parents.append(parents_by_cell[cell][within_birth_rows[cell]:])
        child = torch.cat(children) if children else empty
        parent = torch.cat(parents) if parents else empty
        detail = dict(excess_cells=int(comparison["excess"].sum()),deficit_cells=int(comparison["deficit"].sum()),
                      discoveries=int((comparison["excess"]|comparison["deficit"]).sum()),
                      inaccessible_birth_quota=int(inaccessible),budget=budget,moves=len(child),
                      death_allocation=deaths,birth_allocation=births,eligible_parent_counts=eligible_counts,
                      target_counts=target,clean_counts=clean_counts,table_counts=table_counts,clean_vacancies=vacancies,
                      unique_parent_count=len(torch.unique(parent)),mass_topology=dict(self.mass_topology),
                      group_target_counts=group_target,group_clean_counts=group_clean,
                      within_group_moves=int(within.sum()),between_group_moves=cross_pairs)
        supported_children = child[~flags[child]]
        detail.update(isolation_guard_rejects=isolation_guard_rejects,
                      ordinary_flagged_deaths=int(flags[child].sum()),
                      death_policy="count_certified_flagged_only" if isolation_guard_rejects else "supported_surplus",
                      planned_supported_counts=clean_counts-torch.bincount(ids[supported_children],minlength=self.cells)
                          +torch.bincount(ids[parent],minlength=self.cells))
        return child,parent,detail


    @torch.no_grad()
    def _ordinary_support_transport(self, query_features, flags, comparison, *, generator, pvalues=None, max_moves=None,
                                    reserved_children=None, reserved_parents=None, supported_counts=None):
        """Certified outside-to-inside recovery, with supported mass reserved."""
        features = self.transform(query_features)
        if flags.shape != (len(features),) or flags.dtype != torch.bool or flags.device != self.device:
            raise ValueError("invalid isolation flags")
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        categories, _ = self._count_categories_metric(features)
        inside = categories.remainder(2)==0
        reserved = torch.zeros(len(features),device=self.device,dtype=torch.bool)
        if reserved_children is not None:
            reserved[reserved_children] = True
        if reserved_parents is not None:
            reserved[reserved_parents] = True
        ids,pool,pool_counts,eligible_counts = self._pool(features,~flags&(pvalues>Q)&inside&~reserved,generator)
        difference = comparison["difference"].reshape(self.cells,2)
        outside_excess = comparison["excess"].reshape(self.cells,2)[:,1]
        inside_deficit = comparison["deficit"].reshape(self.cells,2)[:,0]
        death_capacity = torch.floor(len(features)*difference[:,1].clamp_min(0.)+1e-10).long()*outside_excess
        birth_capacity = torch.floor(len(features)*(-difference[:,0]).clamp_min(0.)+1e-10).long()*inside_deficit
        raw_death_capacity,raw_birth_capacity = death_capacity.clone(),birth_capacity.clone()
        spent_deaths = torch.bincount(categories[reserved_children],minlength=2*self.cells).reshape(self.cells,2)[:,1] if reserved_children is not None else torch.zeros_like(death_capacity)
        spent_births = torch.bincount(categories[reserved_parents],minlength=2*self.cells).reshape(self.cells,2)[:,0] if reserved_parents is not None else torch.zeros_like(birth_capacity)
        # Reserve gross certificate use by the first phase as well as row
        # identities. Opposite-direction moves do not replenish this budget.
        death_capacity = (death_capacity-spent_deaths).clamp_min(0)
        birth_capacity = (birth_capacity-spent_births).clamp_min(0)
        residual_death_capacity,residual_birth_capacity = death_capacity.clone(),birth_capacity.clone()
        target = self._mass_targets(len(features))
        table_counts = torch.bincount(ids,minlength=self.cells)
        budget = math.floor(Q*len(features)) if max_moves is None else min(int(max_moves),math.floor(Q*len(features)))
        isolation_guard_rejects = int(flags.sum()) > Q*len(features)
        # Only flagged outside rows can supply ordinary deaths. Even under a
        # passing isolation guard they do not reserve supported mass.
        clean_counts = torch.bincount(ids[~flags],minlength=self.cells) if supported_counts is None else supported_counts.clone()
        vacancies = (target-clean_counts).clamp_min(0)
        birth_capacity = torch.minimum(birth_capacity,vacancies)
        death_eligible = flags&~inside&~reserved
        available_deaths = torch.bincount(ids[death_eligible],minlength=self.cells)
        death_capacity = torch.minimum(death_capacity,available_deaths)
        inaccessible = birth_capacity[pool_counts==0].sum()
        birth_capacity = torch.minimum(birth_capacity,pool_counts)
        groups = self._mass_topology()
        group_target,group_clean = self._group_counts(target),self._group_counts(clean_counts)
        within_capacity = torch.minimum(self._group_counts(death_capacity),self._group_counts(birth_capacity))
        within_capacity = torch.minimum(within_capacity,(group_target-group_clean).clamp_min(0))
        within = _integer_allocate(within_capacity,max(0,budget))
        within_deaths = _group_integer_allocate(death_capacity,groups,within)
        within_births = _group_integer_allocate(birth_capacity,groups,within)
        remaining_death,remaining_birth = death_capacity-within_deaths,birth_capacity-within_births
        group_death = self._group_counts(remaining_death)
        remaining_vacancies = (group_target-group_clean-self._group_counts(within_births)).clamp_min(0)
        group_birth = torch.minimum(self._group_counts(remaining_birth),remaining_vacancies)
        cross_pairs = min(int(group_death.sum()),int(group_birth.sum()),max(0,budget-int(within.sum())))
        cross_group_death,cross_group_birth = _integer_allocate(group_death,cross_pairs),_integer_allocate(group_birth,cross_pairs)
        cross_deaths = _group_integer_allocate(remaining_death,groups,cross_group_death)
        cross_births = _group_integer_allocate(remaining_birth,groups,cross_group_birth)
        deaths,births = within_deaths+cross_deaths,within_births+cross_births
        # One bounded metadata transfer preserves the original cell/RNG order.
        death_rows,birth_rows,within_death_rows,within_birth_rows,pool_rows,group_rows = torch.stack(
            (deaths,births,within_deaths,within_births,pool_counts,groups)).cpu().tolist()
        group_cells = [[] for _ in range(self.mass_groups)]
        for cell,group in enumerate(group_rows):
            group_cells[group].append(cell)
        priorities = torch.rand(len(features),device=self.device,dtype=torch.float64,generator=generator)
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        children_by_cell,parents_by_cell = [],[]
        for cell in range(self.cells):
            count = death_rows[cell]
            if count:
                rows = ((ids==cell)&death_eligible).nonzero().flatten()
                children_by_cell.append(rows[priorities[rows].argsort(stable=True)[:count]])
            else:
                children_by_cell.append(empty)
            count = birth_rows[cell]
            if count:
                draw = torch.randperm(pool_rows[cell],device=self.device,generator=generator)[:count]
                parents_by_cell.append(pool[cell,draw])
            else:
                parents_by_cell.append(empty)
        children,parents = [],[]
        # Match the conservative within-neighborhood phase inside its group.
        # Only the explicit residual phase can move mass between groups.
        for group in range(self.mass_groups):
            for index in group_cells[group]:
                children.append(children_by_cell[index][:within_death_rows[index]])
                parents.append(parents_by_cell[index][:within_birth_rows[index]])
        for cell in range(self.cells):
            children.append(children_by_cell[cell][within_death_rows[cell]:])
            parents.append(parents_by_cell[cell][within_birth_rows[cell]:])
        child = torch.cat(children) if children else empty
        parent = torch.cat(parents) if parents else empty
        detail = dict(excess_cells=int(comparison["excess"].sum()),deficit_cells=int(comparison["deficit"].sum()),
                      discoveries=int((comparison["excess"]|comparison["deficit"]).sum()),
                      inaccessible_birth_quota=int(inaccessible),budget=budget,moves=len(child),
                      death_allocation=deaths,birth_allocation=births,eligible_parent_counts=eligible_counts,
                      target_counts=target,clean_counts=clean_counts,table_counts=table_counts,clean_vacancies=vacancies,
                      unique_parent_count=len(torch.unique(parent)),mass_topology=dict(self.mass_topology),
                      group_target_counts=group_target,group_clean_counts=group_clean,
                      within_group_moves=int(within.sum()),between_group_moves=cross_pairs)
        supported_children = child[~flags[child]]
        detail.update(isolation_guard_rejects=isolation_guard_rejects,
                      ordinary_flagged_deaths=int(flags[child].sum()),
                      death_policy="count_certified_flagged_outside_only",
                      count_partition=dict(self.count_partition),count_boundary=self.count_boundary.clone(),
                      query_category_ids=categories,child_category_ids=categories[child],parent_category_ids=categories[parent],
                      death_certified_categories=comparison["excess"],birth_certified_categories=comparison["deficit"],
                      death_category_allocation=torch.bincount(categories[child],minlength=2*self.cells),
                      birth_category_allocation=torch.bincount(categories[parent],minlength=2*self.cells),
                      raw_certified_death_capacity=raw_death_capacity,raw_certified_birth_capacity=raw_birth_capacity,
                      spent_certified_death_capacity=spent_deaths,spent_certified_birth_capacity=spent_births,
                      residual_certified_death_capacity=residual_death_capacity,
                      residual_certified_birth_capacity=residual_birth_capacity,
                      planned_supported_counts=clean_counts-torch.bincount(ids[supported_children],minlength=self.cells)
                          +torch.bincount(ids[parent],minlength=self.cells))
        return child,parent,detail


    @torch.no_grad()
    def _ordinary_global_transport(self, query_features, flags, comparison, *, generator, pvalues=None, max_moves=None,
                                    reserved_children=None, reserved_parents=None, supported_counts=None):
        """Aggregate support recovery after gross prior certificate reservations."""
        features = self.transform(query_features)
        if flags.shape != (len(features),) or flags.dtype != torch.bool or flags.device != self.device:
            raise ValueError("invalid isolation flags")
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        categories, _ = self._count_categories_metric(features)
        inside = categories.remainder(2)==0
        reserved = torch.zeros(len(features),device=self.device,dtype=torch.bool)
        if reserved_children is not None:
            reserved[reserved_children] = True
        if reserved_parents is not None:
            reserved[reserved_parents] = True
        ids,pool,pool_counts,eligible_counts = self._pool(features,~flags&(pvalues>Q)&inside&~reserved,generator)
        difference = comparison["difference"]
        raw_death_capacity = torch.floor(len(features)*difference[1].clamp_min(0.)+1e-10).long()*comparison["excess"][1]
        raw_birth_capacity = torch.floor(len(features)*(-difference[0]).clamp_min(0.)+1e-10).long()*comparison["deficit"][0]
        spent_deaths = (categories[reserved_children].remainder(2)==1).sum() if reserved_children is not None else torch.zeros_like(raw_death_capacity)
        spent_births = (categories[reserved_parents].remainder(2)==0).sum() if reserved_parents is not None else torch.zeros_like(raw_birth_capacity)
        # Both previous phases spend these aggregate certificates, including
        # supported mass children outside the fixed fitted boundary. Count
        # gross use; an opposite action never restores certificate capacity.
        residual_death_capacity = (raw_death_capacity-spent_deaths).clamp_min(0)
        residual_birth_capacity = (raw_birth_capacity-spent_births).clamp_min(0)
        target = self._mass_targets(len(features))
        table_counts = torch.bincount(ids,minlength=self.cells)
        budget = math.floor(Q*len(features)) if max_moves is None else min(int(max_moves),math.floor(Q*len(features)))
        isolation_guard_rejects = int(flags.sum()) > Q*len(features)
        certified_budget = max(0,min(budget,int(residual_death_capacity),int(residual_birth_capacity)))
        # Global evidence supplies the direction; actual cell/group vacancies
        # and unused inside parents supply physical birth capacity. Flagged
        # outside deaths do not remove any supported mass.
        clean_counts = torch.bincount(ids[~flags],minlength=self.cells) if supported_counts is None else supported_counts.clone()
        vacancies = (target-clean_counts).clamp_min(0)
        birth_capacity = vacancies.clone()
        death_eligible = flags&~inside&~reserved
        available_deaths = torch.bincount(ids[death_eligible],minlength=self.cells)
        death_capacity = available_deaths
        inaccessible = birth_capacity[pool_counts==0].sum()
        birth_capacity = torch.minimum(birth_capacity,pool_counts)
        groups = self._mass_topology()
        group_target,group_clean = self._group_counts(target),self._group_counts(clean_counts)
        within_capacity = torch.minimum(self._group_counts(death_capacity),self._group_counts(birth_capacity))
        within_capacity = torch.minimum(within_capacity,(group_target-group_clean).clamp_min(0))
        within = _integer_allocate(within_capacity,certified_budget)
        within_deaths = _group_integer_allocate(death_capacity,groups,within)
        within_births = _group_integer_allocate(birth_capacity,groups,within)
        remaining_death,remaining_birth = death_capacity-within_deaths,birth_capacity-within_births
        group_death = self._group_counts(remaining_death)
        remaining_vacancies = (group_target-group_clean-self._group_counts(within_births)).clamp_min(0)
        group_birth = torch.minimum(self._group_counts(remaining_birth),remaining_vacancies)
        cross_pairs = min(int(group_death.sum()),int(group_birth.sum()),max(0,certified_budget-int(within.sum())))
        cross_group_death,cross_group_birth = _integer_allocate(group_death,cross_pairs),_integer_allocate(group_birth,cross_pairs)
        cross_deaths = _group_integer_allocate(remaining_death,groups,cross_group_death)
        cross_births = _group_integer_allocate(remaining_birth,groups,cross_group_birth)
        deaths,births = within_deaths+cross_deaths,within_births+cross_births
        # One bounded metadata transfer preserves the original cell/RNG order.
        death_rows,birth_rows,within_death_rows,within_birth_rows,pool_rows,group_rows = torch.stack(
            (deaths,births,within_deaths,within_births,pool_counts,groups)).cpu().tolist()
        group_cells = [[] for _ in range(self.mass_groups)]
        for cell,group in enumerate(group_rows):
            group_cells[group].append(cell)
        priorities = torch.rand(len(features),device=self.device,dtype=torch.float64,generator=generator)
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        children_by_cell,parents_by_cell = [],[]
        for cell in range(self.cells):
            count = death_rows[cell]
            if count:
                rows = ((ids==cell)&death_eligible).nonzero().flatten()
                children_by_cell.append(rows[priorities[rows].argsort(stable=True)[:count]])
            else:
                children_by_cell.append(empty)
            count = birth_rows[cell]
            if count:
                draw = torch.randperm(pool_rows[cell],device=self.device,generator=generator)[:count]
                parents_by_cell.append(pool[cell,draw])
            else:
                parents_by_cell.append(empty)
        children,parents = [],[]
        # Match the conservative within-neighborhood phase inside its group.
        # Only the explicit residual phase can move mass between groups.
        for group in range(self.mass_groups):
            for index in group_cells[group]:
                children.append(children_by_cell[index][:within_death_rows[index]])
                parents.append(parents_by_cell[index][:within_birth_rows[index]])
        for cell in range(self.cells):
            children.append(children_by_cell[cell][within_death_rows[cell]:])
            parents.append(parents_by_cell[cell][within_birth_rows[cell]:])
        child = torch.cat(children) if children else empty
        parent = torch.cat(parents) if parents else empty
        detail = dict(excess_cells=int(comparison["excess"].sum()),deficit_cells=int(comparison["deficit"].sum()),
                      discoveries=int((comparison["excess"]|comparison["deficit"]).sum()),
                      inaccessible_birth_quota=int(inaccessible),budget=budget,moves=len(child),
                      death_allocation=deaths,birth_allocation=births,eligible_parent_counts=eligible_counts,
                      target_counts=target,clean_counts=clean_counts,table_counts=table_counts,clean_vacancies=vacancies,
                      unique_parent_count=len(torch.unique(parent)),mass_topology=dict(self.mass_topology),
                      group_target_counts=group_target,group_clean_counts=group_clean,
                      within_group_moves=int(within.sum()),between_group_moves=cross_pairs)
        supported_children = child[~flags[child]]
        detail.update(isolation_guard_rejects=isolation_guard_rejects,
                      ordinary_flagged_deaths=int(flags[child].sum()),
                      death_policy="global_count_certified_flagged_outside_only",certified_budget=certified_budget,
                      count_partition=dict(self.count_partition),count_boundary=self.count_boundary.clone(),
                      query_category_ids=categories,child_category_ids=categories[child],parent_category_ids=categories[parent],
                      child_region_ids=categories[child].remainder(2),parent_region_ids=categories[parent].remainder(2),
                      death_certified_regions=comparison["excess"],birth_certified_regions=comparison["deficit"],
                      death_category_allocation=torch.bincount(categories[child],minlength=2*self.cells),
                      birth_category_allocation=torch.bincount(categories[parent],minlength=2*self.cells),
                      raw_certified_death_capacity=raw_death_capacity,raw_certified_birth_capacity=raw_birth_capacity,
                      spent_certified_death_capacity=spent_deaths,spent_certified_birth_capacity=spent_births,
                      residual_certified_death_capacity=residual_death_capacity,
                      residual_certified_birth_capacity=residual_birth_capacity,
                      planned_supported_counts=clean_counts-torch.bincount(ids[supported_children],minlength=self.cells)
                          +torch.bincount(ids[parent],minlength=self.cells))
        return child,parent,detail


    @torch.no_grad()
    def ordinary_transport(self, query_features, flags, comparison, *, generator, pvalues=None, max_moves=None):
        """Mass, local support, then aggregate support under one ordinary budget."""
        if (comparison.get("multiplicity") != 3*self.cells+2
                or comparison.get("cutoff") != Q/(3*self.cells+2)
                or comparison["mass"]["difference"].shape != (self.cells,)
                or comparison["support"]["difference"].shape != (2*self.cells,)
                or comparison["global_support"]["difference"].shape != (2,)):
            raise ValueError("joint transport requires the common 3K+2 comparison")
        self._input(query_features)
        rows = len(query_features)
        if flags.shape != (rows,) or flags.dtype != torch.bool or flags.device != self.device:
            raise ValueError("invalid isolation flags")
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        budget = math.floor(Q*rows) if max_moves is None else min(int(max_moves),math.floor(Q*rows))
        budget = max(0,budget)
        mass_child,mass_parent,mass = self._ordinary_mass_transport(query_features,flags,comparison["mass"],
                    generator=generator,pvalues=pvalues,max_moves=budget)
        ids, _ = self.assign(query_features)
        clean = torch.bincount(ids[~flags],minlength=self.cells)
        mass_supported_child = mass_child[~flags[mass_child]]
        after_mass = clean-torch.bincount(ids[mass_supported_child],minlength=self.cells)
        after_mass += torch.bincount(ids[mass_parent],minlength=self.cells)
        remaining = budget-len(mass_child)
        possible = (remaining>0 and bool(flags.any())
                    and bool(comparison["support"]["excess"][1::2].any())
                    and bool(comparison["support"]["deficit"][0::2].any()))
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        if possible:
            support_child,support_parent,support = self._ordinary_support_transport(query_features,flags,
                    comparison["support"],generator=generator,pvalues=pvalues,max_moves=remaining,
                    reserved_children=mass_child,reserved_parents=mass_parent,supported_counts=after_mass)
            support["ran"] = True
        else:
            support_child,support_parent = empty,empty
            support = dict(ran=False,moves=0,budget=remaining,planned_supported_counts=after_mass.clone(),
                    death_allocation=torch.zeros(self.cells,device=self.device,dtype=torch.long),
                    birth_allocation=torch.zeros(self.cells,device=self.device,dtype=torch.long),
                    within_group_moves=0,between_group_moves=0,inaccessible_birth_quota=0)
        child,parent = torch.cat((mass_child,support_child)),torch.cat((mass_parent,support_parent))
        categories = self.count_categories(query_features)
        after_local = after_mass+torch.bincount(ids[support_parent],minlength=self.cells)
        remaining = budget-len(child)
        global_comparison = comparison["global_support"]
        possible = (remaining>0 and bool(flags.any()) and bool(global_comparison["excess"][1])
                    and bool(global_comparison["deficit"][0]))
        if possible:
            global_child,global_parent,global_phase = self._ordinary_global_transport(query_features,flags,
                    global_comparison,generator=generator,pvalues=pvalues,max_moves=remaining,
                    reserved_children=child,reserved_parents=parent,supported_counts=after_local)
            global_phase["ran"] = True
        else:
            global_child,global_parent = empty,empty
            global_phase = dict(ran=False,moves=0,budget=remaining,planned_supported_counts=after_local.clone(),
                    death_allocation=torch.zeros(self.cells,device=self.device,dtype=torch.long),
                    birth_allocation=torch.zeros(self.cells,device=self.device,dtype=torch.long),
                    within_group_moves=0,between_group_moves=0,inaccessible_birth_quota=0)
        child,parent = torch.cat((child,global_child)),torch.cat((parent,global_parent))
        planned = after_local+torch.bincount(ids[global_parent],minlength=self.cells)
        detail = dict(excess_cells=mass["excess_cells"]+int(comparison["support"]["excess"].sum())+int(global_comparison["excess"].sum()),
                deficit_cells=mass["deficit_cells"]+int(comparison["support"]["deficit"].sum())+int(global_comparison["deficit"].sum()),
                discoveries=mass["discoveries"]+int((comparison["support"]["excess"]|comparison["support"]["deficit"]).sum())
                    +int((global_comparison["excess"]|global_comparison["deficit"]).sum()),
                inaccessible_birth_quota=mass["inaccessible_birth_quota"]+support["inaccessible_birth_quota"]+global_phase["inaccessible_birth_quota"],
                budget=budget,moves=len(child),mass_moves=len(mass_child),support_moves=len(support_child),global_moves=len(global_child),
                mass_phase=mass,support_phase=support,mass_children=mass_child,mass_parents=mass_parent,
                support_children=support_child,support_parents=support_parent,
                global_phase=global_phase,global_children=global_child,global_parents=global_parent,
                action_kinds=torch.cat((torch.zeros_like(mass_child),torch.ones_like(support_child),torch.full_like(global_child,2))),
                death_allocation=mass["death_allocation"]+support["death_allocation"]+global_phase["death_allocation"],
                birth_allocation=mass["birth_allocation"]+support["birth_allocation"]+global_phase["birth_allocation"],
                eligible_parent_counts=mass["eligible_parent_counts"],target_counts=mass["target_counts"],
                clean_counts=clean,table_counts=mass["table_counts"],planned_supported_counts=planned,
                supported_after_mass=after_mass,supported_after_local=after_local,unique_parent_count=len(torch.unique(parent)),
                mass_topology=dict(self.mass_topology),group_target_counts=mass["group_target_counts"],
                group_clean_counts=self._group_counts(clean),
                within_group_moves=mass["within_group_moves"]+support["within_group_moves"]+global_phase["within_group_moves"],
                between_group_moves=mass["between_group_moves"]+support["between_group_moves"]+global_phase["between_group_moves"],
                isolation_guard_rejects=mass["isolation_guard_rejects"],ordinary_flagged_deaths=int(flags[child].sum()),
                death_policy="v4_mass_then_local_then_global_certified_flagged_outside",multiplicity=3*self.cells+2,cutoff=Q/(3*self.cells+2),
                query_cell_ids=ids,query_category_ids=categories,child_category_ids=categories[child],
                parent_category_ids=categories[parent],count_partition=dict(self.count_partition),
                count_boundary=self.count_boundary.clone())
        return child,parent,detail


class FeatureCellBirthDeath(ParticleBirthDeath):
    """Actual GANTrainer backend; keeps original FIFO and row-copy mechanics."""
    BACKEND_SCHEMA = 4

    def __init__(self,trainer,seed):
        super().__init__(trainer,seed)
        self.population_policy = population_policy(self.N)
        if not self.population_policy["finite_resolution_feasible"]:
            raise ValueError("use make_feature_cell_birth_death for an infeasible small population")
        recipe = trainer.recipe
        self.settings = dict(cells=recipe.birth_death_cells, rank=recipe.birth_death_metric_rank, chunk=recipe.birth_death_chunk, parent_policy=recipe.birth_death_parent_policy, lloyd=LLOYD_PASSES, power=POWER_PASSES, parent_reservoir=PARENT_RESERVOIR, real_anchors=1, parent_rank=None, latent_kernel='bounded_local_dv12_lineage', latent_neighbors=PARENT_RESERVOIR, latent_rank=recipe.birth_death_metric_rank, lineage_degree=min(recipe.birth_death_metric_rank, PARENT_RESERVOIR, self.N - 1), lineage_policy='symmetric_copy_links_newest_first_invalidate_overwrites', latent_candidate_bound=PARENT_RESERVOIR + min(recipe.birth_death_metric_rank, PARENT_RESERVOIR, self.N - 1), mass_policy='joint_mass_local_global_common_3K_plus_2_unique_parents_v1', isolation_parent_pool='supported_same_cell_without_replacement', count_partition='even_fit_score_order_statistic_2K', count_family='original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2')
        self.lineage = LatentLineage(self.N,self.settings["lineage_degree"],trainer.device)
        self.latent_geometry = BoundedLatentGeometry(rank=recipe.birth_death_metric_rank,
                          neighbors=PARENT_RESERVOIR,chunk=recipe.birth_death_chunk,lineage=self.lineage)
        self._copy_parent_rows = None
        self._sampling_prior, self._sampling_controller = trainer.prior, trainer.controller
        self.snapshot = None
        self.snapshot_serial = 0
        self.counters.update({key:0 for key in ("cell_evals","cell_discoveries","ordinary_moves","feature_rebuilds",
                                                "feature_distance_cells","projection_products","count_test_terms",
                                                "feature_forward_rows","invalidated_cells")})

    @torch.no_grad()
    def _jitter(self,trainer,latent,prior,noise):
        return self.latent_geometry.displacement(latent, prior, trainer.controller.latent_bandwidth, noise,
                                                 rows=self._copy_parent_rows)

    @torch.no_grad()
    def _move(self,trainer,child,parent):
        self._copy_parent_rows = parent
        try:
            super()._move(trainer,child,parent)
        finally:
            self._copy_parent_rows = None
        self.lineage.register_copies(child,parent)

    def perturb_latent(self,latent,stream,controller=None,record=False,*,prior=None,rows=None):
        controller = self._sampling_controller if controller is None else controller
        prior = self._sampling_prior if prior is None else prior
        noise = torch.randn(latent.shape,device=latent.device,dtype=latent.dtype,generator=stream)
        delta = self.latent_geometry.displacement(latent, prior, controller.latent_bandwidth, noise,
                            record=controller.latent_applications if record else None,rows=rows)
        return latent+delta

    @torch.no_grad()
    def _capture_generated(self,trainer,latent,*,sigma=0.,jitter=False,rows=None):
        features = []
        for start in range(0,len(latent),self.settings["chunk"]):
            block=latent[start:start+self.settings["chunk"]]
            if jitter:
                noise = torch.randn(block.shape,device=block.device,dtype=block.dtype,generator=self.stream)
                block = block+self.latent_geometry.displacement(block,trainer.prior,
                    trainer.controller.latent_bandwidth,noise,
                    rows=None if rows is None else rows[start:start+len(block)])
            raw = trainer.G(block)
            if tuple(raw.shape[1:]) != self.sample_shape:
                raise ValueError("generator output shape does not match the real FIFO")
            if sigma:
                raw = raw+float(sigma)*torch.randn(raw.shape,device=raw.device,dtype=raw.dtype,generator=self.stream)
            features.append(self._features(trainer,raw,chunk=self.settings["chunk"]))
            self.counters["feature_forward_rows"] += len(block)
        return torch.cat(features)

    @torch.no_grad()
    def maybe_apply(self,trainer,sigma_out):
        self.moved_rows = None
        if not self.ready():
            return None
        sigma = 0. if sigma_out is None else float(sigma_out)
        if not math.isfinite(sigma) or sigma<0:
            raise ValueError("output sigma must be finite and nonnegative")
        self.rows_since_eval = 0
        self.counters["evals"] += 1
        self.counters["cell_evals"] += 1
        self.snapshot_serial += 1
        self.S.zero_(); self.W.zero_(); self.n.zero_(); self.pending.zero_()
        modes = [(module,module.training) for root in (trainer.G,trainer.D) for module in root.modules()]
        begin = time.perf_counter()
        try:
            trainer.G.eval(); trainer.D.eval()
            z = trainer.prior.z.detach()
            q = self._capture_generated(trainer,z)
            R = self._features(trainer,self.reservoir,chunk=self.settings["chunk"])
            self.counters["feature_forward_rows"] += self.N
            snapshot = FeatureCellSnapshot.fit(R,generator=self.stream,cells=self.settings["cells"],
                                              rank=self.settings["rank"],chunk=self.settings["chunk"])
            self.snapshot = snapshot
            pick = torch.randint(self.N,(self.N,),device=z.device,generator=self.stream)
            F = self._capture_generated(trainer,z[pick],sigma=sigma,jitter=True,rows=pick)
            flags,pvalues,_ = snapshot.support(q)
            if not self.isolation:
                flags.zero_()
            snapshot.cache_queries(q)
            comparison = snapshot.cell_comparison(F)
            child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=self.stream,pvalues=pvalues)
            iso_child,iso_parent,anchored = snapshot.select_parents(q,flags,ordinary_children=child,ordinary_parents=parent,
                                                                 generator=self.stream,pvalues=pvalues)
            c = self.counters
            c["feature_rebuilds"] += 1
            c["cell_discoveries"] += ordinary["discoveries"]
            c["discoveries"] += ordinary["discoveries"]
            c["realised_deaths"] += len(child); c["realised_births"] += len(parent)
            if self.isolation:
                c["iso_evals"] += 1; c["iso_flagged"] += int(flags.sum())
                guard = bool(anchored["guard_passed"])
                c["iso_acted" if guard else "iso_skipped"] += bool(flags.any())
                c["iso_dup_skips"] += snapshot.duplicate_fraction>Q
                self.iso_log = (self.iso_log+[[trainer.completed_steps+1,int(flags.sum()),int(guard)]])[-30:]
            last = dict(step=trainer.completed_steps + 1, backend='feature_cells', snapshot=self.snapshot_serial, cells=snapshot.cells, metric_rank=snapshot.rank, k=None, d_R=None, d_F=None, eligible=int((pvalues > Q).sum()), discoveries=ordinary['discoveries'], ordinary_discoveries=ordinary['discoveries'], ordinary_excess_cells=ordinary['excess_cells'], ordinary_deficit_cells=ordinary['deficit_cells'], ordinary_budget=ordinary['budget'], ordinary_inaccessible_birth_quota=ordinary['inaccessible_birth_quota'], ordinary_moves=0 if self.dry_run else len(child), iso_flagged=int(flags.sum()), iso_moves=0 if self.dry_run else len(iso_child), moves=0, anchored_inaccessible_deficit=anchored['inaccessible_deficit'], calibration_rows=snapshot.calibration_rows, pvalue_floor=1 / (snapshot.calibration_rows + 1), minimum_bh_flag_fraction=1 / (Q * (snapshot.calibration_rows + 1)), duplicate_fraction=snapshot.duplicate_fraction, statistic='conditional_categorical_transport', density_ratio=None, statistical_limit='conditional iid cell test; no repeated adaptive guarantee', mass_topology=dict(snapshot.mass_topology), ordinary_within_group_moves=ordinary['within_group_moves'], ordinary_between_group_moves=ordinary['between_group_moves'], unique_reaction_parents=len(torch.unique(torch.cat((parent, iso_parent)))), isolation_unfilled=anchored.get('unfilled_repairs', int(flags.sum())), count_partition=dict(snapshot.count_partition), count_boundary=float(snapshot.count_boundary), count_categories=comparison['categories'], count_multiplicity=comparison['multiplicity'], count_cutoff=comparison['cutoff'], ordinary_mass_moves=ordinary['mass_moves'], ordinary_support_moves=ordinary['support_moves'], ordinary_global_moves=ordinary['global_moves'], ordinary_death_policy=ordinary['death_policy'])
            if not snapshot.valid_metric:
                last["skip"] = "no varying learned features"
            if not self.dry_run:
                if len(child):
                    self._move(trainer,child,parent)
                if len(iso_child):
                    self._move(trainer,iso_child,iso_parent)
                moved = torch.cat((child,iso_child))
                if len(moved):
                    self.moved_rows = moved
                    refreshed = self._capture_generated(trainer,trainer.prior.z.detach()[moved])
                    invalidated,affected = snapshot.refresh_rows(moved,refreshed)
                    self.S[invalidated]=0.; self.W[invalidated]=0.; self.n[invalidated]=0
                    c["stale_resets"] += int(invalidated.sum()); c["invalidated_cells"] += int(affected.sum())
                    last.update(stale_resets=int(invalidated.sum()),invalidated_cells=int(affected.sum()))
                c["ordinary_moves"] += len(child); c["moves"] += len(child)
                c["matched"] += len(child)
                if self.isolation:
                    c["iso_moves"] += len(iso_child)
                last["moves"] = len(child)+len(iso_child)
            c["feature_distance_cells"] += snapshot.work["distance_cells"]
            c["projection_products"] += snapshot.work["projection_products"]
            c["count_test_terms"] += snapshot.work["count_test_terms"]
            last.update(work=dict(snapshot.work),eval_seconds=time.perf_counter()-begin)
            self.last = last
            return last
        finally:
            for module,mode in modes:
                module.training = mode

    def diagnostics(self):
        result = super().diagnostics()
        result.update(backend="feature_cells",settings=dict(self.settings),snapshot_serial=self.snapshot_serial,
                      snapshot_work=None if self.snapshot is None else dict(self.snapshot.work),
                      population_policy=dict(self.population_policy),
                      lineage_edges=int((self.lineage.neighbors>=0).sum())//2)
        return result

    def state_dict(self):
        result = super().state_dict()
        result.update(backend="feature_cells",backend_schema=self.BACKEND_SCHEMA,settings=dict(self.settings),
                      sample_shape=self.sample_shape,snapshot_serial=self.snapshot_serial,
                      population_policy=dict(self.population_policy),lineage_neighbors=self.lineage.neighbors.clone())
        return result

    def check_state(self,state):
        extra = {"backend","backend_schema","settings","sample_shape","snapshot_serial","population_policy","lineage_neighbors"}
        base = set(self._TENSORS)|{"fill","cursor","rows_since_eval","counters","last","stream"}
        if (not isinstance(state,dict) or set(state)!=base|extra or state.get("backend")!="feature_cells"
                or state.get("backend_schema")!=self.BACKEND_SCHEMA or state.get("settings")!=self.settings
                or state.get("population_policy")!=self.population_policy):
            raise ValueError("invalid or incompatible feature-cell checkpoint backend/settings")
        super().check_state({key:state[key] for key in base})
        self.lineage.validate(state["lineage_neighbors"])
        for key in ("fill","cursor","rows_since_eval","snapshot_serial"):
            value=state[key]
            if type(value) is not int or value<0:
                raise ValueError("invalid feature-cell checkpoint counters")
        if state["fill"]>self.N or state["cursor"]>=self.N:
            raise ValueError("invalid feature-cell FIFO state")
        if (not isinstance(state["counters"],dict) or set(state["counters"])!=set(self.counters)
                or any(type(value) is not int or value<0 for value in state["counters"].values())
                or not isinstance(state["last"],dict)):
            raise ValueError("invalid feature-cell diagnostics state")
        for key in ("S","W","n","anchor","radius","pending"):
            value,expected=state[key],getattr(self,key)
            if (value is None or value.shape!=expected.shape or value.dtype!=expected.dtype
                    or (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
                raise ValueError("invalid feature-cell row state")
        shape=state["sample_shape"]
        if shape is not None and (not isinstance(shape,(tuple,list)) or not shape
                or any(type(value) is not int or value<=0 for value in shape)):
            raise ValueError("invalid feature-cell sample shape")
        reservoir=state["reservoir"]
        if (reservoir is None) != (shape is None):
            raise ValueError("inconsistent feature-cell reservoir/sample shape")
        if reservoir is None and (state["fill"] or state["cursor"] or state["rows_since_eval"]):
            raise ValueError("feature-cell FIFO counters require a reservoir")
        if reservoir is not None and (reservoir.ndim!=2 or reservoir.shape[1]!=math.prod(shape)
                                      or reservoir.dtype!=self.radius.dtype or not bool(torch.isfinite(reservoir).all())):
            raise ValueError("invalid feature-cell reservoir")

    def load_state_dict(self,state):
        self.check_state(state)
        base=set(self._TENSORS)|{"fill","cursor","rows_since_eval","counters","last","stream"}
        # Base load calls self.check_state, so copy the validated base fields
        # directly while preserving the backend-specific schema validation.
        for key in self._TENSORS:
            value=state[key]
            setattr(self,key,None if value is None else value.clone().to(self.S.device))
        self.fill,self.cursor,self.rows_since_eval=state["fill"],state["cursor"],state["rows_since_eval"]
        self.counters,self.last=dict(state["counters"]),dict(state["last"])
        self.stream.set_state(state["stream"].cpu())
        self.sample_shape=None if state["sample_shape"] is None else tuple(state["sample_shape"])
        self.snapshot_serial=state["snapshot_serial"]
        self.snapshot=None
        self.moved_rows=None
        self._copy_parent_rows=None
        self.lineage.neighbors=state["lineage_neighbors"].clone().to(self.S.device)
        self.latent_geometry.clear()
        self._heads=None
