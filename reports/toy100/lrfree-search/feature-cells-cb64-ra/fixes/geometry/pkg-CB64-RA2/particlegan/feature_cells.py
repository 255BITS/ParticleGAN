"""Bounded learned-feature cells, categorical transport and real-anchor repair.

This is an alternative reaction law, not the kNN/Beta law in birth_death.py.
No data-space distances, evaluator imports or full-table neighbor searches.
"""
import math
import time
import weakref
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
    def __init__(self, *, rank=8, neighbors=64, chunk=256):
        for name, value in (("rank", rank), ("neighbors", neighbors), ("chunk", chunk)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.rank, self.neighbors, self.chunk = rank, neighbors, chunk
        self.clear()

    def clear(self):
        self._entries = {}
        self.work = dict(builds=0, distance_pairs=0, max_query_rows=0, max_candidates=0)

    @torch.no_grad()
    def _orders(self, points):
        key = id(points)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is points and entry[1] == points._version:
            return entry[2]
        spread = points.detach().std(0, unbiased=False)
        count = min(self.rank, int((spread > 0).sum()))
        axes = spread.argsort(descending=True, stable=True)[:count]
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
    def radius(self, latent, prior):
        return self._local_geometry(latent, prior)[0]

    @torch.no_grad()
    def _local_geometry(self, latent, prior):
        points = prior.z
        orders = self._orders(points)
        if not orders:
            return latent.new_zeros(len(latent)), torch.zeros_like(latent)
        output, widths = [], []
        for block in latent.detach().split(self.chunk):
            distances, point_ids = [], []
            for index, (axis, values, rows) in enumerate(orders):
                count = self.neighbors//len(orders)+(index < self.neighbors % len(orders))
                if not count:
                    continue
                offsets = torch.arange(-(count//2), count-count//2, device=block.device)
                positions = torch.searchsorted(values, block[:, axis].contiguous())[:, None]+offsets
                valid = (positions >= 0) & (positions < len(values))
                candidates = rows[positions.clamp(0, len(values)-1)]
                distance = (block[:, None]-points.detach()[candidates]).square().sum(-1)
                distance.masked_fill_(~valid | (distance == 0), float("inf"))
                distances.append(distance); point_ids.append(candidates)
                self.work["distance_pairs"] += len(block)*count
            self.work["max_query_rows"] = max(self.work["max_query_rows"], len(block))
            self.work["max_candidates"] = max(self.work["max_candidates"], self.neighbors)
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

    def displacement(self, latent, prior, bandwidth, noise, *, record=None):
        radius, local_width = self._local_geometry(latent, prior)
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
                               backend="feature_cells", kernel="bounded_local_dv12"))
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
    if (real_counts.shape != fake_counts.shape or real_counts.ndim != 1
            or real_counts.dtype != torch.long or fake_counts.dtype != torch.long
            or real_counts.device != fake_counts.device or real_rows <= 0 or fake_rows <= 0
            or bool((real_counts < 0).any()) or bool((fake_counts < 0).any())
            or int(real_counts.sum()) != real_rows or int(fake_counts.sum()) != fake_rows):
        raise ValueError("invalid categorical counts")
    device = real_counts.device
    factorial = torch.lgamma(torch.arange(real_rows + fake_rows + 1, device=device, dtype=torch.float64) + 1)
    result = torch.ones(len(real_counts), device=device, dtype=torch.float64)
    for cell in range(len(real_counts)):
        observed, total = int(real_counts[cell]), int(real_counts[cell] + fake_counts[cell])
        low, high = max(0, total - fake_rows), min(real_rows, total)
        j = torch.arange(low, high + 1, device=device, dtype=torch.long)
        log_probability = (factorial[real_rows] - factorial[j] - factorial[real_rows-j]
                           + factorial[fake_rows] - factorial[total-j] - factorial[fake_rows-total+j]
                           - factorial[real_rows+fake_rows] + factorial[total] + factorial[real_rows+fake_rows-total])
        # Numerical equality tolerance includes both mathematical equal tails.
        observed_log_probability = log_probability[observed-low]
        selected = log_probability <= observed_log_probability + 1e-10
        result[cell] = torch.logsumexp(log_probability[selected], 0).exp().clamp_max(1.)
        if work is not None:
            work["count_test_terms"] += len(j)
    return result


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
        selected = int((x-x.mean(0)).square().sum(1).argmin())
        nearest = x.new_full((len(x),), float("inf"))
        for cell in range(self.cells):
            centers[cell] = x[selected]
            dist2 = (x-centers[cell]).square().sum(1)
            self.work["distance_cells"] += len(x)
            nearest = torch.minimum(nearest, dist2)
            selected = int(nearest.argmax())
        self.centers = centers
        for _ in range(LLOYD_PASSES):
            ids, _ = self._assign_metric(x)
            # Fixed K loops retain row order and avoid nondeterministic CUDA
            # floating scatter atomics, needed for checkpoint continuation.
            for cell in range(self.cells):
                members = x[ids == cell]
                if len(members):
                    self.centers[cell] = members.mean(0)
        ids, dist2 = self._assign_metric(x)
        self.reference_counts = torch.bincount(ids, minlength=self.cells)
        sse = torch.stack([dist2[ids == cell].sum() for cell in range(self.cells)])
        radius = (sse / self.reference_counts.clamp_min(1)).sqrt()
        positive = radius[(self.reference_counts > 0) & (radius > 0)]
        global_radius = float(torch.quantile(positive, .5)) if len(positive) else 1.
        self.cell_scale = ((sse + SCALE_PRIOR * global_radius**2) /
                           (self.reference_counts + SCALE_PRIOR)).sqrt().clamp_min(global_radius*1e-3)
        self.real_representatives = x.new_empty((self.cells, self.rank))
        self.real_representative_rows = torch.empty(self.cells, device=self.device, dtype=torch.long)
        for cell in range(self.cells):
            members = (ids == cell).nonzero().flatten()
            if len(members):
                chosen = members[dist2[members].argmin()]
            else:
                chosen = (x-self.centers[cell]).square().sum(1).argmin()
                self.work["distance_cells"] += len(x)
            self.real_representatives[cell] = x[chosen]
            self.real_representative_rows[cell] = 2*chosen
        calibration = self.transform(real[1::2])
        calibration_ids, _ = self._assign_metric(calibration)
        self.real_calibration_counts = torch.bincount(calibration_ids, minlength=self.cells)
        self.calibration_rows = len(calibration)
        self.null_scores = self._scores_metric(calibration).sort().values
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
        eligible_counts = torch.bincount(ids[eligible],minlength=self.cells)
        for cell in range(self.cells):
            members = (eligible & (ids == cell)).nonzero().flatten()
            selected = members[priorities[members].argsort(stable=True)[:PARENT_RESERVOIR]]
            pool[cell,:len(selected)] = selected
            counts[cell] = len(selected)
        return ids,pool,counts,eligible_counts

    @torch.no_grad()
    def select_parents(self, query_features, flags, *, ordinary_children=None, generator, pvalues=None):
        """Real-anchor isolation plan; detail contains the actual final candidate ball."""
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
        ids,pool,counts,eligible_counts = self._pool(features,eligible,generator)
        accessible = counts > 0
        detail["accessible_cells"] = int(accessible.sum())
        deficit = (len(features)*self.real_calibration_counts.double()/self.calibration_rows-eligible_counts).clamp_min(0.)
        detail["inaccessible_deficit"] = float(deficit[~accessible].sum())
        if not bool(accessible.any()) or not len(dead):
            return empty,empty,detail
        children,parents,candidates,masks,anchor_rows,anchor_cells = [],[],[],[],[],[]
        anchors_to_query = min(REAL_ANCHORS,self.cells)
        for block_rows in dead.split(self.chunk):
            distances = self._distance(features[block_rows],self.real_representatives)
            distances[:,~accessible] = float("inf")
            closest = distances.argsort(dim=1,stable=True)[:,:anchors_to_query]
            anchor = closest[:,0]
            candidate = pool[closest].reshape(len(block_rows),-1)
            valid = candidate >= 0
            candidate_features = features[candidate.clamp_min(0)]
            distance = (candidate_features-self.real_representatives[anchor,None]).norm(dim=2)
            distance[~valid] = float("inf")
            self.work["distance_cells"] += distance.numel()
            self.work["parent_candidate_cells"] += distance.numel()
            self.work["max_distance_rows"] = max(self.work["max_distance_rows"],len(block_rows))
            self.work["max_distance_columns"] = max(self.work["max_distance_columns"],distance.shape[1])
            cap = min(PARENT_RANK_CAP,distance.shape[1])
            limit = torch.minimum(2*distance.min(1).values,distance.topk(cap,dim=1,largest=False).values[:,-1])
            ball = valid & (distance <= limit[:,None])
            draw = torch.rand(distance.shape,device=self.device,dtype=torch.float64,generator=generator)
            selected = draw.masked_fill(~ball,-1).argmax(1)
            accepted = ball.any(1)
            children.append(block_rows[accepted])
            parents.append(candidate[torch.arange(len(block_rows),device=self.device),selected][accepted])
            candidates.append(candidate[accepted])
            masks.append(ball[accepted])
            anchor_rows.append(self.real_representative_rows[anchor][accepted])
            anchor_cells.append(anchor[accepted])
        child,parent = torch.cat(children),torch.cat(parents)
        detail.update(candidate_ids=torch.cat(candidates),candidate_mask=torch.cat(masks),
                      anchor_reference_rows=torch.cat(anchor_rows),anchor_cell_ids=torch.cat(anchor_cells),
                      parent_cell_ids=ids[parent],parent_cell_deficits=deficit[ids[parent]])
        return child,parent,detail

    @torch.no_grad()
    def cell_comparison(self, fake_features):
        fake_ids,_ = self.assign(fake_features)
        fake_counts = torch.bincount(fake_ids,minlength=self.cells)
        p = conditional_count_pvalues(self.real_calibration_counts,fake_counts,self.calibration_rows,len(fake_features),self.work)
        difference = fake_counts.double()/len(fake_features)-self.real_calibration_counts.double()/self.calibration_rows
        significant = p <= Q/self.cells
        if not self.valid_metric:
            significant.zero_()
        return dict(real_counts=self.real_calibration_counts,fake_counts=fake_counts,pvalues=p,
                    difference=difference,excess=significant&(difference>0),deficit=significant&(difference<0))

    @torch.no_grad()
    def ordinary_transport(self, query_features, flags, comparison, *, generator, pvalues=None, max_moves=None):
        """One count-certified quota allocation; no inherited sequential evidence."""
        features = self.transform(query_features)
        if pvalues is None:
            _,pvalues,_ = self.support(query_features)
        ids,pool,pool_counts,eligible_counts = self._pool(features,~flags&(pvalues>Q),generator)
        difference = comparison["difference"]
        death_capacity = torch.floor(len(features)*difference.clamp_min(0.)+1e-10).long()*comparison["excess"]
        birth_capacity = torch.floor(len(features)*(-difference).clamp_min(0.)+1e-10).long()*comparison["deficit"]
        available_deaths = torch.bincount(ids[~flags],minlength=self.cells)
        death_capacity = torch.minimum(death_capacity,available_deaths)
        inaccessible = birth_capacity[pool_counts==0].sum()
        birth_capacity[pool_counts==0] = 0
        budget = math.floor(Q*len(features)) if max_moves is None else min(int(max_moves),math.floor(Q*len(features)))
        pairs = min(int(death_capacity.sum()),int(birth_capacity.sum()),max(0,budget))
        deaths = _integer_allocate(death_capacity,pairs)
        births = _integer_allocate(birth_capacity,pairs)
        priorities = torch.rand(len(features),device=self.device,dtype=torch.float64,generator=generator)
        children,parents = [],[]
        for cell in range(self.cells):
            count = int(deaths[cell])
            if count:
                rows = ((ids==cell)&~flags).nonzero().flatten()
                children.append(rows[priorities[rows].argsort(stable=True)[:count]])
            count = int(births[cell])
            if count:
                draw = torch.randint(int(pool_counts[cell]),(count,),device=self.device,generator=generator)
                parents.append(pool[cell,draw])
        empty = torch.empty(0,device=self.device,dtype=torch.long)
        child = torch.cat(children) if children else empty
        parent = torch.cat(parents) if parents else empty
        detail = dict(excess_cells=int(comparison["excess"].sum()),deficit_cells=int(comparison["deficit"].sum()),
                      discoveries=int((comparison["excess"]|comparison["deficit"]).sum()),
                      inaccessible_birth_quota=int(inaccessible),budget=budget,moves=len(child),
                      death_allocation=deaths,birth_allocation=births,eligible_parent_counts=eligible_counts)
        return child,parent,detail


class FeatureCellBirthDeath(ParticleBirthDeath):
    """Actual GANTrainer backend; keeps original FIFO and row-copy mechanics."""
    BACKEND_SCHEMA = 2

    def __init__(self,trainer,seed):
        super().__init__(trainer,seed)
        recipe = trainer.recipe
        self.settings = dict(cells=recipe.birth_death_cells,rank=recipe.birth_death_metric_rank,
                             chunk=recipe.birth_death_chunk,parent_policy=recipe.birth_death_parent_policy,
                             lloyd=LLOYD_PASSES,power=POWER_PASSES,parent_reservoir=PARENT_RESERVOIR,
                             real_anchors=REAL_ANCHORS,parent_rank=PARENT_RANK_CAP,
                             latent_kernel="bounded_local_dv12",latent_neighbors=PARENT_RESERVOIR,
                             latent_rank=recipe.birth_death_metric_rank)
        self.latent_geometry = BoundedLatentGeometry(rank=recipe.birth_death_metric_rank,
                                                   neighbors=PARENT_RESERVOIR, chunk=recipe.birth_death_chunk)
        self._sampling_prior, self._sampling_controller = trainer.prior, trainer.controller
        self.snapshot = None
        self.snapshot_serial = 0
        self.counters.update({key:0 for key in ("cell_evals","cell_discoveries","ordinary_moves","feature_rebuilds",
                                                "feature_distance_cells","projection_products","count_test_terms",
                                                "feature_forward_rows","invalidated_cells")})

    @torch.no_grad()
    def _jitter(self,trainer,latent,prior,noise):
        return self.latent_geometry.displacement(latent, prior, trainer.controller.latent_bandwidth, noise)

    def perturb_latent(self,latent,stream,controller=None,record=False,*,prior=None):
        controller = self._sampling_controller if controller is None else controller
        prior = self._sampling_prior if prior is None else prior
        noise = torch.randn(latent.shape,device=latent.device,dtype=latent.dtype,generator=stream)
        delta = self.latent_geometry.displacement(latent, prior, controller.latent_bandwidth, noise,
                            record=controller.latent_applications if record else None)
        return latent+delta

    @torch.no_grad()
    def _capture_generated(self,trainer,latent,*,sigma=0.,jitter=False):
        features = []
        for block in latent.split(self.settings["chunk"]):
            if jitter:
                noise = torch.randn(block.shape,device=block.device,dtype=block.dtype,generator=self.stream)
                block = block+self._jitter(trainer,block,trainer.prior,noise)
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
            F = self._capture_generated(trainer,z[pick],sigma=sigma,jitter=True)
            flags,pvalues,_ = snapshot.support(q)
            if not self.isolation:
                flags.zero_()
            snapshot.cache_queries(q)
            comparison = snapshot.cell_comparison(F)
            child,parent,ordinary = snapshot.ordinary_transport(q,flags,comparison,generator=self.stream,pvalues=pvalues)
            iso_child,iso_parent,anchored = snapshot.select_parents(q,flags,ordinary_children=child,
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
            last = dict(step=trainer.completed_steps+1,backend="feature_cells",snapshot=self.snapshot_serial,
                        cells=snapshot.cells,metric_rank=snapshot.rank,k=None,d_R=None,d_F=None,
                        eligible=int((pvalues>Q).sum()),discoveries=ordinary["discoveries"],
                        ordinary_discoveries=ordinary["discoveries"],ordinary_excess_cells=ordinary["excess_cells"],
                        ordinary_deficit_cells=ordinary["deficit_cells"],ordinary_budget=ordinary["budget"],
                        ordinary_inaccessible_birth_quota=ordinary["inaccessible_birth_quota"],
                        ordinary_moves=0 if self.dry_run else len(child),iso_flagged=int(flags.sum()),
                        iso_moves=0 if self.dry_run else len(iso_child),moves=0,
                        anchored_inaccessible_deficit=anchored["inaccessible_deficit"],
                        calibration_rows=snapshot.calibration_rows,pvalue_floor=1/(snapshot.calibration_rows+1),
                        minimum_bh_flag_fraction=1/(Q*(snapshot.calibration_rows+1)),
                        duplicate_fraction=snapshot.duplicate_fraction,statistic="conditional_categorical_transport",
                        density_ratio=None,statistical_limit="conditional iid cell test; no repeated adaptive guarantee")
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
                      snapshot_work=None if self.snapshot is None else dict(self.snapshot.work))
        return result

    def state_dict(self):
        result = super().state_dict()
        result.update(backend="feature_cells",backend_schema=self.BACKEND_SCHEMA,settings=dict(self.settings),
                      sample_shape=self.sample_shape,snapshot_serial=self.snapshot_serial)
        return result

    def check_state(self,state):
        extra = {"backend","backend_schema","settings","sample_shape","snapshot_serial"}
        base = set(self._TENSORS)|{"fill","cursor","rows_since_eval","counters","last","stream"}
        if (not isinstance(state,dict) or set(state)!=base|extra or state.get("backend")!="feature_cells"
                or state.get("backend_schema")!=self.BACKEND_SCHEMA or state.get("settings")!=self.settings):
            raise ValueError("invalid or incompatible feature-cell checkpoint backend/settings")
        super().check_state({key:state[key] for key in base})
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
        self.latent_geometry.clear()
        self._heads=None
