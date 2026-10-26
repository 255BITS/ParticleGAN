"""Frozen learned-feature cell snapshot: metric, support, counts and planning.

One snapshot is fitted from the real FIFO at each feature-cell reaction. It
owns the even-fit metric chart and cells, the odd-calibrated support null,
the count partition and test families, the real-only mass topology and the
isolation/ordinary transport planners. No data-space distances or full-table
neighbor searches are used.
"""
import math

import torch

from .cell_statistics import (LLOYD_PASSES, PARENT_RESERVOIR, POWER_PASSES, Q, SCALE_PRIOR,
                              _features_ok, _fit_cell_count, check_flags, count_family,
                              move_budget, q_order_statistic, require_positive_ints)
from .cell_transport import (GLOBAL_TRANSPORT, MASS_TRANSPORT, SUPPORT_TRANSPORT,
                             plan_ordinary_transport)
from .birth_phase import plan_residual_global_copies


def _empty_phase(snapshot, budget, planned):
    """Diagnostics of an ordinary phase that had no possible certified move."""
    zeros = torch.zeros(snapshot.cells, device=snapshot.device, dtype=torch.long)
    return dict(ran=False, moves=0, budget=budget, planned_supported_counts=planned.clone(),
                death_allocation=zeros, birth_allocation=zeros.clone(),
                within_group_moves=0, between_group_moves=0, inaccessible_birth_quota=0)


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
        require_positive_ints(cells=cells, rank=rank, chunk=chunk)
        if (not isinstance(generator, torch.Generator)
                or torch.device(generator.device) != real_features.device):
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
            start = torch.randn((self.width, self.rank), device=self.device,
                                dtype=torch.float64, generator=generator)
            basis, _ = torch.linalg.qr(start, mode="reduced")
            for _ in range(POWER_PASSES):
                basis, _ = torch.linalg.qr(centered.T @ (centered @ basis), mode="reduced")
                self.work["projection_products"] += 2 * len(ref) * self.width * self.rank
            self.basis = basis
        else:
            self.basis = real.new_zeros((self.width, 0))
        x = self.transform(ref)
        self.cells = _fit_cell_count(cells, len(x), self.rank)
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
        self.cell_scale = ((sse + SCALE_PRIOR * global_radius**2)
                           / (self.reference_counts + SCALE_PRIOR)).sqrt().clamp_min(global_radius*1e-3)
        self.real_representatives = x.new_empty((self.cells, self.rank))
        self.real_representative_rows = torch.empty(self.cells, device=self.device, dtype=torch.long)
        for cell, (start, end) in enumerate(bounds):
            members = order[start:end]
            if len(members):
                closest = dist2.index_select(0, members).argmin().reshape(1)
                chosen = members.index_select(0, closest).squeeze(0)
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
        calibration_categories, calibration_scores = self._count_categories_metric(
            calibration, calibration_ids)
        self.real_calibration_category_counts = torch.bincount(calibration_categories,
                                                               minlength=2*self.cells)
        self.null_scores = calibration_scores.sort().values
        self.duplicate_fraction = 1. - len(torch.unique(real, dim=0)) / len(real)
        self.row_features = self.query_cell_ids = self.query_counts = None
        self.cache_version = 0
        self._update_storage()
        return self

    # Metric chart -------------------------------------------------------------

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
        squared = query.square().sum(1)[:, None]+points.square().sum(1)[None]-2*query@points.T
        return squared.clamp_min(0.)

    def _assign_metric(self, features):
        ids, dist = [], []
        for block in features.split(self.chunk):
            d, i = self._distance(block, self.centers).min(1)
            ids.append(i)
            dist.append(d)
        return torch.cat(ids), torch.cat(dist)

    def _scores_metric(self, features):
        scores = []
        for block in features.split(self.chunk):
            scaled = self._distance(block, self.centers)/self.cell_scale.square()[None]
            scores.append(scaled.min(1).values.sqrt())
        return torch.cat(scores)

    def _fit_count_partition_metric(self, fitted_features, fitted_ids):
        """Fixed global even-fit score boundary, retaining all 2K categories.

        This empirical fitted boundary is not a conformal p-value threshold.
        Its only role is to define a partition before independent odd counts.
        """
        if hasattr(self, "count_boundary"):
            raise ValueError("count partition is already frozen")
        scores = self._scores_metric(fitted_features)
        ordinal = q_order_statistic(len(scores))
        self.count_boundary = scores.sort().values[max(0, ordinal-1)].clone()
        self.count_partition = dict(rule="even_fit_score_order_statistic", q=Q,
                                    fitted_rows=len(scores), ordinal=ordinal,
                                    ties="inside", categories=2*self.cells)
        categories = 2*fitted_ids+(scores > self.count_boundary).long()
        self.reference_category_counts = torch.bincount(categories, minlength=2*self.cells)

    def _count_categories_metric(self, features, ids=None):
        if not hasattr(self, "count_boundary"):
            raise ValueError("fit count partition before assigning count categories")
        if ids is None:
            ids, _ = self._assign_metric(features)
        scores = self._scores_metric(features)
        return 2*ids+(scores > self.count_boundary).long(), scores

    @torch.no_grad()
    def count_categories(self, features):
        categories, _ = self._count_categories_metric(self.transform(features))
        return categories

    @torch.no_grad()
    def assign(self, features):
        return self._assign_metric(self.transform(features))

    def _support_pvalues(self, scores):
        null = self.null_scores
        return (1.+len(null)-torch.searchsorted(null, scores))/(1.+len(null))

    @torch.no_grad()
    def support(self, query_features):
        scores = self._scores_metric(self.transform(query_features))
        p = self._support_pvalues(scores)
        ordered = p.sort().values
        levels = Q*torch.arange(1, len(p)+1, device=self.device, dtype=p.dtype)/len(p)
        passed = (ordered <= levels).nonzero().flatten()
        if len(passed):
            flags = p <= ordered[passed[-1]]
        else:
            flags = torch.zeros(len(p), device=self.device, dtype=torch.bool)
        if self.duplicate_fraction > Q or not self.valid_metric:
            flags.zero_()
        return flags, p, scores

    # Query cache --------------------------------------------------------------

    @torch.no_grad()
    def cache_queries(self, query_features):
        self.row_features = self.transform(query_features)
        self.query_cell_ids, _ = self._assign_metric(self.row_features)
        self.query_counts = torch.bincount(self.query_cell_ids, minlength=self.cells)
        self._update_storage()
        return self.query_cell_ids

    @torch.no_grad()
    def refresh_rows(self, rows, repaired_features):
        if self.query_cell_ids is None:
            raise ValueError("cache_queries must precede row refresh")
        if (rows.ndim != 1 or rows.dtype != torch.long or rows.device != self.device
                or len(rows) != len(repaired_features)):
            raise ValueError("invalid repaired rows")
        if (len(torch.unique(rows)) != len(rows)
                or bool(((rows < 0) | (rows >= len(self.query_cell_ids))).any())):
            raise ValueError("invalid repaired row indices")
        moved = self.transform(repaired_features)
        after_ids, _ = self._assign_metric(moved)
        affected = torch.zeros(self.cells, device=self.device, dtype=torch.bool)
        affected[self.query_cell_ids[rows]] = True
        affected[after_ids] = True
        invalidated = affected[self.query_cell_ids]
        invalidated[rows] = True
        self.row_features[rows] = moved
        self.query_cell_ids[rows] = after_ids
        self.query_counts = torch.bincount(self.query_cell_ids, minlength=self.cells)
        self.cache_version += 1
        self._update_storage()
        return invalidated, affected

    # Parent pools, quotas and real topology -----------------------------------

    def _pool(self, features, eligible, generator):
        ids, _ = self._assign_metric(features)
        priorities = torch.rand(len(features), device=self.device, dtype=torch.float64,
                                generator=generator)
        pool = torch.full((self.cells, PARENT_RESERVOIR), -1, device=self.device, dtype=torch.long)
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
        return ids, pool, counts, eligible_counts

    def _mass_targets(self, rows):
        """Full reference mass for action quotas; tests still use heldout rows."""
        counts = self.reference_counts + self.real_calibration_counts
        quota = counts.double() * (rows / int(counts.sum()))
        target = quota.floor().long()
        remaining = rows - int(target.sum())
        if remaining:
            order = (quota-target).argsort(descending=True, stable=True)
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
        if hasattr(self, "mass_group_ids"):
            return self.mass_group_ids
        k = self.cells
        device = self.device
        group_ids = torch.arange(k, device=device, dtype=torch.long)
        threshold = None
        if k > 2:
            distance = self._distance(self.centers, self.centers)
            distance.fill_diagonal_(float("inf"))
            used = torch.zeros(k, device=device, dtype=torch.bool)
            used[0] = True
            nearest = distance[0].clone()
            parent = torch.zeros(k, device=device, dtype=torch.long)
            edge_parent, edge_child, edge_length = [], [], []
            for _ in range(k-1):
                child = nearest.masked_fill(used, float("inf")).argmin()
                edge_parent.append(parent[child].clone())
                edge_child.append(child.clone())
                edge_length.append(nearest[child].clone())
                used[child] = True
                shorter = distance[child] < nearest
                nearest = torch.minimum(nearest, distance[child])
                parent[shorter] = child
            edge_parent, edge_child, edge_length = map(torch.stack,
                                                       (edge_parent, edge_child, edge_length))
            tiny = torch.finfo(edge_length.dtype).tiny
            logs = edge_length.clamp_min(tiny).log().sort().values
            cumulative = logs.cumsum(0)
            cut = torch.arange(1, len(logs), device=device, dtype=logs.dtype)
            left = cumulative[:-1]/cut
            right = (cumulative[-1]-cumulative[:-1])/(len(logs)-cut)
            separation = cut*(len(logs)-cut)*(left-right).square()
            split = separation.argmax()
            threshold_tensor = ((logs[split]+logs[split+1])*.5).exp()
            threshold = float(threshold_tensor)
            connected = torch.eye(k, device=device, dtype=torch.bool)
            retained = edge_length <= threshold_tensor
            a, b = edge_parent[retained], edge_child[retained]
            connected[a, b] = True
            connected[b, a] = True
            for _ in range(k):
                updated = group_ids[None].expand(k, -1).masked_fill(~connected, k).min(1).values
                if torch.equal(updated, group_ids):
                    break
                group_ids = updated
            group_ids = torch.unique(group_ids, sorted=True, return_inverse=True)[1]
        self.mass_group_ids = group_ids
        self.mass_groups = int(group_ids.max())+1
        self.mass_topology = dict(rule="reference_center_mst_log_between_class_variance",
                                  groups=self.mass_groups, threshold_squared=threshold)
        self._update_storage()
        return group_ids

    def _group_counts(self, counts):
        groups = self._mass_topology()
        if (counts.dtype in (torch.int32, torch.int64) and counts.ndim == 1
                and len(counts) == len(groups)):
            members = groups[None] == torch.arange(self.mass_groups, device=groups.device)[:, None]
            # Integer sums retain exact quotas; the matrix is bounded by K*K.
            return counts[None].expand(self.mass_groups, -1).masked_fill(~members, 0).sum(1)
        return torch.stack([counts[groups == group].sum() for group in range(self.mass_groups)])

    # Isolation repair ---------------------------------------------------------

    @torch.no_grad()
    def select_parents(self, query_features, flags, *, ordinary_children=None,
                       ordinary_parents=None, generator, pvalues=None, supported_counts=None,
                       reserved_rows=None):
        """Vacancy-limited anchor repair with distinct supported parents.

        Clean supported rows, including planned ordinary moves, reserve their
        reference mass first. Each target cell supplies only its own bounded
        parent ball. A parent can seed at most one copy in the whole reaction;
        unfilled holes wait for later snapshots rather than borrowing full-cell
        mass or multiplying one surviving rare row.
        """
        features = self.transform(query_features)
        check_flags(flags, len(features), self.device)
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        dead = flags.nonzero().flatten()
        detail = dict(candidate_ids=empty.reshape(0, 1),
                      candidate_mask=torch.empty((0, 1), device=self.device, dtype=torch.bool),
                      anchor_reference_rows=empty, anchor_cell_ids=empty, parent_cell_ids=empty,
                      inaccessible_deficit=0., accessible_cells=0,
                      guard_passed=0 < len(dead) <= Q*len(features))
        if not detail["guard_passed"]:
            return empty, empty, detail
        if pvalues is None:
            _, pvalues, _ = self.support(query_features)
        eligible = ~flags & (pvalues > Q)
        for excluded in (reserved_rows, ordinary_children):
            if excluded is not None:
                eligible[excluded] = False
                allowed = torch.ones(len(features), device=self.device, dtype=torch.bool)
                allowed[excluded] = False
                dead = dead[allowed[dead]]
        if ordinary_parents is not None:
            if ordinary_children is None or len(ordinary_parents) != len(ordinary_children):
                raise ValueError("ordinary children/parents must be paired")
            eligible[ordinary_parents] = False
        ids, pool, counts, eligible_counts = self._pool(features, eligible, generator)
        target = self._mass_targets(len(features))
        groups = self._mass_topology()
        kept = torch.bincount(ids[~flags], minlength=self.cells)
        if supported_counts is not None:
            if (supported_counts.shape != (self.cells,) or supported_counts.device != self.device
                    or supported_counts.dtype != torch.long or bool((supported_counts < 0).any())):
                raise ValueError("invalid explicit isolation supported ledger")
            kept = supported_counts.clone()
        elif ordinary_parents is not None:
            supported_children = ordinary_children[~flags[ordinary_children]]
            kept -= torch.bincount(ids[supported_children], minlength=self.cells)
            kept += torch.bincount(ids[ordinary_parents], minlength=self.cells)
        vacancies = (self._group_counts(target)-self._group_counts(kept)).clamp_min(0)
        # Partition membership and the unchanged support p>Q criterion
        # define this own-cell pool. The old cross-cell 2*nearest ball could
        # shrink 64 supported rows to one and repeatedly clone that family;
        # it is redundant once parent and vacancy must share the same cell.
        balls = pool >= 0
        self.work["parent_candidate_cells"] += pool.numel()
        supply = balls.sum(1)
        capacity = supply*(vacancies[groups] > 0)
        accessible = capacity > 0
        detail["accessible_cells"] = int(accessible.sum())
        group_supply = self._group_counts(supply)
        detail.update(inaccessible_deficit=float(vacancies[group_supply == 0].sum()),
                      target_counts=target, kept_counts=kept, group_vacancies=vacancies,
                      eligible_parent_counts=eligible_counts, unique_parent_capacity=capacity,
                      requested_repairs=len(dead),
                      candidate_policy="supported_same_cell_without_replacement",
                      mass_topology=dict(self.mass_topology))
        if not bool(accessible.any()) or not len(dead):
            return empty, empty, detail
        distances = torch.cat([self._distance(features[block], self.real_representatives)
                               for block in dead.split(self.chunk)])
        available = torch.ones(len(dead), device=self.device, dtype=torch.bool)
        remaining_capacity = capacity.clone()
        remaining_vacancies = vacancies.clone()
        assignments = torch.full((len(dead),), -1, device=self.device, dtype=torch.long)
        # Child-first proposals preserve learned-feature locality. Each cell
        # accepts its nearest proposals up to its vacancy and parent supply;
        # rejected children try the next nonfull cell. A contested round fills
        # at least one cell, so there are at most K rounds (no table search).
        for _ in range(self.cells):
            accessible = (remaining_capacity > 0) & (remaining_vacancies[groups] > 0)
            if not bool(available.any()) or not bool(accessible.any()):
                break
            current = distances.masked_fill(~accessible[None], float("inf"))
            nearest_distance, nearest = current.min(1)
            nearest_group = groups[nearest]
            same_group = groups[None] == nearest_group[:, None]
            alternative = current.masked_fill(same_group, float("inf")).min(1).values
            regret = alternative-nearest_distance
            for group in range(self.mass_groups):
                rows = (available & (nearest_group == group)).nonzero().flatten()
                if not len(rows) or not int(remaining_vacancies[group]):
                    continue
                # Preserve children with the worst alternative when a local
                # mass quota is contested; absolute closeness alone can use
                # that quota on flexible children and force distant repairs.
                rows = rows[nearest_distance[rows].argsort(stable=True)]
                rows = rows[regret[rows].argsort(descending=True, stable=True)]
                proposals = rows[:int(remaining_vacancies[group])]
                for cell_tensor in (groups == group).nonzero().flatten():
                    cell = int(cell_tensor)
                    local = proposals[nearest[proposals] == cell]
                    take = min(len(local), int(remaining_capacity[cell]))
                    if take:
                        chosen = local[:take]
                        assignments[chosen] = cell
                        available[chosen] = False
                        remaining_capacity[cell] -= take
                        remaining_vacancies[group] -= take
        detail["birth_allocation"] = capacity-remaining_capacity
        detail["unfilled_repairs"] = int(available.sum())
        detail["assignment_policy"] = "nearest_reference_group_regret_then_capacity"
        children, parents, candidates, masks, anchor_rows, anchor_cells = [], [], [], [], [], []
        for cell in range(self.cells):
            selected_child = (assignments == cell).nonzero().flatten()
            count = len(selected_child)
            if not count:
                continue
            slots = balls[cell].nonzero().flatten()
            priority = torch.rand(len(slots), device=self.device, dtype=torch.float64,
                                  generator=generator)
            selected_slot = slots[priority.argsort(stable=True)[:count]]
            candidate = pool[cell].expand(count, -1).clone()
            mask = balls[cell].expand(count, -1).clone()
            order = torch.arange(count, device=self.device)
            previous = (order[:, None] > order[None]).nonzero()
            if len(previous):
                mask[previous[:, 0], selected_slot[previous[:, 1]]] = False
            children.append(dead[selected_child])
            parents.append(pool[cell, selected_slot])
            candidates.append(candidate)
            masks.append(mask)
            anchor_rows.append(self.real_representative_rows[cell].expand(count))
            anchor_cells.append(torch.full((count,), cell, device=self.device, dtype=torch.long))
        child, parent = torch.cat(children), torch.cat(parents)
        detail.update(candidate_ids=torch.cat(candidates), candidate_mask=torch.cat(masks),
                      anchor_reference_rows=torch.cat(anchor_rows),
                      anchor_cell_ids=torch.cat(anchor_cells), parent_cell_ids=ids[parent],
                      parent_cell_deficits=vacancies[groups[ids[parent]]],
                      remaining_group_vacancies=remaining_vacancies)
        return child, parent, detail

    # Count tests --------------------------------------------------------------

    @torch.no_grad()
    def cell_comparison(self, fake_features):
        fake_categories = self.count_categories(fake_features)
        fake_counts = torch.bincount(fake_categories, minlength=2*self.cells)
        real_counts = self.real_calibration_category_counts
        rows, fake_rows = self.calibration_rows, len(fake_features)
        multiplicity = 3*self.cells+3
        cutoff = Q/multiplicity

        def family(real, fake):
            return count_family(real, fake, rows, fake_rows, cutoff, self.valid_metric, self.work)

        support = family(real_counts, fake_counts)
        # K original cells and 2K refinements are overlapping hypotheses,
        # not a single disjoint sample with twice the actual number of rows.
        mass = family(self.real_calibration_counts, fake_counts.reshape(self.cells, 2).sum(1))
        global_support = family(real_counts.reshape(self.cells, 2).sum(0),
                                fake_counts.reshape(self.cells, 2).sum(0))
        return dict(mass=mass, support=support, global_support=global_support,
                    pvalues=torch.cat((mass["pvalues"], support["pvalues"],
                                       global_support["pvalues"])),
                    categories=2*self.cells, multiplicity=multiplicity, cutoff=cutoff,
                    family_sizes=(self.cells, 2*self.cells, 2, 1),
                    count_partition=dict(self.count_partition),
                    count_boundary=self.count_boundary.clone())

    # Ordinary transport -------------------------------------------------------

    @torch.no_grad()
    def _ordinary_mass_transport(self, query_features, flags, comparison, *, generator,
                                 pvalues=None, max_moves=None):
        """One count-certified quota allocation; no inherited sequential evidence."""
        return plan_ordinary_transport(self, MASS_TRANSPORT, query_features, flags, comparison,
                                       generator=generator, pvalues=pvalues, max_moves=max_moves)

    @torch.no_grad()
    def _ordinary_support_transport(self, query_features, flags, comparison, *, generator,
                                    pvalues=None, max_moves=None, reserved_children=None,
                                    reserved_parents=None, supported_counts=None):
        """Certified outside-to-inside recovery, with supported mass reserved."""
        return plan_ordinary_transport(self, SUPPORT_TRANSPORT, query_features, flags, comparison,
                                       generator=generator, pvalues=pvalues, max_moves=max_moves,
                                       reserved_children=reserved_children,
                                       reserved_parents=reserved_parents,
                                       supported_counts=supported_counts)

    @torch.no_grad()
    def _ordinary_global_transport(self, query_features, flags, comparison, *, generator,
                                   pvalues=None, max_moves=None, reserved_children=None,
                                   reserved_parents=None, supported_counts=None):
        """Aggregate support recovery after gross prior certificate reservations."""
        return plan_ordinary_transport(self, GLOBAL_TRANSPORT, query_features, flags, comparison,
                                       generator=generator, pvalues=pvalues, max_moves=max_moves,
                                       reserved_children=reserved_children,
                                       reserved_parents=reserved_parents,
                                       supported_counts=supported_counts)

    @torch.no_grad()
    def ordinary_transport(self, query_features, flags, comparison, *, generator, pvalues=None,
                           max_moves=None, birth_planner=None):
        """Mass, local support, then aggregate support under one ordinary budget."""
        multiplicity = 3*self.cells+3
        if (comparison.get("multiplicity") != multiplicity
                or comparison.get("cutoff") != Q/multiplicity
                or comparison["mass"]["difference"].shape != (self.cells,)
                or comparison["support"]["difference"].shape != (2*self.cells,)
                or comparison["global_support"]["difference"].shape != (2,)):
            raise ValueError("joint transport requires the common 3K+3 comparison")
        self._input(query_features)
        rows = len(query_features)
        check_flags(flags, rows, self.device)
        if pvalues is None:
            _, pvalues, _ = self.support(query_features)
        budget = max(0, move_budget(rows, max_moves))
        empty = torch.empty(0, device=self.device, dtype=torch.long)

        def bincount(rows):
            return torch.bincount(ids[rows], minlength=self.cells)

        mass_child, mass_parent, mass = self._ordinary_mass_transport(
            query_features, flags, comparison["mass"], generator=generator, pvalues=pvalues,
            max_moves=budget)
        ids, _ = self.assign(query_features)
        clean = torch.bincount(ids[~flags], minlength=self.cells)
        after_mass = clean-bincount(mass_child[~flags[mass_child]])
        after_mass += bincount(mass_parent)
        remaining = budget-len(mass_child)
        support_comparison = comparison["support"]
        possible = (remaining > 0 and bool(flags.any())
                    and bool(support_comparison["excess"][1::2].any())
                    and bool(support_comparison["deficit"][0::2].any()))
        if possible:
            support_child, support_parent, support = self._ordinary_support_transport(
                query_features, flags, support_comparison, generator=generator, pvalues=pvalues,
                max_moves=remaining, reserved_children=mass_child, reserved_parents=mass_parent,
                supported_counts=after_mass)
            support["ran"] = True
        else:
            support_child, support_parent = empty, empty
            support = _empty_phase(self, remaining, after_mass)
        child, parent = torch.cat((mass_child, support_child)), torch.cat((mass_parent, support_parent))
        categories = self.count_categories(query_features)
        after_local = after_mass+bincount(support_parent)
        birth_plan = None if birth_planner is None else birth_planner(child, parent, after_local, budget)
        birth_rows = empty if birth_plan is None else birth_plan["children"]
        remaining = budget-len(child)-len(birth_rows)
        global_comparison = comparison["global_support"]
        possible = (remaining > 0 and bool(flags.any()) and bool(global_comparison["excess"][1])
                    and bool(global_comparison["deficit"][0]))
        if birth_plan is not None:
            global_child, global_parent, global_phase = plan_residual_global_copies(
                self, query_features, flags, pvalues, comparison, birth_plan, generator=generator,
                previous_children=child, previous_copy_parents=parent)
        elif possible:
            global_child, global_parent, global_phase = self._ordinary_global_transport(
                query_features, flags, global_comparison, generator=generator, pvalues=pvalues,
                max_moves=remaining, reserved_children=child, reserved_parents=parent,
                supported_counts=after_local)
            global_phase["ran"] = True
        else:
            global_child, global_parent = empty, empty
            global_phase = _empty_phase(self, remaining, after_local)
        child, parent = torch.cat((child, global_child)), torch.cat((parent, global_parent))
        supported_before_global = (after_local if birth_plan is None
                                   else birth_plan["planned_supported_counts"])
        planned = supported_before_global+bincount(global_parent)
        phases = (mass, support, global_phase)

        def total(key):
            return phases[0][key]+phases[1][key]+phases[2][key]

        detail = dict(
            excess_cells=(mass["excess_cells"]+int(support_comparison["excess"].sum())
                          + int(global_comparison["excess"].sum())),
            deficit_cells=(mass["deficit_cells"]+int(support_comparison["deficit"].sum())
                           + int(global_comparison["deficit"].sum())),
            discoveries=(mass["discoveries"]
                         + int((support_comparison["excess"] | support_comparison["deficit"]).sum())
                         + int((global_comparison["excess"] | global_comparison["deficit"]).sum())),
            inaccessible_birth_quota=total("inaccessible_birth_quota"),
            budget=budget, moves=len(child), mass_moves=len(mass_child),
            support_moves=len(support_child), global_moves=len(global_child),
            mass_phase=mass, support_phase=support, mass_children=mass_child,
            mass_parents=mass_parent, support_children=support_child,
            support_parents=support_parent, global_phase=global_phase,
            global_children=global_child, global_parents=global_parent,
            action_kinds=torch.cat((torch.zeros_like(mass_child), torch.ones_like(support_child),
                                    torch.full_like(global_child, 2))),
            death_allocation=total("death_allocation"), birth_allocation=total("birth_allocation"),
            eligible_parent_counts=mass["eligible_parent_counts"],
            target_counts=mass["target_counts"], clean_counts=clean,
            table_counts=mass["table_counts"], planned_supported_counts=planned,
            supported_after_mass=after_mass, supported_after_local=after_local,
            unique_parent_count=len(torch.unique(parent)), mass_topology=dict(self.mass_topology),
            group_target_counts=mass["group_target_counts"],
            group_clean_counts=self._group_counts(clean),
            within_group_moves=total("within_group_moves"),
            between_group_moves=total("between_group_moves"),
            isolation_guard_rejects=mass["isolation_guard_rejects"],
            ordinary_flagged_deaths=int(flags[child].sum()),
            death_policy="v4_mass_then_local_then_global_certified_flagged_outside",
            multiplicity=multiplicity, cutoff=Q/multiplicity, query_cell_ids=ids,
            query_category_ids=categories, child_category_ids=categories[child],
            parent_category_ids=categories[parent], count_partition=dict(self.count_partition),
            count_boundary=self.count_boundary.clone())
        if birth_plan is not None:
            source_ids = ids[birth_rows]
            destination_ids = birth_plan["target_cell_ids"]
            topology = self._mass_topology()
            same_group = int((topology[source_ids] == topology[destination_ids]).sum())
            detail.update(
                novel_birth_plan=birth_plan, novel_birth_moves=len(birth_rows),
                moves=len(child)+len(birth_rows), copy_moves=len(child),
                death_policy="paired_copy_and_even_real_anchor_birth_shared_3K_plus_3",
                within_group_moves=detail["within_group_moves"]+same_group,
                between_group_moves=detail["between_group_moves"]+len(birth_rows)-same_group,
                ordinary_flagged_deaths=detail["ordinary_flagged_deaths"]+len(birth_rows),
                death_allocation=detail["death_allocation"]+torch.bincount(source_ids,
                                                                           minlength=self.cells),
                birth_allocation=detail["birth_allocation"]+torch.bincount(destination_ids,
                                                                           minlength=self.cells),
                action_children=torch.cat((child, birth_rows)),
                action_destination_category_ids=torch.cat((categories[parent],
                                                           birth_plan["destination_category_ids"])),
                action_kinds=torch.cat((detail["action_kinds"], torch.full_like(birth_rows, 3))),
                supported_after_births=supported_before_global)
        return child, parent, detail


def paired_average_geometry(snapshot, ema_features, *, step, snapshot_serial,
                            finite_coordinates=None):
    """Current clean anti-blur geometry, not stationarity or quality evidence.

    The FAST cache must contain every applied copy and novel birth. No random
    draws or count tests are added. One joint Q-sized exception set covers EMA
    support, the even-fitted inside region and paired real-topology agreement.
    """
    if snapshot.row_features is None or snapshot.query_cell_ids is None:
        raise ValueError("paired-average geometry needs the refreshed FAST cache")
    n = len(snapshot.row_features)
    if (ema_features.ndim != 2 or len(ema_features) != n
            or ema_features.shape[1] != snapshot.width or ema_features.device != snapshot.device):
        raise ValueError("paired-average EMA features do not match the current chart")
    finite = torch.isfinite(snapshot.row_features).all(1) & torch.isfinite(ema_features).all(1)
    if finite_coordinates is not None:
        if (finite_coordinates.shape != (n,) or finite_coordinates.dtype != torch.bool
                or finite_coordinates.device != snapshot.device):
            raise ValueError("invalid paired-average finite-coordinate mask")
        finite &= finite_coordinates
    stamp = dict(schema=1, policy="ema_support_inside_current_real_group_Q_v1",
                 snapshot=snapshot_serial, step=step, rows=n, required=n-math.floor(Q*n),
                 cells=snapshot.cells, rank=snapshot.rank, groups=0,
                 calibration_rows=snapshot.calibration_rows,
                 chart_valid=bool(snapshot.valid_metric),
                 duplicate_ok=bool(snapshot.duplicate_fraction <= Q),
                 finite_rows=int(finite.sum()), same_group_rows=0, ema_eligible_rows=0,
                 coherent_rows=0, eligible=False)
    if not bool(finite.all()):
        return stamp
    groups = snapshot._mass_topology()
    metric = snapshot.transform(ema_features)
    categories, scores = snapshot._count_categories_metric(metric)
    null = snapshot.null_scores
    pvalues = (1.+len(null)-torch.searchsorted(null, scores))/(1.+len(null))
    same_group = groups[snapshot.query_cell_ids] == groups[categories//2]
    eligible = (pvalues > Q) & (categories.remainder(2) == 0)
    stamp.update(groups=snapshot.mass_groups, same_group_rows=int(same_group.sum()),
                 ema_eligible_rows=int(eligible.sum()),
                 coherent_rows=int((same_group & eligible).sum()))
    stamp["eligible"] = bool(stamp["chart_valid"] and stamp["duplicate_ok"]
                             and stamp["coherent_rows"] >= stamp["required"])
    return stamp
