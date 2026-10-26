"""Ordinary count-certified feature-cell transport: one planner, three policies.

The mass (K cells), local-support (2K categories) and global-support (2
regions) phases share one pipeline:

1. build the same-cell supported parent pool (one priority draw);
2. turn count certificates into death/birth capacities (policy hook);
3. limit them by actual rows, vacancies and parent supply (policy hook);
4. allocate pairs within real-topology groups, then between groups;
5. select rows in fixed cell order (one priority draw, then per-cell
   ``randperm`` parent draws).

Policies supply only what differs. The random draws, their order and every
arithmetic step are those of the three historical planners, so reaction
trajectories are bit-identical.
"""
from types import SimpleNamespace

import torch

from .cell_statistics import (Q, _group_integer_allocate, _integer_allocate,
                              certified_capacity, check_flags, move_budget)


class TransportPolicy:
    """Hooks distinguishing one ordinary transport phase."""
    #: Whether the phase reads the fixed 2K count categories and reservations.
    categorical = True

    def certify(self, plan):
        """Set ``plan.death_capacity``/``plan.birth_capacity`` from evidence."""
        raise NotImplementedError

    def limit(self, plan):
        """Return (death_eligible, death_capacity, birth_capacity) from rows."""
        raise NotImplementedError

    def conserves_group_vacancies(self, plan):
        """Whether both allocation phases are bounded by group vacancies."""
        return True

    def allocation_budget(self, plan):
        return max(0, plan.budget)

    def death_policy(self, plan):
        raise NotImplementedError

    def details(self, plan, child, parent):
        return {}


class MassTransportPolicy(TransportPolicy):
    """K-cell mass recovery; supported surplus or certified flagged deaths."""
    categorical = False

    def certify(self, plan):
        difference = plan.comparison["difference"]
        plan.death_capacity = certified_capacity(plan.rows, difference, plan.comparison["excess"])
        plan.birth_capacity = certified_capacity(plan.rows, -difference, plan.comparison["deficit"])

    def limit(self, plan):
        snapshot, ids, flags = plan.snapshot, plan.ids, plan.flags
        # A small flagged set remains reserved for the unchanged isolation
        # guard. During widespread isolation, ordinary count evidence can
        # certify replacing a flagged row within its separate fixed budget.
        # Supported survivors always reserve their mass and never supply a
        # death in this recovery branch.
        flagged_counts = plan.table_counts-plan.clean_counts
        excess = (plan.clean_counts-plan.target).clamp_min(0)
        # Emitted iid count evidence certifies direction. Actual clean rows
        # determine how much mass can move, avoiding repeated moves generated
        # solely by emission/pool fluctuation after the table is balanced.
        physical = flagged_counts if plan.guard_rejects else excess
        death_capacity = torch.minimum(plan.death_capacity, physical)
        birth_capacity = torch.minimum(plan.birth_capacity, plan.vacancies)
        # Retain an actual eligible parent, rather than merely leaving an
        # arbitrary nonflagged row that might itself have p<=Q.
        protected = torch.full((snapshot.cells,), -1, device=snapshot.device, dtype=torch.long)
        retain = (plan.target > 0) & (plan.pool_counts > 0)
        protected[retain] = plan.pool[retain, 0]
        if plan.guard_rejects:
            death_eligible = flags
        else:
            positions = torch.arange(plan.rows, device=snapshot.device)
            death_eligible = ~flags & (positions != protected[ids])
        available_deaths = torch.bincount(ids[death_eligible], minlength=snapshot.cells)
        death_capacity = torch.minimum(death_capacity, available_deaths)
        return death_eligible, death_capacity, birth_capacity

    def conserves_group_vacancies(self, plan):
        return plan.guard_rejects

    def death_policy(self, plan):
        return "count_certified_flagged_only" if plan.guard_rejects else "supported_surplus"


def _spent_reserved(plan, rows, region, fallback, *, global_regions):
    """Gross certificate use by reserved rows of one count-category parity.

    ``global_regions`` pools the use over all cells (one scalar per region)
    instead of returning a per-cell count.
    """
    if rows is None:
        return torch.zeros_like(fallback)
    categories = plan.categories[rows]
    if global_regions:
        return (categories.remainder(2) == region).sum()
    counts = torch.bincount(categories, minlength=2*plan.snapshot.cells)
    return counts.reshape(plan.snapshot.cells, 2)[:, region]


class SupportTransportPolicy(TransportPolicy):
    """Certified outside-to-inside recovery, with supported mass reserved."""
    global_regions = False

    def certify(self, plan):
        snapshot, comparison = plan.snapshot, plan.comparison
        difference = comparison["difference"].reshape(snapshot.cells, 2)
        outside_excess = comparison["excess"].reshape(snapshot.cells, 2)[:, 1]
        inside_deficit = comparison["deficit"].reshape(snapshot.cells, 2)[:, 0]
        raw_death = certified_capacity(plan.rows, difference[:, 1], outside_excess)
        raw_birth = certified_capacity(plan.rows, -difference[:, 0], inside_deficit)
        self._reserve(plan, raw_death, raw_birth)
        # Reserve gross certificate use by the first phase as well as row
        # identities. Opposite-direction moves do not replenish this budget.
        plan.death_capacity = plan.residual_death_capacity.clone()
        plan.birth_capacity = plan.residual_birth_capacity.clone()

    def _reserve(self, plan, raw_death, raw_birth):
        plan.raw_death_capacity, plan.raw_birth_capacity = raw_death, raw_birth
        plan.spent_deaths = _spent_reserved(
            plan, plan.reserved_children, 1, raw_death, global_regions=self.global_regions)
        plan.spent_births = _spent_reserved(
            plan, plan.reserved_parents, 0, raw_birth, global_regions=self.global_regions)
        plan.residual_death_capacity = (raw_death-plan.spent_deaths).clamp_min(0)
        plan.residual_birth_capacity = (raw_birth-plan.spent_births).clamp_min(0)

    def _death_eligible(self, plan):
        # Only flagged outside rows can supply ordinary deaths. Even under a
        # passing isolation guard they do not reserve supported mass.
        return plan.flags & ~plan.inside & ~plan.reserved

    def limit(self, plan):
        birth_capacity = torch.minimum(plan.birth_capacity, plan.vacancies)
        death_eligible = self._death_eligible(plan)
        available_deaths = torch.bincount(plan.ids[death_eligible], minlength=plan.snapshot.cells)
        return death_eligible, torch.minimum(plan.death_capacity, available_deaths), birth_capacity

    def death_policy(self, plan):
        return "count_certified_flagged_outside_only"

    def _category_details(self, plan, child, parent):
        categories, comparison = plan.categories, plan.comparison
        detail = dict(count_partition=dict(plan.snapshot.count_partition),
                      count_boundary=plan.snapshot.count_boundary.clone(),
                      query_category_ids=categories, child_category_ids=categories[child],
                      parent_category_ids=categories[parent])
        if self.global_regions:
            detail.update(child_region_ids=categories[child].remainder(2),
                          parent_region_ids=categories[parent].remainder(2),
                          death_certified_regions=comparison["excess"],
                          birth_certified_regions=comparison["deficit"])
        else:
            detail.update(death_certified_categories=comparison["excess"],
                          birth_certified_categories=comparison["deficit"])
        minlength = 2*plan.snapshot.cells
        detail.update(
            death_category_allocation=torch.bincount(categories[child], minlength=minlength),
            birth_category_allocation=torch.bincount(categories[parent], minlength=minlength),
            raw_certified_death_capacity=plan.raw_death_capacity,
            raw_certified_birth_capacity=plan.raw_birth_capacity,
            spent_certified_death_capacity=plan.spent_deaths,
            spent_certified_birth_capacity=plan.spent_births,
            residual_certified_death_capacity=plan.residual_death_capacity,
            residual_certified_birth_capacity=plan.residual_birth_capacity)
        return detail

    def details(self, plan, child, parent):
        return self._category_details(plan, child, parent)


class GlobalTransportPolicy(SupportTransportPolicy):
    """Aggregate support recovery after gross prior certificate reservations.

    Global evidence supplies the direction and a scalar certified budget;
    actual cell/group vacancies and unused inside parents supply physical
    birth capacity. Flagged outside deaths do not remove any supported mass.
    """
    global_regions = True

    def certify(self, plan):
        comparison = plan.comparison
        difference = comparison["difference"]
        raw_death = certified_capacity(plan.rows, difference[1], comparison["excess"][1])
        raw_birth = certified_capacity(plan.rows, -difference[0], comparison["deficit"][0])
        # Both previous phases spend these aggregate certificates, including
        # supported mass children outside the fixed fitted boundary. Count
        # gross use; an opposite action never restores certificate capacity.
        self._reserve(plan, raw_death, raw_birth)

    def limit(self, plan):
        death_eligible = self._death_eligible(plan)
        available_deaths = torch.bincount(plan.ids[death_eligible], minlength=plan.snapshot.cells)
        return death_eligible, available_deaths, plan.vacancies.clone()

    def allocation_budget(self, plan):
        plan.certified_budget = max(0, min(plan.budget, int(plan.residual_death_capacity),
                                           int(plan.residual_birth_capacity)))
        return plan.certified_budget

    def death_policy(self, plan):
        return "global_count_certified_flagged_outside_only"

    def details(self, plan, child, parent):
        return dict(certified_budget=plan.certified_budget,
                    **self._category_details(plan, child, parent))


MASS_TRANSPORT = MassTransportPolicy()
SUPPORT_TRANSPORT = SupportTransportPolicy()
GLOBAL_TRANSPORT = GlobalTransportPolicy()


def _allocate_pairs(snapshot, plan, death_capacity, birth_capacity, conserve, limit):
    """Within-group pairs first, then the explicit residual cross-group phase."""
    groups = snapshot._mass_topology()
    group_counts = snapshot._group_counts
    group_target, group_clean = group_counts(plan.target), group_counts(plan.clean_counts)
    within_capacity = torch.minimum(group_counts(death_capacity), group_counts(birth_capacity))
    if conserve:
        within_capacity = torch.minimum(within_capacity, (group_target-group_clean).clamp_min(0))
    within = _integer_allocate(within_capacity, limit)
    within_deaths = _group_integer_allocate(death_capacity, groups, within)
    within_births = _group_integer_allocate(birth_capacity, groups, within)
    remaining_death, remaining_birth = death_capacity-within_deaths, birth_capacity-within_births
    if conserve:
        group_death = group_counts(remaining_death)
        remaining_vacancies = (group_target-group_clean-group_counts(within_births)).clamp_min(0)
        group_birth = torch.minimum(group_counts(remaining_birth), remaining_vacancies)
    else:
        group_death = torch.minimum(group_counts(remaining_death),
                                    (group_clean-group_target).clamp_min(0))
        group_birth = torch.minimum(group_counts(remaining_birth),
                                    (group_target-group_clean).clamp_min(0))
    cross_pairs = min(int(group_death.sum()), int(group_birth.sum()),
                      max(0, limit-int(within.sum())))
    cross_group_death = _integer_allocate(group_death, cross_pairs)
    cross_group_birth = _integer_allocate(group_birth, cross_pairs)
    cross_deaths = _group_integer_allocate(remaining_death, groups, cross_group_death)
    cross_births = _group_integer_allocate(remaining_birth, groups, cross_group_birth)
    return SimpleNamespace(groups=groups, group_target=group_target, group_clean=group_clean,
                           within=within, within_deaths=within_deaths, within_births=within_births,
                           deaths=within_deaths+cross_deaths, births=within_births+cross_births,
                           cross_pairs=cross_pairs)


def _select_rows(snapshot, plan, allocation, death_eligible, generator):
    """Fixed cell-order child/parent draws; within-group pairs come first."""
    a = allocation
    # One bounded metadata transfer preserves the original cell/RNG order.
    death_rows, birth_rows, within_death_rows, within_birth_rows, pool_rows, group_rows = torch.stack(
        (a.deaths, a.births, a.within_deaths, a.within_births, plan.pool_counts, a.groups)).cpu().tolist()
    group_cells = [[] for _ in range(snapshot.mass_groups)]
    for cell, group in enumerate(group_rows):
        group_cells[group].append(cell)
    priorities = torch.rand(plan.rows, device=snapshot.device, dtype=torch.float64,
                            generator=generator)
    empty = torch.empty(0, device=snapshot.device, dtype=torch.long)
    children_by_cell, parents_by_cell = [], []
    for cell in range(snapshot.cells):
        count = death_rows[cell]
        if count:
            rows = ((plan.ids == cell) & death_eligible).nonzero().flatten()
            children_by_cell.append(rows[priorities[rows].argsort(stable=True)[:count]])
        else:
            children_by_cell.append(empty)
        count = birth_rows[cell]
        if count:
            draw = torch.randperm(pool_rows[cell], device=snapshot.device, generator=generator)
            parents_by_cell.append(plan.pool[cell, draw[:count]])
        else:
            parents_by_cell.append(empty)
    children, parents = [], []
    # Match the conservative within-neighborhood phase inside its group.
    # Only the explicit residual phase can move mass between groups.
    for group in range(snapshot.mass_groups):
        for index in group_cells[group]:
            children.append(children_by_cell[index][:within_death_rows[index]])
            parents.append(parents_by_cell[index][:within_birth_rows[index]])
    for cell in range(snapshot.cells):
        children.append(children_by_cell[cell][within_death_rows[cell]:])
        parents.append(parents_by_cell[cell][within_birth_rows[cell]:])
    child = torch.cat(children) if children else empty
    parent = torch.cat(parents) if parents else empty
    return child, parent


@torch.no_grad()
def plan_ordinary_transport(snapshot, policy, query_features, flags, comparison, *, generator,
                            pvalues=None, max_moves=None, reserved_children=None,
                            reserved_parents=None, supported_counts=None):
    """One ordinary phase: paired (child, parent) copy rows and diagnostics."""
    features = snapshot.transform(query_features)
    rows = len(features)
    check_flags(flags, rows, snapshot.device)
    if pvalues is None:
        _, pvalues, _ = snapshot.support(query_features)
    plan = SimpleNamespace(snapshot=snapshot, rows=rows, flags=flags, comparison=comparison,
                           reserved_children=reserved_children, reserved_parents=reserved_parents)
    eligible = ~flags & (pvalues > Q)
    if policy.categorical:
        plan.categories, _ = snapshot._count_categories_metric(features)
        plan.inside = plan.categories.remainder(2) == 0
        plan.reserved = torch.zeros(rows, device=snapshot.device, dtype=torch.bool)
        if reserved_children is not None:
            plan.reserved[reserved_children] = True
        if reserved_parents is not None:
            plan.reserved[reserved_parents] = True
        eligible = eligible & plan.inside & ~plan.reserved
    plan.ids, plan.pool, plan.pool_counts, eligible_counts = snapshot._pool(features, eligible, generator)
    policy.certify(plan)
    plan.target = snapshot._mass_targets(rows)
    plan.table_counts = torch.bincount(plan.ids, minlength=snapshot.cells)
    plan.budget = move_budget(rows, max_moves)
    plan.guard_rejects = int(flags.sum()) > Q*rows
    plan.clean_counts = (torch.bincount(plan.ids[~flags], minlength=snapshot.cells)
                         if supported_counts is None else supported_counts.clone())
    plan.vacancies = (plan.target-plan.clean_counts).clamp_min(0)
    death_eligible, death_capacity, birth_capacity = policy.limit(plan)
    inaccessible = birth_capacity[plan.pool_counts == 0].sum()
    birth_capacity = torch.minimum(birth_capacity, plan.pool_counts)
    allocation = _allocate_pairs(snapshot, plan, death_capacity, birth_capacity,
                                 policy.conserves_group_vacancies(plan), policy.allocation_budget(plan))
    child, parent = _select_rows(snapshot, plan, allocation, death_eligible, generator)
    detail = dict(excess_cells=int(comparison["excess"].sum()),
                  deficit_cells=int(comparison["deficit"].sum()),
                  discoveries=int((comparison["excess"] | comparison["deficit"]).sum()),
                  inaccessible_birth_quota=int(inaccessible), budget=plan.budget, moves=len(child),
                  death_allocation=allocation.deaths, birth_allocation=allocation.births,
                  eligible_parent_counts=eligible_counts, target_counts=plan.target,
                  clean_counts=plan.clean_counts, table_counts=plan.table_counts,
                  clean_vacancies=plan.vacancies, unique_parent_count=len(torch.unique(parent)),
                  mass_topology=dict(snapshot.mass_topology),
                  group_target_counts=allocation.group_target,
                  group_clean_counts=allocation.group_clean,
                  within_group_moves=int(allocation.within.sum()),
                  between_group_moves=allocation.cross_pairs)
    supported_children = child[~flags[child]]
    detail.update(isolation_guard_rejects=plan.guard_rejects,
                  ordinary_flagged_deaths=int(flags[child].sum()),
                  death_policy=policy.death_policy(plan))
    detail.update(policy.details(plan, child, parent))
    ids = plan.ids
    detail["planned_supported_counts"] = (
        plan.clean_counts-torch.bincount(ids[supported_children], minlength=snapshot.cells)
        + torch.bincount(ids[parent], minlength=snapshot.cells))
    return child, parent, detail
