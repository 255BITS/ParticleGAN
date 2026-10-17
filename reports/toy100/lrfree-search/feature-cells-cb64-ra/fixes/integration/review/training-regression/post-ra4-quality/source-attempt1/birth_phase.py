"""Bounded new-latent births under the existing 3K+2 count certificates.

Run after mass/local copies are planned and before aggregate support copies.
Source rows are solver seeds, never supported-copy parents. Later phases must
reserve both seeds and children, and use actual birth destination categories.
"""
import math
import torch
try:
    from .anchor_birth import propose_inaccessible_anchors
except ImportError:
    from anchor_birth import propose_inaccessible_anchors


def _rows(rows, n, device):
    if rows is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.dtype != torch.long
            or rows.device != device or bool(((rows < 0) | (rows >= n)).any())
            or len(torch.unique(rows)) != len(rows)):
        raise ValueError('invalid reserved birth rows')
    return rows


def global_certificate_residual(snapshot, query_features, comparison, *, previous_children=None,
        previous_birth_categories=None):
    """Gross, nonreplenishing reservations use destinations rather than seeds."""
    if (comparison.get('multiplicity') != 3 * snapshot.cells + 2
            or comparison.get('cutoff') != .05 / (3 * snapshot.cells + 2)
            or comparison.get('family_sizes') != (snapshot.cells, 2 * snapshot.cells, 2)):
        raise ValueError('birth requires the unchanged common 3K+2 count family')
    n = len(query_features)
    categories = snapshot.count_categories(query_features)
    children = _rows(previous_children, n, categories.device)
    destinations = torch.empty(0, dtype=torch.long, device=categories.device) if previous_birth_categories is None else previous_birth_categories
    if (destinations.ndim != 1 or destinations.dtype != torch.long or destinations.device != categories.device
            or len(destinations) != len(children) or bool(((destinations < 0) | (destinations >= 2 * snapshot.cells)).any())):
        raise ValueError('each earlier action needs its actual birth category')
    global_law = comparison['global_support']
    if global_law['difference'].shape != (2,):
        raise ValueError('invalid global support law')
    raw_death = torch.floor(n * global_law['difference'][1].clamp_min(0) + 1e-10).long() * global_law['excess'][1]
    raw_birth = torch.floor(n * (-global_law['difference'][0]).clamp_min(0) + 1e-10).long() * global_law['deficit'][0]
    spent_death = (categories[children].remainder(2) == 1).sum()
    spent_birth = (destinations.remainder(2) == 0).sum()
    return dict(raw_death=raw_death, raw_birth=raw_birth, spent_death=spent_death, spent_birth=spent_birth,
        residual_death=(raw_death - spent_death).clamp_min(0),
        residual_birth=(raw_birth - spent_birth).clamp_min(0),
        outside_certified=bool(global_law['excess'][1]), inside_certified=bool(global_law['deficit'][0]),
        multiplicity=comparison['multiplicity'], cutoff=comparison['cutoff'])


def _accepted(snapshot, value, cell):
    if value is None or not value.get('accepted', False):
        return False
    features, latent = value['features'], value['latent']
    if features.ndim != 2 or len(features) != 1 or not bool(torch.isfinite(latent).all()):
        return False
    flags, pvalues, _ = snapshot.support(features)
    category = snapshot.count_categories(features)
    return bool(not flags[0] and pvalues[0] > .05 and category[0] == 2 * cell)


def allocate_anchor_births(snapshot, query_features, flags, pvalues, comparison, attempts, *,
        previous_children=None, previous_copy_parents=None, previous_birth_categories=None,
        supported_counts=None, max_moves=None):
    """At most four distinct accepted anchor births, sharing the ordinary cap.

    Input supported_counts is the exact ledger after preceding mass/local
    actions. The caller supplies prior destination categories explicitly.
    New deaths are flagged outside rows, so no supported mass is removed.
    """
    n = len(query_features)
    ids, _ = snapshot.assign(query_features)
    categories = snapshot.count_categories(query_features)
    if flags.shape != (n,) or flags.dtype != torch.bool or pvalues.shape != (n,):
        raise ValueError('invalid birth support table')
    prior_children = _rows(previous_children, n, ids.device)
    prior_parents = _rows(previous_copy_parents, n, ids.device)
    if len(prior_children) != len(prior_parents) or bool(torch.isin(prior_children, prior_parents).any()):
        raise ValueError('preceding copies must be distinct paired actions')
    if previous_birth_categories is None:
        previous_birth_categories = categories[prior_parents]
    certificate = global_certificate_residual(snapshot, query_features, comparison,
        previous_children=prior_children, previous_birth_categories=previous_birth_categories)
    initial = torch.bincount(ids[~flags], minlength=snapshot.cells)
    expected = initial - torch.bincount(ids[prior_children[~flags[prior_children]]], minlength=snapshot.cells)
    expected += torch.bincount(ids[prior_parents], minlength=snapshot.cells)
    supported = expected if supported_counts is None else supported_counts.clone()
    if supported.shape != (snapshot.cells,) or not torch.equal(supported, expected):
        raise ValueError('birth requires the exact supported ledger after earlier copies')
    total_budget = math.floor(.05 * n) if max_moves is None else max(0, min(int(max_moves), math.floor(.05 * n)))
    remaining = max(0, total_budget - len(prior_children))
    capacity = min(4, remaining, int(certificate['residual_death']), int(certificate['residual_birth']))
    empty = torch.empty(0, dtype=torch.long, device=ids.device)
    reserved = torch.zeros(n, dtype=torch.bool, device=ids.device)
    reserved[prior_children] = True
    reserved[prior_parents] = True
    inside = categories.remainder(2) == 0
    eligible = ~flags & (pvalues > .05) & inside & ~reserved
    pool_counts = torch.bincount(ids[eligible], minlength=snapshot.cells)
    targets = snapshot._mass_targets(n)
    groups = snapshot._mass_topology()
    planned = supported.clone()
    children, seeds, cells, latents, ema_latents, accepted_attempts = [], [], [], [], [], []
    # Reserve all attempted solver sources before choosing a donor. No source
    # used to construct an accepted birth can be erased later in this phase.
    attempt_sources = [int(a['seed_row']) for a in attempts]
    if len(attempts) > 4 or len(set(attempt_sources)) != len(attempt_sources):
        raise ValueError('birth proposals exceed distinct four-source work bound')
    for source in attempt_sources:
        if not 0 <= source < n or bool(reserved[source]):
            raise ValueError('birth source overlaps an earlier action')
    protected_sources = reserved.clone()
    if attempt_sources:
        protected_sources[torch.tensor(attempt_sources, device=ids.device)] = True
    donor_rows = (flags & ~inside & ~protected_sources).nonzero().flatten()
    if snapshot.valid_metric and certificate['outside_certified'] and certificate['inside_certified']:
        for attempt in attempts:
            if len(children) >= capacity or not len(donor_rows):
                break
            cell, source = int(attempt['cell']), int(attempt['seed_row'])
            if not 0 <= cell < snapshot.cells or cell in cells:
                continue
            vacancies = (targets - planned).clamp_min(0)
            group_vacancies = (snapshot._group_counts(targets) - snapshot._group_counts(planned)).clamp_min(0)
            if (not bool(attempt.get('accepted', False)) or pool_counts[cell] != 0
                    or snapshot.reference_counts[cell] <= 0 or vacancies[cell] <= 0
                    or group_vacancies[groups[cell]] <= 0 or not _accepted(snapshot, attempt['current'], cell)
                    or not _accepted(snapshot, attempt['average'], cell)):
                continue
            child = donor_rows[len(children)]
            children.append(child); seeds.append(source); cells.append(cell)
            latents.append(attempt['current']['latent'].detach().clone())
            ema_latents.append(attempt['average']['latent'].detach().clone())
            accepted_attempts.append(attempt)
            planned[cell] += 1
    child_rows = torch.stack(children) if children else empty
    seed_rows = torch.tensor(seeds, dtype=torch.long, device=ids.device) if seeds else empty
    target_cells = torch.tensor(cells, dtype=torch.long, device=ids.device) if cells else empty
    born_categories = 2 * target_cells
    residual_after = dict(certificate,
        spent_death=certificate['spent_death'] + len(children),
        spent_birth=certificate['spent_birth'] + len(children),
        residual_death=(certificate['residual_death'] - len(children)).clamp_min(0),
        residual_birth=(certificate['residual_birth'] - len(children)).clamp_min(0))
    return dict(kind='new_latent_birth', children=child_rows, source_seed_rows=seed_rows,
        copy_parent_rows=empty, target_cell_ids=target_cells, destination_category_ids=born_categories,
        death_category_ids=categories[child_rows], source_category_ids=categories[seed_rows],
        new_latents=torch.stack(latents) if latents else None,
        paired_ema_latents=torch.stack(ema_latents) if ema_latents else None,
        attempts=attempts, accepted_attempts=accepted_attempts, attempted_cells=len(attempts), moves=len(children),
        budget=total_budget, earlier_moves=len(prior_children), remaining_budget=remaining,
        remaining_budget_after=remaining - len(children), certificates_before=certificate,
        certificates_after=residual_after, supported_before=supported, planned_supported_counts=planned,
        target_counts=targets, group_counts_before=snapshot._group_counts(supported),
        group_counts_after=snapshot._group_counts(planned),
        reserved_rows_after=torch.cat((prior_children, prior_parents, child_rows, seed_rows)),
        previous_children_after=torch.cat((prior_children, child_rows)),
        previous_birth_categories_after=torch.cat((previous_birth_categories, born_categories)),
        work_bound=dict(cells=4, linearizations_per_model=4, paired_models=2),
        policy='new_latent_even_real_anchor_shared_global_certificates')


def plan_real_anchor_births(snapshot, query_features, flags, pvalues, comparison, latents, feature_of_latent, *,
        ema_latents, ema_feature_of_latent, previous_children=None, previous_copy_parents=None,
        supported_counts=None, max_moves=None):
    """Production entry point: skip solver work without residual certificates."""
    n = len(query_features)
    categories = snapshot.count_categories(query_features)
    child = _rows(previous_children, n, categories.device)
    parent = _rows(previous_copy_parents, n, categories.device)
    if len(child) != len(parent):
        raise ValueError('preceding copy actions must be paired')
    certificate = global_certificate_residual(snapshot, query_features, comparison,
        previous_children=child, previous_birth_categories=categories[parent])
    budget = math.floor(.05 * n) if max_moves is None else max(0, min(int(max_moves), math.floor(.05 * n)))
    limit = max(0, min(4, budget - len(child), int(certificate['residual_death']), int(certificate['residual_birth'])))
    attempts = []
    if snapshot.valid_metric and limit and certificate['outside_certified'] and certificate['inside_certified']:
        attempts = propose_inaccessible_anchors(snapshot, query_features, flags, pvalues, latents, feature_of_latent,
            ema_latents=ema_latents, ema_feature_of_latent=ema_feature_of_latent,
            reserved_rows=torch.cat((child, parent)), supported_counts=supported_counts,
            candidate_limit=limit, passes=4)
        for attempt in attempts:
            for model in ('current', 'average'):
                attempt[model].pop('seconds', None)
    return allocate_anchor_births(snapshot, query_features, flags, pvalues, comparison, attempts,
        previous_children=child, previous_copy_parents=parent, supported_counts=supported_counts, max_moves=max_moves)


def learned_latent_features(birth_death, trainer, generator_model):
    """Differentiable form of the already selected learned scalar-head inputs."""
    if not birth_death._heads:
        raise ValueError('original learned scalar heads must be selected before birth solving')
    def capture(latents):
        fired = {}
        handles = [head.register_forward_pre_hook(lambda module, inputs: fired.__setitem__(id(module), inputs[0]))
            for head in birth_death._heads]
        modes = [(module, module.training) for root in (generator_model, trainer.D) for module in root.modules()]
        try:
            generator_model.eval(); trainer.D.eval()
            raw = generator_model(latents)
            if tuple(raw.shape[1:]) != birth_death.sample_shape:
                raise ValueError('generator output shape differs from the real FIFO')
            trainer.D(raw)
            return torch.cat([fired[id(head)] for head in birth_death._heads], 1)
        finally:
            for handle in handles:
                handle.remove()
            for module, mode in modes:
                module.training = mode
    return capture


def plan_residual_global_copies(snapshot, query_features, flags, pvalues, comparison, birth_plan, *,
        generator, previous_children, previous_copy_parents):
    """Reuse the unchanged global copy planner with honest remaining capacity.

    Source seeds reserve rows without claiming they supplied supported mass.
    The certified max_moves includes actual novel-birth destinations, so the
    legacy planner cannot spend a capacity already consumed by a new birth.
    """
    child = torch.cat((previous_children, birth_plan['children']))
    reserved_parent_rows = torch.cat((previous_copy_parents, birth_plan['source_seed_rows']))
    certificate = birth_plan['certificates_after']
    budget = min(birth_plan['remaining_budget_after'], int(certificate['residual_death']),
        int(certificate['residual_birth']))
    empty = torch.empty(0, dtype=torch.long, device=query_features.device)
    if not snapshot.valid_metric or budget <= 0 or not certificate['outside_certified'] or not certificate['inside_certified']:
        return empty, empty, dict(ran=False, moves=0, budget=budget,
            planned_supported_counts=birth_plan['planned_supported_counts'].clone(),
            all_prior_certificates=certificate)
    next_child, next_parent, detail = snapshot._ordinary_global_transport(query_features, flags,
        comparison['global_support'], generator=generator, pvalues=pvalues, max_moves=budget,
        reserved_children=child, reserved_parents=reserved_parent_rows,
        supported_counts=birth_plan['planned_supported_counts'])
    # All newborn destinations are inside, so the source-based gross spending
    # is no greater than actual spending. The external true cap is binding.
    # Keep its raw receipt for inspection and publish actual destination use.
    source_spending = dict(death=detail['spent_certified_death_capacity'],
        birth=detail['spent_certified_birth_capacity'])
    detail.update(ran=True, all_prior_certificates=certificate,
        actual_new_birth_destinations=birth_plan['destination_category_ids'],
        reserved_source_seed_rows=birth_plan['source_seed_rows'], source_based_spending=source_spending,
        raw_certified_death_capacity=certificate['raw_death'], raw_certified_birth_capacity=certificate['raw_birth'],
        spent_certified_death_capacity=certificate['spent_death'], spent_certified_birth_capacity=certificate['spent_birth'],
        residual_certified_death_capacity=certificate['residual_death'], residual_certified_birth_capacity=certificate['residual_birth'])
    return next_child, next_parent, detail


def plan_residual_isolation(snapshot, query_features, flags, pvalues, birth_plan, *, generator,
        copy_children, copy_parents, supported_counts):
    """Existing isolation with an explicit ledger and separate novel reserves.

    Requires the narrow supported_counts/reserved_rows extension to the
    existing select_parents method. Copy children/parents remain paired.
    """
    reserves = torch.cat((birth_plan['children'], birth_plan['source_seed_rows']))
    return snapshot.select_parents(query_features, flags, ordinary_children=copy_children,
        ordinary_parents=copy_parents, generator=generator, pvalues=pvalues,
        supported_counts=supported_counts, reserved_rows=reserves)


@torch.no_grad()
def invalidate_birth_rows(lineage, children):
    """Novel points invalidate overwritten incarnations without copy links."""
    lineage._rows(children)
    if len(torch.unique(children)) != len(children):
        raise ValueError('novel birth children must be distinct')
    if len(children):
        lineage._remove_reciprocals(children, lineage.neighbors[children].clone())
        lineage.neighbors[children] = -1


@torch.no_grad()
def apply_anchor_births(trainer, birth_death, plan):
    """Commit certified new coordinates; do not invoke copy jitter/history."""
    prior, ema = trainer.prior, trainer.ema_prior
    children = _rows(plan['children'], len(prior.z), prior.z.device)
    seeds = _rows(plan['source_seed_rows'], len(prior.z), prior.z.device)
    live, average = plan['new_latents'], plan['paired_ema_latents']
    if len(children) != len(seeds) or bool(torch.isin(children, seeds).any()):
        raise ValueError('new births require distinct paired child/source rows')
    if not len(children):
        return
    if (live.shape != prior.z[children].shape or average.shape != ema.z[children].shape
            or live.device != prior.z.device or average.device != ema.z.device
            or not bool(torch.isfinite(live).all()) or not bool(torch.isfinite(average).all())):
        raise ValueError('invalid paired new-latent coordinates')
    birth_death.lineage._rows(children)
    # Child optimizer state describes its overwritten incarnation. Zero only
    # per-row tensors; the optimizer's shared scalar step remains unchanged.
    prior.z[children] = live
    ema.z[children] = average
    state = trainer.opt_g.state.get(prior.z, {})
    for value in state.values():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(prior.z):
            value[children] = 0
    history = getattr(trainer.opt_g, 'latent_history', None)
    if history is not None:
        history[children] = 0
    for key in ('S', 'W', 'n', 'pending', 'radius'):
        getattr(birth_death, key)[children] = 0
    if birth_death.anchor is not None and birth_death.anchor.shape == prior.z.shape:
        birth_death.anchor[children] = live
    invalidate_birth_rows(birth_death.lineage, children)
