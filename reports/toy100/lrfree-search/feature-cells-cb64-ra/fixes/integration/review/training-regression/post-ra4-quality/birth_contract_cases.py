"""Focused fixed-input birth/copy/isolation contracts, without training."""
from copy import deepcopy
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import torch


def copy_phases(snapshot, value, generator, pvalues, budget):
    q, flags = value['q'], value['flags']
    law = snapshot.cell_comparison(value['fake_features'])
    mc, mp, md = snapshot._ordinary_mass_transport(q, flags, law['mass'], generator=generator,
        pvalues=pvalues, max_moves=budget)
    ids, _ = snapshot.assign(q)
    clean = torch.bincount(ids[~flags], minlength=snapshot.cells)
    after_mass = clean - torch.bincount(ids[mc[~flags[mc]]], minlength=snapshot.cells)
    after_mass += torch.bincount(ids[mp], minlength=snapshot.cells)
    possible = (budget > len(mc) and bool(flags.any()) and bool(law['support']['excess'][1::2].any())
        and bool(law['support']['deficit'][0::2].any()))
    empty = torch.empty(0, dtype=torch.long, device=q.device)
    if possible:
        sc, sp, sd = snapshot._ordinary_support_transport(q, flags, law['support'], generator=generator,
            pvalues=pvalues, max_moves=budget-len(mc), reserved_children=mc, reserved_parents=mp,
            supported_counts=after_mass)
    else:
        sc, sp, sd = empty, empty, dict(moves=0)
    child, parent = torch.cat((mc, sc)), torch.cat((mp, sp))
    return law, child, parent, after_mass + torch.bincount(ids[sp], minlength=snapshot.cells), dict(mass=md, local=sd)


def mechanical_attempts(snapshot, value, pvalues, children, parents, supported, *, forced_cells=None):
    """Allocator only: accepted even reference features, no solver/quality claim."""
    q, flags = value['q'], value['flags']
    ids, _ = snapshot.assign(q); cat = snapshot.count_categories(q)
    reserved = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    reserved[children] = True; reserved[parents] = True
    pools = torch.bincount(ids[(~flags) & (pvalues > .05) & (cat.remainder(2) == 0) & ~reserved], minlength=snapshot.cells)
    targets = snapshot._mass_targets(len(q)); vacancy = (targets-supported).clamp_min(0)
    groups = snapshot._mass_topology()
    gv = (snapshot._group_counts(targets)-snapshot._group_counts(supported)).clamp_min(0)
    accessible = (pools == 0) & (vacancy > 0) & (gv[groups] > 0)
    cells = vacancy.masked_fill(~accessible, -1).argsort(descending=True, stable=True)
    cells = cells[accessible[cells]][:4].tolist() if forced_cells is None else forced_cells
    real = value['real_features'][::2]
    real_ids, _ = snapshot.assign(real); real_cat = snapshot.count_categories(real)
    rf, rp, _ = snapshot.support(real)
    projected = snapshot.transform(q)
    rows = []
    for cell in cells:
        reference = ((real_ids == cell) & (~rf) & (rp > .05) & (real_cat == 2*cell)).nonzero().flatten()
        if not len(reference):
            continue
        distance = (projected-snapshot.real_representatives[cell]).square().sum(1).masked_fill(reserved, float('inf'))
        source = int(distance.argmin())
        if not math.isfinite(float(distance[source])):
            break
        reserved[source] = True
        feature = real[reference[0]:reference[0]+1].clone()
        # These coordinates exercise commit/reset only. The separate saved
        # response receipt supplies actual model-generated accepted latents.
        latent = torch.arange(7, dtype=q.dtype, device=q.device) + cell + .25
        proposal = dict(accepted=True, features=feature, latent=latent)
        rows.append(dict(kind='new_latent_birth', cell=cell, seed_row=source, accepted=True,
            current=proposal, average=dict(proposal, latent=latent+1)))
    return rows


def commit_contract(module, birth, value, plan):
    if not plan['moves']:
        return {}
    n, width = len(value['q']), 7
    device = value['q'].device
    z = torch.arange(n*width, dtype=value['q'].dtype, device=device).reshape(n, width)
    prior = SimpleNamespace(z=torch.nn.Parameter(z.clone()))
    ema = SimpleNamespace(z=torch.nn.Parameter(-z.clone()))
    moments = {key:z.clone()+i for i, key in enumerate(('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'))}
    moments['step'] = torch.tensor(23., device=device)
    opt = SimpleNamespace(state={prior.z:moments}, latent_history=z.clone()+9)
    trainer = SimpleNamespace(prior=prior, ema_prior=ema, opt_g=opt)
    lineage = module.LatentLineage(n, 4, device)
    child = plan['children']; source = plan['source_seed_rows']
    # Old incarnations have actual reciprocal links, distinct from seeds.
    other = torch.tensor([i for i in range(n) if i not in set(torch.cat((child, source)).tolist())][:len(child)], device=device)
    lineage.register_copies(child, other)
    lineage_before = lineage.neighbors.clone(); copied_before = lineage.work['copied_rows']
    controller = SimpleNamespace(lineage=lineage, S=torch.ones(n, device=device), W=torch.ones(n, device=device),
        n=torch.ones(n, device=device, dtype=torch.long), pending=torch.ones(n, device=device, dtype=torch.bool),
        radius=torch.ones(n, device=device), anchor=z.clone())
    before = {key:t.clone() for key, t in moments.items()}
    history_before = opt.latent_history.clone(); live_before=prior.z.detach().clone(); ema_before=ema.z.detach().clone()
    live_version, ema_version = prior.z._version, ema.z._version
    birth.apply_anchor_births(trainer, controller, plan)
    untouched = torch.ones(n, dtype=torch.bool, device=device); untouched[child] = False
    lineage.validate(lineage.neighbors)
    return dict(
        committed_live_exact=torch.equal(prior.z[child], plan['new_latents']),
        committed_ema_exact=torch.equal(ema.z[child], plan['paired_ema_latents']),
        source_rows_untouched=torch.equal(prior.z[source], live_before[source]) and torch.equal(ema.z[source], ema_before[source]),
        other_live_rows_untouched=torch.equal(prior.z[untouched], live_before[untouched]),
        other_ema_rows_untouched=torch.equal(ema.z[untouched], ema_before[untouched]),
        child_moments_zero=all(not bool(v[child].any()) for k, v in moments.items() if k != 'step'),
        other_moments_untouched=all(torch.equal(v[untouched], before[k][untouched]) for k, v in moments.items() if k != 'step'),
        shared_adam_step_retained=torch.equal(moments['step'], before['step']),
        child_history_zero=not bool(opt.latent_history[child].any()),
        other_history_untouched=torch.equal(opt.latent_history[untouched], history_before[untouched]),
        child_evidence_zero=all(not bool(getattr(controller,k)[child].any()) for k in ('S','W','n','pending','radius')),
        new_live_anchor_exact=torch.equal(controller.anchor[child], plan['new_latents']),
        old_child_links_invalidated=bool((lineage.neighbors[child] == -1).all()) and not bool(torch.isin(lineage.neighbors, child).any()),
        no_new_seed_link=not bool((lineage.neighbors[child] >= 0).any()),
        source_graph_rows_untouched=torch.equal(lineage.neighbors[source], lineage_before[source]),
        copy_counter_unchanged=lineage.work['copied_rows'] == copied_before,
        live_ema_versions_incremented=prior.z._version > live_version and ema.z._version > ema_version)


def run_contracts(module, birth, frozen_snapshot, planning_stream, convert, data, device):
    results = []
    variants = [(name, v, None, 51, None) for name, v in data['cases'].items()]
    saved = data['cases']['saved_toy_1000']
    variants += [('absent_copy_parents', saved, torch.zeros_like(saved['pvalues']), 51, None),
        ('birth_budget_zero', saved, torch.zeros_like(saved['pvalues']), 0, None),
        ('birth_budget_two', saved, torch.zeros_like(saved['pvalues']), 2, None)]
    for count in (51,52):
        flags = torch.zeros_like(saved['flags'])
        snapshot = frozen_snapshot(module, saved, 'cpu')
        available = (saved['flags'] & (snapshot.count_categories(saved['q']).remainder(2) == 1)).nonzero().flatten()
        flags[available[:count]] = True
        variants.append((f'small_guard_{count}', dict(saved, flags=flags), None, 51, None))
    for name, original, p_override, budget, forced in variants:
        value = convert(deepcopy(original), device)
        snapshot = frozen_snapshot(module, original, device)
        stream = planning_stream(original, device)
        pvalues = value['pvalues'] if p_override is None else p_override.to(device)
        law, copy_child, copy_parent, supported, phases = copy_phases(snapshot, value, stream, pvalues, budget)
        boundary_before = snapshot.count_boundary.clone()
        attempts = mechanical_attempts(snapshot, value, pvalues, copy_child, copy_parent, supported)
        plan = birth.allocate_anchor_births(snapshot, value['q'], value['flags'], pvalues, law, attempts,
            previous_children=copy_child, previous_copy_parents=copy_parent, supported_counts=supported, max_moves=budget)
        gc, gp, gd = birth.plan_residual_global_copies(snapshot, value['q'], value['flags'], pvalues, law, plan,
            generator=stream, previous_children=copy_child, previous_copy_parents=copy_parent)
        ids, _ = snapshot.assign(value['q']); categories = snapshot.count_categories(value['q'])
        after_global = plan['planned_supported_counts'] + torch.bincount(ids[gp], minlength=snapshot.cells)
        actual_copy_child = torch.cat((copy_child, gc)); actual_copy_parent = torch.cat((copy_parent, gp))
        ic, ip, iso = birth.plan_residual_isolation(snapshot, value['q'], value['flags'], pvalues, plan,
            generator=stream, copy_children=actual_copy_child, copy_parents=actual_copy_parent, supported_counts=after_global)
        all_children = torch.cat((actual_copy_child, plan['children'], ic))
        all_parents = torch.cat((actual_copy_parent, ip))
        all_sources = torch.cat((all_parents, plan['source_seed_rows']))
        initial = torch.bincount(ids[~value['flags']], minlength=snapshot.cells)
        planned = initial-torch.bincount(ids[copy_child[~value['flags'][copy_child]]], minlength=snapshot.cells)
        planned += torch.bincount(ids[copy_parent], minlength=snapshot.cells)
        planned += torch.bincount(plan['target_cell_ids'], minlength=snapshot.cells)
        planned += torch.bincount(ids[gp], minlength=snapshot.cells)
        combined = planned+torch.bincount(ids[ip], minlength=snapshot.cells)
        certificate = plan['certificates_after']
        actual_previous_categories = torch.cat((categories[copy_parent], plan['destination_category_ids']))
        gross_death = (categories[torch.cat((copy_child, plan['children']))].remainder(2) == 1).sum()
        gross_birth = (actual_previous_categories.remainder(2) == 0).sum()
        checks = dict(
            common_family_unchanged=law['multiplicity'] == 3*snapshot.cells+2 and law['cutoff'] == .05/(3*snapshot.cells+2),
            frozen_boundary_unchanged=torch.equal(boundary_before, snapshot.count_boundary),
            ordinary_budget_shared=len(copy_child)+plan['moves']+len(gc) <= budget,
            birth_bound_four=plan['moves'] <= plan['attempted_cells'] <= 4,
            distinct_children=len(torch.unique(all_children)) == len(all_children),
            distinct_copy_parents_and_seed_reservations=len(torch.unique(all_sources)) == len(all_sources),
            no_copy_parent_or_seed_deleted=not bool(torch.isin(all_children, all_sources).any()),
            birth_only_flagged_outside=bool((value['flags'][plan['children']] & (categories[plan['children']].remainder(2) == 1)).all()),
            born_destinations_inside=bool((plan['destination_category_ids'] == 2*plan['target_cell_ids']).all()),
            source_not_claimed_copy_parent=len(plan['copy_parent_rows']) == 0,
            paired_birth_latents_explicit=not plan['moves'] or plan['new_latents'].shape == plan['paired_ema_latents'].shape,
            supported_ledger_exact=torch.equal(planned, after_global),
            ordinary_cell_caps=bool((planned <= torch.maximum(initial, plan['target_counts'])).all()),
            combined_group_caps=bool((snapshot._group_counts(combined) <= torch.maximum(snapshot._group_counts(initial), snapshot._group_counts(plan['target_counts']))).all()),
            correct_gross_death_use=torch.equal(certificate['spent_death'], gross_death),
            correct_actual_destination_use=torch.equal(certificate['spent_birth'], gross_birth),
            global_residual_death_exact=torch.equal(certificate['residual_death'], (certificate['raw_death']-gross_death).clamp_min(0)),
            global_residual_birth_exact=torch.equal(certificate['residual_birth'], (certificate['raw_birth']-gross_birth).clamp_min(0)),
            later_global_no_double_spend=len(gc) <= min(int(certificate['residual_death']), int(certificate['residual_birth'])),
            isolation_guard_retained=iso['guard_passed'] == (0<int(value['flags'].sum()) <= .05*len(value['q'])),
            isolation_no_duplicate_flag_deaths=len(ic) <= int(value['flags'].sum())-int(value['flags'][torch.cat((actual_copy_child,plan['children']))].sum()),
            isolation_actual_supported_override=not iso['guard_passed'] or torch.equal(iso['kept_counts'], after_global),
            true_global_metadata=not gd['ran'] or (torch.equal(gd['spent_certified_birth_capacity'], gross_birth) and torch.equal(gd['spent_certified_death_capacity'], gross_death)))
        if name == 'absent_copy_parents':
            checks.update(birth_actuates_without_copy_supply=plan['moves'] > 0,
                ineligible_sources_are_new_births=bool((pvalues[plan['source_seed_rows']] <= .05).all()),
                copy_contract_retained=len(all_parents) == 0)
        if name == 'birth_budget_two':
            checks['exact_two_shared_slots'] = plan['moves'] == 2 and len(actual_copy_child) == 0
        if name in ('birth_budget_zero','global_no_signal','global_opposite_signal','supported_balanced_control','supported_mass_imbalance'):
            checks['birth_correctly_inert'] = plan['moves'] == 0
        if name == 'supported_mass_imbalance':
            checks['pure_mass51_preserved'] = len(copy_child) == phases['mass']['moves'] == 51
        if name == 'rare_hole':
            group_targets = snapshot._group_counts(plan['target_counts']); rare = int(group_targets.argmin())
            groups = snapshot._mass_topology(); rare_rows = ((~value['flags']) & (groups[ids] == rare)).nonzero().flatten()
            checks.update(rare_two_real_survivors_retained=len(rare_rows) == 2 and not bool(torch.isin(rare_rows, all_children).any()),
                rare_full_group_no_inflation=int(snapshot._group_counts(combined)[rare]) == 2)
        if name == 'global_certificate_exhaustion':
            checks.update(exhausted_birth_phase_inert=plan['moves'] == 0, exhausted_global_inert=len(gc) == 0,
                prior32_exhaust_global=int(certificate['spent_birth']) == int(certificate['raw_birth']) == 32)
        checks.update(commit_contract(module, birth, value, plan))
        row = dict(name=name, mass=phases['mass']['moves'], local=phases['local']['moves'], birth=plan['moves'],
            global_copy=len(gc), isolation=len(ic), attempted=plan['attempted_cells'], budget=budget,
            target_cells=plan['target_cell_ids'].tolist(), source_categories=plan['source_category_ids'].tolist(),
            global_raw=int(certificate['raw_birth']), global_prior_use=int(certificate['spent_birth']),
            checks=checks)
        results.append(row)
        print(__import__('json').dumps(dict(event='birth_contract_case',**row)),flush=True)
        assert all(checks.values()), name+': '+', '.join(k for k,v in checks.items() if not v)
    return results
