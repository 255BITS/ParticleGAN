"""Rejection and reservation edge contracts for the stable birth helpers."""
from copy import deepcopy
import json
import torch
from birth_contract_cases import copy_phases, mechanical_attempts


def run_edges(module,birth,frozen_snapshot,planning_stream,convert,data,device):
    original=data['cases']['saved_toy_1000'];value=convert(deepcopy(original),device)
    snapshot=frozen_snapshot(module,original,device);stream=planning_stream(original,device)
    pvalues=torch.zeros_like(value['pvalues'])
    law,child,parent,supported,_=copy_phases(snapshot,value,stream,pvalues,51)
    attempts=mechanical_attempts(snapshot,value,pvalues,child,parent,supported)
    assert attempts
    def allocate(rows,snap=snapshot,comparison=law):
        return birth.allocate_anchor_births(snap,value['q'],value['flags'],pvalues,comparison,rows,
            previous_children=child,previous_copy_parents=parent,supported_counts=supported,max_moves=51)
    outside=(value['flags'] & (snapshot.count_categories(value['q']).remainder(2)==1)).nonzero().flatten()[0]
    invalid=value['q'][outside:outside+1]
    rows=[]
    def record(name,checks):
        row=dict(name=name,checks=checks);rows.append(row)
        print(json.dumps(dict(event='birth_edge_contract',**row)),flush=True)
        assert all(checks.values()),name+': '+', '.join(k for k,v in checks.items() if not v)
    for model in ('current','average'):
        rejected=deepcopy(attempts)
        for a in rejected:a[model]['features']=invalid
        plan=allocate(rejected)
        record('reject_'+model,dict(original_gate_rejects_new_point=plan['moves']==0,
            no_latents_committed=plan['new_latents'] is None and plan['paired_ema_latents'] is None,
            supported_ledger_unchanged=torch.equal(plan['planned_supported_counts'],supported)))
    rejected=deepcopy(attempts)
    for a in rejected:a['average']=None
    record('paired_ema_required',dict(no_unpaired_commit=allocate(rejected)['moves']==0))
    degenerate=deepcopy(snapshot);degenerate.valid_metric=False
    record('degenerate_metric',dict(inert_without_geometry=allocate(attempts,snap=degenerate)['moves']==0))
    # Parent eligibility remains unchanged: full groups cannot be enlarged
    # even when an artificial accepted target is presented to the allocator.
    rare_original=data['cases']['rare_hole'];rare=convert(deepcopy(rare_original),device)
    rs=frozen_snapshot(module,rare_original,device);rt=rs._mass_targets(len(rare['q']));groups=rs._mass_topology()
    rare_group=int(rs._group_counts(rt).argmin());rare_cell=int((groups==rare_group).nonzero().flatten()[0])
    rlaw=rs.cell_comparison(rare['fake_features']);ids,_=rs.assign(rare['q'])
    rc=torch.bincount(ids[~rare['flags']],minlength=rs.cells);empty=torch.empty(0,dtype=torch.long,device=device)
    ra=mechanical_attempts(rs,rare,torch.zeros_like(rare['pvalues']),empty,empty,rc,forced_cells=[rare_cell])
    rplan=birth.allocate_anchor_births(rs,rare['q'],rare['flags'],torch.zeros_like(rare['pvalues']),rlaw,ra,
        supported_counts=rc,max_moves=51)
    record('full_rare_target',dict(real_rare_survivors_exactly_two=int(rs._group_counts(rc)[rare_group])==2,
        no_rare_birth=not bool((groups[rplan['target_cell_ids']]==rare_group).any()),
        group_ledger_unchanged=torch.equal(rs._group_counts(rplan['planned_supported_counts']),rs._group_counts(rc))))
    # Invalid row overlap is rejected before a plan can be committed.
    duplicate=deepcopy(attempts);duplicate[1]['seed_row']=duplicate[0]['seed_row']
    duplicate_rejected=False
    try:allocate(duplicate)
    except ValueError:duplicate_rejected=True
    record('source_uniqueness',dict(duplicate_seed_rejected=duplicate_rejected))
    old_law=dict(law,multiplicity=3*snapshot.cells,cutoff=.05/(3*snapshot.cells))
    old_rejected=False
    try:allocate(attempts,comparison=old_law)
    except ValueError:old_rejected=True
    record('actual_multiplicity',dict(stale3K_law_rejected=old_rejected))
    # Production performs no solver work without directional certificates.
    no_original=data['cases']['global_no_signal'];no=convert(deepcopy(no_original),device)
    ns=frozen_snapshot(module,no_original,device);nlaw=ns.cell_comparison(no['fake_features'])
    calls=[]
    def forbidden(latent):
        calls.append(1);raise AssertionError('solver called without count evidence')
    z=no['q'][:,:7].clone()
    np=birth.plan_real_anchor_births(ns,no['q'],no['flags'],no['pvalues'],nlaw,z,forbidden,
        ema_latents=z.clone(),ema_feature_of_latent=forbidden)
    nc,npar,nd=birth.plan_residual_global_copies(ns,no['q'],no['flags'],no['pvalues'],nlaw,np,
        generator=planning_stream(no_original,device),previous_children=empty,previous_copy_parents=empty)
    record('no_signal_production',dict(no_solver_work=not calls,no_births=np['moves']==0,
        no_global_copies=len(nc)==len(npar)==0,
        deterministic_zero_allocations=nd['death_allocation'].shape==(ns.cells,) and not bool(nd['death_allocation'].any())
            and nd['birth_allocation'].shape==(ns.cells,) and not bool(nd['birth_allocation'].any()),
        outer_detail_fields_present=all(k in nd for k in ('within_group_moves','between_group_moves','inaccessible_birth_quota')),
        no_semantic_timing='seconds' not in json.dumps(birth.novel_birth_diagnostics(np))))
    # The accepted/failed proposal selection never changes the fixed count
    # partition or law. This is a structural null-family preservation check.
    before=deepcopy(law);allocate(attempts);after=snapshot.cell_comparison(value['fake_features'])
    unchanged=all(torch.equal(before[f][k],after[f][k]) for f in ('mass','support','global_support')
        for k in ('real_counts','fake_counts','pvalues','difference','excess','deficit'))
    record('fixed_null_family',dict(actual_counts_and_decisions_unchanged=unchanged,
        family_sizes_unchanged=after['family_sizes']==before['family_sizes']==(snapshot.cells,2*snapshot.cells,2),
        common_cutoff_unchanged=after['cutoff']==before['cutoff']==.05/(3*snapshot.cells+2)))
    return rows
