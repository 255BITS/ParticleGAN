"""Small shared CPU/CUDA fixed-input action contracts for the joint3K law."""
from copy import deepcopy
import json
import math
import torch


def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def convert(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:convert(v,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return type(value)(convert(v,device) for v in value)
    return value


def frozen_snapshot(module,value,device):
    snap=module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    snap.__dict__.update(convert(deepcopy(value["snapshot"]),device))
    snap.device=torch.device(device)
    # Both actual count samples use the same current-device fixed partition.
    # The boundary is copied from the fitted CPU input, never refitted on odd.
    if "real_features" in value:
        odd_categories=snap.count_categories(value["real_features"][1::2].to(device))
        snap.real_calibration_category_counts=torch.bincount(odd_categories,minlength=2*snap.cells)
        snap.real_calibration_counts=snap.real_calibration_category_counts.reshape(snap.cells,2).sum(1)
    return snap


def planning_stream(value,device):
    generator=torch.Generator(device=device).manual_seed(value["seed"])
    if str(device)=="cpu" and value.get("planning_rng") is not None:
        generator.set_state(value["planning_rng"])
    return generator


def run_plan(module,value,device,*,flags=None,pvalues=None,max_moves=None):
    snap=frozen_snapshot(module,value,device)
    q=value["q"].to(device)
    flags=(value["flags"] if flags is None else flags).to(device).clone()
    pvalues=(value["pvalues"] if pvalues is None else pvalues).to(device).clone()
    generator=planning_stream(value,device)
    initial_rng=generator.get_state().clone()
    comparison=snap.cell_comparison(value["fake_features"].to(device))
    child,parent,detail=snap.ordinary_transport(q,flags,comparison,generator=generator,pvalues=pvalues,max_moves=max_moves)
    ordinary_rng=generator.get_state().clone()
    iso_child,iso_parent,isolation=snap.select_parents(q,flags,ordinary_children=child,
                        ordinary_parents=parent,generator=generator,pvalues=pvalues)
    return snap,q,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation,initial_rng,ordinary_rng


def checks_for(items,v4_method,device,max_moves=None):
    snap,q,flags,pvalues,comparison,child,parent,detail,iso_child,iso_parent,isolation,initial_rng,ordinary_rng=items
    ids=detail["query_cell_ids"];categories=detail["query_category_ids"]
    mass_child,mass_parent=detail["mass_children"],detail["mass_parents"]
    support_child,support_parent=detail["support_children"],detail["support_parents"]
    supported=torch.bincount(ids[~flags],minlength=snap.cells)
    after_mass=supported-torch.bincount(ids[mass_child[~flags[mass_child]]],minlength=snap.cells)
    after_mass+=torch.bincount(ids[mass_parent],minlength=snap.cells)
    planned=after_mass+torch.bincount(ids[support_parent],minlength=snap.cells)
    combined=planned+torch.bincount(ids[iso_parent],minlength=snap.cells)
    budget=math.floor(.05*len(q)) if max_moves is None else max(0,min(max_moves,math.floor(.05*len(q))))
    ref=deepcopy(snap)
    ref_stream=torch.Generator(device=device).set_state(initial_rng)
    ref_child,ref_parent,ref_detail=v4_method(ref,q,flags,comparison["mass"],generator=ref_stream,pvalues=pvalues,max_moves=budget)
    checks=dict(
        common_multiplicity=comparison["multiplicity"]==3*snap.cells and comparison["cutoff"]==.05/(3*snap.cells),
        family_sizes=comparison["family_sizes"]==(snap.cells,2*snap.cells),
        all_hypotheses_included=len(comparison["pvalues"])==3*snap.cells,
        ordinary_budget=len(child)<=budget and detail["budget"]==budget,
        residual_ordinary_budget=detail["support_phase"]["budget"]==budget-len(mass_child),
        paired_actions=len(child)==len(parent) and len(iso_child)==len(iso_parent),
        unique_children=len(torch.unique(torch.cat((child,iso_child))))==len(child)+len(iso_child),
        unique_parents=len(torch.unique(torch.cat((parent,iso_parent))))==len(parent)+len(iso_parent),
        no_parent_deleted=not bool(torch.isin(torch.cat((parent,iso_parent)),torch.cat((child,iso_child))).any()),
        ordinary_supported_parents=bool((~flags[parent]&(pvalues[parent]>.05)).all()),
        support_flagged_outside_deaths=bool((flags[support_child]&(categories[support_child].remainder(2)==1)).all()),
        support_inside_parents=bool((categories[support_parent].remainder(2)==0).all()),
        mass_death_certificates=bool(comparison["mass"]["excess"][ids[mass_child]].all()),
        mass_birth_certificates=bool(comparison["mass"]["deficit"][ids[mass_parent]].all()),
        support_death_certificates=bool(comparison["support"]["excess"][categories[support_child]].all()),
        support_birth_certificates=bool(comparison["support"]["deficit"][categories[support_parent]].all()),
        v4_mass_actions_exact=torch.equal(ref_child,mass_child) and torch.equal(ref_parent,mass_parent),
        mass_supported_ledger=torch.equal(after_mass,detail["supported_after_mass"]),
        ordinary_supported_ledger=torch.equal(planned,detail["planned_supported_counts"]),
        ordinary_cell_caps=bool((planned<=torch.maximum(supported,detail["target_counts"])).all()),
        combined_group_caps=bool((snap._group_counts(combined)<=torch.maximum(snap._group_counts(supported),snap._group_counts(detail["target_counts"]))).all()),
        isolation_guard_unchanged=isolation["guard_passed"]==(0<int(flags.sum())<=.05*len(q)),
        isolation_remaining_flags=bool(flags[iso_child].all()) and len(iso_child)<=int(flags.sum())-int(flags[child].sum()),
        separate_real_sample_sizes=all(int(comparison[name]["real_counts"].sum())==snap.calibration_rows for name in ("mass","support")),
        separate_fake_sample_sizes=all(int(comparison[name]["fake_counts"].sum())==len(q) for name in ("mass","support")),
        real_aggregation_exact=torch.equal(comparison["mass"]["real_counts"],comparison["support"]["real_counts"].reshape(snap.cells,2).sum(1)),
        fake_aggregation_exact=torch.equal(comparison["mass"]["fake_counts"],comparison["support"]["fake_counts"].reshape(snap.cells,2).sum(1)))
    if int(flags.sum())>.05*len(q):
        checks.update(broad_mass_flagged_only=bool(flags[mass_child].all()),broad_isolation_rejects=len(iso_child)==0)
    else:
        checks["small_mass_supported_surplus_only"]=bool((~flags[mass_child]).all())
    for name in ("mass","support"):
        item=comparison[name]
        significant=item["pvalues"]<=comparison["cutoff"]
        if not snap.valid_metric:significant.zero_()
        checks[name+"_fresh_corrected_decisions"]=torch.equal(item["excess"],significant&(item["difference"]>0)) and torch.equal(item["deficit"],significant&(item["difference"]<0))
    if detail["support_phase"]["ran"]:
        phase=detail["support_phase"]
        expected_deaths=torch.bincount(categories[mass_child],minlength=2*snap.cells).reshape(snap.cells,2)[:,1]
        expected_births=torch.bincount(categories[mass_parent],minlength=2*snap.cells).reshape(snap.cells,2)[:,0]
        checks.update(
            support_starting_ledger=torch.equal(phase["clean_counts"],after_mass),
            certificate_death_reservations=torch.equal(phase["spent_certified_death_capacity"],expected_deaths),
            certificate_birth_reservations=torch.equal(phase["spent_certified_birth_capacity"],expected_births),
            support_residual_death_certificate=bool((phase["death_allocation"]<=phase["residual_certified_death_capacity"]).all()),
            support_residual_birth_certificate=bool((phase["birth_allocation"]<=phase["residual_certified_birth_capacity"]).all()),
            residual_death_formula=torch.equal(phase["residual_certified_death_capacity"],(phase["raw_certified_death_capacity"]-expected_deaths).clamp_min(0)),
            residual_birth_formula=torch.equal(phase["residual_certified_birth_capacity"],(phase["raw_certified_birth_capacity"]-expected_births).clamp_min(0)))
    else:
        checks["no_support_preserves_mass_rng"]=torch.equal(ref_stream.get_state(),ordinary_rng)
    if isolation["guard_passed"]:
        checks["isolation_shared_ledger_exact"]=torch.equal(isolation["kept_counts"],planned)
    return checks


def run_contracts(module,data,device,v4_method):
    rows=[]
    def check(name,value,*,flags=None,pvalues=None,max_moves=None,expect=None):
        items=run_plan(module,value,device,flags=flags,pvalues=pvalues,max_moves=max_moves)
        snap,q,flag_values,p_values,comparison,child,parent,detail,iso_child,iso_parent,isolation,*_=items
        checks=checks_for(items,v4_method,device,max_moves=max_moves)
        if expect:checks.update(expect(items))
        row=dict(name=name,flags=int(flag_values.sum()),mass_moves=len(detail["mass_children"]),
                 support_moves=len(detail["support_children"]),ordinary=len(child),isolation=len(iso_child),
                 ordinary_budget=detail["budget"],total_moves=len(child)+len(iso_child),
                 mass_discoveries=detail["mass_phase"]["discoveries"],
                 support_discoveries=int((comparison["support"]["excess"]|comparison["support"]["deficit"]).sum()),
                 unique_parents=len(torch.unique(torch.cat((parent,iso_parent)))),checks=checks)
        rows.append(plain(row));print(json.dumps(dict(event="case_before_assertions",**plain(row))),flush=True)
        assert all(checks.values()),name+": "+", ".join(k for k,v in checks.items() if not v)
        return items
    for name,value in data["cases"].items():
        def expected(items,name=name):
            snap,q,flags,p,comparison,child,parent,detail,iso_child,iso_parent,isolation,*_=items
            checks={}
            if name=="supported_mass_imbalance":
                checks.update(no_flags=not bool(flags.any()),pure_supported_mass_acts=len(child)>0,
                              v4_mass_budget_preserved=len(child)==51,only_mass=len(detail["support_children"])+len(iso_child)==0)
            if name=="supported_mass_small_holes":
                checks.update(mass_and_isolation_both_act=len(detail["mass_children"])>0 and len(iso_child)>0,
                              supported_mass_deaths_present=bool((~flags[detail["mass_children"]]).all()),
                              original46_small_holes_repaired=len(iso_child)+int(flags[child].sum())==46)
            if name=="supported_balanced_control":checks["actual_balanced_table_inert"]=len(child)==0
            if name in ("nominal","rare_hole"):
                checks["original46_holes_all_repaired"]=len(iso_child)+int(flags[child].sum())==46
                ids=detail["query_cell_ids"];groups=snap._mass_topology()
                group_target=snap._group_counts(detail["target_counts"])
                rare=int(group_target.argmin())
                initial=snap._group_counts(torch.bincount(ids,minlength=snap.cells))
                after=initial-snap._group_counts(torch.bincount(ids[torch.cat((child,iso_child))],minlength=snap.cells))
                after+=snap._group_counts(torch.bincount(ids[torch.cat((parent,iso_parent))],minlength=snap.cells))
                if name=="rare_hole":
                    rare_supported=(~flags)&(groups[ids]==rare)
                    checks.update(rare_target_two=int(group_target[rare])==2,
                        legitimate_rare_survivors_preserved=not bool(torch.isin(rare_supported.nonzero().flatten(),torch.cat((child,iso_child))).any()),
                        rare_full_group_not_inflated=int(after[rare])==int(rare_supported.sum())==2)
            return checks
        check(name,value,expect=expected)
    value=data["cases"]["saved_toy_1000"]
    check("no_eligible_parent",value,pvalues=torch.zeros_like(value["pvalues"]),
          expect=lambda x:dict(no_actions=len(x[5])+len(x[8])==0))
    check("joint_ordinary_budget3",value,max_moves=3,
          expect=lambda x:dict(shared_budget_bound=len(x[5])<=3 and x[7]["support_phase"]["budget"]==3-x[7]["mass_moves"]))
    # Existing saved rows and flags, no feature/score boundary modification.
    proto=frozen_snapshot(module,value,device)
    categories=proto.count_categories(value["q"].to(device))
    available=(value["flags"].to(device)&(categories.remainder(2)==1)).nonzero().flatten().cpu()
    for n in (51,52):
        flags=torch.zeros_like(value["flags"]);flags[available[:n]]=True
        check(f"small_guard_{n}",value,flags=flags,
            expect=lambda x,n=n:dict(original_guard_boundary=x[10]["guard_passed"]==(n==51)))
    return rows
