"""Fixed-snapshot local power and global2-category descriptive diagnosis."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"pkg-joint-count"))
import particlegan.feature_cells as module
from contract_cases import run_plan,plain
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=[Path(__file__),HERE/"inputs.pt",HERE/"contract_cases.py",HERE/"pkg-joint-count/particlegan/feature_cells.py"]
before={str(p):sha(p) for p in sources}
data=torch.load(HERE/"inputs.pt",map_location="cpu",weights_only=False)
rows=[]
for step in (1000,2000):
    value=data["cases"][f"saved_toy_{step}"]
    snap,q,flags,p,comparison,child,parent,detail,iso_child,iso_parent,isolation,*_=run_plan(module,value,"cpu")
    ids=detail["query_cell_ids"];categories=detail["query_category_ids"]
    region=comparison["support"]
    real=region["real_counts"].reshape(snap.cells,2)
    fake=region["fake_counts"].reshape(snap.cells,2)
    difference=region["difference"].reshape(snap.cells,2)
    discoveries=region["deficit"].reshape(snap.cells,2)[:,0]
    unconfirmed=(difference[:,0]<0)&~discoveries
    reserved=torch.zeros(len(q),dtype=torch.bool)
    reserved[torch.cat((child,parent,iso_child,iso_parent))]=True
    available=(~flags)&(p>.05)&(categories.remainder(2)==0)&~reserved
    unused_outside=flags&(categories.remainder(2)==1)&~reserved
    eligible=torch.bincount(ids[available],minlength=snap.cells)
    outsiders=torch.bincount(ids[unused_outside],minlength=snap.cells)
    planned=detail["planned_supported_counts"]+torch.bincount(ids[iso_parent],minlength=snap.cells)
    cell_vacancies=(detail["target_counts"]-planned).clamp_min(0)
    group_vacancies=(snap._group_counts(detail["target_counts"])-snap._group_counts(planned)).clamp_min(0)
    cell_physical=torch.minimum(cell_vacancies,eligible.clamp_max(module.PARENT_RESERVOIR))
    physical_capacity=int(torch.minimum(snap._group_counts(cell_physical),group_vacancies).sum())
    unconfirmed_physical=int(torch.minimum(snap._group_counts(cell_physical*unconfirmed),group_vacancies).sum())
    phase=detail["support_phase"]
    remaining_birth=(phase["residual_certified_birth_capacity"]-phase["birth_allocation"]).clamp_min(0)
    remaining_death=(phase["residual_certified_death_capacity"]-phase["death_allocation"]).clamp_min(0)
    certified_birth=torch.minimum(cell_physical,remaining_birth)
    certified_birth_capacity=int(torch.minimum(snap._group_counts(certified_birth),group_vacancies).sum())
    certified_death_capacity=int(torch.minimum(outsiders,remaining_death).sum())
    remaining_budget=detail["budget"]-len(child)
    local_action_cap=min(remaining_budget,certified_birth_capacity,certified_death_capacity)
    global_real=real.sum(0);global_fake=fake.sum(0)
    global_p=module.conditional_count_pvalues(global_real,global_fake,snap.calibration_rows,len(q))
    global_difference=global_fake.double()/len(q)-global_real.double()/snap.calibration_rows
    candidate_multiplicity=3*snap.cells+2
    candidate_cutoff=.05/candidate_multiplicity
    global_significant=global_p<=candidate_cutoff
    # This is a capacity diagnosis only: no global controller/action is added.
    global_death_raw=int(torch.floor(len(q)*global_difference[1].clamp_min(0)+1e-10)) if bool(global_significant[1] and global_difference[1]>0) else 0
    global_birth_raw=int(torch.floor(len(q)*(-global_difference[0]).clamp_min(0)+1e-10)) if bool(global_significant[0] and global_difference[0]<0) else 0
    gross_outside_deaths=int((categories[child].remainder(2)==1).sum())
    gross_inside_births=int((categories[parent].remainder(2)==0).sum())
    global_death_residual=max(0,global_death_raw-gross_outside_deaths)
    global_birth_residual=max(0,global_birth_raw-gross_inside_births)
    hypothetical_cap=min(remaining_budget,physical_capacity,int(unused_outside.sum()),global_death_residual,global_birth_residual)
    corrected_counts={}
    for name in ("mass","support"):
        family=comparison[name];significant=family["pvalues"]<=candidate_cutoff
        corrected_counts[name]=int((significant&(family["difference"]!=0)).sum())
    row=dict(step=step,ordinary_moves=len(child),mass_moves=detail["mass_moves"],support_moves=detail["support_moves"],
        remaining_ordinary_budget=remaining_budget,inside_deficit_cells=int((difference[:,0]<0).sum()),
        locally_certified_inside_deficit_cells=int(discoveries.sum()),
        unconfirmed_inside_deficit_cells=int(unconfirmed.sum()),unused_inside_eligible_rows=int(eligible.sum()),
        unused_inside_eligible_rows_in_unconfirmed_deficit_cells=int(eligible[unconfirmed].sum()),
        unused_flagged_outside_rows=int(unused_outside.sum()),physical_cell_group_birth_capacity=physical_capacity,
        physical_birth_capacity_in_unconfirmed_deficit_cells=unconfirmed_physical,
        remaining_local_certified_birth_capacity=certified_birth_capacity,
        remaining_local_certified_death_capacity=certified_death_capacity,remaining_local_action_cap=local_action_cap,
        descriptive_global2=dict(real_counts=global_real,fake_counts=global_fake,pvalues=global_p,
            difference=global_difference,candidate_total_hypotheses=candidate_multiplicity,cutoff=candidate_cutoff,
            significance=global_significant,raw_outside_death_capacity=global_death_raw,
            raw_inside_birth_capacity=global_birth_raw,spent_outside_deaths=gross_outside_deaths,
            spent_inside_births=gross_inside_births,residual_outside_death_capacity=global_death_residual,
            residual_inside_birth_capacity=global_birth_residual,physical_additional_action_cap=hypothetical_cap,
            existing_family_discoveries_at_candidate_common_cutoff=corrected_counts,
            scope="No new controller, actual action, level, boundary or quality gate; hypothetical aggregate capacity"),
        by_cell=dict(real_inside=real[:,0],fake_inside=fake[:,0],inside_difference=difference[:,0],
            inside_pvalues=region["pvalues"].reshape(snap.cells,2)[:,0],local_inside_discovery=discoveries,
            available_inside_eligible=eligible,actual_cell_vacancies=cell_vacancies,cell_physical_birth_capacity=cell_physical))
    rows.append(plain(row))
    print(json.dumps(dict(event="power_case",**{k:plain(row[k]) for k in
        ("step","ordinary_moves","remaining_ordinary_budget","unconfirmed_inside_deficit_cells",
         "unused_inside_eligible_rows","physical_cell_group_birth_capacity","remaining_local_action_cap")},
         hypothetical_global2_additional_cap=hypothetical_cap,global2_pvalues=plain(global_p))),flush=True)
assert before=={p:sha(Path(p)) for p in before} and not torch.cuda.is_initialized()
result=dict(status="DIAGNOSED",cases=rows,source_sha256=before,sources_unchanged=True,cuda_initialized=False,
            new_seeds=0,optimizer_updates=0,source_law_changed=False,
            scope="Saved fixed-snapshot empirical local-deficit and aggregate capacity diagnosis only")
(HERE/"power-diagnosis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:result[k] for k in ("status","sources_unchanged","cuda_initialized","source_law_changed")}),flush=True)
