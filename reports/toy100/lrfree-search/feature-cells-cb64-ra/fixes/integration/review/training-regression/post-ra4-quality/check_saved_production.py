"""Actual selected-head callback + bounded production entry on saved RA4 states.

The count sample here is the saved clean table, explicitly not a replay of
unrecorded historical GPU emissions. No optimizer update or birth is applied.
"""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import importlib
import json
from types import SimpleNamespace
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PACKAGE=HERE/'pkg-ANCHOR-CONTRACT'
sys.path.insert(0,str(PACKAGE))
from particlegan import feature_cells as module, birth_phase as birth
from birth_contract_cases import copy_phases
from measure_saved_utils import tensor_state_hash
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def linear(weight,bias):
    # Construct from exact saved weights without random initialization.
    layer=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(layer)
    layer.in_features=weight.shape[1];layer.out_features=weight.shape[0]
    layer.weight=torch.nn.Parameter(weight.clone());layer.bias=torch.nn.Parameter(bias.clone())
    return layer


def network(weights):
    modules=[]
    for index in (0,2,4):
        modules.append(linear(weights[f'{index}.weight'],weights[f'{index}.bias']))
        if index!=4:modules.append(torch.nn.LeakyReLU(.2))
    return torch.nn.Sequential(*modules)


states=[ROOT/f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{step:04d}.pt' for step in (1250,2000)]
sources=[*sorted(PACKAGE.rglob('*.py')),Path(__file__),HERE/'birth_contract_cases.py',HERE/'measure_saved_utils.py',*states]
before={str(p):sha(p) for p in sources};rng=torch.get_rng_state().clone();rows=[]
for path in states:
    state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];original=tensor_state_hash(state)
    weights=state['models'];G=network(weights['G']);D=network(weights['D']);ema_G=network(weights['ema_G'])
    trainer=SimpleNamespace(G=G,D=D);controller=SimpleNamespace(_heads=[D[4]],sample_shape=(2,))
    current=birth.learned_latent_features(controller,trainer,G)
    average=birth.learned_latent_features(controller,trainer,ema_G)
    modes=[(m,m.training) for root in (G,D,ema_G) for m in root.modules()]
    z=weights['prior']['z'];ez=weights['ema_prior']['z']
    with torch.no_grad():
        q=current(z).double();r=D[:4](state['birth_death']['reservoir']).double()
        callback_exact=torch.equal(q,D[:4](G(z)).double())
        snapshot=module.FeatureCellSnapshot.fit(r,generator=torch.Generator().set_state(state['cpu_rng']),cells=64,rank=8,chunk=256)
        flags,pvalues,_=snapshot.support(q)
    stream=torch.Generator().set_state(state['cpu_rng'])
    value=dict(q=q,flags=flags,pvalues=pvalues,real_features=r,fake_features=q)
    law,child,parent,supported,phases=copy_phases(snapshot,value,stream,pvalues,51)
    plan=birth.plan_real_anchor_births(snapshot,q,flags,pvalues,law,z,current,ema_latents=ez,ema_feature_of_latent=average,
        previous_children=child,previous_copy_parents=parent,supported_counts=supported,max_moves=51)
    diag=birth.novel_birth_diagnostics(plan)
    gc,gp,gd=birth.plan_residual_global_copies(snapshot,q,flags,pvalues,law,plan,generator=stream,
        previous_children=child,previous_copy_parents=parent)
    supplied=supported+torch.bincount(plan['target_cell_ids'],minlength=snapshot.cells)
    checks=dict(callback_exact_selected_head=callback_exact,
        finite_count_budget=len(child)+plan['moves']+len(gc)<=51,
        paired_acceptance_all=all(a['current']['accepted'] and a['average']['accepted'] for a in plan['accepted_attempts']),
        no_checkpoint_mutation=tensor_state_hash(state)==original,
        model_modes_restored=all(m.training==mode for m,mode in modes),
        parameter_gradients_untouched=all(p.grad is None for root in (G,D,ema_G) for p in root.parameters()),
        exact_supported_destination_ledger=torch.equal(supplied,plan['planned_supported_counts']),
        four_cell_work_bound=len(plan['attempts'])<=4,
        four_iteration_work_bound=all(a[key]['linearizations']<=4 for a in plan['attempts'] for key in ('current','average')),
        no_semantic_timing='seconds' not in json.dumps(diag),
        no_production_oracle_input=True)
    assert all(checks.values()),checks
    record=dict(step=state['completed_steps'],count_sample_scope='saved clean table, no historical emitted-feature replay',
        mass=phases['mass']['moves'],local=phases['local']['moves'],birth=plan['moves'],global_copy=len(gc),
        novel_birth=diag,checks=checks,training_state_sha256=original)
    rows.append(record);print(json.dumps(dict(event='saved_production_birth',**record)),flush=True)
assert before=={str(p):sha(p) for p in sources} and torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
result=dict(status='PASS',records=rows,source_sha256=before,global_rng_unchanged=True,cuda_initialized=False,
    new_training_steps=0,committed_births=0,generated_emissions=0,quality_verdict=None)
(HERE/'saved-production-contract.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(event='saved_production_contract_complete',status='PASS')),flush=True)
