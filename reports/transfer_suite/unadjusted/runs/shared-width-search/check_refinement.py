"""Static derivative/D-only/dedup checks and pre-result plan freeze; no training."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import torch
from particlegan import GradientPenalty
from benchmarks.transfer_suite.compare_defaults import plan,effective_spec
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.shared_width_refinement import ARCHITECTURES
from benchmarks.transfer_suite.shared_width_catalog import ARCHITECTURES as INITIAL
from benchmarks.transfer_suite.shared_critic_research import constructor,variant
from benchmarks.transfer_suite.shared_variants import architecture_spec,architecture_identity
from benchmarks.transfer_suite.shared_discriminator_search import recipe
ROOT=Path('/tmp/pr38-shared-width-search');REPO=Path.cwd()
def write(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
assert not (ROOT/'refinement_scope.json').exists()
jobs=[j for j in plan() if j['spec']['runner']=='vector'];width=next(j for j in jobs if j['spec']['name']=='vector_unequal_width')
leader=json.loads((REPO/'reports/transfer_suite/unadjusted/leaderboard.json').read_text())
old=next(row for row in leader['rows'] if row['name']=='shared_c6')['records']['vector_unequal_width']['trials']
old_ids=[architecture_identity(architecture_spec(width['spec'],row['discriminator_variant'])) for row in old]
old_ids.extend(architecture_identity(architecture_spec(width['spec'],variant(card))) for card in INITIAL)
seen=[];checks=[];torch.set_num_threads(1)
for card in ARCHITECTURES:
 identity=architecture_identity(architecture_spec(width['spec'],variant(card)))
 assert identity not in old_ids and identity not in seen;seen.append(identity)
 for job in jobs:
  before=axes(effective_spec(job['spec'],recipe()),'vector');after=axes(effective_spec(architecture_spec(job['spec'],variant(card)),recipe()),'vector')
  assert before['architecture']['generator']==after['architecture']['generator']
  assert all(before[k]==after[k] for k in ['formulation','training','resources','target'])
 torch.manual_seed(0);model=constructor(card)(2,card['hidden'],card['layers'],card['fourier'])
 x=torch.tensor([[.3,-.7],[1.8,1.2]],requires_grad=True);score=model(x)
 assert torch.allclose(score,torch.cat([model(p[None]) for p in x]),atol=1e-7)
 derivative=torch.autograd.grad(score.sum(),x,create_graph=True)[0]
 (derivative.square().sum()+score.square().sum()).backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 model.zero_grad(set_to_none=True)
 with torch.no_grad():model.main.head.weight.mul_(1000.)
 penalty=GradientPenalty('b_cap',coeff=6.,kappa=1.25)(model,x.detach(),-x.detach(),step=1,generator=torch.Generator().manual_seed(0))
 assert torch.isfinite(penalty) and penalty.item()>0
 penalty.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
 checks.append(dict(name=card['name'],parameters=sum(p.numel() for p in model.parameters()),pointwise=True,finite_input_and_parameter_second_derivatives=True,active_cap_double_backprop=True,static_only_output_multiplier=1000.,no_prior_duplicate=True))
write('refinement_checks.json',checks)
declaration=dict(purpose='Adaptive12-card head/skip refinement after16 negative width cards; exact shared_c6 and original allother setup/gates. No more candidates after this batch; every width winner crosses to rare unchanged.',architectures=[c['name'] for c in ARCHITECTURES],tasks=['vector_unequal_width'])
write('refinement_plan.json',declaration)
write('refinement_scope.json',dict(scope_version=2,created_utc=datetime.now(timezone.utc).isoformat(),source_commit='f10cfb1b025aa6c843b61ea77da5f474444bc7cc',parent_initial_plan='study_plan.json',initial_plan_and_numerical_sources_preserved=True,initial_count=16,refinement_limit=12,authorized_extension='Parent requested up to12 head/input/skip refinements after reviewing the initial near-miss; this explicitly supersedes the initial optional4-card limit without rewriting its frozen plan.',trigger='Initial Softplus128x3 beta5 passes final bounds but only suffix1. SiLU160x2 has good mass/HQ/covariance but min eigen.10366. Lower/higher fixed score parameterization, smoothness and skips may alter the contracting field; no causal remedy is assumed.',architectures=ARCHITECTURES,recipe=recipe().to_dict(),selection='Run all12 frozen width variants; run every sustained width winner on rare mass unchanged. No further refinement in this batch.',command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_width_refinement --plan /tmp/pr38-shared-width-search/refinement_plan.json --output /tmp/pr38-shared-width-search/refinement > /tmp/pr38-shared-width-search/refinement.log 2>&1'))
print('Frozen12 refinement cards; all static and dedup checks pass')
