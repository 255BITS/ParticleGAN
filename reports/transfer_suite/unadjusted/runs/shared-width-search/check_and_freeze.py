"""Static derivative/D-only/dedup checks and pre-result plan freeze; no training."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import torch
from particlegan import GradientPenalty
from benchmarks.transfer_suite.compare_defaults import plan,effective_spec
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.shared_width_catalog import ARCHITECTURES
from benchmarks.transfer_suite.shared_critic_research import constructor,variant
from benchmarks.transfer_suite.shared_variants import architecture_spec,architecture_identity
from benchmarks.transfer_suite.shared_discriminator_search import recipe
ROOT=Path('/tmp/pr38-shared-width-search');REPO=Path.cwd()
def write(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
assert not (ROOT/'study_plan.json').exists()
jobs=[j for j in plan() if j['spec']['runner']=='vector'];width=next(j for j in jobs if j['spec']['name']=='vector_unequal_width')
leader=json.loads((REPO/'reports/transfer_suite/unadjusted/leaderboard.json').read_text())
old=next(row for row in leader['rows'] if row['name']=='shared_c6')['records']['vector_unequal_width']['trials']
old_ids=[architecture_identity(architecture_spec(width['spec'],row['discriminator_variant'])) for row in old]
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
write('static_checks.json',checks)
screen=dict(purpose='Sixteen frozen D-only cards target unequal-width late stability. Exact shared_c6 recipe, original G/data/init rules/resources/budgets/gates, seed0. No per-task optimizer changes.',architectures=[c['name'] for c in ARCHITECTURES],tasks=['vector_unequal_width'])
write('screen_plan.json',screen)
write('study_plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_commit='f10cfb1b025aa6c843b61ea77da5f474444bc7cc',architectures=ARCHITECTURES,recipe=recipe().to_dict(),screen=screen,selection='Finish all16 initial width cards. Every sustained width winner runs rare mass with exactly the same architecture and recipe. No all-six/universal-architecture claim. If none wins, permit at most4 separately declared focused refinement cards only for a candidate with all final bounds passing or mean normalized bound shortfall<.03; retain initial source/results unchanged and freeze refinement before running.',trigger='Existing raw_silu128_l3 passes all final width metrics but has suffix1; last-five min eigen [.17746,.11343,.05728,.10595,.25858] oscillates around.15.',command='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_width_search --plan /tmp/pr38-shared-width-search/screen_plan.json --output /tmp/pr38-shared-width-search/screen > /tmp/pr38-shared-width-search/screen.log 2>&1'))
print('Frozen16 D-only cards; all static/dedup checks pass')
