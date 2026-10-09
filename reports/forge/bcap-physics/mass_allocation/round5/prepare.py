"""Preregister the one balanced-assignment successor and exact local-v2 control."""
from pathlib import Path
from copy import deepcopy
import json
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).parent
TASKS=['gaussian1d_smoke','gaussian1d_stability','vector_unequal_mass','vector_two_broad','vector_unequal_width','grid100']
IDS={r:f'mass-allocation-round5-{r}-v1' for r in ('control','candidate')}
VIEW='mass-allocation-round5-diagnostic-v1';CAMPAIGN='mass-allocation-round5-v1'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 base=json.loads((ROOT/'configs/forge/ideas/projection_transport-round4-transport-v1.json').read_text())
 budget=sum(json.loads((ROOT/f'configs/forge/tasks/{t}.json').read_text())['resources']['timeout_seconds'] for t in TASKS)
 assert budget==9720 and 2*budget+120+120+120<=21600
 for role,cid in IDS.items():
  c=deepcopy(base);c.update(id=cid,parent=base['id'],guide='reports/forge/bcap-physics/mass_allocation/round5/README.md',
   changed_factors=['Replace sliced transport with conservative joint-coordinate assignment, globally fixed block128; weight1 and exact local-v2 weight1 unchanged.'] if role=='candidate' else ['Import exact local-v2 as matched new-source control; no trainer delta.'],
   api_changes=['Recipe.kinetic_transport_mode and kinetic_transport_block_size; reusable balanced_assignment_loss.'] if role=='candidate' else [],
   mechanism_rationale='One conservative joint coupling per128-row contiguous block replaces projection-specific pairings; all consumed rows retained. No target geometry/labels, extra draws, weight changes or acceptance controller.' if role=='candidate' else 'Exact retained local-v2 rare/broad positive baseline; source-bound archived scores remain contextual.',
   prior_art=['https://proceedings.mlr.press/v108/fatras20a.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html','reports/forge/bcap-physics/mass_allocation/round5/prior-diagnostics.json'])
  if role=='candidate':c['recipe_overrides'].update(kinetic_transport_mode='balanced_assignment',kinetic_transport_block_size=128)
  write(ROOT/f'configs/forge/ideas/{cid}.json',c)
  study=dict(schema_version=1,id=f'mass-allocation-round5-{role}-study-v1',status='ready',candidate=cid,
   control=dict(candidate_id=IDS['control' if role=='candidate' else 'candidate'],task_map={}),
   hypothesis='Balanced joint-coordinate block transport improves native genuine quality mass while preserving local-v2 rare/broad sustained passes; original full density and temporal gates decide repair.',
   scope=dict(view=VIEW,through_tier=1,execution_backend='cuda',cuda_model='NVIDIA RTX A6000'),
   campaign=dict(id=CAMPAIGN,candidate_budget_seconds=9720,budget_seconds=19440),max_rounds=1,
   prior_evidence=[dict(path='reports/forge/bcap-physics/mass_allocation/round5/prior-diagnostics.json',selector=[],identity=dict(qualification_input=False),use='motivation_only')],
   prediction=dict(task_id='grid100',metric='precision',op='>=',threshold=.97,phase='final'),
   falsifier=dict(task_id='grid100',metric='precision',op='<',threshold=.97,phase='final'),
   competing_explanation='Balanced finite batch coupling still contains sampled mass fluctuations and shared-map distortion. Cross-label pairing decreases are diagnostic only; minibatch OT is not a population divergence and can bias shape. Native local-v2 is unmeasured before this study. Local covariance or occupancy improvements can coexist with spill, missing quality mass and retention failure.',
   terminal_rules=dict(falsified='stop_revision',prediction_observed='review_saved_diagnostics',inconclusive='stop_and_readout',incomplete='request_missing_evidence'))
  write(ROOT/f'configs/forge/studies/{study["id"]}.json',study)
 write(ROOT/f'configs/forge/views/{VIEW}.json',dict(schema_version=1,id=VIEW,revision=1,goal='discriminator_stability',evidence_scope='research_diagnostic',assignments=[dict(task=t,qualification_tier=1,importance='diagnostic',order=i) for i,t in enumerate(TASKS)],eligibility={},calibration=dict(status='provisional',adoption_blocker='One bounded allocation diagnostic; no ordinary qualification or promotion.'),ranking=dict(policy='qualified_tier_only_with_raw_metrics',compare_compatible_cohorts=True,cost_separate=True),policy_change_reason='Authorized two-arm mass allocation study, full unchanged native7k and vector/Gaussian budgets/gates. Own failed smoke still blocks its stability.'))
 write(OUT/'freeze.json',dict(schema_version=1,roles=IDS,tasks=TASKS,view=VIEW,campaign=CAMPAIGN,main_reservation_seconds=19440,track_ceiling_seconds=21600,ancillary_allowances=dict(saved_readonly_diagnostics=120,synthetic_capacity_no_training=120,software_fixtures=120),unadmitted_remainder_seconds=1800,stop='One candidate only; complete every runnable declared job and publish even negative. No sweep, seed study, second candidate, continuation or promotion.',base='Exact local-v2, projection none; rates/nonsaturating/BCAP/fullDualNorm identical.',trainer_delta=dict(kinetic_transport_mode=['sliced','balanced_assignment'],block_size=128),control_selection='Retain measured rare/broad positives and isolate one allocation coupling change; winner is historical context, not another matched arm.'))
if __name__=='__main__':main()
