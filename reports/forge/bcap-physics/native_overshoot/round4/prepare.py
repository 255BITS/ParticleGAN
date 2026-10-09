"""Freeze one matched, finite-step diagnostic comparison; launches no training."""
from copy import deepcopy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
WINNER='bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36'
IDS={'control':'native-overshoot-round4-control-v1','candidate':'native-overshoot-round4-armijo-v1'}
VIEW='native-overshoot-round4-diagnostic-v1'
CAMPAIGN='native-overshoot-round4-v1'
TASKS=['two_pole','gaussian1d_smoke','gaussian1d_stability','vector_two_broad','grid100']

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def main():
    winner=json.loads((ROOT/f'configs/forge/configurations/{WINNER}.json').read_text())
    for role,identifier in IDS.items():
        candidate={key:deepcopy(winner[key]) for key in ('api_version','claim_contract','execution_path','recipe_overrides','recipe_preset','requires_capabilities','schema_version')}
        candidate.update(id=identifier,parent=WINNER,mechanism_class='structural',
            guide='reports/forge/bcap-physics/native_overshoot/round4/README.md',
            prior_art=['reports/forge/bcap-tier2-search/failure-state-analysis.json',
                       'https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf',
                       'https://arxiv.org/abs/1905.09997'],
            api_changes=['Public Recipe.finite_step_mode and checkpointed GANTrainer finite replay; conditional/fixed two-pole components explicitly unsupported.'] if role=='candidate' else [],
            changed_factors=['Recipe.finite_step_mode=armijo; every actual joint G/prior proposal is checked, including locally downhill ones; c0.1, ten halvings, zero on finite rejection'] if role=='candidate' else ['Exact archived winner recipe imported unchanged as the matched new-source control'],
            mechanism_rationale='Preserve the actual winner optimizer directions and D updates; realize sufficient same-batch G/prior decrease using the actual displacement and unchanged kernel draws. No target geometry or scoring label enters acceptance.' if role=='candidate' else 'Exact winner baseline on the same exercised source/runtime/seed0 tasks as Armijo. Archived winner remains original-source context.')
        if role=='candidate':candidate['recipe_overrides']['finite_step_mode']='armijo'
        write(ROOT/f'configs/forge/ideas/{identifier}.json',candidate)
        op='>=' if role=='candidate' else '<'
        study=dict(schema_version=1,id=f'native-overshoot-round4-{role}-study-v1',status='ready',candidate=identifier,
            control=dict(candidate_id=IDS['control' if role=='candidate' else 'candidate'],task_map={}),
            hypothesis='Bounded Armijo realization of actual G/prior motion repairs sustained native precision without losing distribution shape or broad/Gaussian guardrails.' if role=='candidate' else 'The unchanged winner still fails native precision under the new matched source.',
            scope=dict(view=VIEW,through_tier=1,execution_backend='cuda',cuda_model='NVIDIA RTX A6000'),
            campaign=dict(id=CAMPAIGN,candidate_budget_seconds=6420,budget_seconds=12840),max_rounds=1,
            prior_evidence=[dict(path='reports/forge/bcap-physics/native_overshoot/round4/saved-probe.json',selector=[],
                                 identity=dict(attempt_id='db79021404c14c91aa5f38f9f1be40fb',source_digest='2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed'),use='motivation_only')],
            prediction=dict(task_id='grid100',metric='precision',op=op,threshold=.97,phase='final'),
            falsifier=dict(task_id='grid100',metric='precision',op='<' if role=='candidate' else '>=',threshold=.97,phase='final'),
            competing_explanation='Finite descent on a changing batch/critic is not population fidelity. Armijo may suppress rare useful movement, select critic exploitation, or satisfy scalar loss while worsening full local covariance. The archived endpoint probe is a one-state hypothesis, not a trained result.',
            terminal_rules=dict(falsified='stop_revision',prediction_observed='review_saved_diagnostics',inconclusive='stop_and_readout',incomplete='request_missing_evidence'))
        write(ROOT/f'configs/forge/studies/{study["id"]}.json',study)
    view=dict(schema_version=1,id=VIEW,revision=1,goal='discriminator_stability',evidence_scope='research_diagnostic',
              assignments=[dict(task=task,qualification_tier=1,importance='diagnostic',order=i) for i,task in enumerate(TASKS)],
              eligibility={},calibration=dict(status='provisional',adoption_blocker='Explicit bounded finite-step diagnostic; no ordinary-tier qualification or default promotion.'),
              ranking=dict(policy='qualified_tier_only_with_raw_metrics',compare_compatible_cohorts=True,cost_separate=True),
              policy_change_reason='User-authorized native overshoot study, unchanged grid1007k plus scalar smoke/own stability and broad guardrail, with explicit two-pole applicability. No failed-own-smoke dependency bypass.')
    write(ROOT/f'configs/forge/views/{VIEW}.json',view)

if __name__=='__main__':main()
