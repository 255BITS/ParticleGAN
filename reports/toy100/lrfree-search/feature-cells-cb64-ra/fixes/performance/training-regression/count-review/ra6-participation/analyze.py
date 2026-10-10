"""Read-only saved RA6 displacement participation and incarnation touch analysis."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
RUN=ROOT/'validation-cb64-ra6/learned/training/toy/CB64-RA6'
FINAL=ROOT/'performance/training-regression/count-review/ra6-prospective/FINAL-RECEIPT.json'
STEPS=[0,100,250,500,750,1000,1250,1500,1750,2000]
N=1024; BATCH=128; LAM=.02
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p: json.loads(Path(p).read_text())
write=lambda p,v: Path(p).write_text(json.dumps(v,indent=2,allow_nan=True)+'\n')
assert not (HERE/'receipt.json').exists(), 'receipt already exists'
frozen=read(FINAL);assert frozen['status']=='VALID' and frozen['quality_status']=='FAIL'
verified={str(FINAL):sha(FINAL)}
for name in ('continuous.py','training.py','row_evidence.py','particle_prior.py'):
    p=ROOT/'pkg-CB64-RA6/particlegan'/name
    assert sha(p)==frozen['verified_hashes'][str(p)]
    verified[str(p)]=sha(p)
cpu_rng=torch.get_rng_state().clone()
rows=[]; matrices={}; participant_masks={}; previous_step=0; previous_seconds=0.

def quantiles(x):
    return dict(n=len(x),min=float(x.min()) if len(x) else None,
        q10=float(x.quantile(.1)) if len(x) else None,median=float(x.median()) if len(x) else None,
        q90=float(x.quantile(.9)) if len(x) else None,max=float(x.max()) if len(x) else None)

for step in STEPS:
    p=RUN/f'checkpoint-{step:04d}.pt';before=sha(p)
    saved=torch.load(p,map_location='cpu',weights_only=False)
    assert sha(p)==before==next(x['checkpoint_sha256'] for x in frozen['population_trace'] if x['step']==step)
    verified[str(p)]=before
    state=saved['trainer'];table=state['lr_settle'][0][1];ev=state['row_evidence']
    assert state['completed_steps']==step
    mats={key:torch.isfinite(torch.stack(table[key])) if table[key]
        else torch.empty((0,N),dtype=torch.bool) for key in ('r_b','r_2b')}
    matrices[step]=mats
    counts={key:mat.sum(0) for key,mat in mats.items()}
    masks={key:count>=2 for key,count in counts.items()}
    participant_masks[step]=masks
    # W is deterministic in the number of nonzero-gradient touches since reset.
    # It does not recover the wall-clock reset time or the sampled row sequence.
    touches=torch.log1p(-LAM*ev['W'])/math.log1p(-LAM)
    assert bool(torch.isfinite(touches).all()) and float((touches-touches.round()).abs().max())<1e-8
    touch_count=touches.round().long()
    neff=ev['W'].square()/ev['S'].clamp_min(1e-30)
    details={}
    for key,mat in mats.items():
        details[key]=dict(pair_count=len(mat),pair_finite_rows=mat.sum(1).tolist(),
            finite_observation_histogram=torch.bincount(counts[key]).tolist(),
            participating_rows=int(masks[key].sum()),participating_indices=masks[key].nonzero().flatten().tolist(),
            increasing_finite_sets=bool((~mat[:-1]|mat[1:]).all()) if len(mat)>1 else True,
            finite_then_missing_rows=int((mat[:-1]&~mat[1:]).any(0).sum()) if len(mat)>1 else 0,
            last_two_pairs_equal_eligibility=bool(torch.equal(mat[-2:].all(0),masks[key])) if len(mat)>=2 else None,
            touch_count_participants=quantiles(touch_count[masks[key]].double()),
            touch_count_nonparticipants=quantiles(touch_count[~masks[key]].double()))
    seconds=saved['record']['training_seconds']
    row=dict(step=step,s=table['s'],b=table['b'],tau=table['tau'],blocks=table['blocks_in_window'],
        unfinished_quad_blocks=len(table['blocks']),last_decision=table['last'],last_population=table['last_population'],
        last_look=table['last_look'],required_rows=N-math.floor(.05*N),
        population_active=table['population_active'],accepted_stationary=table['counts']['stationary'],
        expiries=table['counts']['population_expiries'],coverage_rejections=table['counts']['population_coverage_rejections'],
        rebases=table['counts']['rebases'],group_restarts=table['counts']['restarts'],group_reopens=table['counts']['reopens'],
        invalid_current_block_rows=int(table['invalid_block_rows'].sum()) if table['invalid_block_rows'] is not None else None,
        excluded_rows=int(table['exclude'].sum()) if table['exclude'] is not None else 0,hold_descent=table['hold_descent'],
        paired_displacement_observations=details,gradient_touches_since_last_reset=quantiles(touch_count.double()),
        zero_touch_rows=int((touch_count==0).sum()),row_effective_n=quantiles(neff),mature_rows=int((neff>=384).sum()),
        row_resets=ev['counters']['resets'],moves=state['birth_death']['counters']['moves'],
        isolation_moves=state['birth_death']['counters']['iso_moves'],novel_births=state['birth_death']['counters']['novel_birth_moves'],
        evaluations=state['birth_death']['counters']['evals'],
        g_learning_rates=[x['lr'] for x in state['optimizers'][0]['param_groups']],
        d_learning_rates=[x['lr'] for x in state['optimizers'][1]['param_groups']],
        g_network_scale=state['lr_settle'][0][0]['s'],d_own_scale=state['lr_settle'][1][0]['s'],
        all_training_parameters_require_grad=all(v for role in ('G','D','prior') for v in state['requires_grad'][role].values()),
        controller_closed=state['controller']['closed'],controller_updates=state['controller']['updates'],
        training_seconds=seconds,interval_seconds_per_update=(seconds-previous_seconds)/(step-previous_step) if step else None,
        record_metrics=saved['record']['metrics'])
    assert row['row_resets']==row['moves']+row['isolation_moves']
    if step>=1000:
        assert table['s']==1 and table['b']==64 and table['blocks_in_window']*64+table['tau']==step-904
        assert row['group_restarts']==0 and row['group_reopens']==0
        assert row['excluded_rows']==0 and row['hold_descent'] is False
    rows.append(row);previous_step=step;previous_seconds=seconds

cohorts=[]
for earlier in (1250,1500,1750):
    start=matrices[earlier]['r_b'];final=matrices[2000]['r_b'][:len(start)]
    # Every finite observation lost between snapshots is a rebase invalidation;
    # the window is identical, no whole-table restart occurs, and pairs are never overwritten.
    assert not bool((final & ~start).any())
    eligible=participant_masks[earlier]['r_b']
    retained=(final.sum(0)>=2)&eligible
    cohorts.append(dict(start_step=earlier,start_rows=int(eligible.sum()),
        retained_original_pair_participants_at2000=int(retained.sum()),
        lost_original_pair_participants_by2000=int((eligible&~retained).sum()),
        retained_indices=retained.nonzero().flatten().tolist(),reason='rebase invalidation of this same window; no restart'))

p_touch=1-(1-1/N)**BATCH
reference=[]
for b in (16,32,64):
    block_touch=1-(1-p_touch)**b
    pair_valid=block_touch**2
    four_pairs_two_or_more=1-(1-pair_valid)**4-4*pair_valid*(1-pair_valid)**3
    # This is an illustrative iid-uniform replacement assumption, not the real
    # count-guided donor law and not a new statistical test or simulation.
    survive4b=(1-51/N)**(4*b/8)
    survive8b=(1-51/N)**(8*b/8)
    reference.append(dict(b=b,uniform_sample_touch_probability_per_block=block_touch,
        four_pairs_at_least_two_if_touch_implies_displacement=four_pairs_two_or_more,
        iid_uniform_51_per8_survival_for4b=survive4b,iid_uniform_51_per8_survival_for8b=survive8b,
        iid_uniform_95pct_survival_max_moves_per8=N*(1-.95**(8/(4*b)))))
by_step={r['step']:r for r in rows}
final=by_step[2000]
since1000=dict(moves=final['moves']-by_step[1000]['moves'],
    evaluations=final['evaluations']-by_step[1000]['evaluations'])
since1000['average_moves_per_evaluation']=since1000['moves']/since1000['evaluations']
since1000['iid_uniform_mean_lifetime_steps']=8*N/since1000['average_moves_per_evaluation']
since1000['iid_uniform_mean_touches_before_reset']=p_touch*since1000['iid_uniform_mean_lifetime_steps']
assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
write(HERE/'observations.json',dict(status='VALID',endpoints=rows,cohort_retention=cohorts,
    observed_movement_since1000=since1000,analytic_references=reference))
receipt=dict(status='VALID',created_at=datetime.now(timezone.utc).isoformat(),quality_status='FAIL',
    scope='saved CPU analysis; no source law, model evaluation, gradient, optimizer update, replay or random experiment',
    findings=dict(rejected525_was_tested_b32_at904=True,b64_clock_advances_each_update=True,
        final_blocks=17,final_tau=8,earliest_full_window_step_if_s1=2440,
        final_current_participants_b=204,final_current_participants_2b=53,required=973,
        last_look_step=1928,last_look_logbf=[-.7045399794443015,.6249864170130379],required_early_logbf=math.log(80),
        table_stationarity_accepted=0,table_population_expiries=0,
        excludes_or_holds_cause=False,row_evidence_never_mature_in128d=True,
        final_median_gradient_touches_since_reset=final['gradient_touches_since_last_reset']['median'],
        cohort_retention=cohorts,controller_not_a_training_freeze=True),
    limitations=['The rejected525/599 row identities were cleared by _conclude and are not recoverable; only saved counts are exact.',
        'W recovers nonzero-gradient touches since reset, not reset wall time or all sampled indices.',
        'Uniform survival/sample formulas are analytic reference assumptions; real donors are count-guided and nonuniform.',
        'Ongoing transport can eventually qualify if movement concentrates in at most51 rows or enough lineages stop moving; it is not universally impossible.',
        'Population participation does not certify each row stationary, G/D convergence, emitted quality or the required Grid100 gate.'],
    production_changes=0,new_tests=0,new_model_evaluations=0,new_seeds=0,optimizer_updates=0,
    global_cpu_rng_unchanged=True,cuda_initialized=False,verified_input_hashes=verified)
write(HERE/'receipt.json',receipt)
report='''# RA6 table participation after the completed toy failure

RA6 toy remains FAIL: emitted precision .516968,23/25 modes,TV .483032. The saved artifact audit is VALID. This report uses saved parameter-displacement vectors and gradient sufficient statistics only.

## What the coverage count measures

The525-row rejection at904 tested b32 and changed b to64. It was a negative **average displacement-direction** test over a surviving subset, followed by a population participation veto. Participation requires two finite cosine observations for the same current row incarnation at the negative scale. It is not a count of sampled indices, gradient updates, independently stationary rows, or G/D convergence. The rejected decision clears its observation arrays; the525 and earlier599 row identities are no longer saved.

Each b cosine uses two complete blocks; two such observations need at least four clean b blocks. A2b cosine uses four blocks; two need eight clean b blocks. Copy and novel birth rebase keeps the clock, erases the moved row in **every previously completed pair**, and masks its unfinished block. Future gradients do not repair an erased old pair. More sampling therefore cannot preserve evidence for repeatedly replaced incarnations.

## The clock ran; participation and evidence were insufficient

| Step | b | Completed blocks since904 | Finite-pair participants b /2b | Moves cumulative | Median touches since reset |
|---|---:|---:|---:|---:|---:|
'''
for step in (1000,1250,1500,1750,2000):
    r=by_step[step]
    report+=f"| {step} | {r['b']:.0f} | {r['blocks']} | {r['paired_displacement_observations']['r_b']['participating_rows']} / {r['paired_displacement_observations']['r_2b']['participating_rows']} | {r['moves']} | {r['gradient_touches_since_last_reset']['median']:.0f} |\n"
report+='''
All these checkpoints have table s1, LR .0085, zero excluded rows, no hold, and no whole-table restart. At2000,17×64+8=1096 intrinsic updates exactly equals2000−904. The full24-block window would first finish at2440 if s stays1. At the latest early look1928, log Bayes factors −.70454/.62499 were below log80=4.38203, so no new negative direction decision occurred. This is not a stalled clock or another coverage rejection at b64.

Saved evidence directly demonstrates lineage turnover: the143 eligible b-row incarnations at1250 retain only3 voters in those same original pairs at2000;140 were invalidated by rebase. At final, only204 rows have two b observations and53 have two2b observations, versus973 required. Their finite sets are nested; participation equals the final two clean-pair intersection at each scale.82 rows have zero gradient touches since the latest reset. Median touches are12 overall,52 among b participants,86 among2b participants. Current-mask identities and observation histograms are retained in observations.json.

## Sampling reference and continued transport

Uniform sampling of128 indices from1024 gives each row touch probability .117557 per update. Without replacement, a64-step block has touch probability .999666; ordinary sampling by itself provides ample opportunities. W=50(1−.98^touches) recovers the **actual** nonzero-gradient touch count to numerical precision. The all-row median12 is far below the approximately235 expected sampled touches over2000 updates without resets. Sparse sampling also causes some missing pairs at shorter b, but cannot explain this final cohort loss.

The actual second half makes5763 moves over125 evaluations,46.104 moves per8-step evaluation on average. Under an illustrative uniform replacement assumption, that corresponds to a mean incarnation lifetime about178 updates and21 gradient touches. Under the maximum51-per8 rate, survival across the four64-step blocks needed for two b pairs is about.195; across eight blocks it is about.038. A95% survival rate across four blocks would allow only about1.64 uniformly spread replacements per8, not51. These formulas are descriptive assumptions; the actual count-guided donors are not uniform, and no simulation or statistical level was fitted.

It is mathematically possible to certify during continuing moves if the moves stay within at most51 rows and at least973 other incarnations remain unchanged and gather negative-direction evidence. It is also possible after sufficiently few rows are replaced. The observed broadly changing table does not meet that condition. Doubling b after a coverage rejection increases the lineage lifetime needed by the next test; it does not cure churn. This explains the scheduler's lack of a whole-table certificate, without showing that lowering the coverage gate would be valid.

## Separate optimizer and row-evidence limitations

G/D/prior parameters remain trainable at every endpoint. controller.closed is already true at500 and is a mobility diagnostic; under stationarity control it does not freeze optimizers. Final G LR is .00006640625, prior LR .0085, actual D LR about.0031875 due the existing .75 prior-rate floor. D's own small s is not its applied LR. The speed change after1072 therefore is not explained by a newly closed optimizer in the saved state.

RowEvidence is a separate touch-gradient test. Its effective sample cap99 is below3×128=384; every checkpoint has zero mature rows and no flags, so it cannot provide the intended hot-row/exclusion/hold signal. Even an infinite untouched run cannot mature this configured full128-dimensional test. A projection would need its own independently specified subspace, sufficient statistics and null interpretation; existing scalar norms cannot justify reusing p-values in a smaller rank.

No production change is qualified by this report. Preserve the population continuity veto and unchanged serving/gates. Any next transport scheduling or gradient-subspace mechanism requires a fresh declared law and prospective toy plus original Grid100 validation. An EMA serving override would not establish current-population stationarity or repair fast training drift.
'''
(HERE/'REPORT.md').write_text(report)
for p in (HERE/'analyze.py',HERE/'observations.json',HERE/'receipt.json',HERE/'REPORT.md'):
    verified[str(p)]=sha(p)
write(HERE/'FROZEN.json',dict(status='VALID',quality_status='FAIL',files=verified))
print(json.dumps(dict(status='VALID',receipt=str(HERE/'receipt.json'),receipt_sha256=sha(HERE/'receipt.json'),
    freeze_sha256=sha(HERE/'FROZEN.json'),cohorts=cohorts,uniform_reference=reference[-1],
    observed_since1000=since1000)))
