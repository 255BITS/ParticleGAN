from pathlib import Path
import datetime,gzip,hashlib,json
OUT=Path(__file__).parent
TAG='h_n05r06_mixup_c0p01_lr15'
read=lambda p:json.loads(Path(p).read_text())
plan=read(OUT/'verification-plan.json')
cold=read(OUT/'cold-replay'/TAG/'status.json')
strict_extra=read(OUT/'remaining-replay'/TAG/'status.json')
cold_audit=read(OUT/'cold-evidence-audit.json')
remaining_audit=read(OUT/'remaining-evidence-audit.json')
row=read(OUT/'one.json')[0]

def episode_info(stage):
    artifact=Path(stage['artifact']); saved=json.loads(gzip.decompress(artifact.read_bytes()))
    verdict=saved['verdict']
    return {'task':stage['gate'],'status':verdict['status'],'budget':saved['spec']['steps'],
            'passing_suffix':verdict['convergence']['passing_suffix'],
            'required_suffix':verdict['convergence']['minimum_stable_checks'],
            'live':saved['result']['live'],'verdict':verdict,
            'elapsed_seconds':stage['seconds'],'episode':str(artifact),
            'evidence_directory':str(artifact.parent.parent),
            'source_archive':str(artifact.parent.parent/'source.tar.gz'),
            'signal_policy_receipt':str(artifact.parent.parent/'signal-policy.json.gz')}

cold_rows=[episode_info(s) for s in cold['stages']]
first_failure=episode_info(strict_extra['stages'][0])
diagnostic_rows=[]
for task in plan['remaining_order'][1:]:
    status=read(OUT/'diagnostics'/task/TAG/'status.json')
    diagnostic_rows.append(episode_info(status['stages'][0]))
strict_rows=cold_rows+[first_failure]+[
    {'task':t,'status':'SKIPPED','reason':'Strict expansion stopped after unipolar sustained-gate FAIL; any later older-host diagnostic is separately labeled.'}
    for t in plan['remaining_order'][1:]+plan['native_order']]
assert len(strict_rows)==22
all_older=cold_rows+[first_failure]+diagnostic_rows
assert len({r['task'] for r in all_older})==19
passed=sum(r['status']=='PASS' for r in all_older)
failed=sum(r['status']=='FAIL' for r in all_older)
assert (passed,failed)==(13,6)
refs={
 'unused_parameters':str(OUT/'repo/benchmarks/locked_shared/hosts/unused_token_hold.py')+':141',
 'unused_embedding':str(OUT/'repo/benchmarks/locked_shared/hosts/unused_token_hold.py')+':147',
 'unused_adversarial_path':str(OUT/'repo/benchmarks/locked_shared/hosts/unused_token_hold.py')+':259',
 'unused_hold_loss':str(OUT/'repo/benchmarks/locked_shared/hosts/unused_token_hold.py')+':270',
 'unused_metrics':str(OUT/'repo/benchmarks/locked_shared/hosts/unused_token_hold.py')+':167',
 'ae_encoder_and_optimizers':str(OUT/'repo/benchmarks/locked_shared/hosts/ae_gan_hold.py')+':163',
 'ae_encoder_reconstruction_path':str(OUT/'repo/benchmarks/locked_shared/hosts/ae_gan_hold.py')+':210',
 'ae_loss_paths':str(OUT/'repo/benchmarks/locked_shared/hosts/ae_gan_hold.py')+':219',
 'ae_evaluation':str(OUT/'repo/benchmarks/locked_shared/hosts/ae_gan_hold.py')+':115',
}
report={'candidate':TAG,'verification_status':'FAIL','completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Frozen older-toy cold baseline; exact archived candidate plus explicit faithful host plumbing. Native tests skipped after older failures.',
 'candidate_declaration':row,'cold_replay':{'passed':10,'required':10,'seconds':cold['seconds'],'all_live_metrics_exactly_match_original':True,'source_count':125,'audit':str(OUT/'cold-evidence-audit.json'),'sample_regraded':9,'sample_missing':['residual_student']},
 'strict_fail_fast_22':{'passed':10,'failed':1,'skipped':11,'first_failure':'unipolar','cases':strict_rows},
 'authorized_post_failure_diagnostic':{'passed':3,'failed':5,'cases':diagnostic_rows},
 'all_measured_older_19':{'passed':passed,'failed':failed,'cases':all_older},
 'native_100':{'passed':0,'failed':0,'skipped':3,'cases':[{'task':t,'status':'SKIPPED','reason':'Older-host strict and diagnostic failures; no 100-mode training authorized after those failures.'} for t in plan['native_order']]},
 'separate_continuation':{'status':'PRIOR_REPORTED_FAIL','freshly_replayed_here':False,'additional_updates':1200,'final_modes':5,'final_hq':.209228515625,'source':'/ml2/hypergan/gan-attempts/batch-20260924T153037Z/critic_signal/20260924T153037Z-1298555/repo/reports/toy100/critic_signal_attempt/h-own-hold/diagnostic-summary.json'},
 'training_receipts':{'source_config_episode_audits_passed':19,'constant_actual_lrs':{'g':.0015,'d':.0015,'prior':.003},'adam_betas':[0.,.999],'input_noise_sigma':.05,'additional_updates':0,'fixed_seed_older':0,'native_seed_declared':1234,'no_seed_sweeps':True,'max_training_workers':4,'threads_per_worker':1,'python':'/tmp/pr38-default-env/bin/python','cpu_capability':'avx2'},
 'host_plumbing':{'script':str(OUT/'repo/reports/toy100/selected_h_remaining.py'),'extension':str(OUT/'repo/reports/toy100/selected_h_extension.py'),'fixture_audit':str(OUT/'plumbing-audit.json'),'conditional_critic':'Expose the existing critic and its NoisePolicy stream; same callable forward, gradients, and RNG, no extra forward/update.','auxiliary_removal':{'ae_gan_hold':{'reconstruction_weight':0.},'unused_token_hold':{'hold_weight':0.}},'grading_architecture_budgets_seeds_unchanged':True},
 'source_inferences':{'source_lines':refs,'unused_token_hold':'With the hold term zero, shared and concept-slot parameters have identical gradients/Adam settings and zero initialization; the unused correction receives zero gradients. Thus concept displacement is twice unused displacement. concept_move>=0.85 requires normalized concept norm>=0.85, hence unused_hold<=0.575, incompatible with the frozen >=0.85 gate. This is a source-level invariant under the preserved optimizer/host, not a general GAN impossibility claim.','ae_gan_hold':'With reconstruction_weight=0 and cover/fm/particle fits zero, the encoder has no adversarial path: GAN generation samples the prior directly. The frozen reconstruction metric still evaluates that untrained encoder. This identifies disconnected training, not a proof that no accidental initialization can pass.'},
 'setup_errors':[{'path':str(OUT/'cold'),'training_updates':0,'reason':'Supplied replay packaging omitted frozen leading_profile.json; exact dependency copied before replay.'},{'path':str(OUT/'remaining'),'training_updates':0,'reason':'Conditional lambda lacks modules(); fixed in separately archived interface-only extension.'}],
 'evidence':{name:str(OUT/name) for name in ['cold-parity.json','cold-evidence-audit.json','remaining-evidence-audit.json','isolation.json','frozen-reference-receipt.json','plumbing-audit.json','tests.jsonl','diagnostic-tests.jsonl','run-cold.sh','run-remaining-replay.sh','run-diagnostics.py']}}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')

lead={
 'trajectory':'MSE 0.002905043', 'mode_hold':'8 modes; HQ 0.999267578',
 'residual_student':'MSE 0.002931164; success 1; wrong-pad 0',
 'img_stripes2':'2 modes; HQ 0.96875','img_bars4':'4 modes; HQ 1',
 'vector_overlap':'SW1 0.030681546','img_blobs4':'4 modes; HQ 1',
 'img_intensity2':'2 modes; HQ 0.96875','vector_unequal_mass':'mass TV 0.041552730; covariance error 0.417076070',
 'vector_unequal_width':'mass TV 0.032226563; covariance error 0.307263948',
 'unipolar':'endpoint passes, but suffix 1/5; neutral hold 0.850326702',
 'two_pole':'mean_abs 0.032636743 < 0.30',
 'cover_leftover':'u_kept 0.4491; content 0.4419; pole errors 0.4156/0.4542',
 'mid_scale_identity':'identity_at_0 0.806213874 < 0.85',
 'unused_token_hold':'hold 0.729775608; concept_move 0.540403429; both require 0.85',
 'ae_gan_hold':'reconstruction MSE 3.917919397 > 0.05; hold 0.002231841 passes',
 'vector_anisotropic':'SW1 0.058866644; covariance error 0.275090615',
 'vector_two_broad':'SW1 0.040083604; covariance error 0.133632258',
 'vector_spiral':'SW1 0.036653005; covariance error 0.069570340',
}
# Do not hard-code any diagnostic metric incorrectly in prose.
lead['vector_overlap']=f"SW1 {next(r for r in cold_rows if r['task']=='vector_overlap')['live']['sw1_normalized']:.9f}"
lines=['# Selected H verification: FAIL','',
 '**Cold acquisition replay: 10/10 PASS, with all live metrics and non-timing verdict fields exactly matching the archived run.** The subsequent strict 22-case sequence stopped at unipolar: **10 PASS, 1 FAIL, 11 SKIPPED**. An explicitly authorized one-time diagnostic of the other eight older hosts added three passes and five failures. Across all 19 measured older hosts: **13 PASS, 6 FAIL**. grid100, rotated100, and staggered100 remain SKIPPED.','',
 'The exact selected archive has 125 verified source files, SHA256 `ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334`. All remain unchanged. Ten cold gates took 77.22 s on one CPU/AVX2 worker. Subsequent diagnostics used at most four one-thread workers. Fixed older seed 0; no seed sweep or candidate tuning.','',
 'The conditional-host extension exposes the existing lambda critic and noise stream to H’s mixup helper. A fixture proves identical forward values, gradients, penalty, and RNG state. The explicit pure-GAN overrides set AE reconstruction and unused-token hold weights to zero. Frozen architecture, budgets, measurement samples, metrics, thresholds, and sustained-gate rules are unchanged. Actual G/D LR is 0.0015, particle LR 0.003, Adam betas (0, 0.999), and discriminator input sigma 0.05 throughout all 19 measured hosts.','',
 '| Case | Strict sequence | Later diagnostic | Budget | Result detail |',
 '| --- | --- | --- | ---: | --- |']
diagmap={r['task']:r for r in diagnostic_rows}
for r in strict_rows:
    task=r['task'];d=diagmap.get(task)
    measured=r if r['status']!='SKIPPED' else d
    budget=measured['budget'] if measured else 7000
    detail=lead.get(task,'SKIPPED after older failures')
    lines.append(f"| `{task}` | {r['status']} | {d['status'] if d else '—'} | {budget} | {detail} |")
lines += ['',
 '**Unipolar is a sustained-gate failure:** its final cover 0.922150949, off-caption 0.0000238003 and neutral hold 0.850326702 satisfy endpoint bounds, but its last passing run has only one check; the frozen minimum is five.','',
 '**Auxiliary-host blockers inferred from source:** [unused-token parameters]('+refs['unused_parameters']+'), [embedding rule]('+refs['unused_embedding']+'), [concept-only adversarial update]('+refs['unused_adversarial_path']+'), [hold loss]('+refs['unused_hold_loss']+'), and [metrics]('+refs['unused_metrics']+') imply an invariant when hold_weight=0: shared and concept-slot parameters stay equal, and the unused correction stays zero. A concept_move of at least 0.85 therefore forces unused_hold at most 0.575, below its required 0.85. This applies to this preserved host and optimizer; critic tuning cannot remove that conflict.','',
 'The AE [encoder path]('+refs['ae_encoder_reconstruction_path']+') feeds reconstruction only, while the [adversarial loss]('+refs['ae_loss_paths']+') samples the prior directly. Removing reconstruction weight disconnects encoder training although the [frozen metric]('+refs['ae_evaluation']+') still evaluates the encoder. This is a missing training signal, not an observed instability of the unconditional generator. Its hold metric passes.','',
 'All 19 source/config/episode and optimizer/noise receipts independently regrade. Cold raw final draws regrade for trajectory/ring; restored frozen evaluation draws exactly reproduce every metric for the four image and three vector gates. `residual_student` has no checkpoint in the archived screen, so its evidence is the 24-check episode and full update/noise receipts. The original own-state continuation remains a prior reported FAIL (5 modes/HQ 0.209228516); this verification did not rerun it.','',
 'Evidence: [verification.json](verification.json), [cold parity](cold-parity.json), [cold evidence audit](cold-evidence-audit.json), [remaining audit](remaining-evidence-audit.json), [plumbing fixture](plumbing-audit.json), [cold artifacts](cold-replay/'+TAG+'/status.json), [first failure](remaining-replay/'+TAG+'/status.json), [diagnostic summary](diagnostics/summary.json). Exact commands are [run-cold.sh](run-cold.sh), [run-remaining-replay.sh](run-remaining-replay.sh), and [run-diagnostics.py](run-diagnostics.py). Tail `diagnostics.log` or individual candidate `run.log` files.','',
 'Two setup errors are retained: the supplied replay script omitted the exact frozen leading_profile.json dependency; then the unextended H helper rejected a conditional lambda before any optimizer step. Neither is counted as a GAN failure. All work is isolated here; no source-attempt/PR checkout edits or GitHub writes were performed.','']
(OUT/'verification.md').write_text('\n'.join(lines))
print(json.dumps({'strict':{'PASS':10,'FAIL':1,'SKIPPED':11},'all_older':{'PASS':13,'FAIL':6},'native':'3 SKIPPED','report':str(OUT/'verification.md')}))
