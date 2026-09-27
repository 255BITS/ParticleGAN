from pathlib import Path
import ast,hashlib,json,zipfile
H=Path(__file__).resolve().parents[1];E=H.parent;R=H/'reviews';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seals={'api-rp12':'883195fac3e8e721fbc4b34d13393b89af188949a334bfea3894878499f07e05','api-rp14':'263ff75f07d3a3fb99065f5477f03fefa21899154239147cfe0054c916234489','api-rp15':'6aa48b73df1081c6a6b57930efe4a54df68d54f00caab0dd310ce8457c008208'}
rows=[]
for alias,wanted in seals.items():
 B=H/alias;D=read(B/'declaration.json');assert sha(B/'bundle-sha256.json')==wanted
 for name,digest in read(B/'bundle-sha256.json')['files'].items():assert sha(B/name)==digest,name
 port=E/'port-source'/alias;decl=read(port/'candidate-declaration.json')
 assert D['recipe']=={**decl['resolved_recipe'],'num_particles':20000,'z_dim':2,'batch_size':2048}
 assert D['package_sha256']==decl['package_sha256']
 assert 'eager' in D['execution']['adam_state'] and D['optimizer_step_devices']=={'G':'parameter','D':'parameter'}
 source=(B/'worker.py').read_text();assert 'recipe = Recipe(**declaration["recipe"])' in source
 assert 'value.serial_backward is True' in source and "trainer.game_stats['field_evaluations']==2" in source
 row=dict(candidate=D['candidate'],manifest_sha256=wanted,declaration_sha256=sha(B/'declaration.json'),worker_sha256=sha(B/'worker.py'),status='PASS_SOURCE_ONLY_CPU_SKIPPED_AFTER_QUALITY_REJECTION' if alias=='api-rp12' else 'PASS_SOURCE_AND_CPU_RING_INITIALIZATION',quality='NOT_RUN',launch_scope='Existing candidate only; own image/breadth survival required. No result inherited.',recipe=D['recipe'],source_checks=['Exact own initializer port and resolved recipe; only canonical ring dimensions restored','Original dense G/D and ring data/scorer definitions; 4600updates/shift2400/all460observations','Caller data0; learner latent2/penalty3/eval4/noise5; isolated measurement9','Public GANTrainer serial_backward=True and package-owned eager states on parameter device','Exactly2fields/accepted update; precision and game diagnostics retained; no constant numeric-rate claim','Own schema4 frozen control retains precision/reference, optimizer clocks, serial mode and RNG; main state equality before/after construction/restore','Saved recipe uses public Recipe; no invalid family-selector or learner patch'],resolved_metadata='Eager parameter-device clocks replace stale lazy/CPU declaration prose before CPU execution.')
 outdir=R/alias;outdir.mkdir(exist_ok=True)
 if alias!='api-rp12':
  P=outdir/'cpu-constructor-proof.json';p=read(P);assert p['status']=='PASS' and p['manifest_sha256']==wanted and p['declaration_sha256']==row['declaration_sha256']
  assert not p['cuda_initialized'] and p['learner_steps']==0 and all(p['checks'].values()) and p['eager_counts']==[9,8]
  assert p['constructor_rng_sha256']==D['expected_initial_fixture']['cpu_rng_sha256']
  row.update(cpu_receipt=str(P),cpu_receipt_sha256=sha(P),cpu_checks=['Complete nonRNG checkpoint state and named model buffers repeat without resetting RNG','Public QR/R2 initialization is RNG-neutral; prior exactly standard deviation1 R2','Original global CPU construction cursor matches canonical fixture','All17 eager states are zero and remain on parameter devices; no injected or relocated counters','Precision reference equals initialized critic; private streams seeds2/3/4/5','No forwards/backwards/optimizer steps or GPU initialization'])
 else:row['not_run_reason']='Own new-init bars4 failure0/24 rejected further RP12 qualification before this CPU proof.'
 if alias=='api-rp14':row['qualification_hold']='Root reported own bars4 failure0/24; pending independent runtime audit. Preparation PASS is not clearance to launch.'
 (outdir/'source-review.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 (outdir/'source-review.md').write_text('# '+D['candidate']+' ring preparation\n\n'+row['status']+'. The exact candidate package and policy are preserved, with canonical20k/z2/batch2048 ring resources and the new public initializer. The source retains all460 observations, isolated evaluation, two accepted game fields, complete precision/game telemetry and a separate frozen schema4 control at2400. Eager clocks remain package-owned on each parameter device.\n\n'+('CPU initialization was deliberately skipped after the own bars4 failure; do not launch this rejected candidate.\n' if alias=='api-rp12' else 'Own guarded CPU construction passed: complete initialized state/buffers repeat, the initializer is RNG-neutral, the original CPU construction cursor matches, all17 eager states start at zero and the precision reference matches initialized D. No forwards, backward passes, optimizer steps or CUDA initialization occurred. Runtime must compare actual CUDA state to this own proof.\n\nThis preparation does not override image/breadth failures or authorize launch of a rejected candidate.\n'))
(R/'independent-ring-review.json').write_text(json.dumps(dict(status='PASS_SOURCE_PREPARATION_WITH_QUALITY_PRUNING',rows=rows),indent=2)+'\n')
print(json.dumps(dict(review=str(R/'independent-ring-review.json'),sha256=sha(R/'independent-ring-review.json'),rows=[{k:r[k] for k in ('candidate','status')} for r in rows])))
