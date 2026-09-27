"""Bind nine own image initialization receipts to the already reviewed frozen host."""
from pathlib import Path
import hashlib,itertools,json,zipfile
E=Path(__file__).resolve().parents[1];R=E/'image-screen-review';B=E/'port-source/new-init-image-screen-logging-v2'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(B/'bundle-sha256.json');assert sha(B/'bundle-sha256.json')=='057da9a9e3ac8fda9fbf35404d745cf1418bc6c65e468ddb5b970e15c2db1a17'
for name,want in seal.items():assert sha(B/name)==want,name
base=read(R/'source-audit.json');assert base['six_frozen_definitions_exact'] and base['four_frozen_specs_exact']
rows=[]
for candidate,task in itertools.product(('api-rp12','api-rp14','api-rp15'),('img_bars4','img_blobs4','img_stripes2')):
 C=R/(candidate+'-'+task+'-cpu');proof=read(C/'cpu-preflight.json');D=E/'port-source'/candidate/'candidate-declaration.json';decl=read(D)
 assert proof['status']=='PASS_CPU_ZERO_STEP' and proof['cuda_initialized'] is False and proof['initializer_rng_neutral'] and proof['repeated_without_rng_reset']
 assert proof['candidate_declaration_sha256']==sha(D) and proof['task']==task
 assert proof['initial_material']['completed_steps']==0
 assert decl['optimizer_step_devices']=={'G':'parameter','D':'parameter'} and decl['initial_optimizer_state']=='declared_eager'
 assert read(E/(candidate+'-independent-init-audit.json'))['status']=='PASS_INITIALIZATION_ONLY'
 expected=dict(decl['resolved_recipe']);expected.update(num_particles=32,z_dim=8,batch_size=32)
 assert proof['initial_material']['recipe']==expected
 assert {role:len(v) for role,v in proof['optimizer'].items()}=={'G':9,'D':6}
 assert all(v['step']==0 and v['step_device']==v['parameter_device']=='cpu' for values in proof['optimizer'].values() for v in values)
 for name,want in proof['source_sha256'].items():
  if name.startswith('particlegan/'):assert decl['package_sha256'][name]==want
  else:assert sha(B/name)==want,name
 assert sha(C/'source.zip')==proof['source_zip_sha256']
 with zipfile.ZipFile(C/'source.zip') as z:
  for name,want in proof['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==want,name
 review=dict(status='PASS_SOURCE_AND_CPU_IMAGE_INITIALIZATION',candidate=decl['candidate'],task=task,harness_manifest_sha256=sha(B/'bundle-sha256.json'),declaration_sha256=sha(D),cpu_receipt=str(C/'cpu-preflight.json'),cpu_receipt_sha256=sha(C/'cpu-preflight.json'),source_zip_sha256=proof['source_zip_sha256'],checker_sha256=sha(R/'breadth_constructor_check.py'),previous_source_review_sha256=sha(R/(candidate+'-image-source-review.json')),recipe=expected,learner_steps=0,cuda_initialized=False,source_checks=['Exact unchanged sealed logging-v2 and complete own candidate package','Own task-specific constructor receipt with complete non-RNG state repeatability','Public new initialization before reference/optimizer construction; RNG-neutral','Sentries prohibit all forward/backward/optimizer steps','Frozen600-update task,24observations,final5 scoring; actualCUDA modelbytes must match own CPUproof','Own candidate adaptive rates/eager clocks/precision/game reporting preserved'],scope='Source and initialization clearance only; fresh own task outcome required. No old image pass or new quality inferred.')
 out=R/(candidate+'-'+task+'-source-review.json');out.write_text(json.dumps(review,indent=2)+'\n');review['review']=str(out);review['review_sha256']=sha(out);rows.append(review)
(R/'precision-nine-image-review.json').write_text(json.dumps(dict(status='PASS_INITIALIZATION_ONLY',harness_manifest_sha256=sha(B/'bundle-sha256.json'),rows=rows),indent=2)+'\n')
(R/'precision-nine-image-review.md').write_text('# Nine remaining-image initialization reviews\n\nPASS for each own RP12, RP14 and RP15 construction on bars4, blobs4 and stripes2. The frozen logging-v2 image host and candidate packages remain unchanged. Each CPU receipt independently checks all non-RNG initial state, repeatability without resetting the constructor RNG, initializer RNG neutrality and package-owned eager states (9 generator/prior and 6 critic parameters). Forward, backward and optimizer-step sentries remained unentered; CUDA was not initialized.\n\nRuntime must compare actual CUDA model bytes against its own task receipt before updating and retain all 24 observations from the frozen 600-update task. These are preparation clearances only; no image result is inferred or inherited. Exact source, recipe and proof hashes are in the JSON and per-task reviews.\n')
print(json.dumps(dict(status='PASS_INITIALIZATION_ONLY',rows=len(rows),group=str(R/'precision-nine-image-review.json'),sha256=sha(R/'precision-nine-image-review.json'))))
