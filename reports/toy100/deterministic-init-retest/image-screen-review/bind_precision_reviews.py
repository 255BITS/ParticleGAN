"""Bind actual CPU constructor proofs to isolated logging-only v2 image runner."""
from pathlib import Path
import ast,hashlib,json,zipfile
E=Path(__file__).resolve().parents[1];R=E/'image-screen-review';B=E/'port-source/new-init-image-screen-logging-v2';V=E/'port-source/new-init-image-screen';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();seal=read(B/'bundle-sha256.json');assert {f.name:sha(f) for f in B.iterdir() if f.is_file() and f.name!='bundle-sha256.json'}==seal
old=(V/'image_screen.py').read_text();new=(B/'image_screen.py').read_text();functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)};a,b=functions(old),functions(new);assert a.keys()==b.keys();assert [n for n in a if a[n]!=b[n]]==['run_cuda']
added="""                if hasattr(trainer, 'game_stats'):
                    row['game_stats'] = material(torch, trainer.game_stats)
                precision = getattr(trainer, 'precision', None)
                if precision is not None:
                    if hasattr(precision, 'state'):
                        row['precision'] = material(torch, precision.state)
                    elif hasattr(precision, 'diagnostics'):
                        row['precision'] = material(torch, precision.diagnostics())
"""
assert new.replace(added,'')==old
for n in seal:
 if n not in ('image_screen.py','README.md'):assert (B/n).read_bytes()==(V/n).read_bytes()
base_review=read(R/'source-audit.json');assert base_review['six_frozen_definitions_exact'] and base_review['four_frozen_specs_exact']
out=[]
for candidate in ['api-rp12','api-rp14','api-rp15']:
 C=R/(candidate+'-intensity2-cpu');receipt=read(C/'cpu-preflight.json');decl=read(E/'port-source'/candidate/'candidate-declaration.json');port_review=read(E/(candidate+'-independent-init-audit.json'))
 assert receipt['status']=='PASS_CPU_ZERO_STEP' and not receipt['cuda_initialized'] and receipt['initializer_rng_neutral'] and receipt['repeated_without_rng_reset']
 assert receipt['candidate_declaration_sha256']==sha(E/'port-source'/candidate/'candidate-declaration.json')==port_review['declaration_sha256']
 assert port_review['status']=='PASS_INITIALIZATION_ONLY'
 assert decl['initial_optimizer_state']=='declared_eager' and decl['optimizer_step_devices']=={'G':'parameter','D':'parameter'}
 assert decl['serial_backward_argument'] and decl['evaluation_generate']=='plain'
 if candidate in ['api-rp12','api-rp14']:
  # Original actual CPU execution retained; v2 changes post-update logging only.
  assert not (C/'executed-v1-cpu-preflight.json').exists()
  (C/'executed-v1-cpu-preflight.json').write_bytes((C/'cpu-preflight.json').read_bytes())
  for n in list(receipt['source_sha256']):
   if (B/n).is_file():receipt['source_sha256'][n]=sha(B/n)
  receipt['review_rebinding']={'executed_receipt_sha256':sha(C/'executed-v1-cpu-preflight.json'),'only_change':'post-update game/precision logging, no initialization or learner operations','all_constructor_functions_AST_identical':True,'original_source_zip_sha256':receipt['source_zip_sha256']}
  (C/'cpu-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
 for n,want in receipt['source_sha256'].items():
  if (B/n).is_file():assert sha(B/n)==want
  elif n.startswith('particlegan/'):assert want==decl['package_sha256'][n]
 assert receipt['initial_material']['completed_steps']==0
 recipe=receipt['initial_material']['recipe'];expected=dict(decl['resolved_recipe']);expected.update(num_particles=32,z_dim=8,batch_size=32);assert recipe==expected
 counts={role:len(values) for role,values in receipt['optimizer'].items()};assert counts=={'G':9,'D':6}
 for values in receipt['optimizer'].values():
  assert all(v['step']==0 and v['step_device']==v['parameter_device']=='cpu' for v in values)
 r={'status':'PASS_SOURCE_AND_CPU_IMAGE_INITIALIZATION','candidate':decl['candidate'],'task':'img_intensity2','harness_manifest_sha256':sha(B/'bundle-sha256.json'),'declaration_sha256':receipt['candidate_declaration_sha256'],'port_review_sha256':sha(E/(candidate+'-independent-init-audit.json')),'cpu_receipt':str(C/'cpu-preflight.json'),'cpu_receipt_sha256':sha(C/'cpu-preflight.json'),'recipe':recipe,'eager_counts':counts,'changes':'Only32particles/z8/batch32 image host resources plus predeclared new initialization; unchanged candidate policy','verified':['Complete nonRNG initialstate repeats without RNGreset','All newR2 prior/model state before reference/optimizer derivation','Initializer RNGneutral,0steps/CUDAfalse; forbiddenforward/backward/optimizerstep sentries','Exact frozen image source/scorer/task600updates/24checks/final5','ActualruntimeCUDA modelbytes must equal CPUproof before updates','Candidate-owned eager parameter clocks preserved; CPUproof clocksCPU because parametersCPU','Sharedglobal data/latent transaction and private noise/measurement stream unchanged','v2 preserves nested game telemetry/precisionstate without loss coercion or learner changes'],'limits':['Initialization/source clearance only; fresh ownimage outcome required','RP15 has no prior completed image quality comparison in this review','RP12/RP14 CPUproof rebound to logging-only v2 with originalexecuted proof preserved; RP15 executedv2 directly']}
 (R/(candidate+'-image-source-review.json')).write_text(json.dumps(r,indent=2)+'\n');(R/(candidate+'-image-source-review.md')).write_text('# '+decl['candidate']+' image preparation review\n\nPASS source and own guarded CPU construction on frozen intensity2. Exact ported package and recipe remain; only32particles/z8/batch32 resources apply. All non-RNG state repeats without resetting the global stream, public initialization consumes no RNG, and eager states follow parameter devices (9G/6D). No forwards, backwards, updates or GPU execution occurred.\n\nThe same frozen600-update image host,24 checks and final-five scorer are preserved. Logging-v2 differs only by nested game/precision telemetry after the accepted update, retaining field cost/corrections and adaptive state. It does not coerce nested diagnostics into floats or change candidate arithmetic. Runtime must validate exact CUDA initialized model bytes, full600 data/latent receipts, serial steps and parameter-device clocks.\n\nThis clears own execution preparation only. No old image pass is inherited; see the JSON for source/proof hashes and exact recipe.\n')
 out.append(r)
(R/'precision-three-image-review.json').write_text(json.dumps({'status':'PASS_INITIALIZATION_ONLY','harness_manifest_sha256':sha(B/'bundle-sha256.json'),'rows':out},indent=2)+'\n');print(json.dumps([{'candidate':r['candidate'],'cpu_receipt':r['cpu_receipt'],'cpu_receipt_sha256':r['cpu_receipt_sha256']} for r in out],indent=2))
