"""Independent exact-source binding for first six historical constructor proofs."""
from pathlib import Path
import ast,hashlib,json
Q=Path(__file__).resolve().parent;E=Q.parent;K=E/'research-mode-hold-preparation';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text());rows=read(Q/'prepared-index.json')['rows'][:6];out=[]
base=(K/'run_research_mode_hold.py').read_text();kplan=read(K/'source-plan.json')
for row in rows:
 B=Path(row['directory']);p=read(B/'source-plan.json');seal=read(B/'manifest.json');proof=read(Path(row['required_review']));R=Path(row['required_review']).parent
 assert sha(B/'manifest.json')==row['manifest_sha256']==proof['manifest_sha256']
 assert sha(B/'source-plan.json')==row['source_plan_sha256']==proof['source_plan_sha256']
 assert sha(B/'initialization_bridge.py')==sha(K/'initialization_bridge.py')==proof['bridge_sha256']
 s=(B/'run_research_mode_hold.py').read_text().replace('exact grouped research screen','exact research-KA2 screen').replace("HERE/'candidate-source'","HERE/'ka2-source'").replace("candidate=plan['candidate']","candidate='RESEARCH-KA2-new-init'")
 assert s==base
 assert p['historical_runtime_files']==kplan['historical_runtime_files'] and p['initializer_package']==kplan['initializer_package']
 assert p['frozen_host_sha256']==kplan['frozen_host_sha256'] and p['train_mode_hold_sha256']==kplan['train_mode_hold_sha256']
 originals=Path(p['source_candidate']);checked=[];fallback=[]
 for n,h in p['candidate_files'].items():
  assert sha(B/'candidate-source'/n)==h
  if (originals/n).is_file():assert sha(originals/n)==h;checked.append(n)
  else:
   assert n in ['probe.py','convergence_gate.py','checkpoint.py'] and sha(K/'ka2-source'/n)==h,n;fallback.append(n)
 for n in ['probe.py','latent.py','response.py','checkpoint.py']:
  assert sha(B/'candidate-source'/n)==sha(K/'ka2-source'/n)
 for n,h in seal['files'].items():assert sha(B/n)==h
 assert proof['status']=='PASS' and all(proof['checks'].values()) and proof['learner_steps']==0 and not proof['cuda_initialized']
 report={'status':'PASS_SOURCE_AND_CPU_INITIALIZATION','candidate':row['candidate'],'manifest_sha256':row['manifest_sha256'],'source_plan_sha256':row['source_plan_sha256'],'cpu_proof_sha256':sha(Path(row['required_review'])),'bridge_sha256':proof['bridge_sha256'],'runner_sha256':sha(B/'run_research_mode_hold.py'),'exact_original_candidate_files':checked,'exact_reviewed_probe_dependency_fallbacks':fallback,'unchanged_shared_runtime_files':len(p['historical_runtime_files']),'cpu_checks':proof['checks'],'source_definition':p['source_definition']['historical_eligibility'],'scope':'Frozen dense research mode_hold; custom original learner and new public constructor initialization only','limits':['CPU constructors only; zero forwards, backwards, optimizers or GPU.','Fresh external quality required; no old22 scores inherited.','When original runtime receipt is unresolved, common declared retest runtime is explicit; no historical execution bit-parity claim.','Historical recipe/horizon eligibility remains separate from small-screen quality.']}
 (R/'source-review.json').write_text(json.dumps(report,indent=2)+'\n')
 (R/'source-review.md').write_text('# '+row['candidate']+' independent preparation review\n\nPASS. Every retained candidate file matches the original source authority (or the explicitly reviewed common probe dependency). The initializer bridge, original probe/latent/response/checkpoint and complete200-file historical runtime match the reviewed research KA2 template. The runner changes only the source directory, report label and docstring; all arithmetic and host construction/sampling/scoring remain unchanged.\n\nA fresh isolated CPU process passed all seven required constructor checks: all parameters/buffers match public new-initialization material, repeat without RNG reset, preserve original constructor RNG cursor, initializer operations consume no RNG, preserve historical prior registration, restore bindings after exceptions, and reject unsupported batch-distance heads. Forward/backward/optimizer steps were forbidden. No CUDA initialized and no learner steps ran.\n\nThis clears preparation only. Existing horizon/eligibility limitations remain and no old quality is inherited. See source-review.json and cpu-constructor-proof.json for exact source seals and proof.\n')
 out.append(report)
(Q/'priority-six-source-review.json').write_text(json.dumps({'status':'PASS','rows':out},indent=2)+'\n')
print(json.dumps([{'candidate':r['candidate'],'status':r['status'],'cpu_proof_sha256':r['cpu_proof_sha256']} for r in out],indent=2))
