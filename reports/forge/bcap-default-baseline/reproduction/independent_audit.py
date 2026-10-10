"""Read-only verification of the saved frozen ordinary pair; no qualification writes."""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, json, math, subprocess, sys, torch
ROOT=Path('/home/martyn/dev/ParticleGAN-bcap-projection-baseline-review')
A=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-default-adoption-20261010')
OUT=ROOT/'reports/forge/bcap-default-baseline'
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result
from experiments.forge.api import resolve_public_recipe
from dataclasses import asdict
from particlegan import get_recipe,Recipe
from PIL import Image
FROZEN='d378734f40b09ce223a389e8f54a9783ec6a0c75'
DIGEST='6a225fcdd6922cdad37c9c947e163fb6f164f6ad4390293a3d3091b8f741ce44'
def require(b,m):
 if not b: raise ValueError(m)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w=module('independent_ordinary_workflow',OUT/'workflow.py')
p=w.shared.publisher(ROOT);p.ROLES=w.ROLES
reg=read_json(OUT/'registration.json');spec=read_json(OUT/'spec.json')
p.required_questions=lambda root:reg['original_requirements']
results=read_json(OUT/'results.json');audit=read_json(OUT/'audit.json');media=read_json(OUT/'media/index.json')
require(w.head(ROOT)==FROZEN,'HEAD changed before saved-only independent audit')
require(results['source_commit']==audit['source_commit']==FROZEN and results['source_digest']==audit['source_digest']==DIGEST,'publication source differs')
require(results['registration_sha256']==file_hash(OUT/'registration.json'),'registration differs')
require(audit['results_file_sha256']==file_hash(OUT/'results.json') and audit['task_results_sha256']==stable_hash(results['task_results']),'publication hashes differ')
require(reg['spec_sha256']==file_hash(OUT/'spec.json'),'spec differs')
require(read_json(A/'progress.json')['phase']=='ordinary_complete','original queue not complete')
data=p.collect(argparse.Namespace(repository=ROOT,queue=A/'queue',progress=A/'progress.json',diagnostic_queue=None,diagnostic_progress=None,allow_partial=False))
require(data['cells']==results['task_cells'],'published cells differ from original queue/certificates')
require([e['item'] for e in data['final']]==results['task_results'],'published compact results differ from originals')
require(data['accounting']==results['accounting'],'published accounting differs from original charges')
require([dict(a['compact']) for a in data['attempts'].values()]==results['paid_attempt_history'],'paid attempt history differs')
actual=module('independent_saved_ordinary_audit',ROOT/'reports/forge/bcap-develop-integration/audit.py')
sources,requests=actual.source_bindings(data,ROOT)
require(len(sources)==1 and sources[0]['source_digest']==DIGEST and sources[0]['source_commits']==[FROZEN],'actual sources differ')
require(sources[0]['scientific_files_verified']==1206,'unexpected scientific manifest count')
changes=subprocess.check_output(['git','diff','--name-only',FROZEN],cwd=ROOT,text=True).splitlines()
require(not set(changes)&set(data['scopes'][0]['submissions']['control']['request']['source']['files']),'scientific tracked source changes')
recipe_delta=None
for context in data['scopes']:
 recipes={}
 for role,sub in context['submissions'].items():
  req=sub['request'];require(w.summary(req)==reg['arms'][role],f'{role}: actual submission differs from frozen admission')
  require(req['request_id']==read_json(A/'progress.json')['requests'][role],f'{role}: request id differs')
  recipes[role]=asdict(resolve_public_recipe(req['candidate']))
  require(stable_hash(recipes[role])==stable_hash(spec['resolved_recipes'][role]),f'{role}: effective recipe differs')
  require(req['through_tier']==3 and req['execution_policy']['mode']=='complete_current_tier' and req['protocol']['seed']==0,'ordinary/seed contract differs')
 recipe_delta={k:{role:recipes[role][k] for role in w.ROLES} for k in recipes['control'] if recipes['control'][k]!=recipes['candidate'][k]}
 require(recipe_delta=={'constraint_geometry_mode':{'control':'none','candidate':'direction_blend'}},'unexpected trainer delta')
 require(asdict(get_recipe('bcap'))==recipes['candidate'],'named preset differs from actual candidate')
require(Recipe().constraint_geometry_mode=='none','plain inactive default changed')
required={r['task'] for r in reg['original_requirements'] if r['importance']=='required' and r['qualification_tier']<=2}
expected={(r,t) for r in w.ROLES for t in required}
entries={(e['item']['role'],e['row']['task_id']):e for e in data['final']}
require(expected<=set(entries) and len(required)==27 and len(entries)==56,'54 required and2 diagnostic cells missing/duplicated')
require(all(entries[k]['row']['gate_status'] in {'PASS','FAIL'} for k in expected),'unresolved required cell')
recomputed=[];receipts={}
for aid,a in data['attempts'].items():
 require(not a['result'].get('retry_of'),'unregistered repeat/retry')
 receipts[aid]={'canonical_result_hash':a['compact']['provenance']['canonical_result_hash'],
  'original_files_sha256':{k:v['sha256'] for k,v in a['compact']['provenance']['original_files'].items()}}
for key,e in sorted(entries.items()):
 role,tid=key;grade=grade_result(e['task'],e['row'])
 require(grade['status']==e['row']['gate_status'],f'{role}/{tid}: saved numerical regrade differs')
 for field in ('metrics','evaluator_result'):
  if field in grade: require(stable_hash(grade[field])==stable_hash(e['row'][field]),f'{role}/{tid}: {field} differs')
 require(stable_hash({k:v for k,v in e['task'].items() if k not in {'field_ownership','preflight_blockers'}})==reg['task_contracts'][tid],f'{tid}: admitted task differs from registration')
 for rel,d in e['task']['evaluation'].get('sources',{}).items():
  require(e['request']['source']['files'][rel]==d==file_hash(ROOT/rel),f'{tid}: evaluator source differs')
 recomputed.append({'role':role,'task_id':tid,'attempt_id':e['item']['attempt_id'],'gate_status':grade['status'],'saved_numeric_grade_sha256':stable_hash(grade)})
# Independently compare actual checkpoint contents, rather than trusting audit booleans.
pairs=[];direct=[];inferred=[]
for tid in sorted(required):
 left,right=(entries[r,tid] for r in w.ROLES)
 require(left['saved'] is not None and right['saved'] is not None,f'{tid}: final checkpoint missing')
 require(actual.equal(actual.metadata(left),actual.metadata(right)),f'{tid}: initial/prior proof differs')
 lb,ls=actual.non_eval(left['saved']);rb,rs=actual.non_eval(right['saved'])
 require(actual.equal(lb,rb) and actual.equal(ls,rs),f'{tid}: named non-eval actual states differ')
 steps={r:entries[r,tid]['item']['provenance_checkpoint']['completed_steps'] for r in w.ROLES}
 require(len(set(steps.values()))==1,f'{tid}: unmatched completed prefixes')
 batch={r:entries[r,tid]['row']['evidence'].get('data_sha256') for r in w.ROLES}
 require(batch['control']==batch['candidate'],f'{tid}: present direct batch hash differs')
 has_hash=batch['control'] is not None
 (direct if has_hash else inferred).append(tid)
 for e in (left,right):
  guards=e['row']['evidence'].get('guards',{})
  require(guards.get('unintended_rng_deviations',0)==0,'RNG guard deviation')
  require(all(x.get('unintended_rng_deviations',0)==0 for x in e['row']['evidence'].get('rng_audits',[])),'RNG audit deviation')
  if 'applied' in e['saved']:
   applied=e['row'].get('applied',e['row']['evidence'].get('applied'))
   require(stable_hash(applied)==stable_hash(e['saved']['applied']),'consumed applied packet differs')
 proof=next(x for x in audit['saved_state_comparisons'] if x['task_id']==tid and x['saved_variant']=='final')
 require(set(proof['completed_roles'])==set(w.ROLES) and proof['initialization_and_prior_equal'] and proof['named_training_bindings_equal'] and proof['all_declared_arms_present'] and proof['consumed_non_eval_streams_and_batches_equal'],'shared state proof unverified')
 require(proof['saved_state_references']=={r:entries[r,tid]['item']['provenance_checkpoint'] for r in w.ROLES} and proof['completed_steps']==steps,'shared checkpoint references differ')
 pairs.append({'task_id':tid,'completed_steps':steps,'initial_prior_sha256':state_digest(actual.metadata(left)),
  'non_eval_bindings_sha256':stable_hash(lb),'non_eval_states_sha256':state_digest(ls),
  'stored_batch_hash_proof':batch['control'] if has_hash else None,
  'batch_identity_basis':'direct stored rolling hash equality' if has_hash else 'identical actual non-eval/data RNG states and bindings plus identical frozen deterministic host/task sampling law; no stored batch-byte hash'})
require(not audit['unavailable_complete_states'],'unexpected incomplete actual state')
# All six saved clock branches also retain actual same starts, budgets and consumed non-eval streams.
clock=[]
for role in w.ROLES:
 e=entries[role,'clockfree_audit_measurement_v1']
 variants=dict(actual.saved_variants(e));clock.append({'role':role,'variants':list(variants),'proof_file_sha256':next(iter(variants.values()))['item']['provenance_checkpoint']['artifact_sha256']})
 if role=='control':clock_reference=variants
 else:
  for name,v in variants.items():
   first=clock_reference[name];require(actual.equal(actual.metadata(first),actual.metadata(v)),f'clock {name}: initial/prior differ')
   fb,fs=actual.non_eval(first['saved']);vb,vs=actual.non_eval(v['saved'])
   require(actual.equal(fb,vb) and actual.equal(fs,vs),f'clock {name}: actual consumed states differ')
restored,uninterrupted=w.own_producers(data,actual)
require(restored==audit['own_checkpoint_producers'] and uninterrupted==audit['own_uninterrupted_groups'],'own producer report differs')
selected_producer_pairs=[]
for tid in sorted({x['parent_task_id'] for x in restored}):
 selected={}
 for role in w.ROLES:
  e=entries[role,tid];descriptor=e['row']['evidence']['checkpoint']
  path=Path(e['row']['evidence']['artifact_root'])/descriptor['path']
  require(file_hash(path)==descriptor['sha256'],'selected producer bytes differ')
  saved=torch.load(path,map_location='cpu',weights_only=False)
  require(state_digest(saved)==descriptor['state_sha256'],'selected producer state differs')
  selected[role]=(dict(e,saved=saved),dict(descriptor,completed_steps=descriptor.get('completed_steps',saved.get('trainer',{}).get('completed_steps'))))
 left,right=(selected[r][0] for r in w.ROLES)
 require(actual.equal(actual.metadata(left),actual.metadata(right)),f'{tid}: selected producer initialization/prior differs')
 lb,ls=actual.non_eval(left['saved']);rb,rs=actual.non_eval(right['saved'])
 require(actual.equal(lb,rb) and actual.equal(ls,rs),f'{tid}: selected producer consumed states differ')
 require(selected['control'][1]['completed_steps']==selected['candidate'][1]['completed_steps'],f'{tid}: selected producer budgets differ')
 selected_producer_pairs.append({'task_id':tid,'completed_steps':selected['control'][1]['completed_steps'],
  'actual_non_eval_states_sha256':state_digest(ls),'initialization_prior_equal':True,'named_consumed_states_equal':True,
  'own_selected_checkpoint_sha256':{r:selected[r][1]['sha256'] for r in w.ROLES}})
word=[]
for role in w.ROLES:
 consumer=entries[role,'five_word_joint_hold'];producer=entries[role,'five_word_joint_smoke']
 continuity=consumer['row']['evidence']['continuity'];psteps=producer['row']['evidence']['checkpoint']['completed_steps'];csteps=consumer['item']['provenance_checkpoint']['completed_steps']
 require(continuity['prefix_steps']==psteps==834 and continuity['parent_confirmed_step']==834 and csteps==4834,'word own selected prefix differs')
 require(continuity['same_recipe_prior_architecture'] is True and continuity['restored_exactly'] is True and continuity['history_reset'] is False,'word continuation contract differs')
 require(continuity['parent_candidate_revision']==consumer['request']['candidate_revision']==producer['request']['candidate_revision'],'cross-arm word producer')
 word.append({'role':role,'producer_attempt_id':producer['item']['attempt_id'],'consumer_attempt_id':consumer['item']['attempt_id'],
  'producer_checkpoint_sha256':continuity['parent_checkpoint_sha256'],'producer_state_sha256':continuity['parent_state_sha256'],
  'producer_compatibility_key':continuity['parent_compatibility_key'],'prefix_steps':psteps,'producer_full_provenance_steps':producer['item']['provenance_checkpoint']['completed_steps'],'final_steps':csteps,'restored_exactly':True,'history_reset':False})
# Verify bytes and their existing recorded input trajectories without generating media.
require(len(media['media'])==results['actual_training_gifs']==audit['actual_training_gifs']==56,'all56 media required')
seen=set();gif_hashes={}
for m in media['media']:
 k=(m['role'],m['task_id']);require(k in entries and k not in seen,'media cell duplicated/foreign');seen.add(k)
 gif=OUT/m['gif'];require(gif.is_relative_to(OUT) and file_hash(gif)==m['gif_sha256'],'GIF hash differs')
 with Image.open(gif) as image: require(image.n_frames==m['frames'] and image.n_frames>=2,'GIF framecount differs')
 require(m['optimizer_updates_added']==m['sampling_draws_added']==0,'publication executed updates/draws')
 for path,d in m['source_inputs'].items(): require(file_hash(Path(path))==d,'media original saved source differs')
 e=entries[k];require(m['recorded_grade']==e['row']['gate_status'],'media recorded gate differs')
 if m['kind']=='actual_training_saved_observations_gif':
  curve=e['row']['evidence'].get('observations',e['row']['evidence'].get('dense'))
  if e['task']['adapter']=='clockfree_audit':
   proof_path=Path(e['row']['evidence']['artifact_root'])/'comparisons.pt'
   proof=torch.load(proof_path,map_location='cpu',weights_only=True)
   initial=proof['initial']['trainer']['models']['G']
   parameter=next(key for key,value in initial.items() if value.is_floating_point() and value.numel())
   curve=[{'step':i+1,**{name:float((proof['trajectories'][name][i]['trainer']['models']['G'][parameter]-initial[parameter]).abs().mean()) for name in proof['trajectories']}} for i in range(e['task']['execution']['probe_steps'])]
  require(stable_hash(curve)==m['observations_sha256'],f'{k}: media observation curve differs')
 gif_hashes[f'{k[0]}/{k[1]}']=m['gif_sha256']
require(seen==set(entries),'completed cell has no media')
paid=sum(c['selected_paid_seconds'] for c in data['accounting']);require(math.isclose(paid,audit['paid_seconds'],abs_tol=1e-8),'paid total differs')
require(paid<=96000 and all(c['execution_retries']==0 and not c['pending_work'] for c in data['accounting']),'ceiling/repeat/pending violation')
t1={r:{e['row']['task_id'] for e in data['final'] if e['item']['role']==r and e['item']['tier']==1 and e['item']['importance']=='required' and e['row']['gate_status']=='PASS'} for r in w.ROLES}
t2={r:{e['row']['task_id'] for e in data['final'] if e['item']['role']==r and e['item']['tier']==2 and e['row']['gate_status']=='PASS'} for r in w.ROLES}
require(all(len(v)==6 for v in t1.values()) and t2['control']<t2['candidate'],'frozen winner predicate fails')
blocked=[c for c in data['cells'] if c['tier']==3]
require(len(blocked)==4 and all(c['gate_status']=='BLOCKED' for c in blocked),'ordinary Tier3 veto missing')
review=read_json(A/'independent-source-config-review.json');admission=read_json(A/'independent-admission-audit.json')
for rel,b in review['reviewed_paths'].items(): require(file_hash(ROOT/rel)==b['reviewed_worktree_sha256'],f'reviewed path changed: {rel}')
receipt={
 'schema_version':1,'scope':'independent_saved_ordinary_bcap_default_baseline_audit','status':'PASS',
 'source_commit':FROZEN,'source_digest':DIGEST,'registration_sha256':file_hash(OUT/'registration.json'),
 'results_file_sha256':file_hash(OUT/'results.json'),'task_results_sha256':stable_hash(results['task_results']),
 'shared_audit_sha256':file_hash(OUT/'audit.json'),'media_index_sha256':file_hash(OUT/'media/index.json'),
 'scientific_source_files_verified':1206,'original_certified_attempts_verified':len(data['attempts']),
 'required_tier1_tier2_cells_verified':54,'optional_clock_cells_verified':2,
 'numeric_gates_independently_recomputed':56,'numeric_grade_receipt_sha256':stable_hash(recomputed),
 'original_receipt_hashes_sha256':stable_hash(receipts),'checkpoint_pair_count':27,'saved_checkpoint_pairs':pairs,
 'direct_stored_batch_hash_tasks':direct,'batch_identity_inferred_tasks':inferred,
 'batch_proof_limit':'None/None is not batch-byte proof. For listed inferred tasks, equality is inferred from identical actual checkpointed non-eval/data RNG bindings and states, frozen host/task/data law and equal completed budgets. This does not prove stored bytes directly.',
 'own_checkpoint_producers':restored,'own_uninterrupted_groups':uninterrupted,'exact_word_producer_prefix':word,'selected_own_producer_pair_proofs':selected_producer_pairs,'optional_clock_saved_branches':clock,
 'accounting':data['accounting'],'paid_seconds':paid,'execution_retries':0,'full_paired_reservation_seconds':93240,'paid_ceiling_seconds':96000,
 'actual_training_gifs_verified':56,'gif_bytes_manifest_sha256':stable_hash(gif_hashes),
 'outcomes':{r:{'tier1':{'PASS':len(t1[r]),'total':6},'tier2':{'PASS':len(t2[r]),'FAIL':21-len(t2[r]),'total':21},'tier3':'2 BLOCKED by ordinary prerequisite veto'} for r in w.ROLES},
 'selection_eligibility':{'frozen_predicate_satisfied':True,'control_tier2_passes':sorted(t2['control']),'candidate_tier2_passes':sorted(t2['candidate']),
   'repaired':sorted(t2['candidate']-t2['control']),'regressed':sorted(t2['control']-t2['candidate']),'unknown_tier1_tier2':[],
   'scope':'Best-observed named research preset and sameglobal configured family standard; no calibrated public default or endurance claim.'},
 'effective_recipe_delta':recipe_delta,'named_public_bcap_matches_candidate':True,'plain_Recipe_geometry_inactive':True,
 'source_config_adapter_review':review,'source_config_adapter_review_sha256':file_hash(A/'independent-source-config-review.json'),
 'admission_review_sha256':file_hash(A/'independent-admission-audit.json'),'admission_review':admission,
 'pr_notation_review':'Actual rounded joint displacement after one base clock; conflict iff any protected g_i dot d>0. Average nearest feasible projection and norm(d) downhill unit-gradient mean. Rollback when mean norm<=1e-12 including near opposition; no probes/draws; finite loss decrease not guaranteed.',
 'optimizer_updates_added':0,'sampling_draws_added':0,'qualification_input':False,'board_or_selection_modified':False,
 'audit_script_sha256':file_hash(Path(__file__)),
 'full_independent_evidence_archive':str(A/'independent-original-evidence.json')}
atomic_json(A/'independent-original-evidence.json',{'original_receipts':receipts,'saved_numeric_grades':recomputed,'actual_gif_sha256':gif_hashes,'requests':requests,'source_checks':sources})
atomic_json(OUT/'independent-audit.json',receipt)
atomic_json(A/'independent-audit.json',receipt)
print(json.dumps({'event':'independent_saved_audit_complete','path':str(OUT/'independent-audit.json'),'sha256':file_hash(OUT/'independent-audit.json'),'paid_seconds':paid,'eligibility':receipt['selection_eligibility']}),flush=True)
