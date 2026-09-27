"""Independent source/receipt binding, standard library only."""
from pathlib import Path
import ast,hashlib,itertools,json,zipfile
E=Path(__file__).resolve().parents[1];R=E/'precision-vector-review';B=E/'port-source/new-init-vector-screen';V=E/'port-source/new-init-dv12-unequal-mass'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();bh=lambda b:hashlib.sha256(b).hexdigest()
assert sha(B/'bundle-sha256.json')=='27e107244b13da511680c50d77a62e660d54dea360d10d708e844c04a0d1a7d1'
for name,want in read(B/'bundle-sha256.json').items():assert sha(B/name)==want,name
proof=read(B/'host-source-proof.json');specs=read(B/'task-specs.json');fresh=(B/'frozen_vector_host.py').read_text();old_frozen=(V/'frozen_vector_host.py').read_text()
def span(text,name):
 n=next(n for n in ast.parse(text).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
 return ast.get_source_segment(text,n)
for d in proof['definitions']:assert span(fresh,d['definition'])==span(old_frozen,d['definition'])
assert (B/'receipts.py').read_bytes()==(V/'receipts.py').read_bytes()
authorities=[]
for a in proof['task_authorities']:
 assert sha(Path(a['source_zip']))==a['source_zip_sha256']
 assert sha(Path(a['declaration_path']))==a['declaration_sha256']
 d=read(Path(a['declaration_path']));local=specs[a['task']]
 assert local=={'spec':d['original_spec'],'discriminator_card':d['card']}
 differences={k:[local['spec'].get(k),d['spec'].get(k)] for k in set(local['spec'])|set(d['spec']) if local['spec'].get(k)!=d['spec'].get(k)}
 assert set(differences)<={'d_hidden','d_layers','fourier','research_discriminator'}
 assert not differences or local['discriminator_card'] is not None
 for name in ('target_scale','sample_target','score_samples','test_verdict','requirements'):
  text=span(fresh,name)
  assert all(repr(k) not in text and ('"'+k+'"') not in text for k in differences)
 with zipfile.ZipFile(a['source_zip']) as z:
  assert z.read('benchmarks/transfer_suite/shared_critic_research.py')==(B/'frozen_shared_critic.py').read_bytes()
  original=z.read('benchmarks/transfer_suite/vector_tasks.py').decode()
  for name in ('resolve','target_scale','sample_target','score_samples'):
   actual=span(fresh,name)
   expected=span(original,name)
   if name=='score_samples':expected=expected.replace('torch.Generator().manual_seed(991)','torch.Generator(device=torch.get_default_device()).manual_seed(991)')
   assert actual==expected,(a['task'],name)
  assert span(fresh,'SimpleMLPDiscriminator')==span(z.read('lib/toy_models.py').decode(),'SimpleMLPDiscriminator')
 authorities.append(dict(task=a['task'],source_zip_sha256=a['source_zip_sha256'],declaration_sha256=a['declaration_sha256'],original_spec_exact=True,card_exact=True,promoted_cfg_differences=differences,effective_host='Promoted critic constructed directly from exact card. The four D-only metadata overrides are unused by target/scorer and generator; no unconditional fallback.'))
# The full source was read: callback is materialized once before probes. Verify that ordering structurally.
for candidate in ('api-rp12','api-rp14','api-rp15'):
 code=(E/'port-source'/candidate/'package/particlegan/game_update.py').read_text();node=next(n for n in ast.parse(code).body if getattr(n,'name',None)=='joint_step')
 assert isinstance(node.body[0],ast.If) and ast.unparse(node.body[0].test)=='callable(generator_real)'
 assert ast.unparse(node.body[0].body[0])=='generator_real = generator_real()'
assert 'expected = [math.ceil(i * cfg[\'steps\'] / 24)' in (B/'vector_screen.py').read_text()
rows=[]
for candidate,task in itertools.product(('api-rp12','api-rp14','api-rp15'),specs):
 C=R/(candidate+'-'+task+'-cpu');P=C/'cpu-preflight.json';p=read(P);D=E/'port-source'/candidate/'candidate-declaration.json';d=read(D)
 assert p['status']=='PASS_CPU_ZERO_STEP' and p['cuda_initialized'] is False and p['initializer_rng_neutral'] and p['repeated_without_rng_reset']
 assert p['task']==task and p['candidate_declaration_sha256']==sha(D)
 assert p['initial_material']['completed_steps']==0
 expected=dict(d['resolved_recipe']);expected.update(num_particles=256,z_dim=4,batch_size=128)
 assert p['initial_material']['recipe']==expected
 assert all(v['step']==0 and v['step_device']==v['parameter_device']=='cpu' for vs in p['optimizer'].values() for v in vs)
 for name,want in p['source_sha256'].items():
  if name.startswith('particlegan/'):assert d['package_sha256'][name]==want
  elif name=='candidate-declaration.json':assert sha(D)==want
  else:assert sha(B/name)==want
 with zipfile.ZipFile(C/'source.zip') as z:
  for name,want in p['source_sha256'].items():assert bh(z.read(name))==want
 row=dict(status='PASS_SOURCE_AND_CPU_VECTOR_INITIALIZATION',candidate=d['candidate'],task=task,harness_manifest_sha256=sha(B/'bundle-sha256.json'),candidate_declaration_sha256=sha(D),cpu_receipt=str(P),cpu_receipt_sha256=sha(P),source_zip_sha256=sha(C/'source.zip'),recipe=expected,critic_card=specs[task]['discriminator_card'],optimizer_parameter_counts={k:len(v) for k,v in p['optimizer'].items()},checks=['Own fresh task initialization repeats complete serialized model/optimizer/precision/reference state without RNG reset','Public prior R2std.5 and RNG-neutral network initialization; exact original CPU prior-first construction','All three frozen critic routes retained; registered frequencies/scales and custom branches retained','No forward/backward/optimizer step/GPU; package eager counters remain on parameter devices','Separate data0/latent1/penalty2; whole-step serial public transaction; callback G-real drawn once','All original data/latent dry receipts checked after accepted steps; learner RNG not advanced by dry host','Isolated evaluation latent990/target991/projection992/output2303; all24ceil observations/final5 original bounds','Two field evaluations per accepted update, game/precision/applied rates retained'],scope='Initialization/source clearance only. No historical quality inherited.',launch_eligibility='NOT_ELIGIBLE: valid own bars4 failure0/24; preserve completed CPU preparation only' if candidate in ('api-rp12','api-rp14') else 'CONDITIONAL_ON_OWN_REMAINING_IMAGE_SURVIVAL')
 (R/(candidate+'-'+task+'-source-review.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
review=dict(status='PASS_SOURCE_AND_INITIALIZATION_WITH_QUALITY_PRUNING',harness_manifest_sha256=sha(B/'bundle-sha256.json'),source_authorities=authorities,rows=rows,scorer_device_binding='Same reviewed DV12 explicit generator device for target991 under host-only CUDA scope; no global Generator hook.',limits=['No GPU qualification in this review.','RP12/RP14 already rejected by own bars4; their CPU proofs completed before pruning and do not authorize further qualification.','RP15 remains conditional on remaining images.','Canonical original_spec JSON plus exact promoted card is behaviorally equal to promoted spec; the four D-only metadata keys are not falsely claimed identical.'])
(R/'independent-vector-review.json').write_text(json.dumps(review,indent=2)+'\n')
(R/'independent-vector-review.md').write_text('# Precision vector preparation review\n\nPASS for source and18 own guarded CPU initializations across RP12/RP14/RP15 and the six frozen tasks. Each complete non-RNG serialized state repeats without a seed reset. Priorstd.5, public initialization, eager parameter-device counters and every custom critic branch/buffer are retained. No forward pass, backward pass, optimizer update or GPU operation was run.\n\nThe three critic routes match the frozen authority: original SimpleMLP for two_broad/spiral, public BatchDistance for unequal_mass, and complete shared research critics for width/anisotropic/overlap. Task JSON retains original_spec plus the exact promoted card; the old promoted spec adds only four discriminator metadata overrides which the direct-card constructor already applies and which target/scorer code never reads. Data/scorer source and all bounds remain unchanged apart from the previously reviewed explicit device for evaluation generator991.\n\nSeparate data0/latent1/penalty2 streams, two real batches per accepted update, one materialized G-real callback, two game fields, all24ceil observations and final-five scoring are preserved. Spiral remains1600 updates; the others1200. Runtime must verify every dry data/latent receipt and initialized CUDA model bytes against its own CPU proof.\n\nRP12 and RP14 are already rejected by valid own bars4 failures; preserve their completed preparation without further GPU qualification. Only RP15 remains conditional on its own remaining image outcomes. No quality result is inferred or inherited.\n')
print(json.dumps(dict(review=str(R/'independent-vector-review.json'),sha256=sha(R/'independent-vector-review.json'),rows=len(rows))))
