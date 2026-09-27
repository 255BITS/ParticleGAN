"""Independent initial source/runtime/frozen-control cross-check; stdlib only."""
from pathlib import Path
import hashlib,importlib.util,json,sys,zipfile
E=Path(__file__).resolve().parents[1];H=E/'dv23-single-shift-preparation';R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('raw',E/'audit_public3_runtime.py');raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
base=Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/20260927T031740Z/dv23-new-init-ring/20260927T031740Z-1983204/repo/reports/reviewed-probe-output/fixed-batch-0ii7edgr')
rows=[]
for alias in ('api-dv2','api-dv3'):
 S=base/(alias+'-single-shift');B=H/alias;P=H/'reviews'/alias/'cpu-constructor-proof.json';D=read(B/'declaration.json');proof=read(P);manifest=read(B/'bundle-sha256.json')
 assert sha(B/'bundle-sha256.json')==proof['manifest_sha256'] and sha(B/'declaration.json')==proof['declaration_sha256']
 assert read(S/'reviewed-cpu-proof.json')==proof
 artifact=read(S/'artifact-sha256.json')
 for n,want in artifact.items():assert sha(S/n)==want,n
 runtime_decl=read(S/'declaration.json');expected_sources={**manifest['files'],'bundle-sha256.json':sha(B/'bundle-sha256.json')}
 assert runtime_decl['source_sha256']==expected_sources
 assert {k:v for k,v in runtime_decl.items() if k not in ('started_utc','source_sha256','status')}=={k:v for k,v in D.items() if k!='status'}
 assert runtime_decl['status']=='DECLARED_BEFORE_EXECUTION'
 with zipfile.ZipFile(S/'source.zip') as z:
  assert set(z.namelist())==set(expected_sources)
  for n,want in expected_sources.items():assert hashlib.sha256(z.read(n)).hexdigest()==want==sha(B/n),n
 runtime=read(S/'runtime.json');expected=D['runtime_expected']
 for k in ('torch','cuda','gpu','threads','interop_threads','deterministic','tf32'):assert runtime[k]==expected[k],k
 assert runtime['default_factory_device']=='cpu'
 assert runtime['execution']==dict(serial_backward=True,scope='full GANTrainer.step')
 assert runtime['environment']['CUBLAS_WORKSPACE_CONFIG']==expected['cublas_workspace']
 for name,r in runtime['pinned_torch_sources'].items():assert sha(Path(r['path']))==r['sha256']==expected['source_hashes'][name]
 for name,r in runtime['imported_package'].items():assert sha(Path(r['path']))==r['sha256']==D['package_sha256']['particlegan/'+Path(r['path']).name]
 assert read(S/'initial-material.json')==proof['all_initial_material']
 assert read(S/'initial-model-cpu-cuda-proof.json')==dict(status='PASS',all_initial_non_rng_state_and_named_parameters_and_buffers_equal=True)
 envelope,clocks=raw.checkpoint(S/'initial-state.pt');trainer=envelope['trainer'];assert clocks==[[],[]]
 state={k:v for k,v in trainer.items() if k not in ('streams','cpu_rng','cuda_rng','device')}
 assert state==proof['all_initial_material']['state']
 assert trainer['schema']==4 and trainer['completed_steps']==0 and trainer['recipe']==D['recipe']
 assert envelope['schema']==1 and envelope['execution']==runtime['execution']
 assert envelope['identity']['bundle_manifest_sha256']==proof['manifest_sha256']
 init=read(S/'initial.json')
 for name,wanted in D['expected_initial_fixture'].items():assert init[name]==wanted,name
 audit=read(S/'initialization-audit.json');assert all(audit['matches'].values()) and audit['adam_state_empty'] is True
 result=read(S/'result.json');assert result['status']=='COMPLETE' and result['metrics']['completed_updates']==4600
 assert result['metrics']['source_and_runtime_assertions_passed'] is True
 frozen=result['metrics']['frozen_control'];assert frozen['checkpoint']==2400 and frozen['updates_after_copy']==0 and frozen['observations']==220
 for label,step in [('main-0001',1),('frozen-2400',2400),('main-4600',4600)]:
  states=read(S/('optimizer-device-proof-'+label+'.json'))
  assert {k:len(v) for k,v in states.items()}=={'generator':9,'critic':8}
  assert all(r['step']==step and r['step_device']=='cpu' and r['parameter_device']==r['exp_avg_device']==r['exp_avg_sq_device']=='cuda:0' for values in states.values() for r in values)
 source=(B/'worker.py').read_text()
 assert 'frozen.load_state_dict(saved["trainer"])' in source and 'assert host.digest(envelope()) == before_control' in source and 'assert own_frozen_digest() == frozen_digest' in source
 row=dict(status='PASS_INITIAL_SOURCE_RUNTIME_AND_FROZEN_CONTROL_RECEIPTS',candidate=D['candidate'],source=str(S),manifest_sha256=proof['manifest_sha256'],cpu_proof_sha256=sha(P),source_zip_sha256=sha(S/'source.zip'),artifact_manifest_sha256=sha(S/'artifact-sha256.json'),checks=['Full sealed source and own CPU proof identity','Exact actualCUDA complete initial material equals independent CPUproof','Raw initial schema4 state equals proof excluding only device/actualRNG states; nativeAdam empty','Original declared global/private/caller RNG and means fixture retained','Pinned runtime/Torch source, CPU factory default and full-step serial mode','Separate frozen control copied at2400; original source asserts main restoration and frozen immutability atall220 checks','Native17CPU clocks verified from retained first/frozen/final receipts, no repair'],limits=['Independent cross-check supports constant_lr_evidence terminal metrics/rawfinal/rate audit; no duplicate quality determination here.','No Torch/GPU/model execution. Frozen invariance evidence is completed own guarded execution, not a new replay.'])
 (R/(alias+'-initial-runtime-crosscheck.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
(R/'summary.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n');assert 'torch' not in sys.modules
print(json.dumps(dict(status='PASS',candidates=[r['candidate'] for r in rows],summary=str(R/'summary.json'),sha256=sha(R/'summary.json'))))
