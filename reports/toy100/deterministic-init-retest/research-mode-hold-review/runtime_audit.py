"""Read retained research KA2 tensors/source without importing Torch."""
from pathlib import Path
import importlib.util,hashlib,io,json,math,struct,zipfile
E=Path(__file__).resolve().parents[1];B=E/'research-mode-hold-preparation'
s=importlib.util.spec_from_file_location('raw',E/'audit_public3_runtime.py');raw=importlib.util.module_from_spec(s);s.loader.exec_module(raw)
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=next(Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927').glob('*/*/*/repo/reports/reviewed-probe-output/*/research-ka2/result.json')).parent
plan=read(B/'source-plan.json');proof=read(E/'research-mode-hold-review/cpu-constructor-proof.json');seal=read(B/'manifest.json');source=read(p/'prepared-source-sha256.json');result=read(p/'result.json');init=read(p/'initialization-receipt.json');runtime=read(p/'runtime.json')
assert read(p/'source-plan.json')==plan and read(p/'prepared-source-manifest.json')==seal
assert source==dict(source_zip_sha256=sha(p/'prepared-source.zip'),prepared_manifest_sha256=sha(B/'manifest.json'),reviewed_cpu_proof_sha256=sha(E/'research-mode-hold-review/cpu-constructor-proof.json'))
with zipfile.ZipFile(p/'prepared-source.zip') as z:
 assert set(z.namelist())==set(seal['files'])|{'manifest.json','reviewed-cpu-proof.json'}
 for n,h in seal['files'].items():assert raw.sha(z.read(n))==h,n
 assert raw.sha(z.read('manifest.json'))==source['prepared_manifest_sha256']
 assert raw.sha(z.read('reviewed-cpu-proof.json'))==source['reviewed_cpu_proof_sha256']
 with zipfile.ZipFile(io.BytesIO(z.read('historical-runtime-source.zip'))) as h:
  assert {n:raw.sha(h.read(n)) for n in h.namelist()}==plan['historical_runtime_files']
for name,path in runtime['historical_imports'].items():
 path=Path(path);rel=str(path.relative_to(plan['historical_runtime']));assert sha(path)==plan['historical_runtime_files'][rel]
assert (runtime['torch'],runtime['cuda'],runtime['gpu'])==('2.13.0+cu126','12.6','NVIDIA RTX A6000')
assert runtime['deterministic'] and not runtime['tf32'] and runtime['serial_backward'] and runtime['learner_default_device']=='cuda:0'
assert runtime['threads']==runtime['interop_threads']==1
assert init['source_plan_sha256']==sha(B/'source-plan.json') and not init['historical_tensor_fixture_loaded'] and not init['old22_scores_inherited']
assert init['transform']==proof['transform'] and init['serial_context_restored']
def load(path):
 with zipfile.ZipFile(path) as z:
  n=next(n for n in z.namelist() if n.endswith('/data.pkl'));prefix=n[:-8];d=raw.Reader(io.BytesIO(z.read(n))).load()
  def binary(t):
   kind,key,device,count=t.storage;width={'FloatStorage':4,'ByteStorage':1}[kind];v=z.read(prefix+'data/'+key);assert len(v)==count*width
   stride=1
   for length,st in reversed(list(zip(t.shape,t.stride))):assert length<=1 or st==stride;stride*=length
   return v[t.offset*width:(t.offset+math.prod(t.shape))*width]
  def material(v):
   if isinstance(v,raw.Tensor):return dict(shape=list(v.shape),dtype={'FloatStorage':'torch.float32','ByteStorage':'torch.uint8'}[v.storage[0]],sha256=raw.sha(binary(v)))
   if isinstance(v,dict):return {str(k):material(x) for k,x in v.items()}
   if isinstance(v,(list,tuple)):return [material(x) for x in v]
   assert v is None or isinstance(v,(float,int,str,bool));return v
  clocks=[]
  if isinstance(d,dict):
   for opt in d.get('optimizers',[]):
    clocks.append([{'step':struct.unpack('<f',binary(st['step']))[0],'device':st['step'].storage[2],'moment_device':st['exp_avg'].storage[2]} for st in opt['state'].values()])
  return material(d),clocks
initial,_=load(p/'new-initial-state.pt');final,clocks=load(p/'final-state.pt');fixture=read(E/'mode-hold-source/fixture-receipt.json')
assert {r:initial[r] for r in ['generator','critic','prior']}==proof['all_initial_material']
assert {r:init['initial_material'][r] for r in ['generator','critic','prior']}==proof['all_initial_material']
for k in ['cpu_rng','data_rng']:assert initial[k]==fixture['expected_initial'][k],k
assert initial['cuda_rng']==[fixture['expected_initial']['cuda_rng']]
assert init['initial_material']['data_rng_sha256']==initial['data_rng']['sha256']
assert list(final['streams'].values())==[fixture['expected_final_data_rng']]
assert sum(map(len,clocks))==17 and all(c==dict(step=1200.,device='cuda:0',moment_device='cuda:0') for role in clocks for c in role)
assert result['checkpoint']['sha256']==sha(p/'final-state.pt')
rows=result['result']['observations'];assert [r['step'] for r in rows]==list(range(50,1201,50))
passing=[r for r in rows if r['modes']>=8 and r['hq']>=.9];assert not passing
assert result['status']==result['verdict']['status']=='FAIL' and result['result']['convergence']['passing_suffix']==0
assert result['result']['live']==dict(modes=rows[-1]['modes'],hq=rows[-1]['hq'])
mech=read(p/'mechanism-receipt.json');assert mech['calls']==mech['critic_steps']==1200 and mech['pure_a_calls']==799 and mech['first_blend_call']==800 and mech['blend_calls']==401
assert mech['ema_updates']+mech['ema_skips']==401 and mech['ema_reseeds']==0
out=dict(status='PASS',scope='Independent retained source/runtime/raw-state audit, no Torch/model/GPU execution',quality_status='FAIL',source=str(p),artifacts={f.name:sha(f) for f in p.iterdir() if f.is_file()},prepared_source_sha256=source,initial_cuda_material_matches_cpu_proof=True,all_initial_global_and_shared_rng_equal_frozen=True,final_shared_cursor_equal_frozen=True,initialization_fixture_loaded=False,observations=24,passing_observations=0,passing_suffix=0,final=result['result']['live'],seconds=result['seconds'],optimizer_clocks=clocks,mechanism=mech,limits=['Historical custom research loop with original eager CUDA Adam state, not public GANTrainer.','Inner raw host verdict PASS has a looser diagnostic rule and is not authoritative. Outer frozen eight-mode/final-five gate FAIL is authoritative.','No per-update data batch receipts were emitted by this historical host; exact sealed sampler/update source plus original initial/final shared cursors and retained randomness receipt are verified, not an invented1200-row equality claim.','No old22 quality inherited; fresh result only.'])
(E/'research-ka2-runtime-audit.json').write_text(json.dumps(out,indent=2)+'\n')
(E/'research-ka2-runtime-audit.md').write_text('''# Research KA2 fixed initialization runtime audit\n\nThe new-initialization research KA2 result is a valid FAIL: 0/24 observations pass, final7/8 modes with HQ .995361328125 and no passing suffix. All24 observations use the strict eight-mode/HQ.9 gate; the nested old host diagnostic labelled PASS is looser and does not determine qualification. Runtime cost47.1656s is retained without a comparative speed claim.\n\nIndependent stdlib inspection verified the complete39-file prepared-source seal, the retained200-file historical runtime, exact original mechanism/config/latent/response/probe, initializer source, transform and independent CPU proof. Raw captured CUDA construction tensors, including critic buffers, exactly equal the reviewed deterministic public-initializer CPU material. Initial global/shared RNG and final shared sampling cursor equal the frozen host. The original constructor draws remain; archived random model weights were never loaded.\n\nThe historical learner remains a custom research loop. Its raw final17 eager CUDA scalar Adam clocks are1200 with CUDA moments; no public lazy-counter normalization was introduced. KA2 records799 pure-A calls then401 blend calls beginning800, with376 reference EMA updates and25 skips. Complete final checkpoints and original randomness receipts are preserved. This old host emits no1200-row batch-receipt table, so the audit verifies exact sealed source and boundary cursors rather than claiming an unrecorded per-batch comparison.\n\nNo Torch import, model execution, training, replay or GPU work was performed by this audit. Full hashes and clocks are in research-ka2-runtime-audit.json. This fresh failure inherits none of the old22 quality results.\n''')
print(json.dumps({k:out[k] for k in ['status','quality_status','observations','passing_observations','final','seconds']}))
