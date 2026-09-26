"""Full CUDA API regression under explicitly checkpointed serial_backward=True.
Cold1100 updates, save at1000; separate fresh process restores1000→1100.
This is state-correctness evidence, not a ring-quality promotion or a new policy.
"""
import argparse,hashlib,json,os,time,zipfile
from pathlib import Path
import torch
import worker as w


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['reference','resume']);a=p.parse_args()
 out=w.ROOT/'reports/data-drift-api/runs'/('serial-checkpoint-'+a.mode);out.mkdir(exist_ok=False)
 sources=[Path(__file__),*sorted((w.ROOT/'particlegan').glob('*.py'))]
 declaration=dict(mode=a.mode,purpose=__doc__,serial_backward=True,seed=0,reference_end=1100,checkpoint=1000,
                  dimensions={'particles':20000,'z_dim':2,'batch':2048,'width':96,'layers':3},
                  sources={str(p.relative_to(w.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 (out/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
 with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sources:z.write(p,str(p.relative_to(w.ROOT)))
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 recipe=w.make_recipe('dv1',1100)
 torch.manual_seed(w.SEED)
 generator=w.SimpleMLPGenerator(recipe.z_dim,w.mode_hold.HIDDEN,w.mode_hold.N_HIDDEN,2).to('cuda:0')
 critic=w.SimpleMLPDiscriminator(2,w.mode_hold.HIDDEN,w.mode_hold.N_HIDDEN,w.mode_hold.FOURIER).to('cuda:0')
 trainer=w.GANTrainer(recipe,generator,critic,seed=w.SEED,optimizer_options={'foreach':False,'fused':False},serial_backward=True)
 stream=torch.Generator(device='cuda:0').manual_seed(w.SEED);means=w.mode_hold.ring_means().cuda()
 if a.mode=='resume':
  checkpoint=torch.load(out.parent/'serial-checkpoint-reference/checkpoint.pt',weights_only=False)
  trainer.load_state_dict(checkpoint['trainer']);stream.set_state(checkpoint['real_stream']);means=checkpoint['means']
  assert w.digest(trainer.state_dict())==w.digest(checkpoint['trainer'])
 initial=w.state_receipt(trainer,stream,means)
 (out/'initial.json').write_text(json.dumps(initial,indent=2)+'\n')
 before=torch.autograd.is_multithreading_enabled();start=time.monotonic();observations=[]
 for step in range(trainer.completed_steps+1,1101):
  idx=torch.randint(0,8,(recipe.batch_size,),device='cuda:0',generator=stream)
  real=means[idx]+w.mode_hold.SIGMA*torch.randn(recipe.batch_size,2,device='cuda:0',generator=stream)
  stats=trainer.step(real,collect_stats=step%10==0)
  if step%10==0:
   point=dict(step=step,live=w.measure(trainer,means),ema=w.measure(trainer,means,ema=True),
              losses={k:float(v) for k,v in stats.items() if isinstance(v,torch.Tensor)},policy=trainer.controller.diagnostics())
   observations.append(point)
  if step==1000:w.save_checkpoint(out/'checkpoint.pt',trainer,stream,means)
  if step%100==0:print(json.dumps({'step':step,'mode':a.mode,'seconds':time.monotonic()-start}),flush=True)
 receipt=w.state_receipt(trainer,stream,means)
 w.save_checkpoint(out/'final.pt',trainer,stream,means)
 result=dict(final=receipt,observations=[p for p in observations if p['step']>1000],
             restores_caller_autograd=before==torch.autograd.is_multithreading_enabled(),seconds=time.monotonic()-start)
 if a.mode=='resume':
  reference=json.loads((out.parent/'serial-checkpoint-reference/result.json').read_text())
  result['all_state_exact']=receipt==reference['final']
  result['all_observations_exact']=result['observations']==reference['observations']
  result['cuda_models_and_moments']=all(p.is_cuda for m in (trainer.G,trainer.D,trainer.prior) for p in m.parameters()) and all(v.is_cuda for opt in (trainer.opt_g,trainer.opt_d) for st in opt.state.values() for k,v in st.items() if isinstance(v,torch.Tensor) and k!='step')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 status='PASS' if result['restores_caller_autograd'] and (a.mode=='reference' or (result['all_state_exact'] and result['all_observations_exact'] and result['cuda_models_and_moments'])) else 'FAIL'
 with (w.ROOT.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(dict(candidate='regression',gate='serial_backward_'+a.mode,status=status,seconds=result['seconds'],metrics=result,artifact=str(out)))+'\n')
 print(json.dumps({'status':status,'all_state_exact':result.get('all_state_exact'),'all_observations_exact':result.get('all_observations_exact')}),flush=True)

if __name__=='__main__':main()
