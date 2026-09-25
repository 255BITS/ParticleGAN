import sys,pathlib,time,json,torch
root=pathlib.Path(__file__).resolve().parent;attempt=root.parents[2].parent
steering=(attempt/'supervisor.md').read_text() if (attempt/'supervisor.md').exists() else ''
if 'STOP' in steering:raise SystemExit('Supervisor STOP')
sys.path.insert(0,str(root/'prepared/repos/cuda'));sys.path.insert(0,str(root/'candidates/direct_particle_response'))
import response
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.perf_counter()
class O:pass
p=torch.nn.Parameter(torch.zeros(2,1,device='cuda'));p.grad=torch.tensor([[-1.],[1.]],device='cuda')
o=O();o.param_groups=[dict(params=[p],lr=.0085,betas=(0.,.999),_comparison_prior=True)]
a=response.begin(o);assert o.param_groups[0]['betas']==(0.,.9);response.end(a)
a=response.begin(o);assert abs(o.param_groups[0]['lr']-.017)<1e-8;response.end(a)
assert o.param_groups[0]['lr']==.0085 and o.param_groups[0]['betas']==(0.,.999)
response.prior_ids.add(id(p));a=response.begin(o)
assert not a and o.param_groups[0]['betas']==(0.,.999) and o.param_groups[0]['lr']==.0085
assert torch.equal(p.grad,torch.tensor([[-1.],[1.]],device='cuda')) and torch.count_nonzero(p)==0
row=dict(candidate='regression',gate='direct_particle_structural_scope',status='PASS',seconds=time.perf_counter()-start,metrics=dict(direct_particle_response=True,latent_prior_excluded=True,scheduled_lr_and_betas_restored=True,optimizer_updates=0),artifact=str(root/'direct-mechanism-check.json'))
pathlib.Path(row['artifact']).write_text(json.dumps(row,indent=2)+'\n')
with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))
