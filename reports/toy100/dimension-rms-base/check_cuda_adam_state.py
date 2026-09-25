"""Confirm CUDA scalar state storage preserves baseline noncapturable Adam math."""
import hashlib,json,time
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;attempt=root.parents[3]
steering=(attempt/'supervisor.md').read_text() if (attempt/'supervisor.md').exists() else ''
if 'STOP' in steering:raise SystemExit('Supervisor STOP')
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.perf_counter()
a=torch.nn.Parameter(torch.linspace(-.2,.3,32,device='cuda'));b=torch.nn.Parameter(a.detach().clone())
control=torch.optim.Adam([a],lr=.00425,betas=(0.,.999),foreach=False,fused=False)
checked=torch.optim.Adam([b],lr=.00425,betas=(0.,.999),foreach=False,fused=False)
checked.state[b].update(step=torch.zeros((),device='cuda'),exp_avg=torch.zeros_like(b),exp_avg_sq=torch.zeros_like(b))
for i in range(25):
 gradient=(torch.arange(32,device='cuda',dtype=torch.float32)+1).sin()*(i+1)/25
 a.grad=gradient.clone();b.grad=gradient.clone()
 control.step();checked.step()
 assert torch.equal(a,b)
 for key in ('exp_avg','exp_avg_sq'):assert torch.equal(control.state[a][key],checked.state[b][key])
assert all(v.device.type=='cuda' for v in checked.state[b].values())
row={'candidate':'regression','gate':'cuda_adam_state_equivalence','status':'PASS','seconds':time.perf_counter()-start,'metrics':{'steps':25,'parameter_and_moment_bitwise_parity':True,'all_candidate_state_cuda':True,'gpu':torch.cuda.get_device_name(0),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid)},'artifact':str(root/'cuda-adam-state-check.json'),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(row['artifact']).write_text(json.dumps(row,indent=2)+'\n')
with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))
