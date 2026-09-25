"""Small CUDA checks of optimizer formulas, replay, and accepted update clocks."""
import copy,importlib.util,json,os,time,traceback
from pathlib import Path
import torch
root=Path(__file__).resolve().parent;attempt=root.parents[2].parent
supervisor=attempt/'supervisor.md'
if supervisor.exists():
    message=supervisor.read_text();print('SUPERVISOR: '+message.strip(),flush=True)
    if 'STOP' in message:raise SystemExit('Supervisor STOP')
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_device('cuda')
torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
def correction(name):
    spec=importlib.util.spec_from_file_location(name,root/'candidates'/name/'mechanism.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.Correction()
def parameter(values):return torch.nn.Parameter(torch.tensor(values,dtype=torch.float32))
def adam(p,lr=.1,role=None):
    opt=torch.optim.Adam([p],lr=lr,betas=(0.,.999),foreach=False,fused=False);opt._game_role=role;return opt
def equal(a,b):torch.testing.assert_close(a,b,rtol=1e-6,atol=2e-7)
def optimistic():
    c=correction('optimistic_adam');p=parameter([1.,-2.]);opt=adam(p);previous=None
    for lr,values in [(.1,[.3,-.7]),(.1,[-.2,.4]),(.013,[.8,-.1])]:
        opt.param_groups[0]['lr']=lr;p.grad=torch.tensor(values)
        ref=parameter(p.detach().clone());other=adam(ref,lr);other.load_state_dict(copy.deepcopy(opt.state_dict()));ref.grad=p.grad.clone()
        before=ref.detach().clone();other.step();direction=(ref.detach()-before)/lr
        expected=ref.detach().clone()
        if previous is not None:expected.add_(direction-previous,alpha=lr)
        c.step(opt,torch.optim.Adam.step);equal(p,expected);previous=direction
    restored=parameter(p.detach().clone());replay=adam(restored);replay.load_state_dict(copy.deepcopy(opt.state_dict()))
    p.grad=torch.tensor([.17,.42]);restored.grad=p.grad.clone()
    c.step(opt,torch.optim.Adam.step);correction('optimistic_adam').step(replay,torch.optim.Adam.step);equal(p,restored)
def gradient():
    c=correction('optimistic_gradient');p=parameter([1.,-2.]);opt=adam(p);previous=None
    for values in [[.3,-.7],[-.2,.4],[.8,-.1]]:
        raw=torch.tensor(values);p.grad=raw
        ref=parameter(p.detach().clone());other=adam(ref);other.load_state_dict(copy.deepcopy(opt.state_dict()));ref.grad=raw if previous is None else 2*raw-previous
        other.step();c.step(opt,torch.optim.Adam.step);equal(p,ref);assert p.grad is raw;previous=raw
        equal(opt.state[p]['exp_avg_sq'],other.state[ref]['exp_avg_sq'])
def predictive():
    c=correction('predictive_critic');p=parameter([1.]);q=parameter([2.]);g=adam(p,role='g');d=adam(q,role='d')
    q.grad=-p.detach().clone();c.step(d,torch.optim.Adam.step);equal(q,torch.tensor([2.2]));p.grad=q.detach().clone()
    c.step(g,torch.optim.Adam.step);equal(q,torch.tensor([2.1]));assert not c.pending and c.restores==1

def lookahead():
    c=correction('joint_lookahead');p=parameter([1.]);q=parameter([2.]);g=adam(p,role='g');d=adam(q,role='d')
    for step in range(5):
        q.grad=torch.tensor([-1.]);c.step(d,torch.optim.Adam.step)
        if step==4:equal(q,torch.tensor([2.5]))
        p.grad=torch.tensor([1.]);c.step(g,torch.optim.Adam.step)
    equal(p,torch.tensor([.75]));equal(q,torch.tensor([2.25]));assert c.synchronizations==1

def extragradient():
    c=correction('extragradient');p=parameter([1.]);q=parameter([2.]);g=adam(p,role='g');d=adam(q,role='d')
    stream=torch.Generator(device='cuda').manual_seed(923);draws=[];global_draws=[]
    def sgd(opt):
        with torch.no_grad():
            for group in opt.param_groups:
                for v in group['params']:
                    v.add_(v.grad,alpha=-group['lr']);opt.state[v]['updates']=opt.state[v].get('updates',0)+1
    for phase in c.passes():
        draws.append(torch.randn(8,generator=stream));global_draws.append(torch.randn(8))
        q.grad=-p.detach().clone();c.step(d,sgd)
        equal(q,torch.tensor([2. if phase=='predictor' else 2.1]))
        p.grad=q.detach().clone();c.step(g,sgd)
    equal(p,torch.tensor([.79]));equal(q,torch.tensor([2.08]));equal(draws[0],draws[1]);equal(global_draws[0],global_draws[1]);assert g.state[p]['updates']==d.state[q]['updates']==1
    c=correction('extragradient');p=parameter([1.]);q=parameter([2.]);g=adam(p,role='g');d=adam(q,role='d')
    for _ in range(2):
        for phase in c.passes():
            q.grad=-p.detach().clone();c.step(d,torch.optim.Adam.step);p.grad=q.detach().clone();c.step(g,torch.optim.Adam.step)
    assert g.state[p]['step'].item()==d.state[q]['step'].item()==2
    assert c.calls==8 and c.receipt()['accepted_optimizer_updates']==4

rows=[]
for name,test in [('optimistic_direction_decay_resume',optimistic),('optimistic_gradient_preconditioning',gradient),('predictive_critic_restoration',predictive),('joint_lookahead_sync',lookahead),('extragradient_math_rng_update_counts',extragradient)]:
    start=time.perf_counter();row=dict(candidate='regression',gate=name,status='PASS',metrics={},artifact=str(root/'mechanism-checks.json'))
    try:test()
    except Exception:row.update(status='ERROR',error=traceback.format_exc())
    row['seconds']=time.perf_counter()-start;rows.append(row);print(json.dumps(row),flush=True)
    with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
(root/'mechanism-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(r['status']!='PASS' for r in rows))
