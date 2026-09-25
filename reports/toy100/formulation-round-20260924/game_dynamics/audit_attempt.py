"""Reuse frozen grading and verify this attempt's mechanism-specific receipts."""
import collections,copy,gzip,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root.parents[2];attempt=repo.parent
supervisor=attempt/'supervisor.md'
if supervisor.exists():
    directive=supervisor.read_text();print('SUPERVISOR: '+directive.strip(),flush=True)
    if 'STOP' in directive:raise SystemExit('Supervisor STOP')
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sys.path.insert(0,str(root/'prepared/repos/cuda'))
from benchmarks.transfer_suite.protocol import test_verdict

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def receipts():
    results=[];streams={};commands=[json.loads(s) for s in (root/'commands.jsonl').read_text().splitlines()]
    manifest=json.loads((root/'prepared/prepared-sources.json').read_text())['cuda']
    for name,expected in manifest.items():assert sha(root/'prepared/repos/cuda'/name)==expected,name
    for candidate in sorted((root/'candidates').iterdir()):
        if not (root/'runs'/candidate.name).exists():continue
        declaration=json.loads((candidate/'declaration.json').read_text())
        assert declaration['config_sha256']==sha(repo/'configs/toy100/constraints_simple_regularization.json')
        assert declaration['prepared_manifest_sha256']==sha(root/'prepared/prepared-sources.json')
        for name,expected in declaration['file_hashes'].items():assert sha(candidate/name)==expected
        for task in ['mode_hold','vector_unequal_mass']:
            path=root/'runs'/candidate.name/task/'result.json';r=json.loads(path.read_text())
            launch=next(v for v in commands if v['candidate']==candidate.name and v['gate']==task)
            assert launch['declaration_sha256']==sha(candidate/'declaration.json')
            assert declaration['declared_utc']<=launch['utc']
            assert r['worker_sha256']==declaration['file_hashes']['probe.py']
            assert r['backend']=='cuda' and not r['cpu_random'] and r['spec']['steps']==1200
            assert r['environment']['CUDA_VISIBLE_DEVICES']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
            fixture=repo/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/task
            assert r['initialization_fixture_sha256']==sha(fixture/'initial-values.pt')
            with gzip.open(fixture/'result.json.gz','rt') as f:initial=json.load(f)
            assert r['proof']['initial_optimizers']==initial['proof']['initial_optimizers']
            verdict=test_verdict(r['spec'],r['result']);assert verdict==r['verdict']
            assert verdict['convergence']['complete'] and verdict['convergence']['observations']==24
            correction=r['proof']['game_correction'];counts=sorted(v['calls'] for v in r['proof']['optimizers'].values())
            if candidate.name=='extragradient':
                assert counts==[2400,2400] and r['proof']['adam_calls']==4800
                assert correction['roles']=={'predictor':{'d':1200,'g':1200},'corrector':{'d':1200,'g':1200},'ordinary':{'d':0,'g':0}}
                assert correction['accepted_optimizer_updates']==2400 and correction['rng_replays']==1200 and not correction['pending']
            else:
                assert counts==[1200,1200] and r['proof']['adam_calls']==2400
                assert correction['extra_forward_evaluations']==correction['extra_backward_evaluations']==correction['extra_optimizer_updates']==0
                stream=(r['randomness']['calls'],r['randomness']['elements'],r['randomness']['sha256'])
                if task in streams:assert stream==streams[task]
                streams[task]=stream
            if candidate.name=='optimistic_critic':
                assert r['final_state_sha256']==sha(path.parent/'final-state.pt')
                for c in correction['adam_state_clocks']:assert c['minimum']==c['maximum']==1200
                saved=torch.load(path.parent/'final-state.pt',map_location='cpu',weights_only=False)
                for optimizer in saved['optimizers']:
                    assert all(int(s['step'])==1200 for s in optimizer['state']['state'].values())
                    assert all(('game_previous_direction' in s)==(optimizer['role']=='d') for s in optimizer['state']['state'].values())
            results.append(dict(candidate=candidate.name,gate=task,status=r['status'],source_and_declaration_verified=True,CPU_fixture_verified=True,frozen_regrade_verified=True,updates_verified=True))
    assert len(results)==12
    return dict(executed_gates=12,candidates=6,counts=dict(collections.Counter(r['status'] for r in results)),one_evaluation_streams_equal=True,prepared_files_verified=len(manifest),checks=results)

def critic_scope():
    torch.set_default_device('cuda');torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    name='optimistic_critic';spec=importlib.util.spec_from_file_location(name,root/'candidates'/name/'mechanism.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.Correction()
    p=torch.nn.Parameter(torch.tensor([1.,-2.]));q=torch.nn.Parameter(torch.tensor([.5,1.]));ref=torch.nn.Parameter(p.detach().clone())
    g=torch.optim.Adam([p],lr=.03,betas=(0.,.999),foreach=False,fused=False);g._game_role='g'
    d=torch.optim.Adam([q],lr=.03,betas=(0.,.999),foreach=False,fused=False);d._game_role='d'
    reference=torch.optim.Adam([ref],lr=.03,betas=(0.,.999),foreach=False,fused=False)
    for grad in ([.2,-.1],[-.4,.3],[.1,.5]):
        q.grad=torch.tensor(grad);c.step(d,torch.optim.Adam.step)
        p.grad=torch.tensor(grad);ref.grad=p.grad.clone();c.step(g,torch.optim.Adam.step);reference.step()
        assert torch.equal(p,ref)
        assert torch.equal(g.state[p]['exp_avg_sq'],reference.state[ref]['exp_avg_sq'])
    assert 'game_previous_direction' not in g.state[p] and 'game_previous_direction' in d.state[q]
    clone=torch.nn.Parameter(q.detach().clone());other=torch.optim.Adam([clone],lr=.03,betas=(0.,.999),foreach=False,fused=False);other._game_role='d';other.load_state_dict(copy.deepcopy(d.state_dict()))
    q.grad=torch.tensor([.3,-.7]);clone.grad=q.grad.clone();c.step(d,torch.optim.Adam.step);m.Correction().step(other,torch.optim.Adam.step);assert torch.equal(q,clone)
    return dict(generator_bitwise_original=True,critic_memory_roundtrip_exact=True,CUDA_memory=True)

rows=[]
for name,test in [('critic_scope_and_resume',critic_scope),('executed_gate_receipts',receipts)]:
    started=time.perf_counter();row=dict(candidate='regression',gate=name,status='PASS',artifact=str(root/'attempt-audit.json'))
    try:row['metrics']=test()
    except Exception:row.update(status='ERROR',metrics={},error=traceback.format_exc())
    row['seconds']=time.perf_counter()-started;rows.append(row)
    with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    visible={**row,'metrics':{k:v for k,v in row.get('metrics',{}).items() if k!='checks'}};print(json.dumps(visible),flush=True)
(root/'attempt-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(r['status']!='PASS' for r in rows))
