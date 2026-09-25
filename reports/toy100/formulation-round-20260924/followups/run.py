"""Run three declared compositions, serial gates per candidate, three CUDA workers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip, hashlib, importlib.util, json, os, subprocess, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/ml2/hypergan/ParticleGAN-selected-h-stability-base')
RECORDS=json.loads((ROOT/'batch.json').read_text())
RUNS=[sorted(Path(row['directory']).glob('20*'))[-1] for row in RECORDS]
SOURCE=Path(json.loads((RUNS[0]/'declaration.json').read_text())['source'])
MANIFEST=SOURCE.parents[1]/'prepared-sources.json'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in json.loads(MANIFEST.read_text())['cuda'].items():
    assert sha(SOURCE/name)==h,name
sys.path.insert(0,str(SOURCE))
from benchmarks.transfer_suite.protocol import test_verdict
import torch
# Analytic regularizer check. It does not train a model or consume benchmark RNG.
module_spec=importlib.util.spec_from_file_location('hybrid_check',RUNS[0]/'hybrid_install.py')
m=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(m)
from particlegan.grad_regularizers import GradientPenalty
class Quadratic(torch.nn.Module):
    def __init__(self,w):
        super().__init__(); self.w=torch.nn.Parameter(torch.tensor(w,dtype=torch.float64));self.calls=0
    def forward(self,x):
        self.calls+=1
        return self.w*x.square().sum(dim=1,keepdim=True)
for w in (0.,.05,.4):
    d=Quadratic(w);real=torch.tensor([[1.],[2.]],dtype=torch.float64);fake=torch.tensor([[3.],[4.]],dtype=torch.float64)
    reg=GradientPenalty(arm='a_r1r2',coeff=1.,kappa=1.)
    actual,_=reg.penalty(d,real,fake)
    expected=.5*((2*d.w*real).square().sum(1).mean()+((2*d.w*fake).square().sum(1)+1e-12).sqrt().sub(1).relu().square().mean())
    ga=torch.autograd.grad(actual,d.w,retain_graph=True)[0];ge=torch.autograd.grad(expected,d.w)[0]
    assert torch.allclose(actual,expected,atol=1e-12) and torch.allclose(ga,ge,atol=1e-12)
    assert d.calls==2
m.GradientPenalty.penalty=m._original_penalty
(ROOT/'hybrid-check.json').write_text(json.dumps(dict(status='PASS',cases=3,checks=['penalty value','parameter gradient','two critic forwards','zero slope finite']))+'\n')
env=os.environ.copy()
for key in ('LD_PRELOAD','PYTHONPATH','CODEX_THREAD_ID'):env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATEN_CPU_CAPABILITY='avx2',MKL_ENABLE_INSTRUCTIONS='AVX2',ONEDNN_MAX_CPU_ISA='AVX2',DNNL_MAX_CPU_ISA='AVX2',PYTHONHASHSEED='0')
def worker(run):
    d=json.loads((run/'declaration.json').read_text());name=d['candidate'];rows=[]
    (run/'status.txt').write_text('running\n')
    try:
        for filename,h in d['file_hashes'].items():assert sha(run/filename)==h
        assert sha(MANIFEST)==d['source_manifest_sha256']
        tasks=['mode_hold','vector_unequal_mass']
        for task in tasks:
            output=run/task
            fixture=BASE/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/task/'initial-values.pt'
            command=[sys.executable,str(run/'probe.py'),'--repo',str(SOURCE),'--config',str(run/'config.json'),'--task',task,'--backend','cuda','--initial-state',str(fixture),'--output',str(output)]
            launch=dict(command=command,env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED','ATEN_CPU_CAPABILITY','MKL_ENABLE_INSTRUCTIONS','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA')},utc=datetime.now(timezone.utc).isoformat(),declaration_sha256=sha(run/'declaration.json'))
            with (run/'commands.jsonl').open('a') as f:f.write(json.dumps(launch)+'\n')
            print('START',name,task,flush=True)
            with (run/(task+'.log')).open('w') as log:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            assert code==0,(name,task,code)
            r=json.loads((output/'result.json').read_text())
            baseline=json.loads(gzip.decompress((BASE/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')).read_bytes()))
            assert r['spec']==baseline['spec']
            assert r['status']==test_verdict(r['spec'],r['result'])['status']
            assert r['verdict']==test_verdict(r['spec'],r['result'])
            assert r['backend']=='cuda' and not r['cpu_random']
            assert r['worker_sha256']==d['file_hashes']['probe.py']
            assert r['initialization_fixture_sha256']==sha(fixture)
            assert r['proof']['initial_optimizers']==baseline['proof']['initial_optimizers']
            assert r['randomness']==baseline['randomness']
            assert r['proof']['adam_calls']==2400
            assert sorted(v['calls'] for v in r['proof']['optimizers'].values())==[1200,1200]
            assert all(v['device']=='cuda:0' for v in r['proof']['optimizers'].values())
            row=dict(candidate=name,gate=task,status=r['status'],seconds=r['seconds'],metrics=r['result']['live'],convergence=r['verdict']['convergence'],artifact=str(output/'result.json'),audit='PASS')
            with (run/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            rows.append(row)
            print(json.dumps(row),flush=True)
        (run/'audit.json').write_text(json.dumps(dict(status='PASS',gates=2,source_files=1614,code=True,frozen_verdict=True,initialization=True,randomness=True,cuda_updates=True),indent=2)+'\n')
        (run/'result.md').write_text('Direct composition: '+name+'\n\n'+'\n'.join(x['gate']+': '+x['status'] for x in rows)+'\n\nFull 22 and continuation NOT_RUN. Exact commands, code, declaration and audit retained.\n')
        (run/'status.txt').write_text('completed (inspect result.md; exit 0 does not mean a win)\n')
        (run/'exit-code.txt').write_text('0\n')
    except Exception:
        error=traceback.format_exc();(run/'error.txt').write_text(error)
        (run/'status.txt').write_text('failed (execution or audit error; inspect error.txt)\n')
        print(error,flush=True)
        raise
    return rows
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=sum(list(pool.map(worker,RUNS)),[])
(ROOT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE',len(rows),'gates',flush=True)
