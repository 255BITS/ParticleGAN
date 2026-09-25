"""One foreground CUDA worker; wraps the retained benchmark, never grades itself."""
import argparse, datetime, hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('candidate'); p.add_argument('gates',nargs='+'); a=p.parse_args()
root=Path(__file__).resolve().parent; repo=root.parents[2]; attempt=repo.parent
supervisor=attempt/'supervisor.md'
if supervisor.exists():
    directive=supervisor.read_text(); print('SUPERVISOR: '+directive.strip(),flush=True)
    if 'STOP' in directive: raise SystemExit('Supervisor STOP')
c=root/'candidates'/a.candidate; declaration=json.loads((c/'declaration.json').read_text())
for name, expected in declaration['file_hashes'].items():
    assert hashlib.sha256((c/name).read_bytes()).hexdigest()==expected, name
assert hashlib.sha256((repo/'configs/toy100/constraints_simple_regularization.json').read_bytes()).hexdigest()==declaration['config_sha256']
env=os.environ.copy(); env.pop('PYTHONPATH',None); env.pop('LD_PRELOAD',None)
env.update(CUDA_VISIBLE_DEVICES='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0', ATEN_CPU_CAPABILITY='avx2', MKL_ENABLE_INSTRUCTIONS='AVX2', ONEDNN_MAX_CPU_ISA='AVX2', DNNL_MAX_CPU_ISA='AVX2')
for gate in a.gates:
    output=root/'runs'/a.candidate/gate; output.parent.mkdir(parents=True,exist_ok=True)
    fixture=repo/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/gate/'initial-values.pt'
    assert fixture.exists(), str(fixture)
    command=[sys.executable,'-u',str(c/'probe.py'),'--repo',str(root/'prepared/repos/cuda'),'--config',str(repo/'configs/toy100/constraints_simple_regularization.json'),'--task',gate,'--backend','cuda','--initial-state',str(fixture),'--output',str(output)]
    log=output.with_suffix('.log'); started=time.perf_counter()
    launch=dict(candidate=a.candidate,gate=gate,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,shell_command=' '.join(f'{k}={shlex.quote(env[k])}' for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED','ATEN_CPU_CAPABILITY','MKL_ENABLE_INSTRUCTIONS','ONEDNN_MAX_CPU_ISA','DNNL_MAX_CPU_ISA'])+' '+shlex.join(command),log=str(log),declaration_sha256=hashlib.sha256((c/'declaration.json').read_bytes()).hexdigest())
    with (root/'commands.jsonl').open('a') as f:f.write(json.dumps(launch)+'\n')
    print(json.dumps(dict(event='START',candidate=a.candidate,gate=gate,log=str(log))),flush=True)
    with log.open('x') as stream:
        process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    path=output/'result.json'
    if path.exists():
        r=json.loads(path.read_text()); result=r.get('result',{}); live=result.get('live',{})
        metrics={k:v for k,v in live.items() if not isinstance(v,(list,dict))}
        metrics.update(adam_calls=r.get('proof',{}).get('adam_calls'),optimizer_calls=[v['calls'] for v in r.get('proof',{}).get('optimizers',{}).values()],frozen_steps=r.get('spec',{}).get('steps'),convergence=r.get('verdict',{}).get('convergence'))
        row=dict(candidate=a.candidate,gate=gate,status=r['status'],seconds=r['seconds'],metrics=metrics,artifact=str(path))
        if r.get('error'):row['error']=r['error']
    else:row=dict(candidate=a.candidate,gate=gate,status='ERROR',seconds=time.perf_counter()-started,metrics={},artifact=str(log),error=f'exit {process.returncode}, no result')
    with (attempt/'tests.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (root/'batch.log').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)
