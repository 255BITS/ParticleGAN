#!/usr/bin/env python3
"""Prepared exact research-KA2 screen; requires independent CPU proof first.

Never run by the source-audit process. Intended for one authorized external
worker only, using the original research learner and new public initialization.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys
import zipfile

sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reviewed-cpu-proof',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((HERE/'source-plan.json').read_text())
    proof=json.loads(a.reviewed_cpu_proof.read_text())
    if proof.get('status')!='PASS' or proof.get('source_plan_sha256')!=sha(HERE/'source-plan.json') or proof.get('bridge_sha256')!=sha(HERE/'initialization_bridge.py'):
        raise RuntimeError('matching independent CPU source/constructor proof required')
    required=('all_initial_tensors_match_public_host','repeat_without_rng_reset','constructor_rng_cursor_preserved','initializer_rng_neutral','historical_prior_registration_preserved','all_bindings_restore_on_exception','batch_distance_scope_explicit')
    if any(proof.get('checks',{}).get(k) is not True for k in required):
        raise RuntimeError('CPU proof is incomplete')
    if a.output.exists():raise RuntimeError('fresh output directory required')
    seal=json.loads((HERE/'manifest.json').read_text())
    for rel,want in seal['files'].items():
        if sha(HERE/rel)!=want:raise RuntimeError('prepared source changed: '+rel)
    runtime=Path(plan['historical_runtime'])
    for rel,want in plan['historical_runtime_files'].items():
        if sha(runtime/rel)!=want:raise RuntimeError('historical runtime changed: '+rel)
    source=HERE/'ka2-source'
    for rel,want in plan['candidate_files'].items():
        if sha(source/rel)!=want:raise RuntimeError('research candidate changed: '+rel)
    old_path=list(sys.path);old_argv=list(sys.argv)
    sys.path[:0]=[str(source),str(runtime)]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(key,'1')
    # All imports below occur only inside the authorized external execution.
    import torch
    from initialization_bridge import load_initializer,bind_mode_hold
    from benchmarks.locked_shared import mode_hold
    if sha(Path(mode_hold.__file__))!=plan['frozen_host_sha256']:
        raise RuntimeError('host came from an unreviewed runtime')
    expected=plan['runtime_expected']
    assert str(torch.__version__)==expected['torch']
    assert torch.version.cuda==expected['cuda'] and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)==expected['gpu']
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    public=load_initializer(HERE/'initializer-authority',plan['initializer_package'])
    captures=[]
    def capture(generator,critic,prior,stream):
        assert str(torch.get_default_device())=='cuda:0'
        assert str(stream.device)=='cuda:0'
        assert all(str(p.device)=='cuda:0' and p.dtype==torch.float32 for m in (generator,critic,prior) for p in m.parameters())
        assert torch.get_num_threads()==torch.get_num_interop_threads()==1
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        assert not torch.autograd.is_multithreading_enabled()
        def tensors(model):
            return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state=dict(generator=tensors(generator),critic=tensors(critic),prior=tensors(prior),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),data_rng=stream.get_state())
        torch.save(state,a.output/'new-initial-state.pt')
        def h(t):return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        material={role:{k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=h(v)) for k,v in state[role].items()} for role in ['generator','critic','prior']}
        assert material==proof['all_initial_material'],'actual CUDA initial tensors differ from reviewed public CPU constructor proof'
        material['data_rng_sha256']=h(state['data_rng'].cpu());captures.append(material)
        runtime_receipt=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),threads=1,interop_threads=1,deterministic=True,tf32=False,matmul_precision=torch.get_float32_matmul_precision(),serial_backward=True,learner_default_device='cuda:0',optimizer_state='Original research probe eager CUDA scalar counters; not native public lazy Adam',historical_imports={k:str(v.__file__) for k,v in sys.modules.items() if k.split('.')[0] in ('particlegan','benchmarks','lib') and getattr(v,'__file__',None)},environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')})
        (a.output/'runtime.json').write_text(json.dumps(runtime_receipt,indent=2)+'\n')
    before=torch.autograd.is_multithreading_enabled();exit_code=0
    try:
        sys.argv=[str(source/'probe.py'),'--repo',str(runtime),'--config',str(source/'config.json'),'--task','mode_hold','--backend','cuda','--output',str(a.output)]
        with bind_mode_hold(mode_hold,public,plan['train_mode_hold_sha256'],capture) as transformed:
            with torch.autograd.set_multithreading_enabled(False):
                try:runpy.run_path(str(source/'probe.py'),run_name='__main__')
                except SystemExit as done:exit_code=int(done.code or 0)
    finally:
        sys.path[:]=old_path;sys.argv[:]=old_argv
        if a.output.exists():
            with zipfile.ZipFile(a.output/'prepared-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for rel in sorted(seal['files']):archive.write(HERE/rel,rel)
                archive.write(HERE/'manifest.json','manifest.json')
                archive.write(a.reviewed_cpu_proof,'reviewed-cpu-proof.json')
            (a.output/'prepared-source-sha256.json').write_text(json.dumps(dict(source_zip_sha256=sha(a.output/'prepared-source.zip'),prepared_manifest_sha256=sha(HERE/'manifest.json'),reviewed_cpu_proof_sha256=sha(a.reviewed_cpu_proof)),indent=2)+'\n')
        if torch.autograd.is_multithreading_enabled()!=before:raise RuntimeError('serial context leaked')
    if len(captures)!=1:raise RuntimeError('expected exactly one complete initialized construction')
    receipt=dict(candidate='RESEARCH-KA2-new-init',initialization_scope='Exact standalone public factory plus keys0/1; original research learner unchanged.',source_plan_sha256=sha(HERE/'source-plan.json'),initializer_commit=plan['initializer_commit'],transform={k:v for k,v in transformed.items() if k!='source'},initial_material=captures[0],historical_tensor_fixture_loaded=False,serial_context_restored=True,old22_scores_inherited=False,scope='RESEARCH_HOST; not a publicGANTrainer quality result')
    (a.output/'initialization-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.output/'source-plan.json').write_bytes((HERE/'source-plan.json').read_bytes())
    (a.output/'prepared-source-manifest.json').write_bytes((HERE/'manifest.json').read_bytes())
    print(json.dumps({'result':str(a.output/'result.json'),'initialization_receipt':str(a.output/'initialization-receipt.json'),'exit_code':exit_code}),flush=True)
    raise SystemExit(exit_code)

if __name__=='__main__':main()
