"""CPU-only post-exit validation of existing root CUDA trace artifacts."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import ast
import hashlib
import json
from pathlib import Path
import struct
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'integration/review/training-regression/post-ra8-quality/positive-lease-replay'
GPU=ROOT/'integration/review/ra8-positive-lease-gpu'
PRIOR=HERE.parent/'positive-lease-review'
CHECKER=ROOT/'integration/review/audit_learned.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def require(condition,message):
    if not condition:raise AssertionError(message)
torch.set_num_threads(1)

def main():
    assert not (HERE/'receipt.json').exists() and not torch.cuda.is_initialized()
    maps=dict(read(PRIOR/'receipt.json')['reviewed_sha256'])
    phase_ready=ROOT/'quality/positive-lease-phase/READY.json'
    assert sha(phase_ready)=='0320ea8caae255352d008b702f39365aae7a7c6d78eaf061e048c3bd5f560fe3'
    maps.update(read(phase_ready)['source_sha256'])
    maps.update({str(p):sha(p) for p in sorted(GPU.rglob('*')) if p.is_file()})
    maps.update({str(p):sha(p) for p in (phase_ready,CHECKER,PRIOR/'receipt.json',PRIOR/'FROZEN.json',Path(__file__))})
    for p,d in maps.items():assert sha(p)==d,p
    (HERE/'INPUT-FROZEN.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_CPU_ARTIFACT_LOAD',sha256=maps),indent=2)+'\n')
    rng=torch.get_rng_state().clone()
    result=read(GPU/'result.json');phase=read(GPU/'PHASE-RESULT.json');launch=read(GPU/'LAUNCH.json')
    assert phase['status']==result['status']=='PASS' and phase['returncode']==0
    assert phase['source_integrity']=='VALID' and phase['quality_verdict'] is None and phase['numerical_parallelism']==1
    assert phase['result_sha256']==sha(GPU/'result.json') and phase['phase_ready_sha256']==sha(phase_ready)
    assert result['source_freeze_sha256']==sha(OWNER/'SOURCE-FROZEN.json')
    assert result['state_excluded_fields']==[] and result['training_updates']==result['critic_forwards']==result['quality_metric_evaluations']==0
    assert result['generator_forwards']==4 and result['chunk_rows']==256 and result['chunks_per_branch']==2
    assert result['generated_rows_per_branch']==512 and result['original_scorer_seed']==314259
    assert result['actual_paired_average']['coherent_rows']==977 and result['actual_paired_average']['eligible'] is True
    process=Path(f"/proc/{launch['pid']}/stat")
    if process.exists():
        fields=process.read_text().rsplit(')',1)[1].split()
        assert fields[19]!=str(launch['startticks']) or fields[0]=='Z','GPU evidence writer still active'
    assert result['runtime']['physical_gpu']['uuid']==phase['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    assert result['runtime']['deterministic_algorithms'] is True and result['runtime']['tf32_matmul'] is False
    tree=ast.parse(CHECKER.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('load_cpu','digest','rng_placement')]
    namespace=dict(torch=torch,hashlib=hashlib,struct=struct,require=require)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CHECKER),'exec'),namespace)
    load,digest,placement=[namespace[name] for name in ('load_cpu','digest','rng_placement')]
    checkpoint,devices=load(result['checkpoint']);saved=checkpoint['trainer']
    assert sha(result['checkpoint'])==result['checkpoint_sha256']
    expected=digest(saved,devices);sections={name:digest(value,devices) for name,value in saved.items()}
    findings=[];branch_records=[]
    for branch in result['branches']:
        path=Path(branch['endpoint']);assert sha(path)==branch['endpoint_sha256']
        artifact,labels=load(path);state=artifact['trainer']
        assert artifact['source_checkpoint_sha256']==result['checkpoint_sha256']
        assert digest(state,labels)==expected
        assert branch['state_sections']==[sections,sections]
        placement(state,labels)
        traces=artifact['sample_traces'];assert len(traces)==2
        assert [digest(t,labels) for t in traces]==branch['traces']
        assert [{name:digest(value,labels) for name,value in t.items()} for t in traces]==branch['trace_components']
        for t in traces:
            assert t['rows'].shape==(256,) and t['rows'].dtype==torch.int64
            assert t['latent'].shape==t['perturbed_latent'].shape==(256,128)
            assert t['noisy_output'].shape==(256,2) and t['sigma']>0
            assert torch.equal(saved['models']['ema_prior']['z'][t['rows']],t['latent'])
        assert digest(traces[0]['scorer_stream_after'],labels)==digest(traces[1]['scorer_stream_before'],labels)
        assert digest(artifact['scorer_stream'],labels)==digest(traces[1]['scorer_stream_after'],labels)==branch['final_scorer_stream_sha256']
        assert branch['serving']['retained_fast_matches_saved'] and branch['serving']['live_matches_paired_average']
        assert branch['serving']['fast_differs_from_average']
        for cache in branch['caches']:
            assert cache['warm_reuse'] and cache['chart_absent'] and cache['heads_absent']
            assert cache['current_parameter_version_matches'] and cache['orders']==8
            assert cache['work']['max_query_rows']<=256 and cache['work']['max_candidates']<=72
        findings.append(dict(branch=branch['branch'],state_exact_original_GPU_typed_fingerprint=True,
            all23_serialized_sections=len(sections),trace_components_exact=True,scorer_cursor_chain_exact=True,
            RNG=placement(state,labels)))
        branch_records.append((artifact,labels))
    assert result['branches'][0]['traces']==result['branches'][1]['traces']
    for chunk in range(2):
        assert result['branches'][0]['caches'][chunk]['axis_orders_sha256']==result['branches'][1]['caches'][chunk]['axis_orders_sha256']
    middle,labels=load(GPU/'intermediate-reload.pt')
    assert digest(middle['trainer'],labels)==expected
    assert middle['source_checkpoint_sha256']==result['checkpoint_sha256']
    assert digest(middle['scorer_stream'],labels)==result['branches'][1]['trace_components'][0]['scorer_stream_after']
    for p,d in maps.items():assert sha(p)==d,p
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    out=dict(status='PASS',evidence='VALID',scope='Independent CPU post-exit original GPU-typed trace/state audit, no new numerical run',
        root_result_sha256=sha(GPU/'result.json'),root_phase_result_sha256=sha(GPU/'PHASE-RESULT.json'),
        owner_ready_sha256=sha(OWNER/'READY.json'),actual_saved_stamp=result['actual_paired_average'],
        full_saved_state_fingerprint=expected,serialized_fields_excluded=[],branch_checks=findings,
        exact_midstream_saved_reload=True,exact_sampled_EMA_rows=True,exact_trace_continuation=True,
        original_noise_law_retained=True,geometry_cache_orders_exact=True,
        cpu_only=True,cuda_initialized=False,global_CPU_RNG_unchanged=True,
        reviewer_training_updates=0,reviewer_draws=0,reviewer_forwards=0,quality_verdict=None,
        known_Grid100_quality='FAIL',reviewed_sha256=maps)
    (HERE/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
    (HERE/'REPORT.md').write_text('# Positive saved serving lease: post-execution audit\n\nPASS/VALID. CPU storage loads preserve original device tags and reproduce the original CUDA typed fingerprints for all saved training fields, with no exclusions. Two independent branch endpoints equal the checkpoint exactly. All stored row/code/perturbed-code/noisy-output/scorer-cursor trace fingerprints agree; codes equal saved own EMA rows. Actual saved midstream trainer/scorer cursor matches the first-chunk continuation. Both cache-order fingerprints agree and reported work stays256 rows/72 candidates. Owner/source/runtime/result/closed log guards unchanged.\n\nRoot executed four256-row G calls, zero training and zero quality scoring. Reviewer performed only saved CPU loads and comparisons, zero forwards/draws/CUDA. This mechanical saved977/973 lease passes; original Grid100 remains VALID/FAIL and no quality acceptance is implied.\n')
    print(json.dumps(dict(status='PASS',reviewed_files=len(maps),receipt_sha256=sha(HERE/'receipt.json'))))

if __name__=='__main__':main()
