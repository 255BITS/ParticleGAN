"""CPU saved-artifact pilot and failure controls for the declared API adapter."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import torch
import indexed_adapter as adapter

TASKS=('mode_hold','img_blobs4','vector_unequal_mass')


def main():
    destination=adapter.HERE/'PILOT.json'
    if destination.exists() or adapter.OUTPUT.exists():raise RuntimeError('Pilot output already exists')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    guard=adapter.guard();api=adapter.resolve_declared_api()
    source_paths=[Path(p) for p in adapter.read(adapter.READY)['exact_source_sha256']]
    inputs=[]
    for task in TASKS:
        directory=adapter.SCREEN_ROOT/'runs'/task
        inputs.extend(p for p in directory.rglob('*') if p.is_file())
        inputs.append(adapter.ROOT/'integration/review/ra4-validation-monitor/canonical-receipts/screens/runs'/task/'acceptance-receipt.json')
    before={str(p):adapter.sha(p) for p in inputs+source_paths}
    spec=importlib.util.spec_from_file_location('lane',adapter.SCREEN_ROOT/'lane.py')
    lane=importlib.util.module_from_spec(spec);sys.modules['lane']=lane;spec.loader.exec_module(lane)
    strict,adapted,integrity,proof=adapter.modules(lane)
    originals=[]
    for task in TASKS:
        old=strict.collect(task,require_attempt=True);new=adapted.collect(task,require_attempt=True)
        annotated=adapter.annotate(task,old,new,integrity,proof)
        assert old['primary_status']=='PASS' and old['acceptance_status']=='ERROR'
        assert annotated['acceptance_status']=='PASS' and annotated['canonical_fixture_validity']=='VALID'
        assert len(old['validity_reasons'])==1 and not new['validity_reasons']
        originals.append(dict(task=task,original_strict_status=old['acceptance_status'],adapted_status=new['acceptance_status'],
                              all_other_fields_exact=True,result_sha256=old['result_sha256']))
    task='vector_unequal_mass';result_path=adapter.SCREEN_ROOT/'runs'/task/'result.json'
    actual=adapter.read(result_path);normal_read=adapted.read
    controls=[]
    for name in ('wrong_other_option','wrong_package','wrong_stream','wrong_api_mode','quality_fail'):
        value=deepcopy(actual)
        if name=='wrong_other_option':value['header']['options']['strict_streams']=False
        elif name=='wrong_package':value['header']['package_sha256']='wrong-package'
        elif name=='wrong_stream':value['stream_deviations']=1
        elif name=='wrong_api_mode':value['header']['options']['evaluation_generate']='plain'
        else:value['status']='FAIL'
        adapted.read=lambda path,v=value:deepcopy(v) if Path(path)==result_path else normal_read(path)
        record=adapted.collect(task,require_attempt=True)
        expected='FAIL' if name=='quality_fail' else 'ERROR'
        assert record['acceptance_status']==expected,(name,record['acceptance_status'])
        if name!='quality_fail':assert record['canonical_fixture_validity']=='INVALID' and record['validity_reasons']
        controls.append(dict(name=name,status=record['acceptance_status'],validity=record['canonical_fixture_validity'],
                             reasons=record['validity_reasons'],memory_overlay_only=True))
    adapted.read=normal_read
    started=time.perf_counter()
    log=adapter.HERE/'pilot-monitor.log'
    with log.open('x') as output:
        result=subprocess.run(['/tmp/pr38-default-env/bin/python','-u','-B',str(adapter.HERE/'run_indexed_monitor.py')],
                              stdout=output,stderr=subprocess.STDOUT,check=False,cwd=adapter.ROOT)
    assert result.returncode==0,'Original monitor pilot failed; see pilot-monitor.log'
    summary=adapter.read(adapter.OUTPUT/'summary.json')
    completed=[r for r in summary['records'] if r['acceptance_status']!='PENDING']
    assert set(TASKS)<={r['task'] for r in completed}
    for record in completed:
        assert record['canonical_fixture_validity']=='VALID'
        assert record['acceptance_status']==record['primary_status'] in ('PASS','FAIL')
        annotation=record['indexed_api_expectation_adapter']
        assert annotation['original_strict_status']=='ERROR' and annotation['quality_verdict_unchanged']
        assert annotation['exact_declared_api_difference']['actual']=='indexed'
    identity=adapter.read(adapter.OUTPUT/'CHECKER-IDENTITY.json')
    assert identity['write_redirection_only'] is False and identity['original_canonical_checks_unchanged'] is False
    assert identity['indexed_api_expectation_adapter']['all_other_canonical_checks_unchanged']
    assert before=={str(p):adapter.sha(p) for p in inputs+source_paths},'Frozen input or old ERROR receipt changed'
    adapter.guard();assert not torch.cuda.is_initialized()
    receipt=dict(status='VALID',scope='saved-artifact original collector checks with one declared indexed expectation; no numerical reruns',
        ready_sha256=adapter.sha(adapter.READY),source_guard=guard,api=api,collector_ast_proof=proof,
        original_vs_adapted=originals,negative_controls=controls,monitor_pilot_records=len(completed),
        monitor_output=str(adapter.OUTPUT),monitor_returncode=result.returncode,monitor_wall_seconds=time.perf_counter()-started,
        source_and_input_sha256=before,old_error_receipts_preserved=True,cpu_threads=1,cuda_initialized=False,
        numerical_reruns=0,quality_gates_unchanged=True,script_sha256=adapter.sha(__file__),
        pilot_monitor_log_sha256=adapter.sha(log))
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='VALID',pilot_sha256=adapter.sha(destination),records=len(completed),output=str(destination))),flush=True)


if __name__=='__main__':main()
