"""Close the single exited owner CPU mechanics proof using raw byte guards."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    for name in ('CPU-RECEIPT.json','READY.json','FROZEN.json'):
        assert not (HERE/name).exists(),name
    source=read(HERE/'SOURCE-FROZEN.json')
    approval=read(HERE/'ROOT-GO.json')
    launch=read(HERE/'LAUNCH-attempt1.json')
    exited=read(HERE/'EXIT-attempt1.json')
    result_path=HERE/'cpu-contract-attempt1/result.json'
    result=read(result_path)
    assert result['status']=='PASS' and result['device']=='cpu' and not result['cuda_initialized']
    assert result['genuine_newlaw_checkpoints']
    assert exited['exit_code']==0 and exited['pid']==launch['pid']
    assert exited['startticks']==launch['startticks']
    assert launch['command']==source['cpu_command']
    assert len(result['records'])==2 and {r['case'] for r in result['records']}=={'grid','toy'}
    assert all(r['status']=='PASS' and all(r['checks'].values()) for r in result['records'])
    files=dict(source['source_and_input_sha256'])
    for path,digest in approval['independent_review_sha256'].items():
        assert sha(path)==digest,path
        files[path]=digest
    for record in result['records']:
        for path,digest in record['output_sha256'].items():
            assert sha(path)==digest,path
            files[path]=digest
    for path in (HERE/'SOURCE-FROZEN.json',HERE/'ROOT-GO.json',HERE/'LAUNCH-attempt1.json',
                 HERE/'EXIT-attempt1.json',HERE/'numerical-attempt1.log',result_path,Path(__file__)):
        files[str(path)]=sha(path)
    for path in (HERE/'REPORT.md',):
        if path.exists():files[str(path)]=sha(path)
    for path,digest in files.items():assert sha(path)==digest,path
    receipt=dict(status='PASS',scope='one actual fresh backend10 CPU mechanics reaction each grid/toy input',
        closed_utc=datetime.now(timezone.utc).isoformat(),result_path=str(result_path),result_sha256=sha(result_path),
        source_freeze_sha256=sha(HERE/'SOURCE-FROZEN.json'),numerical_invocations=1,per_case_reactions=1,
        exit_code=0,records=result['records'],protected_sha256=dict(sorted(files.items())),
        genuine_backend10_states=True,old_backend_state_loaded=False,CPU_only=True,new_quality_emissions=0,
        quality_verdict=None,root_actual_API_pending=True,root_actual_CUDA_pending=True)
    with (HERE/'CPU-RECEIPT.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    files[str(HERE/'CPU-RECEIPT.json')]=sha(HERE/'CPU-RECEIPT.json')
    ready=dict(status='FROZEN_CPU_QUALIFIED',closed_utc=datetime.now(timezone.utc).isoformat(),
        backend_schema=10,trainer_schema=5,mean_schema=2,package_root=source['package_root'],
        package_source_sha256=source['package_source_sha256'],package_sha256=source['package_sha256'],
        config_path=source['config_path'],config_sha256=source['config_sha256'],
        source_and_input_sha256=source['source_and_input_sha256'],evidence_sha256=dict(sorted(files.items())),
        cpu_receipt_sha256=sha(HERE/'CPU-RECEIPT.json'),cpu_command=source['cpu_command'],gpu_command=source['gpu_command'],
        affected_scope='even-only linear output moment frame, schema2 metadata and exact prepared output packet; learned critic support/count chart unchanged',
        changed_original_modules=source['changed_original_modules'],added_modules=source['added_modules'],unchanged_original_modules=28,
        root_composed_source_review_pending=True,root_composed_API_review_pending=True,root_CUDA_pending=True,
        quality_verdict=None,quality_acceptance='both unchanged final learned toy and full canonical grid gates remain required',
        no_covariance_certificate=True,default_package_promoted=False)
    with (HERE/'READY.json').open('x') as f:f.write(json.dumps(ready,indent=2)+'\n')
    files[str(HERE/'READY.json')]=sha(HERE/'READY.json')
    for path,digest in files.items():assert sha(path)==digest,path
    frozen=dict(status='FROZEN_CPU_QUALIFIED',files=dict(sorted(files.items())),
        ready_sha256=sha(HERE/'READY.json'),CPU_receipt_sha256=sha(HERE/'CPU-RECEIPT.json'),
        numerical_process_closed=True,log_closed=True,quality_verdict=None)
    with (HERE/'FROZEN.json').open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=ready['status'],guarded_files=len(files),
        ready_sha256=sha(HERE/'READY.json'),frozen_sha256=sha(HERE/'FROZEN.json'),
        CPU_receipt_sha256=sha(HERE/'CPU-RECEIPT.json'),package_sha256=ready['package_sha256'])))


if __name__=='__main__':main()
