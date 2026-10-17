"""Bind completed original CUDA toy artifacts as raw bytes before PT audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
STEPS=(0,100,250,500,750,1000,1250,1500,1750,2000)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,default=ROOT/'validation-cb64-ra11')
    parser.add_argument('--ready',type=Path,default=ROOT/'quality/ra11/READY.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.validation=args.validation.resolve();args.ready=args.ready.resolve();args.output=args.output.resolve()
    assert args.output.is_relative_to(HERE) and not args.output.exists()
    checker=read(HERE/'CHECKER-FROZEN.json')
    files={}
    def bind(path,digest=None):
        p=Path(path).resolve();actual=sha(p)
        assert digest is None or actual==digest,p
        assert str(p) not in files or files[str(p)]==actual,p
        files[str(p)]=actual
    for field in ('files','protected_file_sha256'):
        for name,digest in checker[field].items():bind(name,digest)
    bind(HERE/'CHECKER-FROZEN.json');bind(args.ready)
    ready=read(args.ready)
    assert ready['package_root']==checker['package_root'] and ready['package_sha256']==checker['package_sha256']
    assert ready['backend_schema']==10 and ready['trainer_schema']==5
    for name,digest in ready['numerical_source_sha256'].items():bind(name,digest)
    bind(args.validation/'source-freeze.json')
    lane=read(args.validation/'source-freeze.json')
    for name,digest in lane['local_sources'].items():bind(args.validation/name,digest)
    for name,digest in lane['external_sources'].items():bind(name,digest)
    learned=args.validation/'learned'
    for name in ('SOURCE-FREEZE.json','INPUTS.json','PROTOCOL.md','preparation-receipt.json'):bind(learned/name)
    source=read(learned/'SOURCE-FREEZE.json')
    for name,digest in source['local_source_sha256'].items():bind(learned/name,digest)
    inputs=read(learned/'INPUTS.json')
    for name,digest in inputs['read_only_file_sha256'].items():bind(name,digest)
    run=learned/'training/toy/CB64-RA11'
    result=read(run/'result.json')
    assert result['status']=='COMPLETE' and result['steps']==result['final']['step']==2000
    assert set(result['checkpoint_sha256'])=={f'checkpoint-{step:04d}.pt' for step in STEPS}
    journal=(args.validation/'run.log').read_bytes()
    prefix=bytearray();events=[]
    for line in journal.splitlines(keepends=True):
        prefix.extend(line)
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('event')=='job_complete' and event.get('name')=='learned-toy-CB64-RA11':
            events.append(event);break
    assert len(events)==1,'Completed original RA11 toy event is required before sealing.'
    event=events[0]
    assert event['status']=='COMPLETE' and event['returncode']==0
    assert Path(event['result']).resolve()==(run/'result.json').resolve()
    assert event['result_sha256']==sha(run/'result.json')
    bind(event['log'])
    for path in sorted(run.rglob('*')):
        if path.is_file():bind(path)
    for name,digest in result['checkpoint_sha256'].items():assert files[str((run/name).resolve())]==digest
    args.output.mkdir(parents=True)
    (args.output/'journal-prefix.log').write_bytes(bytes(prefix))
    (args.output/'toy-job-event.json').write_text(json.dumps(event,indent=2)+'\n')
    bind(args.output/'journal-prefix.log');bind(args.output/'toy-job-event.json')
    frozen=dict(status='COMPLETED_TOY_INPUTS_FROZEN',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        validation=str(args.validation),package_root=checker['package_root'],package_sha256=checker['package_sha256'],
        root_ready_path=str(args.ready),root_ready_sha256=sha(args.ready),lane_source_freeze_sha256=sha(args.validation/'source-freeze.json'),
        checker_freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'),steps=list(STEPS),source_and_input_sha256=files,
        checkpoint_sha256=result['checkpoint_sha256'],journal_capture=dict(live_path=str(args.validation/'run.log'),
            captured_prefix_bytes=len(prefix),captured_prefix_sha256=sha(args.output/'journal-prefix.log'),
            completed_event=event,live_journal_not_required_immutable=True),
        numerical_checkpoint_loads_before_seal=0,model_forwards=0,training_updates=0,new_quality_emissions=0,
        quality_verdict=None)
    with (args.output/'INPUTS-FROZEN.json').open('x') as stream:stream.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=frozen['status'],input_freeze_sha256=sha(args.output/'INPUTS-FROZEN.json'),
        guarded_files=len(files),checkpoints=len(STEPS))))

if __name__=='__main__':main()
