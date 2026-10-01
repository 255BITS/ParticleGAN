"""Watch only original post-save sealed milestones with a pre-frozen CPU parser."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from compare_saved import HERE,ROOT,STEPS,compare,sha,source_guard

RESULT=ROOT/'validation-cb64-ra9/learned/training/toy/CB64-RA9/result.json'
OUT=HERE/'accepted-attempt1'


def log(event,**values):
    print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values),allow_nan=True),flush=True)


def seals(variant):
    path=ROOT/f'validation-cb64-{variant.lower()}/logs/learned-toy-CB64-{variant}.log'
    if not path.exists():return {}
    data=path.read_bytes();complete=data[:data.rfind(b'\n')+1];result={}
    for line in complete.splitlines():
        try:record=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if record.get('event')=='training_checkpoint' and record.get('problem')=='toy' and record.get('variant')==f'CB64-{variant}':
            result[record['step']]=dict(log_path=str(path),line_sha256=hashlib.sha256(line).hexdigest(),
                event='training_checkpoint',step=record['step'],utc=record.get('utc'),post_checkpoint_save=True)
    return result


def main():
    source_guard();OUT.mkdir(exist_ok=False)
    log('parity_watch_start',pid=os.getpid(),source_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),output=str(OUT))
    rows=[]
    while len(rows)<len(STEPS):
        available={v:seals(v) for v in ('RA8','RA9')}
        for step in STEPS:
            if any(r['step']==step for r in rows) or any(step not in available[v] for v in available):continue
            paths=[ROOT/f'validation-cb64-{v.lower()}/learned/training/toy/CB64-{v}/checkpoint-{step:04d}.pt' for v in available]
            if not all(p.exists() for p in paths):continue
            out=OUT/f'checkpoint-{step:04d}.json'
            receipt=compare(step,{v:available[v][step] for v in available},out)
            summary=dict(step=step,status=receipt['status'],unexpected_difference_count=receipt['unexpected_difference_count'],
                receipt_path=str(out),receipt_sha256=sha(out),paired_average_lease_live=receipt['paired_average_lease_live'],
                actual_cells=receipt['resolution_metadata_candidate']['actual_cells'],
                metric_rank=receipt['resolution_metadata_candidate']['metric_rank'],
                coherent_rows=receipt['resolution_metadata_candidate']['coherent_rows'])
            rows.append(summary)
            frozen=dict(status='FROZEN_SAVED_ENDPOINT_CPU_PARSE_PARITY',files={str(out):sha(out)},
                helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),checkpoint_sha256=receipt['checkpoint_sha256'])
            with (OUT/f'checkpoint-{step:04d}-FROZEN.json').open('x') as target:target.write(json.dumps(frozen,indent=2)+'\n')
            (OUT/'live-index.json').write_text(json.dumps(dict(status='WATCHING',records=rows,numerical_replay=False,quality_verdict=None),indent=2)+'\n')
            log('saved_endpoint_compared',**summary)
        if len(rows)==len(STEPS):break
        if RESULT.exists() and json.loads(RESULT.read_text()).get('status')=='ERROR':
            raise RuntimeError('Toy job ended ERROR before all endpoints were sealed')
        time.sleep(10)
    value=dict(status='COMPLETE_EXACT_TRAINING_PARITY' if all(r['status']=='EXACT_LEGACY_TRAINING_PARITY' for r in rows) else 'COMPLETE_WITH_TRAINING_DIVERGENCES',
        records=sorted(rows,key=lambda r:r['step']),quality_verdict=None,cpu_only=True,new_training_steps=0,new_emissions=0,
        numerical_replay=False,helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'))
    with (OUT/'summary.json').open('x') as target:target.write(json.dumps(value,indent=2)+'\n')
    log('parity_watch_complete',status=value['status'],endpoints=len(rows))


if __name__=='__main__':
    try:main()
    except Exception as failure:
        OUT.mkdir(exist_ok=True)
        error=dict(status='HELPER_ERROR',type=type(failure).__name__,message=str(failure),utc=datetime.now(timezone.utc).isoformat())
        with (OUT/'ERROR.json').open('x') as target:target.write(json.dumps(error,indent=2)+'\n')
        log('parity_watch_error',**error)
        raise
