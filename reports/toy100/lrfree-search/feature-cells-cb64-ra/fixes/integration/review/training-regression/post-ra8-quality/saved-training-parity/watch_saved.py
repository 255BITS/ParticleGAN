"""Watch only post-save sealed toy milestones with a pre-frozen CPU helper."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from compare_saved import HERE,ROOT,STEPS,compare,sha,source_guard

LOG=ROOT/'validation-cb64-ra8/logs/learned-toy-CB64-RA8.log'
RESULT=ROOT/'validation-cb64-ra8/learned/training/toy/CB64-RA8/result.json'
OUT=HERE/'accepted-attempt1'


def log(event,**values):
    print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**values),allow_nan=True),flush=True)


def seals():
    if not LOG.exists():return {}
    data=LOG.read_bytes();complete=data[:data.rfind(b'\n')+1]
    result={}
    for line in complete.splitlines():
        try:record=json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
        if record.get('event')=='training_checkpoint' and record.get('problem')=='toy' and record.get('variant')=='CB64-RA8':
            result[record['step']]=dict(log_path=str(LOG),line_sha256=hashlib.sha256(line).hexdigest(),
                event='training_checkpoint',step=record['step'],utc=record.get('utc'),post_checkpoint_save=True)
    return result


def main():
    source_guard();OUT.mkdir(exist_ok=False)
    log('parity_watch_start',pid=os.getpid(),source_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),output=str(OUT))
    rows=[]
    while len(rows)<len(STEPS):
        available=seals()
        for step in STEPS:
            if any(r['step']==step for r in rows) or step not in available:continue
            path=ROOT/f'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-{step:04d}.pt'
            if not path.exists():continue
            out=OUT/f'checkpoint-{step:04d}.json'
            receipt=compare(step,available[step],out)
            summary=dict(step=step,status=receipt['status'],unexpected_difference_count=receipt['unexpected_difference_count'],
                receipt_path=str(out),receipt_sha256=sha(out),paired_average_lease_live=receipt['paired_average_lease_live'],
                coherent_rows=receipt['allowed_metadata_candidate']['paired_average']['coherent_rows'])
            rows.append(summary)
            frozen=dict(status='FROZEN_SAVED_ENDPOINT_PARITY',files={str(out):sha(out)},
                helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'),checkpoint_sha256=receipt['checkpoint_sha256'])
            (OUT/f'checkpoint-{step:04d}-FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
            (OUT/'live-index.json').write_text(json.dumps(dict(status='WATCHING',records=rows,quality_verdict=None),indent=2)+'\n')
            log('saved_endpoint_compared',**summary)
        if len(rows)==len(STEPS):break
        if RESULT.exists():
            result=json.loads(RESULT.read_text())
            if result.get('status')=='ERROR':raise RuntimeError('Toy job ended ERROR before all endpoints were sealed')
        time.sleep(10)
    value=dict(status='COMPLETE_EXACT_TRAINING_PARITY' if all(r['status']=='EXACT_LEGACY_TRAINING_PARITY' for r in rows) else 'COMPLETE_WITH_TRAINING_DIVERGENCES',
        records=sorted(rows,key=lambda r:r['step']),quality_verdict=None,cpu_only=True,new_training_steps=0,new_emissions=0,
        helper_frozen_sha256=sha(HERE/'SOURCE-FROZEN.json'))
    (OUT/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    log('parity_watch_complete',status=value['status'],endpoints=len(rows))


if __name__=='__main__':
    try:main()
    except Exception as failure:
        OUT.mkdir(exist_ok=True)
        error=dict(status='HELPER_ERROR',type=type(failure).__name__,message=str(failure),utc=datetime.now(timezone.utc).isoformat())
        (OUT/'ERROR.json').write_text(json.dumps(error,indent=2)+'\n')
        log('parity_watch_error',**error)
        raise
