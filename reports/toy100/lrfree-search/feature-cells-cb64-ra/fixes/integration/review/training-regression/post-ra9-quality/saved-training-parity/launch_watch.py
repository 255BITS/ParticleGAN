"""Launch one CPU-only detached saved-endpoint watcher after helper/source freeze."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
PYTHON='/tmp/pr38-default-env/bin/python'


def main():
    assert (HERE/'SOURCE-FROZEN.json').is_file() and not (HERE/'WATCH-PID.json').exists()
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (HERE/'watch-attempt1.log').open('x') as log:
        process=subprocess.Popen([PYTHON,'-B',str(HERE/'watch_saved.py')],cwd=HERE,env=env,
                                 stdout=log,stderr=log,start_new_session=True,close_fds=True)
    fields=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()
    value=dict(status='CPU_WATCHER_STARTED',pid=process.pid,start_ticks=fields[19],CPU_only=True,
               source_frozen_sha256=hashlib.sha256((HERE/'SOURCE-FROZEN.json').read_bytes()).hexdigest(),
               launched_utc=datetime.now(timezone.utc).isoformat(),numerical_replay=False,new_emissions=0,new_training_steps=0)
    with (HERE/'WATCH-PID.json').open('x') as target:target.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
