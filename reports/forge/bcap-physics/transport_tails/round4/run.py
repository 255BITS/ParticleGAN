"""Drain this bounded diagnostic only; full qualification compilation is disabled."""
from pathlib import Path
import sys
import json, threading
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.queue import Queue, drain
import particlegan


def mirror_events(queue, stopped):
    """Stream the queue's numerical events to unbuffered coordinator stdout."""
    with (queue.root/'events.jsonl').open() as source:
        while True:
            for line in source:
                event=json.loads(line)
                if event.get('event')=='worker_observation':print(line.rstrip(),flush=True)
            if stopped.wait(1):
                for line in source:print(line.rstrip(),flush=True)
                return

if __name__=='__main__':
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT)
    queue=Queue(Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/transport_tails/queue'),
                report_root=ROOT/'reports/forge',on_completion=None)
    print({'time':datetime.now(timezone.utc).isoformat(),'event':'bounded_drain_start','package':particlegan.__file__},flush=True)
    stopped=threading.Event();monitor=threading.Thread(target=mirror_events,args=(queue,stopped))
    monitor.start()
    try:
        drain(queue,['1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign='transport_tails_round4')
    finally:
        stopped.set();monitor.join()
    print({'time':datetime.now(timezone.utc).isoformat(),'event':'bounded_drain_complete'},flush=True)
