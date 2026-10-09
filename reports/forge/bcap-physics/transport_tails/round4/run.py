"""Drain this bounded diagnostic only; full qualification compilation is disabled."""
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.queue import Queue, drain
import particlegan

if __name__=='__main__':
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT)
    queue=Queue(Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/transport_tails/queue'),
                report_root=ROOT/'reports/forge',on_completion=None)
    print({'time':datetime.now(timezone.utc).isoformat(),'event':'bounded_drain_start','package':particlegan.__file__},flush=True)
    drain(queue,['1'],workers_per_gpu=1,allow_sharing=True,watch=False,campaign='transport_tails_round4')
    print({'time':datetime.now(timezone.utc).isoformat(),'event':'bounded_drain_complete'},flush=True)
