"""Bounded public Queue/drain; enqueue both frozen arms before starting."""
from pathlib import Path
from experiments.forge.queue import Queue, drain

ROOT = Path(__file__).resolve().parents[5]
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/queue')
if __name__ == '__main__':
    import particlegan
    assert Path(particlegan.__file__).resolve().is_relative_to(ROOT)
    print(dict(phase='bounded_drain',package=particlegan.__file__,queue=str(QUEUE)),flush=True)
    queue=Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None)
    print(drain(queue,['0'],workers_per_gpu=1,allow_sharing=True,watch=False),flush=True)
