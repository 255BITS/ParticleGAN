"""Run the frozen residual-curvature cards through the audited shared-D runner."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch
from . import shared_discriminator_search as runner
from .shared_residual_curvature_research import ARCHITECTURES,constructor,variant

@contextmanager
def registry():
    with patch.object(runner,'ARCHITECTURES',ARCHITECTURES),patch.object(runner,'constructor',constructor),patch.object(runner,'variant',variant):
        yield

def episode(job,card):
    with registry():return runner.episode(job,card)

def run(declaration,output):
    with registry():return runner.run(declaration,output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(p)
    a=p.parse_args()
    apply_device_policy(a.device, log=True)
    run(json.loads(a.plan.read_text()),a.output)
