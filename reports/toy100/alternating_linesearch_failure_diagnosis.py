"""Classify a failed bounded Armijo search without changing its trajectory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from particlegan.gan_loss import GANLoss
from reports.toy100 import alternating_linesearch_scratch as adapter
from reports.toy100 import alternating_linesearch_probe as controller

RAW_LOSS={}


class FailureObserver(adapter.DLineSearchRecorder):
    def __init__(self,**options):
        super().__init__(**options);self.diagnostic_mode=False

    def step(self,opt,ordinary_step,closure=None):
        if not self.diagnostic_mode:return super().step(opt,ordinary_step,closure)
        if opt is self.optimizers[0]:
            error=sum(float((p.grad.detach().double()-g.double()).square().sum()) for p,g in zip(self._params(opt),self.gd0))**.5
            norm=sum(float(g.double().square().sum()) for g in self.gd0)**.5
            total=self.loss_reader('d')
            self.diagnostic_value=dict(total=total,raw_gan=RAW_LOSS['value'],cap=total-RAW_LOSS['value'],
                base_gradient_relative_error=error/max(norm,1e-300))
        return None

    def phases(self,step,opt_d,opt_g,local):
        try:yield from super().phases(step,opt_d,opt_g,local)
        except RuntimeError as error:
            if 'exhausted declared retry budget' not in str(error):raise
            self.diagnostic_mode=True
            self.row['failure_diagnosis']=dict(base_total=self.ld0,base_raw_gan=math.log(2)-self.advantage,
                base_cap=self.ld0-(math.log(2)-self.advantage),points=[])
            for alpha in (0.,2**-13,2**-16,2**-20,-2**-20):
                self._place(opt_d,[b if alpha==0 else torch.lerp(b,n,alpha) for b,n in zip(self.d0,self.d1)])
                self._place(opt_g,self.g_base);self._prepare_replay()
                displacement=sum(float((p.detach().double()-b.double()).square().sum()) for p,b in zip(self._params(opt_d),self.d0))**.5
                yield 1
                self._check_rng()
                self.row['failure_diagnosis']['points'].append(dict(alpha=alpha,parameter_displacement_norm=displacement,**self.diagnostic_value))
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True);args=parser.parse_args()
    original=GANLoss.d_loss
    def observe(gan,real,fake):
        value=original(gan,real,fake);RAW_LOSS['value']=float(value.detach());return value
    sys.argv=[sys.argv[0],'--phase','cold','--previous',str(args.previous),'--output',str(args.output)]
    with patch.object(adapter,'DLineSearchRecorder',FailureObserver),patch.object(GANLoss,'d_loss',observe),\
         patch.dict(controller.OPTIONS,{'reject_exhausted':False}):
        controller.main()
    (args.output/'failure-observer.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'failure-observer-sha256.txt').write_text(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')


if __name__=='__main__':main()
