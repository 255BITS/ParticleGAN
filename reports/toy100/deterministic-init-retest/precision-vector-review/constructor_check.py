"""Own frozen vector task constructor proof; learner execution forbidden."""
from pathlib import Path
import argparse,runpy,sys
from unittest.mock import patch
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1];B=E/'port-source/new-init-vector-screen'
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--task',required=True);a=p.parse_args();sys.path.insert(0,str(B))
import torch
assert not torch.cuda.is_initialized()
def forbidden(*args,**kwargs):raise AssertionError('no forward/backward/optimizer step allowed')
sys.argv=[str(B/'vector_screen.py'),'--package-root',str(E/'port-source'/a.candidate/'package'),'--declaration',str(E/'port-source'/a.candidate/'candidate-declaration.json'),'--task',a.task,'--cpu-only','--output',str(E/'precision-vector-review'/(a.candidate+'-'+a.task+'-cpu'))]
with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.Tensor,'backward',forbidden),patch.object(torch.optim.Adam,'step',forbidden):runpy.run_path(str(B/'vector_screen.py'),run_name='__main__')
assert not torch.cuda.is_initialized()
