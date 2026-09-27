"""Single image-specific constructor proof; no learner execution."""
from pathlib import Path
import runpy,sys
from unittest.mock import patch
E=Path(__file__).resolve().parents[1];B=E/'port-source/new-init-image-screen';sys.path.insert(0,str(B))
import torch
assert not torch.cuda.is_initialized()
def forbidden(*args,**kwargs):raise AssertionError('no forward/backward/optimizer step allowed')
sys.argv=[str(B/'image_screen.py'),'--package-root',str(E/'port-source/api-dv12/package'),'--declaration',str(E/'port-source/api-dv12/candidate-declaration.json'),'--task','img_intensity2','--cpu-only','--output',str(E/'image-screen-review/dv12-intensity2-cpu')]
with patch.object(torch.nn.Module,'_call_impl',forbidden),patch.object(torch.Tensor,'backward',forbidden),patch.object(torch.optim.Adam,'step',forbidden):runpy.run_path(str(B/'image_screen.py'),run_name='__main__')
assert not torch.cuda.is_initialized()
