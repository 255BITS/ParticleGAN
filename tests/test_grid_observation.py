import importlib.util
from pathlib import Path

import torch


def test_grid_observer_preserves_training_and_forwards_cap(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('grid_example',Path(__file__).parents[1]/'examples/100gaussians.py')
    grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
    torch.set_num_threads(1)
    opts=dict(epochs=1,steps_per_epoch=6,batch_size=8,num_particles=12,seed=0,
              device_str='cpu',return_details=True,save_plots=False,reg_kappa=1.25)
    caps=[]
    original=grid.get_recipe
    def recipe(**kwargs):
        caps.append(kwargs['reg_kappa'])
        return original(**kwargs)
    monkeypatch.setattr(grid,'get_recipe',recipe)
    baseline=grid.train(**opts,out_dir=str(tmp_path/'plain'))
    points=[]
    def observe(step,g,prior,eg,ep,seconds):
        torch.randn(100)
        g.eval()
        points.append((step,seconds))
    measured=grid.train(**opts,out_dir=str(tmp_path/'measured'),metric_callback=observe,metric_interval=2)
    for model in ('G','D','prior','ema_G','ema_prior'):
        for key,value in baseline[model].state_dict().items():
            assert torch.equal(value,measured[model].state_dict()[key]),(model,key)
    assert caps==[1.25,1.25]
    assert [step for step,_ in points]==[2,4,6]
    assert all(b[1]>=a[1] for a,b in zip(points,points[1:]))
