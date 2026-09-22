import importlib.util
from pathlib import Path

import torch
import pytest


@pytest.mark.parametrize("overrides", [{}, {"reg_kappa": 1.25, "reg_coeff": 3., "lr": .00051, "lambda_ep": .05}])
def test_public_trainer_matches_reference_updates(tmp_path, overrides):
    spec = importlib.util.spec_from_file_location("grid_parity", Path(__file__).parents[1] / "examples/100gaussians.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    torch.set_num_threads(1)
    options = dict(epochs=1, steps_per_epoch=20, batch_size=16, num_particles=32,
                   seed=0, device_str="cpu", return_details=True, save_plots=False, **overrides)
    reference = grid.train(**options, out_dir=str(tmp_path / "reference"))
    public = grid.train(**options, out_dir=str(tmp_path / "public"), use_training_api=True)
    for model in ("G", "D", "prior", "ema_G", "ema_prior"):
        for key, tensor in reference[model].state_dict().items():
            assert torch.equal(tensor, public[model].state_dict()[key]), (model, key)


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
