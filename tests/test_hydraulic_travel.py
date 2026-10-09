"""Software checks; tiny public API updates confer no scientific qualification."""
import json
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context, CapabilityError
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from experiments.forge.state import state_digest
from particlegan.hydraulic import HydraulicTravel


def test_nonlinear_accepted_output_is_bounded_and_direction_preserved():
    weight = torch.nn.Parameter(torch.tensor([[1.0]]))
    optimizer = torch.optim.SGD([weight], lr=10.)
    weight.grad = torch.tensor([[-1.]])
    limiter = HydraulicTravel(1.)
    real = torch.tensor([[0.], [.1], [.2]])
    old = weight.detach().clone() ** 3
    def probe():
        result = weight ** 3
        return result, result
    limiter.step(optimizer, real, probe)
    assert 0 < float(weight.detach() - 1) < 10
    assert float((weight.detach() ** 3 - old).abs()) <= .1
    assert limiter.summary['max_accepted_radius_ratio'] <= 1
    assert limiter.summary['limited'] == 1


def build():
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root/'configs/forge/ideas/hydraulic-output-travel-v1.json').read_text())
    task = json.loads((root/'configs/forge/tasks/gaussian1d_smoke.json').read_text())
    context = task_formulation_context(candidate, task, device='cpu', root=root)
    g, d = build_vector_models(context, resolve_vector_spec(task))
    return context, context.build_trainer(g, d, max_steps=4)


def test_public_checkpoint_replay_and_no_probe_rng_consumption():
    torch.set_num_threads(1)
    context, trainer = build()
    real = torch.linspace(1., 3., 128).reshape(-1, 1)
    trainer.step(real, generator_real=real)
    saved = context.state_dict()
    trainer.step(real, generator_real=real)
    expected = state_digest(context.state_dict())
    context.load_state_dict(saved)
    trainer.step(real, generator_real=real)
    assert state_digest(context.state_dict()) == expected
    assert trainer.hydraulic.summary['updates'] == 2
    assert trainer.hydraulic.summary['max_accepted_radius_ratio'] <= 1
    # Different replay/probe counts never consume another stream; compared with
    # the disabled mechanism's identical role draws on the same initial models.
    control_context, control = build()
    control.hydraulic = None
    control.step(real, generator_real=real)
    control.step(real, generator_real=real)
    active = context.state_dict()
    inactive = control_context.state_dict()
    def stream_tensors(tree):
        if isinstance(tree, dict):
            return {k:stream_tensors(v) for k,v in tree.items()}
        if isinstance(tree, torch.Tensor):
            return tree.tolist()
        if isinstance(tree, (list, tuple)):
            return [stream_tensors(v) for v in tree]
        return tree
    assert stream_tensors(active['streams']) == stream_tensors(inactive['streams'])
    for name in trainer._STREAMS:
        assert torch.equal(getattr(trainer,name).get_state(), getattr(control,name).get_state())
    packet = trainer.state_dict()
    packet['hydraulic']['fraction'] = .5
    before = state_digest(trainer.state_dict())
    with pytest.raises(ValueError, match='hydraulic'):
        trainer.load_state_dict(packet)
    assert state_digest(trainer.state_dict()) == before


def test_component_host_blocks_before_training():
    root=Path(__file__).resolve().parents[1]
    candidate=json.loads((root/'configs/forge/ideas/hydraulic-output-travel-v1.json').read_text())
    task=json.loads((root/'configs/forge/tasks/two_pole.json').read_text())
    with pytest.raises(CapabilityError, match='unsupported by public_components'):
        task_formulation_context(candidate, task, device='cpu', root=root)
