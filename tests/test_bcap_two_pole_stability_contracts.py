"""Software contract checks for independently enabled public transport terms."""
import pytest
import torch

from particlegan.conditional_transport import OutputMarginalTransport
from particlegan import Recipe


@pytest.mark.parametrize('active', ['global', 'local'])
def test_only_enabled_transport_term_is_called_and_resumes(monkeypatch, active):
    recipe=Recipe(kinetic_transport_weight=float(active=='global'),
                  kinetic_transport_local_weight=float(active=='local'))
    disabled='kinetic_transport_local_loss' if active=='global' else 'kinetic_transport_loss'
    def forbidden(*_args):
        raise AssertionError('disabled transport signal was called')
    monkeypatch.setattr(Recipe,disabled,forbidden)
    consumer=OutputMarginalTransport(recipe)
    fake=torch.tensor([[-.4],[.5]],requires_grad=True)
    real=torch.tensor([[-1.],[1.]])
    value=consumer.add(fake.new_zeros(()),fake,real)
    value.backward()
    assert torch.isfinite(value) and value>0
    assert torch.isfinite(fake.grad).all() and fake.grad.norm()>0
    restored=OutputMarginalTransport(recipe)
    restored.load_state_dict(consumer.state_dict())
    assert restored.state_dict()==consumer.state_dict()
    assert restored.active_calls==restored.calls==1


def test_disabled_transport_has_no_panel_or_hook_requirement():
    recipe=Recipe()
    value=torch.tensor(3.)
    consumer=OutputMarginalTransport(recipe)
    assert consumer.add(value,None,None) is value
    assert consumer.calls==consumer.active_calls==0
