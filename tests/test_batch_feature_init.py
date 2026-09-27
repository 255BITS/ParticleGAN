"""Behavioral contracts for the opt-in batch-distance initializer."""
import os
import subprocess
import sys


def run(script):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_neutral_branch_is_trainable_and_preserves_host_assignments():
    run('''
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, ParticlePrior
from particlegan.init_registry import install, family_of
assert family_of('batch_feature_zero') == 'batch_feature_zero'
install('batch_feature_zero')
torch.manual_seed(0)
g = nn.Linear(2, 2)
with torch.no_grad():
    g.weight.copy_(torch.eye(2))
    g.bias.zero_()
prior = ParticlePrior(8, 2)
critic = BatchDistanceDiscriminator(hidden_dim=8, n_hidden=2)
torch.optim.Adam([*g.parameters(), *prior.parameters()])
torch.optim.Adam(critic.parameters())
assert torch.equal(g.weight, torch.eye(2)) and not g.bias.any()
assert not critic.head.weight[:, -4:].any()
assert critic.head.weight[:, :-4].abs().sum() > 0
x = torch.tensor([[-.2,.1],[.3,-.1],[.05,.4],[-.3,-.3]], requires_grad=True)
score = critic(x)
derivative = torch.autograd.grad(score[0], x, retain_graph=True)[0]
assert not derivative[1:].any()
assert derivative[0].abs().sum() > 0
real = x.detach()*2 + torch.tensor([.4,-.5])
loss = torch.nn.functional.softplus(-(critic(real)-score)).mean()
loss.backward()
assert critic.head.weight.grad[:, -4:].abs().sum() > 0
# A second optimizer must not silently erase learned batch-feature weights.
with torch.no_grad(): critic.head.weight[:, -4:].fill_(.25)
torch.optim.Adam(critic.parameters())
assert torch.equal(critic.head.weight[:, -4:], torch.full((1,4), .25))
''')


def test_reinstall_repeats_fresh_models_and_consumes_the_original_rng_stream():
    run('''
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, ParticlePrior
from particlegan import batch_feature_init as init
original = (torch.Tensor.uniform_, torch.Tensor.normal_, torch.optim.Adam.__init__)
def build(enabled):
    if enabled: init.install()
    torch.manual_seed(0)
    generator = nn.Sequential(nn.Linear(4,8), nn.LeakyReLU(.2), nn.Linear(8,2))
    prior = ParticlePrior(8,4)
    critic = BatchDistanceDiscriminator(hidden_dim=8,n_hidden=2)
    torch.optim.Adam([*generator.parameters(),*prior.parameters()])
    torch.optim.Adam(critic.parameters())
    params = [p.detach().clone() for model in (generator,prior,critic) for p in model.parameters()]
    return params, torch.get_rng_state().clone()
_, control_rng = build(False)
a, rng_a = build(True)
b, rng_b = build(True)
assert all(torch.equal(x,y) for x,y in zip(a,b,strict=True))
assert torch.equal(control_rng,rng_a) and torch.equal(rng_a,rng_b)
init.uninstall()
assert original == (torch.Tensor.uniform_,torch.Tensor.normal_,torch.optim.Adam.__init__)
_, restored_rng = build(False)
assert torch.equal(control_rng,restored_rng)
''')


def test_nonprior_normal_parameters_are_initialized_at_declared_rms():
    run('''
import torch
from torch import nn
from particlegan.batch_feature_init import install
install()
parameter = nn.Parameter(torch.empty(4,7))
with torch.no_grad(): parameter.normal_(.4,.2)
torch.optim.Adam([parameter])
assert torch.allclose(parameter.square().mean(),torch.tensor(.4**2+.2**2))
''')


def test_launcher_passes_script_arguments_without_editing_the_script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text('''
import sys
import torch
from particlegan import BatchDistanceDiscriminator
assert sys.argv[1:] == ['--example', 'value']
critic = BatchDistanceDiscriminator(hidden_dim=8,n_hidden=2)
torch.optim.Adam(critic.parameters())
assert not critic.head.weight[:, -4:].any()
print('neutral readout active')
''')
    before = script.read_bytes()
    done = subprocess.run([sys.executable, "-m", "particlegan.init_registry",
                           "--init", "batch_feature_zero", "--", str(script),
                           "--example", "value"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "neutral readout active" in done.stdout
    assert script.read_bytes() == before
