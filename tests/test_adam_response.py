"""The proposed geometry must be a positive adversarial-gradient preconditioner."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reports/toy100'), str(ROOT/'reports/toy100/h_stability')]
from adam_response import BlockObserver, SplitObserver, PROPOSALS, response_policy
from benchmarks import learned_lr_evaluation as bridge


def test_block_denominator_matches_hand_update_and_preserves_adam_moments():
    p = torch.nn.Parameter(torch.tensor([[.4, -.3], [.6, -.2]], dtype=torch.float64))
    reference = torch.nn.Parameter(p.detach().clone())
    opt = torch.optim.Adam([p], lr=.012, betas=(0., .9))
    plain = torch.optim.Adam([reference], lr=.012, betas=(0., .9))
    observer = BlockObserver(dict(network_eps=1e-8, prior_eps=1e-8), ('g',))
    observer.register(opt, 'g')
    for gradient in (torch.tensor([[.1, -.02], [.4, -.5]], dtype=p.dtype),
                     torch.tensor([[-.3, .2], [.01, -.6]], dtype=p.dtype)):
        p.grad = gradient.clone(); reference.grad = gradient.clone()
        anchor = p.detach().clone()
        plain.step()
        observer.step(opt, torch.optim.Adam.step)
        state = opt.state[p]
        denominator = (state['exp_avg_sq'].mean()/(1-.9**int(state['step']))).sqrt()+1e-8
        expected = anchor-.012*gradient/denominator
        torch.testing.assert_close(p, expected, rtol=1e-14, atol=1e-14)
        assert torch.all((p-anchor)*gradient < 0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            assert torch.equal(state[key], plain.state[reference][key])


@pytest.mark.parametrize('name', list(PROPOSALS))
def test_scoped_policy_restores_adam_and_records_fixed_roles(name):
    ordinary = torch.optim.Adam.step
    with response_policy(dict(adam_response=name)) as receipt:
        g = torch.nn.Parameter(torch.tensor([.1, .2]))
        d = torch.nn.Parameter(torch.tensor([.3, .4]))
        opt_g = torch.optim.Adam([g], lr=.001125, betas=(0., .999))
        opt_d = torch.optim.Adam([d], lr=.0015, betas=(0., .999))
        bridge.optimizer_role(opt_g, {'opt_g': opt_g})
        bridge.optimizer_role(opt_d, {'opt_d': opt_d})
        for opt, p in ((opt_d, d), (opt_g, g)):
            p.grad = torch.tensor([.01, -.02]); opt.step()
        assert receipt['additional_gradient_evaluations'] == 0
        assert [u['optimizer_role'] for u in receipt['updates']] == ['d', 'g']
        assert [u['groups'][0]['lr'] for u in receipt['updates']] == [.0015, .001125]
    assert torch.optim.Adam.step is ordinary


def test_geometry_resume_is_exact_without_extra_history():
    p = torch.nn.Parameter(torch.tensor([.4, -.2], dtype=torch.float64))
    opt = torch.optim.Adam([p], lr=.01, betas=(0., .9))
    observer = BlockObserver(dict(network_eps=1e-8, prior_eps=1e-8), ('g',))
    observer.register(opt, 'g')
    p.grad = torch.tensor([.02, -.3], dtype=p.dtype)
    observer.step(opt, torch.optim.Adam.step)
    q = torch.nn.Parameter(p.detach().clone())
    replay = torch.optim.Adam([q])
    replay.load_state_dict(deepcopy(opt.state_dict()))
    other = BlockObserver(dict(network_eps=1e-8, prior_eps=1e-8), ('g',))
    other.register(replay, 'g')
    p.grad = torch.tensor([-.2, .08], dtype=p.dtype); q.grad = p.grad.clone()
    observer.step(opt, torch.optim.Adam.step); other.step(replay, torch.optim.Adam.step)
    assert torch.equal(p, q)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        assert torch.equal(opt.state[p][key], replay.state[q][key])


def test_split_particle_geometry_is_positive_and_replays_extra_moments_exactly():
    eps = dict(network_eps=1e-8, prior_eps=1e-8)
    p = torch.nn.Parameter(torch.tensor([[.2, -.1], [-.3, .5]], dtype=torch.float64))
    opt = torch.optim.Adam([{'params': [p], '_comparison_prior': True}], lr=.01, betas=(0., .9))
    observer = SplitObserver(eps); observer.register(opt, 'g')
    gradient = torch.tensor([[.03, -.02], [-.01, .04]], dtype=p.dtype)
    anchor = p.detach().clone(); p.grad = gradient.clone()
    observer.step(opt, torch.optim.Adam.step)
    center = gradient.mean(0, keepdim=True); relative = gradient-center
    expected = anchor-.01*(center/(center.abs()+1e-8) +
                          relative/(relative.square().mean(0, keepdim=True).sqrt()+1e-8))
    torch.testing.assert_close(p, expected, rtol=1e-14, atol=1e-14)
    assert ((p-anchor)*gradient).sum() < 0
    # Both the centroid and relative spaces have strictly positive mobility.
    for key in ('response_centroid_sq', 'response_relative_sq'):
        assert torch.all(opt.state[p][key] > 0)
    q = torch.nn.Parameter(p.detach().clone())
    replay = torch.optim.Adam([{'params': [q], '_comparison_prior': True}])
    replay.load_state_dict(deepcopy(opt.state_dict()))
    other = SplitObserver(eps); other.register(replay, 'g')
    p.grad = -gradient; q.grad = p.grad.clone()
    observer.step(opt, torch.optim.Adam.step); other.step(replay, torch.optim.Adam.step)
    assert torch.equal(p, q)
    for key in opt.state[p]:
        assert torch.equal(opt.state[p][key], replay.state[q][key])
