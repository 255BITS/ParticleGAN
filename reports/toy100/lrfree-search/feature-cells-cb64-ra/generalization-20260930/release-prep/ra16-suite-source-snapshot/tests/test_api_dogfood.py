"""Compatibility contracts exercised by the repository's API consumers."""
import pytest
import torch
from torch import nn

from particlegan import ParticlePrior, UCD, get_recipe, ucd_scores
from particlegan.diffusion import DrawSource


@pytest.mark.parametrize("kind", ["learned", "fixed", "gaussian", "zero"])
def test_research_draw_source_preserves_archived_state_and_rng(kind):
    source = DrawSource(kind, 16, 3, 41, "cpu")
    assert isinstance(source, ParticlePrior)
    reference = torch.randn(16, 3, generator=torch.Generator().manual_seed(41))
    assert list(source.state_dict()) == ["table"]
    assert torch.equal(source.table, reference)
    assert bool(list(source.parameters())) == (kind == "learned")
    actual_rng = torch.Generator().manual_seed(42)
    expected_rng = torch.Generator().manual_seed(42)
    actual, indices = source.sample(7, actual_rng)
    if kind == "gaussian":
        expected = torch.randn(7, 3, generator=expected_rng)
        assert indices is None
    elif kind == "zero":
        expected = torch.zeros(7, 3)
        assert indices is None
    else:
        expected_indices = torch.randint(16, (7,), generator=expected_rng)
        assert torch.equal(indices, expected_indices)
        expected = reference[expected_indices]
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng.get_state(), expected_rng.get_state())
    source.load_state_dict({"table": reference + 1}, strict=True)
    assert torch.equal(source(torch.tensor([0, 3])), (reference + 1)[[0, 3]])


@pytest.mark.parametrize("target,heads", [("class", 3), ("time_class", 6)])
def test_existing_logits_selection_matches_wrapper_and_gradients(target, heads):
    class Network(nn.Module):
        def forward(self, x, **kwargs):
            return x

    logits = torch.randn(4, heads, requires_grad=True)
    labels, times = torch.tensor([0, 1, 2, 0]), torch.tensor([1, 2, 1, 2])
    scores = ucd_scores(logits, labels, times, num_classes=3,
                        target=target, num_steps=2)
    wrapped, _ = UCD(Network(), 3, target=target, num_steps=2)(logits, labels, t=times)
    assert torch.equal(scores, wrapped)
    scores.sum().backward()
    selected = labels if target == "class" else (times - 1) * 3 + labels
    assert torch.equal(logits.grad, torch.nn.functional.one_hot(selected, heads).float())
    with pytest.raises(ValueError, match="logits"):
        ucd_scores(logits[:, :-1], labels, times, num_classes=3,
                   target=target, num_steps=2)


def test_optimizer_factory_preserves_adam_updates_with_execution_options():
    recipe = get_recipe(z_dim=2, num_particles=8, lr=.001, betas=(0., .9))
    g, d, prior = nn.Linear(2, 2), nn.Linear(2, 1), recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(g, d, prior, foreach=False, eps=1e-6)
    clones = [nn.Parameter(p.detach().clone()) for p in (*g.parameters(), prior.z)]
    reference = torch.optim.Adam([
        {"params": clones[:-1], "lr": recipe.lr},
        {"params": clones[-1:], "lr": recipe.lr * recipe.prior_lr_mult},
    ], lr=recipe.lr, betas=recipe.betas, foreach=False, eps=1e-6)
    for _ in range(2):
        for index, (p, clone) in enumerate(zip((*g.parameters(), prior.z), clones)):
            p.grad = torch.full_like(p, .1 * (index + 1))
            clone.grad = p.grad.clone()
        opt_g.step()
        reference.step()
    for p, clone in zip((*g.parameters(), prior.z), clones):
        assert torch.equal(p, clone)
    assert opt_d.defaults["eps"] == 1e-6 and opt_d.defaults["foreach"] is False
