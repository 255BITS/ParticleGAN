"""Separate CUDA software fixture: zero smoothing preserves frozen develop."""
from contextlib import contextmanager
import subprocess

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from experiments.forge.state import state_digest
import particlegan.optim.dualnorm as dualnorm

BASE = "83b099d1e4330dda953d5fce4f68ce00f75fa9a6"


def test_zero_smoothing_matches_develop_public_updates_and_checkpoint_packets():
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for default compatibility")
    source = subprocess.check_output(["git", "show", BASE + ":particlegan/optim/dualnorm.py"], text=True)
    namespace = {"__name__": "particlegan.optim._develop_reference", "__package__": "particlegan.optim"}
    exec(compile(source, "develop-dualnorm.py", "exec"), namespace)

    @contextmanager
    def factory(value, optimizer_class):
        old = dualnorm.make_normalized_optimizer
        old_class = dualnorm.NormalizedOptimizer
        dualnorm.make_normalized_optimizer = value
        dualnorm.NormalizedOptimizer = optimizer_class
        try:
            yield
        finally:
            dualnorm.make_normalized_optimizer = old
            dualnorm.NormalizedOptimizer = old_class

    def build():
        recipe = get_recipe("bcap", optimizer_family="dualnorm", num_particles=8,
                            z_dim=2, batch_size=4, total_steps=8, prior_kind="mog",
                            sigma_rel=.1, standardize=False, lr_floor=1., network_lr_floor=1.)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        critic = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for index, module in enumerate((generator, critic, prior)):
            deterministic_orthogonal_(module, seed=index)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator(device="cuda:0").manual_seed(0))

    with torch.device("cuda:0"), torch.autograd.set_multithreading_enabled(False):
        torch.manual_seed(0)
        reference_factory = namespace["make_normalized_optimizer"]
        with factory(reference_factory, namespace["NormalizedOptimizer"]):
            reference = build()
        current = build()
        # Compare learned components/optimizer packets and consumed streams;
        # independent construction may leave different ambient RNG envelopes.
        def digest(trainer):
            state = trainer.state_dict()
            state.pop("cpu_rng")
            state.pop("cuda_rng")
            return state_digest(state)
        assert digest(reference) == digest(current)
        for i in range(3):
            batch = torch.tensor([[-1.], [-.2], [.4], [1.]]) + i * .1
            reference.step(batch)
            current.step(batch)
            assert digest(reference) == digest(current)
