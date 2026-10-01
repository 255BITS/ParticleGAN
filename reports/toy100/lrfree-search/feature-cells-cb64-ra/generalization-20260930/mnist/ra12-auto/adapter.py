"""Original learned fixtures on the current public API; no scorer changes."""
from copy import deepcopy

from models_metrics import draw_samples as _original_draw_samples
from models_metrics import toy_metrics as _original_toy_metrics


def initialize_fixture_models(generator, critic):
    """Reproduce the frozen optimizer-time G=0/D=1 construction values."""
    from particlegan.init import deterministic_orthogonal_
    deterministic_orthogonal_(generator, seed=0)
    deterministic_orthogonal_(critic, seed=1)


class OriginalPrimarySampler:
    """Supply the frozen scorer with its original noisy primary sample law."""
    def __init__(self, trainer):
        self.trainer = trainer

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def sample(self, n, *, generator=None):
        return self.trainer.sample(n, generator=generator, output_noise=True)


def draw_samples(trainer, n, seed):
    return _original_draw_samples(OriginalPrimarySampler(trainer), n, seed)


def toy_metrics(trainer, seed):
    return _original_toy_metrics(OriginalPrimarySampler(trainer), seed)


def selection_diagnostics(trainer):
    selection = trainer.policy._feature_selection
    if selection is None:
        raise RuntimeError('candidate must expose its opt-in auto backend selection')
    return deepcopy(selection.state_dict())
