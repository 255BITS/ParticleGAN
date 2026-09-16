import torch

from experiments.diagnose_memory_future_rank import ranking
from experiments.diagnose_memory_local_signal import ranking as next_ranking


class Writer:
    def initial(self, observations):
        return torch.zeros(len(observations), 1, 2)
    def write(self, memory, x):
        return .5*memory+x[:, None]


class Generator:
    clock_bands = 1
    def __init__(self):
        self.calls = []
    def __call__(self, z, memory, time_index):
        self.calls.append((z.clone(), memory.clone(), time_index.clone()))
        return .2*memory[:, 0]+z, None


class Critic:
    writer = Writer()
    def __init__(self):
        self.calls = []
    def score_candidate(self, candidate, memory, times, *, horizon=0):
        self.calls.append((candidate.detach().clone(), memory.detach().clone(), times.clone(), horizon))
        return -(candidate-memory[:, 0]/2-horizon*.01).square().sum(-1)


def panel():
    rng = torch.Generator().manual_seed(914)
    return torch.randn(8, 28, 2, generator=rng), torch.randn(8, 2, generator=rng)


def test_horizon_zero_matches_existing_clean_and_full_write_ranking():
    observed, z = panel()
    actual = ranking(Critic(), Generator(), z, observed, 8)
    for name, strength in (('clean', None), ('generated_write', 1.)):
        expected = next_ranking(Critic(), Generator(), z, observed, 8, strength)
        assert actual['donor_endpoint_distance'] == expected['donor_endpoint_distance']
        for kind in ('nearest', 'shuffle'):
            new = actual['contexts'][name]['0']['donors'][kind]
            for metric in ('correct_rank_fraction', 'mean_margin', 'median_margin'):
                assert new[metric] == expected[kind][metric]


def test_future_queries_index_targets_and_share_causal_memories():
    observed, z = panel()
    d, g = Critic(), Generator()
    actual = ranking(d, g, z, observed, 8)
    assert len(g.calls) == 1
    torch.testing.assert_close(g.calls[0][0], z)
    assert torch.equal(g.calls[0][2], torch.full((8,), 7))
    for offset, horizon in enumerate((0, 4, 12)):
        candidate, memory, times, queried = d.calls[3*offset]
        torch.testing.assert_close(candidate, observed[:, 8+horizon])
        torch.testing.assert_close(memory, d.calls[0][1])
        assert queried == horizon and times.eq(8).all()
        assert actual['contexts']['clean'][str(horizon)]['target_index'] == 8+horizon
    modified = observed.clone()
    modified[:, 8:] += 100
    altered_d, altered_g = Critic(), Generator()
    altered = ranking(altered_d, altered_g, z, modified, 8)
    assert actual['donor_endpoint_distance'] == altered['donor_endpoint_distance']
    for original, altered_call in zip(d.calls, altered_d.calls):
        torch.testing.assert_close(original[1], altered_call[1])
    torch.testing.assert_close(g.calls[0][1], altered_g.calls[0][1])
