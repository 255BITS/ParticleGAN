import pytest
import torch

from experiments.diagnose_memory_gibbs import gibbs_panel, diagnose


class Writer:
    def __init__(self):
        self.calls = []

    def initial(self, observations):
        return torch.zeros(len(observations), 1, 2)

    def write(self, memory, x):
        self.calls.append(x.clone())
        return .5*memory+x[:, None]


class D:
    def __init__(self):
        self.writer = Writer()


class Generator:
    clock_bands = 1

    def __init__(self):
        self.calls = []

    def __call__(self, z, memory, time_index=None):
        return self.joint(z, memory, time_index=time_index)[0], None

    def infer(self, memory, point, time_index=None):
        return .5*point+.1*memory[:, 0]

    def decode(self, z, memory, latent, time_index=None):
        return .3*latent+.1*z+.2*memory[:, 0], None

    def joint(self, z, memory, time_index=None, steps=None):
        self.calls.append((z.clone(), memory.clone(), time_index.clone(), steps))
        latent = z.clone()
        for _ in range(2 if steps is None else steps):
            point, _ = self.decode(z, memory, latent, time_index=time_index)
            producer = latent
            latent = self.infer(memory, point, time_index=time_index)
        return point, producer


class K:
    def __init__(self, offset=False):
        self.calls = []
        self.offset = offset

    def score_candidate(self, candidate, anchor, time_index=None):
        self.calls.append((candidate.clone(), anchor.clone(), time_index.clone()))
        latent, point = candidate.chunk(2, -1)
        score = -(latent-.5*point-.1*anchor[:, 0]).square().sum(-1)
        if self.offset:
            score = score+3*anchor.flatten(1).sum(-1)
        return score


def panel():
    rng = torch.Generator().manual_seed(831)
    return torch.randn(8, 12, 2, generator=rng), torch.randn(8, 2, generator=rng)


def test_opposing_pairs_use_actual_producer_and_fixed_event():
    observed, z = panel()
    g, d, k = Generator(), D(), K()
    rows = gibbs_panel(g, d, z, observed, prefix=8, critic=k)
    assert len(d.writer.calls) == 8  # Real prefix only, never generated/refined observations.
    anchor = k.calls[0][1]
    fake, producer = g.joint(z, anchor, time_index=torch.full((len(z),), 8))
    torch.testing.assert_close(k.calls[0][0][:, :2], g.infer(anchor, observed[:, 8]))
    torch.testing.assert_close(k.calls[0][0][:, 2:], observed[:, 8])
    torch.testing.assert_close(k.calls[1][0][:, :2], producer)
    torch.testing.assert_close(k.calls[1][0][:, 2:], fake)
    assert not torch.equal(producer, g.infer(anchor, fake))
    for particle, _, clock, _ in g.calls:
        torch.testing.assert_close(particle, z)
        assert clock.eq(8).all()
    for _, memory, _, steps in g.calls:
        if steps is not None:
            torch.testing.assert_close(memory, anchor)
    for _, _, clock in k.calls:
        assert clock.eq(8).all()
    assert set(rows['refinement']) == {'1', '2', '3', '7'}
    assert rows['refinement']['2']['output_change_from_configured_mse'] == 0
    assert rows['joint_critic']['real_vs_shuffled_latent']['positive_rank_fraction'] == 1


def test_target_and_future_do_not_enter_generated_reads_or_refinement():
    observed, z = panel()
    g, altered_g = Generator(), Generator()
    original = gibbs_panel(g, D(), z, observed, 8, K())
    altered = observed.clone()
    altered[:, 8:] += 10
    changed = gibbs_panel(altered_g, D(), z, altered, 8, K())
    for first, second in zip(g.calls, altered_g.calls):
        for a, b in zip(first[:3], second[:3]):
            torch.testing.assert_close(a, b)
    assert original['point_mse'] != changed['point_mse']
    for steps in original['refinement']:
        for key in ('output_change_from_configured_mse', 'producer_vs_inferred_generated_mse'):
            assert original['refinement'][steps][key] == changed['refinement'][steps][key]


def test_baseline_no_critic_and_completed_run_requirement(tmp_path):
    class Baseline:
        clock_bands = 1

        def __call__(self, z, memory, time_index=None):
            return z+.2*memory[:, 0], None

    observed, z = panel()
    rows = gibbs_panel(Baseline(), D(), z, observed, 8)
    assert rows['joint_critic'] is None
    assert rows['latent'] is None
    assert rows['refinement'] is None
    assert rows['point_mse'] >= 0
    with pytest.raises(FileNotFoundError, match='summary.json'):
        diagnose(tmp_path, observed, torch.arange(len(z)))


def test_joint_comparisons_cancel_anchor_only_offsets():
    observed, z = panel()
    plain = gibbs_panel(Generator(), D(), z, observed, 8, K())['joint_critic']
    offset = gibbs_panel(Generator(), D(), z, observed, 8, K(offset=True))['joint_critic']
    for name in plain:
        for key in ('mean_margin', 'median_margin'):
            assert offset[name][key] == pytest.approx(plain[name][key], abs=1e-6)
        assert offset[name]['positive_rank_fraction'] == plain[name]['positive_rank_fraction']
