import pytest
import torch

from experiments.diagnose_memory_transition import transition_panel, diagnose


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


class TransitionCritic:
    include_x = True

    def __init__(self, conditioned=True):
        self.calls = []
        self.conditioned = conditioned

    def candidate(self, x, memory):
        return torch.cat((x, memory.flatten(1)), -1)

    def score_candidate(self, candidate, anchor):
        self.calls.append((candidate.clone(), anchor.clone()))
        if self.conditioned:
            return -(candidate[:, 2:]-candidate[:, :2]-.5*anchor[:, 0]).square().sum(-1)
        return -candidate.square().sum(-1)


def panel():
    rng = torch.Generator().manual_seed(9231)
    return torch.randn(8, 12, 2, generator=rng), torch.randn(8, 2, generator=rng)


def test_one_write_reads_use_same_particle_and_clock_without_future_leak():
    observed, z = panel()
    g, k = Generator(), TransitionCritic()
    result = transition_panel(g, Critic(), z, observed, prefix=8, critic=k)
    assert len(g.calls) == 4
    for i, (particle, _, clock) in enumerate(g.calls):
        torch.testing.assert_close(particle, z)
        assert clock.eq(8 if i == 0 else 9).all()
    initial = g.calls[0][1]
    torch.testing.assert_close(g.calls[1][1], Critic.writer.write(initial, observed[:, 8]))
    fake = .2*initial[:, 0]+z
    torch.testing.assert_close(g.calls[2][1], Critic.writer.write(initial, fake))
    torch.testing.assert_close(g.calls[3][1], g.calls[1][1].roll(1, 0))
    for _, anchor in k.calls[4:]:
        torch.testing.assert_close(anchor, initial)
    altered = observed.clone()
    altered[:, 9:] += 100
    g2 = Generator()
    other = transition_panel(g2, Critic(), z, altered, prefix=8, critic=TransitionCritic())
    for first, second in zip(g.calls, g2.calls):
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
    assert result['transition_critic'] == other['transition_critic']
    assert result['next_read_mse'] != other['next_read_mse']


def test_consistency_critic_detects_hybrids_and_anchor_shuffle():
    observed, z = panel()
    rows = transition_panel(Generator(), Critic(), z, observed, 8, TransitionCritic())['transition_critic']
    assert abs(rows['real_vs_generated']['mean_margin']) < 1e-10
    for key in ('real_anchor_vs_shuffled_anchor', 'real_vs_real_x_generated_memory',
                'real_vs_generated_x_real_memory', 'real_vs_other_episode_successor'):
        assert rows[key]['positive_rank_fraction'] == 1.
        assert rows[key]['mean_margin'] > 0
    unconditional = transition_panel(Generator(), Critic(), z, observed, 8,
                                    TransitionCritic(False))['transition_critic']
    assert unconditional['real_anchor_vs_shuffled_anchor']['mean_margin'] == 0
    assert unconditional['generated_anchor_vs_shuffled_anchor']['mean_margin'] == 0


def test_baseline_supported_and_partial_run_rejected(tmp_path):
    observed, z = panel()
    result = transition_panel(Generator(), Critic(), z, observed, 8)
    assert result['transition_critic'] is None
    assert result['normalized_successor_memory_mse'] >= 0
    with pytest.raises(FileNotFoundError, match='summary.json'):
        diagnose(tmp_path, observed, torch.arange(len(z)))


def test_paired_anchor_comparison_cancels_anchor_only_score_offsets():
    class NonzeroMarginCritic(TransitionCritic):
        def score_candidate(self, candidate, anchor):
            return super().score_candidate(candidate, anchor)+candidate[:, 0]*anchor[:, 0, 0]

    class OffsetCritic(NonzeroMarginCritic):
        def score_candidate(self, candidate, anchor):
            return super().score_candidate(candidate, anchor)+7*anchor.flatten(1).sum(-1)

    observed, z = panel()
    base = transition_panel(Generator(), Critic(), z, observed, 8, NonzeroMarginCritic())['transition_critic']
    offset = transition_panel(Generator(), Critic(), z, observed, 8, OffsetCritic())['transition_critic']
    for key in ('real_vs_generated', 'real_vs_other_episode_successor', 'real_vs_generated_shuffled_anchor',
                'paired_margin_correct_vs_shuffled_anchor'):
        for metric in ('mean_margin', 'median_margin'):
            assert offset[key][metric] == pytest.approx(base[key][metric], abs=3e-6)
        assert offset[key]['positive_rank_fraction'] == base[key]['positive_rank_fraction']
    assert offset['real_anchor_vs_shuffled_anchor']['median_margin'] != pytest.approx(
        base['real_anchor_vs_shuffled_anchor']['median_margin'])


def test_read_space_candidates_and_hybrids_reuse_next_reads():
    class ReadCritic(TransitionCritic):
        space = 'read'

        def candidate(self, x, read):
            assert read.ndim == 2 and read.shape == x.shape
            return torch.cat((x, read), -1)

    observed, z = panel()
    g, k = Generator(), ReadCritic()
    result = transition_panel(g, Critic(), z, observed, 8, k)
    assert result['transition_candidate_space'] == 'read'
    assert len(g.calls) == 4  # Initial proposal plus three already-required diagnostic reads.
    real_read = .2*g.calls[1][1][:, 0]+z
    generated_read = .2*g.calls[2][1][:, 0]+z
    torch.testing.assert_close(k.calls[0][0][:, 2:], real_read)
    torch.testing.assert_close(k.calls[1][0][:, 2:], generated_read)
    torch.testing.assert_close(k.calls[4][0], k.calls[0][0].roll(1, 0))
    torch.testing.assert_close(k.calls[5][0][:, :2], observed[:, 8])
    torch.testing.assert_close(k.calls[5][0][:, 2:], generated_read)
    torch.testing.assert_close(k.calls[6][0][:, :2], k.calls[1][0][:, :2])
    torch.testing.assert_close(k.calls[6][0][:, 2:], real_read)
    assert 'real_vs_real_x_generated_read' in result['transition_critic']
    assert 'real_vs_real_x_generated_memory' not in result['transition_critic']
