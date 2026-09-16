import torch

from experiments.diagnose_memory_information import collect, fit_probe, make_splits, metrics


def test_diagnostic_splits_are_disjoint_and_in_support():
    splits = make_splits([64, 32, 32], 40)
    labels = torch.cat([row[1] for row in splits.values()])
    assert len(torch.unique(labels, dim=0)) == 128
    assert labels[:, 0].min() >= .6 and labels[:, 0].max() <= 1.4
    assert labels[:, 1].abs().min() >= .12 and labels[:, 1].abs().max() <= .4
    for observed, _, ids in splits.values():
        assert observed.shape[1:] == (40, 2)
        assert ids.min() >= 0 and ids.max() < 512


def test_ridge_oracle_calibrates_and_test_cannot_change_fit():
    targets = {k: row[1] for k, row in make_splits([128, 64, 64], 32).items()}
    pred, selection = fit_probe(targets, targets, 'ridge', 'cpu')
    result = metrics(pred(targets['test']), targets['test'])
    assert result['radius_r2'] > .99999
    assert result['signed_speed_r2'] > .99999
    changed = {**targets, 'test': targets['test']*100}
    pred2, selection2 = fit_probe(changed, changed, 'ridge', 'cpu')
    assert selection == selection2
    torch.testing.assert_close(pred(targets['test']), pred2(targets['test']))


def test_collection_generated_branch_ignores_future_observations_and_keeps_z():
    class Writer:
        def initial(self, x):
            return torch.zeros(x.shape[0], 1, 2)
        def write(self, memory, x):
            return memory*.7+x[:, None]
    class G(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(1.))
            self.calls = []
        def forward(self, z, memory, time_index):
            self.calls.append((z.clone(), time_index))
            return memory[:, 0]*.5+z, None
    class D:
        writer = Writer()
    prior = lambda ids: torch.stack((ids.float(), ids.float()), -1)*.01
    observed, _, ids = make_splits([4, 2, 2], 35)['train']
    g = G()
    original, _ = collect(g, D(), prior, observed, ids, [0, 1, 3])
    changed = observed.clone()
    changed[:, 32:] += 100
    altered, _ = collect(g, D(), prior, changed, ids, [0, 1, 3])
    for depth in (0, 1, 3):
        torch.testing.assert_close(original['generated'][depth], altered['generated'][depth])
    assert not torch.allclose(original['real'][3], altered['real'][3])
    assert [time for _, time in g.calls] == [32, 33, 34]*2
    for z, _ in g.calls:
        torch.testing.assert_close(z, prior(ids))
