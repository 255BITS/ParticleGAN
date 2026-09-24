"""Independent numerical checks of the frozen forward-KL free-output source."""

import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'reports/toy100/continuous-evidence/forward-kl-independent-v2'
SOURCE = ARCHIVE / 'source/forward_kl_free_filter.py'
SOURCE_SHA = '0cc098d63efb2460dc038b897d5db532b9f87115ffbd5a78e42ee042ed21c873'


def frozen():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('forward_kl_frozen_audit', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt():
    return json.loads(gzip.decompress((ARCHIVE/'first-bank-result.json.gz').read_bytes()))


def independent_ce(locations, weights, points, variance):
    # Literal equal-weight 2-D Gaussian-mixture density, independent of the
    # tested implementation's log_kernels/logsumexp helper.
    density = torch.exp(-torch.cdist(locations, points).square()/(2*variance))
    density = density.mean(1)/(2*math.pi*variance)
    return float(-(weights*density.log()).sum())


def test_quadrature_and_density_have_exact_single_gaussian_ce():
    module = frozen()
    real = torch.tensor([[.2, -.4], [1.1, .7]], dtype=torch.float64)
    point = torch.tensor([[-.1, .5]], dtype=torch.float64)
    width, sigma = .3, .2
    variance = width**2+sigma**2
    expected = math.log(2*math.pi*variance) + \
        float((real-point).square().sum(1).mean()+2*width**2)/(2*variance)
    for order in (5, 9):
        locations, weights = module.quadrature(real, width, order)
        assert abs(float(weights.sum())-1) < 1e-14
        assert abs(module.cross_entropy(locations, weights, point, variance)-expected) < 1e-13
        assert abs(independent_ce(locations, weights, point, variance)-expected) < 1e-13


def test_global_donor_matches_exhaustive_one_move_ce():
    module = frozen()
    real = torch.tensor([[0., 0.], [2.8, .1], [.2, 2.7]], dtype=torch.float64)
    points = torch.tensor([[.1, -.2], [2.6, .3]], dtype=torch.float64)
    width, variance = .23, .23**2+.12**2
    target = module.quadrature(real, width, 5)
    audit = module.quadrature(real, width, 9)
    expected = []
    for donor in range(len(points)):
        for index in range(len(real)):
            proposal = points.clone()
            proposal[donor] = real[index]
            expected.append((independent_ce(*target, proposal, variance), donor, index))
    optimum = min(expected)
    final, moves = module.global_donor(real, points, *target, variance, limit=1, audit=audit)
    assert len(moves) == 1
    assert abs(moves[0]['after']-optimum[0]) < 1e-13
    assert (moves[0]['donor_index'], moves[0]['real_index']) == optimum[1:]
    assert abs(independent_ce(*target, final, variance)-optimum[0]) < 1e-13
    assert abs(moves[0]['predicted']-moves[0]['after']) < 1e-13


def test_equal_weight_fixed_covariance_em_matches_independent_m_step():
    module = frozen()
    real = torch.tensor([[0., 0.], [1.2, .2], [.1, 1.3]], dtype=torch.float64)
    points = torch.tensor([[.15, -.1], [1.0, .3]], dtype=torch.float64)
    width, variance = .26, .26**2+.18**2
    locations, weights = module.quadrature(real, width, 5)
    density = torch.exp(-torch.cdist(locations, points).square()/(2*variance))
    responsibility = density/density.sum(1, keepdim=True)
    weighted = weights[:,None]*responsibility
    expected = (weighted.T@locations)/weighted.sum(0)[:,None]
    before = independent_ce(locations, weights, points, variance)
    after = independent_ce(locations, weights, expected, variance)
    final, rows = module.em_centroids(points, locations, weights, variance,
                                      limit=1, audit=module.quadrature(real, width, 9))
    assert len(rows) == 1 and after < before
    torch.testing.assert_close(final, expected, atol=1e-13, rtol=0)
    assert abs(rows[0]['after']-after) < 1e-13


def test_native_first_real_bank_and_rng_match_frozen_receipt():
    module = frozen()
    from reports.toy100.sample_anchor_free1200 import load_states

    row = receipt()
    cold, warm, hashes = load_states()
    assert hashes == row['declaration']['input_hashes']
    before = torch.random.get_rng_state().clone()
    for name, state in (('cold1', cold), ('warm1324', warm)):
        bank, after = module.first_native_bank(state)
        assert torch.equal(bank, torch.tensor(row['real_banks'][name]))
        assert hashlib.sha256(bank.contiguous().numpy().tobytes()).hexdigest() == \
            row['declaration']['native_real_bank_sha256'][name]
        assert after == row['declaration']['data_rng_after_bank_sha256'][name]
    assert torch.equal(before, torch.random.get_rng_state())


def test_cold_emitted_sigma_and_late_noisy_quality_are_distinct():
    row = receipt()['cases']['cold1']
    assert row['actual_sigma'] == 0.
    # The HQ utility always uses .029 late output noise. For a singleton mode,
    # its variance ratio is therefore (.029/.07)^2 although the actual cold
    # emitted component variance from that atom is exactly zero.
    from reports.toy100.sample_anchor_mmd_filter import SIGMA_OUT
    from benchmarks.locked_shared import mode_hold

    assert SIGMA_OUT == .029
    singleton = next(i for i, count in enumerate(row['final_quality']['nearest_mode_counts'])
                     if count == 1)
    ratios = row['final_quality']['per_mode_coordinate_variance_ratios'][singleton]
    expected_late = (SIGMA_OUT/mode_hold.SIGMA)**2
    assert all(abs(value-expected_late) < 1e-13 for value in ratios)
