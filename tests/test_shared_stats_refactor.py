"""Contracts of the behaviour-preserving BISPF split and the shared statistics helpers.

Each helper must reproduce the exact operations of the copies it replaced; the
references below are those copies, kept verbatim.
"""
import ast
import inspect
import math

import pytest
import torch
from torch import nn

import particlegan.birth_death as birth_death
import particlegan.bispf as bispf
import particlegan.mean_transport as mean_transport
import particlegan.output_moments as output_moments
from particlegan import _stats
from particlegan._bispf import transport
from particlegan.continuous import DataDriftController, SequentialSettleTest
from particlegan.drift_variants import DRIFT_VARIANTS, drift_features
from particlegan.ka2 import _median, _surprise_of
from particlegan.population_continuity import PopulationSequentialSettleTest


def _legacy_bh_count(p, q):
    m = len(p)
    ranks = torch.arange(1, m + 1, device=p.device, dtype=p.dtype)
    passed = (p <= ranks * q / m).nonzero()
    return int(passed[-1]) + 1 if len(passed) else 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bh_rejections_matches_legacy_step_up(dtype):
    g = torch.Generator().manual_seed(0)
    for trial in range(200):
        m = 1 + trial % 40
        p = torch.rand(m, generator=g, dtype=torch.float64).pow(1 + trial % 5).to(dtype).sort().values
        assert _stats.bh_rejections(p, .05) == _legacy_bh_count(p, .05)
    assert _stats.bh_rejections(torch.tensor([.5, .9], dtype=dtype), .05) == 0
    assert _stats.bh_rejections(torch.tensor([1e-6, .02, .9], dtype=dtype), .05) == 2


def test_knn_helpers_match_legacy_sbnp_repulsion_bitwise():
    z = torch.randn(37, 5, generator=torch.Generator().manual_seed(1), requires_grad=True)
    knn, r_med = _stats.knn_sq_and_median_bandwidth(z, 4)
    sq = torch.cdist(z, z, p=2.0).pow(2)
    with torch.no_grad():
        ref_med = torch.median(sq[~torch.eye(37, dtype=torch.bool)]).clamp_min(1e-12)
    ref_knn, _ = torch.topk(sq + torch.eye(37) * 1e9, k=4, dim=1, largest=False)
    assert torch.equal(knn, ref_knn) and torch.equal(r_med, ref_med)
    knn.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_group_means_has_one_implementation():
    assert output_moments.group_means is _stats.group_means is mean_transport.group_means
    values = torch.arange(12.).reshape(6, 2)
    means, counts = _stats.group_means(values, torch.tensor([0, 0, 2, 2, 2, 0]), 4)
    assert counts.tolist() == [3, 0, 3, 0]
    assert means.dtype == torch.float64 and means[1].tolist() == [0., 0.]


def test_pooled_reference_moments_shared_by_tether_and_sbnp():
    from particlegan.moment_tether import MomentTetherRegularizer, capture_initial_moments
    priors = []
    for seed, n in ((0, 7), (1, 11)):
        prior = nn.Module()
        prior.z = nn.Parameter(torch.randn(n, 3, generator=torch.Generator().manual_seed(seed)))
        capture_initial_moments(prior)
        priors.append(prior)
    tether = MomentTetherRegularizer(priors)
    sbnp = bispf.SBNPPriorRegularizer(priors)
    mean, cov = tether.reference_moments()
    s_mean, s_cov, count = sbnp._reference_moments()
    assert torch.equal(mean, s_mean) and torch.equal(cov, s_cov) and count == 18
    rows = torch.cat([p.z.detach() for p in priors])
    torch.testing.assert_close(cov, rows.T.cov(), rtol=1e-5, atol=1e-6)


def test_ka2_surprise_batched_transfer_equals_per_tensor_floats():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    opt = torch.optim.Adam(model.parameters(), betas=(0., .99))
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 3)).square().mean().backward()
        opt.step()
    model(torch.randn(8, 3)).square().mean().backward()
    expected = []
    for group in opt.param_groups:
        beta2 = group["betas"][1]
        for p in group["params"]:
            st = opt.state[p]
            vhat = st["exp_avg_sq"].mean() / (1.0 - beta2 ** st["step"])
            expected.append(float(p.grad.square().mean().sqrt() / vhat.clamp_min(1e-30).sqrt()))
    assert _surprise_of(opt) == _median(expected)


_LEGACY_TUPLES = {
    "data_memory": ("dv2", "dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "calm_anchor_hold": ("dv2",),
    "fast_reference": ("dv3", "dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "game_trust": ("dv4", "dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "payoff": ("dv5", "dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "drive_evidence": ("dv6", "dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "critic_balance": ("dv7", "dv8", "dv9", "dv10", "dv11", "dv12"),
    "pair": ("dv8", "dv9"),
    "pair_target": ("dv8",),
    "pair_prior": ("dv9",),
    "latent_bandwidth": ("dv10", "dv11", "dv12"),
    "support_trust": ("dv11",),
    "local_support": ("dv12",),
}


@pytest.mark.parametrize("variant", [None, "dv0", "dv13", *DRIFT_VARIANTS])
def test_drift_features_reproduce_legacy_variant_tuples(variant):
    features = drift_features(variant)
    assert features._fields == tuple(_LEGACY_TUPLES)
    for name, members in _LEGACY_TUPLES.items():
        assert getattr(features, name) == (variant in members), name


def test_drift_controller_features_are_not_checkpointed():
    controller = DataDriftController("dv12")
    assert controller.features is drift_features("dv12")
    assert "features" not in controller.state_dict()


def test_population_tester_inherits_sequential_settle_methods():
    assert issubclass(PopulationSequentialSettleTest, SequentialSettleTest)
    for name in ("_scale_values", "_early_stat", "_final_verdict", "_look", "_decide", "observe"):
        assert name not in vars(PopulationSequentialSettleTest), name


def _asserts(function):
    return [node for node in ast.walk(ast.parse(inspect.getsource(function))) if isinstance(node, ast.Assert)]


def test_invariants_are_explicit_exceptions_not_asserts():
    """python -O strips asserts; these invariants must still be checked."""
    assert not _asserts(output_moments.odd_witness)
    assert not _asserts(mean_transport.propose_pairs)


def test_birth_death_docstring_matches_paired_only_moves():
    doc = birth_death.__doc__
    assert "unmatched deaths clone" not in doc
    assert "pending" in doc


def test_bispf_facade_reexports_implementation_objects():
    from particlegan._bispf import critic, penalty, prior, thermal
    assert bispf._LAST_THERMAL_SCALE is bispf._LEGACY_THERMAL_CHANNEL is thermal._LEGACY_THERMAL_CHANNEL
    assert bispf.sinkhorn_coupling is transport.sinkhorn_coupling
    assert bispf.SBNPPriorRegularizer is prior.SBNPPriorRegularizer
    assert bispf._BISPFPenaltyKernel is penalty._BISPFPenaltyKernel
    assert bispf._BISPFObserverStepRecord is critic._BISPFObserverStepRecord
    for name in bispf.__all__:
        assert getattr(bispf, name) is not None
    assert bispf.FROZEN_THERMAL_SCALE == 1e-4


def test_one_transport_patch_reaches_engine_loss_and_penalty(monkeypatch):
    calls = []
    original = transport.sinkhorn_coupling

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(transport, "sinkhorn_coupling", counted)
    D = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    x_r = torch.randn(16, 2, generator=torch.Generator().manual_seed(0))
    x_f = torch.randn(16, 2, generator=torch.Generator().manual_seed(1))
    engine = bispf.BISPFEngine()
    engine.d_step(D, x_r, x_f, generator=torch.Generator().manual_seed(2))
    engine.g_step(D, x_f, x_r)
    loss = bispf.SinkhornRelativisticLoss()
    loss.d_loss(D(x_r), D(x_f))
    loss.g_loss(D(x_f), D(x_r))
    kernel = bispf._BISPFPenaltyKernel()
    kernel.penalty(D, x_r, x_f, step=0)
    assert len(calls) == 5
    assert math.isfinite(float(kernel.record.ema_residual))


def test_recipe_variant_list_mirrors_drift_variants():
    from particlegan import get_recipe
    for variant in DRIFT_VARIANTS:
        assert get_recipe("ka2", continuous_policy=variant, total_steps=None).continuous_policy == variant
    with pytest.raises(ValueError, match="unknown continuous_policy"):
        get_recipe("ka2", continuous_policy="dv13", total_steps=None)
