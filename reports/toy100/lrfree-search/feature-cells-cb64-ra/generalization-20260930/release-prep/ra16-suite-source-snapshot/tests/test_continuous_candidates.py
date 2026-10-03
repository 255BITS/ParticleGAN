"""The fixed-epsilon probe must preserve PyTorch Adam and identify prior groups."""

import pytest
import torch

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
from benchmarks.gan_v3 import gan_v3_recipe
from particlegan import GANTrainer
from particlegan.particle_prior import ParticlePrior
from reports.toy100.continuous_candidates import candidate_update


OPTIONS = {"network_eps": 1e-3, "prior_eps": 1e-8}


def _recipe():
    return gan_v3_recipe("gan", num_particles=8, z_dim=2, batch_size=4,
                      total_steps=2, lr=0.002, d_lr_mult=1,
                      prior_lr_mult=2, lr_anneal_start=0, lr_floor=1,
                      reg_arm="f_none", reg_coeff=0, prior_reg=0)


def test_native_trainer_receipt_has_every_actual_rate_and_role():
    torch.manual_seed(7)
    with candidate_update(OPTIONS) as receipt:
        trainer = GANTrainer(_recipe(), torch.nn.Linear(2, 2),
                             torch.nn.Linear(2, 1), seed=11)
        for _ in range(2):
            trainer.step(torch.ones(4, 2), generator_real=torch.ones(4, 2))
    assert [update["optimizer_role"] for update in receipt["updates"]] == ["d", "g", "d", "g"]
    assert [[group["role"] for group in update["groups"]]
            for update in receipt["updates"]] == [["d"], ["g", "prior"], ["d"], ["g", "prior"]]
    for update in receipt["updates"]:
        for group in update["groups"]:
            assert group["eps"] == (OPTIONS["prior_eps"] if group["role"] == "prior"
                                    else OPTIONS["network_eps"])
            assert group["lr"] == (0.004 if group["role"] == "prior" else 0.002)
            if update["optimizer_step"] == 1:
                assert group["gradient_rms"] >= 0
                assert group["update_rms"] >= 0
                assert group["denominator_min"] >= group["eps"]
    assert receipt["shared_gate_eligible"] is False


def test_legacy_constructor_split_and_direct_particle_role():
    torch.manual_seed(3)
    with candidate_update(OPTIONS) as receipt:
        applied = []
        with optimizer_defaults(_recipe(), applied):
            prior = ParticlePrior(8, 2)
            generator = torch.nn.Parameter(torch.ones(2))
            critic = torch.nn.Parameter(torch.ones(2))
            opt_g = torch.optim.Adam([generator, prior.z], lr=0.002)
            opt_d = torch.optim.Adam([critic], lr=0.002)
            bridge.optimizer_role(opt_d, {"opt_d": opt_d})
            bridge.optimizer_role(opt_g, {"opt_g": opt_g})
            for optimizer in (opt_d, opt_g):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()
            direct = torch.nn.Parameter(torch.ones(2))
            opt_p = torch.optim.Adam([direct], lr=0.002)
            bridge.optimizer_role(opt_p, {"opt_p": opt_p})
            direct.grad = torch.ones_like(direct)
            opt_p.step()
    assert [[g["role"] for g in u["groups"]] for u in receipt["updates"]] == [
        ["d"], ["g", "prior"], ["prior"],
    ]
    assert receipt["updates"][-1]["groups"][0]["eps"] == 1e-8


def test_adapter_matches_plain_adam_with_same_fixed_group_eps():
    a = torch.nn.Parameter(torch.tensor([0.7, -0.2], dtype=torch.float64))
    b = torch.nn.Parameter(a.detach().clone())
    plain = torch.optim.Adam([a], lr=0.03, betas=(0.0, 0.999), eps=1e-3)
    adapted = torch.optim.Adam([b], lr=0.03, betas=(0.0, 0.999), eps=1e-8)
    ordinary_step = torch.optim.Adam.step
    with candidate_update(OPTIONS) as receipt:
        # Exercise the public context through the same bridge used by legacy hosts.
        bridge.optimizer_role(adapted, {"opt_g": adapted})
        for gradient, rate in (([0.01, -0.08], 0.03), ([0.2, -0.03], 0.03),
                               ([-0.005, 0.4], 0.015)):
            a.grad = torch.tensor(gradient, dtype=a.dtype)
            b.grad = a.grad.clone()
            plain.param_groups[0]["lr"] = adapted.param_groups[0]["lr"] = rate
            ordinary_step(plain)
            adapted.step()
    assert len(receipt["updates"]) == 3
    assert torch.equal(a, b)
    assert torch.equal(plain.state[a]["exp_avg"], adapted.state[b]["exp_avg"])
    assert torch.equal(plain.state[a]["exp_avg_sq"], adapted.state[b]["exp_avg_sq"])


def test_asymmetric_eps_changes_only_declared_player():
    options = {"network_eps": 1e-8, "prior_eps": 1e-8,
               "g_eps": 0.03, "d_eps": 0.001}
    with candidate_update(options) as receipt:
        trainer = GANTrainer(_recipe(), torch.nn.Linear(2, 2),
                             torch.nn.Linear(2, 1), seed=11)
        trainer.step(torch.ones(4, 2), generator_real=torch.ones(4, 2))
    assert receipt["effective_eps"] == {"g": 0.03, "d": 0.001, "prior": 1e-8}
    assert [[g["eps"] for g in u["groups"]] for u in receipt["updates"]] == [
        [0.001], [0.03, 1e-8],
    ]


@pytest.mark.parametrize("options", [
    {}, {"network_eps": 0, "prior_eps": 1e-8},
    {"network_eps": float("nan"), "prior_eps": 1e-8},
    {"network_eps": 1e-3, "prior_eps": True},
])
def test_invalid_options_and_exception_restore_patches(options):
    original_step = torch.optim.Adam.step
    original_prior = ParticlePrior.__init__
    with pytest.raises(ValueError):
        with candidate_update(options):
            pass
    assert torch.optim.Adam.step is original_step
    assert ParticlePrior.__init__ is original_prior

    with pytest.raises(RuntimeError, match="stop"):
        with candidate_update(OPTIONS):
            raise RuntimeError("stop")
    assert torch.optim.Adam.step is original_step
    assert ParticlePrior.__init__ is original_prior
