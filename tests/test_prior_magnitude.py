"""CUDA mechanism and matched-host contracts for magnitude-sensitive priors."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init
from particlegan.extrapolation import stateless_directions
from particlegan.optim.dualnorm import NormalizedOptimizer
from benchmarks.toy_audit import gaussian_combined_magnitude as study
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.api import CapabilityError, task_formulation_context
from experiments.forge.state import state_digest

DEVICE = "cuda:0"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")


def test_capped_field_tracks_gradient_magnitude_and_exact_preview():
    table = nn.Parameter(torch.zeros(4, 2, device=DEVICE))
    opt = NormalizedOptimizer([dict(params=[table], role="prior", row_gradient_scale=.001)], lr=.03)
    table.grad = torch.tensor([[.0003, .0004], [.003, .004], [0., 0.], [1., 1.]], device=DEVICE)
    opt.set_sampled_rows(table, torch.tensor([0, 1, 2, 1], device=DEVICE))
    before = state_digest(opt.state_dict())
    field = stateless_directions(opt)[table]
    assert state_digest(opt.state_dict()) == before
    expected = torch.tensor([[.3, .4], [.6, .8], [0., 0.], [0., 0.]], device=DEVICE)
    assert torch.allclose(field, expected)
    opt.step()
    assert torch.equal(table, -.03*field)
    assert not opt._sampled_rows


def test_dense_gradient_cannot_move_unsampled_rows_and_missing_ownership_is_atomic():
    table = nn.Parameter(torch.zeros(4, 2, device=DEVICE))
    opt = NormalizedOptimizer([dict(params=[table], role="prior", row_gradient_scale=.001)])
    table.grad = torch.ones_like(table)
    with pytest.raises(ValueError, match="actual sampled rows"):
        opt.step()
    assert not opt.state
    assert not bool(table.any())


def test_checkpoint_scale_mismatch_rejected_before_parameter_mutation():
    table = nn.Parameter(torch.zeros(4, 2, device=DEVICE))
    opt = NormalizedOptimizer([dict(params=[table], role="prior", row_gradient_scale=.001)])
    before = state_digest(opt.state_dict())
    bad = deepcopy(opt.state_dict())
    bad["param_groups"][0]["row_gradient_scale"] = .01
    with pytest.raises(ValueError, match="row_gradient_scale differs"):
        opt.load_state_dict(bad)
    assert state_digest(opt.state_dict()) == before


def tiny(mode):
    recipe = get_recipe("bcap", optimizer_family="dualnorm", game_update=mode,
                        prior_update="row_capped", prior_gradient_scale=.001,
                        network_update="spectral_capped", network_gradient_scale=.1,
                        num_particles=16, z_dim=2, batch_size=8, total_steps=8)
    g = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).to(DEVICE)
    d = nn.Sequential(nn.Linear(1, 8), nn.LeakyReLU(.2), nn.Linear(8, 1)).to(DEVICE)
    init.deterministic_orthogonal_(g)
    init.deterministic_orthogonal_(d)
    return GANTrainer(recipe, g, d, serial_backward=True)


@reproducible_execution
def resume_probe(*, device):
    real = torch.linspace(1., 3., 8, device=device)[:, None]
    trainer = tiny("extrapolation_from_past")
    for _ in range(3):
        trainer.step(real)
    checkpoint = trainer.state_dict()
    for _ in range(3):
        trainer.step(real)
    expected = trainer.state_dict()
    restored = tiny("extrapolation_from_past")
    restored.load_state_dict(checkpoint)
    for _ in range(3):
        restored.step(real)
    assert state_digest(restored.state_dict()) == state_digest(expected)
    assert restored.opt_g.param_groups[-1]["algorithm"] == "row_capped"
    assert restored.opt_g.param_groups[0]["algorithm"] == restored.opt_d.param_groups[0]["algorithm"] == "dualnorm"
    assert restored.opt_g.param_groups[0]["network_update"] == "spectral_capped"
    assert restored.opt_d.param_groups[0]["network_update"] == "spectral_capped"
    bad = deepcopy(expected)
    bad["optimizers"][0]["param_groups"][-1]["row_gradient_scale"] = .01
    before = state_digest(restored.state_dict())
    with pytest.raises(ValueError, match="optimizer state"):
        restored.load_state_dict(bad)
    assert before == state_digest(restored.state_dict())


def test_cuda_exact_resume_with_cached_capped_prior_field():
    resume_probe(device=DEVICE)


@reproducible_execution
def matched_probe(task_id, *, device):
    protocol = study.declaration()
    states = []
    for arm in protocol["candidates"]:
        context, trainer, _ = study.build(arm, task_id, device)
        assert study.initial_proof(context, task_id, protocol)["matched"]
        assert trainer.prior.z.device == torch.device(device)
        assert trainer.recipe.prior_gradient_scale == .001
        assert trainer.recipe.lr_floor == trainer.recipe.network_lr_floor == 1.
        states.append(context.state_dict()["trainer"])
    for key in ("models", "streams", "initial_lrs", "optimizers"):
        assert len({state_digest(state[key]) for state in states}) == 1


@pytest.mark.parametrize("task_id", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_same_initial_tensors_data_law_and_scale_across_timing_arms(task_id):
    matched_probe(task_id, device=DEVICE)


def test_strict_acquisition_hold_and_shift_gates():
    protocol = study.declaration()
    rows = [dict(step=step, full_pass=True, metrics={}) for step in study.checkpoints("gaussian1d_acquisition", "stationary", protocol)]
    grade = study.summarize(rows, "gaussian1d_acquisition", "stationary", protocol)
    assert grade["combined_verdict"] == "PASS" and grade["hold_total_checks"] == 72
    rows[30]["full_pass"] = False
    assert study.summarize(rows, "gaussian1d_acquisition", "stationary", protocol)["hold_verdict"] == "FAIL"
    with pytest.raises(ValueError, match="missing"):
        study.summarize(rows[:-1], "gaussian1d_acquisition", "stationary", protocol)
    assert len(study.checkpoints("gaussian1d_acquisition", "shift", protocol)) == 48


def test_cpu_and_unsupported_component_hosts_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        study.execute(tmp_path/"absent", device="cpu")
    protocol = study.declaration()
    candidate = json.loads((study.ROOT/protocol["candidates"]["alternating"]).read_text())
    task = json.loads((study.ROOT/protocol["tasks"]["gaussian1d_acquisition"]["path"]).read_text())
    task["adapter"] = "transfer_behavior"
    with pytest.raises(CapabilityError, match="GANTrainer"):
        task_formulation_context(candidate, task, {"seed": 0}, device=DEVICE, root=study.ROOT)


def test_default_packet_and_unsupported_optimizer_validation():
    assert "prior_update" not in get_recipe("bcap").to_dict()
    assert "prior_gradient_scale" not in get_recipe("bcap").to_dict()
    with pytest.raises(ValueError, match="row-normalized"):
        get_recipe("bcap", optimizer_family="adam", prior_update="row_capped", prior_gradient_scale=.001)
    with pytest.raises(ValueError, match="finite positive"):
        get_recipe("bcap", optimizer_family="dualnorm", prior_update="row_capped", prior_gradient_scale=0)


def test_bound_runner_restores_archived_globals_and_replay_refuses_new_source():
    original = study.host.PROTOCOL, study.host.initial_proof
    with pytest.raises(RuntimeError, match="probe"):
        with study.bound_runner():
            assert study.host.PROTOCOL == study.PROTOCOL
            raise RuntimeError("probe")
    assert (study.host.PROTOCOL, study.host.initial_proof) == original
    with pytest.raises(ValueError, match="frozen study input changed"):
        study.host.declaration()


def test_prior_response_is_structural_and_scale_only_active_on_learned_capped_rows():
    from experiments.forge.techniques import recipe_field_active, validate_same_technique
    base = get_recipe("bcap", optimizer_family="dualnorm")
    capped = base.replace(prior_update="row_capped", prior_gradient_scale=.001)
    with pytest.raises(ValueError, match="prior_update"):
        validate_same_technique(base, capped)
    validate_same_technique(capped, capped.replace(prior_gradient_scale=.002))
    assert not recipe_field_active("prior_gradient_scale", base)
    assert recipe_field_active("prior_gradient_scale", capped)
    assert not recipe_field_active("prior_gradient_scale", capped, task={"execution":{"prior":{"learnable":False}}})
