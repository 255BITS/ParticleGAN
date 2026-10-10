"""Prior ownership checks construct public priors without training or regrading."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from particlegan import MoGParticlePrior, ParticlePrior
from experiments.forge.adapters import _context, adapter_preflight
from experiments.forge.api import CapabilityError
from experiments.forge.behavior_adapters import BehaviorComponents, behavior_preflight
from experiments.forge.priors import task_prior
from experiments.forge.views import load_tasks, task_execution_fingerprint


ROOT = Path(__file__).resolve().parents[1]
MOG = dict(kind="mog", sigma=.025, standardize=False, learnable=True)
PARTICLES = dict(kind="particle_cloud", sigma=0., standardize=False, learnable=True,
                 exception_reason="Explicit finite-cloud contract fixture")


def task():
    return json.loads((ROOT / "configs/forge/tasks/vector_two_broad.json").read_text())


@pytest.mark.parametrize("field", [None, "kind", "sigma", "standardize", "learnable"])
def test_catalog_rejects_missing_prior_and_partial_declarations(tmp_path, field):
    value = task()
    if field is None:
        del value["execution"]["prior"]
    else:
        del value["execution"]["prior"][field]
    path = tmp_path / "configs/forge/tasks/vector_two_broad.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="vector_two_broad: execution.prior requires explicit"):
        load_tasks(tmp_path)


@pytest.mark.parametrize("prior", [
    [], {**MOG, "kind": "particles"}, {**MOG, "kind": []},
    {**MOG, "sigma": float("nan")}, {**MOG, "sigma": -1},
    {**MOG, "sigma": True}, {**MOG, "standardize": 0}, {**MOG, "learnable": 1},
    {**MOG, "sigma": 0}, {**PARTICLES, "sigma": .025},
    {**PARTICLES, "standardize": True}, {**PARTICLES, "exception_reason": " "},
    {**MOG, "unknown_field": 1},
])
def test_task_rejects_invalid_prior_before_execution(prior):
    value = task()
    value["execution"]["prior"] = prior
    with pytest.raises(ValueError, match="vector_two_broad: execution."):
        task_prior(value)
    assert adapter_preflight(value, {})


def test_runtime_cannot_fall_back_to_candidate_prior():
    value = task()
    del value["execution"]["prior"]
    candidate = dict(prior=deepcopy(MOG))
    assert "prior requires explicit" in adapter_preflight(value, candidate)[0]
    with pytest.raises(ValueError, match="prior requires explicit"):
        spec = value["execution"]["host_definition"]
        _context(dict(candidate=candidate, protocol=dict(seed=0)), value, "cpu",
                 dict(num_particles=spec["particles"], z_dim=spec["z_dim"], batch_size=spec["batch"]))


@pytest.mark.parametrize("prior,other,expected", [
    (MOG, PARTICLES, MoGParticlePrior), (PARTICLES, MOG, ParticlePrior),
])
def test_experiment_kind_selects_actual_code_path_despite_candidate_prior(prior, other, expected):
    value = task()
    value["execution"]["host_definition"].update(particles=12, z_dim=2, batch=4)
    value["execution"]["prior"] = deepcopy(prior)
    value["requires_capabilities"] = ["mog_prior" if expected is MoGParticlePrior else "particle_cloud"]
    candidate = dict(prior=deepcopy(other), recipe_overrides={})
    context = _context(dict(candidate=candidate, protocol=dict(seed=0)), value, "cpu",
                       dict(num_particles=12, z_dim=2, batch_size=4))
    assert type(context.build_prior()) is expected
    assert context.receipt()["prior"] == prior
    assert context.recipe.prior_kind == ("mog" if expected is MoGParticlePrior else "particles")


def test_prior_change_changes_execution_identity_without_mutating_declaration():
    value = task()
    original = deepcopy(value)
    task_prior(value)
    assert value == original
    value["execution"]["prior"] = deepcopy(PARTICLES)
    assert task_execution_fingerprint(value) != task_execution_fingerprint(original)


@pytest.mark.parametrize("prior,expected", [(MOG, MoGParticlePrior), (PARTICLES, ParticlePrior)])
def test_released_gan_v3_task_adaptation_uses_the_experiment_prior(prior, expected):
    """One released trainer card works with both public prior implementations."""
    candidate = json.loads((ROOT / "configs/forge/ideas/release07-gan-v3-task-adapted-v1.json").read_text())
    value = task()
    value["execution"]["host_definition"].update(particles=12, z_dim=2, batch=4)
    value["execution"]["prior"] = deepcopy(prior)
    value["requires_capabilities"] = ["mog_prior" if expected is MoGParticlePrior else "particle_cloud"]
    before = deepcopy(candidate)
    context = _context(dict(candidate=candidate, protocol=dict(seed=0)), value, "cpu",
                       dict(num_particles=12, z_dim=2, batch_size=4))
    assert type(context.build_prior()) is expected
    assert context.recipe.batch_size == 4 and context.recipe.num_particles == 12
    assert context.receipt()["prior"] == prior
    assert candidate == before


def test_frozen_behavior_host_cannot_claim_a_different_prior_code_path():
    value = json.loads((ROOT / "configs/forge/tasks/trajectory.json").read_text())
    value["execution"]["prior"] = deepcopy(MOG)
    value["requires_capabilities"] = ["mog_prior"]
    assert any("requires prior kind particle_cloud" in reason for reason in behavior_preflight(value, {}))


def test_behavior_checks_constructed_prior_before_binding_models_or_updates():
    value = json.loads((ROOT / "configs/forge/tasks/ae_gan_hold.json").read_text())
    components = BehaviorComponents(dict(candidate={}, protocol=dict(seed=0)), value)
    wrong = ParticlePrior(12, 2, generator=torch.Generator().manual_seed(0))
    with pytest.raises(CapabilityError, match="constructed prior differs"):
        components.bind(generator=None, critic=None, opt_g=None, opt_d=None, priors=[wrong])
    assert not components.bound and components.models == {} and components.optimizers == {}
