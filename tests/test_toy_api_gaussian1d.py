"""Scientific scorer counterexamples and public scalar-host integration."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from benchmarks.toy_audit import api_contract, api_gaussian1d
from benchmarks.toy_audit.api_vectors import _bounds
from benchmarks.toy_audit.gaussian1d_quality import sample_target, score_samples
from experiments.forge import adapters
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result
from particlegan import GANTrainer, MoGParticlePrior, get_recipe, init

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case():
    return api_gaussian1d.list_cases()[0]


def test_exact_gaussian_oracle_and_destructive_controls():
    spec = case()["spec"]
    oracle = sample_target(spec, 4096, torch.Generator().manual_seed(78013), 0)
    assert not _bounds(score_samples(oracle, spec), spec["thresholds"])
    controls = {
        "collapsed": torch.full((4096, 1), 2.),
        "shifted": oracle + .5,
        "too_wide": 2. + 2 * (oracle - 2.),
        "same_moment_atoms": torch.tensor([1.5, 2.5]).repeat(2048)[:, None],
        "same_moment_uniform": (2. + np.sqrt(3.) * .5 * torch.linspace(-1, 1, 4096))[:, None],
        "nonfinite": torch.full((4096, 1), float("nan")),
        "undersized": oracle[:128],
    }
    for name, points in controls.items():
        assert _bounds(score_samples(points, spec), spec["thresholds"]), name
    atoms = score_samples(controls["same_moment_atoms"], spec)
    assert atoms["mean_error_sigma"] == 0. and atoms["std_ratio"] == 1.
    assert atoms["cdf_ks"] > .05


def test_public_trainer_scalar_mog_and_observer_isolation():
    fixture = api_gaussian1d.build_case(api_gaussian1d.CASE_ID, max_steps=2)
    assert isinstance(fixture.trainer, GANTrainer)
    assert isinstance(fixture.trainer.prior, MoGParticlePrior)
    fixture.step()
    before = state_digest(fixture.state_dict())
    observed = api_contract.validate_observation(fixture.observe())
    assert state_digest(fixture.state_dict()) == before
    assert observed["metrics"]["sample_count"] == 4096
    view = observed["views"][0]
    assert view["kind"] == "bar" and len(view["bin_centers"]) == len(view["samples"])
    assert fixture.trainer.sample(16, output_noise=False).shape == (16, 1)
    assert view["xlim"] == [-2., 5.]


@pytest.mark.parametrize("family", ["k3p", "bispf"])
def test_scalar_initial_prior_law_is_task_bound_without_trainer_tuning(family):
    task = json.loads(api_gaussian1d.TASK_PATH.read_text())
    host, law = task["execution"]["host_definition"], task["execution"]["prior"]
    metadata = case()
    assert "recipe_overrides" not in metadata
    fixture = api_gaussian1d.build_case(api_gaussian1d.CASE_ID, recipe_name=family, max_steps=1)
    expected = get_recipe(family, num_particles=host["particles"], z_dim=host["z_dim"],
                          batch_size=host["batch"], total_steps=host["steps"],
                          prior_kind=law["kind"], sigma_rel=0., standardize=law["standardize"])
    assert fixture.recipe == expected
    prior = fixture.trainer.prior
    assert isinstance(prior, MoGParticlePrior)
    assert prior.z.shape == (host["particles"], host["z_dim"])
    assert prior.standardize is law["standardize"] is False
    assert torch.equal(prior.sigma, prior.sigma.new_tensor(law["sigma"]))
    assert prior.sigma_rel == 0.
    reference = expected.make_prior(init_std=.5, sigma=law["sigma"])
    init.deterministic_orthogonal_(reference, seed=2)
    assert torch.equal(prior.z, reference.z)
    actual, indices = prior.sample(32, generator=torch.Generator().manual_seed(19))
    declared, declared_indices = reference.sample(32, generator=torch.Generator().manual_seed(19))
    assert torch.equal(indices, declared_indices) and torch.equal(actual, declared)


def test_forge_scalar_adapter_and_incomplete_short_protocol(tmp_path):
    task = json.loads((ROOT / "configs/forge/tasks/gaussian1d_acquisition.json").read_text())
    task["execution"]["steps"] = 2  # Integration proof; full task still needs 1,000.
    candidate = {"recipe_overrides": {"name": "k3p", "critic_formulation": "k3p"}}
    assert adapters.adapter_preflight(task, candidate, root=ROOT) == []
    request = {"candidate": candidate, "candidate_revision": "unit-scalar",
               "protocol": {"seed": 0}, "tasks": {task["id"]: task}}
    raw = adapters.run_task(request, {"task_id": task["id"]}, tmp_path, "cpu")
    assert raw["execution_path"] == "public_trainer"
    assert raw["evidence"]["live"]["sample_count"] == 4096
    assert "cdf_ks" in raw["evidence"]["live"]
    assert raw["evidence"]["host"]["models"]["generator"]["parameter_shapes"]["net.4.weight"][0] == 1
    full_task = deepcopy(task)
    full_task["execution"]["steps"] = 1000
    assert grade_result(full_task, raw)["gate_status"] == "INCOMPLETE"


def test_forge_preflight_rejects_wrong_scalar_target_or_missing_scorer():
    task = json.loads((ROOT / "configs/forge/tasks/gaussian1d_acquisition.json").read_text())
    candidate = {"recipe_overrides": {"name": "k3p", "critic_formulation": "k3p"}}
    missing = deepcopy(task)
    missing["evaluation"].pop("sample_evaluator")
    assert any("scalar scorer" in message for message in adapters.adapter_preflight(missing, candidate))
    task["execution"]["host_definition"]["covariances"] = [[[0.]]]
    assert any("positive sigma" in message for message in adapters.adapter_preflight(task, candidate))


def test_declared_scorer_is_bound_before_execution_in_a_fresh_process():
    # Parent imports must not accidentally supply a scorer missing from a new receipt.
    code = """
from hashlib import sha256
from pathlib import Path
from benchmarks.toy_audit import api_gaussian1d, api_run
case = api_gaussian1d.list_cases()[0]
assert case['sample_evaluator'] == 'benchmarks.toy_audit.gaussian1d_quality:score_samples'
path = 'benchmarks/toy_audit/gaussian1d_quality.py'
assert api_run.source_identity()['files_sha256'][path] == sha256(Path(path).read_bytes()).hexdigest()
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                   capture_output=True, text=True, timeout=30)
