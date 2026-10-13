"""Saved-only Ring geometry and read-only prior-freeze public binding checks.

No generative model/optimizer construction, draw, backward or update is performed.
Public Recipe checks may construct their parameter-free Coulomb validator.
The manual Jacobian is a local proxy at saved learned rows, not historical force.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import stable_hash
from experiments.forge.preflight import task_preflight
from experiments.forge.views import load_tasks
from particlegan.coulomb_prior import CoulombPriorRegularizer

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parents[1]
SOURCE = "2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12"
SOURCE_COMMIT = "67920d9090727bd74adbd331187f0e6f22533e48"
HOSTS = ("gaussian1d_smoke", "two_pole", "unused_token_hold", "ae_gan_hold",
         "ring16_acquisition", "five_word_joint_smoke", "clockfree_audit_measurement_v1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(row, means, target_variance):
    checkpoint = row["checkpoint"]
    path = Path(checkpoint["artifact_root"]) / checkpoint["path"]
    assert sha(path) == checkpoint["sha256"]
    archive = path.parent.parent
    panel_path = archive / "observed-samples.pt"
    raw = json.loads((archive / "raw-result.json").read_text())
    assert sha(panel_path) == raw["evidence"]["saved_observer_outputs"]["sha256"]
    panels = torch.load(panel_path, map_location="cpu", weights_only=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert len(panels) == 96 and checkpoint["completed_steps"] == panels[-1]["step"] == 1600
    models = state["trainer"]["models"]
    weights = models["G"]
    z = models["prior"]["z"].double()
    w = lambda i: weights[f"net.{i}.weight"].double()
    b = lambda i: weights[f"net.{i}.bias"].double()
    a1 = z @ w(0).T + b(0)
    a2 = torch.nn.functional.leaky_relu(a1, .2) @ w(2).T + b(2)
    output = torch.nn.functional.leaky_relu(a2, .2) @ w(4).T + b(4)
    jacobian = torch.einsum("oi,bi,ij,bj,jk->bok", w(4), torch.where(a2 >= 0, 1., .2),
                           w(2), torch.where(a1 >= 0, 1., .2), w(0))
    sigma = float(models["prior"]["sigma"])
    assert sigma == float(torch.tensor(.1)) and z.shape == (256, 4)
    local = sigma ** 2 * (jacobian @ jacobian.transpose(1, 2))
    x = panels[-1]["samples"].double()
    assigned, row_assigned = torch.cdist(x, means).argmin(1), torch.cdist(output, means).argmin(1)
    components = []
    for k, mean in enumerate(means):
        panel, centers = x[assigned == k], output[row_assigned == k]
        assert len(panel) > 1 and len(centers) > 1
        centered, row_centered = panel - panel.mean(0), centers - centers.mean(0)
        covariance = centered.T @ centered / len(panel)
        between = row_centered.T @ row_centered / len(centers)
        proxy = local[row_assigned == k].mean(0)
        radial = mean / mean.norm()
        tangent = torch.stack([-radial[1], radial[0]])
        values = {f"{name}_{direction}": float(axis @ matrix @ axis / target_variance)
                  for name, matrix in (("output", covariance), ("mapped_centers", between), ("local_noise", proxy))
                  for direction, axis in (("radial", radial), ("tangent", tangent))}
        components.append(dict(component=k, samples=len(panel), mapped_rows=len(centers), **values))
    summary = {name: dict(mean=sum(c[name] for c in components) / 16,
                         minimum=min(c[name] for c in components), maximum=max(c[name] for c in components))
               for name in values}
    curve = raw["evidence"]["observations"]
    return dict(attempt_id=row["attempt_id"], candidate_revision=row["candidate_revision"],
        gate_status=row["gate_status"], checkpoint_sha256=sha(path), panel_sha256=sha(panel_path),
        endpoint_metrics={k: v for k, v in row["metrics"].items() if type(v) in (int, float)}, summary=summary,
        mapped_centers_beyond_3sigma=int(((output - means[row_assigned]).norm(dim=1) > .3).sum()),
        mapped_centers_beyond_4sigma=int(((output - means[row_assigned]).norm(dim=1) > .4).sum()),
        covariance_passes=sum(c["component_covariance_error"] <= .85 for c in curve),
        best_covariance=min((dict(value=c["component_covariance_error"], step=c["step"]) for c in curve),
                            key=lambda c: c["value"]),
        core_covariance=raw["evidence"]["live"]["component_core_covariance_error"])


def main():
    torch.set_num_threads(1)
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-only research cannot construct a model/optimizer or draw")
    for name in ("rand", "randn", "randint", "randperm", "normal", "multinomial"):
        setattr(torch, name, forbidden)
    original_module_init = torch.nn.Module.__init__
    def only_recipe_validator(module, *args, **kwargs):
        if type(module) is not CoulombPriorRegularizer:
            return forbidden()
        return original_module_init(module, *args, **kwargs)
    torch.nn.Module.__init__ = only_recipe_validator
    torch.optim.Optimizer.__init__ = forbidden
    publication = json.loads((REPORT / "publication/results.json").read_text())
    registration = json.loads((REPORT / "registration.json").read_text())
    assert publication["complete"] and publication["source_digest"] == registration["source_digest"] == SOURCE
    energy_row = next(r for r in publication["certified_results"]
                      if r["role"] == "energy_u" and r["task_id"] == "ring16_acquisition")
    attempt_root = Path(energy_row["checkpoint"]["artifact_root"]).parent
    request_path = attempt_root / "request.json"
    request = json.loads(request_path.read_text())["request"]
    source = request["source"]
    assert source["digest"] == stable_hash(source["files"]) == SOURCE and len(source["files"]) == 1216
    assert source["origin_commit"] == SOURCE_COMMIT
    snapshot = Path(source["snapshot_path"])
    for relative, digest in source["files"].items():
        assert sha(ROOT / relative) == sha(snapshot / relative) == digest, relative
    tasks = load_tasks(ROOT)
    definition = tasks["ring16_acquisition"]["execution"]["host_definition"]
    means = torch.tensor(definition["means"], dtype=torch.float64)
    target_variance = definition["covariances"][0][0][0]
    rows = {r["role"]: r for r in publication["certified_results"] if r["task_id"] == "ring16_acquisition"}
    result = dict(schema_version=1, qualification_input=False, source_digest=SOURCE,
        reproduction_source_sha256=sha(__file__), publication_sha256=sha(REPORT / "publication/results.json"),
        registration_sha256=sha(REPORT / "registration.json"),
        frozen_source_binding=dict(request_sha256=sha(request_path), origin_commit=source["origin_commit"],
            source_digest=source["digest"], source_files_checked=1216, current_and_snapshot_bytes_match=True),
        scope="saved endpoints and declared deterministic target axes; no historical gradient or initialization reconstruction",
        rings={role: geometry(row, means, target_variance) for role, row in rows.items()})
    card = json.loads((ROOT / "configs/forge/ideas/r1r2-non-saturating-energy-u-v1.json").read_text())
    frozen = deepcopy(card)
    frozen["recipe_overrides"]["prior_update"] = "frozen"
    frozen["requires_capabilities"] = [c for c in card["requires_capabilities"] if c != "learned_locations"]
    checks = {}
    for name in HOSTS:
        task = tasks[name]
        blockers = task_preflight(task, frozen, {"seed": 0}, root=ROOT, tasks=tasks)
        assert blockers == []
        parent = task_formulation_context(card, task, {"seed": 0}, root=ROOT)
        child = task_formulation_context(frozen, task, {"seed": 0}, root=ROOT)
        a, b = asdict(parent.recipe), asdict(child.recipe)
        delta = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
        assert delta == {"prior_update": ["learned", "frozen"]}
        assert {k: v for k, v in parent.prior_config.items() if k != "learnable"} == {
               k: v for k, v in child.prior_config.items() if k != "learnable"}
        assert child.prior_config["learnable"] is False
        checks[name] = dict(blockers=blockers, effective_recipe_delta=delta,
                           unchanged_initial_prior=task["execution"]["prior"])
    result["prior_freeze_read_only_prerequisite"] = dict(status="PASS", contexts=checks,
        candidate_capability_removed="learned_locations", numerical_repair_claim=False,
        runtime_frozen_energy_checkpoint_integration="not run by this saved-only probe")
    paid = publication["aggregate_total_paid_seconds"]
    result["finite_budget"] = dict(existing_attempts=publication["aggregate_paid_attempts"],
        aggregate_paid_seconds=paid, cap_seconds=8000, headroom_seconds=8000-paid,
        new_recipe_full_allowance_seconds=2520, two_role_full_allowance_seconds=5040,
        new_cells=7, exact_cached_parent_cells=7, word_full_reservation_seconds=900,
        maximum_nonword_paid_before_word_reservation_seconds=8000-paid-900)
    assert random.getstate() == py and np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[0] == np_state[0] and np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(cpu, torch.get_rng_state())
    result.update(optimizer_updates_added=0, sampling_draws_added=0, ambient_rng_unchanged=True)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
