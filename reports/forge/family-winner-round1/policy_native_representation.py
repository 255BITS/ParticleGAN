"""Zero-update capacity witnesses for the exact public-policy native toys.

This constructs model/prior parameters, never a learned-policy history or a
training result. Every observation uses the original full numeric gate and
the actual selected public sampler. Large states and arrays stay outside Git.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.toy_audit import api_contract, api_family_search, api_run, api_vectors
from benchmarks.toy100.problems import evaluation_geometry
from particlegan.training import GANTrainer


def state_hash(value):
    """Include tensors and documented policy-history sentinels exactly."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(str((array.dtype, tuple(array.shape))).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def construct(case, family, *, unique_support, device="cpu", geometry="line"):
    fixture = api_contract.build(case, device=device, seed=24002, recipe_name=family)
    trainer = fixture.trainer
    centers, sigma = evaluation_geometry(case["problem"], device=device)
    copies = fixture.recipe.num_particles // len(centers)
    if copies * len(centers) != fixture.recipe.num_particles:
        raise ValueError("balanced construction requires an exact equal mode roster")
    cloud = centers.repeat_interleave(copies, 0)
    if unique_support:
        # Each same-mode row is a distinct supported point. The public DV12
        # nonzero-neighbor radius now derives from actual near support rather
        # than the gap between modes. Symmetry preserves each mode centroid.
        if geometry == "line":
            offsets = (torch.arange(copies, device=device) - (copies - 1) / 2) * 5e-6
            cloud[:, 0] += offsets.repeat(len(centers))
        elif geometry == "lattice":
            if copies != 200:
                raise ValueError("the fixed 20x10 construction requires200 rows per mode")
            x, y = torch.meshgrid(torch.arange(20, device=device), torch.arange(10, device=device), indexing="ij")
            offsets = torch.stack((x.flatten() - 9.5, y.flatten() - 4.5), 1) * .001
            cloud += offsets.repeat(len(centers), 1)
        else:
            # Break coordinate ties between modes as well as between rows.
            # The bounded Atlas sampler keeps one representative per sorted
            # coordinate; common within-mode offsets alone leave grid ties.
            mode = torch.arange(len(centers), device=device).repeat_interleave(copies)
            row = torch.arange(copies, device=device).repeat(len(centers))
            cloud[:, 0] += ((mode % 10) * copies + row - 999.5) * 1e-6
            cloud[:, 1] += ((mode // 10) * copies + row - 999.5) * 1e-6
    with torch.no_grad():
        trainer.G.weight.copy_(torch.eye(2, device=device)); trainer.G.bias.zero_()
        trainer.prior.z.copy_(cloud)
    # Fresh construction derives all policy/EMA/optimizer state from the
    # installed supported model and prior. No controller history is assigned.
    fixture.trainer = GANTrainer(fixture.recipe, trainer.G, trainer.D,
        prior=trainer.prior, seed=24002, max_steps=case["default_steps"],
        serial_backward=True, optimizer_options={"foreach": False, "fused": False})
    with torch.no_grad():
        fixture.trainer.log_output_sigma.fill_(math.log(sigma))
    policy = fixture.trainer.policy
    # Atlas selects its real output shape through this public lifecycle. It
    # also observes ordinary real data/metadata; abort performs no update and
    # leaves that prelude recorded, rather than pretending it is shape-only.
    real = api_vectors._target(case, case["batch_size"], fixture.data_rng, 1, device=device)
    policy.begin_step(real, execution_limit=case["default_steps"])
    policy.abort_step()
    if fixture.completed_steps or policy.completed_steps:
        raise ValueError("capacity construction performed an ordinary update")
    return fixture


def scalar_observation(fixture, *, count, seed, return_samples=False):
    result = api_contract.validate_observation(fixture.observe(n=count, seed=seed))
    samples = fixture.trainer.sample(count, generator=torch.Generator(device=fixture.device).manual_seed(seed), output_noise=True).cpu()
    primary = api_vectors.score_case(fixture.metadata, samples, 0)
    if primary["passed"] != result["passed"] or any(primary["metrics"][key] != result["metrics"][key] for key in primary["metrics"]):
        raise ValueError("retained samples differ from the actual fixture observer")
    observation = {"samples": count, "evaluation_seed": seed, "samples_key": str(seed), "completed_steps": 0,
                   "metrics": primary["metrics"], "passed": result["passed"], "failed_bounds": result["failed_bounds"]}
    return (observation, samples.numpy()) if return_samples else observation


def run_case(case, family, output, *, device="cpu", geometry="line"):
    start = time.monotonic()
    failed_alternative = construct(case, family, unique_support=False, device=device, geometry=geometry)
    alternative = scalar_observation(failed_alternative, count=case["eval_samples"], seed=34002)
    fixture = construct(case, family, unique_support=True, device=device, geometry=geometry)
    initial_state = deepcopy(fixture.state_dict())
    initial_hash = state_hash(initial_state)
    global_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    observations, arrays = [], {}
    for seed in range(34002, 34007):
        observation, arrays[str(seed)] = scalar_observation(fixture, count=case["eval_samples"], seed=seed, return_samples=True)
        observations.append(observation)
    observation, arrays["134002"] = scalar_observation(fixture, count=100000, seed=134002, return_samples=True)
    observations.append(observation)
    pure = state_hash(fixture.state_dict()) == initial_hash
    rng_pure = torch.equal(global_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []))
    if not pure or not rng_pure:
        raise ValueError("capacity observer changed fixture, optimizer or training RNG state")
    with torch.no_grad():
        saved = fixture.trainer.prior.z.clone()
        fixture.trainer.prior.z.zero_()
    negative = scalar_observation(fixture, count=case["eval_samples"], seed=34002)
    with torch.no_grad():
        fixture.trainer.prior.z.copy_(saved)
    if negative["passed"]:
        raise ValueError("collapsed-cloud control passed the original density gate")
    directory = output / family / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    state_path, sample_path = directory / "state.pt", directory / "samples.npz"
    torch.save(initial_state, state_path)
    alternative_path = directory / "exact-duplicate-state.pt"
    torch.save(failed_alternative.state_dict(), alternative_path)
    np.savez_compressed(sample_path, **arrays)
    passed = all(row["passed"] for row in observations)
    selection = fixture.trainer.policy.served_snapshot().get("backend_selection", {})
    return {"family": family, "case_id": case["id"],
        "status": "SUPPORTED" if passed else "UNRESOLVED",
        "claim_scope": "Exact full-horizon public fixture at zero optimizer updates after one ordinary real-data lifecycle prelude; actual selected served-law density capacity only. Shared LR/prior-rate overrides leave this sampling law unchanged at this clock. No adaptive-training reachability or stable-learning claim.",
        "bindings": api_family_search.proof_bindings(case, family),
        "observations": observations,
        "construction": {"generator": "identity affine host", "prior": "balanced200 near-unique rows per mode",
                         "geometry": geometry, "device": device,
                         "offsets": {"lattice": "20x10 lattice spacing.001", "line": "x line spacing5e-6",
                                     "axis_unique": "task-grid coordinate groups receive distinct1e-6 row offsets within±.001"}[geometry],
                         "learned_output_sigma": fixture.trainer.output_sigma(),
                         "ordinary_updates": 0, "reference_fit_updates": 0,
                         "public_preludes": 1, "completed_steps": 0,
                         "backend_selection": selection},
        "failed_exact_duplicate_alternative": alternative, "collapsed_control": negative,
        "observer_purity": {"state_sha256": initial_hash, "state_unchanged": pure, "global_rng_unchanged": rng_pure},
        "elapsed_seconds": time.monotonic() - start,
        "artifacts": {"state": {"path": str(state_path.resolve()), "sha256": api_run.file_hash(state_path)},
                      "samples": {"path": str(sample_path.resolve()), "sha256": api_run.file_hash(sample_path)},
                      "exact_duplicate_state": {"path": str(alternative_path.resolve()), "sha256": api_run.file_hash(alternative_path)}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--geometry", choices=("line", "lattice", "axis_unique"), default="line")
    parser.add_argument("--family", choices=("atlas", "e22"), action="append")
    parser.add_argument("--case", choices=("api-grid100", "api-rotated100", "api-staggered100"), action="append")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cases = api_contract.discover()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reproducer.py").write_bytes(Path(__file__).read_bytes())
    records = []
    for family in args.family or ("atlas", "e22"):
        for name in args.case or ("api-grid100", "api-rotated100", "api-staggered100"):
            record = run_case(cases[name], family, args.output, device=args.device, geometry=args.geometry)
            records.append(record)
            api_run.write_json(args.output / "receipt.json", {
                "schema_version": 1, "kind": "zero-update-policy-native-capacity", "records": records})
            print(json.dumps({"family": family, "case": name, "status": record["status"],
                              "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
