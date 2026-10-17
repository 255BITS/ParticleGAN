"""Zero-update image capacity under the actual Atlas/E22 public served law.

Retained supervised reference parameters are construction inputs. They carry no
ordinary training qualification into this separate policy/architecture cohort.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

# Direct report-script invocation uses the selected checkout's public modules.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from particlegan import GANTrainer
from benchmarks.toy_audit import api_contract, api_images
from benchmarks.toy_audit.api_family_search import proof_bindings
from benchmarks.toy_audit.api_run import json_value
from benchmarks.transfer_suite.image_tasks import templates
from experiments.forge.contracts import stable_hash


CASES = (
    ("image-develop-img_intensity2-source-transpose12", "img_intensity2",
     "462a04ede97889cec691426834d1f583acef8ab9c4904416166776db46d8a61c"),
    ("image-develop-img_bars4-source-transpose12", "img_bars4",
     "f9c3d898fc42875f74387ff0cfb5a8b407004bd63af5b5d41827b4fd2cd7c009"),
)
FAMILIES = ("atlas", "e22")
SEED = 24002
EVAL_SEED = SEED + 10000
NEIGHBOR_SPACING = 2. ** -20


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def install_reference(fixture, reference, *, local_neighbors=False):
    """Initialize the same public trainer from explicit endpoint parameters.

    This fresh constructor initializes fast/EMA parameters and DV12 bandwidth
    together from the restored prior. No learned controller, tester evidence,
    training clock or optimizer trajectory is transplanted or fabricated.
    """
    if fixture.case.get("query") or fixture.case["architecture"] != "transpose":
        raise ValueError("only the declared unconditional source-transpose hosts are supported")
    source_spec = reference["task"]["execution"]["host_definition"]
    for field in ("architecture", "width", "z_dim", "pattern"):
        if source_spec[field] != fixture.case[field]:
            raise ValueError(f"reference and public host differ in {field}")
    fixture.G.load_state_dict(reference["generator"], strict=True)
    fixture.prior.load_state_dict(reference["prior"], strict=True)
    if local_neighbors:
        if torch.count_nonzero(fixture.prior.z[:, 7]).item() != 0:
            raise ValueError("declared constructive coordinate7 is not unused in reference rows")
        # Rows stay balanced at each original mode. These explicit learned
        # positions supply near same-mode support to the unmodified DV12 law.
        with torch.no_grad():
            offsets = torch.arange(len(fixture.prior.z), device=fixture.device, dtype=fixture.prior.z.dtype)
            fixture.prior.z[:, 7] += (offsets - (len(offsets) - 1) / 2) * NEIGHBOR_SPACING
    fixture.trainer = GANTrainer(
        fixture.recipe, fixture.G, fixture.D, prior=fixture.prior,
        seed=fixture.seed, max_steps=fixture.max_steps,
        model_generator=torch.Generator(device=fixture.device).manual_seed(fixture.seed + 8))
    fixture.policy = fixture.trainer.policy
    if (fixture.completed_steps != 0 or fixture.trainer.opt_g.state
            or fixture.trainer.opt_d.state):
        raise ValueError("capacity construction must leave the public optimizer clock at zero")
    return fixture


def evaluate(archive, output, *, local_neighbors=False, lifecycle_prelude=False):
    """Construct and retain four parameter witnesses, with original API bounds."""
    archive, output = Path(archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    declaration = dict(schema_version=1, kind="public_policy_capacity_construction",
                       families=list(FAMILIES), cases=[case for case, _, _ in CASES],
                       device="cpu", cpu_threads=1, seed=SEED, evaluation_seed=EVAL_SEED,
                       evaluation_samples="Original case.eval_samples (1024)",
                       new_fitting_updates=0, ordinary_training_updates=0,
                       ordinary_qualification_credit=False,
                       scope="Gate-tolerance capacity of exact public selected/served law; no learning or robustness claim",
                       construction="Restore reference G/prior into actual host and initialize fresh public GANTrainer; retain DV12 and native selected-serving policy",
                       same_mode_neighbors=local_neighbors,
                       neighbor_spacing=NEIGHBOR_SPACING if local_neighbors else None,
                       neighbor_coordinate=7 if local_neighbors else None,
                       lifecycle_prelude="Public begin_step(actual first real batch, full case horizon) then abort_step; retain controller/reservoir metadata observations; no forward/backward/optimizer update" if lifecycle_prelude else "none; pre-shape policy sampling only",
                       reproducer_sha256=file_hash(__file__))
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    summary_path = archive / "result.json"
    summary = json.loads(summary_path.read_text())
    if summary["declaration"].get("ordinary_training_updates") != 0:
        raise ValueError("reference fitting must be separated from ordinary GAN qualification")
    definitions = api_contract.discover()
    records = []
    for family in FAMILIES:
        for case_id, task_id, expected_hash in CASES:
            start = time.monotonic()
            case = definitions[case_id]
            bindings = proof_bindings(case, family)
            reference_path = archive / f"{task_id}-witness.pt"
            if file_hash(reference_path) != expected_hash:
                raise ValueError("retained reference state differs from the frozen construction input")
            result = next(row for row in summary["results"] if row["task"] == task_id)
            if result["artifact"]["sha256"] != expected_hash or result["ordinary_training_updates"] != 0:
                raise ValueError("reference result does not bind the zero-ordinary-update parameter artifact")
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            if (stable_hash(reference["task"]) != result["task_sha256"]
                    or api_images.ordered_bank_sha256(templates(reference["task"]["execution"]["host_definition"]))
                    != case["ordered_template_sha256"]):
                raise ValueError("reference task/ordered targets differ from the exact public question")
            global_rng = torch.random.get_rng_state().clone()
            fixture = api_contract.build(case, device="cpu", seed=SEED,
                                         recipe_name=family, max_steps=case["default_steps"])
            install_reference(fixture, reference, local_neighbors=local_neighbors)
            real = None
            if lifecycle_prelude:
                real, context = fixture._real_batch()
                if context is not None:
                    raise ValueError("these capacity cases require unconditional source batches")
                fixture.policy.begin_step(real, execution_limit=case["default_steps"])
                fixture.policy.abort_step()
            before = deepcopy(fixture.state_dict())
            rng_before = torch.random.get_rng_state().clone()
            observation = api_contract.validate_observation(fixture.observe(n=case["eval_samples"], seed=EVAL_SEED))
            purity = dict(fixture_state_unchanged=same(before, fixture.state_dict()),
                          observation_global_rng_unchanged=torch.equal(rng_before, torch.random.get_rng_state()),
                          construction_global_rng_unchanged=torch.equal(global_rng, rng_before),
                          completed_steps=fixture.completed_steps,
                          optimizer_states_empty=not fixture.trainer.opt_g.state and not fixture.trainer.opt_d.state)
            if not all(purity[key] for key in ("fixture_state_unchanged", "observation_global_rng_unchanged",
                                              "construction_global_rng_unchanged", "optimizer_states_empty")) or purity["completed_steps"] != 0:
                raise ValueError("zero-update capacity observer altered the actual training state or RNG")
            if proof_bindings(case, family) != bindings:
                raise ValueError("physical capacity source changed during observation")
            controls = api_images.oracle_controls(case_id)
            if any(control["passed"] != control["expected_pass"] for control in controls.values()):
                raise ValueError("original image gate does not discriminate its declared controls")
            directory = output / family / case_id
            directory.mkdir(parents=True)
            state_path, samples_path = directory / "state.pt", directory / "samples.npz"
            torch.save(before, state_path)
            view = observation["views"][0]
            np.savez_compressed(samples_path, target=api_contract.array(view["target"]),
                                samples=api_contract.array(view["samples"]),
                                **({} if real is None else {"lifecycle_real": real.detach().cpu().numpy()}))
            served = fixture.trainer.served_model()
            record = dict(family=family, case_id=case_id,
                          status="SUPPORTED" if observation["passed"] else "UNRESOLVED",
                          claim_scope=declaration["scope"], bindings=bindings,
                          observations=[dict(metrics=observation["metrics"], passed=observation["passed"],
                                             failed_bounds=observation["failed_bounds"])],
                          artifacts={name: dict(path=str(path), sha256=file_hash(path))
                                     for name, path in (("state", state_path), ("samples", samples_path))},
                          reference=dict(path=str(reference_path), sha256=expected_hash,
                                         summary_path=str(summary_path), summary_sha256=file_hash(summary_path),
                                         original_source_sha256=result["source_sha256"],
                                         original_task_sha256=result["task_sha256"],
                                         original_reference_fit_updates=result["diagnostic_fit_updates"],
                                         qualification_credit=False),
                          resolved_recipe=fixture.recipe.to_dict(),
                          serving=dict(source=served.source, output_noise_added=False,
                                       continuous_policy=fixture.recipe.continuous_policy,
                                       controller=None if served.controller is None else served.controller.state_dict(),
                                       prior_kind=fixture.recipe.prior_kind,
                                       backend_selection=fixture.policy.served_snapshot().get("backend_selection"),
                                       max_steps=fixture.max_steps,
                                       ordered_template_sha256=case["ordered_template_sha256"]),
                          controls={name: {key: value for key, value in control.items() if key != "views"}
                                    for name, control in controls.items()},
                          purity=purity, ordinary_training_updates=0, new_fitting_updates=0,
                          ordinary_qualification_credit=False, elapsed_seconds=time.monotonic() - start)
            record = json_value(record)
            json.dumps(record, allow_nan=False)
            records.append(record)
            print(json.dumps({"family": family, "case_id": case_id, "status": record["status"],
                              "metrics": observation["metrics"], "failed_bounds": observation["failed_bounds"]}), flush=True)
    receipt = dict(declaration=declaration, records=records, ordinary_qualification_credit=False)
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-neighbors", action="store_true", help="Declared zero-fit near same-mode support construction")
    parser.add_argument("--lifecycle-prelude", action="store_true", help="Select actual first-real-shape backend through begin_step/abort_step without optimizer updates")
    args = parser.parse_args()
    evaluate(args.archive, args.output, local_neighbors=args.local_neighbors, lifecycle_prelude=args.lifecycle_prelude)


if __name__ == "__main__":
    main()
