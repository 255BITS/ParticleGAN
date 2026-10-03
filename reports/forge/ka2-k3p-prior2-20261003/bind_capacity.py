"""Fresh zero-update public capacity for the fixed KA2/K3P prior2 contrast.

Retained fast parameters are explicit inputs; vector/native populations are
analytic, zero-fit constructions on the original public host scaffolds.
Old policy, optimizer, averages, clocks, arrays and verdicts supply no new credit.
Only ``main --output NEW_DIRECTORY`` captures the sixteen new CPU witnesses.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
from statistics import NormalDist
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from particlegan import GANTrainer
from benchmarks.toy_audit import api_contract as contract, api_family_search as study, api_run, api_vectors
from benchmarks.toy100 import problems as native_problems

SCHEMA = "particlegan_ka2_k3p_prior2_capacity_v1"
SCIENTIFIC_BASE = "4749b2780add539df4bd8d2dd1d3cc9f002f77ad"
import importlib.util
_PROTOCOL_PATH = Path(__file__).with_name("protocol.py")
_PROTOCOL_SPEC = importlib.util.spec_from_file_location("ka2_k3p_prior2_capacity_protocol", _PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(_PROTOCOL_SPEC)
_PROTOCOL_SPEC.loader.exec_module(protocol)
SHARED_OVERRIDES = {"lr": .006375, "prior_lr_mult": 2.0, "d_lr_mult": 1.0}
FAMILIES = ("ka2", "k3p")
CASE_IDS = tuple(case_id for case_id, _ in study.DEFAULT_CASES)
SEED, EVALUATION_SEED = 24002, 34002
INPUT_CARD = ROOT / "reports/forge/family-winner-round1/policy-representation.json"
INPUT_CARD_SHA256 = "729f084b5e7b076cfb5d06d65810cec1b823f5df0ebe645767d8cd43fcb74c38"
CLAIM_SCOPE = ("Necessary CPU snapshot capacity under the actual selected public served law, "
               "at zero optimizer updates after one original real-data lifecycle prelude. "
               "Fixed KA2/K3P prior2 tuple .006375/2/1, fast-only, no DV12, "
               "fixed output-noise schedule at clock zero. "
               "No fitting, CUDA training, terminal-horizon solvability, convergence or default qualification.")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(value):
    digest = hashlib.sha256()
    def visit(part):
        digest.update(type(part).__name__.encode())
        if isinstance(part, torch.Tensor):
            array = part.detach().cpu().contiguous()
            digest.update(str((array.dtype, tuple(array.shape))).encode())
            digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(part, np.ndarray):
            digest.update(str((part.dtype, part.shape)).encode())
            digest.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, dict):
            for key in sorted(part, key=str):
                visit(key); visit(part[key])
        elif isinstance(part, (list, tuple)):
            for child in part:
                visit(child)
        else:
            digest.update(repr(part).encode())
    visit(value)
    return digest.hexdigest()


def _globals():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def _runtime():
    if torch.get_num_threads() != 1 or torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
        raise ValueError("capacity requires one CPU thread, no initialized CUDA context and no visible CUDA devices")
    return dict(device="cpu", python=platform.python_version(), torch=str(torch.__version__),
                cuda_build=torch.version.cuda, machine=platform.machine(), cpu_threads=1)


@contextmanager
def _capacity_session():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    try:
        with api_run.isolated_evaluation():
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            # Ambient checkpoint RNG is explicit for the construction only;
            # original provider-owned named streams still use their seeds.
            torch.manual_seed(SEED)
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark


def selected_cases():
    definitions = contract.discover()
    return {name: definitions[name] for name in CASE_IDS}


def _canonical(case, family):
    if (tuple(protocol.FAMILIES) != FAMILIES
            or _fingerprint(protocol.OVERRIDES) != _fingerprint(SHARED_OVERRIDES)):
        raise ValueError("prior2 protocol must resolve the exact declared families and three-field tuple")
    if family not in FAMILIES or case.get("id") not in CASE_IDS:
        raise ValueError("capacity is limited to the declared two families and eight cases")
    actual = selected_cases()[case["id"]]
    if study.digest(case) != study.digest(actual):
        raise ValueError("case, gates, sampling law or full default resources changed")
    contract.validate_recipe_overrides(actual, family, SHARED_OVERRIDES)
    return actual


def _check_import_origins():
    prefixes = ("particlegan", "benchmarks.toy_audit", "benchmarks.toy100",
                "benchmarks.transfer_suite", "experiments.forge", "lib.toy_models")
    for name, module in tuple(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        location = getattr(module, "__file__", None)
        expected = ROOT.joinpath(*name.split("."))
        if location is None:
            paths = {Path(path).resolve() for path in getattr(module, "__path__", [])}
            if not paths or paths != {expected.resolve()} or (expected / "__init__.py").exists():
                raise ValueError(f"public namespace was imported from a different source: {name}")
        elif Path(location).resolve() not in {expected.with_suffix(".py").resolve(), (expected / "__init__.py").resolve()}:
            raise ValueError(f"public module was imported from a different source: {name}")


@lru_cache(maxsize=None)
def _pinned_hash(name):
    blob = subprocess.run(["git", "show", f"{SCIENTIFIC_BASE}:{name}"], cwd=ROOT,
                          check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


@lru_cache(maxsize=None)
def _git_blob_hash(commit, name):
    blob = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                          check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def _verify_committed_source(source):
    commit = source.get("reproducer_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("capacity needs an exact committed reproducer identity")
    manifest = {**source["files_sha256"], **source["new_test_files_sha256"],
                Path(__file__).relative_to(ROOT).as_posix(): source["binder_sha256"]}
    if any(_git_blob_hash(commit, name) != value for name, value in manifest.items()):
        raise ValueError("committed reproducer/scientific source differs from captured bytes")


def _source_binding(case, family):
    if contract.ROOT.resolve() != ROOT.resolve():
        raise ValueError("public modules were imported from a different checkout")
    _check_import_origins()
    bindings = study.proof_bindings(case, family)
    # The new contrast covers both providers: bind the entire scientific
    # source union even for a record from just one host.
    paths = {name for selected in selected_cases().values()
             for name in study.proof_bindings(selected, family)["source_files_sha256"]}
    paths.update(bindings["source_files_sha256"])
    paths.update(("benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_family_search.py"))
    paths.update(("experiments/forge/contracts.py", "experiments/forge/boundaries.py",
                  "experiments/forge/configuration_search.py"))
    paths.add("configs/forge/tasks/ring16_acquisition.json")
    manifest = {}
    for name in sorted(paths):
        manifest[name] = _hash(ROOT / name)
        if manifest[name] != _pinned_hash(name):
            raise ValueError(f"protected scientific source changed: {name}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    source = dict(scientific_base_commit=SCIENTIFIC_BASE, files_sha256=manifest,
                  new_test_files_sha256={_PROTOCOL_PATH.relative_to(ROOT).as_posix(): _hash(_PROTOCOL_PATH)},
                  binder_sha256=_hash(__file__), reproducer_commit=commit)
    _verify_committed_source(source)
    return source


def _construction_record(case, family):
    if _hash(INPUT_CARD) != INPUT_CARD_SHA256:
        raise ValueError("frozen construction-input card changed")
    records = json.loads(INPUT_CARD.read_text())["records"]
    # This explicit provenance source does not assume the input card contains
    # a KA2/K3P record. Its verdict and family law confer no new credit.
    matches = [record for record in records if (record["family"], record["case_id"]) == ("atlas", case["id"])]
    if len(matches) != 1:
        raise ValueError("construction input must have one exact original Atlas/case parameter record")
    record = matches[0]
    if record["bindings"]["case_sha256"] != study.digest(case):
        raise ValueError("construction input describes a different host/question")
    return dict(card=dict(path=str(INPUT_CARD), sha256=INPUT_CARD_SHA256),
                original_record=deepcopy(record),
                use="Images: fast G/D/prior only; vector/native: fast D only plus analytic current-host G/prior construction. No old noise, policy, optimizer, averages, clocks, samples or verdict credit")


def _input_state(construction, case):
    old = construction["original_record"]
    for artifact in old["artifacts"].values():
        path = Path(artifact["path"])
        if not path.is_absolute() or not path.is_file() or _hash(path) != artifact["sha256"]:
            raise ValueError("retained construction artifact is missing or changed")
    state = torch.load(old["artifacts"]["state"]["path"], map_location="cpu", weights_only=True)
    api = state.get("trainer", state.get("api_state"))
    if (not isinstance(api, dict) or type(api.get("completed_steps")) is not int
            or api["completed_steps"] != 0 or api.get("device") != "cpu"
            or api.get("max_steps", api.get("recipe", {}).get("total_steps")) != case["default_steps"]):
        raise ValueError("construction input must be the actual zero-update full-horizon CPU capture")
    models = {name: api["models"][name] for name in ("G", "D", "prior")}
    if not study._finite_tensors(models):
        raise ValueError("constructed learned parameters are nonfinite")
    return api


def _coordinate_identity(generator):
    """Install an exact linear coordinate map on the existing MLP scaffold.

    phi(x)-phi(-x)=(1+a)x transports each coordinate using its signed pair;
    every original hidden layer, width and parameter owner stays in place.
    """
    if not isinstance(getattr(generator, "net", None), torch.nn.Sequential):
        raise ValueError("vector identity construction requires the original sequential MLP")
    layers = list(generator.net)
    if (len(layers) < 3 or len(layers) % 2 != 1
            or any(not isinstance(layers[i], torch.nn.Linear) for i in range(0, len(layers), 2))
            or any(not isinstance(layers[i], torch.nn.LeakyReLU) or layers[i].negative_slope != .2
                   for i in range(1, len(layers), 2))):
        raise ValueError("vector identity needs original Linear/LeakyReLU(.2) hidden layers")
    linears = layers[::2]
    if (linears[0].in_features < 2 or linears[-1].out_features != 2
            or any(layer.out_features < 4 for layer in linears[:-1])):
        raise ValueError("original MLP has insufficient signed-coordinate slots")
    pair = torch.tensor([[1., -1., 0., 0.], [-1., 1., 0., 0.],
                         [0., 0., 1., -1.], [0., 0., -1., 1.]]) / 1.2
    with torch.no_grad():
        for layer in linears:
            layer.weight.zero_()
            if layer.bias is None:
                raise ValueError("original coordinate scaffold requires its ordinary bias parameters")
            layer.bias.zero_()
        linears[0].weight[:4, :2].copy_(torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]))
        for layer in linears[1:-1]:
            layer.weight[:4, :4].copy_(pair)
        linears[-1].weight[:, :4].copy_(torch.tensor([[1., -1., 0., 0.], [0., 0., 1., -1.]]) / 1.2)


def _component_counts(masses, particles):
    expected = np.asarray(masses, dtype=np.float64) * particles
    if (type(particles) is not int or particles < 1 or expected.ndim != 1
            or not np.isfinite(expected).all() or (expected <= 0).any()
            or not math.isclose(float(sum(masses)), 1., abs_tol=1e-12)):
        raise ValueError("construction requires positive normalized component mass")
    counts = np.floor(expected).astype(int)
    order = sorted(range(len(counts)), key=lambda i: (-(expected[i] - counts[i]), i))
    counts[order[:particles - int(counts.sum())]] += 1
    if (counts < 4).any():
        raise ValueError("finite Gaussian design needs at least four original rows per component")
    return counts


def _vector_population(case):
    """Nonrandom finite Gaussian design; no sampler, optimizer or fitting."""
    spec = case["spec"]
    if spec.get("kind") != "gaussian_mixture" or case["z_dim"] < 2:
        raise ValueError("analytic row design only covers the declared Gaussian vector laws")
    counts = _component_counts(spec["masses"], case["particles"])
    normal = NormalDist()
    points = []
    for count, center, covariance in zip(counts, spec["means"], spec["covariances"]):
        bits = int(np.ceil(np.log2(count)))
        inverse = [int(f"{j:0{bits}b}"[::-1], 2) for j in range(count)]
        uniform = np.stack([(np.arange(count) + .5) / count,
                            (np.asarray(inverse) + .5) / 2 ** bits], axis=1)
        z = np.asarray([[normal.inv_cdf(float(value)) for value in row] for row in uniform])
        z -= z.mean(axis=0)
        values, vectors = np.linalg.eigh(z.T @ z / count)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("finite coordinate design is singular")
        z = z @ (vectors @ np.diag(1 / np.sqrt(values)) @ vectors.T)
        chol = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
        points.append(np.asarray(center) + z @ chol.T)
    rows = np.zeros((case["particles"], case["z_dim"]), dtype=np.float32)
    rows[:, :2] = np.concatenate(points)
    if not np.isfinite(rows).all():
        raise ValueError("analytic vector row construction is nonfinite")
    return torch.from_numpy(rows)


def _native_population(case):
    """200 original rows per mode; fixed radial quantiles/antipodal pairs."""
    if (case.get("kind") != "native100" or case.get("moving")
            or case.get("particles") != 20000 or case.get("z_dim") != 2):
        raise ValueError("native construction requires the three original static 20k-row hosts")
    centers, sigma = native_problems.evaluation_geometry(case["problem"], dtype=torch.float64)
    if centers.shape != (100, 2) or sigma != .03:
        raise ValueError("native target geometry differs from its original 100-mode law")
    offsets = _radial_pairs(100, sigma)
    rows = centers.numpy()[:, None, :] + offsets[None, :, :]
    return torch.from_numpy(rows.reshape(20000, 2).astype(np.float32))


def _radial_pairs(pairs, sigma):
    if type(pairs) is not int or pairs < 2 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive radial-pair count and scale required")
    j = np.arange(pairs, dtype=np.float64)
    radius = sigma * np.sqrt(-2 * np.log(1 - (j + .5) / pairs))
    theta = 2 * np.pi * j / ((1 + np.sqrt(5)) / 2)
    half = radius[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return np.concatenate([half, -half], axis=0)


def _construct(case, family, original):
    fixture = contract.build(case, device="cpu", seed=SEED, recipe_name=family,
                             max_steps=case["default_steps"], recipe_overrides=SHARED_OVERRIDES)
    initial = fixture.trainer
    initial.D.load_state_dict(original["models"]["D"], strict=True)
    if case["provider"] == "api_images":
        initial.G.load_state_dict(original["models"]["G"], strict=True)
        initial.prior.load_state_dict(original["models"]["prior"], strict=True)
    elif case["kind"] == "native100":
        if (not isinstance(initial.G, torch.nn.Linear) or initial.G.in_features != 2
                or initial.G.out_features != 2 or initial.G.bias is None):
            raise ValueError("native construction must use the original public learned Linear(2,2) scaffold")
        with torch.no_grad():
            initial.G.weight.copy_(torch.eye(2)); initial.G.bias.zero_()
            initial.prior.z.copy_(_native_population(case))
    elif case["kind"] == "vector":
        _coordinate_identity(initial.G)
        with torch.no_grad():
            initial.prior.z.copy_(_vector_population(case))
    else:
        raise ValueError("unsupported host construction; no alternate model or sampler is permitted")
    # The constructor derives all EMA/controller/tester/optimizer ownership
    # afresh from the installed fast parameters and the candidate Recipe.
    options = dict(seed=SEED, max_steps=case["default_steps"],
                   serial_backward=initial.serial_backward,
                   optimizer_options=deepcopy(initial.optimizer_options),
                   penalty_options=deepcopy(initial.penalty_options))
    if initial.model_generator is not None:
        options["model_generator"] = torch.Generator(device="cpu").manual_seed(SEED + 8)
    fixture.trainer = GANTrainer(fixture.recipe, initial.G, initial.D, prior=initial.prior, **options)
    if (fixture.trainer.log_output_sigma is not None or fixture.trainer.policy.controller is not None
            or fixture.trainer.recipe.output_noise_mode != "fixed" or fixture.trainer.recipe.serve_average != 0):
        raise ValueError("fresh family must retain fixed noise, fast serving and no DV12 controller")
    if case["provider"] == "api_images":
        fixture.policy = fixture.trainer.policy
        real, context = fixture._real_batch()
        if context is not None:
            raise ValueError("only the unconditional source image hosts are declared")
    else:
        real = api_vectors._target(case, case["batch_size"], fixture.data_rng, 1, device="cpu")
    fixture.trainer.policy.begin_step(real, execution_limit=case["default_steps"])
    fixture.trainer.policy.abort_step()
    if (fixture.completed_steps != 0 or fixture.trainer.completed_steps != 0
            or fixture.trainer.opt_g.state or fixture.trainer.opt_d.state):
        raise ValueError("capacity construction advanced an optimizer or fixture clock")
    return fixture, real.detach().cpu().numpy()


def _observe(fixture, case):
    state_before = deepcopy(fixture.state_dict())
    global_before = _fingerprint(_globals())
    observed = contract.validate_observation(fixture.observe(n=case["eval_samples"], seed=EVALUATION_SEED))
    if (_fingerprint(fixture.state_dict()) != _fingerprint(state_before)
            or _fingerprint(_globals()) != global_before):
        raise ValueError("capacity observer altered public state or global/training RNG")
    arrays, views = {}, []
    for index, view in enumerate(observed["views"]):
        for role in ("target", "samples"):
            array = contract.array(view[role])
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError("capacity view arrays must be finite numeric values")
            arrays[f"view{index}_{role}"] = np.array(array, copy=True)
        views.append(api_run.json_value({key: value for key, value in view.items() if key not in {"target", "samples"}}))
    samples = arrays["view0_samples"]
    if len(samples) != case["eval_samples"]:
        raise ValueError("capacity must retain the full original evaluation count")
    primary = study._score(case, samples, 0)
    failures = sorted(set(primary["failed_bounds"]))
    if observed["passed"] is not primary["passed"] or observed["failed_bounds"] != failures:
        raise ValueError("actual observer verdict differs from its original primary gate")
    row = dict(completed_steps=0, evaluation_seed=EVALUATION_SEED, samples=case["eval_samples"],
               samples_key="view0_samples", metrics=primary["metrics"],
               passed=primary["passed"], failed_bounds=failures,
               observer_metrics=observed["metrics"], views=views)
    return api_run.json_value(row), arrays, state_before


def _base_record(case, family):
    runtime = _runtime()
    case = _canonical(case, family)
    return dict(schema=SCHEMA, family=family, case_id=case["id"], case=deepcopy(case),
                claim_scope=CLAIM_SCOPE, bindings=study.proof_bindings(case, family),
                source=_source_binding(case, family), runtime=runtime, seed=SEED,
                recipe_overrides=deepcopy(SHARED_OVERRIDES),
                requested_recipe_overrides=deepcopy(SHARED_OVERRIDES),
                resolved_recipe=protocol.resolved_recipe(case, family, SHARED_OVERRIDES),
                family_law=protocol.family_law(case, family),
                construction_inputs=_construction_record(case, family),
                sampling_execution=dict(device="cpu", cpu_threads=1, ambient_cpu_seed=SEED,
                                        deterministic_algorithms=True, cudnn_benchmark=False),
                ordinary_training_updates=0, fitting_updates=0, ordinary_qualification_credit=False)


def _construction_metadata(case):
    copied = ["G", "D", "prior"] if case["provider"] == "api_images" else ["D"]
    design = ("explicit retained image fast parameters" if case["provider"] == "api_images" else
              "native 100 radial antipodal pairs per mode, sigma .03 at zero clock" if case["kind"] == "native100" else
              "mass-allocated stratified Gaussian row design with complete covariance; exact coordinate-identity MLP")
    return dict(parameter_fields=copied, constructed_population=design, original_input_family="atlas",
                original_host_scaffolds_preserved=True, fitting_updates=0,
                learned_noise_transplanted=False, terminal_horizon_capacity_claim=False,
                old_policy_optimizer_averages_clocks_transplanted=False,
                fresh_policy_initialized_from_installed_fast_models=True,
                optimizer_states_empty=True, ambient_cpu_seed=SEED)


def _serving_metadata(fixture, state, case):
    served = fixture.trainer.served_model()
    api = state.get("trainer", state.get("api_state"))
    return api_run.json_value(dict(source=served.source,
                                  output_noise_primary=case.get("kind") == "native100",
                                  output_sigma=fixture.trainer.output_sigma(),
                                  latent_perturbation_enabled=served.controller is not None,
                                  backend_selection=api.get("backend_selection")))


def capture_record(case, family, output):
    """One declared sampler capture; no fitting or ordinary optimizer update."""
    record = _base_record(case, family)
    original = _input_state(record["construction_inputs"], case)
    directory = Path(output).resolve() / family / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    with _capacity_session():
        fixture, real = _construct(case, family, original)
        observation, arrays, state = _observe(fixture, case)
        arrays["lifecycle_real"] = real
        arrays["construction_population"] = fixture.trainer.prior.z.detach().cpu().numpy().copy()
        record.update(status="SUPPORTED" if observation["passed"] else "UNRESOLVED",
                      observations=[observation],
                      lifecycle=dict(public_preludes=1, first_real_step=1,
                                     execution_limit=case["default_steps"], completed_steps=0,
                                     scope="Original first real batch: begin_step then abort_step; controller/reservoir observation retained"),
                      serving=_serving_metadata(fixture, state, case),
                      observer_purity=dict(state_unchanged=True, global_rng_unchanged=True,
                                           complete_public_state_sha256=_fingerprint(state)),
                      construction=_construction_metadata(case))
        state_path, samples_path = directory / "state.pt", directory / "samples.npz"
        torch.save(state, state_path)
        np.savez_compressed(samples_path, **arrays)
    record["artifacts"] = {role: dict(path=str(path), sha256=_hash(path), bytes=path.stat().st_size)
                           for role, path in (("state", state_path), ("samples", samples_path))}
    if record["source"] != _source_binding(case, family):
        raise ValueError("source changed during the capacity capture")
    return api_run.json_value(record)


def _artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"}:
        raise ValueError("exact artifact path/hash/byte binding required")
    path = Path(artifact["path"])
    if (not path.is_absolute() or not path.is_file() or type(artifact["bytes"]) is not int
            or path.stat().st_size != artifact["bytes"] or _hash(path) != artifact["sha256"]):
        raise ValueError("capacity artifact missing or changed")
    return path


def verify_record(record, case, family):
    """Validate positive or honest negative evidence; never upgrade a verdict.

    Complete UNRESOLVED records get the same exact state/array/sampler checks.
    BLOCKED records bind the preparation error and carry no capacity proof.
    """
    expected = _base_record(case, family)
    if "reproducer_commit" in expected["source"]:
        # Later publication-only commits do not invalidate an immutable
        # capture; its original committed blobs must still match every byte.
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("reproducer_commit"), str):
            raise ValueError("capacity needs its committed reproducer/source binding")
        expected["source"]["reproducer_commit"] = source["reproducer_commit"]
    if any(_fingerprint(record.get(key)) != _fingerprint(value)
           for key, value in api_run.json_value(expected).items()):
        raise ValueError("capacity source/case/Recipe/input/runtime/request binding differs")
    if "reproducer_commit" in expected["source"]:
        _verify_committed_source(record["source"])
    if "elapsed_seconds" in record and (isinstance(record["elapsed_seconds"], bool)
            or not isinstance(record["elapsed_seconds"], (int, float))
            or not math.isfinite(record["elapsed_seconds"]) or record["elapsed_seconds"] < 0):
        raise ValueError("capacity software elapsed time must be finite and nonnegative")
    if record.get("status") == "BLOCKED":
        if (record.get("observations") != [] or record.get("artifacts") != {}
                or record.get("capacity_credit") is not False or not isinstance(record.get("error"), dict)):
            raise ValueError("blocked preparation cannot contain a capacity certificate")
        error_path = _artifact(record["error_artifact"])
        if json.loads(error_path.read_text()) != record["error"]:
            raise ValueError("blocked error differs from its retained failure artifact")
        if set(record["error"]) != {"stage", "type", "message", "traceback"} or not all(
                isinstance(value, str) and value for value in record["error"].values()):
            raise ValueError("exact nonempty preparation error required")
        for artifact in record.get("available_artifacts", {}).values():
            _artifact(artifact)
        return deepcopy(record)
    if record.get("status") not in {"SUPPORTED", "UNRESOLVED"} or set(record.get("artifacts", {})) != {"state", "samples"}:
        raise ValueError("complete capacity record needs its state and sampled arrays")
    state_path = _artifact(record["artifacts"]["state"])
    samples_path = _artifact(record["artifacts"]["samples"])
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    api = state.get("trainer", state.get("api_state"))
    if (not isinstance(api, dict) or type(api.get("completed_steps")) is not int or api["completed_steps"] != 0
            or api.get("max_steps", api.get("recipe", {}).get("total_steps")) != case["default_steps"]
            or api_run.json_value(api.get("recipe")) != expected["resolved_recipe"] or api.get("device") != "cpu"):
        raise ValueError("candidate needs an exact zero-update CPU Recipe and original full horizon")
    study._check_health(state, case, expected["resolved_recipe"], completed_steps=0)
    if any(optimizer.get("state") for optimizer in api["optimizers"]):
        raise ValueError("capacity optimizer state contains update credit")
    if record.get("observer_purity") != dict(state_unchanged=True, global_rng_unchanged=True,
                                             complete_public_state_sha256=_fingerprint(state)):
        raise ValueError("capacity state/purity fingerprint differs")
    observations = record.get("observations")
    if not isinstance(observations, list) or len(observations) != 1:
        raise ValueError("one preregistered full-count snapshot required")
    original = _input_state(expected["construction_inputs"], case)
    with _capacity_session(), np.load(samples_path, allow_pickle=False) as arrays:
        fixture, real = _construct(case, family, original)
        if _fingerprint(state) != _fingerprint(fixture.state_dict()):
            raise ValueError("capacity checkpoint differs from fresh public parameter construction")
        if case["provider"] == "api_vectors":
            fixture.load_state_dict(deepcopy(state))
        else:
            fixture.trainer.load_state_dict(deepcopy(state["api_state"]))
            fixture.data_generator.set_state(state["data_generator"])
        actual, expected_arrays, _ = _observe(fixture, case)
        expected_arrays["lifecycle_real"] = real
        expected_arrays["construction_population"] = fixture.trainer.prior.z.detach().cpu().numpy().copy()
        if record.get("serving") != _serving_metadata(fixture, state, case):
            raise ValueError("selected serving or backend metadata differs")
        if _fingerprint(observations) != _fingerprint([actual]):
            raise ValueError("capacity metrics/count/seed/verdict differs from actual public observation")
        if set(arrays.files) != set(expected_arrays) or any(
                arrays[key].dtype != value.dtype or arrays[key].shape != value.shape
                or np.ascontiguousarray(arrays[key]).tobytes() != np.ascontiguousarray(value).tobytes()
                for key, value in expected_arrays.items()):
            raise ValueError("capacity arrays differ from the exact actual public sampler/targets")
    expected_status = "SUPPORTED" if actual["passed"] else "UNRESOLVED"
    if record["status"] != expected_status:
        raise ValueError("capacity status differs from its original binary metric")
    if record.get("construction") != _construction_metadata(case):
        raise ValueError("fresh parameter-only construction contract differs")
    if record.get("lifecycle") != dict(public_preludes=1, first_real_step=1,
                                       execution_limit=case["default_steps"], completed_steps=0,
                                       scope="Original first real batch: begin_step then abort_step; controller/reservoir observation retained"):
        raise ValueError("original public lifecycle prelude differs")
    return deepcopy(record)


def export(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases, records = selected_cases(), []
    started = time.monotonic()
    def save(*, prerequisite=None):
        packet = dict(schema=SCHEMA, status="BLOCKED_PREREQUISITE" if prerequisite else
                      "COMPLETE" if len(records) == 16 else "CAPTURING",
                      claim_scope=CLAIM_SCOPE, required_records=16, records=records,
                      requested_cells=[dict(family=family, case_id=name) for family in FAMILIES for name in CASE_IDS],
                      ordinary_training_updates=0, fitting_updates=0, ordinary_qualification_credit=False,
                      elapsed_seconds=time.monotonic() - started)
        if prerequisite:
            packet["prerequisite_failure"] = prerequisite
        api_run.write_json(output / "capacity.json", packet)
        return packet
    for family in FAMILIES:
        for name in CASE_IDS:
            case, began = cases[name], time.monotonic()
            try:
                base = api_run.json_value(_base_record(case, family))
            except Exception as error:
                # An unattested source/card prerequisite cannot truthfully
                # become a verified per-case certificate. Preserve its raw
                # failure and entire requested denominator, then stop.
                problem = dict(stage="unattested_source_or_card_prerequisite", type=type(error).__name__,
                               message=str(error), traceback=traceback.format_exc())
                error_path = output / "prerequisite-error.json"
                api_run.write_json(error_path, problem)
                return save(prerequisite=dict(family=family, case_id=name, error=problem,
                            artifact=dict(path=str(error_path), sha256=_hash(error_path), bytes=error_path.stat().st_size),
                            qualification_credit=False))
            try:
                record = capture_record(case, family, output)
                verify_record(record, case, family)
            except Exception as error:
                record = base
                problem = dict(stage="capacity_preparation_or_verification", type=type(error).__name__,
                               message=str(error), traceback=traceback.format_exc())
                error_path = output / family / name / "preparation-error.json"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                api_run.write_json(error_path, problem)
                record.update(status="BLOCKED", observations=[], artifacts={}, capacity_credit=False,
                              error=problem, error_artifact=dict(path=str(error_path), sha256=_hash(error_path),
                                                                 bytes=error_path.stat().st_size))
                record["available_artifacts"] = {
                    role: dict(path=str(path), sha256=_hash(path), bytes=path.stat().st_size)
                    for role, filename in (("state", "state.pt"), ("samples", "samples.npz"))
                    if (path := error_path.parent / filename).is_file()}
            record["elapsed_seconds"] = time.monotonic() - began
            records.append(record)
            save()
            print(json.dumps(dict(family=family, case_id=name, status=record["status"],
                                  elapsed_seconds=record["elapsed_seconds"])), flush=True)
    return save()


def verify_packet(packet):
    cases = selected_cases()
    keys = [(row.get("family"), row.get("case_id")) for row in packet.get("records", [])]
    requested = [dict(family=family, case_id=name) for family in FAMILIES for name in CASE_IDS]
    if (packet.get("schema") != SCHEMA or packet.get("status") != "COMPLETE"
            or packet.get("claim_scope") != CLAIM_SCOPE or packet.get("required_records") != 16
            or packet.get("requested_cells") != requested or packet.get("ordinary_training_updates") != 0
            or packet.get("fitting_updates") != 0 or packet.get("ordinary_qualification_credit") is not False
            or set(keys) != {(family, name) for family in FAMILIES for name in CASE_IDS} or len(keys) != 16):
        raise ValueError("all sixteen distinct declared outcomes and zero-update scope must be retained")
    for record in packet["records"]:
        verify_record(record, cases[record["case_id"]], record["family"])
    return deepcopy(packet)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if (args.output is None) == (args.verify is None):
        parser.error("choose exactly one fresh export directory or retained packet to verify")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.output is not None:
        packet = export(args.output)
    else:
        try:
            packet_hash = _hash(args.verify)
            packet = json.loads(args.verify.read_text())
            verify_packet(packet)
            if _hash(args.verify) != packet_hash:
                raise ValueError("capacity packet changed during CPU verification")
        except Exception as error:
            print(json.dumps(dict(status="INVALID", verified=False, type=type(error).__name__,
                                  error=str(error))), flush=True)
            return 2
        print(json.dumps(dict(status="VERIFIED", verified=True, packet_sha256=packet_hash,
                              required_records=16, ordinary_training_updates=0,
                              family_supported={family: all(record["status"] == "SUPPORTED"
                                                for record in packet["records"] if record["family"] == family)
                                                for family in FAMILIES})), flush=True)
    if packet["status"] != "COMPLETE":
        return 2
    return 0 if all(record["status"] == "SUPPORTED" for record in packet["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
