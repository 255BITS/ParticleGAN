"""Fresh zero-update public capacity for the declared critic-balance contrast.

Retained constructed fast parameters are explicit inputs. Their old policy,
optimizer, averages, clocks, sampled arrays and verdicts supply no new credit.
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

SCHEMA = "particlegan_critic_balance_capacity_v1"
SCIENTIFIC_BASE = "4749b2780add539df4bd8d2dd1d3cc9f002f77ad"
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5, "d_lr_mult": 2.25}
FAMILIES = ("atlas", "e22")
CASE_IDS = tuple(case_id for case_id, _ in study.DEFAULT_CASES)
SEED, EVALUATION_SEED = 24002, 34002
INPUT_CARD = ROOT / "reports/forge/family-winner-round1/policy-representation.json"
INPUT_CARD_SHA256 = "729f084b5e7b076cfb5d06d65810cec1b823f5df0ebe645767d8cd43fcb74c38"
CLAIM_SCOPE = ("Necessary CPU snapshot capacity under the actual selected public served law, "
               "at zero optimizer updates after one original real-data lifecycle prelude. "
               "No fitting, CUDA training, convergence, robustness or default qualification.")


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
    manifest = {**source["files_sha256"], Path(__file__).relative_to(ROOT).as_posix(): source["binder_sha256"]}
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
    manifest = {}
    for name in sorted(paths):
        manifest[name] = _hash(ROOT / name)
        if manifest[name] != _pinned_hash(name):
            raise ValueError(f"protected scientific source changed: {name}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    source = dict(scientific_base_commit=SCIENTIFIC_BASE, files_sha256=manifest,
                  binder_sha256=_hash(__file__), reproducer_commit=commit)
    _verify_committed_source(source)
    return source


def _construction_record(case, family):
    if _hash(INPUT_CARD) != INPUT_CARD_SHA256:
        raise ValueError("frozen construction-input card changed")
    records = json.loads(INPUT_CARD.read_text())["records"]
    matches = [record for record in records if (record["family"], record["case_id"]) == (family, case["id"])]
    if len(matches) != 1:
        raise ValueError("construction input must have one exact family/case record")
    record = matches[0]
    if record["bindings"]["case_sha256"] != study.digest(case):
        raise ValueError("construction input describes a different host/question")
    return dict(card=dict(path=str(INPUT_CARD), sha256=INPUT_CARD_SHA256),
                original_record=deepcopy(record),
                use="Fast G/D/prior and learned output-noise parameters only; no old verdict/source credit")


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
    if not study._finite_tensors(models) or not study._finite_tensors(api.get("output_noise", {})):
        raise ValueError("constructed learned parameters are nonfinite")
    return api


def _construct(case, family, original):
    fixture = contract.build(case, device="cpu", seed=SEED, recipe_name=family,
                             max_steps=case["default_steps"], recipe_overrides=SHARED_OVERRIDES)
    initial = fixture.trainer
    for name in ("G", "D", "prior"):
        getattr(initial, name).load_state_dict(original["models"][name], strict=True)
    # The constructor derives all EMA/controller/tester/optimizer ownership
    # afresh from the installed fast parameters and the candidate Recipe.
    options = dict(seed=SEED, max_steps=case["default_steps"],
                   serial_backward=initial.serial_backward,
                   optimizer_options=deepcopy(initial.optimizer_options),
                   penalty_options=deepcopy(initial.penalty_options))
    if initial.model_generator is not None:
        options["model_generator"] = torch.Generator(device="cpu").manual_seed(SEED + 8)
    fixture.trainer = GANTrainer(fixture.recipe, initial.G, initial.D, prior=initial.prior, **options)
    if fixture.trainer.log_output_sigma is not None:
        noise = original.get("output_noise", {}).get("log_sigma")
        if not isinstance(noise, torch.Tensor):
            raise ValueError("construction lacks its explicit learned output-noise parameter")
        with torch.no_grad():
            fixture.trainer.log_output_sigma.copy_(noise)
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
                resolved_recipe=study.resolved_recipe(case, family, SHARED_OVERRIDES),
                construction_inputs=_construction_record(case, family),
                sampling_execution=dict(device="cpu", cpu_threads=1, ambient_cpu_seed=SEED,
                                        deterministic_algorithms=True, cudnn_benchmark=False),
                ordinary_training_updates=0, fitting_updates=0, ordinary_qualification_credit=False)


def _construction_metadata():
    return dict(parameter_fields=["G", "D", "prior", "output_noise.log_sigma"],
                old_policy_optimizer_averages_clocks_transplanted=False,
                fresh_policy_initialized_from_installed_fast_models=True,
                optimizer_states_empty=True, ambient_cpu_seed=SEED)


def _serving_metadata(fixture, state, case):
    served = fixture.trainer.served_model()
    api = state.get("trainer", state.get("api_state"))
    return api_run.json_value(dict(source=served.source,
                                  output_noise_primary=case.get("kind") == "native100",
                                  output_sigma=fixture.trainer.output_sigma(),
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
        record.update(status="SUPPORTED" if observation["passed"] else "UNRESOLVED",
                      observations=[observation],
                      lifecycle=dict(public_preludes=1, first_real_step=1,
                                     execution_limit=case["default_steps"], completed_steps=0,
                                     scope="Original first real batch: begin_step then abort_step; controller/reservoir observation retained"),
                      serving=_serving_metadata(fixture, state, case),
                      observer_purity=dict(state_unchanged=True, global_rng_unchanged=True,
                                           complete_public_state_sha256=_fingerprint(state)),
                      construction=_construction_metadata())
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
    if any(record.get(key) != value for key, value in api_run.json_value(expected).items()):
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
        if record.get("serving") != _serving_metadata(fixture, state, case):
            raise ValueError("selected serving or backend metadata differs")
        if observations != [actual]:
            raise ValueError("capacity metrics/count/seed/verdict differs from actual public observation")
        if set(arrays.files) != set(expected_arrays) or any(
                arrays[key].dtype != value.dtype or arrays[key].shape != value.shape
                or np.ascontiguousarray(arrays[key]).tobytes() != np.ascontiguousarray(value).tobytes()
                for key, value in expected_arrays.items()):
            raise ValueError("capacity arrays differ from the exact actual public sampler/targets")
    expected_status = "SUPPORTED" if actual["passed"] else "UNRESOLVED"
    if record["status"] != expected_status:
        raise ValueError("capacity status differs from its original binary metric")
    if record.get("construction") != _construction_metadata():
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
