"""Root-only, once-only Gaussian observer through the maintained supervisor."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
QUEUE = Path("/ml2/hypergan/ParticleGAN-single-recipe/runs/forge")
OBSERVER = "benchmarks/toy_audit/gaussian2d_observer.py"
A_PROTOCOL = "reports/forge/gaussian2d-current-c6-20261004/protocol.json"
OBSERVER_SHA = "ef607609d377c9747d74a3e490a0092e59f1a4b027f99e23f60a10b014ba0717"
API_RUN_SHA = "c3d54398256eb0aba6dd20bb77e6e9a9a71bb29cc5bd4df8b8f9bce1ef5687b9"
A_PROTOCOL_SHA = "2de6fb8894a53b7d15c0787132247316a6ac1267550cee811350d1dcb66c7c26"
WRAPPER = "observer-controls/gaussian2d-v1/run_supervised.py"
PROTOCOL_RELATIVE = "observer-controls/gaussian2d-v1/protocol.json"
PROPOSAL_RELATIVE = "observer-controls/gaussian2d-v1/proposal.json"
SCOPE_RELATIVE = "observer-controls/gaussian2d-v1/PROPOSAL.md"
CASE = "api_gaussian2d_c6_observer_gpu_v1"
FAMILY = "atlas_gaussian2d_c6_observer"
LIMIT = 180
EXTRA_DATA = ("configs/forge/tasks/ring16_acquisition.json", "reports/toy_audit/catalog.json")
REQUIRED_SOURCE = frozenset((OBSERVER, "benchmarks/toy_audit/api_run.py",
    "benchmarks/toy_audit/api_contract.py", "benchmarks/toy_audit/api_vectors.py",
    "benchmarks/toy_audit/api_publish.py", "experiments/forge/policy_adapters.py",
    "experiments/forge/policy_execution.py", "experiments/forge/queue.py",
    "experiments/forge/sources.py", "experiments/forge/contracts.py",
    "particlegan/policy.py", "particlegan/recipes.py", A_PROTOCOL, *EXTRA_DATA))
SOURCE_DIRS = ("particlegan", "experiments", "benchmarks", "lib")
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}
ENV = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0", CUBLAS_WORKSPACE_CONFIG=":4096:8",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
FLAGS = dict(ordinary_current_26_slot_credit=False, new_catalog_question=False,
    historical_credit=False, default_adoption=False, speed_ranking=False)
METRIC_STEPS = [0, *(math.ceil(1000 * i / 24) for i in range(1, 25))]
MEDIA_STEPS = list(range(0, 1001, 125))
THRESHOLDS = [["sample_count", ">=", 1024], ["mean_error_sigma", "<=", .10],
    ["min_cov_eigen", ">=", .85], ["max_cov_eigen", "<=", 1.15],
    ["radial_ks", "<=", .075], ["max_projection_ks", "<=", .06]]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(value):
    path = Path(value["path"])
    if path.is_symlink() or not path.is_file() or pin(path) != value:
        raise ValueError("changed pinned Gaussian supervisor input")
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid measured Gaussian paid time")
    return value


def declared_protocol():
    return dict(schema="pg_gaussian2d_observer_supervision_v1", id=CASE,
        case_id="api-gaussian2d", cohort="api_gaussian2d_c6_gpu_v1", family=FAMILY,
        recipe_name="atlas", recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5},
        updates=1000, evaluation_samples=4096, metric_steps=METRIC_STEPS,
        media_steps=MEDIA_STEPS, terminal_observations=5, seed=24002, evaluation_seed=34002,
        physical_gpu="0", logical_device="cuda:0", inclusive_paid_seconds=LIMIT,
        export_grace_seconds=0, attempts=1, retries=0, fallback=False,
        resources=dict(cpu_threads=1, host_memory_mb=2048, memory_fraction=.2,
            minimum_free_gpu_memory_mib=12288, maximum_gpu_temperature_c=82),
        required_source_paths=sorted(REQUIRED_SOURCE),
        fixed_observer_sources={OBSERVER: OBSERVER_SHA,
            "benchmarks/toy_audit/api_run.py": API_RUN_SHA, A_PROTOCOL: A_PROTOCOL_SHA},
        question="Recover the mean, full covariance and radial/projected law of N((1,1),.04I) under the existing public selected sampler; retain original last-five gates and actual training media.",
        timing_scope="One admitted inclusive deadline covers child startup, construction, all updates/observations, original GIF/checkpoint export and source/observer attestation.",
        **FLAGS)


def validate_proposal(value):
    execution = dict(updates=1000, eval_samples=4096, protocol_seed=24002, heldout_seed=34002,
        metric_steps=METRIC_STEPS, media_steps=MEDIA_STEPS, terminal_observations=5)
    budget = dict(physical_gpu=0, logical_device="cuda:0", cpu_threads=1, cuda_memory_fraction=.2,
        inclusive_paid_seconds=180, attempts=1, export_grace_seconds=0)
    case = value.get("case", {})
    required_case = dict(id="api-gaussian2d", legacy_ids=["source-family-16"], kind="gaussian",
        default_steps=1000, batch_size=2048, eval_samples=4096, particles=20000, z_dim=2,
        thresholds=THRESHOLDS, law=dict(kind="normal2d", mean=[1., 1.], covariance=[[.04, 0.], [0., .04]]),
        profile=dict(kind="batch_distance", width=96, layers=3, scales=[.1, .25, .5, 1.], init_std=1.),
        provider="api_vectors", evaluation_observations=24)
    recipe = value.get("resolved_recipe", {})
    if (value.get("schema") != "pg_gaussian2d_additional_question_proposal_v1"
            or value.get("status") != "PROPOSED_NOT_EXECUTED" or value.get("catalog_id") != "source-family-16"
            or value.get("requested_recipe") != "atlas"
            or value.get("requested_recipe_overrides") != {"lr": .0053125, "prior_lr_mult": 1.5}
            or digest({key: case.get(key) for key in required_case}) != digest(required_case)
            or digest(value.get("execution")) != digest(execution) or digest(value.get("budget")) != digest(budget)
            or digest(value.get("qualification")) != digest(dict(new_catalog_question=False, old_cpu_credit=False,
                ordinary_current_26_slot_credit=False, speed_or_default_credit=False))
            or not isinstance(recipe, dict) or not recipe or recipe.get("total_steps", "missing") is not None
            or any(recipe.get(key) != item for key, item in dict(lr=.0053125, prior_lr_mult=1.5,
                num_particles=20000, z_dim=2, batch_size=2048).items())):
        raise ValueError("existing Gaussian proposal/gates/full recipe/resources differ")
    # Final exact public-factory equality is checked by A.run_bound_case before
    # construction; no new mechanism or numerical threshold is introduced here.
    json.dumps(value, allow_nan=False)
    return value


def validate_manifest(source):
    if (type(source.get("schema_version")) is not int or source.get("schema_version") != 1
            or not re.fullmatch(r"[a-f0-9]{40}", str(source.get("origin_commit", "")))
            or not isinstance(source.get("files"), dict) or not REQUIRED_SOURCE.issubset(source["files"])
            or digest(source["files"]) != source.get("digest")):
        raise ValueError("complete CURRENT observer/public/config source manifest required")
    for relative, value in source["files"].items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or relative != path.as_posix()
                or not re.fullmatch(r"[a-f0-9]{64}", str(value))):
            raise ValueError("invalid frozen source entry")
    if any(source["files"].get(path) != value for path, value in declared_protocol()["fixed_observer_sources"].items()):
        raise ValueError("stable observer/hook/proposal source bytes differ")
    return source


def verify_source(source):
    validate_manifest(source)
    root = Path(source["snapshot_path"]).resolve()
    expected = {key: value for key, value in source.items() if key != "snapshot_path"}
    if read(root / "forge-source.json") != expected:
        raise ValueError("frozen source header/origin changed")
    for relative, value in source["files"].items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root) or sha(path) != value:
            raise ValueError("frozen Gaussian source changed: " + relative)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts
        and path.name != "forge-source.json"}
    if actual - set(source["files"]):
        raise ValueError("unexpected executable/config source in frozen snapshot")
    return root


def select_snapshot_path(snapshot):
    snapshot = Path(snapshot).resolve()
    sys.path[:] = [str(snapshot), *(entry for entry in sys.path if Path(entry).resolve() != snapshot)]
    return snapshot


def guard_imports(source, modules=None):
    root = Path(source["snapshot_path"]).resolve()
    imported = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        path = getattr(module, "__file__", None)
        protected = name.split(".", 1)[0] in {"particlegan", "experiments", "benchmarks", "lib"}
        if path is None:
            if protected:
                relative = name.replace(".", "/")
                if (list(getattr(module, "__path__", [])) != [str(root / relative)]
                        or not any(item.startswith(relative + "/") for item in source["files"])):
                    raise ValueError("foreign/unbound repository namespace: " + name)
            continue
        if not isinstance(path, (str, os.PathLike)):
            if protected:
                raise ValueError("protected module has no real source file: " + name)
            continue
        file = Path(path)
        # Torch dynamic pseudo __file__ values are not real imported sources.
        if not file.is_file():
            if protected:
                raise ValueError("protected module has missing source file: " + name)
            continue
        file = file.resolve()
        if not file.is_relative_to(root):
            if protected:
                raise ValueError("foreign repository import: " + name)
            continue
        relative = file.relative_to(root).as_posix()
        if source["files"].get(relative) != sha(file):
            raise ValueError("unpinned imported repository file: " + name)
        imported[name] = dict(path=relative, sha256=sha(file))
        if getattr(module, "__path__", None) is not None and list(module.__path__) != [str(file.parent)]:
            raise ValueError("foreign/duplicate repository package path: " + name)
    return imported


def bootstrap_files(root):
    """Pre-import byte guard, matched exactly to A's maintained source_manifest."""
    root = Path(root).resolve()
    paths = {path for directory in SOURCE_DIRS for path in (root / directory).rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
        and not {"__pycache__", ".venv", "runs"}.intersection(path.relative_to(root).parts)}
    paths.update(path for path in (root / "configs").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "forge" not in path.relative_to(root).parts)
    paths.update((root / "examples").rglob("*.py"))
    paths.update(root / relative for relative in (*EXTRA_DATA, A_PROTOCOL))
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("non-file/foreign source before metadata import")
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(paths)}


def runtime():
    from platform import python_version, python_implementation, machine, system
    return dict(python=python_version(), implementation=python_implementation(), machine=machine(), system=system(),
        packages={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")})


def sidecar(output):
    output = Path(output).resolve()
    return output.parent / ("." + output.name + ".gaussian-supervision.json")


def spec(protocol_pin):
    return dict(id=CASE, paid_cap_seconds=LIMIT, export_grace_seconds=0, frames=9,
        representation_card=protocol_pin, resources=declared_protocol()["resources"],
        physical_attempt_limit=1, retries=0, **FLAGS)


def case_definition(proposal, source, inputs):
    return dict(id=CASE, question_id="api-gaussian2d", observer_cohort="api_gaussian2d_c6_gpu_v1",
        original_case=proposal["case"], full_recipe=proposal["resolved_recipe"],
        recipe_overrides=proposal["requested_recipe_overrides"], execution=proposal["execution"],
        source_digest=source["digest"], wrapper_sha256=inputs["wrapper"]["sha256"],
        protocol_sha256=inputs["protocol"]["sha256"], proposal_sha256=inputs["proposal"]["sha256"],
        scope_sha256=inputs["scope"]["sha256"],
        inclusive_timeout_seconds=LIMIT, export_grace_seconds=0, media_frames=9, **FLAGS)


def verify(packet):
    if set(packet.get("inputs", {})) != {"wrapper", "protocol", "proposal", "scope"}:
        raise ValueError("exact external driver/protocol/proposal/scope inputs required")
    for item in packet["inputs"].values():
        checked(item)
    if packet["inputs"]["wrapper"]["sha256"] != sha(__file__):
        raise ValueError("Gaussian wrapper bytes changed")
    if digest(read(checked(packet["inputs"]["protocol"]))) != digest(declared_protocol()):
        raise ValueError("fixed180s Gaussian supervisor protocol changed")
    proposal = validate_proposal(read(checked(packet["inputs"]["proposal"])))
    parent = validate_manifest(packet["parent_source"])
    additions = {WRAPPER: packet["inputs"]["wrapper"]["sha256"],
        PROTOCOL_RELATIVE: packet["inputs"]["protocol"]["sha256"], PROPOSAL_RELATIVE: packet["inputs"]["proposal"]["sha256"],
        SCOPE_RELATIVE: packet["inputs"]["scope"]["sha256"]}
    files = {**parent["files"], **additions}
    expected = {**parent, "files": files, "digest": digest(files), "snapshot_path": packet["source"]["snapshot_path"]}
    if packet["source"] != expected or packet["execution_source"] != expected:
        raise ValueError("Gaussian complete current-source closure changed")
    verify_source(expected)
    frozen_proposal = read(Path(expected["snapshot_path"]) / A_PROTOCOL)
    if any(digest(proposal.get(key)) != digest(frozen_proposal.get(key)) for key in proposal):
        raise ValueError("Gaussian proposal/full Recipe differs from the fixed source declaration")
    current_runtime = runtime()
    lane = {**current_runtime, "device": "cuda:0", "physical_gpu": "0", "torch_threads": 1,
            "compute": packet["compute_profile"]}
    required = dict(schema="pg_gaussian2d_supervised_packet_v1", spec=spec(packet["inputs"]["protocol"]),
        case_definitions={CASE: case_definition(proposal, expected, packet["inputs"])},
        runtime_contract=current_runtime, lane_runtime=lane,
        family_paid_budget_seconds={FAMILY: LIMIT},
        capacity_preflight=dict(kind="metadata_only_source_and_original_question", capacity_proved=False,
            learned_quality_proved=False, original_gate_credit=False), **FLAGS)
    if (any(digest(packet.get(key)) != digest(value) for key, value in required.items())
            or packet.get("spec_sha256") != digest(required["spec"])
            or packet["compute_profile"].get("backend") != "cuda"
            or packet["compute_profile"].get("threads") != 1
            or packet["compute_profile"].get("model") != "NVIDIA RTX A6000"
            or packet["compute_profile"].get("deterministic") is not True
            or packet["compute_profile"].get("tf32") is not False):
        raise ValueError("Gaussian question/runtime/resource/credit packet changed")
    return proposal


def prepare(output, checkout, commit):
    """Root calls this only after the observer's final clean source freeze."""
    output, checkout = Path(output).resolve(), Path(checkout).resolve()
    if sidecar(output).exists() or output.exists() or not re.fullmatch(r"[a-f0-9]{40}", str(commit)):
        raise ValueError("fresh Gaussian output and exact final commit required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
        raise ValueError("Gaussian requires the final CLEAN committed checkout")
    bootstrap = dict(schema_version=1, origin_commit=commit, files=bootstrap_files(checkout), snapshot_path=str(checkout))
    bootstrap["digest"] = digest(bootstrap["files"])
    validate_manifest(bootstrap)
    select_snapshot_path(checkout)
    guard_imports(bootstrap)
    observer = importlib.import_module("benchmarks.toy_audit.gaussian2d_observer")
    guard_imports(bootstrap)
    parent = observer.source_manifest(checkout, supervisor_source_paths=(A_PROTOCOL,))
    if parent != {key: value for key, value in bootstrap.items() if key != "snapshot_path"}:
        raise ValueError("actual complete current source_manifest differs from bootstrap/commit")
    # Metadata-only preparation has not called build, observe, a model or CUDA.
    torch = sys.modules.get("torch")
    if torch is None or torch.cuda.is_initialized():
        raise ValueError("source preparation initialized CUDA")
    proposal = validate_proposal(observer.declaration())
    from experiments.forge.sources import snapshot_source, compute_profile
    compute = compute_profile("cuda", "NVIDIA RTX A6000", threads=1)
    if compute.get("availability") == "unavailable":
        raise ValueError("physical CUDA runtime unavailable for declaration")
    metadata = output.parent / ("." + output.name + ".gaussian-preparation")
    if metadata.exists():
        raise ValueError("preserve prior preparation; use a fresh parent")
    metadata.mkdir(parents=True)
    proposal_path = metadata / "proposal.json"
    write(proposal_path, proposal)
    inputs = dict(wrapper=pin(__file__), protocol=pin(PROTOCOL), proposal=pin(proposal_path), scope=pin(HERE / "PROPOSAL.md"))
    files = {**parent["files"], WRAPPER: inputs["wrapper"]["sha256"],
        PROTOCOL_RELATIVE: inputs["protocol"]["sha256"], PROPOSAL_RELATIVE: inputs["proposal"]["sha256"],
        SCOPE_RELATIVE: inputs["scope"]["sha256"]}
    derived = {**parent, "files": files, "digest": digest(files)}
    with tempfile.TemporaryDirectory(prefix="gaussian-source-staging-", dir=metadata) as temporary:
        stage = Path(temporary)
        originals = {WRAPPER: Path(__file__), PROTOCOL_RELATIVE: PROTOCOL, PROPOSAL_RELATIVE: proposal_path,
            SCOPE_RELATIVE: HERE / "PROPOSAL.md"}
        for relative, value in files.items():
            data = (originals.get(relative) or checkout / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != value:
                raise ValueError("source changed during exact Gaussian snapshot capture")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        snapshot = snapshot_source(stage, output.parent / "gaussian-source" / commit, derived)
    source = {**derived, "snapshot_path": str(snapshot)}
    rt = runtime()
    declaration = spec(inputs["protocol"])
    packet = dict(schema="pg_gaussian2d_supervised_packet_v1", status="PREPARED", inputs=inputs,
        parent_source=parent, source=source, execution_source=source, spec=declaration,
        spec_sha256=digest(declaration), case_definitions={CASE: case_definition(proposal, source, inputs)},
        runtime_contract=rt, compute_profile=compute,
        lane_runtime={**rt, "device": "cuda:0", "physical_gpu": "0", "torch_threads": 1, "compute": compute},
        family_paid_budget_seconds={FAMILY: LIMIT},
        capacity_preflight=dict(kind="metadata_only_source_and_original_question", capacity_proved=False,
            learned_quality_proved=False, original_gate_credit=False), **FLAGS)
    verify(packet)
    write(sidecar(output), packet)
    return pin(sidecar(output))


def metadata_directory(output):
    output = Path(output).resolve()
    return output.parent / ("." + output.name + ".gaussian-metadata-check")


def metadata_stage(path):
    """Real copied-source bootstrap only; constructs no model or fixture."""
    resolved = read(path)
    packet = resolved["packet"]
    proposal = verify(packet)
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(key) != value for key, value in ENV.items() if key != "CUDA_VISIBLE_DEVICES")):
        raise ValueError("metadata-only bootstrap requires CUDA hidden and CPU1")
    snapshot = Path(packet["source"]["snapshot_path"]).resolve()
    if Path.cwd().resolve() != snapshot:
        raise ValueError("metadata bootstrap must use the exact copied directory")
    select_snapshot_path(snapshot)
    guard_imports(packet["source"])
    observer = importlib.import_module("benchmarks.toy_audit.gaussian2d_observer")
    torch = sys.modules["torch"]
    if torch.cuda.is_initialized():
        raise ValueError("metadata bootstrap initialized CUDA")
    before = observer.typed_state_digest(observer.global_state())
    declaration = observer.declaration()
    parent = observer.source_manifest(snapshot, supervisor_source_paths=(A_PROTOCOL,))
    derived = observer.source_manifest(snapshot,
        supervisor_source_paths=(A_PROTOCOL, WRAPPER, PROTOCOL_RELATIVE, PROPOSAL_RELATIVE, SCOPE_RELATIVE))
    # Snapshot execution has a source header, not a Git checkout. File bytes
    # carry the scientific identity; no enclosing Git origin is substituted.
    if (digest(declaration) != digest(proposal) or parent["files"] != packet["parent_source"]["files"]
            or derived["files"] != packet["source"]["files"]):
        raise ValueError("real current child closure/proposal differs from frozen packet")
    after = observer.typed_state_digest(observer.global_state())
    if before != after or torch.cuda.is_initialized():
        raise ValueError("metadata declaration/source inspection drew RNG or initialized CUDA")
    imported = guard_imports(packet["source"])
    verify(packet)
    record = dict(schema="pg_gaussian2d_copied_source_metadata_v1", status="PASS_METADATA_ONLY",
        source_digest=packet["source"]["digest"], source_commit=packet["source"]["origin_commit"],
        wrapper_sha256=sha(__file__), proposal_sha256=packet["inputs"]["proposal"]["sha256"],
        runtime=packet["runtime_contract"], imported_sources_after=imported,
        parent_source_files=len(parent["files"]), derived_source_files=len(derived["files"]),
        current_runtime_closure_subset=True, copied_external_closure_exact=True,
        model_constructions=0, training_updates=0, evaluation_draws=0, numerical_scorer_calls=0,
        cuda_initialized=False, global_rng_before_sha256=before, global_rng_after_sha256=after,
        canonical_snapshot_entries=sum(Path(item).resolve() == snapshot for item in sys.path), **FLAGS)
    write(Path(resolved["output"]) / "receipt.json", record)
    return 0


def verify_metadata(output, packet):
    path = metadata_directory(output) / "receipt.json"
    record = read(path)
    if (record.get("schema") != "pg_gaussian2d_copied_source_metadata_v1"
            or record.get("status") != "PASS_METADATA_ONLY"
            or record.get("source_digest") != packet["source"]["digest"]
            or record.get("source_commit") != packet["source"]["origin_commit"]
            or record.get("wrapper_sha256") != sha(__file__)
            or record.get("proposal_sha256") != packet["inputs"]["proposal"]["sha256"]
            or record.get("runtime") != packet["runtime_contract"]
            or record.get("parent_source_files") != len(packet["parent_source"]["files"])
            or record.get("derived_source_files") != len(packet["source"]["files"])
            or record.get("current_runtime_closure_subset") is not True
            or record.get("copied_external_closure_exact") is not True
            or any(type(record.get(key)) is not int or record[key] != 0 for key in
                ("model_constructions", "training_updates", "evaluation_draws", "numerical_scorer_calls"))
            or record.get("cuda_initialized") is not False
            or type(record.get("canonical_snapshot_entries")) is not int or record["canonical_snapshot_entries"] != 1
            or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("global_rng_before_sha256", "")))
            or record["global_rng_before_sha256"] != record.get("global_rng_after_sha256")
            or not record.get("imported_sources_after")
            or any(record.get(key) is not value for key, value in FLAGS.items())):
        raise ValueError("no exact successful copied-source metadata prerequisite")
    for item in record["imported_sources_after"].values():
        if packet["source"]["files"].get(item["path"]) != item["sha256"]:
            raise ValueError("metadata prerequisite imported source changed")
    return pin(path)


def metadata_preflight(output):
    output = Path(output).resolve()
    packet = read(sidecar(output))
    verify(packet)
    directory = metadata_directory(output)
    if directory.exists():
        return verify_metadata(output, packet)
    directory.mkdir()
    resolved = directory / "resolved.json"
    write(resolved, dict(packet=packet, output=str(directory)))
    snapshot = Path(packet["source"]["snapshot_path"])
    env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(snapshot)}
    command = [sys.executable, "-u", str(snapshot / WRAPPER), "--metadata-stage", str(resolved)]
    with (directory / "run.log").open("wb") as log:
        result = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode != 0:
        raise ValueError("copied-source metadata bootstrap failed; preserve " + str(directory / "run.log"))
    return verify_metadata(output, packet)


def readiness(query=None):
    command = ["nvidia-smi", "--id=0", "--query-gpu=index,name,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
    fields = (query or (lambda command: subprocess.check_output(command, text=True)))(command).strip().split(",")
    if len(fields) != 4:
        raise RuntimeError("physical GPU0 telemetry unavailable")
    index, model, free, temperature = (field.strip() for field in fields)
    if index != "0" or model != "NVIDIA RTX A6000" or int(free) < 12288 or int(temperature) > 82:
        raise RuntimeError("physical GPU0 unsafe/unavailable")
    return dict(physical_gpu="0", model=model, free_memory_mib=int(free), temperature_c=int(temperature))


def charge(paid, terminal):
    paid = number(paid)
    reserve = 0. if terminal is not None and terminal.get("attempt_status") == "completed" else max(0., LIMIT - paid)
    return dict(paid_wall_seconds=paid, unmeasured_interrupt_reserved_seconds=reserve, charged_seconds=paid + reserve)


def verify_lease(fd, resolved):
    if type(fd) is not int or fd < 0:
        raise ValueError("Gaussian child requires the inherited admitted descriptor")
    path = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
    if path != Path(resolved["worker"]["lease_path"]).resolve():
        raise ValueError("foreign inherited Gaussian descriptor")
    os.fstat(fd)
    request = read(path.parent / "supervisor-request.json")
    command = [sys.executable, "-u", str(Path(resolved["packet"]["source"]["snapshot_path"]) / WRAPPER),
        "--child", str(Path(resolved["output"]) / "resolved.json"), "--lease-fd", str(fd)]
    if (request.get("token") != resolved["worker"]["token"] or fd not in request.get("lease_fds", [])
            or request.get("source") != resolved["packet"]["execution_source"]
            or request.get("command") != command
            or len(request["lease_fds"]) != 2 or len(set(request["lease_fds"])) != 2
            or any(type(item) is not int or item < 0 for item in request["lease_fds"])
            or not request["deadline_monotonic"] > time.monotonic()
            or abs(request["deadline_monotonic"] - request["started_monotonic"] - LIMIT) > 1e-8):
        raise ValueError("Gaussian source/token/180s inherited deadline differs")
    return request


def validate_raw_identity(receipt, packet, proposal):
    if (digest(receipt.get("case")) != digest(proposal["case"])
            or digest(receipt.get("recipe")) != digest(proposal["resolved_recipe"])
            or digest(receipt.get("requested_recipe_overrides")) != digest(proposal["requested_recipe_overrides"])
            or type(receipt.get("seed")) is not int or receipt.get("seed") != 24002
            or receipt.get("source", {}).get("commit") != packet["source"]["origin_commit"]
            or receipt.get("historical_results_changed") is not False):
        raise ValueError("Gaussian original receipt case/recipe/source/seed differs")
    protocol = receipt.get("protocol", {})
    required = dict(updates=1000, default_updates=1000, evaluation_samples=4096, default_evaluation_samples=4096,
        evaluation_steps=METRIC_STEPS, metric_evaluation_steps=METRIC_STEPS, media_steps=MEDIA_STEPS,
        metric_observations=24, media_frames=9, terminal_observations=5, wall_cap_seconds=180)
    if digest(protocol) != digest(required):
        raise ValueError("Gaussian original full protocol/cadence differs")
    recorded_runtime = receipt.get("runtime", {})
    if (recorded_runtime.get("device") != "cuda:0" or type(recorded_runtime.get("torch_threads")) is not int
            or recorded_runtime["torch_threads"] != 1
            or recorded_runtime.get("cuda_device_model") != "NVIDIA RTX A6000"
            or recorded_runtime.get("python") != packet["runtime_contract"]["python"]
            or recorded_runtime.get("torch") != packet["runtime_contract"]["packages"]["torch"]):
        raise ValueError("Gaussian actual child runtime differs")
    bound = receipt.get("gaussian_bound_protocol", {})
    admitted = dict(status="running", device="cuda:0", physical_gpu=0, threads=1, memory_fraction=.2,
        allowance_seconds=180, grace_seconds=0, lease_verified=True, single_attempt=True)
    if (digest(bound.get("declaration")) != digest(proposal)
            or bound.get("source") != packet["execution_source"]
            or digest(bound.get("admission")) != digest(admitted)
            or bound.get("scientific_status") != receipt.get("status")
            or bound.get("final_raw_receipt_is_verdict_authority") is not True):
        raise ValueError("Gaussian declared source/admitted-resource receipt differs")
    sources = receipt["source"].get("files_sha256", {})
    if (not sources or OBSERVER not in sources
            or any(packet["source"]["files"].get(path) != value for path, value in sources.items())):
        raise ValueError("Gaussian original receipt does not bind the new observer/current source")
    observer = receipt.get("policy_observer", {})
    if (observer.get("schema") != "pg_gaussian2d_policy_observer_sidecar_v1"
            or observer.get("cohort") != "api_gaussian2d_c6_gpu_v1" or observer.get("case_id") != "api-gaussian2d"
            or observer.get("completed_updates") != receipt.get("completed_updates")
            or observer.get("observer_source_sha256") != packet["source"]["files"][OBSERVER]
            or observer.get("quality_from_owner_evidence") is not False
            or observer.get("training_or_rescoring_added") is not False):
        raise ValueError("Gaussian actual observer source/scope differs")
    return receipt


def complete_outcome(output, packet, proposal):
    path = Path(output) / "case" / "receipt.json"
    value = validate_raw_identity(read(path), packet, proposal)
    if (value.get("status") != "COMPLETE" or value.get("verdict") not in {"PASS", "FAIL"}
            or type(value.get("passed")) is not bool or value["passed"] != (value["verdict"] == "PASS")):
        raise ValueError("Gaussian original execution is not complete")
    observer = value["policy_observer"]
    if (type(value.get("completed_updates")) is not int or value.get("completed_updates") != 1000
            or value.get("default_protocol_complete") is not True or value.get("source_unchanged") is not True
            or observer.get("policy_protocol_complete") is not True or value.get("gif_frames") != 9
            or observer.get("pre_export_numerical_status") != "COMPLETE"
            or observer.get("pre_export_numerical_verdict") != value["verdict"]
            or [row.get("step") for row in value.get("observations", [])] != METRIC_STEPS
            or [row.get("completed_steps") for row in observer.get("observations", [])] != METRIC_STEPS):
        raise ValueError("Gaussian full1000/25read/nine-frame evidence missing")
    attestation = read(Path(output) / "observer-control.json")
    if (attestation.get("schema") != "pg_gaussian2d_supervised_observer_receipt_v1"
            or attestation.get("raw_receipt") != pin(path)
            or attestation.get("original_verdict") != value.get("verdict")
            or attestation.get("source") != packet["execution_source"]
            or attestation.get("wrapper_sha256") != sha(__file__)
            or not attestation.get("imported_sources_after")
            or any(attestation.get(key) is not expected for key, expected in FLAGS.items())):
        raise ValueError("Gaussian supervised final attestation differs")
    for item in attestation.get("artifacts", {}).values():
        checked(item)
    if set(attestation.get("artifacts", {})) != {"goal.gif", "observations.npz", "final-state.pt"}:
        raise ValueError("complete original Gaussian media/state/array artifacts required")
    for name, item in attestation["artifacts"].items():
        if (item != pin(Path(output) / "case" / name)
                or value["artifacts"].get(name) != {key: item[key] for key in ("sha256", "bytes")}):
            raise ValueError("Gaussian artifact borrowed or differs from original receipt")
    for module in attestation["imported_sources_after"].values():
        if packet["source"]["files"].get(module["path"]) != module["sha256"]:
            raise ValueError("Gaussian final imported source attestation differs")
    return dict(raw_receipt=pin(path), attestation=pin(Path(output) / "observer-control.json"),
        original_status="COMPLETE", original_verdict=value["verdict"], media=attestation["artifacts"]["goal.gif"])


def partial_outcome(output, packet, proposal):
    """A valid retained timeout prefix confers no original numerical grade."""
    path = Path(output) / "case" / "receipt.json"
    value = validate_raw_identity(read(path), packet, proposal)
    completed = value.get("completed_updates")
    if (value.get("status") != "INCOMPLETE" or value.get("passed") is not False
            or value.get("verdict") != "FAIL" or value.get("default_protocol_complete") is not False
            or type(completed) is not int or not 0 <= completed <= 1000):
        raise ValueError("Gaussian timeout prefix is not explicitly incomplete")
    steps = [row.get("step") for row in value.get("observations", [])]
    expected = [step for step in METRIC_STEPS if step <= completed]
    if value.get("partial_terminal_observation_added") is True and completed not in expected:
        expected.append(completed)
    observer = value["policy_observer"]
    if (steps != expected or observer.get("policy_protocol_complete") is not False
            or observer.get("pre_export_numerical_status") != "INCOMPLETE"
            or observer.get("pre_export_numerical_verdict") != "FAIL"
            or [row.get("completed_steps") for row in observer.get("observations", [])] != steps):
        raise ValueError("Gaussian timeout prefix/cursor/cadence differs")
    for name, identity in value.get("artifacts", {}).items():
        if name not in {"goal.gif", "observations.npz", "final-state.pt"}:
            raise ValueError("foreign Gaussian timeout artifact")
        actual = pin(Path(output) / "case" / name)
        if identity != {key: actual[key] for key in ("sha256", "bytes")}:
            raise ValueError("Gaussian timeout artifact differs")
    return dict(raw_receipt=pin(path), original_status="INCOMPLETE", original_verdict=None,
        completed_updates=completed, qualified=False)


def child(resolved_path, fd):
    resolved = read(resolved_path)
    packet = resolved["packet"]
    proposal = verify(packet)
    if any(os.environ.get(key) != value for key, value in ENV.items()):
        raise ValueError("one physicalGPU0/thread/determinism environment required")
    verify_lease(fd, resolved)  # Refuse before any protected/ML import.
    if resolved.get("metadata_preflight") != verify_metadata(resolved["output"], packet):
        raise ValueError("admitted Gaussian child lacks copied-source metadata prerequisite")
    snapshot = Path(packet["source"]["snapshot_path"]).resolve()
    if Path.cwd().resolve() != snapshot:
        raise ValueError("Gaussian child must use its exact admitted frozen directory")
    select_snapshot_path(snapshot)
    before = guard_imports(packet["source"])
    import torch
    torch.set_num_threads(1)
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != "NVIDIA RTX A6000"):
        raise ValueError("actual admitted visible CUDA0 model differs")
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    def source_guard():
        verify_source(packet["source"])
        return guard_imports(packet["source"])

    def admission_guard():
        verify_lease(fd, resolved)
        if torch.get_num_threads() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("actual admitted thread/GPU mapping changed")
        return dict(status="running", device="cuda:0", physical_gpu=0, threads=1, memory_fraction=.2,
            allowance_seconds=180, grace_seconds=0, lease_verified=True, single_attempt=True)

    source_guard()
    observer = importlib.import_module("benchmarks.toy_audit.gaussian2d_observer")
    source_guard()
    original = observer.run_bound_case(Path(resolved["output"]) / "case", proposal=proposal,
        frozen_source=packet["execution_source"], source_guard=source_guard, admission_guard=admission_guard)
    if original["status"] != "COMPLETE":
        return observer.result_exit_code(original)
    # Existing draw-free validator checks the ORIGINAL cadence, grades, actual
    # retained arrays and GIF. All export/validation remains inside180s.
    from benchmarks.toy_audit.api_publish import verify_run
    if verify_run(Path(resolved["output"]) / "case") != original:
        raise ValueError("Gaussian retained receipt differs from returned original evidence")
    validate_raw_identity(original, packet, proposal)
    after = source_guard()
    admission_guard()
    output = Path(resolved["output"])
    record = dict(schema="pg_gaussian2d_supervised_observer_receipt_v1", source=packet["execution_source"],
        wrapper_sha256=sha(__file__), raw_receipt=pin(output / "case" / "receipt.json"),
        original_verdict=original["verdict"], imported_sources_before=before, imported_sources_after=after,
        artifacts={name: pin(output / "case" / name) for name in ("goal.gif", "observations.npz", "final-state.pt")},
        **FLAGS)
    write(output / "observer-control.json", record)
    admission_guard()
    return observer.result_exit_code(original)


def result_for(output, packet, admission, error=None):
    path = Path(admission["lease_path"]).parent / "supervisor-terminal.json"
    terminal = read(path) if path.exists() else None
    if terminal is not None and terminal.get("token") != admission["token"]:
        raise ValueError("fenced Gaussian supervisor terminal")
    paid = 0. if terminal is None else terminal.get("paid_wall_seconds")
    result = dict(status="INCOMPLETE", original_verdict=None,
        token_sha256=hashlib.sha256(admission["token"].encode()).hexdigest(), **charge(paid, terminal), **FLAGS)
    if terminal is not None:
        result["terminal"] = pin(path)
    if terminal is not None and terminal.get("attempt_status") == "completed":
        result["status"] = "INVALID"
        try:
            proposal = read(checked(packet["inputs"]["proposal"]))
            outcome = complete_outcome(output, packet, proposal)
            expected_code = 0 if outcome["original_verdict"] == "PASS" else 1
            if terminal.get("child_returncode") != expected_code:
                raise ValueError("scientific PASS/FAIL disagrees with exact observer exit code")
            result.update(status="COMPLETE", original_verdict=outcome["original_verdict"], evidence=outcome)
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            if terminal.get("child_returncode") == 2:
                try:
                    partial = partial_outcome(output, packet, proposal)
                    result.update(status="INCOMPLETE", evidence=partial)
                except Exception:
                    pass  # Actual ERROR/BLOCKED/source/observer faults stay INVALID.
    if result["charged_seconds"] > LIMIT:
        result["status"] = "BUDGET_EXCEEDED"
    result["overrun_seconds"] = max(0., result["charged_seconds"] - LIMIT)
    if error is not None:
        result.setdefault("reason", f"{type(error).__name__}: {error}")
    raw = Path(output) / "case" / "receipt.json"
    if raw.exists():
        result["retained_original_receipt"] = pin(raw)
    return result


def run(output):
    output = Path(output).resolve()
    packet = read(sidecar(output))
    verify(packet)
    metadata = verify_metadata(output, packet)
    os.environ.update(ENV)
    select_snapshot_path(packet["execution_source"]["snapshot_path"])
    guard_imports(packet["source"])
    from experiments.forge.policy_execution import PolicyCoordinator
    guard_imports(packet["source"])
    coordinator = PolicyCoordinator(QUEUE, report_root=Path(packet["source"]["snapshot_path"]) / "reports/forge")
    if (output / "study.json").exists():
        saved = read(output / "study.json")
        fields = ("spec", "spec_sha256", "source", "execution_source", "parent_source", "case_definitions",
                  "runtime_contract", "family_paid_budget_seconds", "inputs")
        if any(saved.get(key) != packet[key] for key in fields):
            raise ValueError("retained Gaussian study identity differs; no retries")
        if saved.get("result") is not None:
            result = saved["result"]
            if "terminal" in result:
                terminal = read(checked(result["terminal"]))
                if (hashlib.sha256(terminal["token"].encode()).hexdigest() != result["token_sha256"]
                        or any(result[key] != value for key, value in charge(terminal["paid_wall_seconds"], terminal).items())):
                    raise ValueError("retained Gaussian cost differs")
            if result["status"] == "COMPLETE" and result["evidence"] != complete_outcome(output, packet, verify(packet)):
                raise ValueError("retained Gaussian evidence changed")
            return result
    try:
        telemetry = readiness()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        return dict(status="WAITING", reason=f"{type(exc).__name__}: {exc}", **FLAGS)
    key, actual = coordinator.register(packet, output, FAMILY, packet["lane_runtime"])
    if actual != output:
        raise ValueError("compatible Gaussian attempt already registered elsewhere; attach instead of retry")
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:
            return dict(status="WAITING", reason="compatible study has a live owner", **FLAGS)
        row = dict(id=CASE, timeout_seconds=LIMIT)
        attempt = coordinator.attempt_key(packet, dict(family=FAMILY,
            recipe_overrides={"lr": .0053125, "prior_lr_mult": 1.5}), row)
        with coordinator.admit(attempt, packet, row, "cuda:0") as (admission, lease):
            if admission["status"] == "busy":
                return dict(status="WAITING", reason=admission["reason"], **FLAGS)
            error = None
            if admission["status"] == "running" and lease is not None:
                try:
                    telemetry = readiness()
                    verify(packet)
                    resolved = output / "resolved.json"
                    write(resolved, dict(packet=packet, output=str(output), telemetry=telemetry,
                        metadata_preflight=metadata,
                        worker=dict(token=admission["token"], lease_path=admission["lease_path"])))
                    helper = Path(packet["source"]["snapshot_path"]) / WRAPPER
                    command = [sys.executable, "-u", str(helper), "--child", str(resolved), "--lease-fd", str(lease.fileno())]
                    coordinator.launch(command, packet, output / "run.log", (study_lease, lease), LIMIT)
                except BaseException as exc:
                    error = exc
            result = result_for(output, packet, admission, error)
            result["attempt_key"] = attempt
            if admission["status"] in {"running", "awaiting_certification"}:
                coordinator.complete(attempt, result)
            elif admission.get("charged_seconds") != result["charged_seconds"]:
                raise ValueError("shared/local durable Gaussian cost differs")
            packet.update(status=result["status"], result=result, spent_seconds=result["charged_seconds"])
            write(output / "study.json", packet)
            write(output / "cost.json", result)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--metadata-stage", type=Path)
    parser.add_argument("--child", type=Path)
    parser.add_argument("--lease-fd", type=int)
    args = parser.parse_args(argv)
    if args.metadata_stage is not None:
        return metadata_stage(args.metadata_stage)
    if args.child is not None:
        if args.lease_fd is None:
            raise ValueError("Gaussian child requires inherited admitted descriptor")
        return child(args.child, args.lease_fd)
    if args.output is None:
        raise ValueError("explicit fresh Gaussian output required")
    if args.preflight_only:
        print(json.dumps(dict(status="PASS_METADATA_ONLY", receipt=metadata_preflight(args.output), **FLAGS)))
        return 0
    if args.prepare_only:
        if args.checkout is None or args.expected_commit is None:
            raise ValueError("final clean source checkout/commit required before preparation")
        print(json.dumps(dict(status="PREPARED", packet=prepare(args.output, args.checkout, args.expected_commit), **FLAGS)))
        return 0
    result = run(args.output)
    print(json.dumps(result))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
