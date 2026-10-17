"""One root-admitted half-base word contrast; no private training loop.

Preparation and copied-source preflight are metadata-only. Actual execution
delegates exclusively to maintained Forge runtime/evaluate and public policy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
DIRECTORY = "observer-controls/word-half-base-v1"
SELF = DIRECTORY + "/run_supervised.py"
CONTRACT = "experiments/forge/word_joint_rate_policy_contracts.py"
CONTRACT_SHA = "0c725c5c4b8ea7e09554bf7022b62950bb42a3d41ea4fb49dca932428a61ca11"
COHORT = "word_joint_policy_min11_rates_v1"
FAMILY = "atlas_word_joint_min11_rates"
PROFILE = "half_base"
CANDIDATE = "word-min11-half_base-rates-v1"
PARENT = "five_word_joint_acquisition"
TASK = PARENT + "_" + COHORT
SCHEMA = "pg_word_half_base_supervision_v1"
ORIGIN = "f9f7ed9d7a06c48d4ec56999107658983d7e8efc"
OVERRIDES = dict(lr=.00265625, prior_lr_mult=1.5, d_lr_mult=1.)
LIMIT = 900
QUEUE = Path("/ml2/hypergan/ParticleGAN-single-recipe/runs/forge")
STEPS = [math.ceil(i * 20001 / 24) for i in range(1, 25)]
SELECTED = [0, 3, 6, 9, 12, 14, 17, 20, 23]
MEDIA_STEPS = [STEPS[i] for i in SELECTED]
THRESHOLDS = [["sample_count", ">=", 1024], ["quality_fraction", ">=", .95],
    ["modes", "==", 5], ["mass_tv", "<=", .1], ["reconstruction_exact", "==", 1],
    ["minimum_reconstruction_token_probability", ">=", .9]]
PARENTS = ("two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition", PARENT,
    "trajectory", "residual_student", "unipolar", "cover_leftover", "mid_scale_identity", "mode_hold",
    "vector_two_broad", "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
    "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2", "grid100", "rotated100",
    "staggered100", "ring_hold", "ring_extension")
ENV = dict(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
    PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
FLAGS = dict(qualification_input=False, ordinary_tier_credit=False, historical_credit=False,
    default_adoption=False, speed_ranking=False, cross_tuple_pooling=False)
SOURCE_DIRS = ("particlegan", "experiments", "benchmarks", "lib")
SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}
REQUIRED = {CONTRACT, "experiments/forge/word_joint_policy_contracts.py",
    "experiments/forge/word_joint_policy_adapters.py", "experiments/forge/policy_cohorts.py",
    "experiments/forge/policy_adapters.py", "experiments/forge/adapters.py", "experiments/forge/api.py",
    "experiments/forge/boundaries.py", "experiments/forge/named_policy_planning.py",
    "experiments/forge/planning.py", "experiments/forge/runtime.py", "experiments/forge/evaluate.py",
    "experiments/forge/sampling.py", "experiments/forge/views.py", "experiments/forge/rng.py",
    "experiments/forge/artifacts.py", "experiments/forge/sources.py", "experiments/forge/policy_execution.py",
    "experiments/forge/queue.py", "particlegan/policy.py", "particlegan/recipes.py",
    "benchmarks/toy_audit/api_images.py", "benchmarks/toy_audit/api_run.py", "benchmarks/toy_audit/api_contract.py",
    "configs/forge/views/discriminator_stability.json", "configs/forge/defaults.json",
    "configs/forge/tasks/ring16_acquisition.json", "reports/toy_audit/catalog.json",
    f"configs/forge/task-variants/{COHORT}/{TASK}.json",
    *(f"configs/forge/tasks/{p}.json" for p in PARENTS)}
ADVISORY = "new v1 declaration is not immutable legacy evidence; use a v2 decision_contract"
BASE_PATH = Path("/ml2/hypergan/pg-gaussian2d-observer-supervisor-20261004/run_supervised.py")
BASE_SHA = "1152522e2cc3215e0e707cf265697f596f3aa19244d905ffe0eea411b5daee97"
HISTORY_PATH = Path("/ml2/hypergan/pg-atlas-named-native-v4b-publication-20261003/publish_results.py")
HISTORY_SHA = "99970901e2793961c433164f5f8c73094316c19d2f92c77c5c8d3e5e6617187a"
PRIOR_DIR = Path("/ml2/hypergan/pg-atlas-named-native-v4b-publication-20261003/publication-final")
PRIOR_PINS = {
    "results": (PRIOR_DIR / "results.json", "0a175519b59b5180078d9669519060b79dc30d596daca3c8be1aca2dd9ce1d5f"),
    "index": (PRIOR_DIR / "input-index.json", "8c57ff6e02a69eda41b39cb10c0bdd8c3f41040504b3eda1f11534b2a6d82ce4"),
    "card": (PRIOR_DIR.parent / "final-inputs.json", "f287d51757c4e9397afb5435e7b3848877915b6326ba59d0659b32fab20c6be2")}
PRIOR_LANES = {"0": 234.82608077581972, "1": 675.4130623831879}
PRIOR_TOTAL = 910.2391431590077


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_wire_identity(actual, recorded):
    """Compare the complete canonical JSON request, including every value/type.

    Tuple/list representation is intentionally identical on the JSON wire.
    No scientific field, source identity or compiled annotation is omitted.
    """
    if digest(actual)!=digest(recorded):
        raise ValueError("actual copied planner/JSON/request differs")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(item):
    if (not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}
            or type(item["bytes"]) is not int or item["bytes"] < 0
            or not re.fullmatch("[a-f0-9]{64}", str(item["sha256"]))):
        raise ValueError("complete typed input identity required")
    path = Path(item["path"])
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or any(p.is_symlink() for p in path.parents) or pin(path) != item):
        raise ValueError("changed/foreign pinned input")
    return path


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite/nonnegative")
    return value


def close(a, b):
    if not math.isclose(number(a), number(b), rel_tol=0, abs_tol=1e-8):
        raise ValueError("cost arithmetic changed")


def delegate(name, original, expected):
    copied = HERE / name
    path = copied if copied.is_file() else original
    if sha(path) != expected:
        raise ValueError("frozen metadata delegate changed")
    key = "_word_half_" + name.replace(".", "_")
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec); sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def guards():
    # Only established canonical-path/import guards/runtime metadata are used.
    # No Gaussian declaration, observer, preparation or scientific code is called.
    return delegate("source_guards.py", BASE_PATH, BASE_SHA)


def runtime():
    return {**guards().runtime(),"render_packages":{p:importlib.metadata.version(p) for p in ("Pillow","matplotlib")}}


def protocol():
    return dict(schema=SCHEMA, id=CANDIDATE, profile=PROFILE, task_id=TASK, parent_id=PARENT,canonical_origin_commit=ORIGIN,
        cohort=COHORT, family=FAMILY, recipe_preset="atlas", recipe_overrides=OVERRIDES,
        seed=0, updates=20001, original_schedule_horizon=20000, observations=24, eval_samples=1024,
        metric_steps=STEPS, terminal_steps=STEPS[-5:], thresholds=THRESHOLDS,
        frames=9, media_indices=SELECTED, media_steps=MEDIA_STEPS,
        required_slots=26, tiers={"1":5,"2":19,"3":2}, unscheduled_slots=25,
        physical_gpu="1", logical_device="cuda:0", allowance_seconds=LIMIT, export_grace_seconds=0,
        attempts=1, retries=0, prior_inclusive_charged_seconds=PRIOR_TOTAL,
        prior_lane_charged_seconds=PRIOR_LANES, original_lane_caps={"0":7500,"1":3000}, original_total_cap=10500,
        resources=dict(cpu_threads=1, host_memory_mb=2048, memory_fraction=.2,
            minimum_free_gpu_memory_mib=12288, maximum_gpu_temperature_c=82),
        required_source_paths=sorted(REQUIRED), fixed_implementation_sources={CONTRACT:CONTRACT_SHA},
        fixed_delegates={"source_guards.py":BASE_SHA,"history_projection.py":HISTORY_SHA},
        rate_scope="global base rate half; nominal G/E/prior/critic rates change, not generator-only",
        timing_scope="construct/train/all24reads/state/CPUgrade/nine retained views/final attestation inside900s",
        **FLAGS)


def validate_protocol(value):
    if digest(value) != digest(protocol()):
        raise ValueError("exact single half-base tuple/protocol changed")
    return value


def history():
    old = delegate("history_projection.py", HISTORY_PATH, HISTORY_SHA)
    values = {}
    for name, (path, expected) in PRIOR_PINS.items():
        if sha(path) != expected:
            raise ValueError("immutable named cost cut changed")
        values[name] = read(path)
    indexed = values["index"]
    if indexed.get("file_count") != len(indexed["files"]) or len({i["path"] for i in indexed["files"]}) != len(indexed["files"]):
        raise ValueError("missing/duplicate prior consumed pins")
    for item in indexed["files"]:
        checked(item)
    card = values["card"]
    if card.get("schema") != old.INPUT_SCHEMA or card.get("qualification_input") is not False:
        raise ValueError("wrong prior immutable terminal card")
    inputs = old.Inputs(); packet = inputs.json(card["prepared"])
    old.packet_identity(packet, inputs)
    engineering = old.engineering(packet, inputs); continuation = old.continuation(packet, inputs)
    families = {f: old.project_family(inputs, packet, f, card["studies"].get(f), card["studies"]) for f in old.FAMILIES}
    old.validate_lanes(families, packet); inputs.recheck()
    costs = values["results"]["cost"]
    lanes = {}
    for gpu in ("0","1"):
        lanes[gpu] = old.DEBITS[gpu] + old.PREVIOUS_DEBITS[gpu] + sum(f["charged_seconds"] for f in families.values() if f["physical_gpu"] == gpu)
        close(lanes[gpu], PRIOR_LANES[gpu]); close(costs["lanes"][gpu]["inclusive_charged_seconds"], lanes[gpu])
    close(sum(lanes.values()), PRIOR_TOTAL); close(costs["inclusive_charged_seconds"], PRIOR_TOTAL)
    if (costs["current_reserved_seconds"] != 0 or costs["original_cap_seconds"] != 10500
            or {g:costs["lanes"][g]["cap_seconds"] for g in lanes} != {"0":7500,"1":3000}
            or values["results"].get("qualification_input") is not False):
        raise ValueError("prior caps/reserve/qualification changed")
    close(engineering["paid_seconds"], 25.32295504095964)
    close(continuation["paid_seconds"], 235.7073353389278)
    return dict(pins={k:pin(p) for k,(p,_) in PRIOR_PINS.items()}, prior_charged_seconds=PRIOR_TOTAL,
        prior_lane_charged_seconds=PRIOR_LANES, source=values["results"]["source"],
        consumed_pins=len(indexed["files"]), terminal_cost_joins_verified=True, outcomes_reused=False,
        engineering_paid_seconds=engineering["paid_seconds"], prior_v3_paid_seconds=continuation["paid_seconds"],
        prior_v4b_paid_seconds=costs["current_paid_seconds"], **FLAGS)


def validate_history(value):
    if value != history():
        raise ValueError("history cannot reset/rebind/import outcomes")
    return value


def source_paths(root):
    root = Path(root).resolve()
    paths = {p for d in SOURCE_DIRS for p in (root/d).rglob("*") if p.is_file() and p.suffix in SUFFIXES
        and not {"__pycache__",".venv","runs"}.intersection(p.relative_to(root).parts)}
    paths.update(p for p in (root/"configs").rglob("*") if p.is_file() and p.suffix in SUFFIXES)
    paths.update((root/"examples").rglob("*.py"))
    paths.update(root/p for p in REQUIRED)
    for p in paths:
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError("complete regular source/config closure required")
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}


def verify_source(source):
    root = Path(source["snapshot_path"]).resolve()
    if (source.get("schema_version") != 1 or source.get("origin_commit")!=ORIGIN
            or not REQUIRED.issubset(source.get("files",{})) or source["files"].get(CONTRACT)!=CONTRACT_SHA
            or digest(source["files"]) != source.get("digest")
            or read(root/"forge-source.json") != {k:v for k,v in source.items() if k != "snapshot_path"}):
        raise ValueError("wrong frozen source origin/header/closure")
    for rel, expected in source["files"].items():
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or p.as_posix() != rel or not re.fullmatch("[a-f0-9]{64}",str(expected)):
            raise ValueError("unsafe frozen source field")
        path = root/p
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root) or sha(path) != expected:
            raise ValueError("frozen source byte drift: " + rel)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        and p.suffix in SUFFIXES and "__pycache__" not in p.parts and p.name != "forge-source.json"}
    if actual - set(source["files"]):
        raise ValueError("unbound executable/config file in snapshot")
    return root


def forge(name):
    return importlib.import_module("experiments.forge." + name)


def declaration(root):
    rate = forge("word_joint_rate_policy_contracts")
    base = forge("planning").load_idea(root, "atlas-c6-observed-policy-current-v1")
    reference = dict(path="configs/forge/ideas/atlas-c6-observed-policy-current-v1.json",
        sha256=sha(root/"configs/forge/ideas/atlas-c6-observed-policy-current-v1.json"),
        decision_contract=base.pop("decision_contract"), execution_authorized=False)
    base.update(rate.candidate(PROFILE), schema_version=1,
        hypothesis="Test one global half-base rate tuple on the unchanged N11/free-E joint word question.",
        changed_factors=["Global base LR .00265625; prior multiplier1.5 and critic multiplier1 remain fixed"],
        ordinary_decision_reference=reference,
        requires_capabilities=["a2","named_rng","policy_controls","policy_serving"])
    return base


def validate_request(request, source=None):
    candidate = request["candidate"]
    expected = {"id":CANDIDATE,"word_rate_profile":PROFILE,"task_cohort":COHORT,"trainer_family":FAMILY,
        "recipe_preset":"atlas","recipe_overrides":OVERRIDES,"execution_path":"public_trainer"}
    if digest({k:candidate.get(k) for k in expected})!=digest(expected):
        raise ValueError("exact half-base candidate/profile/tuple/reference binding required")
    ids = [TASK if p == PARENT else p for p in PARENTS]
    view = request["view"]; assignments = view["assignments"]
    if (set(request["tasks"]) != set(ids) or [a["task"] for a in assignments] != ids
            or len(assignments) != 26 or [sum(a["qualification_tier"] == t for a in assignments) for t in (1,2,3)] != [5,19,2]
            or any(a["importance"] != "required" for a in assignments) or view.get("revision") != 4
            or view.get("policy_family") != FAMILY or view.get("task_cohort") != COHORT
            or request["protocol"].get("seed") != 0 or type(request["protocol"]["seed"]) is not int):
        raise ValueError("full26/original order/view/seed changed")
    task = request["tasks"][TASK]
    execution, evaluation = task["execution"], task["evaluation"]
    if (task.get("policy_family") != FAMILY or task.get("task_cohort") != COHORT
            or task["policy_parent"]["id"] != PARENT or task.get("preflight_blockers") != []
            or execution.get("steps") != 20001 or execution.get("original_schedule_horizon") != 20000
            or execution.get("execution_path") != "public_components" or execution.get("device") != "cuda"
            or execution.get("resources") != {"num_particles":11,"z_dim":2,"batch_size":256}
            or execution.get("prior",{}).get("kind") != "particle_cloud"
            or execution["prior"].get("sigma") != 0 or execution["prior"].get("standardize") is not False
            or evaluation.get("thresholds") != THRESHOLDS or evaluation.get("observations") != 24
            or evaluation.get("minimum_stable_checks") != 5 or evaluation.get("eval_samples") != 1024
            or evaluation.get("scoring_weights") != "state_selected" or task["resources"].get("timeout_seconds") != LIMIT):
        raise ValueError("word parent/physical resources/raw prior/gates/horizon drift")
    jobs = [j for j in request["jobs"] if TASK in j["task_ids"]]
    if len(jobs) != 1 or jobs[0]["task_ids"] != [TASK] or jobs[0]["budget_seconds"] != LIMIT:
        raise ValueError("exact one full word job required")
    if source is not None and request.get("source") != source:
        raise ValueError("request source changed")
    return jobs[0]


def build_request(root, source=None):
    root = Path(root).resolve()
    planned = forge("planning").resolve_idea(root, CANDIDATE, declaration=declaration(root),
        view_id="discriminator_stability", through_tier=3, freeze_source=False,
        execution_backend="cuda", cuda_model="NVIDIA RTX A6000")
    blockers = list(planned["preflight_blockers"])
    if blockers.count(ADVISORY) != 1:
        raise ValueError("exact inactive legacy advisory must be retained")
    blockers.remove(ADVISORY)
    if blockers:
        raise ValueError("source/API/capability blockers remain: " + "; ".join(blockers))
    result = deepcopy(planned)
    result.update(preflight_blockers=[], ordinary_admission={"status":"BLOCKED","blockers":[ADVISORY],"execution_authorized":False},
        campaign_id=CANDIDATE, evidence_scope="named_word_global_half_base_diagnostic", qualification_reuse=False)
    if source is not None:
        if any(source["files"].get(p) != h for p,h in planned["source"]["files"].items()):
            raise ValueError("maintained planner escaped complete source closure")
        result["source"] = deepcopy(source)
        result["candidate_revision"] = forge("planning").candidate_revision_for(source["digest"], result["candidate"])
        for job in result["jobs"]:
            job["science"]["candidate_revision"] = result["candidate_revision"]
        forge("planning").rekey_jobs(result["jobs"])
    validate_request(result, source)
    return result


def sidecar(output):
    p = Path(output).resolve(); return p.parent/("."+p.name+".word-half-supervision.json")


def metadata_dir(output):
    p = Path(output).resolve(); return p.parent/("."+p.name+".word-half-metadata")


def verify(packet):
    if packet.get("schema") != SCHEMA+"_packet" or packet.get("spec_sha256") != digest(packet["spec"]):
        raise ValueError("wrong supervisor packet")
    validate_protocol(read(checked(packet["inputs"]["protocol"])))
    if sha(__file__) != packet["inputs"]["wrapper"]["sha256"]:
        raise ValueError("supervisor wrapper changed")
    for item in packet["inputs"].values(): checked(item)
    parent = packet["parent_source"]
    additions = {SELF:packet["inputs"]["wrapper"]["sha256"], DIRECTORY+"/protocol.json":packet["inputs"]["protocol"]["sha256"],
        DIRECTORY+"/source_guards.py":BASE_SHA, DIRECTORY+"/history_projection.py":HISTORY_SHA}
    files = {**parent["files"], **additions}
    expected = {**parent,"files":files,"digest":digest(files),"snapshot_path":packet["source"]["snapshot_path"]}
    if packet["source"] != expected or packet["execution_source"] != expected:
        raise ValueError("complete parent/derived source closure changed")
    root = verify_source(expected)
    if packet["history"] != history(): raise ValueError("history changed")
    expected_spec = dict(id=CANDIDATE, profile=PROFILE, recipe_overrides=OVERRIDES, paid_cap_seconds=LIMIT,
        export_grace_seconds=0, frames=9, representation_card=packet["inputs"]["protocol"],
        resources=protocol()["resources"], history_sha256=digest(packet["history"]), **FLAGS)
    if packet["spec"] != expected_spec or any(packet.get(k) is not v for k,v in FLAGS.items()):
        raise ValueError("fixed candidate/cap/nonqualification packet changed")
    validate_request(packet["request"], expected)
    # Canonical parent bytes for every one of26 slots, never a reduced roster.
    canonical_tasks={}
    for name in PARENTS:
        actual = read(root/f"configs/forge/tasks/{name}.json")
        wire = packet["request"]["tasks"][TASK if name == PARENT else name]
        if (not isinstance(wire.get("preflight_blockers"),list)
                or any(type(v) is not str for v in wire["preflight_blockers"])
                or ("field_ownership" in wire and not isinstance(wire["field_ownership"],dict))):
            raise ValueError("malformed inert compiler annotations")
        declaration_only={k:v for k,v in wire.items() if k not in {"preflight_blockers","field_ownership"}}
        if name == PARENT:
            variant=read(root/f"configs/forge/task-variants/{COHORT}/{TASK}.json")
            if (digest(declaration_only)!=digest(variant) or wire["policy_parent"]["task_sha256"] != sha(root/f"configs/forge/tasks/{name}.json")):
                raise ValueError("word parent bytes changed")
        else:
            if digest(declaration_only)!=digest(actual): raise ValueError("unexecuted parent definition changed")
        canonical_tasks[TASK if name==PARENT else name]=declaration_only
    view=packet["request"]["view"]
    if (view.get("parent_view_fingerprint")!=digest(read(root/"configs/forge/views/discriminator_stability.json"))
            or view.get("cohort_fingerprint")!=digest(canonical_tasks)
            or packet["case_definitions"]!={TASK:dict(task=deepcopy(packet["request"]["tasks"][TASK]),view=deepcopy(view),
                full26_sha256=digest(packet["request"]["tasks"]),profile=PROFILE,recipe_overrides=OVERRIDES,seed=0,
                metric_steps=STEPS,media_steps=MEDIA_STEPS)}):
        raise ValueError("complete view/case/request identities changed")
    if packet["runtime_contract"] != runtime(): raise ValueError("Python/package runtime changed")
    return root


def prepare(output, checkout, expected_commit):
    output, checkout = Path(output).resolve(), Path(checkout).resolve()
    if output.exists() or sidecar(output).exists(): raise ValueError("fresh preparation parent required")
    commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=checkout,text=True).strip()
    if commit != expected_commit or commit!=ORIGIN or subprocess.check_output(["git","status","--porcelain"],cwd=checkout,text=True):
        raise ValueError("root must supply clean exact source commit")
    parent = dict(schema_version=1,origin_commit=commit,files=source_paths(checkout))
    parent["digest"] = digest(parent["files"])
    guards().select_snapshot_path(checkout)
    guards().guard_imports({**parent,"snapshot_path":str(checkout)})
    base = build_request(checkout); old = history()
    for relative, expected in base["source"]["files"].items():
        path=checkout/relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(checkout) or sha(path)!=expected:
            raise ValueError("planner source support changed before capture")
        parent["files"][relative]=expected
    parent["digest"]=digest(parent["files"])
    inputs = dict(wrapper=pin(__file__),protocol=pin(PROTOCOL), source_guards=pin(BASE_PATH),history_projection=pin(HISTORY_PATH))
    validate_protocol(read(PROTOCOL))
    additions = {SELF:Path(__file__),DIRECTORY+"/protocol.json":PROTOCOL,
        DIRECTORY+"/source_guards.py":BASE_PATH,DIRECTORY+"/history_projection.py":HISTORY_PATH}
    files = {**parent["files"],**{p:sha(f) for p,f in additions.items()}}
    derived = {**parent,"files":files,"digest":digest(files)}
    with tempfile.TemporaryDirectory(prefix="word-half-source-",dir=output.parent) as tmp:
        stage = Path(tmp)
        for p,h in files.items():
            original = additions.get(p) or checkout/p; data=original.read_bytes()
            if hashlib.sha256(data).hexdigest() != h: raise ValueError("source changed during capture")
            target=stage/p; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        snapshot = forge("sources").snapshot_source(stage,output.parent/"word-half-source"/commit,derived)
    source = {**derived,"snapshot_path":str(snapshot)}
    request = build_request(checkout,source)
    rt = runtime(); compute = request["compute_profiles"]["cuda"]
    if compute.get("availability") == "unavailable": raise ValueError("runtime CUDA metadata unavailable")
    spec = dict(id=CANDIDATE,profile=PROFILE,recipe_overrides=OVERRIDES,paid_cap_seconds=LIMIT,export_grace_seconds=0,
        frames=9,representation_card=inputs["protocol"],resources=protocol()["resources"],history_sha256=digest(old),**FLAGS)
    packet = dict(schema=SCHEMA+"_packet",status="PREPARED",spec=spec,spec_sha256=digest(spec),inputs=inputs,
        parent_source=parent,source=source,execution_source=source,history=old,request=request,runtime_contract=rt,
        lane_runtime={**rt,"device":"cuda:0","physical_gpu":"1","torch_threads":1,"compute":compute},
        family_paid_budget_seconds={FAMILY:LIMIT},capacity_preflight=dict(kind="structural_metadata_only",capacity_proved=False,learned_quality_proved=False),
        case_definitions={TASK:dict(task=deepcopy(request["tasks"][TASK]),view=deepcopy(request["view"]),
            full26_sha256=digest(request["tasks"]), profile=PROFILE, recipe_overrides=OVERRIDES, seed=0,
            metric_steps=STEPS,media_steps=MEDIA_STEPS)},**FLAGS)
    verify(packet); write(sidecar(output),packet); return pin(sidecar(output))


def actual_metadata(packet):
    root = verify(packet); guards().select_snapshot_path(root); guards().guard_imports(packet["source"])
    import torch
    rate = forge("word_joint_rate_policy_contracts")
    producer = forge("word_joint_policy_adapters")
    # Import all real dispatch/evaluator paths without constructing their owners.
    for module in ("api","adapters","sampling","views","runtime","evaluate"):
        forge(module)
    if torch.cuda.is_initialized(): raise ValueError("metadata initialized CUDA")
    before = producer.typed_state_digest(producer.word_global_rng())
    def forbidden(*args,**kwargs):
        raise ValueError("metadata must not construct/forward/draw/score/execute")
    with ExitStack() as stack:
        for owner,name in ((torch.nn.Module,"__init__"),(torch.nn.Module,"__call__"),
                (producer,"run_word"),(forge("runtime"),"execute"),(forge("evaluate"),"evaluate"),
                (forge("views"),"grade_result")):
            stack.enter_context(patch.object(owner,name,forbidden))
        for name in ("rand","randn","randint","randperm","multinomial","normal","bernoulli"):
            stack.enter_context(patch.object(torch,name,forbidden))
        request = build_request(root,packet["source"])
        require_wire_identity(request,packet["request"])
        task = request["tasks"][TASK]
        declaration_only = forge("policy_cohorts").policy_task_declaration(task)
        rate.validate_task(declaration_only,root=root)
        recipe = rate.validate_request(request,declaration_only,root=root)
        context = forge("api").task_formulation_context(request["candidate"],task,request["protocol"],device="cpu",root=root)
        if context.recipe.to_dict() != recipe.to_dict(): raise ValueError("actual public context Recipe differs")
        binding = rate.binding_receipt(recipe,PROFILE)
    after = producer.typed_state_digest(producer.word_global_rng())
    if before != after or torch.cuda.is_initialized(): raise ValueError("metadata drew RNG/initialized CUDA")
    imported = guards().guard_imports(packet["source"])
    return dict(word_rate_binding=binding, request_sha256=digest(request), full26_task_sha256=digest(request["tasks"]),
        global_rng_before_sha256=before,global_rng_after_sha256=after,imported_sources=imported,
        canonical_snapshot_entries=sum(Path(p).resolve()==root for p in sys.path), cuda_initialized=False,
        model_constructions=0,forwards=0,draws=0,updates=0,scorer_calls=0)


def metadata_stage(path):
    value=read(path); packet=value["packet"]
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k)!=v for k,v in ENV.items() if k!="CUDA_VISIBLE_DEVICES")):
        raise ValueError("metadata requires hidden CUDA/CPU1")
    root=verify(packet)
    if Path.cwd().resolve()!=root: raise ValueError("metadata must use exact copied source cwd")
    record=actual_metadata(packet)
    record.update(schema=SCHEMA+"_metadata",status="PASS_METADATA_ONLY",source=packet["source"],wrapper_sha256=sha(__file__),**FLAGS)
    write(Path(value["output"])/"receipt.json",record); return 0


def verify_metadata(output,packet):
    path=metadata_dir(output)/"receipt.json"; value=read(path)
    if (value.get("schema")!=SCHEMA+"_metadata" or value.get("status")!="PASS_METADATA_ONLY"
            or value.get("source")!=packet["source"] or value.get("wrapper_sha256")!=sha(__file__)
            or value.get("request_sha256")!=digest(packet["request"]) or value.get("full26_task_sha256")!=digest(packet["request"]["tasks"])
            or value.get("canonical_snapshot_entries")!=1 or value.get("cuda_initialized") is not False
            or any(type(value.get(k)) is not int or value[k]!=0 for k in ("model_constructions","forwards","draws","updates","scorer_calls"))
            or not re.fullmatch("[a-f0-9]{64}",str(value.get("global_rng_before_sha256")))
            or value["global_rng_before_sha256"]!=value.get("global_rng_after_sha256") or not value.get("imported_sources")
            or any(value.get(k) is not v for k,v in FLAGS.items())):
        raise ValueError("successful exact copied-source preflight required")
    binding=value["word_rate_binding"]
    if binding.get("profile")!=PROFILE or binding.get("tuple_id")!=CANDIDATE or binding.get("overrides")!=OVERRIDES or binding.get("resolved_recipe_sha256")!=digest(binding["resolved_recipe"]):
        raise ValueError("metadata complete half-base Recipe binding differs")
    for item in value["imported_sources"].values():
        if packet["source"]["files"].get(item["path"])!=item["sha256"]: raise ValueError("metadata imports changed")
    return pin(path)


def preflight(output):
    output=Path(output).resolve(); packet=read(sidecar(output)); verify(packet)
    if output.exists(): raise ValueError("preflight must precede canonical output/admission")
    directory=metadata_dir(output)
    if directory.exists(): return verify_metadata(output,packet)
    directory.mkdir(); resolved=directory/"resolved.json"; write(resolved,{"packet":packet,"output":str(directory)})
    root=Path(packet["source"]["snapshot_path"])
    env={**os.environ,**ENV,"CUDA_VISIBLE_DEVICES":"","PYTHONPATH":str(root)}
    with (directory/"run.log").open("wb") as log:
        result=subprocess.run([sys.executable,"-u",str(root/SELF),"--metadata-stage",str(resolved)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    if result.returncode!=0: raise ValueError("real copied-source metadata failed; preserve its log")
    return verify_metadata(output,packet)


def readiness(query=None):
    run=query or subprocess.check_output
    row=run(["nvidia-smi","--id=1","--query-gpu=memory.free,temperature.gpu,name","--format=csv,noheader,nounits"],text=True).strip().split(",")
    if len(row)!=3: raise ValueError("one numeric physical GPU1 telemetry row required")
    free,temp,name=int(row[0]),int(row[1]),row[2].strip()
    if free<12288 or temp>82 or name!="NVIDIA RTX A6000": raise ValueError("GPU1 resources not ready")
    return dict(physical_gpu="1",free_memory_mib=free,temperature_c=temp,model=name)


def allowance_fits(prior, paid=0., reserve=0.):
    if prior.get("prior_charged_seconds")!=PRIOR_TOTAL or prior.get("prior_lane_charged_seconds")!=PRIOR_LANES:
        raise ValueError("original cost debit cannot reset")
    charged=number(paid)+number(reserve)
    return (PRIOR_LANES["1"]+charged+LIMIT<=3000 and PRIOR_TOTAL+charged+LIMIT<=10500)


def charge(paid, terminal):
    paid=number(paid)
    # Match maintained _released_attempt: only a committed completed terminal
    # closes the full allowance, regardless of its scientific/child exit code.
    completed=terminal is not None and terminal.get("attempt_status")=="completed"
    reserve=0. if completed else max(0.,LIMIT-paid)
    return dict(paid_wall_seconds=paid,unmeasured_interrupt_reserved_seconds=reserve,charged_seconds=paid+reserve)


def verify_lease(fd,resolved):
    if type(fd) is not int or fd<0: raise ValueError("inherited attempt FD required")
    worker=resolved["worker"]; path=Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
    if path!=Path(worker["lease_path"]).resolve(): raise ValueError("foreign inherited descriptor")
    os.fstat(fd); request=read(path.parent/"supervisor-request.json")
    expected=[sys.executable,"-u",str(Path(resolved["packet"]["source"]["snapshot_path"])/SELF),"--child",str(resolved["resolved_path"]),"--lease-fd",str(fd)]
    if (request.get("source")!=resolved["packet"]["source"] or request.get("token")!=worker["token"]
            or request.get("command")!=expected or fd not in request.get("lease_fds",[])
            or len(request["lease_fds"])!=2 or len(set(request["lease_fds"]))!=2
            or not request["deadline_monotonic"]>time.monotonic()
            or abs(request["deadline_monotonic"]-request["started_monotonic"]-LIMIT)>1e-8):
        raise ValueError("source/token/single900s/study+attempt lease differs")
    return request


def word_views(task,arrays):
    import numpy as np
    definition=task["execution"]["host_definition"]; chars=definition["characters"]; length=definition["length"]
    def decoded(value):
        rows=np.asarray(value).reshape(-1,len(chars),length)
        return ["".join(chars[int(i)] for i in row) for row in rows.argmax(1)]
    target=arrays["target"]; labels=decoded(target)
    context="Actual selected G/E/prior; DV12 retained, output noise off. Argmax text is display only."
    a=dict(kind="text",title="Five target words and eight actual generated rows",target=target,samples=arrays["generated"],
        target_labels=labels,sample_labels=decoded(arrays["generated"])[:8],caption=context+" All1024 draws determine quality/mass.")
    b=dict(kind="text",title="Every known word and paired free-E reconstruction",target=target,samples=arrays["reconstruction"],
        target_labels=labels,sample_labels=decoded(arrays["reconstruction"]),caption=context+" No reconstruction training loss.")
    c=dict(kind="image",title="Paired token confidence and padding",target=np.asarray(target).reshape(-1,1,len(chars),length),
        samples=np.asarray(arrays["reconstruction"]).reshape(-1,1,len(chars),length),vmin=0.,vmax=1.,
        caption="Original28 character rows and six token positions; actual probabilities, fixed[0,1].")
    return [a,b,c]


def point_pass(row):
    checks=[]
    for key,op,bound in THRESHOLDS:
        value=row.get(key)
        if type(value) not in (int,float) or not math.isfinite(value): raise ValueError("finite original metric required")
        checks.append({">=":value>=bound,"<=":value<=bound,"==":value==bound}[op])
    return all(checks)


def recorded_evidence(directory,packet,metadata):
    """Join saved JSON/file identities without models, restoration or sampling."""
    directory=Path(directory); raw=read(directory/"raw-result.json"); grading=read(directory/"graded-result.json")
    if (grading.get("raw_hash")!=digest(raw) or grading.get("source_digest")!=packet["source"]["digest"]
            or set(grading.get("grades",{}))!={TASK} or grading["grades"][TASK].get("gate_status") not in {"PASS","FAIL"}
            or raw.get("task_id")!=TASK or raw.get("device")!="cuda:0" or raw.get("execution_path")!="public_components"):
        raise ValueError("original raw/grade/source/device join failed")
    evidence=raw["evidence"]; recipe=metadata["word_rate_binding"]["resolved_recipe"]
    applied=raw["applied"]; lifecycle=applied.get("policy_lifecycle",{})
    binding=metadata["word_rate_binding"]
    if (digest(applied.get("recipe"))!=digest(recipe) or applied.get("family")!=FAMILY
            or applied.get("task_cohort")!=COHORT or applied.get("execution_path")!="public_components"
            or digest(applied.get("actual_resources"))!=digest(dict(num_particles=11,z_dim=2,batch_size=256))
            or lifecycle.get("owner")!="particlegan.UpdatePolicy"
            or type(lifecycle.get("completed_steps")) is not int or lifecycle["completed_steps"]!=20001
            or type(lifecycle.get("external_max_steps")) is not int or lifecycle["external_max_steps"]!=20001
            or digest(lifecycle.get("controls",{}).get("word_rate_binding"))!=digest(binding)
            or digest(evidence.get("policy_controls",{}).get("word_rate_binding"))!=digest(binding)
            or type(raw["cost"].get("completed_steps")) is not int or raw["cost"]["completed_steps"]!=20001
            or [p["step"] for p in evidence["observations"]]!=STEPS or evidence.get("live")!=evidence["observations"][-1]
            or evidence.get("scoring_weights")!="state_selected" or evidence["guards"].get("all_finite") is not True
            or digest(evidence["guards"].get("optimizer_updates"))!=digest({r:20001 for r in ("generator","encoder","prior","discriminator")})
            or len(evidence.get("policy_observations",[]))!=24 or len(evidence.get("policy_purity",[]))!=24
            or any(type(p.get("step")) is not int for p in evidence["observations"])
            or [p.get("completed_steps") for p in evidence["policy_observations"]]!=STEPS
            or [p.get("completed_steps") for p in evidence["policy_purity"]]!=STEPS
            or any(type(p.get("completed_steps")) is not int for p in evidence["policy_observations"]+evidence["policy_purity"])
            or any(p.get("family")!=FAMILY for p in evidence["policy_observations"])):
        raise ValueError("full Recipe/profile/owners/original24 reads changed")
    for point in evidence["observations"]:
        point_pass(point)
    for audit in evidence["policy_purity"]:
        if (audit.get("pure") is not True or audit.get("before_sha256")!=audit.get("after_sha256")
                or audit.get("global_rng_before_sha256")!=audit.get("global_rng_after_sha256")
                or any(not re.fullmatch("[a-f0-9]{64}",str(audit.get(k))) for k in ("before_sha256","global_rng_before_sha256"))):
            raise ValueError("all original observations require state/global RNG purity")
    expected="PASS" if all(point_pass(p) for p in evidence["observations"][-5:]) else "FAIL"
    if grading["grades"][TASK]["gate_status"]!=expected: raise ValueError("original final-five gate differs")
    root=Path(evidence["artifact_root"])
    if (root.is_symlink() or not root.is_absolute() or not root.resolve().is_relative_to(directory.resolve())
            or any(p.is_symlink() for p in root.parents)):
        raise ValueError("foreign retained artifacts")
    manifest=evidence["artifact_manifest"]; files=manifest["files"]
    expected_files={"state.pt",*(f"observations/step_{step:06d}.npz" for step in STEPS)}
    if (set(files)!=expected_files or manifest.get("schema_version")!=1 or manifest.get("file_count")!=25
            or manifest.get("sha256")!=digest(files) or manifest.get("total_bytes")!=sum(v["size"] for v in files.values())):
        raise ValueError("complete original checkpoint/all24 arrays manifest required")
    for relative,item in files.items():
        checked(dict(path=str(root/relative),sha256=item["sha256"],bytes=item["size"]))
    checkpoint=evidence["checkpoint"]
    if (checkpoint.get("path")!="state.pt" or checkpoint.get("sha256")!=files["state.pt"]["sha256"]
            or checkpoint.get("digest_kind")!="typed_policy_state_v1"
            or not re.fullmatch("[a-f0-9]{64}",str(checkpoint.get("state_sha256")))):
        raise ValueError("complete typed checkpoint/manifest join required")
    return raw,grading,root


def retained_evidence(directory,packet,metadata):
    raw,grading,root=recorded_evidence(directory,packet,metadata)
    evidence=raw["evidence"]
    task=packet["request"]["tasks"][TASK]
    # Maintained policy guards check owner clocks/purity/artifacts without draws.
    failure=forge("views")._policy_guards(task,evidence)
    if failure is not None: raise ValueError("policy/source/state/purity evidence is not accepted: "+str(failure))
    forge("artifacts").verify_artifacts(root,evidence["artifact_manifest"])
    return raw,grading,root


def render_media(directory,packet,metadata):
    import numpy as np
    from benchmarks.toy_audit.api_run import render_gif
    raw,grading,root=retained_evidence(directory,packet,metadata)
    records=[]; paths=[]; task=packet["request"]["tasks"][TASK]
    for i in SELECTED:
        point=raw["evidence"]["observations"][i]; path=root/"observations"/f"step_{point['step']:06d}.npz"; paths.append(pin(path))
        with np.load(path,allow_pickle=False) as source:
            arrays={k:source[k].copy() for k in source.files}
        if (not {"target","generated","reconstruction"}.issubset(arrays)
                or any(not np.isfinite(v).all() for v in arrays.values())
                or arrays["generated"].shape[0]!=1024 or arrays["target"].shape[0]!=5 or arrays["reconstruction"].shape[0]!=5):
            raise ValueError("full original word media arrays required")
        records.append(dict(step=point["step"],metrics={k:v for k,v in point.items() if k!="step"},passed=point_pass(point),views=word_views(task,arrays)))
    status=grading["grades"][TASK]["gate_status"]; gif=Path(directory)/"goal.gif"
    render_gif(dict(id=CANDIDATE,goal="Acquire five words and paired free-E reconstruction; one global half-base tuple, selected DV12/no output noise; original final-five gate",default_steps=20001),
        records,gif,full_budget=True,requested_steps=20001,final_verdict=status)
    from PIL import Image
    with Image.open(gif) as image:
        if image.n_frames!=9: raise ValueError("nine actual retained frames required")
    record=dict(schema=SCHEMA+"_media",source_digest=packet["source"]["digest"],candidate_id=CANDIDATE,profile=PROFILE,
        original_gate=status,actual_steps=MEDIA_STEPS,selected_indices=SELECTED,inputs=paths,gif=pin(gif),frames=9,
        renderer_sha256=packet["source"]["files"]["benchmarks/toy_audit/api_run.py"],
        wrapper_sha256=packet["source"]["files"][SELF],numerical_observations_changed=False,draws=0,updates=0,**FLAGS)
    write(Path(directory)/"media.json",record); return record


def stage(path,execute,fd):
    resolved=read(path); packet=resolved["packet"]; root=verify(packet)
    if Path.cwd().resolve()!=root: raise ValueError("stage must execute exact snapshot")
    if any(os.environ.get(k)!=v for k,v in ENV.items() if k!="CUDA_VISIBLE_DEVICES"): raise ValueError("CPU1/determinism environment changed")
    guards().select_snapshot_path(root); guards().guard_imports(packet["source"]); verify_lease(fd,resolved)
    if execute:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise ValueError("physicalGPU1 only")
        import torch
        torch.set_num_threads(1)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="NVIDIA RTX A6000": raise ValueError("visible device/model changed")
        torch.cuda.set_per_process_memory_fraction(.2,0)
        code=forge("runtime").execute(Path(path))
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise ValueError("grade/export requires hidden CUDA")
        forge("evaluate").evaluate(Path(path))
        metadata=read(checked(resolved["metadata_preflight"]))
        render_media(Path(path).parent,packet,metadata); code=0
    verify_source(packet["source"]); imports=guards().guard_imports(packet["source"]); verify_lease(fd,resolved)
    write(Path(path).parent/("execution-control.json" if execute else "evaluation-control.json"),
        dict(source_digest=packet["source"]["digest"],imports=imports,code=code,**FLAGS))
    return code


def child(path,fd):
    resolved=read(path); packet=resolved["packet"]; root=verify(packet)
    if resolved.get("request")!=packet["request"] or resolved.get("job")!=validate_request(packet["request"],packet["source"]):
        raise ValueError("admitted wire task/job differs")
    if resolved.get("metadata_preflight")!=verify_metadata(resolved["study_output"],packet): raise ValueError("metadata prerequisite changed")
    if any(os.environ.get(k)!=v for k,v in ENV.items()): raise ValueError("child environment changed")
    guards().select_snapshot_path(root); verify_lease(fd,resolved)
    command=[sys.executable,"-u",str(root/SELF)]; env={**os.environ,"CUDA_VISIBLE_DEVICES":""}
    for flag,environment in (("--execute",None),("--evaluate",env)):
        result=subprocess.run(command+[flag,str(path),"--lease-fd",str(fd)],env=environment,pass_fds=(fd,),check=False)
        if result.returncode!=0: return result.returncode
    metadata=read(checked(resolved["metadata_preflight"])); directory=Path(path).parent
    retained_evidence(directory,packet,metadata); verify_source(packet["source"]); imports=guards().guard_imports(packet["source"]); verify_lease(fd,resolved)
    stages=stage_proof(directory,packet)
    write(directory/"word-control.json",dict(schema=SCHEMA+"_control",source=packet["source"],candidate_id=CANDIDATE,
        profile=PROFILE,word_rate_binding=metadata["word_rate_binding"],raw=pin(directory/"raw-result.json"),grading=pin(directory/"graded-result.json"),
        media=pin(directory/"media.json"),gif=pin(directory/"goal.gif"),imports=imports,stages=stages,**FLAGS))
    verify_lease(fd,resolved); return 0


def stage_proof(directory,packet):
    joined={}; paths=set()
    for name in ("execution","evaluation"):
        path=Path(directory)/(name+"-control.json"); value=read(path)
        if (value.get("source_digest")!=packet["source"]["digest"] or type(value.get("code")) is not int or value["code"]!=0
                or not value.get("imports") or any(value.get(k) is not v for k,v in FLAGS.items())):
            raise ValueError("actual execution/evaluator source stage proof missing")
        for item in value["imports"].values():
            if packet["source"]["files"].get(item["path"])!=item["sha256"]:
                raise ValueError("stage imported wrong source")
            paths.add(item["path"])
        joined[name]=pin(path)
    required={CONTRACT,"experiments/forge/word_joint_policy_adapters.py","experiments/forge/runtime.py",
        "experiments/forge/evaluate.py","experiments/forge/api.py","experiments/forge/views.py",
        "experiments/forge/sampling.py","particlegan/policy.py","particlegan/recipes.py","benchmarks/toy_audit/api_run.py"}
    if not required.issubset(paths): raise ValueError("complete real producer/evaluator imports absent")
    return joined


def outcome(directory,packet,metadata):
    directory=Path(directory); control=read(directory/"word-control.json")
    required=dict(schema=SCHEMA+"_control",source=packet["source"],candidate_id=CANDIDATE,profile=PROFILE,
        word_rate_binding=metadata["word_rate_binding"],raw=pin(directory/"raw-result.json"),grading=pin(directory/"graded-result.json"),
        media=pin(directory/"media.json"),gif=pin(directory/"goal.gif"),stages=stage_proof(directory,packet),**FLAGS)
    if (any(digest(control.get(k))!=digest(v) for k,v in required.items()) or not control.get("imports")
            or any(control.get(k) is not v for k,v in FLAGS.items())):
        raise ValueError("final source/export attestation missing/changed")
    for item in control["imports"].values():
        if packet["source"]["files"].get(item["path"])!=item["sha256"]: raise ValueError("final imported source changed")
    raw,grading,artifact_root=recorded_evidence(directory,packet,metadata)
    media=read(directory/"media.json")
    expected_media=dict(schema=SCHEMA+"_media",source_digest=packet["source"]["digest"],candidate_id=CANDIDATE,profile=PROFILE,
        original_gate=grading["grades"][TASK]["gate_status"],actual_steps=MEDIA_STEPS,selected_indices=SELECTED,
        inputs=[pin(artifact_root/"observations"/f"step_{step:06d}.npz") for step in MEDIA_STEPS],
        gif=pin(directory/"goal.gif"),frames=9,renderer_sha256=packet["source"]["files"]["benchmarks/toy_audit/api_run.py"],
        wrapper_sha256=packet["source"]["files"][SELF],numerical_observations_changed=False,draws=0,updates=0,**FLAGS)
    if digest(media)!=digest(expected_media): raise ValueError("final raw/grade/GIF join differs")
    from PIL import Image
    with Image.open(directory/"goal.gif") as image:
        if image.n_frames!=9: raise ValueError("actual decoded nine-frame GIF required")
    return dict(original_gate=grading["grades"][TASK]["gate_status"],control=pin(directory/"word-control.json"),
        raw=required["raw"],grading=required["grading"],media=required["media"],gif=required["gif"],**FLAGS)


def result_for(directory,packet,admission,metadata,error=None):
    terminal_path=Path(admission["lease_path"]).parent/"supervisor-terminal.json"
    terminal=read(terminal_path) if terminal_path.exists() else None
    if terminal is not None and terminal.get("token")!=admission["token"]: raise ValueError("fenced durable terminal")
    paid=terminal["paid_wall_seconds"] if terminal is not None else getattr(error,"paid_wall_seconds",0.)
    result=dict(status="INCOMPLETE",original_gate="UNAVAILABLE",token_sha256=hashlib.sha256(admission["token"].encode()).hexdigest(),**charge(paid,terminal),**FLAGS)
    if terminal is not None: result["terminal"]=pin(terminal_path)
    elif error is not None:
        path=Path(directory)/"launch-error.json"
        write(path,dict(token_sha256=result["token_sha256"],source_digest=packet["source"]["digest"],
            paid_wall_seconds=paid,error=f"{type(error).__name__}: {error}"))
        result["launch_error"]=pin(path)
    if terminal is not None and terminal.get("attempt_status")=="completed":
        result["status"]="INVALID"
        try:
            if terminal.get("child_returncode")!=0: raise ValueError("scientific source/model/export stage failed")
            result["outcome"]=outcome(directory,packet,metadata); result.update(status="COMPLETE",original_gate=result["outcome"]["original_gate"])
        except Exception as exc: result["reason"]=f"{type(exc).__name__}: {exc}"
    elif terminal is not None and terminal.get("attempt_status")!="timeout": result["status"]="INVALID"
    if result["charged_seconds"]>LIMIT:
        result["status"]="BUDGET_EXCEEDED"
        if "outcome" in result: result["retained_unaccepted_outcome"]=result.pop("outcome")
        result["original_gate"]="UNAVAILABLE"
    result["overrun_seconds"]=max(0.,result["charged_seconds"]-LIMIT)
    if error is not None: result.setdefault("reason",f"{type(error).__name__}: {error}")
    return result


def verify_retained_result(directory,packet,result,metadata):
    if result.get("status") not in {"COMPLETE","INVALID","INCOMPLETE","BUDGET_EXCEEDED"}:
        raise ValueError("unknown retained terminal status")
    if any(result.get(k) is not v for k,v in FLAGS.items()): raise ValueError("retained scope cannot gain credit")
    terminal=None; paid=0.
    if "terminal" in result:
        path=checked(result["terminal"]); terminal=read(path)
        supervisor=read(path.with_name("supervisor-request.json"))
        command=supervisor.get("command",[])
        if (not re.fullmatch("[a-f0-9]{64}",str(result.get("attempt_key"))) or path.parent.name!=result["attempt_key"]
                or path!=QUEUE/"policy/attempts"/result["attempt_key"]/"supervisor-terminal.json"
                or supervisor.get("source")!=packet["source"] or supervisor.get("token")!=terminal.get("token")
                or hashlib.sha256(terminal["token"].encode()).hexdigest()!=result.get("token_sha256")
                or len(command)!=7 or command[0]!=sys.executable
                or command[1:4]!=["-u",str(Path(packet["source"]["snapshot_path"])/SELF),"--child"]
                or command[4]!=str(Path(directory)/"resolved.json") or command[5]!="--lease-fd"
                or not command[6].isdigit() or type(supervisor.get("lease_fds")) is not list
                or len(supervisor["lease_fds"])!=2 or len(set(supervisor["lease_fds"]))!=2
                or any(type(fd) is not int or fd<0 for fd in supervisor["lease_fds"])
                or int(command[6]) not in supervisor["lease_fds"]
                or abs(supervisor["deadline_monotonic"]-supervisor["started_monotonic"]-LIMIT)>1e-8):
            raise ValueError("retained supervisor source/candidate/deadline changed")
        paid=terminal["paid_wall_seconds"]
    elif "launch_error" in result:
        error=read(checked(result["launch_error"]))
        if error.get("token_sha256")!=result.get("token_sha256") or error.get("source_digest")!=packet["source"]["digest"]:
            raise ValueError("foreign retained interruption")
        paid=error["paid_wall_seconds"]
    for key,value in charge(paid,terminal).items(): close(result.get(key),value)
    close(result.get("overrun_seconds"),max(0.,result["charged_seconds"]-LIMIT))
    if (result["charged_seconds"]>LIMIT)!=(result["status"]=="BUDGET_EXCEEDED"):
        raise ValueError("retained budget overrun cannot be relabelled")
    if terminal is None and result["charged_seconds"]<=LIMIT and result["status"]!="INCOMPLETE":
        raise ValueError("missing durable terminal remains incomplete")
    if terminal is not None and terminal.get("attempt_status")=="timeout" and result["charged_seconds"]<=LIMIT and result["status"]!="INCOMPLETE":
        raise ValueError("deadline cannot become accepted outcome")
    if result["status"]=="COMPLETE":
        if terminal is None or terminal.get("attempt_status")!="completed" or terminal.get("child_returncode")!=0:
            raise ValueError("complete retained outcome needs successful terminal")
        actual=outcome(directory,packet,metadata)
        if result.get("outcome")!=actual or result.get("original_gate")!=actual["original_gate"] or result["charged_seconds"]>LIMIT:
            raise ValueError("retained accepted result changed")
    elif result.get("original_gate")!="UNAVAILABLE":
        raise ValueError("fault/partial cannot claim accepted numeric result")
    return result


def verify_saved_study(saved,packet):
    for key in ("spec","spec_sha256","source","execution_source","request","history","case_definitions","inputs",
            "runtime_contract","lane_runtime","family_paid_budget_seconds","capacity_preflight"):
        if digest(saved.get(key))!=digest(packet[key]): raise ValueError("one immutable candidate/source per output; no retry")
    if saved.get("executed_family")!=FAMILY: raise ValueError("foreign retained family")
    if saved.get("result") is None: return
    result=saved["result"]
    slots={name:{"status":"NOT_RUN"} for name in packet["request"]["tasks"]}
    slots[TASK]={"status":result["original_gate"] if result["status"]=="COMPLETE" else result["status"]}
    if digest(saved.get("slots"))!=digest(slots) or saved.get("status")!=result["status"]:
        raise ValueError("retained full26/no pooled results changed")
    close(saved.get("spent_seconds"),result["charged_seconds"])
    close(saved.get("inclusive_lane_charged_seconds"),PRIOR_LANES["1"]+result["charged_seconds"])
    close(saved.get("inclusive_total_charged_seconds"),PRIOR_TOTAL+result["charged_seconds"])


def run(output):
    output=Path(output).resolve(); packet=read(sidecar(output)); verify(packet); prerequisite=verify_metadata(output,packet)
    metadata=read(checked(prerequisite)); os.environ.update(ENV); root=Path(packet["source"]["snapshot_path"])
    guards().select_snapshot_path(root); guards().guard_imports(packet["source"])
    from experiments.forge.policy_execution import PolicyCoordinator
    coordinator=PolicyCoordinator(QUEUE,report_root=root/"reports/forge")
    if (output/"study.json").exists():
        saved=read(output/"study.json")
        verify_saved_study(saved,packet)
        if saved.get("result") is not None:
            return verify_retained_result(output/"attempt",packet,saved["result"],metadata)
    if not allowance_fits(packet["history"]): return {"status":"BLOCKED_BUDGET","reason":"complete900s allowance does not fit original inclusive lane/campaign cap",**FLAGS}
    try: telemetry=readiness()
    except Exception as exc: return {"status":"WAITING","reason":f"{type(exc).__name__}: {exc}",**FLAGS}
    key,canonical=coordinator.register(packet,output,FAMILY,packet["lane_runtime"])
    if canonical!=output: raise ValueError("compatible attempt exists; attach/review without rerun")
    with coordinator.study_lease(key) as study:
        if study is None: return {"status":"WAITING","reason":"live compatible study owner",**FLAGS}
        row=dict(id=TASK,timeout_seconds=LIMIT); attempt=coordinator.attempt_key(packet,dict(family=FAMILY,recipe_overrides=OVERRIDES),row)
        with coordinator.admit(attempt,packet,row,"cuda:0") as (admission,lease):
            if admission["status"]=="busy": return {"status":"WAITING","reason":admission["reason"],**FLAGS}
            error=None; directory=output/"attempt"
            if admission["status"]=="running" and lease is not None:
                try:
                    telemetry=readiness(); verify(packet); directory.mkdir(); path=directory/"resolved.json"
                    write(path,dict(packet=packet,request=packet["request"],job=validate_request(packet["request"],packet["source"]),
                        resolved_path=str(path),study_output=str(output),metadata_preflight=prerequisite,
                        worker=dict(token=admission["token"],lease_path=admission["lease_path"],attempt=attempt,device="cuda:0"),telemetry=telemetry))
                    coordinator.launch([sys.executable,"-u",str(root/SELF),"--child",str(path),"--lease-fd",str(lease.fileno())],
                        packet,directory/"run.log",(study,lease),LIMIT)
                except BaseException as exc: error=exc
            result=result_for(directory,packet,admission,metadata,error); result["attempt_key"]=attempt
            if admission["status"] in {"running","awaiting_certification"}: coordinator.complete(attempt,result)
            elif admission.get("charged_seconds")!=result["charged_seconds"]: raise ValueError("shared/local retained charge differs")
            slots={name:{"status":"NOT_RUN"} for name in packet["request"]["tasks"]}; slots[TASK]={"status":result["original_gate"] if result["status"]=="COMPLETE" else result["status"]}
            packet.update(status=result["status"],result=result,slots=slots,spent_seconds=result["charged_seconds"],
                inclusive_lane_charged_seconds=PRIOR_LANES["1"]+result["charged_seconds"],inclusive_total_charged_seconds=PRIOR_TOTAL+result["charged_seconds"])
            write(output/"study.json",packet); write(output/"cost.json",result); return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path); parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--checkout",type=Path); parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-only",action="store_true"); parser.add_argument("--metadata-stage",type=Path)
    parser.add_argument("--child",type=Path); parser.add_argument("--execute",type=Path); parser.add_argument("--evaluate",type=Path)
    parser.add_argument("--lease-fd",type=int); args=parser.parse_args(argv)
    if args.metadata_stage is not None: return metadata_stage(args.metadata_stage)
    if any(p is not None for p in (args.child,args.execute,args.evaluate)):
        if args.lease_fd is None: raise ValueError("admitted child/stage requires inherited descriptor")
        if args.child is not None: return child(args.child,args.lease_fd)
        return stage(args.execute or args.evaluate,args.execute is not None,args.lease_fd)
    if args.output is None: raise ValueError("explicit new output required")
    if args.prepare_only:
        if args.checkout is None or args.expected_commit is None: raise ValueError("clean root source identity required")
        print(json.dumps(prepare(args.output,args.checkout,args.expected_commit),sort_keys=True)); return 0
    if args.preflight_only:
        print(json.dumps(preflight(args.output),sort_keys=True)); return 0
    result=run(args.output); print(json.dumps(result,sort_keys=True))
    return 0 if result["status"]=="COMPLETE" else 2


if __name__=="__main__": raise SystemExit(main())
