"""Finite GPU diagnostics for eight explicit named Atlas host adaptations.

The public Forge runtime owns all updates; the independent evaluator owns all
numerical decisions. This wrapper owns source reconstruction, disjoint lanes,
durable admission and retained-array goal media. It grants no qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = "reports/forge/atlas-named-gpu-diagnostics-v1"
SELF = DIRECTORY + "/run_diagnostics.py"
DELEGATE = "reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py"
SCHEMA = "particlegan_atlas_named_gpu_diagnostics_v1"
OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
ORDINARY_DECLARATION = "configs/forge/ideas/atlas-c6-observed-policy-current-v1.json"
LEGACY_ADMISSION_BLOCKER = "new v1 declaration is not immutable legacy evidence; use a v2 decision_contract"
FLAGS = ("qualification_input", "ordinary_tier_credit", "calibration_credit",
         "default_adoption", "cross_cohort_pooling", "speed_ranking")
ENVIRONMENT = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
# A fixed family/cohort/physical-lane declaration, not a configurable sweep.
FAMILIES = {
    "atlas_conditional": {"cohort": "conditional_policy_selected_cloud_v1", "gpu": "0", "cap": 7200,
                          "parents": ("trajectory", "residual_student", "unipolar", "mid_scale_identity")},
    "atlas_ae_routed": {"cohort": "ae_routed_policy_v1", "gpu": "0", "cap": 300,
                        "parents": ("ae_gan_hold",)},
    "atlas_routed": {"cohort": "routed_policy_selected_cloud_v1", "gpu": "1", "cap": 300,
                     "parents": ("unused_token_hold",)},
    "atlas_multibank": {"cohort": "multibank_policy_v1", "gpu": "1", "cap": 1800,
                        "parents": ("cover_leftover",)},
    "atlas_word_joint_min11": {"cohort": "word_joint_policy_min11_v1", "gpu": "1", "cap": 900,
                              "parents": ("five_word_joint_acquisition",)},
}
FIXED_HOSTS = {
    "trajectory": (400, 1800), "residual_student": (400, 1800), "unipolar": (400, 1800),
    "mid_scale_identity": (800, 1800), "ae_gan_hold": (250, 300), "unused_token_hold": (200, 300),
    "cover_leftover": (800, 1800), "five_word_joint_acquisition": (20001, 900),
}
_legacy = None


def legacy():
    """Only generic immutable pin/lease/import/charge/selection utilities."""
    global _legacy
    if _legacy is None:
        loader = importlib.util.spec_from_file_location("_atlas_named_diagnostic_retained_utilities", ROOT / DELEGATE)
        _legacy = importlib.util.module_from_spec(loader)
        sys.modules[loader.name] = _legacy
        loader.loader.exec_module(_legacy)
    return _legacy


def forge(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("experiments.forge." + name)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    forge("contracts").atomic_json(Path(path), value)


def pin(path):
    return legacy().pin(path)


def check_pin(value):
    return legacy().check_pin(value)


def active_rows(spec, family):
    return [row for row in spec["cases"] if row["family"] == family]


def validate_spec(spec, root=None):
    fixed = {"schema": SCHEMA, "id": "atlas-named-hosts-current-gpu-diagnostics-v1",
             "view": "discriminator_stability", "recipe_preset": "atlas", "recipe_overrides": OVERRIDES,
             "seed": 0, "cuda_model": "NVIDIA RTX A6000", "required_slots_per_family": 26,
             "families": list(FAMILIES), "executable_slots": 8, "executable_jobs": 8,
             "diagnostic_cap_seconds": 10500, "lane_cap_seconds": {"0": 7500, "1": 3000},
             "full_view_tiers": {"1": 5, "2": 19, "3": 2}, "frames": 9, "export_grace_seconds": 0,
             "failure_policy": "continue_completed_numerical_FAIL_halt_invalid_lane_no_retry",
             "evidence_use": "named_policy_host_diagnostic",
             "resources": {"host_memory_mb": 2048, "cpu_threads": 1, "minimum_free_gpu_memory_mib": 12288,
                           "maximum_gpu_temperature_c": 82, "memory_fraction": .2}}
    if any(canonical(spec.get(k)) != canonical(v) for k, v in fixed.items()) or any(spec.get(k) is not False for k in FLAGS):
        raise ValueError("the one finite named-family/device/budget/nonqualification contract changed")
    rows = spec.get("cases")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("all eight named questions are required")
    expected = [(family, parent) for family, info in FAMILIES.items() for parent in info["parents"]]
    if [(r.get("family"), r.get("parent_id")) for r in rows] != expected:
        raise ValueError("missing, duplicated, borrowed or reordered named family question")
    for row in rows:
        family, parent = row["family"], row["parent_id"]
        info = FAMILIES[family]; name = parent + "_" + info["cohort"]
        path = "configs/forge/task-variants/" + info["cohort"] + "/" + name + ".json"
        if (row.get("id") != name or row.get("cohort") != info["cohort"] or row.get("gpu") != info["gpu"]
                or row.get("definition") != path or type(row.get("steps")) is not int
                or type(row.get("timeout_seconds")) is not int
                or (row["steps"], row["timeout_seconds"]) != FIXED_HOSTS[parent]
                or any(type(row.get(k)) is not str or len(row[k]) != 64 or any(c not in "0123456789abcdef" for c in row[k])
                       for k in ("sha256", "evaluation_sha256", "execution_sha256"))):
            raise ValueError("named task identity/horizon/allowance/source changed")
        if root is not None:
            actual = Path(root) / path
            if actual.is_symlink() or file_hash(actual) != row["sha256"]:
                raise ValueError("named task source drift: " + name)
            task = read(actual)
            forge("policy_cohorts").validate_policy_task(task, root=root, allow_compiler_annotations=False)
            if (task["id"] != name or task["policy_family"] != family or task["policy_parent"]["id"] != parent
                    or digest(task["evaluation"]) != row["evaluation_sha256"]
                    or digest(task["execution"]) != row["execution_sha256"]
                    or task["execution"]["steps"] != row["steps"]
                    or task["resources"]["timeout_seconds"] != row["timeout_seconds"]
                    or task["execution"]["device"] != "cuda" or task["resources"]["gpus"] != 1
                    or task["resources"]["cpu_threads"] != 1 or task["resources"].get("allow_cpu") is not False
                    or task["evaluation"]["observations"] != 24 or task["evaluation"]["minimum_stable_checks"] != 5):
                raise ValueError("actual named task law/gates/cadence/resources differ")
    return spec


def declaration(root, spec, family):
    """Inline source-bound diagnostic declaration; ordinary draft stays intact."""
    base = forge("planning").load_idea(Path(root), "atlas-c6-observed-policy-current-v1")
    base = deepcopy(base)
    ordinary_reference = {"candidate_id": base["id"], "path": ORDINARY_DECLARATION,
                          "sha256": file_hash(Path(root) / ORDINARY_DECLARATION),
                          "decision_contract": base.pop("decision_contract"), "execution_authorized": False}
    base.update(schema_version=1, id=family + "-c6-named-gpu-diagnostic-v1", trainer_family=family,
                task_cohort=FAMILIES[family]["cohort"], execution_path="public_trainer",
                recipe_preset="atlas", recipe_overrides=deepcopy(OVERRIDES),
                requires_capabilities=["a2", "named_rng", "policy_controls", "policy_serving"],
                hypothesis="Execute the explicitly named host adaptation on GPU under one fixed Atlas pair; preserve original gates and all 26 required questions without qualification credit.",
                changed_factors=["Explicit named family/serving/routing/resource law for its original question, independent of ordinary prerequisite promotion"],
                qualification_scope={"diagnostic_only": True, "ordinary_credit": False, "seed": 0,
                                     "full_required_roster": 26, "historical_reuse": False},
                reference_context_scope="Generic Atlas public_trainer Recipe metadata only; all adapted executions are explicitly task-owned public_components with their own prior/row/encoder laws.",
                ordinary_decision_reference=ordinary_reference)
    # The unchanged v2 contract is inactive provenance. Schema-v1 metadata
    # resolution intentionally remains ineligible for ordinary legacy admission;
    # only this separately fixed diagnostic dispatcher authorizes these jobs.
    base["source_files"] = sorted(set(base.get("source_files", ())) | {ORDINARY_DECLARATION})
    return base


def request_from_base(base, spec, family, source):
    validate_spec(spec)
    info = FAMILIES[family]; candidate = base["candidate"]
    if (candidate.get("id") != family + "-c6-named-gpu-diagnostic-v1"
            or candidate.get("trainer_family") != family or candidate.get("task_cohort") != info["cohort"]
            or candidate.get("recipe_preset") != "atlas" or candidate.get("execution_path") != "public_trainer"
            or canonical(candidate.get("recipe_overrides")) != canonical(OVERRIDES)
            or base["protocol"].get("seed") != 0 or base.get("execution_backend") != "cuda"):
        raise ValueError("actual named candidate/Recipe/seed/backend differs")
    reference = candidate.get("ordinary_decision_reference", {})
    if (candidate.get("schema_version") != 1 or "decision_contract" in candidate
            or reference.get("candidate_id") != "atlas-c6-observed-policy-current-v1"
            or reference.get("path") != ORDINARY_DECLARATION or reference.get("execution_authorized") is not False
            or reference.get("decision_contract", {}).get("status") != "draft"
            or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64
            or base.get("decision_review") is not None or base.get("decision_admission") is not None):
        raise ValueError("ordinary decision reference must stay inactive and source-bound")
    blockers = list(base.get("preflight_blockers", []))
    if blockers.count(LEGACY_ADMISSION_BLOCKER) != 1:
        raise ValueError("expected immutable-legacy admission refusal is absent or ambiguous")
    blockers.remove(LEGACY_ADMISSION_BLOCKER)
    if blockers:
        raise ValueError("nondecision source/API blockers: " + "; ".join(blockers))
    assignments = base["view"]["assignments"]
    mapping = {parent: parent + "_" + info["cohort"] for parent in info["parents"]}
    expected = [mapping.get(parent, parent) for parent in legacy().PARENTS]
    if (list(base["tasks"]) != expected or [a["task"] for a in assignments] != expected
            or base["view"].get("revision") != 4 or base["view"].get("policy_family") != family
            or [sum(a["qualification_tier"] == tier for a in assignments) for tier in (1, 2, 3)] != [5, 19, 2]
            or any(a["importance"] != "required" for a in assignments)):
        raise ValueError("each actual named family retains its own complete 5/19/2 view")
    for row in active_rows(spec, family):
        task = base["tasks"][row["id"]]
        if (task.get("policy_family") != family or task.get("task_cohort") != info["cohort"]
                or task["policy_parent"]["id"] != row["parent_id"]
                or task["execution"].get("execution_path") != "public_components"):
            raise ValueError("borrowed named question")
    request = deepcopy(base)
    request.update(source=deepcopy(source), diagnostic_contract=deepcopy(spec), diagnostic_contract_sha256=digest(spec),
                   evidence_scope=spec["evidence_use"], qualification_reuse=False,
                   campaign_id=spec["id"] + "--" + family, ordinary_request_sha256=digest(base),
                   ordinary_admission={"status": "BLOCKED", "blockers": [LEGACY_ADMISSION_BLOCKER],
                                       "execution_authorized": False})
    request["candidate_revision"] = forge("planning").candidate_revision_for(source["digest"], candidate)
    for job in request["jobs"]:
        job["science"].update(candidate_revision=request["candidate_revision"], evidence_use=spec["evidence_use"],
                             diagnostic_contract_sha256=digest(spec), named_family=family)
    forge("planning").rekey_jobs(request["jobs"])
    for row in active_rows(spec, family):
        jobs = [j for j in request["jobs"] if row["id"] in j["task_ids"]]
        if len(jobs) != 1 or jobs[0]["task_ids"] != [row["id"]] or jobs[0]["budget_seconds"] != row["timeout_seconds"]:
            raise ValueError("an adapted host must be one unchanged full-budget job")
    return json.loads(canonical(request))


def build_requests(root, spec, source=None):
    validate_spec(spec, root)
    requests = {family: forge("planning").resolve_idea(Path(root), family + "-c6-named-gpu-diagnostic-v1",
                declaration=declaration(root, spec, family), view_id=spec["view"], through_tier=1,
                freeze_source=False, execution_backend="cuda", cuda_model=spec["cuda_model"]) for family in FAMILIES}
    if source is None:
        extras = {SELF, DELEGATE, DIRECTORY + "/protocol.json", DIRECTORY + "/README.md"}
        extras.update(str(p.relative_to(root)) for p in (Path(root) / "configs/forge").rglob("*.json"))
        for request in requests.values():
            extras.update(request["source"]["files"])
        source = forge("sources").inspect_source(Path(root), sorted(extras))
    elif any(source["files"].get(p) != h for request in requests.values() for p, h in request["source"]["files"].items()):
        raise ValueError("named request reconstruction escaped the frozen source")
    return {family: request_from_base(base, spec, family, source) for family, base in requests.items()}, source


def case_definitions(requests, spec):
    result = {}
    for row in spec["cases"]:
        request = requests[row["family"]]
        job = next(j for j in request["jobs"] if row["id"] in j["task_ids"])
        result[row["id"]] = {"task": request["tasks"][row["id"]], "job_science": job["science"],
                              "family": row["family"], "evidence_use": spec["evidence_use"]}
    return result


def card_for(requests, spec, source):
    return {"schema": SCHEMA + "_structural_readiness", "source_digest": source["digest"], "contract_sha256": digest(spec),
            "task_preflight": {r["id"]: requests[r["family"]]["tasks"][r["id"]].get("preflight_blockers", []) for r in spec["cases"]},
            "required_questions_per_family": 26, "declared_adapted_questions": 8,
            "model_capacity_proved": False, "learned_quality_proved": False, "optimizer_updates": 0,
            "qualification_input": False, "claim": "Metadata/API/source readiness only; no capacity, CPU numerical or learned-quality proof."}


def preparation_path(output):
    output = Path(output).resolve()
    return output.parent / ("." + output.name + ".atlas-named-gpu-diagnostics.json")


def prepare(spec_path, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare a new empty output; no overwritten evidence or failed retry")
    spec = validate_spec(read(spec_path), ROOT)
    requests, source = build_requests(ROOT, spec)
    snapshot = forge("sources").snapshot_source(ROOT, output.parent / "named-diagnostic-source", source)
    source = {**source, "snapshot_path": str(snapshot)}
    requests, _ = build_requests(snapshot, spec, source)
    card = card_for(requests, spec, source)
    card_path = output.parent / ("." + output.name + ".named-structural-readiness.json")
    write(card_path, card)
    runtimes = [r["runtime"] for r in requests.values()]
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("one frozen software runtime is required")
    packet = {"schema": SCHEMA, "spec": {**spec, "representation_card": pin(card_path)}, "spec_sha256": digest(spec),
              "requests": requests, "source": source, "execution_source": source, "capacity_preflight": card,
              "case_definitions": case_definitions(requests, spec), "runtime_contract": runtimes[0],
              "family_paid_budget_seconds": {family: info["cap"] for family, info in FAMILIES.items()}}
    path = preparation_path(output)
    if path.exists() and read(path) != packet:
        raise ValueError("preserve previous preparation/source; declare a new output")
    write(path, packet)
    return packet


def verify_packet(packet):
    spec = deepcopy(packet["spec"]); card_pin = spec.pop("representation_card")
    validate_spec(spec)
    if packet.get("schema") != SCHEMA or packet.get("spec_sha256") != digest(spec) or packet.get("source") != packet.get("execution_source"):
        raise ValueError("prepared source/contract identity changed")
    source = packet["source"]; snapshot = Path(source["snapshot_path"])
    for relative, path in ((SELF, Path(__file__)), (DELEGATE, ROOT / DELEGATE)):
        if source["files"].get(relative) != file_hash(path):
            raise ValueError("current wrapper/delegated utility differs from the frozen source")
    forge("sources").verify_snapshot(snapshot, source)
    if read(snapshot / "forge-source.json") != {k: v for k, v in source.items() if k != "snapshot_path"}:
        raise ValueError("snapshot metadata changed")
    requests, _ = build_requests(snapshot, spec, source)
    if canonical(requests) != canonical(packet["requests"]):
        raise ValueError("candidate/task/Recipe/runtime/compatibility reconstruction changed")
    card = card_for(requests, spec, source)
    if (read(check_pin(card_pin)) != card or packet.get("capacity_preflight") != card
            or packet.get("case_definitions") != case_definitions(requests, spec)
            or packet.get("family_paid_budget_seconds") != {f: i["cap"] for f, i in FAMILIES.items()}
            or any(r["runtime"] != packet.get("runtime_contract") for r in requests.values())):
        raise ValueError("readiness/full-roster/quota/runtime identity changed")
    legacy().guard_imports(source, execution_root=ROOT)
    return spec


def family_packet(packet, family):
    if family not in FAMILIES:
        raise ValueError("unknown named family")
    result = deepcopy(packet)
    result.update(family=family, request=deepcopy(packet["requests"][family]),
                  lane_runtime=lane_runtime(packet["requests"][family], FAMILIES[family]["gpu"]))
    return result


def lane_runtime(request, physical):
    compute = request["compute_profiles"]["cuda"]
    if physical not in {"0", "1"} or compute.get("availability") == "unavailable" or compute.get("model") != "NVIDIA RTX A6000":
        raise ValueError("fixed GPU runtime unavailable")
    return {**request["runtime"], "device": "cuda:0", "physical_gpu": physical,
            "cuda_device_model": compute["model"], "compute": compute, "torch_threads": 1}


def configure_environment(physical=None):
    if physical is not None and physical not in {"0", "1"}:
        raise ValueError("only the two declared disjoint GPU lanes are supported")
    os.environ.update(ENVIRONMENT)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical or ""


def gpu_readiness(spec, physical, query=None):
    if physical not in {"0", "1"}:
        raise ValueError("unknown physical GPU")
    command = ["nvidia-smi", "--id=" + physical, "--query-gpu=index,name,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
    value = query(command) if query else subprocess.check_output(command, text=True)
    rows = [line.strip().split(",") for line in value.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("unavailable/ambiguous physical lane telemetry")
    device, model, free, temperature = [v.strip() for v in rows[0]]
    free = legacy().number(float(free), "free GPU memory"); temperature = legacy().number(float(temperature), "GPU temperature")
    if (device != physical or model != spec["cuda_model"] or free < spec["resources"]["minimum_free_gpu_memory_mib"]
            or temperature > spec["resources"]["maximum_gpu_temperature_c"]):
        raise ValueError("unsafe/wrong physical lane; zero admission/reservation")
    return {"physical_gpu": device, "model": model, "free_memory_mib": free, "temperature_c": temperature}


def selected_jobs(packet):
    names = [r["id"] for r in active_rows(packet["spec"], packet["family"])]
    return [next(j for j in packet["request"]["jobs"] if j["task_ids"] == [name]) for name in names]


def executable_jobs(packet):
    return [job for job in selected_jobs(packet)
            if not packet["request"]["tasks"][job["task_id"]].get("preflight_blockers")]


def initial_state(packet):
    state = deepcopy(packet); family = packet["family"]
    active = {r["id"] for r in active_rows(packet["spec"], family)}
    state.update(status="PREPARED", jobs=[], ordinary_qualified_tier=0, qualification_input=False, default_adoption=False,
                 measured_paid_seconds=0., unmeasured_interrupt_reserved_seconds=0., spent_seconds=0., slots={})
    for assignment in packet["request"]["view"]["assignments"]:
        name = assignment["task"]; blockers = packet["request"]["tasks"][name].get("preflight_blockers", [])
        state["slots"][name] = {"tier": assignment["qualification_tier"], "in_diagnostic_batch": name in active,
            "diagnostic_status": "BLOCKED" if name in active and blockers else "NOT_RUN",
            "preflight_blockers": blockers, "qualification_input": False}
    state["historical_original_word"] = {"prior_rows": 5, "status": "BLOCKED", "execution_credit": False,
        "claim": "Original N5 cannot enable all public row owners; min11 is a distinct declared resource/joint-code law."}
    return state


def verify_state(state):
    family = state["family"]; allowed = selected_jobs(state)
    expected = initial_state(state)["slots"]
    executable = [j for j in allowed if not expected[j["task_id"]]["preflight_blockers"]]
    if len(state["jobs"]) > len(executable):
        raise ValueError("extra/duplicate named job")
    paid = reserved = 0.; halted = False
    for index, row in enumerate(state["jobs"]):
        job = executable[index]
        if paid + reserved + job["budget_seconds"] > FAMILIES[family]["cap"]:
            raise ValueError("the full original allowance was unavailable before this attempt")
        if halted or row.get("compatibility_key") != job["compatibility_key"] or row.get("task_ids") != job["task_ids"]:
            raise ValueError("each family executes an uninterrupted prefix of its own declared questions")
        if "terminal" in row:
            terminal = read(check_pin(row["terminal"]))
            if terminal.get("token") != row.get("token"):
                raise ValueError("foreign durable supervisor token")
            cost = legacy().charge(terminal["paid_wall_seconds"], terminal, job["budget_seconds"])
        else:
            measurement = 0.
            if "launch_error" in row:
                error = read(check_pin(row["launch_error"]))
                if error.get("token") != row.get("token") or error.get("source_digest") != state["source"]["digest"]:
                    raise ValueError("foreign parent interruption")
                measurement = error["paid_wall_seconds"]
            cost = legacy().charge(measurement, None, job["budget_seconds"])
        if any(row.get(k) != v for k, v in cost.items()):
            raise ValueError("paid/reserved/conservative charge differs from durable semantics")
        paid += cost["paid_wall_seconds"]; reserved += cost["unmeasured_interrupt_reserved_seconds"]
        if row.get("status") == "COMPLETE":
            outcome = row.get("outcome", {})
            if (set(outcome.get("statuses", {})) != set(job["task_ids"])
                    or any(v not in {"PASS", "FAIL"} for v in outcome["statuses"].values())
                    or outcome.get("qualification_input") is not False):
                raise ValueError("only complete original numerical grades may advance a diagnostic")
            statuses = outcome["statuses"]
        elif row.get("status") in {"INCOMPLETE", "INVALID"} and "outcome" not in row:
            statuses = {name: row["status"] for name in job["task_ids"]}; halted = True
        else:
            raise ValueError("unknown terminal status")
        for name, status in statuses.items():
            expected[name]["diagnostic_status"] = status
    if expected != state["slots"]:
        raise ValueError("forged grades or credit for unexecuted/cross-family questions")
    for key, value in (("measured_paid_seconds", paid), ("unmeasured_interrupt_reserved_seconds", reserved), ("spent_seconds", paid + reserved)):
        if not math.isclose(legacy().number(state.get(key), key), value, rel_tol=0, abs_tol=1e-8):
            raise ValueError("study totals do not match durable charged costs")
    overrun = paid + reserved > FAMILIES[family]["cap"]
    if (state.get("family_paid_budget_seconds") != {f: info["cap"] for f, info in FAMILIES.items()}
            or state.get("qualification_input") is not False or state.get("default_adoption") is not False
            or state.get("ordinary_qualified_tier") != 0):
        raise ValueError("finite family quota/nonqualification changed")
    if overrun and state.get("status") != "BUDGET_EXCEEDED":
        raise ValueError("a durable measured overshoot must be retained as BUDGET_EXCEEDED and halt")
    if state.get("status") == "COMPLETE_DIAGNOSTIC" and (halted or len(state["jobs"]) != len(executable)):
        raise ValueError("incomplete lane cannot claim diagnostic completion")


def validate_resolved(resolved):
    packet = resolved["packet"]; verify_packet(packet)
    family = packet.get("family")
    if (family not in FAMILIES or resolved.get("packet_sha256") != digest(packet)
            or packet.get("request") != packet["requests"][family] or resolved.get("request") != packet["request"]):
        raise ValueError("child named family/request attribution changed")
    jobs = [j for j in selected_jobs(packet) if j["compatibility_key"] == resolved["job"].get("compatibility_key")]
    worker = resolved["worker"]; physical = FAMILIES[family]["gpu"]
    if (len(jobs) != 1 or jobs[0] != resolved["job"]
            or packet["request"]["tasks"][jobs[0]["task_id"]].get("preflight_blockers")
            or worker.get("device") != physical or worker.get("lane_runtime") != lane_runtime(packet["request"], physical)
            or packet.get("lane_runtime") != worker["lane_runtime"] or resolved.get("prerequisites") != {}):
        raise ValueError("child job/physical lane/runtime/prerequisite changed")
    for dep in packet["request"]["tasks"][jobs[0]["task_id"]].get("dependencies", []):
        if dep.get("kind") != "gate":
            raise ValueError("this diagnostic imports no unknown/failed checkpoint prerequisite")
    return packet


def child_environment(lease_fd, resolved):
    physical = FAMILIES[resolved["packet"]["family"]]["gpu"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != physical or any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("child visibility/determinism/one-thread environment changed")
    legacy().verify_lease(lease_fd, resolved)
    os.environ["FORGE_LEASE_FD"] = str(lease_fd)


def stage(path, *, execute):
    resolved = read(path); packet = validate_resolved(resolved)
    legacy().guard_imports(packet["source"])
    legacy().verify_lease(int(os.environ["FORGE_LEASE_FD"]), resolved)
    if any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("stage determinism/thread environment changed")
    if execute:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != FAMILIES[packet["family"]]["gpu"]:
            raise ValueError("numerical child must use its declared physical GPU")
        import torch
        torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.2, 0)
        code = forge("runtime").execute(Path(path))
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("independent grading/media requires a clean CPU process")
        grade = forge("evaluate").evaluate(Path(path))
        render_media(resolved, Path(path).parent, grade); code = 0
    legacy().guard_imports(packet["source"])
    return code


def child(path, lease_fd, runner=None):
    resolved = read(path); validate_resolved(resolved); child_environment(lease_fd, resolved)
    command = [sys.executable, "-u", str(Path(__file__).resolve())]; run = runner or subprocess.run
    executed = run(command + ["--execute", str(path)], pass_fds=(lease_fd,), check=False)
    if executed.returncode != 0:
        return executed.returncode
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = ""
    evaluated = run(command + ["--evaluate", str(path)], env=environment, pass_fds=(lease_fd,), check=False)
    if evaluated.returncode != 0:
        return evaluated.returncode
    grade = read(Path(path).parent / "graded-result.json")
    return 0 if all(g.get("gate_status") in {"PASS", "FAIL"} for g in grade["grades"].values()) else 2


def media_views(task, arrays):
    """Lossless views of recorded contexts and outputs; no metric computation."""
    import numpy as np
    family = task["policy_family"]; parent = task["policy_parent"]["id"]
    context = "Actual retained selected-policy output; original known training contexts only, no unseen-context claim."
    def view(kind, title, target, samples, caption):
        return {"kind": kind, "title": title, "target": np.asarray(target), "samples": np.asarray(samples),
                "caption": caption, "xlabel": "x" if kind == "line" or kind == "scatter" else "Feature index",
                "ylabel": "y" if kind == "line" or kind == "scatter" else "Context/value"}
    if family == "atlas_conditional":
        if parent in {"trajectory", "residual_student"}:
            paths = {key: np.asarray(arrays[key]).reshape(len(arrays[key]), 8, 2) for key in ("given_context", "target", "samples")}
            return [view("line", "Given slow trajectory → desired fast trajectory", paths["given_context"], paths["target"],
                         "Each path is one original paired episode; gray is the given source, red the desired target."),
                    view("line", "Desired fast trajectory → actual selected prediction", paths["target"], paths["samples"], context)]
        return [view("image", "Desired responses and selected responses at each declared scale", arrays["target"], arrays["samples"],
                     context + " Rows retain the saved scale order: " + str(arrays["given_context"].reshape(-1).tolist()) + "; all feature values are shown.")]
    if family == "atlas_routed":
        desired = np.stack((arrays["neu"][0], np.asarray(arrays["concept_target"]).reshape(-1)))
        return [view("scatter", "Unused slot stays; used slot moves along its concept", desired, arrays["embeds"],
                     "Desired slot 0 is its retained neutral input; desired slot 1 is its retained concept target. Both actual slots are plotted."),
                view("image", "Every coordinate of the two protected/edited slots", desired, arrays["embeds"], context)]
    if family == "atlas_multibank":
        return [view("image", "Desired pole residuals and selected complete-bank residuals", arrays["targets"] - arrays["neu"], arrays["residual"],
                     "Two rows are + and − concept polarity; columns retain all four original content/leftover coordinates. Desired residual equals retained pole minus neutral.")]
    if family == "atlas_ae_routed":
        return [view("scatter", "Original paired inputs and selected AE reconstructions", arrays["target"], arrays["reconstruction"],
                     "All recorded input/reconstruction coordinates; original reconstruction-MSE gate is scored independently."),
                view("scatter", "Protected anchors and actual selected generated population", arrays["anchors"], arrays["samples"],
                     "Declared fixed-width MoG .025 and routed AE family; output noise is off in this observation. Original anchor-hold gate remains separate.")]
    if family == "atlas_word_joint_min11":
        definition = task["execution"]["host_definition"]; chars = definition["characters"]; length = definition["length"]
        def words(value):
            shaped = np.asarray(value).reshape(-1, len(chars), length)
            return ["".join(chars[int(i)] for i in row) for row in shaped.argmax(1)]
        target = arrays["target"]; labels = words(target)
        first = view("text", "Five canonical words and the first eight actual generated draws", target, arrays["generated"],
                     "Argmax text is a display only; the original quality, mass and token-confidence gates use all 1024 recorded draws. Eleven actual prior rows preserve five target words.")
        first.update(target_labels=labels, sample_labels=words(arrays["generated"])[:8])
        second = view("text", "Correctly paired reconstruction of every canonical word", target, arrays["reconstruction"],
                      "The actual free E encodes each known word; selected G uses the same effective code under public DV12. No reconstruction loss is added to training; padding is displayed.")
        second.update(target_labels=labels, sample_labels=words(arrays["reconstruction"]))
        probabilities = view("image", "Paired token probabilities, including padding", np.asarray(target).reshape(-1, 1, len(chars), length),
                             np.asarray(arrays["reconstruction"]).reshape(-1, 1, len(chars), length),
                             "Each tile is one canonical word. Rows follow the declared 28-character alphabet, columns its six token positions. Actual reconstruction probabilities retain confidence and padding, beyond argmax text.")
        probabilities.update(vmin=0., vmax=1.)
        return [first, second, probabilities]
    raise ValueError("unknown named goal media")


def render_media(resolved, directory, grading):
    import numpy as np
    from benchmarks.toy_audit.api_run import render_gif
    name = resolved["job"]["task_id"]; task = resolved["request"]["tasks"][name]
    if set(grading.get("grades", {})) != {name} or grading["grades"][name].get("gate_status") not in {"PASS", "FAIL"}:
        raise ValueError("only independently complete numerical outcomes receive goal media")
    raw = read(directory / "raw-result.json"); evidence = raw["evidence"]
    forge("artifacts").verify_artifacts(Path(evidence["artifact_root"]), evidence["artifact_manifest"])
    points, selected, paths, metadata = legacy().media_selection(task, evidence)
    if len(points) != 24 or metadata:
        raise ValueError("named media requires its complete 24-observation original cadence")
    records = []
    for index, path in zip(selected, paths):
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("nonfinite retained media array")
        point = points[index]
        records.append({"step": point["step"], "metrics": {k: v for k, v in point.items() if k != "step"},
                        "passed": legacy().point_pass(point, task["evaluation"]["thresholds"]), "views": media_views(task, arrays)})
    status = grading["grades"][name]["gate_status"]; goal = task["policy_parent"]["id"] + " — " + task["policy_family"]
    feature_views = [view for record in records for view in record["views"] if view["kind"] == "image" and "vmin" not in view]
    if feature_views:
        low = min(float(np.min(view[key])) for view in feature_views for key in ("target", "samples"))
        high = max(float(np.max(view[key])) for view in feature_views for key in ("target", "samples"))
        if high == low:
            low -= .5; high += .5
        for view in feature_views:
            view.update(vmin=low, vmax=high)
            view["caption"] += f" Fixed display range across all retained frames: [{low:.6g}, {high:.6g}]; signed values are preserved."
    gif = directory / (name + "-goal.gif")
    render_gif({"id": name, "goal": goal + " — original terminal gate; named diagnostic only, no ordinary/default/speed credit", "default_steps": task["execution"]["steps"]}, records, gif,
               full_budget=True, requested_steps=task["execution"]["steps"], final_verdict=status)
    write(directory / (name + "-media.json"), {"schema": SCHEMA + "_media", "task": name, "family": task["policy_family"],
        "source_digest": resolved["request"]["source"]["digest"], "renderer_sha256": file_hash(Path(__file__)),
        "original_gate": status, "qualification_input": False, "draws": 0, "optimizer_updates": 0,
        "actual_steps": [r["step"] for r in records], "inputs": [pin(path) for path in paths], "gif": pin(gif)})


def certified_outcome(path, grader=None, policy_guard=None):
    resolved = read(path); packet = validate_resolved(resolved); directory = Path(path).parent
    raw = read(directory / "raw-result.json"); grade = read(directory / "graded-result.json")
    name = resolved["job"]["task_id"]; task = packet["request"]["tasks"][name]
    if (grade.get("raw_hash") != digest(raw) or grade.get("source_digest") != packet["source"]["digest"]
            or set(grade.get("grades", {})) != {name}):
        raise ValueError("raw/independent grade/source/question changed")
    evidence = raw["evidence"]; artifacts = Path(evidence["artifact_root"]).resolve()
    if not artifacts.is_relative_to(directory.resolve()):
        raise ValueError("foreign candidate/task artifacts")
    forge("artifacts").verify_artifacts(artifacts, evidence["artifact_manifest"])
    guard = (policy_guard or forge("views")._policy_guards)(task, evidence)
    if guard is not None:
        raise ValueError("invalid/incomplete owner/runtime/purity cannot count as numerical FAIL: " + canonical(guard))
    expected = (grader or forge("views").grade_result)(task, raw)
    if expected != grade["grades"][name] or expected.get("gate_status") not in {"PASS", "FAIL"}:
        raise ValueError("complete original numerical grade differs")
    media_path = directory / (name + "-media.json"); receipt = read(media_path)
    points, selected, paths, metadata = legacy().media_selection(task, evidence)
    gif = directory / (name + "-goal.gif")
    fixed = {"schema": SCHEMA + "_media", "task": name, "family": packet["family"],
             "source_digest": packet["source"]["digest"], "renderer_sha256": packet["source"]["files"][SELF],
             "original_gate": expected["gate_status"], "qualification_input": False, "draws": 0, "optimizer_updates": 0,
             "actual_steps": [points[i]["step"] for i in selected], "inputs": [pin(p) for p in paths], "gif": pin(gif)}
    if len(points) != 24 or metadata or receipt != fixed:
        raise ValueError("media source/original verdict/cadence/retained inputs changed")
    from PIL import Image
    with Image.open(gif) as image:
        if image.n_frames != len(selected):
            raise ValueError("GIF frame count differs from actual observations")
    return {"statuses": {name: expected["gate_status"]}, "resolved": pin(path), "raw": pin(directory / "raw-result.json"),
            "grading": pin(directory / "graded-result.json"), "media": {name: {"receipt": pin(media_path), "gif": pin(gif)}}, "qualification_input": False}


def save_state(output, state):
    state["measured_paid_seconds"] = sum(r["paid_wall_seconds"] for r in state["jobs"])
    state["unmeasured_interrupt_reserved_seconds"] = sum(r["unmeasured_interrupt_reserved_seconds"] for r in state["jobs"])
    state["spent_seconds"] = state["measured_paid_seconds"] + state["unmeasured_interrupt_reserved_seconds"]
    if state["spent_seconds"] > FAMILIES[state["family"]]["cap"]:
        state.update(status="BUDGET_EXCEEDED", stop_reason="Durable measured supervisor overrun retained; no further admission or budget reset.")
    verify_state(state); write(Path(output) / "study.json", state)
    lines = ["# " + state["family"] + " GPU diagnostic", "", "All 26 required questions remain visible. Only this family's named adapted subset is executed; no ordinary/default/speed credit.", "",
             f"Measured paid {state['measured_paid_seconds']:.6f}s; reserve {state['unmeasured_interrupt_reserved_seconds']:.6f}s; charged {state['spent_seconds']:.6f}/{FAMILIES[state['family']]['cap']}s.", "",
             "| Actual question | Tier | Original diagnostic gate | Goal GIF |", "|---|---:|---|---|"]
    for assignment in state["request"]["view"]["assignments"]:
        name = assignment["task"]; media = ""
        for job in state["jobs"]:
            if name in job.get("outcome", {}).get("media", {}):
                relative = Path(job["outcome"]["media"][name]["gif"]["path"]).relative_to(output).as_posix()
                media = f"[actual goal GIF]({relative})"
        lines.append(f"| {name} | {assignment['qualification_tier']} | {state['slots'][name]['diagnostic_status']} | {media} |")
    lines += ["", "Original five-row word source remains BLOCKED under full-owner guards. Named min11 is a different explicit resource/joint-code law; neither variant grants the original source credit.", ""]
    (Path(output) / "README.md").write_text("\n".join(lines))


def coordinator_for(queue_root):
    return forge("policy_execution").PolicyCoordinator(Path(queue_root).resolve(), report_root=ROOT / "reports/forge")


def run_family(output, packet, queue_root):
    output = Path(output).resolve(); verify_packet(packet); family = packet["family"]
    physical = FAMILIES[family]["gpu"]; runtime = lane_runtime(packet["request"], physical)
    state = read(output / "study.json") if (output / "study.json").exists() else initial_state(packet)
    for key in ("spec", "spec_sha256", "source", "execution_source", "request", "requests", "family", "case_definitions", "runtime_contract", "family_paid_budget_seconds", "lane_runtime"):
        if state.get(key) != packet.get(key):
            raise ValueError("resume source/family/quota/runtime changed")
    verify_state(state)
    if state["status"] == "BUDGET_EXCEEDED":
        return state
    for row in state["jobs"]:
        if "outcome" in row and certified_outcome(check_pin(row["outcome"]["resolved"])) != row["outcome"]:
            raise ValueError("retained scientific/media result changed")
        if row["status"] != "COMPLETE":
            return state  # No failed infrastructure retry.
    gpu_readiness(packet["spec"], physical)
    coordinator = coordinator_for(queue_root)
    key, actual = coordinator.register(packet, output, family, runtime)
    if actual != output:
        raise ValueError("compatible named study already has a canonical output")
    state.update(executed_family=family, coordinator=packet.get("coordinator")); save_state(output, state)
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:
            raise RuntimeError("another owner holds this exact family study")
        for job in executable_jobs(packet)[len(state["jobs"]):]:
            if state["spent_seconds"] + job["budget_seconds"] > FAMILIES[family]["cap"]:
                state.update(status="INCOMPLETE", stop_reason="next full allowance cannot fit"); break
            telemetry = gpu_readiness(packet["spec"], physical)
            row = {"id": job["task_id"], "timeout_seconds": job["budget_seconds"]}
            attempt = coordinator.attempt_key(packet, {"family": family, "recipe_overrides": OVERRIDES}, row)
            with coordinator.admit(attempt, packet, row, "cuda:0") as (admission, lease):
                if admission["status"] == "busy":
                    state.update(status="INCOMPLETE", stop_reason=admission["reason"]); break
                directory = output / "attempts" / job["execution_group"]; directory.mkdir(parents=True, exist_ok=True)
                path = directory / "resolved.json"; token = admission["token"]; launch_error = None
                if admission["status"] == "running" and lease is not None:
                    gpu_readiness(packet["spec"], physical)
                    resolved = {"schema_version": 1, "packet": packet, "packet_sha256": digest(packet), "request": packet["request"],
                        "job": job, "prerequisites": {}, "worker": {"device": physical, "token": token, "attempt": attempt,
                        "lane_runtime": runtime, "lease_path": admission["lease_path"]}}
                    write(path, resolved)
                    command = [sys.executable, "-u", str(Path(packet["source"]["snapshot_path"]) / SELF), "--child", str(path), "--lease-fd", str(lease.fileno())]
                    try:
                        coordinator.launch(command, packet, directory / "run.log", (study_lease, lease), job["budget_seconds"])
                    except (Exception, KeyboardInterrupt) as error:
                        launch_error = {"token": token, "source_digest": packet["source"]["digest"], "type": type(error).__name__,
                            "message": str(error), "paid_wall_seconds": legacy().number(getattr(error, "paid_wall_seconds", 0.), "parent paid")}
                        write(directory / "launch-error.json", launch_error)
                else:
                    launch_error = read(directory / "launch-error.json") if (directory / "launch-error.json").exists() else None
                terminal_path = Path(admission["lease_path"]).parent / "supervisor-terminal.json"
                terminal = read(terminal_path) if terminal_path.exists() else None
                if terminal is not None and terminal.get("token") != token:
                    raise ValueError("foreign durable supervisor terminal")
                paid = terminal.get("paid_wall_seconds", 0.) if terminal else (launch_error or {}).get("paid_wall_seconds", 0.)
                result = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"], "attempt_key": attempt,
                          "token": token, "status": "INCOMPLETE", "admission": telemetry,
                          **legacy().charge(paid, terminal, job["budget_seconds"])}
                if terminal is not None:
                    result["terminal"] = pin(terminal_path)
                elif launch_error is not None:
                    result["launch_error"] = pin(directory / "launch-error.json")
                if terminal and terminal["attempt_status"] == "completed" and terminal.get("child_returncode") == 0:
                    try:
                        result["outcome"] = certified_outcome(path); result["status"] = "COMPLETE"
                    except Exception as error:
                        result.update(status="INVALID", reason=f"{type(error).__name__}: {error}")
                elif terminal and terminal["attempt_status"] == "completed":
                    result.update(status="INVALID", reason="runtime/grading/media did not produce a valid complete original outcome")
                if admission["status"] in {"running", "awaiting_certification"}:
                    coordinator.complete(attempt, result)
                elif admission.get("charged_seconds") != result["charged_seconds"]:
                    raise ValueError("retained local/central conservative charges differ")
                state["jobs"].append(result)
                for name in job["task_ids"]:
                    state["slots"][name]["diagnostic_status"] = result["outcome"]["statuses"][name] if result["status"] == "COMPLETE" else result["status"]
                state["status"] = "RUNNING" if result["status"] == "COMPLETE" else result["status"]
                save_state(output, state)
                if result["status"] != "COMPLETE" or state["status"] == "BUDGET_EXCEEDED":
                    return state
        else:
            state["status"] = "COMPLETE_DIAGNOSTIC"
        save_state(output, state)
    return state


def run_lane(output, queue_root, physical):
    packet = read(preparation_path(output)); verify_packet(packet)
    summaries = []
    for family, info in FAMILIES.items():
        if info["gpu"] != physical:
            continue
        state = run_family(Path(output) / family, family_packet(packet, family), queue_root)
        summaries.append({"family": family, "status": state["status"], "paid_seconds": state["measured_paid_seconds"],
                          "reserved_seconds": state["unmeasured_interrupt_reserved_seconds"], "qualification_input": False})
        if state["status"] != "COMPLETE_DIAGNOSTIC":
            break  # Infrastructure/invalid evidence stops the physical lane.
    return summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / DIRECTORY / "protocol.json")
    p.add_argument("--output", type=Path); p.add_argument("--queue-root", type=Path)
    p.add_argument("--prepare-only", action="store_true"); p.add_argument("--gpus", choices=("0", "1"))
    stages = p.add_mutually_exclusive_group()
    for flag in ("child", "execute", "evaluate"):
        stages.add_argument("--" + flag, type=Path)
    p.add_argument("--lease-fd", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.child:
        if args.lease_fd is None:
            raise ValueError("child needs an admitted inherited lease descriptor")
        return child(args.child, args.lease_fd)
    if args.execute or args.evaluate:
        return stage(args.execute or args.evaluate, execute=args.execute is not None)
    if args.output is None:
        raise ValueError("declare a fresh external output")
    configure_environment(None if args.prepare_only else args.gpus)
    if args.prepare_only:
        packet = prepare(args.spec, args.output)
        print(canonical({"status": "PREPARED", "source_digest": packet["source"]["digest"], "spec_sha256": packet["spec_sha256"], "qualification_input": False}), flush=True)
        return 0
    if args.gpus is None or args.queue_root is None:
        raise ValueError("run needs an explicit physical lane and the existing shared queue")
    rows = run_lane(args.output, args.queue_root, args.gpus)
    print(canonical({"physical_gpu": args.gpus, "families": rows, "qualification_input": False}), flush=True)
    return 0 if all(r["status"] == "COMPLETE_DIAGNOSTIC" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
