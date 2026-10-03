"""One fixed, non-qualifying Atlas GPU diagnostic batch.

Training and numerical grading belong to the unchanged Forge runtime/evaluator.
This wrapper owns only the finite diagnostic contract, admission and receipts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = "reports/forge/atlas-current-gpu-diagnostics-v1"
SELF = DIRECTORY + "/run_diagnostics.py"
SCHEMA = "particlegan_atlas_current_gpu_diagnostics_v1"
SUFFIX = "_policy_selected_cloud_v1"
OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
BLOCKED = frozenset({"unused_token_hold", "ae_gan_hold", "five_word_joint_acquisition",
                     "trajectory", "residual_student", "unipolar", "cover_leftover", "mid_scale_identity"})
PARENTS = ("two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition", "five_word_joint_acquisition",
           "trajectory", "residual_student", "unipolar", "cover_leftover", "mid_scale_identity", "mode_hold",
           "vector_two_broad", "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
           "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2",
           "grid100", "rotated100", "staggered100", "ring_hold", "ring_extension")
FLAGS = ("qualification_input", "ordinary_tier_credit", "calibration_credit",
         "default_adoption", "cross_cohort_pooling", "speed_ranking")
ENVIRONMENT = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def forge(name):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("experiments.forge." + name)


def write(path, value):
    forge("contracts").atomic_json(Path(path), value)


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid finite nonnegative " + label)
    return float(value)


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def check_pin(record):
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file() or pin(path) != record:
        raise ValueError("missing or changed pinned artifact: " + str(path))
    return path


def validate_spec(spec, root=None):
    fixed = {"schema": SCHEMA, "id": "atlas-c6-policy-current-gpu-diagnostics-v1",
             "candidate_id": "atlas-c6-observed-policy-current-v1", "view": "discriminator_stability",
             "task_cohort": "policy_selected_cloud_v1", "recipe_overrides": OVERRIDES,
             "seed": 0, "cuda_model": "NVIDIA RTX A6000", "physical_gpu": "1",
             "required_slots": 26, "executable_slots": 18, "executable_jobs": 17,
             "diagnostic_cap_seconds": 34800, "full_defined_allowance_seconds": 45300,
             "export_grace_seconds": 0, "frames": 9,
             "failure_policy": "continue_completed_numerical_FAIL_halt_invalid_execution",
             "evidence_use": "current_policy_task_diagnostic"}
    if any(canonical(spec.get(key)) != canonical(value) for key, value in fixed.items()):
        raise ValueError("diagnostic contract differs from the one fixed GPU batch")
    if any(spec.get(key) is not False for key in FLAGS):
        raise ValueError("diagnostic evidence cannot supply qualification/default/speed credit")
    expected_resources = {"host_memory_mb": 2048, "cpu_threads": 1,
                          "minimum_free_gpu_memory_mib": 12288, "maximum_gpu_temperature_c": 82,
                          "memory_fraction": .2}
    if canonical(spec.get("resources")) != canonical(expected_resources):
        raise ValueError("resource/precision admission contract changed")
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 26:
        raise ValueError("retain all 26 required slots")
    seen, groups, tiers = set(), {}, {1: 0, 2: 0, 3: 0}
    for row in cases:
        parent = row.get("parent_id")
        if not isinstance(parent, str) or row.get("id") != parent + SUFFIX or parent in seen:
            raise ValueError("duplicate/unknown parent-slot mapping")
        seen.add(parent)
        if (type(row.get("executable")) is not bool or row["executable"] != (parent not in BLOCKED)
                or type(row.get("tier")) is not int or row["tier"] not in tiers
                or type(row.get("steps")) is not int or row["steps"] <= 0
                or type(row.get("timeout_seconds")) is not int or row["timeout_seconds"] <= 0):
            raise ValueError("invalid fixed task budget or executable roster")
        tiers[row["tier"]] += 1
        expected_path = "configs/forge/task-variants/policy_selected_cloud_v1/" + row["id"] + ".json"
        if (row.get("definition") != expected_path or not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError("unsafe or unbound task declaration")
        if root is not None:
            path = Path(root) / expected_path
            if file_hash(path) != row["sha256"]:
                raise ValueError("frozen task declaration changed: " + row["id"])
            task = read(path)
            if (task["id"] != row["id"] or task["policy_parent"]["id"] != parent
                    or task["execution"]["steps"] != row["steps"]
                    or task["resources"]["timeout_seconds"] != row["timeout_seconds"]
                    or task["execution"].get("execution_group", task["id"]) != row["execution_group"]
                    or task["execution"]["device"] != "cuda" or task["resources"].get("gpus") != 1):
                raise ValueError("actual task budget/group/device differs from the contract")
        group = groups.setdefault(row["execution_group"], {"members": [], "cap": row["timeout_seconds"],
                                                          "execute": row["executable"]})
        if group["cap"] != row["timeout_seconds"] or group["execute"] != row["executable"]:
            raise ValueError("conflicting group budget/ownership")
        group["members"].append(parent)
    if ([r["parent_id"] for r in cases] != list(PARENTS) or tiers != {1: 5, 2: 19, 3: 2}
            or not BLOCKED <= seen or len(groups) != 25
            or sum(g["execute"] for g in groups.values()) != 17
            or sum(r["executable"] for r in cases) != 18
            or sum(g["cap"] for g in groups.values()) != 45300
            or sum(g["cap"] for g in groups.values() if g["execute"]) != 34800):
        raise ValueError("required-slot/job/budget denominator changed")
    multi = [set(g["members"]) for g in groups.values() if len(g["members"]) != 1]
    if multi != [{"ring_hold", "ring_extension"}]:
        raise ValueError("keep the two ring slots in one uninterrupted job")
    return spec


def request_from_base(base, spec, source):
    """Preserve ordinary review; give diagnostic jobs independent identities."""
    validate_spec(spec)
    candidate = base["candidate"]
    if (candidate.get("id") != spec["candidate_id"] or candidate.get("recipe_preset") != "atlas"
            or candidate.get("task_cohort") != spec["task_cohort"]
            or canonical(candidate.get("recipe_overrides")) != canonical(OVERRIDES)
            or base["protocol"].get("seed") != 0 or base.get("execution_backend") != "cuda"):
        raise ValueError("fixed candidate, seed or GPU cohort changed")
    advisory = base.get("decision_review", {}).get("blockers", [])
    remaining = list(base.get("preflight_blockers", []))
    for message in advisory:
        if message in remaining:
            remaining.remove(message)
    if remaining:
        raise ValueError("non-decision candidate/source blockers: " + "; ".join(remaining))
    names = {row["id"] for row in spec["cases"]}
    if set(base["tasks"]) != names:
        raise ValueError("planner task denominator differs from the fixed diagnostic contract")
    assignments = base["view"]["assignments"]
    if ({a["task"] for a in assignments} != names or len(assignments) != 26
            or base["view"]["revision"] != 4):
        raise ValueError("actual policy qualification view/slots changed")
    for row in spec["cases"]:
        task = base["tasks"][row["id"]]
        blockers = task.get("preflight_blockers", [])
        if bool(blockers) == row["executable"]:
            raise ValueError("structural preflight differs from the frozen 18/8 roster: " + row["id"])
        matches = [a for a in assignments if a["task"] == row["id"]]
        if (matches[0]["qualification_tier"] != row["tier"] or matches[0].get("order", 0) != row["order"]
                or matches[0]["importance"] != "required"):
            raise ValueError("qualification slot order/importance changed")
    request = deepcopy(base)
    request.update(source=deepcopy(source), diagnostic_contract=deepcopy(spec),
                   diagnostic_contract_sha256=digest(spec), evidence_scope=spec["evidence_use"],
                   qualification_reuse=False, campaign_id=spec["id"], ordinary_request_sha256=digest(base))
    request["candidate_revision"] = forge("planning").candidate_revision_for(source["digest"], candidate)
    for job in request["jobs"]:
        job["science"].update(candidate_revision=request["candidate_revision"],
                              evidence_use=spec["evidence_use"], diagnostic_contract_sha256=digest(spec))
    forge("planning").rekey_jobs(request["jobs"])
    by_task = {m: job for job in request["jobs"] for m in job["task_ids"]}
    if set(by_task) != names or len(request["jobs"]) != 25:
        raise ValueError("planner split or omitted an execution group")
    for row in spec["cases"]:
        job = by_task[row["id"]]
        if job["budget_seconds"] != row["timeout_seconds"] or job["execution_group"] != row["execution_group"]:
            raise ValueError("planner job allowance/group changed")
    return request


def build_request(root, spec, source=None):
    validate_spec(spec, root)
    base = forge("planning").resolve_idea(Path(root), spec["candidate_id"], view_id=spec["view"],
                                        through_tier=1, freeze_source=False, execution_backend="cuda",
                                        cuda_model=spec["cuda_model"])
    if source is None:
        extras = set(base["source"]["files"])
        extras.update(str(p.relative_to(root)) for p in (Path(root) / "configs/forge").rglob("*.json"))
        extras.update((SELF, DIRECTORY + "/protocol.json", DIRECTORY + "/README.md"))
        source = forge("sources").inspect_source(Path(root), sorted(extras))
    elif any(source["files"].get(n) != h for n, h in base["source"]["files"].items()):
        raise ValueError("reconstructed scientific source is outside the frozen snapshot")
    return request_from_base(base, spec, source)


def card_for(request, spec):
    return {"schema": "atlas_gpu_structural_readiness_v1", "claim": spec["structural_card_claim"],
            "source_digest": request["source"]["digest"], "candidate_revision": request["candidate_revision"],
            "contract_sha256": digest(spec), "task_preflight": {
                row["id"]: request["tasks"][row["id"]].get("preflight_blockers", []) for row in spec["cases"]},
            "required_slots": 26, "executable_slots": 18, "optimizer_updates": 0,
            "model_capacity_proved": False, "learned_quality_proved": False, "qualification_input": False}


def case_definitions(request, spec):
    by_task = {m: job for job in request["jobs"] for m in job["task_ids"]}
    return {row["id"]: {"task": request["tasks"][row["id"]], "evidence_use": spec["evidence_use"],
                        "execution_group_members": {m: request["tasks"][m] for m in by_task[row["id"]]["task_ids"]},
                        "job_science": by_task[row["id"]]["science"]}
            for row in spec["cases"]}


def preparation_path(output):
    output = Path(output).resolve()
    return output.parent / ("." + output.name + ".atlas-gpu-diagnostics.json")


def prepare(spec_path, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("prepare requires a fresh output; no failed retry or archive overwrite")
    spec = validate_spec(read(spec_path), ROOT)
    request = build_request(ROOT, spec)
    source = request["source"]
    snapshot = forge("sources").snapshot_source(ROOT, output.parent / "diagnostic-source", source)
    source = {**source, "snapshot_path": str(snapshot)}
    request = build_request(snapshot, spec, source)
    card_path = output.parent / ("." + output.name + ".structural-readiness.json")
    card = card_for(request, spec)
    write(card_path, card)
    packet = {"schema": SCHEMA, "spec": {**spec, "representation_card": pin(card_path)},
              "spec_sha256": digest(spec), "request": request, "source": source,
              "execution_source": source, "case_definitions": case_definitions(request, spec), "capacity_preflight": card,
              "runtime_contract": request["runtime"], "family_paid_budget_seconds": {"atlas": 34800}}
    target = preparation_path(output)
    if target.exists() and read(target) != packet:
        raise ValueError("preparation identity changed; keep the old source/output and declare a new study")
    write(target, packet)
    return packet


def verify_packet(packet):
    spec = deepcopy(packet["spec"])
    card_pin = spec.pop("representation_card")
    validate_spec(spec)
    if packet.get("schema") != SCHEMA or packet.get("spec_sha256") != digest(spec):
        raise ValueError("prepared contract identity changed")
    source = packet["execution_source"]
    if packet.get("source") != source:
        raise ValueError("source/child snapshot mismatch")
    if source["files"].get(SELF) != file_hash(Path(__file__)):
        raise ValueError("maintained diagnostic wrapper differs from the frozen child source")
    snapshot = Path(source["snapshot_path"])
    forge("sources").verify_snapshot(snapshot, source)
    if read(snapshot / "forge-source.json") != {k: v for k, v in source.items() if k != "snapshot_path"}:
        raise ValueError("snapshot provenance metadata changed")
    expected = build_request(snapshot, spec, source)
    if expected != packet["request"]:
        raise ValueError("candidate/task/Recipe/runtime/compatibility reconstruction changed")
    card = read(check_pin(card_pin))
    if card != card_for(expected, spec) or packet.get("capacity_preflight") != card:
        raise ValueError("structural card changed or claimed learned/capacity credit")
    definitions = case_definitions(expected, spec)
    if (packet.get("case_definitions") != definitions or packet.get("runtime_contract") != expected["runtime"]
            or packet.get("family_paid_budget_seconds") != {"atlas": 34800}):
        raise ValueError("prepared denominator/runtime/paid quota changed")
    guard_imports(source, execution_root=ROOT)
    return spec


def configure_environment(cuda=False):
    os.environ.update(ENVIRONMENT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if cuda else ""


def lane_runtime(request):
    compute = request["compute_profiles"]["cuda"]
    if compute.get("availability") == "unavailable" or compute.get("model") != "NVIDIA RTX A6000":
        raise ValueError("required GPU hardware cohort is unavailable")
    return {**request["runtime"], "device": "cuda:0", "cuda_device_model": compute["model"],
            "compute": compute, "torch_threads": 1}


def gpu_readiness(spec, query=None):
    command = ["nvidia-smi", "--id=1", "--query-gpu=index,name,memory.free,temperature.gpu",
               "--format=csv,noheader,nounits"]
    text = query(command) if query else subprocess.check_output(command, text=True)
    rows = [line.strip().split(",") for line in text.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("GPU admission telemetry is unavailable or ambiguous")
    index, model, memory, temperature = [v.strip() for v in rows[0]]
    memory = number(float(memory), "free GPU memory")
    temperature = number(float(temperature), "GPU temperature")
    if (index != "1" or model != spec["cuda_model"]
            or memory < spec["resources"]["minimum_free_gpu_memory_mib"]
            or temperature > spec["resources"]["maximum_gpu_temperature_c"]):
        raise ValueError("unsafe or wrong GPU lane; no reservation/child is permitted")
    return {"physical_gpu": index, "model": model, "free_memory_mib": memory, "temperature_c": temperature}


def initial_state(packet):
    spec = packet["spec"]
    state = deepcopy(packet)
    state.update(status="PREPARED", jobs=[], slots={row["id"]: {
        "parent_id": row["parent_id"], "tier": row["tier"],
        "diagnostic_status": "NOT_RUN" if row["executable"] else "BLOCKED",
        "reason": None if row["executable"] else packet["request"]["tasks"][row["id"]]["preflight_blockers"],
        "qualification_input": False} for row in spec["cases"]}, measured_paid_seconds=0.,
        unmeasured_interrupt_reserved_seconds=0., spent_seconds=0., ordinary_qualified_tier=0,
        qualification_input=False, default_adoption=False)
    return state


def charge(paid, terminal, allowance):
    paid = number(paid, "paid seconds")
    completed = terminal is not None and terminal.get("attempt_status") == "completed"
    charged = paid if completed else max(number(allowance, "allowance"), paid)
    return {"paid_wall_seconds": paid, "unmeasured_interrupt_reserved_seconds": charged - paid,
            "charged_seconds": charged}


def verify_costs(state):
    seen, paid, reserved = set(), 0., 0.
    allowed = {j["compatibility_key"]: j for j in state["request"]["jobs"]
               if all(state["slots"][m]["parent_id"] not in BLOCKED for m in j["task_ids"])}
    for row in state["jobs"]:
        key = row.get("compatibility_key")
        if key in seen or key not in allowed or row.get("task_ids") != allowed[key]["task_ids"]:
            raise ValueError("duplicate/foreign/split job cost")
        seen.add(key)
        p = number(row.get("paid_wall_seconds"), "job paid")
        r = number(row.get("unmeasured_interrupt_reserved_seconds"), "job reservation")
        if number(row.get("charged_seconds"), "job charged") != p + r:
            raise ValueError("paid/reserved/charged arithmetic changed")
        if "terminal" in row:
            terminal = read(check_pin(row["terminal"]))
            if terminal.get("token") != row.get("token"):
                raise ValueError("foreign durable terminal")
            if number(terminal["paid_wall_seconds"], "durable paid") != p:
                raise ValueError("saved spend differs from durable measured paid time")
            expected = charge(p, terminal, allowed[key]["budget_seconds"])
        else:
            measured = 0.
            if "launch_error" in row:
                error = read(check_pin(row["launch_error"]))
                if error.get("token") != row.get("token") or error.get("source_digest") != state["source"]["digest"]:
                    raise ValueError("foreign parent interruption measurement")
                measured = number(error.get("paid_wall_seconds", 0.), "parent interruption paid")
            expected = charge(measured, None, allowed[key]["budget_seconds"])
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("interruption charge does not retain the original full allowance")
        paid += p; reserved += r
    for key, expected in (("measured_paid_seconds", paid), ("unmeasured_interrupt_reserved_seconds", reserved),
                          ("spent_seconds", paid + reserved)):
        if not math.isclose(number(state.get(key), key), expected, rel_tol=0., abs_tol=1e-8):
            raise ValueError("study spend/quota arithmetic changed")
    if paid + reserved > 34800:
        raise ValueError("finite diagnostic ceiling exhausted")
    if state.get("family_paid_budget_seconds") != {"atlas": 34800} or any(state.get(k) is not False for k in (
            "qualification_input", "default_adoption")) or state.get("ordinary_qualified_tier") != 0:
        raise ValueError("quota or non-qualifying scope changed")


def verify_state(state):
    verify_costs(state)
    expected = initial_state(state)["slots"]
    supported = [j for j in state["request"]["jobs"]
                 if all(expected[m]["diagnostic_status"] != "BLOCKED" for m in j["task_ids"])]
    order = {r["id"]: i for i, r in enumerate(state["spec"]["cases"])}
    supported.sort(key=lambda j: min(order[m] for m in j["task_ids"]))
    if len(state["jobs"]) > len(supported):
        raise ValueError("diagnostic job denominator changed")
    halted = False
    for index, row in enumerate(state["jobs"]):
        if halted or row["compatibility_key"] != supported[index]["compatibility_key"]:
            raise ValueError("diagnostic jobs must retain the uninterrupted ordered prefix")
        if row.get("status") == "COMPLETE":
            outcome = row.get("outcome", {})
            statuses = outcome.get("statuses", {})
            if (set(statuses) != set(row["task_ids"]) or any(v not in {"PASS", "FAIL"} for v in statuses.values())
                    or outcome.get("qualification_input") is not False):
                raise ValueError("complete diagnostic requires every original numerical grade")
        elif row.get("status") in {"INCOMPLETE", "INVALID"} and "outcome" not in row:
            statuses = {m: row["status"] for m in row["task_ids"]}
            halted = True
        else:
            raise ValueError("unknown/forged diagnostic terminal status")
        for member, status in statuses.items():
            expected[member]["diagnostic_status"] = status
    if expected != state.get("slots") or any(state.get(k, False) is not False for k in FLAGS):
        raise ValueError("retained slot grades or non-qualifying scope changed")
    if state.get("status") == "COMPLETE_DIAGNOSTIC" and (halted or len(state["jobs"]) != 17):
        raise ValueError("incomplete comparison cannot claim full diagnostic completion")


def fits_allowance(spent, allowance):
    return number(spent, "spent seconds") + number(allowance, "full allowance") <= 34800


def can_admit(state, job):
    verify_state(state)
    return fits_allowance(state["spent_seconds"], job["budget_seconds"])


def validate_resolved(resolved):
    packet = resolved["packet"]
    if resolved.get("packet_sha256") != digest(packet):
        raise ValueError("child packet identity changed")
    verify_packet(packet)
    if resolved.get("request") != packet["request"]:
        raise ValueError("child request changed")
    jobs = [j for j in packet["request"]["jobs"] if j["compatibility_key"] == resolved["job"]["compatibility_key"]]
    if len(jobs) != 1 or jobs[0] != resolved["job"] or any(
            packet["request"]["tasks"][m].get("preflight_blockers") for m in jobs[0]["task_ids"]):
        raise ValueError("child job is blocked, split or changed")
    worker = resolved["worker"]
    if worker.get("device") != "1" or worker.get("lane_runtime") != lane_runtime(packet["request"]):
        raise ValueError("child GPU/runtime cohort changed")
    # Ignore only an external gate for this diagnostic dispatch. Never import a
    # checkpoint from failed/missing evidence or split the internal ring group.
    members = set(jobs[0]["task_ids"])
    for member in members:
        for dep in packet["request"]["tasks"][member].get("dependencies", []):
            if dep["task"] not in members and dep["kind"] != "gate":
                raise ValueError("external checkpoint/data prerequisite needs its own frozen protocol")
    if resolved.get("prerequisites") != {}:
        raise ValueError("this finite roster imports no external checkpoint")
    return packet


def guard_imports(source, *, execution_root=None, modules=None):
    snapshot = Path(execution_root or source["snapshot_path"]).resolve()
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        if name.split(".", 1)[0] not in {"experiments", "particlegan", "benchmarks", "lib"}:
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            expected = snapshot / name.replace(".", "/")
            locations = list(getattr(module, "__path__", []))
            if (locations != [str(expected)] or not any(n.startswith(name.replace(".", "/") + "/")
                                                       for n in source["files"])):
                raise ValueError("unbound namespace import: " + name)
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError("foreign execution import: " + name)
        relative = path.relative_to(snapshot).as_posix()
        if source["files"].get(relative) != file_hash(path):
            raise ValueError("unpinned execution import: " + name)


def child_environment(lease_fd, resolved):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("child requires the exact GPU1/deterministic/thread environment")
    verify_lease(lease_fd, resolved)
    os.environ["FORGE_LEASE_FD"] = str(lease_fd)


def verify_lease(lease_fd, resolved):
    path = Path(os.readlink(f"/proc/self/fd/{lease_fd}")).resolve()
    if path != Path(resolved["worker"]["lease_path"]).resolve():
        raise ValueError("admitted inherited descriptor is missing or foreign")
    os.fstat(lease_fd)
    supervision = read(path.parent / "supervisor-request.json")
    if (supervision.get("token") != resolved["worker"]["token"]
            or supervision.get("source") != resolved["packet"]["execution_source"]
            or lease_fd not in supervision.get("lease_fds", [])):
        raise ValueError("descriptor is not bound to this durable admitted source/token")


def stage(path, *, execute):
    resolved = read(path)
    packet = validate_resolved(resolved)
    guard_imports(packet["source"])
    if any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError("stage environment changed")
    verify_lease(int(os.environ["FORGE_LEASE_FD"]), resolved)
    if execute:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("numerical execution requires physical GPU1")
        import torch
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.2, 0)
        result = forge("runtime").execute(Path(path))
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("independent grading requires a clean CPU process")
        result = forge("evaluate").evaluate(Path(path))
        render_media(resolved, Path(path).parent, result)
        result = 0
    guard_imports(packet["source"])
    return result


def child(path, lease_fd, runner=None):
    resolved = read(path)
    validate_resolved(resolved)
    child_environment(lease_fd, resolved)
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    run = runner or subprocess.run
    execution = run(command + ["--execute", str(path)], pass_fds=(lease_fd,), check=False)
    if execution.returncode != 0:
        return execution.returncode
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = ""
    evaluation = run(command + ["--evaluate", str(path)], env=environment, pass_fds=(lease_fd,), check=False)
    if evaluation.returncode != 0:
        return evaluation.returncode
    grades = read(Path(path).parent / "graded-result.json")["grades"]
    return 0 if all(g.get("gate_status") in {"PASS", "FAIL"} for g in grades.values()) else 2


def point_pass(point, thresholds):
    operations = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b}
    return all(op in operations and operations[op](point[name], bound) for name, op, bound in thresholds)


def media_selection(task, evidence):
    artifact_root = Path(evidence["artifact_root"])
    files = evidence["artifact_manifest"]["files"]
    points = evidence.get("observations", evidence.get("dense", []))
    native = task["evaluation"]["kind"] == "native_accuracy"
    metadata = []
    if native:
        summaries = [n for n in files if n.endswith("/summary.json")]
        events = [n for n in files if n.endswith("/events.jsonl")]
        if len(summaries) != 1 or len(events) != 1:
            raise ValueError("native media needs its exact summary and primary observations")
        metadata = [artifact_root / summaries[0], artifact_root / events[0]]
        summary = read(metadata[0])
        points = [event for line in metadata[1].read_text().splitlines()
                  if (event := json.loads(line)).get("event") == "eval" and event.get("model") == "live"]
        if [p["step"] for p in points] != summary["eval_steps"]:
            raise ValueError("native recorded observation steps differ from summary")
    if not points:
        raise ValueError("media requires actual recorded observations")
    selected = sorted({round(i * (len(points) - 1) / min(8, len(points) - 1))
                       for i in range(min(9, len(points)))}) if len(points) > 1 else [0]
    paths = []
    for index in selected:
        step = points[index]["step"]
        matches = [name for name in files if name.endswith(f"step_{step:06d}.npz")
                   and ("/snapshots/" in name if native else name.startswith("observations/"))]
        if len(matches) != 1:
            raise ValueError("media needs one actual retained observation: " + str(step))
        paths.append(artifact_root / matches[0])
    return points, selected, paths, metadata


def render_media(resolved, directory, grading):
    """Use only actual retained arrays and recorded measurements; zero draws."""
    import numpy as np
    from benchmarks.toy_audit.api_run import render_gif
    raw = read(directory / "raw-result.json")
    for task_id in resolved["job"]["task_ids"]:
        task = resolved["request"]["tasks"][task_id]
        member = raw.get("task_results", {}).get(task_id, raw)
        evidence = member["evidence"]
        artifact_root = Path(evidence["artifact_root"])
        forge("artifacts").verify_artifacts(artifact_root, evidence["artifact_manifest"])
        files = evidence["artifact_manifest"]["files"]
        points, selected, paths, metadata = media_selection(task, evidence)
        native = task["evaluation"]["kind"] == "native_accuracy"
        records, input_pins = [], []
        for index, path in zip(selected, paths):
            point = points[index]; step = point["step"]
            input_pins.append(pin(path))
            with np.load(path, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("nonfinite retained display arrays")
            samples = arrays.get("samples", arrays.get("live", arrays.get("particles")))
            target = arrays.get("target", arrays.get("real"))
            if target is None:
                definition = task["execution"].get("host_definition", {})
                target = np.asarray(definition.get("means", []), dtype=np.float32)
                if not len(target) and definition.get("kind") == "spiral":
                    theta = np.linspace(0, 2 * np.pi * definition["turns"], 512)
                    radius = np.linspace(definition["radius_min"], definition["radius_max"], 512)
                    target = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
            if samples is None or target is None or not len(target):
                raise ValueError("actual outputs/declared reference are missing")
            kind = "image" if samples.ndim == 4 else "scatter"
            if samples.ndim == 2 and samples.shape[1] == 1:
                samples = np.column_stack((samples[:, 0], np.zeros(len(samples))))
                target = np.column_stack((target[:, 0], np.zeros(len(target))))
            metrics = {k: v for k, v in point.items() if k != "step" and type(v) in (int, float)}
            thresholds = task["evaluation"].get("thresholds", [["modes", ">=", 8], ["hq", ">=", .9]])
            instant = point_pass(point, thresholds) if not native else point["accuracy"].get("passed")
            if type(instant) is not bool:
                raise ValueError("display needs the actual recorded numerical decision")
            caption = ("Actual retained public policy outputs. Reference is the retained target or declared mode means/centerline. "
                       "Diagnostic only; ordinary qualification and defaults receive no credit.")
            views = [{"kind": kind, "title": task_id, "target": target, "samples": samples,
                      "caption": caption, "xlabel": "x", "ylabel": "y"}]
            if "critic_gradient" in arrays:
                gradients = np.abs(arrays["critic_gradient"]).reshape(-1)
                views.append({"kind": "bar", "title": "Original measured critic input gradients",
                              "target": np.ones_like(gradients), "samples": gradients,
                              "caption": "Reference line 1; the original gate scores median absolute gradient, not every bar."})
            if native:
                metrics = {k: v for k, v in point["metrics"].items() if type(v) in (int, float)}
                caption = "Actual retained selected-policy clean draw. Grade uses original 20k checks plus independent 100k accuracy holdout."
                views[0]["caption"] = caption + " Frame badge is the recorded 20k decision; plotted 4096 samples provide no extra credit."
            records.append({"step": step, "metrics": metrics, "passed": instant, "views": views})
        status = grading["grades"][task_id]["gate_status"]
        media_path = directory / (task_id + "-goal.gif")
        render_gif({"id": task_id, "goal": "Frozen Atlas policy diagnostic: " + task_id,
                    "default_steps": task["execution"]["steps"]}, records, media_path,
                   full_budget=True, requested_steps=task["execution"]["steps"],
                   final_verdict=status + " (diagnostic; no ordinary credit)")
        input_pins.extend(pin(path) for path in metadata)
        write(directory / (task_id + "-media.json"), {"schema": SCHEMA + "_media", "task": task_id,
              "source_digest": resolved["request"]["source"]["digest"], "original_gate": status,
              "renderer_sha256": file_hash(Path(__file__)),
              "qualification_input": False, "draws": 0, "optimizer_updates": 0,
              "actual_steps": [r["step"] for r in records], "inputs": input_pins, "gif": pin(media_path)})


def certified_outcome(resolved_path, grader=None, policy_guard=None):
    resolved = read(resolved_path)
    packet = validate_resolved(resolved)
    directory = Path(resolved_path).parent
    raw_path, grade_path = directory / "raw-result.json", directory / "graded-result.json"
    raw, grading = read(raw_path), read(grade_path)
    job = resolved["job"]
    if (grading.get("raw_hash") != digest(raw) or grading.get("source_digest") != packet["source"]["digest"]
            or set(grading.get("grades", {})) != set(job["task_ids"])):
        raise ValueError("raw/independent grade/source/task identity changed")
    grader = grader or forge("views").grade_result
    policy_guard = policy_guard or forge("views")._policy_guards
    statuses, media = {}, {}
    for name in job["task_ids"]:
        task = packet["request"]["tasks"][name]
        member = raw.get("task_results", {}).get(name, raw)
        evidence = member.get("evidence", {})
        root = Path(evidence["artifact_root"]).resolve()
        if not root.is_relative_to(directory.resolve()):
            raise ValueError("foreign case/candidate artifacts")
        forge("artifacts").verify_artifacts(root, evidence["artifact_manifest"])
        guard = policy_guard(task, evidence)
        if guard is not None:
            raise ValueError("invalid/incomplete policy execution cannot count as numerical FAIL: " + canonical(guard))
        expected = grader(task, member)
        if expected != grading["grades"][name]:
            raise ValueError("independent numerical grade changed")
        status = expected.get("gate_status")
        if status not in {"PASS", "FAIL"}:
            raise ValueError("non-numerical outcome halts the diagnostic batch")
        statuses[name] = status
        receipt = read(directory / (name + "-media.json"))
        points, selected, paths, metadata = media_selection(task, evidence)
        if (receipt.get("schema") != SCHEMA + "_media" or receipt.get("task") != name
                or receipt.get("source_digest") != packet["source"]["digest"]
                or receipt.get("original_gate") != status or receipt.get("qualification_input") is not False
                or receipt.get("draws") != 0 or receipt.get("optimizer_updates") != 0
                or receipt.get("renderer_sha256") != packet["source"]["files"][SELF]
                or receipt.get("actual_steps") != [points[i]["step"] for i in selected]
                or receipt.get("inputs") != [pin(p) for p in paths + metadata]
                or receipt.get("gif", {}).get("path") != str((directory / (name + "-goal.gif")).resolve())):
            raise ValueError("media verdict/source/scope changed")
        check_pin(receipt["gif"])
        from PIL import Image
        with Image.open(receipt["gif"]["path"]) as gif:
            if gif.n_frames != len(selected):
                raise ValueError("actual GIF frames differ from retained observation count")
        media[name] = {"receipt": pin(directory / (name + "-media.json")), "gif": receipt["gif"]}
    return {"statuses": statuses, "resolved": pin(resolved_path), "raw": pin(raw_path), "grading": pin(grade_path),
            "media": media, "qualification_input": False}


def save_state(output, state):
    state["measured_paid_seconds"] = sum(r["paid_wall_seconds"] for r in state["jobs"])
    state["unmeasured_interrupt_reserved_seconds"] = sum(r["unmeasured_interrupt_reserved_seconds"] for r in state["jobs"])
    state["spent_seconds"] = state["measured_paid_seconds"] + state["unmeasured_interrupt_reserved_seconds"]
    verify_state(state)
    write(Path(output) / "study.json", state)
    lines = ["# Fixed Atlas GPU diagnostics", "", "Diagnostic evidence only; ordinary tiers/calibration/defaults/speed are not qualified.", "",
             f"Required 26; executable 18; structurally BLOCKED 8. Measured paid {state['measured_paid_seconds']:.6f}s; "
             f"conservative reserve {state['unmeasured_interrupt_reserved_seconds']:.6f}s; charged {state['spent_seconds']:.6f}/34800s.", "",
             "| Actual task | Tier | Diagnostic gate | Ordinary credit | Goal GIF |", "| --- | ---: | --- | --- | --- |"]
    for row in state["spec"]["cases"]:
        slot = state["slots"][row["id"]]
        goal = ""
        for job in state["jobs"]:
            if row["id"] in job["task_ids"] and "outcome" in job:
                directory = Path(job["outcome"]["resolved"]["path"]).parent
                goal = f"[actual goal]({directory.relative_to(output).as_posix()}/{row['id']}-goal.gif)"
        lines.append(f"| {row['id']} | {row['tier']} | {slot['diagnostic_status']} | none | {goal} |")
    (Path(output) / "README.md").write_text("\n".join(lines) + "\n")


def coordinator_for(queue_root):
    return forge("policy_execution").PolicyCoordinator(Path(queue_root).resolve(), report_root=ROOT / "reports/forge")


def run(output, queue_root):
    output = Path(output).resolve()
    packet = read(preparation_path(output)); verify_packet(packet)
    runtime = lane_runtime(packet["request"])
    packet["lane_runtime"] = runtime
    state = read(output / "study.json") if (output / "study.json").exists() else initial_state(packet)
    for key in ("spec", "request", "source", "execution_source", "case_definitions", "spec_sha256", "family_paid_budget_seconds"):
        if state.get(key) != packet.get(key):
            raise ValueError("resume source/contract/quota identity changed")
    verify_state(state)
    for row in state["jobs"]:
        if "outcome" in row and certified_outcome(check_pin(row["outcome"]["resolved"])) != row["outcome"]:
            raise ValueError("retained terminal result changed")
        if row.get("status") != "COMPLETE":
            return state  # Never retry an interrupted or invalid attempt.
    gpu_readiness(packet["spec"])
    coordinator = coordinator_for(queue_root)
    key, canonical_output = coordinator.register(packet, output, "atlas", runtime)
    if canonical_output != output:
        raise ValueError("compatible diagnostic already has a canonical archive; use that output")
    state.update(executed_family="atlas", lane_runtime=runtime, coordinator=packet.get("coordinator"))
    save_state(output, state)
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:
            raise RuntimeError("diagnostic study already has an owner")
        done = {r["compatibility_key"] for r in state["jobs"]}
        order = {r["id"]: i for i, r in enumerate(packet["spec"]["cases"])}
        jobs = sorted(packet["request"]["jobs"], key=lambda j: min(order[m] for m in j["task_ids"]))
        for job in jobs:
            if job["compatibility_key"] in done or any(state["slots"][m]["parent_id"] in BLOCKED for m in job["task_ids"]):
                continue
            if not can_admit(state, job):
                state.update(status="INCOMPLETE", stop_reason="next complete job allowance does not fit")
                break
            telemetry = gpu_readiness(packet["spec"])
            row = {"id": job["task_id"], "timeout_seconds": job["budget_seconds"]}
            trial = {"family": "atlas", "recipe_overrides": OVERRIDES}
            attempt = coordinator.attempt_key(packet, trial, row)
            with coordinator.admit(attempt, packet, row, "cuda:0") as (admission, lease):
                if admission["status"] == "busy":
                    state.update(status="INCOMPLETE", stop_reason=admission["reason"])
                    break
                directory = output / "attempts" / job["execution_group"]
                directory.mkdir(parents=True, exist_ok=True)
                resolved_path = directory / "resolved.json"
                token = admission["token"]
                if admission["status"] == "running" and lease is not None:
                    gpu_readiness(packet["spec"])
                    resolved = {"schema_version": 1, "packet": packet, "packet_sha256": digest(packet),
                                "request": packet["request"], "job": job, "prerequisites": {},
                                "worker": {"device": "1", "token": token, "attempt": attempt,
                                           "lane_runtime": runtime, "lease_path": admission["lease_path"]}}
                    write(resolved_path, resolved)
                    command = [sys.executable, "-u", str(Path(packet["source"]["snapshot_path"]) / SELF),
                               "--child", str(resolved_path), "--lease-fd", str(lease.fileno())]
                    launch_error = None
                    try:
                        coordinator.launch(command, packet, directory / "run.log", (study_lease, lease), job["budget_seconds"])
                    except (Exception, KeyboardInterrupt) as error:
                        launch_error = {"token": token, "source_digest": packet["source"]["digest"],
                                        "type": type(error).__name__, "message": str(error),
                                        "paid_wall_seconds": number(getattr(error, "paid_wall_seconds", 0.), "parent paid")}
                        write(directory / "launch-error.json", launch_error)
                else:
                    launch_error = read(directory / "launch-error.json") if (directory / "launch-error.json").exists() else None
                terminal_path = Path(admission["lease_path"]).parent / "supervisor-terminal.json"
                terminal = read(terminal_path) if terminal_path.exists() else None
                if terminal is not None and terminal.get("token") != token:
                    raise ValueError("foreign durable supervisor terminal")
                paid = terminal.get("paid_wall_seconds", 0.) if terminal else (launch_error or {}).get("paid_wall_seconds", 0.)
                result = {"compatibility_key": job["compatibility_key"], "task_ids": job["task_ids"],
                          "attempt_key": attempt, "token": token, "status": "INCOMPLETE", "admission": telemetry,
                          **charge(paid, terminal, job["budget_seconds"])}
                if terminal is not None:
                    result["terminal"] = pin(terminal_path)
                elif launch_error is not None:
                    result["launch_error"] = pin(directory / "launch-error.json")
                if terminal and terminal["attempt_status"] == "completed" and terminal.get("child_returncode") == 0:
                    try:
                        result["outcome"] = certified_outcome(resolved_path)
                        result["status"] = "COMPLETE"
                    except Exception as error:
                        result.update(status="INVALID", reason=f"{type(error).__name__}: {error}")
                elif terminal and terminal["attempt_status"] == "completed":
                    result.update(status="INVALID", reason="runtime/grading/media child did not produce valid complete evidence")
                if admission["status"] in {"running", "awaiting_certification"}:
                    coordinator.complete(attempt, result)
                elif admission.get("charged_seconds") != result["charged_seconds"]:
                    raise ValueError("recovered central charge differs from the retained attempt")
                state["jobs"].append(result)
                for member in job["task_ids"]:
                    state["slots"][member]["diagnostic_status"] = (
                        result["outcome"]["statuses"][member] if result["status"] == "COMPLETE" else result["status"])
                state["status"] = "RUNNING" if result["status"] == "COMPLETE" else result["status"]
                save_state(output, state)
                if result["status"] != "COMPLETE":
                    return state
        else:
            state["status"] = "COMPLETE_DIAGNOSTIC"
        save_state(output, state)
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / DIRECTORY / "protocol.json")
    p.add_argument("--output", type=Path)
    p.add_argument("--queue-root", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--gpus", choices=("1",))
    stages = p.add_mutually_exclusive_group()
    stages.add_argument("--child", type=Path)
    stages.add_argument("--execute", type=Path)
    stages.add_argument("--evaluate", type=Path)
    p.add_argument("--lease-fd", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.child:
        if args.lease_fd is None:
            raise ValueError("child needs the admitted inherited lease descriptor")
        return child(args.child, args.lease_fd)
    if args.execute or args.evaluate:
        return stage(args.execute or args.evaluate, execute=args.execute is not None)
    if args.output is None:
        raise ValueError("declare a new external output")
    configure_environment(cuda=not args.prepare_only)
    if args.prepare_only:
        packet = prepare(args.spec, args.output)
        print(canonical({"status": "PREPARED", "spec_sha256": packet["spec_sha256"],
                         "source_digest": packet["source"]["digest"], "qualification_input": False}), flush=True)
        return 0
    if args.gpus != "1" or args.queue_root is None:
        raise ValueError("run requires physical GPU1 and the existing shared queue")
    state = run(args.output, args.queue_root)
    print(canonical({"status": state["status"], "paid_seconds": state["measured_paid_seconds"],
                     "reserved_seconds": state["unmeasured_interrupt_reserved_seconds"], "qualification_input": False}), flush=True)
    return 0 if state["status"] == "COMPLETE_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
