"""Draw-free publication of immutable terminal named-host GPU diagnostics.

The trusted input-card SHA is an explicit root attestation of this publication
cut. The unchanged GPU producer and independent evaluator establish numerical
grades; this module checks retained identities and projects their decisions.
It never imports Forge, Torch, NumPy, a producer, or a scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil

SCHEMA = "particlegan_atlas_named_gpu_diagnostics_publication_v3"
INPUT_SCHEMA = SCHEMA + "_inputs"
RUN_SCHEMA = "particlegan_atlas_named_gpu_diagnostics_v1"
SOURCE = {"origin_commit": "ff94453b45e02fd451b21c26d5991f6ed435c292",
          "digest": "87fcd4e28bdcd9f347379b7db1fbcf0af3edcc0fbd5e97e4a98d9a49f8987a87"}
SPEC_SHA256 = "f2cf72408406c5b8ab280b586fe28010ad9bc21cf45a4f4ea94a0d7a82d65790"
SELF = "reports/forge/atlas-named-gpu-diagnostics-v1/run_diagnostics.py"
PRIOR = "reports/forge/atlas-named-gpu-diagnostics-invalid-20261003/summary.json"
PRIOR_SHA256 = "7fee64a49aa56b2cec25ffcf4861eccd91ba6886fd33f9e95565dce1b701cfd7"
DEBITS = {"0": 12.873334385920316, "1": 12.449620655039325}
FAMILIES = {
    "atlas_conditional": {"gpu": "0", "cap": 7200, "cohort": "conditional_policy_selected_cloud_v1",
                          "parents": ["trajectory", "residual_student", "unipolar", "mid_scale_identity"]},
    "atlas_ae_routed": {"gpu": "0", "cap": 300, "cohort": "ae_routed_policy_v1", "parents": ["ae_gan_hold"]},
    "atlas_routed": {"gpu": "1", "cap": 300, "cohort": "routed_policy_selected_cloud_v1", "parents": ["unused_token_hold"]},
    "atlas_multibank": {"gpu": "1", "cap": 1800, "cohort": "multibank_policy_v1", "parents": ["cover_leftover"]},
    "atlas_word_joint_min11": {"gpu": "1", "cap": 900, "cohort": "word_joint_policy_min11_v1",
                             "parents": ["five_word_joint_acquisition"]},
}
PARENTS = ["two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition", "five_word_joint_acquisition",
           "trajectory", "residual_student", "unipolar", "cover_leftover", "mid_scale_identity", "mode_hold",
           "vector_two_broad", "vector_unequal_mass", "vector_unequal_width", "vector_anisotropic", "vector_overlap",
           "vector_spiral", "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2", "grid100", "rotated100",
           "staggered100", "ring_hold", "ring_extension"]
HOSTS = {"trajectory": (400, 1800), "residual_student": (400, 1800), "unipolar": (400, 1800),
         "mid_scale_identity": (800, 1800), "ae_gan_hold": (250, 300), "unused_token_hold": (200, 300),
         "cover_leftover": (800, 1800), "five_word_joint_acquisition": (20001, 900)}
FLAGS = ("qualification_input", "ordinary_tier_credit", "calibration_credit", "default_adoption",
         "cross_cohort_pooling", "speed_ranking")
TERMINAL = {"COMPLETE_DIAGNOSTIC", "INCOMPLETE", "INVALID", "BUDGET_EXCEEDED"}
PRIVATE_KEYS = {"token", "credential", "credentials", "password", "secret", "authorization",
                "access_token", "refresh_token", "api_key", "lease_fd", "lease_path"}
QUESTIONS = {
    "trajectory": "Recover each original identity's conditioned eight-point trajectory.",
    "residual_student": "Preserve each paired trajectory, successful action and unused padding.",
    "unipolar": "Preserve conditioned concept coverage and neutral identity without off-caption leakage.",
    "mid_scale_identity": "Recover both signed concept directions and magnitudes while preserving identity at zero and intermediate scale.",
    "unused_token_hold": "Separate the original unused-token nuisance while preserving concept geometry under complete routed ownership.",
    "cover_leftover": "Keep content and identity while separating the original odd/even residual poles and controlling leakage.",
    "ae_gan_hold": "Reconstruct the original hard-AE inputs and preserve hold under the declared fixed-width routed MoG law.",
    "five_word_joint_acquisition": "Acquire all five canonical words and recover their paired reconstructions using eleven rows, free E and the declared joint-code law.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def safe_relative(value):
    p = PurePosixPath(value)
    if not isinstance(value, str) or not value or p.is_absolute() or ".." in p.parts or "\\" in value or p.as_posix() != value or value == ".":
        raise ValueError("unsafe relative artifact/source path")
    return value


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    return float(value)


def close(left, right):
    if not math.isclose(number(left), number(right), rel_tol=0, abs_tol=1e-8):
        raise ValueError("paid/reserved/charged arithmetic changed")


def public(value, secrets=()):
    if isinstance(value, dict):
        if any(str(key).lower() in PRIVATE_KEYS for key in value):
            raise ValueError("private field in public projection")
        for item in value.values():
            public(item, secrets)
    elif isinstance(value, (list, tuple)):
        for item in value:
            public(item, secrets)
    elif isinstance(value, str) and any(secret and secret in value for secret in secrets):
        raise ValueError("plaintext internal token in public projection")
    return value


class Inputs:
    def __init__(self):
        self.files = {}
        self.secrets = set()

    def check(self, pin):
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "bytes"}:
            raise ValueError("complete file identity required")
        path = Path(pin["path"])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or any(part.is_symlink() for part in path.parents) or binding(path) != pin):
            raise ValueError("missing, unsafe or changed pinned input")
        key = str(path.resolve())
        if key in self.files and self.files[key] != pin:
            raise ValueError("input changed between reads")
        self.files[key] = deepcopy(pin)
        return path

    def json(self, pin):
        return read(self.check(pin))

    def path(self, path):
        return self.check(binding(path))

    def recheck(self):
        for item in list(self.files.values()):
            self.check(item)


def packet_identity(packet, inputs):
    if packet.get("schema") != RUN_SCHEMA or packet["source"] != packet["execution_source"]:
        raise ValueError("prepared cohort/source differs")
    source = packet["source"]
    if any(source.get(k) != v for k, v in SOURCE.items()) or digest(source["files"]) != source["digest"]:
        raise ValueError("wrong scientific source cohort")
    snapshot = Path(source["snapshot_path"])
    manifest = read(inputs.path(snapshot / "forge-source.json"))
    if manifest != {k: v for k, v in source.items() if k != "snapshot_path"}:
        raise ValueError("frozen snapshot manifest differs")
    for relative, sha in source["files"].items():
        path = snapshot / safe_relative(relative)
        if file_sha(inputs.path(path)) != sha:
            raise ValueError("frozen source file drift")
    spec = deepcopy(packet["spec"])
    readiness = inputs.json(spec.pop("representation_card"))
    if packet.get("spec_sha256") != SPEC_SHA256 or digest(spec) != SPEC_SHA256:
        raise ValueError("fixed protocol/gate/task hashes changed")
    if (spec.get("id") != "atlas-named-hosts-current-gpu-diagnostics-v3" or spec.get("seed") != 0
            or spec.get("recipe_overrides") != {"lr": .0053125, "prior_lr_mult": 1.5}
            or spec.get("lane_cap_seconds") != {"0": 7500, "1": 3000}
            or spec.get("diagnostic_cap_seconds") != 10500 or spec.get("frames") != 9
            or spec.get("export_grace_seconds") != 0 or spec.get("required_slots_per_family") != 26
            or spec.get("executable_slots") != 8 or spec.get("executable_jobs") != 8
            or spec.get("families") != list(FAMILIES) or any(spec.get(flag) is not False for flag in FLAGS)):
        raise ValueError("fixed family/recipe/denominator/nonqualification contract changed")
    if (readiness != packet.get("capacity_preflight") or readiness.get("source_digest") != source["digest"]
            or readiness.get("contract_sha256") != SPEC_SHA256 or readiness.get("model_capacity_proved") is not False
            or readiness.get("learned_quality_proved") is not False or readiness.get("optimizer_updates") != 0
            or readiness.get("qualification_input") is not False):
        raise ValueError("structural readiness is not capacity or learning proof")
    if set(packet["requests"]) != set(FAMILIES) or packet["family_paid_budget_seconds"] != {f: i["cap"] for f, i in FAMILIES.items()}:
        raise ValueError("five family denominators/caps required")
    expected_cases = [(f, p) for f, i in FAMILIES.items() for p in i["parents"]]
    if [(r["family"], r["parent_id"]) for r in spec["cases"]] != expected_cases:
        raise ValueError("all eight adapted cases required in fixed order")
    for family, info in FAMILIES.items():
        request = packet["requests"][family]
        mapping = {p: p + "_" + info["cohort"] for p in info["parents"]}
        ids = [mapping.get(p, p) for p in PARENTS]
        view = request["view"]
        candidate = request["candidate"]
        if (set(request["tasks"]) != set(ids) or [a["task"] for a in view["assignments"]] != ids
                or Counter(a["qualification_tier"] for a in view["assignments"]) != {1: 5, 2: 19, 3: 2}
                or any(a["importance"] != "required" for a in view["assignments"])
                or view.get("revision") != 4 or view.get("policy_family") != family
                or candidate.get("trainer_family") != family or candidate.get("task_cohort") != info["cohort"]
                or candidate.get("recipe_preset") != "atlas" or candidate.get("recipe_overrides") != spec["recipe_overrides"]
                or request["source"] != source or request.get("qualification_reuse") is not False
                or request["protocol"].get("seed") != 0 or request["runtime"] != packet["runtime_contract"]
                or request.get("execution_backend") != "cuda"):
            raise ValueError("family-specific full 5/19/2 view/candidate/runtime changed")
        for row in (r for r in spec["cases"] if r["family"] == family):
            task = request["tasks"][row["id"]]
            canonical_task = inputs.json(binding(snapshot / safe_relative(row["definition"])))
            compiled = {k: v for k, v in task.items() if k not in {"field_ownership", "preflight_blockers"}}
            if (canonical_task != compiled or source["files"].get(row["definition"]) != row["sha256"]
                    or digest(task["execution"]) != row["execution_sha256"]
                    or digest(task["evaluation"]) != row["evaluation_sha256"]
                    or (row["steps"], row["timeout_seconds"]) != HOSTS[row["parent_id"]]
                    or task["evaluation"].get("observations") != 24 or task["evaluation"].get("minimum_stable_checks") != 5):
                raise ValueError("canonical task/evaluator/host horizon differs")
    return spec


def engineering(packet, inputs):
    source = packet["source"]
    reference = packet["spec"]["engineering_carryover"]
    prior = inputs.json(binding(Path(source["snapshot_path"]) / PRIOR))
    if (source["files"].get(PRIOR) != PRIOR_SHA256 or reference != packet["engineering_carryover"].get("reference")
            or reference["summary"]["sha256"] != PRIOR_SHA256 or reference["paid_seconds_by_lane"] != DEBITS
            or reference.get("qualification_input") is not False or reference.get("reserved_seconds") != 0
            or prior.get("status") != "INVALID" or prior["counts"]["completed_updates"] != 0
            or prior["counts"]["all_planned_family_cells"] != {"INVALID": 2, "NOT_RUN": 128}
            or prior["source"]["origin_commit"] != reference["source"]["commit"]
            or prior["source"]["digest"] != reference["source"]["digest"]):
        raise ValueError("prior startup costs cannot reset, provide outcomes, or become current source")
    attempts = packet["engineering_carryover"].get("attempts", [])
    if len(attempts) != 2:
        raise ValueError("exactly two prior engineering debits required")
    for item, old in zip(attempts, prior["attempts"]):
        gpu = item["physical_gpu"]
        if item["status"] != "INVALID" or item["paid_seconds"] != DEBITS[gpu] or item["reserved_seconds"] != 0:
            raise ValueError("prior engineering debit changed")
        artifacts = item["artifacts"]
        for pin in artifacts.values():
            inputs.check(pin)
        terminal = inputs.json(artifacts["terminal"])
        oldstudy = inputs.json(artifacts["study"])
        token = terminal.get("token")
        inputs.secrets.add(token)
        if (not isinstance(token, str) or hashlib.sha256(token.encode()).hexdigest() != old["token_sha256"]
                or terminal.get("attempt_status") != "completed" or terminal.get("child_returncode") != 1
                or terminal.get("paid_wall_seconds") != DEBITS[gpu] or oldstudy.get("status") != "INVALID"
                or oldstudy.get("spent_seconds") != DEBITS[gpu]):
            raise ValueError("old durable cost evidence differs")
    return {"reference": public(deepcopy(reference)), "paid_seconds_by_lane": DEBITS,
            "paid_seconds": sum(DEBITS.values()), "reserved_seconds": 0., "outcomes_reused": False,
            "source": reference["source"], "status": "INVALID", "completed_updates": 0}


def jobs_for(packet, family):
    request = packet["requests"][family]
    return [next(j for j in request["jobs"] if j["task_id"] == r["id"])
            for r in packet["spec"]["cases"] if r["family"] == family]


def family_runtime(packet, family):
    request = packet["requests"][family]
    compute = request["compute_profiles"]["cuda"]
    return {**request["runtime"], "device": "cuda:0", "physical_gpu": FAMILIES[family]["gpu"],
            "cuda_device_model": "NVIDIA RTX A6000", "compute": compute, "torch_threads": 1}


def artifact_manifest(inputs, root, manifest):
    root = Path(root)
    files = manifest.get("files", {})
    if (manifest.get("schema_version") != 1 or not files or manifest.get("sha256") != digest(files)
            or manifest.get("file_count") != len(files)
            or manifest.get("total_bytes") != sum(v["size"] for v in files.values())):
        raise ValueError("complete artifact manifest required")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("artifact file set changed")
    for name, pin in files.items():
        inputs.check({"path": str(root / safe_relative(name)), "sha256": pin["sha256"], "bytes": pin["size"]})


def resolved_identity(inputs, pin, packet, study, row, job):
    value = inputs.json(pin)
    token = row["token"]
    inputs.secrets.add(token)
    expected_packet = deepcopy(packet)
    expected_packet.update(family=study["family"], request=study["request"], lane_runtime=study["lane_runtime"],
                           lane_predecessors=study["lane_predecessors"], coordinator=study.get("coordinator"),
                           executed_family=study["family"], spent_seconds=0.)
    if (value.get("packet_sha256") != digest(value["packet"]) or value["packet"] != expected_packet
            or value.get("request") != study["request"] or value.get("job") != job
            or value.get("worker", {}).get("token") != token
            or value["worker"].get("lane_runtime") != study["lane_runtime"]
            or value["worker"].get("attempt") != row["attempt_key"]):
        raise ValueError("foreign task/candidate/Recipe/source/runtime attempt")
    return value


def complete(inputs, packet, study, row, job):
    outcome = row["outcome"]; family = study["family"]; name = job["task_id"]
    resolved = resolved_identity(inputs, outcome["resolved"], packet, study, row, job)
    directory = inputs.check(outcome["resolved"]).parent
    raw = inputs.json(outcome["raw"]); grading = inputs.json(outcome["grading"])
    task = study["request"]["tasks"][name]
    if (grading.get("raw_hash") != digest(raw) or grading.get("source_digest") != packet["source"]["digest"]
            or set(grading.get("grades", {})) != {name} or outcome.get("qualification_input") is not False
            or set(outcome.get("media", {})) != {name}):
        raise ValueError("raw/independent grade/source/media join changed")
    grade = grading["grades"][name]; status = grade.get("gate_status")
    if status not in {"PASS", "FAIL"} or outcome.get("statuses") != {name: status}:
        raise ValueError("scientific grade cannot be inferred from a status stamp")
    evidence = raw["evidence"]; observations = evidence["observations"]
    steps = task["execution"]["steps"]
    cadence = [math.ceil(i * steps / 24) for i in range(1, 25)]
    if (raw.get("task_id") != name or raw.get("execution_path") != "public_components" or raw.get("device") != "cuda:0"
            or raw["cost"].get("completed_steps") != steps or evidence.get("live") != observations[-1]
            or [p["step"] for p in observations] != cadence or evidence.get("scoring_weights") != "state_selected"
            or len(evidence.get("policy_observations", [])) != 24 or len(evidence.get("policy_purity", [])) != 24
            or raw["applied"]["recipe"].get("lr") != .0053125
            or raw["applied"]["recipe"].get("prior_lr_mult") != 1.5):
        raise ValueError("incomplete or wrong public owner/Recipe/cadence evidence")
    for point in observations:
        for metric, _, _ in task["evaluation"]["thresholds"]:
            if type(point.get(metric)) not in (float, int) or not math.isfinite(point[metric]):
                raise ValueError("nonfinite/missing original gate metric")
    root = Path(evidence["artifact_root"])
    if not root.resolve().is_relative_to(directory.resolve()):
        raise ValueError("foreign candidate artifacts")
    artifact_manifest(inputs, root, evidence["artifact_manifest"])
    checkpoint = evidence["checkpoint"]
    if file_sha(inputs.path(root / safe_relative(checkpoint["path"]))) != checkpoint["sha256"]:
        raise ValueError("complete checkpoint hash differs")
    media = outcome["media"][name]
    receipt = inputs.json(media["receipt"]); gif = inputs.check(media["gif"])
    selected = [round(i * (len(observations) - 1) / 8) for i in range(9)]
    expected_steps = [cadence[i] for i in selected]
    if (receipt.get("schema") != RUN_SCHEMA + "_media" or receipt.get("task") != name
            or receipt.get("family") != family or receipt.get("source_digest") != packet["source"]["digest"]
            or receipt.get("renderer_sha256") != packet["source"]["files"][SELF]
            or receipt.get("original_gate") != status or receipt.get("qualification_input") is not False
            or receipt.get("draws") != 0 or receipt.get("optimizer_updates") != 0
            or receipt.get("actual_steps") != expected_steps or receipt.get("gif") != media["gif"]
            or len(receipt.get("inputs", [])) != 9):
        raise ValueError("actual goal media law/source/verdict/cadence changed")
    for pin, step in zip(receipt["inputs"], expected_steps):
        expected_path = root / "observations" / f"step_{step:06d}.npz"
        if inputs.check(pin).resolve() != expected_path.resolve():
            raise ValueError("media is not from this task's recorded observations")
    from PIL import Image
    with Image.open(gif) as image:
        if image.n_frames != 9:
            raise ValueError("actual GIF frame count changed")
    applied = raw["applied"]
    public_applied = {k: applied[k] for k in ("execution_path", "family", "task_cohort", "host", "recipe", "rng",
                        "initialization", "policy_lifecycle", "table_ownership", "optimizer_groups", "resource_adaptation",
                        "prior", "prior_mechanisms", "serving_law", "original_objective", "execution_phase", "actual_device",
                        "all_original_training_contexts_preserved", "heldout_generalization_claim", "independent_atlas_qualification") if k in applied}
    return {"task_id": name, "parent_id": task["policy_parent"]["id"], "status": status,
            "grade": public(deepcopy(grade), inputs.secrets), "grading_input": outcome["grading"],
            "completed_steps": steps, "observations": 24, "primary_steps": cadence,
            "terminal_hold_scope": "Original last five observations; no acquisition-speed or first-window reclassification.",
            "last_metrics": public(observations[-1], inputs.secrets), "applied": public(public_applied, inputs.secrets),
            "owner_receipt_sha256": digest(evidence["policy_controls"]),
            "owners": public(evidence["guards"].get("optimizer_updates", {}), inputs.secrets),
            "checkpoint": deepcopy(checkpoint), "artifact_manifest_sha256": evidence["artifact_manifest"]["sha256"],
            "observer_receipts_sha256": digest({k: evidence[k] for k in ("policy_observations", "policy_purity", "rng_audits")}),
            "raw_input": outcome["raw"], "media_input": media, "media_receipt": public(receipt, inputs.secrets),
            "goal": QUESTIONS[task["policy_parent"]["id"]], "qualification_input": False}


def costs(inputs, row, job, source, auxiliary):
    token = row.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("owned attempt identity required")
    inputs.secrets.add(token)
    paid = 0.; complete_terminal = False; terminal_status = "UNAVAILABLE"; child_returncode = None
    if "terminal" in row:
        terminal = inputs.json(row["terminal"])
        if terminal.get("token") != token:
            raise ValueError("foreign terminal attempt")
        paid = number(terminal["paid_wall_seconds"]); terminal_status = terminal["attempt_status"]
        complete_terminal = terminal_status == "completed"; child_returncode = terminal.get("child_returncode")
        supervisor = inputs.json(auxiliary["supervisor"])
        command = supervisor.get("command", [])
        if (supervisor.get("token") != token or supervisor.get("source") != source
                or len(command) != 7 or command[1:4] != ["-u", str(Path(source["snapshot_path"]) / SELF), "--child"]
                or command[4] != auxiliary["resolved"]["path"] or command[5] != "--lease-fd"
                or not str(command[6]).isdigit()):
            raise ValueError("durable supervisor source/actual child changed")
    elif "launch_error" in row:
        error = inputs.json(row["launch_error"])
        if error.get("token") != token or error.get("source_digest") != source["digest"]:
            raise ValueError("foreign interruption measurement")
        paid = number(error["paid_wall_seconds"])
    reserved = 0. if complete_terminal else max(0., job["budget_seconds"] - paid)
    close(row.get("paid_wall_seconds"), paid)
    close(row.get("unmeasured_interrupt_reserved_seconds"), reserved)
    close(row.get("charged_seconds"), paid + reserved)
    if row.get("status") == "COMPLETE" and (not complete_terminal or child_returncode != 0):
        raise ValueError("complete scientific outcome needs a successful completed supervisor")
    return {"paid_seconds": paid, "reserved_seconds": reserved, "charged_seconds": paid + reserved,
            "terminal_status": terminal_status, "child_returncode": child_returncode,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest()}


def initial_slots(packet, family):
    request = packet["requests"][family]
    active = {r["id"] for r in packet["spec"]["cases"] if r["family"] == family}
    return {a["task"]: {"tier": a["qualification_tier"], "in_diagnostic_batch": a["task"] in active,
             "diagnostic_status": "BLOCKED" if a["task"] in active and request["tasks"][a["task"]].get("preflight_blockers") else "NOT_RUN",
             "preflight_blockers": request["tasks"][a["task"]].get("preflight_blockers", []), "qualification_input": False}
            for a in request["view"]["assignments"]}


def project_family(inputs, packet, family, item, studies):
    info = FAMILIES[family]; request = packet["requests"][family]
    result = {"family": family, "cohort": info["cohort"], "physical_gpu": info["gpu"], "status": "NOT_RUN",
              "view": {k: request["view"][k] for k in ("id", "revision", "goal", "parent_view_fingerprint", "cohort_fingerprint")},
              "candidate_revision": request["candidate_revision"], "runtime": family_runtime(packet, family),
              "required_slots": 26, "tiers": {"1": 5, "2": 19, "3": 2}, "family_cap_seconds": info["cap"],
              "paid_seconds": 0., "reserved_seconds": 0., "charged_seconds": 0., "attempts": [],
              "slots": initial_slots(packet, family), **{flag: False for flag in FLAGS}, "ordinary_qualified_tier": 0}
    if item is None:
        return result
    study = inputs.json(item["study"])
    if study.get("status") not in TERMINAL:
        raise ValueError("RUNNING/PREPARED studies need an immutable terminal boundary before export")
    for key in packet:
        if study.get(key) != packet[key]:
            raise ValueError("study source/request/Recipe/quota/readiness identity changed")
    if (study.get("family") != family or study.get("request") != request or study.get("lane_runtime") != family_runtime(packet, family)
            or study.get("qualification_input") is not False or study.get("default_adoption") is not False
            or study.get("ordinary_qualified_tier") != 0):
        raise ValueError("actual family/runtime/qualification changed")
    prefix = [f for f, i in FAMILIES.items() if i["gpu"] == info["gpu"]][:list(f for f, i in FAMILIES.items() if i["gpu"] == info["gpu"]).index(family)]
    predecessor_refs = study.get("lane_predecessors", [])
    if [r["family"] for r in predecessor_refs] != prefix:
        raise ValueError("complete predecessor-family costs required")
    for ref in predecessor_refs:
        if ref["family"] not in studies or studies[ref["family"]]["study"] != ref["study"]:
            raise ValueError("terminal publication cut must include the exact lane predecessor")
    allowed = [j for j in jobs_for(packet, family) if not result["slots"][j["task_id"]]["preflight_blockers"]]
    if len(study["jobs"]) > len(allowed):
        raise ValueError("extra/duplicate case attempt")
    expected = deepcopy(result["slots"]); halted = False
    for row, job in zip(study["jobs"], allowed):
        if halted or row.get("compatibility_key") != job["compatibility_key"] or row.get("task_ids") != job["task_ids"]:
            raise ValueError("attempts must be the unchanged family job prefix")
        auxiliary = item.get("auxiliary", {}).get(job["task_id"], {})
        cost = costs(inputs, row, job, packet["source"], auxiliary)
        record = {"task_ids": job["task_ids"], "attempt_key": row["attempt_key"], "compatibility_key": row["compatibility_key"],
                  "status": row["status"], "allowance_seconds": job["budget_seconds"], **cost,
                  "terminal": row.get("terminal"), "launch_error": row.get("launch_error"), "qualification_input": False}
        if row["status"] == "COMPLETE":
            record["outcome"] = complete(inputs, packet, study, row, job)
            expected[job["task_id"]]["diagnostic_status"] = record["outcome"]["status"]
        elif row["status"] in {"INVALID", "INCOMPLETE"} and "outcome" not in row:
            expected[job["task_id"]]["diagnostic_status"] = row["status"]; halted = True
            record.update(completed_steps=None, completed_steps_status="UNAVAILABLE", numerical_gate="UNAVAILABLE", goal_gif=None)
            aux = auxiliary
            if "resolved" in aux:
                resolved_identity(inputs, aux["resolved"], packet, study, row, job)
            if "raw" in aux:
                raw = inputs.json(aux["raw"])
                record["raw_error_input"] = aux["raw"]
                record["error"] = public(raw.get("error"), inputs.secrets)
                record["telemetry"] = public(raw.get("telemetry", {}), inputs.secrets)
                record["raw_execution_path_scope"] = "Exception fallback label; it is not an executed public-trainer/owner receipt. Intended task is public_components."
            for pin in aux.values():
                inputs.check(pin)
        else:
            raise ValueError("engineering outcomes cannot become numerical FAIL/PASS")
        result["attempts"].append(record)
        result["paid_seconds"] += cost["paid_seconds"]; result["reserved_seconds"] += cost["reserved_seconds"]
    result["charged_seconds"] = result["paid_seconds"] + result["reserved_seconds"]
    for key, value in (("measured_paid_seconds", result["paid_seconds"]), ("unmeasured_interrupt_reserved_seconds", result["reserved_seconds"]), ("spent_seconds", result["charged_seconds"])):
        close(study[key], value)
    if expected != study["slots"]:
        raise ValueError("130-slot status denominator/qualification tamper")
    if study["status"] == "COMPLETE_DIAGNOSTIC" and (halted or len(study["jobs"]) != len(allowed)):
        raise ValueError("incomplete family cannot be labelled complete")
    if halted and study["status"] not in {study["jobs"][-1]["status"], "BUDGET_EXCEEDED"}:
        raise ValueError("terminal family/attempt status differs")
    result.update(status=study["status"], slots=expected, study_input=item["study"], lane_accounting=study["lane_accounting"])
    return result


def validate_lanes(results, packet):
    for family, result in results.items():
        if "study_input" not in result:
            continue
        gpu = result["physical_gpu"]
        prefix = []
        for f, i in FAMILIES.items():
            if f == family:
                break
            if i["gpu"] == gpu:
                prefix.append(results[f])
        if any(r["status"] != "COMPLETE_DIAGNOSTIC" for r in prefix):
            raise ValueError("uncompleted lane predecessor cannot fund the next family")
        paid = sum(r["paid_seconds"] for r in prefix); reserved = sum(r["reserved_seconds"] for r in prefix)
        expected = {"physical_gpu": gpu, "lane_cap_seconds": packet["spec"]["lane_cap_seconds"][gpu],
                    "historical_engineering_debit_seconds": DEBITS[gpu], "predecessor_paid_seconds": paid,
                    "predecessor_reserved_seconds": reserved, "predecessor_charged_seconds": paid + reserved,
                    "current_family_charged_seconds": result["charged_seconds"],
                    "inclusive_lane_charged_seconds": DEBITS[gpu] + paid + reserved + result["charged_seconds"],
                    "all_predecessor_families_complete": True, "qualification_input": False}
        if result["lane_accounting"] != expected:
            raise ValueError("inclusive cost ledger changed or history double-counted")
        spent = 0.
        for row in result["attempts"]:
            if (spent + row["allowance_seconds"] > result["family_cap_seconds"]
                    or DEBITS[gpu] + paid + reserved + spent + row["allowance_seconds"] > expected["lane_cap_seconds"]):
                raise ValueError("next full original allowance was unavailable before the attempt")
            spent += row["charged_seconds"]
        if ((spent > result["family_cap_seconds"] or expected["inclusive_lane_charged_seconds"] > expected["lane_cap_seconds"])
                and result["status"] != "BUDGET_EXCEEDED"):
            raise ValueError("measured overrun must be retained and halt")


def make_card(prepared, studies, output):
    """Root calls this only after declaring the selected family files immutable."""
    if Path(output).exists():
        raise ValueError("never replace a previous publication cut")
    card = {"schema": INPUT_SCHEMA, "prepared": binding(prepared), "studies": {},
            "qualification_input": False, "claim": "Immutable terminal-family cut; unannounced families remain NOT_RUN."}
    for family, path in studies.items():
        if family not in FAMILIES or read(path).get("status") not in TERMINAL:
            raise ValueError("only explicitly announced terminal family studies may be pinned")
        study = read(path)
        aux = {}
        for row in study["jobs"]:
            name = row["task_ids"][0]
            directory = Path(path).parent / "attempts" / name
            found = {key: binding(directory / filename) for key, filename in
                     (("resolved", "resolved.json"), ("raw", "raw-result.json"), ("grading", "graded-result.json"), ("log", "run.log"))
                     if (directory / filename).is_file()}
            if "terminal" in row:
                found["supervisor"] = binding(Path(row["terminal"]["path"]).with_name("supervisor-request.json"))
            aux[name] = found
        card["studies"][family] = {"study": binding(path), "auxiliary": aux}
    public(card)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(card, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return binding(output)


def markdown(results):
    counts = results["counts"]; cost = results["cost"]
    lines = ["# Named Atlas GPU diagnostic results", "", results["scope"], "",
             f"This immutable cut contains {counts['terminal_families']}/5 terminal family ledgers, {counts['attempted_adapted_questions']}/8 attempted adaptations and all **130 required cells** (five separate 26-slot views, each 5/19/2).",
             "", f"Current paid **{cost['current_paid_seconds']:.6f}s**, conservative reserve **{cost['current_reserved_seconds']:.6f}s**; prior engineering **{cost['prior_engineering_paid_seconds']:.6f}s** is debited once. Inclusive charged **{cost['inclusive_charged_seconds']:.6f}/10500s**. These are supervised costs, not convergence timing or FLOPs.",
             "", "| Family / cohort | GPU | Original numerical PASS | FAIL | INVALID | INCOMPLETE | BLOCKED | NOT_RUN | Family status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for family in results["families"].values():
        c = family["counts"]
        lines.append(f"| {family['family']} / `{family['cohort']}` | {family['physical_gpu']} | " + " | ".join(str(c.get(s, 0)) for s in ("PASS", "FAIL", "INVALID", "INCOMPLETE", "BLOCKED", "NOT_RUN")) + f" | {family['status']} |")
    lines += ["", "Numerical PASS/FAIL applies only to the original named variant's terminal-five gates. INVALID and INCOMPLETE have no numerical gate. Unexecuted questions remain NOT_RUN; blocked original N5 has no min11 resource-law credit.",
              "", "| Adapted question | Original grade / execution status | Original thresholds | Actual goal GIF |", "|---|---|---|---|"]
    for family in results["families"].values():
        attempts = {r["task_ids"][0]: r for r in family["attempts"]}
        for name, slot in family["slots"].items():
            if not slot["in_diagnostic_batch"]:
                continue
            q = results["questions"][name]; row = attempts.get(name, {})
            thresholds = "; ".join(f"{k} {op} {bound}" for k, op, bound in q["thresholds"])
            media = row.get("outcome", {}).get("gif_publication")
            link = f"[recorded {q['steps']}-update goal GIF]({media['path']})" if media else "Unavailable; no invented media or gate"
            lines.append(f"| {q['goal']} (`{name}`) | **{slot['diagnostic_status']}** | {thresholds} | {link} |")
    for family in results["families"].values():
        for row in family["attempts"]:
            if row["status"] != "COMPLETE":
                error = row.get("error") or {}
                memory = row.get("telemetry", {}).get("memory", {})
                lines += ["", f"`{row['task_ids'][0]}` retains **{row['status']}**, not a numerical FAIL: {error.get('type', 'Unavailable error type')} — {error.get('message', 'No complete numerical evidence')}. Completed updates are UNAVAILABLE. Recorded CUDA peak allocation {memory.get('cuda_peak_allocated_bytes', 'UNAVAILABLE')} bytes; peak reserve {memory.get('cuda_peak_reserved_bytes', 'UNAVAILABLE')} bytes. The generic public_trainer exception fallback is not an executed owner receipt; intended law is public_components. No gate or GIF is invented."]
    lines += ["", "The AE variant uses the original fixed .025-width MoG with explicit routed AE ownership. The word variant has eleven actual rows, five target words, a free encoder and a distinct same-effective-code joint law; original N5 remains BLOCKED. Conditional, routed and multibank variants retain their declared source-owned target/probe and sampling laws. Selected states, full checkpoints and optimizer owners are hash-bound, without replay by this publisher.",
              "", "The structural-readiness card is metadata/API readiness only; it proves neither representability nor learned quality. There is no ordinary tier, calibration, shipping-default, speed-ranking or cross-family/cohort qualification credit.",
              "", f"Scientific source `{results['source']['origin_commit']}` / `{results['source']['digest']}`; protocol `{results['spec_sha256']}`. Exact applied Recipes, runtime, independent grades, source/task/evaluator hashes and all 130 statuses are in [results.json](results.json); input identities are in [input-index.json](input-index.json). No raw logs, tensors, state dumps or private attempt tokens are copied.", ""]
    return "\n".join(lines)


def publish(card_path, trusted_sha256, output):
    inputs = Inputs(); card_pin = binding(card_path)
    if card_pin["sha256"] != trusted_sha256:
        raise ValueError("explicit trusted input-card SHA required")
    card = inputs.json(card_pin)
    if (card.get("schema") != INPUT_SCHEMA or card.get("qualification_input") is not False
            or not isinstance(card.get("studies"), dict) or not set(card["studies"]) <= set(FAMILIES)):
        raise ValueError("invalid immutable publication cut")
    packet = inputs.json(card["prepared"])
    packet_identity(packet, inputs); prior = engineering(packet, inputs)
    families = {family: project_family(inputs, packet, family, card["studies"].get(family), card["studies"]) for family in FAMILIES}
    validate_lanes(families, packet)
    questions = {}
    for family, value in families.items():
        value["counts"] = dict(Counter(s["diagnostic_status"] for s in value["slots"].values()))
        for name, slot in value["slots"].items():
            task = packet["requests"][family]["tasks"][name]
            parent = task.get("policy_parent", {}).get("id", name)
            questions[name] = {"id": name, "parent_id": parent, "goal": QUESTIONS.get(parent, parent),
                               "steps": task["execution"]["steps"], "thresholds": task["evaluation"].get("thresholds", []),
                               "evaluation_contract": task["evaluation"],
                               "execution_sha256": digest(task["execution"]), "evaluation_sha256": digest(task["evaluation"]),
                               "sampling": {k: task["evaluation"][k] for k in ("sampling_law", "scoring_weights", "eval_output_noise", "policy_observation") if k in task["evaluation"]},
                               "host_resources": task["execution"].get("resources"), "prior": task["execution"].get("prior"),
                               "policy_contract": task["execution"].get("policy_contract"), "required": True}
    all_counts = dict(Counter(s["diagnostic_status"] for f in families.values() for s in f["slots"].values()))
    paid = sum(f["paid_seconds"] for f in families.values()); reserve = sum(f["reserved_seconds"] for f in families.values())
    lanes = {gpu: {"cap_seconds": cap, "prior_engineering_paid_seconds": DEBITS[gpu],
                  "current_paid_seconds": sum(f["paid_seconds"] for f in families.values() if f["physical_gpu"] == gpu),
                  "current_reserved_seconds": sum(f["reserved_seconds"] for f in families.values() if f["physical_gpu"] == gpu)}
             for gpu, cap in packet["spec"]["lane_cap_seconds"].items()}
    for lane in lanes.values():
        lane["inclusive_charged_seconds"] = lane["prior_engineering_paid_seconds"] + lane["current_paid_seconds"] + lane["current_reserved_seconds"]
    results = {"schema": SCHEMA, "scope": "One fixed LR .0053125 / prior-rate 1.5 pair across five explicitly named host laws, seed 0, GPU-only numerical evidence. Five separate full views; no pooled ranking or qualification.",
               "source": {k: packet["source"][k] for k in SOURCE}, "source_file_count": len(packet["source"]["files"]),
               "spec_sha256": packet["spec_sha256"], "runtime": packet["runtime_contract"], "trusted_cut": card_pin,
               "structural_readiness": packet["capacity_preflight"], "engineering_carryover": prior, "families": families,
               "questions": questions, "counts": {"required_family_cells": 130, "required_slots_per_family": 26, "tiers_per_family": {"1": 5, "2": 19, "3": 2},
                   "terminal_families": len(card["studies"]), "declared_adapted_questions": 8,
                   "attempted_adapted_questions": sum(len(f["attempts"]) for f in families.values()), "statuses": all_counts},
               "cost": {"current_paid_seconds": paid, "current_reserved_seconds": reserve,
                        "prior_engineering_paid_seconds": prior["paid_seconds"], "inclusive_charged_seconds": paid + reserve + prior["paid_seconds"],
                        "original_cap_seconds": 10500, "lanes": lanes, "convergence_time_available": False, "flops_available": False},
               "historical_original_word": {"prior_rows": 5, "status": "BLOCKED", "current_min11_credit": False},
               **{flag: False for flag in FLAGS}, "reuse": False}
    if sum(all_counts.values()) != 130:
        raise ValueError("complete five-view denominator required")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("publish to a new directory; never overwrite a prior cut")
    inputs.recheck(); public(results, inputs.secrets)
    output.mkdir(parents=True)
    for family in families.values():
        for row in family["attempts"]:
            outcome = row.get("outcome")
            if outcome is None:
                continue
            name = outcome["task_id"]
            for kind, original, relative in (
                ("gif_publication", outcome["media_input"]["gif"], "gifs/" + name + ".gif"),
                ("grade_publication", outcome["grading_input"], "grades/" + name + ".json"),
                ("media_receipt_publication", outcome["media_input"]["receipt"], "media/" + name + ".json")):
                source = inputs.check(original); destination = output / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                if kind != "gif_publication":
                    public(read(source), inputs.secrets)
                shutil.copyfile(source, destination)
                copied = binding(destination)
                if copied["sha256"] != original["sha256"] or copied["bytes"] != original["bytes"]:
                    raise ValueError("copied public artifact bytes differ")
                outcome[kind] = {"path": relative, "sha256": copied["sha256"], "bytes": copied["bytes"]}
    publisher_files = [Path(__file__)]
    for name in ("test_publish_results.py", "README.md"):
        path = Path(__file__).with_name(name)
        if path.exists():
            publisher_files.append(path)
    results["publisher_source"] = {"files": {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size} for p in publisher_files},
                                    "model_constructions": 0, "restores": 0, "draws": 0, "rescoring": 0, "training_updates": 0}
    inputs.recheck()
    index = {"schema": SCHEMA + "_input_index", "files": list(inputs.files.values()), "file_count": len(inputs.files), "raw_files_changed": False}
    index_path = output / "input-index.json"
    index_path.write_text(json.dumps(public(index, inputs.secrets), indent=2, sort_keys=True, allow_nan=False) + "\n")
    results["input_index"] = {"path": "input-index.json", "sha256": file_sha(index_path), "bytes": index_path.stat().st_size, "file_count": len(inputs.files)}
    text = markdown(results)
    public(text, inputs.secrets); public(results, inputs.secrets)
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "README.md").write_text(text)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path); parser.add_argument("--trusted-sha256")
    parser.add_argument("--prepared", type=Path); parser.add_argument("--study", action="append", default=[])
    parser.add_argument("--make-card", action="store_true"); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.make_card:
        if args.prepared is None or args.card or args.trusted_sha256:
            raise ValueError("make-card needs only prepared and explicit terminal study pins")
        studies = {}
        for item in args.study:
            family, path = item.split("=", 1)
            if family in studies:
                raise ValueError("duplicate terminal family")
            studies[family] = Path(path)
        result = make_card(args.prepared, studies, args.output)
        print(json.dumps({"card_sha256": result["sha256"], "bytes": result["bytes"]}))
    else:
        if args.card is None or args.trusted_sha256 is None or args.prepared or args.study:
            raise ValueError("publication requires explicit card and trusted SHA")
        result = publish(args.card, args.trusted_sha256, args.output)
        print(json.dumps({"counts": result["counts"], "cost": result["cost"], "qualification_input": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
