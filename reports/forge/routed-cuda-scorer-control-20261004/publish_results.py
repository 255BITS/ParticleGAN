"""Pure retained-byte publication of the two-node CUDA engineering control.

Root explicitly attests an immutable terminal cut. Numerical producers, pytest,
Torch and Forge are never imported or executed by this publisher.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import xml.etree.ElementTree as ET

SCHEMA = "pg_routed_cuda_scorer_engineering_publication_v1"
COMMIT = "fb7acc775b3a1a6184d36b55e035b9da04531492"
PARENT_DIGEST = "f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed"
PARENT_SHA = "638657429eabaf2acfc62164610fa7abbded91a8e0144975d31d7abc1d47729f"
HISTORY_SHA = "2608f280b90bcaf39ac9d688282c5736e3c71f4228ec049d67695ab2e48ddcc3"
V1_HISTORY_SHA = "a9c8f90323b334ffa5f18316417b59cb0e31ebb40efdd7b6eff7db2ab63c94a0"
TEST_PATH = "tests/test_forge_routed_policy_scoring_device.py"
TEST_SHA = "ad865ea5058d24fb032a35185dae04a67fed85048c9f38319f6592ca7d6116d4"
CONFTEST_SHA = "dbfd4549ebc1b8a0a209d616920587fde2e81c1ebd99a1895cc7c45986d6065b"
CONFIG_SHA = "9037d73fe240aed03b781e28b2153618a05d2f661676b0c1e05cf70d9ff59a6d"
NAMES = ("test_cuda_selected_forward_keeps_models_on_cuda_and_original_cpu_scorer_pure",
         "test_cuda_cover_forwards_keep_selected_models_and_references_on_cuda")
NODES = [TEST_PATH + "::" + name for name in NAMES]
CLASSNAME = "tests.test_forge_routed_policy_scoring_device"
FAMILY = "atlas_cuda_scorer_engineering_control"
PAST = (4.234133972087875, 5.1976593940053135)
PRIOR = sum(PAST)
CAP = 120
LIMIT = CAP - PRIOR
OLD_DIGESTS = ("2354ab1ffba4faeb33c77461f4c0732948a2306157529148e8c9df545f287dc7",
               "f4355e10df69baf079dfb5d4e25ae2024d9dac2be7c3cc91f305a9f8f1153cb3")
OLD_WRAPPERS = ("361c3944c3c42934ee500801e9a01f2448264322c399ef9c7721b355ffe8f6bf",
                "c046b6636f06da77dd79fde6b7d363438594a3f67b3c13e01806535d0273b7d6")
FLAGS = {key: False for key in ("qualification_input", "convergence_credit", "default_adoption",
                                "speed_ranking", "named_learning_budget_input")}
PRIVATE = {"token", "tokens", "credential", "credentials", "password", "secret", "authorization",
           "access_token", "refresh_token", "api_key", "lease_fd", "lease_fds", "lease_path"}
TERMINAL = {"COMPLETE_ENGINEERING_CONTROL", "INVALID", "INCOMPLETE", "BUDGET_EXCEEDED"}
FROZEN_HELPER = Path("/ml2/hypergan/pg-routed-cuda-scorer-engineering-control-v4-20261003")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    def nonfinite(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative time/cost required")
    return float(value)


def same_number(a, b):
    if not math.isclose(number(a), number(b), rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("cost/time identity differs")


def flags(value):
    if any(value.get(k) is not False for k in FLAGS):
        raise ValueError("engineering evidence cannot supply scientific/default/speed credit")


def contract(value):
    """Root's trusted card selects the reviewed wrapper/protocol byte identities."""
    if (not isinstance(value, dict) or set(value) != {"version", "case_id", "wrapper_sha256", "protocol_sha256", "collection_schema"}
            or type(value.get("version")) is not int or value["version"] < 4
            or value.get("case_id") != "routed_and_multibank_cuda_scorer_boundary_v" + str(value["version"])
            or value.get("collection_schema") != "pg_routed_cuda_scorer_collection_v2"
            or any(not re.fullmatch("[0-9a-f]{64}", value.get(k, "")) for k in ("wrapper_sha256", "protocol_sha256"))):
        raise ValueError("explicit reviewed post-v3 wrapper/protocol/collection contract required")
    return value


def contract_from_packet(value):
    protocol = read(value["inputs"]["protocol"]["path"])
    match = re.fullmatch("pg_routed_cuda_scorer_engineering_v([1-9][0-9]*)", protocol.get("schema", ""))
    if match is None:
        raise ValueError("unknown reviewed control protocol")
    return contract({"version": int(match[1]), "case_id": protocol["id"],
                     "wrapper_sha256": value["inputs"]["wrapper"]["sha256"],
                     "protocol_sha256": value["inputs"]["protocol"]["sha256"],
                     "collection_schema": "pg_routed_cuda_scorer_collection_v2"})


def relative(value):
    p = PurePosixPath(value)
    if not isinstance(value, str) or not value or p.is_absolute() or ".." in p.parts or "\\" in value or str(p) != value:
        raise ValueError("unsafe source path")
    return p


def pin(path):
    p = Path(path).absolute()
    if p.is_symlink() or not p.is_file() or any(q.is_symlink() for q in p.parents):
        raise ValueError("missing/unsafe evidence path")
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


class Inputs:
    def __init__(self):
        self.files = {}
        self.secrets = set()

    def check(self, record):
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"} or type(record["bytes"]) is not int:
            raise ValueError("exact evidence pin required")
        p = Path(record["path"])
        if not p.is_absolute() or pin(p) != record:
            raise ValueError("changed input hash/bytes/path")
        previous = self.files.get(str(p))
        if previous is not None and previous != record:
            raise ValueError("conflicting input identity")
        self.files[str(p)] = record
        return p

    def json(self, record):
        return read(self.check(record))

    def recheck(self):
        for record in list(self.files.values()):
            self.check(record)


def public(value, secrets=()):
    if isinstance(value, dict):
        if PRIVATE.intersection(value):
            raise ValueError("private credential/nonce/lease field in public output")
        for item in value.values():
            public(item, secrets)
    elif isinstance(value, list):
        for item in value:
            public(item, secrets)
    elif isinstance(value, str) and any(s and s in value for s in secrets):
        raise ValueError("private nonce in public text/path")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite public value")
    return value


def source(value, inputs):
    if value.get("origin_commit") != COMMIT or digest(value.get("files")) != value.get("digest"):
        raise ValueError("source commit/manifest digest differs")
    root = Path(value["snapshot_path"])
    manifest = inputs.json(pin(root / "forge-source.json"))
    if manifest != {k: v for k, v in value.items() if k != "snapshot_path"}:
        raise ValueError("snapshot manifest differs")
    for name, wanted in value["files"].items():
        p = root.joinpath(*relative(name).parts)
        record = pin(p)
        if record["sha256"] != wanted:
            raise ValueError("snapshot source file differs")
        inputs.check(record)
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
              and p.suffix in {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}
              and "__pycache__" not in p.parts and p.name != "forge-source.json"}
    if actual - value["files"].keys():
        raise ValueError("unbound executable snapshot files")


def supervisor(result, resolved, source_value, inputs, limit, terminal_pin=None):
    terminal_pin = terminal_pin or result.get("terminal")
    if not terminal_pin:
        raise ValueError("durable terminal required for a public terminal cut")
    terminal = inputs.json(terminal_pin)
    request_pin = pin(Path(terminal_pin["path"]).with_name("supervisor-request.json"))
    request = inputs.json(request_pin)
    token = terminal.get("token")
    if not isinstance(token, str) or not token or request.get("token") != token or resolved.get("worker", {}).get("token") != token:
        raise ValueError("foreign terminal/supervisor/worker nonce")
    inputs.secrets.add(token)
    token_sha = hashlib.sha256(token.encode()).hexdigest()
    if result.get("token_sha256") != token_sha or request.get("source") != source_value:
        raise ValueError("terminal nonce/source identity differs")
    cmd = request.get("command", [])
    case_id = resolved["packet"]["spec"]["id"]
    match = re.fullmatch("routed_and_multibank_cuda_scorer_boundary_v([1-9][0-9]*)", case_id)
    if match is None:
        raise ValueError("unknown supervised case identity")
    suffix = "" if match[1] == "1" else "-v" + match[1]
    script = Path(source_value["snapshot_path"]) / ("engineering-controls/routed-cuda-scorer" + suffix + "/run_supervised.py")
    if (len(cmd) != 7 or cmd[1] != "-u" or Path(cmd[2]) != script or cmd[3:5] != ["--child", resolved["_input_path"]]
            or cmd[5] != "--lease-fd" or int(cmd[6]) not in request.get("lease_fds", [])):
        raise ValueError("actual supervised child command differs")
    same_number(request.get("deadline_monotonic", 0) - request.get("started_monotonic", 0), limit)
    same_number(result.get("paid_wall_seconds"), terminal.get("paid_wall_seconds"))
    paid = number(result["paid_wall_seconds"])
    reserve = 0. if terminal.get("attempt_status") == "completed" else max(0., limit - paid)
    same_number(result.get("unmeasured_interrupt_reserved_seconds"), reserve)
    same_number(result.get("charged_seconds"), paid + reserve)
    same_number(result.get("paid_cap_seconds"), limit)
    same_number(result.get("overrun_seconds"), max(0., paid + reserve - limit))
    if result.get("retry_authorized") is not False:
        raise ValueError("no automatic retry credit")
    flags(result)
    return {"attempt_status": terminal["attempt_status"], "child_returncode": terminal.get("child_returncode"),
            "token_sha256": token_sha, "terminal": terminal_pin, "supervisor_request": request_pin,
            "paid_wall_seconds": paid, "unmeasured_interrupt_reserved_seconds": reserve, "charged_seconds": paid + reserve}


def history(inputs, history_pin=None):
    history_pin = history_pin or pin(FROZEN_HELPER / "prior-attempts.json")
    if history_pin["sha256"] != HISTORY_SHA:
        raise ValueError("fixed combined historical authority required")
    h = inputs.json(history_pin)
    if (h.get("schema") != "pg_routed_cuda_scorer_engineering_carryover_v2" or h.get("qualification_input") is not False
            or h.get("original_charged_seconds") != PRIOR or h.get("cumulative_cap_seconds") != CAP or h.get("remaining_seconds") != LIMIT
            or h.get("v1", {}).get("sha256") != V1_HISTORY_SHA):
        raise ValueError("historical cost reset/double count/header drift")
    first = inputs.json(h["v1"])
    records = []
    for index, reference in enumerate((first, h["v2"])):
        declared = reference.get("inputs", {})
        if set(declared) != {"study", "cost", "log", "resolved", "junit", "terminal", "supervisor_request", "wrapper", "protocol"}:
            raise ValueError("all nine old evidence bindings required")
        for record in declared.values():
            inputs.check(record)
        study, cost, resolved = (inputs.json(declared[k]) for k in ("study", "cost", "resolved"))
        resolved["_input_path"] = declared["resolved"]["path"]
        src = study["source"]
        if (study.get("status") != "INVALID" or cost.get("status") != "INVALID" or study.get("result") != cost
                or src != study.get("execution_source") or src != resolved.get("packet", {}).get("source")
                or src.get("digest") != OLD_DIGESTS[index] or reference.get("original_status") != "INVALID"
                or reference.get("tests_executed") != 0 or reference.get("source_commit") != COMMIT
                or reference.get("parent_source_digest") != PARENT_DIGEST or reference.get("source_digest") != OLD_DIGESTS[index]
                or reference.get("attempt_key") != cost.get("attempt_key") or cost.get("terminal") != declared["terminal"]
                or declared["wrapper"]["sha256"] != OLD_WRAPPERS[index]):
            raise ValueError("old INVALID attempt/source attribution differs")
        source(src, inputs); flags(study)
        for field in ("original_paid_seconds", "original_charged_seconds"):
            same_number(reference.get(field), PAST[index])
        same_number(reference.get("original_reserved_seconds"), 0)
        same_number(study.get("spent_seconds"), PAST[index])
        proof = supervisor(cost, resolved, src, inputs, CAP - sum(PAST[:index]), declared["terminal"])
        if proof["attempt_status"] != "completed" or proof["child_returncode"] != 1 or proof["paid_wall_seconds"] != PAST[index] or proof["unmeasured_interrupt_reserved_seconds"] != 0:
            raise ValueError("old known-completed INVALID cost differs")
        if proof["supervisor_request"] != declared["supervisor_request"]:
            raise ValueError("old supervisor proof attribution differs")
        if "no tests ran" not in inputs.check(declared["log"]).read_text() or any(row.get("name") in NAMES for row in ET.parse(inputs.check(declared["junit"])).getroot().iter("testcase")):
            raise ValueError("old requested tests were not both unexecuted")
        case = next(iter(study["case_definitions"].values()))
        if case.get("wrapper_sha256") != OLD_WRAPPERS[index] or case.get("test_sha256") != TEST_SHA or case.get("node_ids") != NODES:
            raise ValueError("old exact control nodes/helper differ")
        records.append({"version": index + 1, "status": "INVALID", "source_commit": COMMIT, "source_digest": src["digest"],
                        "attempt_key": cost["attempt_key"], "tests_executed": 0, "learned_quality_credit": False,
                        "study_input": declared["study"], "cost_input": declared["cost"], **proof})
    if records[0]["attempt_key"] == records[1]["attempt_key"]:
        raise ValueError("duplicate historical attempt")
    return {"input": history_pin, "attempts": records, "paid_wall_seconds": PRIOR, "reserved_seconds": 0.,
            "charged_seconds": PRIOR, "cumulative_cap_seconds": CAP, "remaining_seconds": LIMIT, "qualification_input": False}


def packet(value, inputs, selected_contract):
    selected_contract = contract(selected_contract)
    version = str(selected_contract["version"])
    flags(value)
    pins = value.get("inputs", {})
    expected = {"source_packet": PARENT_SHA, "protocol": selected_contract["protocol_sha256"], "wrapper": selected_contract["wrapper_sha256"],
                "test": TEST_SHA, "conftest": CONFTEST_SHA, "pyproject": CONFIG_SHA, "history": HISTORY_SHA}
    if set(pins) != set(expected) or any(pins[k].get("sha256") != h for k, h in expected.items()):
        raise ValueError("fixed seven control inputs/source authority required")
    for record in pins.values():
        inputs.check(record)
    parent = inputs.json(pins["source_packet"])
    original = parent["source"]
    if original != parent["execution_source"] or original.get("digest") != PARENT_DIGEST:
        raise ValueError("original fb7 parent changed")
    source(original, inputs)
    p = inputs.json(pins["protocol"])
    if (p.get("id") != selected_contract["case_id"] or p.get("schema") != "pg_routed_cuda_scorer_engineering_v" + version
            or p.get("source_commit") != COMMIT or p.get("source_digest") != PARENT_DIGEST
            or p.get("physical_gpu") != "1" or p.get("logical_device") != "cuda:0" or p.get("node_ids") != NODES
            or p.get("timeout_seconds") != LIMIT or p.get("export_grace_seconds") != 0
            or p.get("prior_paid_seconds") != PRIOR or p.get("cumulative_engineering_cap_seconds") != CAP
            or p.get("collection_preflight_required") is not True or p.get("retries") != 0 or p.get("physical_attempt_limit") != 1):
        raise ValueError("fixed GPU1/two-control/remaining-cap law changed")
    flags(p)
    src = value["source"]
    extras = {TEST_PATH: TEST_SHA, "tests/conftest.py": CONFTEST_SHA, "pyproject.toml": CONFIG_SHA,
              "engineering-controls/routed-cuda-scorer-v" + version + "/run_supervised.py": selected_contract["wrapper_sha256"],
              "engineering-controls/routed-cuda-scorer-v" + version + "/protocol.json": selected_contract["protocol_sha256"],
              "engineering-controls/routed-cuda-scorer-v" + version + "/prior-attempts.json": HISTORY_SHA}
    expected_source = {**original, "files": {**original["files"], **extras},
                       "digest": digest({**original["files"], **extras}), "snapshot_path": src["snapshot_path"]}
    if (src != value.get("execution_source") or src != expected_source
            or value.get("schema") != "pg_routed_cuda_scorer_supervision_v" + version
            or value.get("family_paid_budget_seconds") != {FAMILY: LIMIT}
            or value.get("spec_sha256") != digest(value["spec"]) or value["spec"].get("paid_cap_seconds") != LIMIT
            or value["spec"].get("export_grace_seconds") != 0 or set(value.get("case_definitions", {})) != {selected_contract["case_id"]}):
        raise ValueError("derived source/case/spec allowance changed")
    source(src, inputs)
    runtime = value["runtime_contract"]
    lane = value["lane_runtime"]
    expected_runtime = deepcopy(parent["runtime_contract"])
    expected_runtime["packages"]["pytest"] = "9.1.1"
    expected_lane = {**expected_runtime, "physical_gpu": "1", "device": "cuda:0", "torch_threads": 1,
                     "compute": parent["requests"]["atlas_routed"]["compute_profiles"]["cuda"]}
    if runtime != expected_runtime or lane != expected_lane:
        raise ValueError("fixed public CPU1/CUDA runtime differs")
    case = value["case_definitions"][selected_contract["case_id"]]
    if (case.get("node_ids") != NODES or case.get("wrapper_sha256") != selected_contract["wrapper_sha256"] or case.get("test_sha256") != TEST_SHA
            or case.get("protocol_sha256") != selected_contract["protocol_sha256"] or case.get("source_commit") != COMMIT
            or case.get("parent_source_digest") != PARENT_DIGEST):
        raise ValueError("actual case identity differs")
    flags(case)
    past = history(inputs, pins["history"])
    resources = {"cpu_threads": 1, "host_memory_mb": 2048, "maximum_gpu_temperature_c": 82,
                 "memory_fraction": .2, "minimum_free_gpu_memory_mib": 12288}
    expected_spec = {"id": selected_contract["case_id"], "representation_card": pins["protocol"],
                     "paid_cap_seconds": LIMIT, "export_grace_seconds": 0, "frames": 0, "resources": resources,
                     "physical_attempt_limit": 1, "retries": 0, "cumulative_engineering_cap_seconds": CAP,
                     "prior_paid_seconds": PRIOR, "prior_attempt_sha256": HISTORY_SHA, "collection_preflight_required": True,
                     "cost_scope": "separate_engineering_control", **FLAGS}
    expected_case = {"id": selected_contract["case_id"], "node_ids": NODES, "pytest_version": "9.1.1",
                     "source_commit": COMMIT, "parent_source_digest": PARENT_DIGEST,
                     "wrapper_sha256": selected_contract["wrapper_sha256"], "test_sha256": TEST_SHA,
                     "protocol_sha256": selected_contract["protocol_sha256"], **FLAGS}
    expected_carryover = {"prior_attempt": pins["history"], "original_attempts": [r["attempt_key"] for r in past["attempts"]],
                          "original_statuses": ["INVALID", "INVALID"], "paid_wall_seconds": PRIOR,
                          "unmeasured_interrupt_reserved_seconds": 0., "charged_seconds": PRIOR,
                          "cumulative_engineering_cap_seconds": CAP, "remaining_seconds": LIMIT,
                          "old_tests_executed": 0, "qualification_input": False}
    if (value["spec"] != expected_spec or case != expected_case or p.get("resources") != resources
            or p.get("prior_attempt_sha256") != HISTORY_SHA or p.get("source_packet_sha256") != PARENT_SHA
            or p.get("test_sha256") != TEST_SHA or p.get("conftest_sha256") != CONFTEST_SHA
            or p.get("pytest_config_sha256") != CONFIG_SHA
            or p.get("junit") != {"tests": 2, "passed": 2, "skipped": 0, "failures": 0, "errors": 0}
            or value.get("engineering_carryover") != expected_carryover
            or value.get("capacity_preflight") != {"kind": "source_bound_software_control", "capacity_proved": False,
                                                  "learned_quality_proved": False, "qualification_input": False}):
        raise ValueError("exact software-only resource/case/debit/representation contract differs")
    return p, past


def imports(value, src):
    if not isinstance(value, dict) or not value:
        raise ValueError("actual imported-source evidence required")
    for row in value.values():
        if not isinstance(row, dict) or src["files"].get(row.get("path")) != row.get("sha256"):
            raise ValueError("imported source is foreign/unpinned")


def boundaries(rows, src, *, collection):
    stages = ["initial_conftests", "configure", "collection"] + ([] if collection else ["teardown", "teardown"])
    if not isinstance(rows, list) or [r.get("stage") for r in rows] != stages:
        raise ValueError("exact collection and two test-teardown guards required")
    for row in rows:
        if type(row.get("canonical_snapshot_entries")) is not int or row["canonical_snapshot_entries"] != 1:
            raise ValueError("duplicate/missing canonical source path")
        namespaces = row.get("namespaces", {})
        if set(namespaces) != {"experiments", "benchmarks"}:
            raise ValueError("namespace owner proof absent")
        for name, paths in namespaces.items():
            expected = [str(Path(src["snapshot_path"]) / name)]
            if paths != expected and (row["stage"] in {"collection", "teardown"} or paths != []):
                raise ValueError("foreign source namespace")


def junit(path, inputs):
    record = pin(path); tree = ET.parse(inputs.check(record)).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree) if tree.tag == "testsuites" else []
    if not suites or any(s.tag != "testsuite" for s in suites):
        raise ValueError("strict pytest JUnit structure required")
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if len(cases) != 2 or {(r.get("classname"), r.get("name")) for r in cases} != {(CLASSNAME, name) for name in NAMES}:
        raise ValueError("exact two unique requested CUDA nodes required")
    for suite in suites:
        if int(suite.get("tests", -1)) != len(suite.findall("testcase")) or any(int(suite.get(k, -1)) != 0 for k in ("failures", "errors", "skipped")):
            raise ValueError("JUnit skipped/failed/error/unknown test is not PASS")
    for row in cases:
        if any(child.tag in {"skipped", "failure", "error"} for child in row):
            raise ValueError("skipped/failed/error CUDA control cannot pass")
        number(float(row.get("time", "nan")))
    return {"node_ids": NODES, "tests": 2, "passed": 2, "skipped": 0, "failures": 0, "errors": 0, "junit": record}


def collection(record, value, inputs, selected_contract):
    x = inputs.json(record); src = value["source"]
    flags(x)
    if (x.get("schema") != selected_contract["collection_schema"] or x.get("status") != "PASS_COLLECT_ONLY"
            or x.get("source_commit") != COMMIT or x.get("source_digest") != src["digest"]
            or x.get("wrapper_sha256") != selected_contract["wrapper_sha256"] or x.get("test_sha256") != TEST_SHA
            or x.get("config_sha256") != CONFIG_SHA or x.get("conftest_sha256") != CONFTEST_SHA
            or x.get("runtime") != value["runtime_contract"] or x.get("pytest_version") != "9.1.1"
            or x.get("pytest_exit_code") != 0 or x.get("node_ids") != NODES or x.get("collect_only") is not True
            or type(x.get("tests_executed")) is not int or x["tests_executed"] != 0
            or type(x.get("fixtures_executed")) is not int or x["fixtures_executed"] != 0 or x.get("cuda_initialized") is not False):
        raise ValueError("real source-bound collect-only prerequisite differs")
    root = Path(src["snapshot_path"])
    expected = ["-q", "-p", "no:cacheprovider", "-c", str(root / "pyproject.toml"), "--rootdir=" + str(root),
                "--confcutdir=" + str(root / "tests"), "--collect-only", *NODES]
    if x.get("arguments") != expected:
        raise ValueError("collection config/node arguments differ")
    imports(x.get("imported_sources_after"), src); boundaries(x.get("path_boundaries"), src, collection=True)
    return x


def current(card, inputs):
    selected_contract = contract(card.get("contract"))
    pins = card.get("current", {})
    if not {"study", "cost", "resolved", "log", "terminal", "supervisor_request"} <= pins.keys():
        raise ValueError("complete announced terminal cut required")
    for record in pins.values():
        inputs.check(record)
    s, cost, resolved = (inputs.json(pins[k]) for k in ("study", "cost", "resolved"))
    if s.get("status") not in TERMINAL or cost.get("status") != s.get("status") or s.get("result") != cost:
        raise ValueError("RUNNING/missing/foreign terminal result")
    if any(resolved.get("packet", {}).get(k) != s.get(k) for k in resolved["packet"] if k not in {"status", "spent_seconds", "coordinator", "result"}):
        raise ValueError("resolved source/request differs from actual study")
    p, past = packet(s, inputs, selected_contract)
    resolved["_input_path"] = pins["resolved"]["path"]
    proof = supervisor(cost, resolved, s["source"], inputs, LIMIT, pins["terminal"])
    if proof["supervisor_request"] != pins["supervisor_request"] or cost.get("terminal") != pins["terminal"]:
        raise ValueError("terminal artifact attribution differs")
    same_number(s.get("spent_seconds"), proof["charged_seconds"])
    same_number(cost.get("prior_engineering_paid_seconds"), PRIOR)
    same_number(cost.get("cumulative_engineering_cap_seconds"), CAP)
    same_number(cost.get("cumulative_engineering_charged_seconds"), PRIOR + proof["charged_seconds"])
    if cost.get("prior_attempt_sha256") != HISTORY_SHA or not isinstance(cost.get("attempt_key"), str):
        raise ValueError("prior authority/attempt identity differs")
    if proof["charged_seconds"] > LIMIT and s["status"] != "BUDGET_EXCEEDED":
        raise ValueError("measured overrun cannot disappear")
    control = None; preflight = None; grade = None
    if s["status"] == "COMPLETE_ENGINEERING_CONTROL":
        if proof["attempt_status"] != "completed" or proof["child_returncode"] != 0 or not {"control", "junit", "collection"} <= pins.keys():
            raise ValueError("no complete supervised two-pass proof")
        if cost.get("control_receipt") != pins["control"] or resolved.get("collection_preflight") != pins["collection"]:
            raise ValueError("control/collection raw artifact attribution differs")
        control = inputs.json(pins["control"]); flags(control)
        preflight = collection(pins["collection"], s, inputs, selected_contract)
        grade = junit(pins["junit"]["path"], inputs)
        expected = {"schema": "pg_routed_cuda_scorer_engineering_receipt_v" + str(selected_contract["version"]), "status": "COMPLETE_ENGINEERING_CONTROL",
                    "source_commit": COMMIT, "parent_source_digest": PARENT_DIGEST, "source_digest": s["source"]["digest"],
                    "wrapper_sha256": selected_contract["wrapper_sha256"], "test_sha256": TEST_SHA, "pytest_version": "9.1.1", "runtime": s["lane_runtime"],
                    "pytest_exit_code": 0, "collection_preflight": pins["collection"], "result": grade,
                    "optimizer_update_scope": "two tiny structural controls", "full_scientific_runs": 0}
        if any(control.get(k) != v for k, v in expected.items()):
            raise ValueError("strict complete engineering receipt differs")
        imports(control.get("imported_sources_before"), s["source"]); imports(control.get("imported_sources_after"), s["source"])
        boundaries(control.get("path_boundaries"), s["source"], collection=False)
    elif s["status"] == "INVALID" and proof["attempt_status"] != "completed":
        raise ValueError("INVALID requires a known completed child")
    elif s["status"] == "INCOMPLETE" and proof["attempt_status"] == "completed" and proof["child_returncode"] == 0:
        raise ValueError("unattributed successful child cannot be called incomplete")
    return s, cost, proof, past, control, preflight, grade


def make_card(study_path, output):
    if Path(output).exists():
        raise ValueError("never overwrite a terminal card")
    s = read(study_path)
    if s.get("status") not in TERMINAL:
        raise ValueError("root must announce a terminal boundary before pinning")
    directory = Path(study_path).parent
    records = {"study": pin(study_path), **{key: pin(directory / name) for key, name in
               (("cost", "cost.json"), ("resolved", "resolved.json"), ("log", "run.log"))}}
    records["terminal"] = s["result"]["terminal"]
    records["supervisor_request"] = pin(Path(records["terminal"]["path"]).with_name("supervisor-request.json"))
    for key, name in (("control", "control-receipt.json"), ("junit", "cuda-scorer-junit.xml")):
        if (directory / name).is_file():
            records[key] = pin(directory / name)
    resolved = read(records["resolved"]["path"])
    if resolved.get("collection_preflight"):
        records["collection"] = resolved["collection_preflight"]
    selected_contract = contract_from_packet(s)
    result = {"schema": SCHEMA + "_inputs", "contract": selected_contract, "current": records, "immutable_terminal_cut": True, **FLAGS}
    public(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return pin(output)


def publish(card_path, trusted_sha, output):
    inputs = Inputs(); card_pin = pin(card_path)
    if card_pin["sha256"] != trusted_sha:
        raise ValueError("explicit root-reviewed trusted card SHA required")
    card = inputs.json(card_pin); flags(card)
    if card.get("schema") != SCHEMA + "_inputs" or card.get("immutable_terminal_cut") is not True:
        raise ValueError("wrong/unannounced publication cut")
    s, cost, proof, past, control, preflight, grade = current(card, inputs)
    status = s["status"]
    results = {"schema": SCHEMA, "status": status, "scope": "Two fixed CUDA scorer/owner structural controls; no scientific convergence or named training-budget credit.",
               "required_control_nodes": NODES, "required_controls": 2,
               "current_controls": {k: grade[k] for k in ("tests", "passed", "skipped", "failures", "errors")} if grade else None,
               "current_control_grade": "PASS" if grade else "UNAVAILABLE", "historical_attempts": past["attempts"],
               "scientific_source_commit": COMMIT, "parent_source_digest": PARENT_DIGEST, "derived_control_source_digest": s["source"]["digest"],
               "source_manifest": pin(Path(s["source"]["snapshot_path"]) / "forge-source.json"), "source_file_count": len(s["source"]["files"]),
               "input_bindings": s["inputs"], "runtime": s["lane_runtime"], "case": s["case_definitions"][card["contract"]["case_id"]],
               "reviewed_control_contract": card["contract"],
               "attempt_key": cost["attempt_key"], "current_proof": proof, "trusted_terminal_cut": card_pin,
               "collection_preflight": {"input": card["current"]["collection"], "tests_executed": 0, "fixtures_executed": 0, "cuda_initialized": False} if preflight else None,
               "cost": {"prior_engineering_paid_seconds": PRIOR, "current_paid_seconds": proof["paid_wall_seconds"],
                        "current_reserved_seconds": proof["unmeasured_interrupt_reserved_seconds"],
                        "current_charged_seconds": proof["charged_seconds"], "inclusive_charged_seconds": PRIOR + proof["charged_seconds"],
                        "cumulative_cap_seconds": CAP, "current_wall_cap_seconds": LIMIT,
                        "named_training_campaign_debit_seconds": 0, "convergence_time_available": False},
               "full_scientific_runs": 0, "learned_quality_credit": False, "ordinary_tier_credit": False,
               "cross_cohort_pooling": False, "retry_authorized": False,
               "raw_availability": "Local immutable paths in input-index.json are required for revalidation; no automatic hydration.", **FLAGS}
    inputs.recheck(); public(results, inputs.secrets)
    destination = Path(output).absolute()
    if destination.exists():
        raise ValueError("new publication directory required")
    destination.mkdir(parents=True)
    if grade:
        for key, filename in (("junit", "cuda-scorer-junit.xml"), ("control", "control-receipt.json"), ("collection", "collection-receipt.json")):
            original = inputs.check(card["current"][key])
            if key == "junit":
                tree = ET.parse(original)
                for node in tree.iter():
                    public(dict(node.attrib), inputs.secrets); public(node.text or "", inputs.secrets)
            else:
                public(read(original), inputs.secrets)
            shutil.copyfile(original, destination / filename)
            copied = pin(destination / filename)
            if copied["sha256"] != card["current"][key]["sha256"] or copied["bytes"] != card["current"][key]["bytes"]:
                raise ValueError("copied strict receipt bytes differ")
            results[key + "_publication"] = {"path": filename, "sha256": copied["sha256"], "bytes": copied["bytes"]}
    inputs.recheck()
    index = {"schema": SCHEMA + "_input_index", "files": list(inputs.files.values()), "file_count": len(inputs.files), "raw_files_changed": False}
    public(index, inputs.secrets)
    index_path = destination / "input-index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    results["input_index"] = {"path": "input-index.json", "sha256": sha(index_path), "bytes": index_path.stat().st_size, "file_count": len(inputs.files)}
    own = [Path(__file__), Path(__file__).with_name("test_publish_results.py"), Path(__file__).with_name("README.md")]
    results["publisher_source"] = {"files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in own if p.is_file()},
                                   "models": 0, "restores": 0, "draws": 0, "scorers": 0, "training_updates": 0, "queue_calls": 0}
    text = ["# CUDA scorer engineering control", "", results["scope"], "",
            f"Current status: **{status}**. Strict requested controls: **2 PASS, 0 SKIP, 0 FAIL, 0 ERROR**." if grade else f"Current status: **{status}**; a complete two-control numerical result is unavailable.",
            "", "The unused-token control checks selected models remain on CUDA while the original CPU scorer stays pure. The cover control checks selected models and source reference tensors remain on CUDA. Both use the exact frozen opt-in test nodes. Collect-only readiness executes no fixtures or tests and grants no CUDA PASS.",
            "", "| Attempt | Source digest | Actual status | Paid seconds | Conservative reserve |", "|---|---|---|---:|---:|"]
    for row in past["attempts"]:
        text.append(f"| v{row['version']} | `{row['source_digest']}` | INVALID; neither requested test ran | {row['paid_wall_seconds']:.9f} | 0 |")
    text += [f"| v{card['contract']['version']} | `{s['source']['digest']}` | {status} | {proof['paid_wall_seconds']:.9f} | {proof['unmeasured_interrupt_reserved_seconds']:.9f} |", "",
             f"Prior engineering debit **{PRIOR:.12f}s**, current paid **{proof['paid_wall_seconds']:.12f}s**, current reserve **{proof['unmeasured_interrupt_reserved_seconds']:.12f}s**; inclusive charged **{PRIOR + proof['charged_seconds']:.12f}/120s**. Each immutable earlier cost is counted once. These controls are separate from the 10,500-second named training campaign and supply no learned-quality, ordinary-tier, default, convergence-time or speed credit.",
             "", f"Scientific source `{COMMIT}` / `{PARENT_DIGEST}`. The separately derived test-wrapper snapshot, actual runtime, complete source/input hashes, nonce SHA256 and durable terminal/cost joins are retained in [results.json](results.json) and [input-index.json](input-index.json). No internal nonce values, lease descriptors, raw logs or checkpoints are copied."]
    if grade:
        text += ["", "[Byte-original JUnit](cuda-scorer-junit.xml), [strict engineering receipt](control-receipt.json), [collect-only prerequisite](collection-receipt.json). Two tiny optimizer/observer controls establish this source's engineering behavior; they are not full scientific task runs."]
    text += ["", "This publisher reads retained bytes only. It imports no Torch/Forge/producer/scorer, constructs or restores no model, draws no samples and submits no jobs. Root separately attests the immutable input card. The compact receipts are portable; independent revalidation requires the local source snapshots and raw proof paths in the input index, which are not hydrated automatically.", ""]
    public(results, inputs.secrets); public(text, inputs.secrets)
    (destination / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (destination / "README.md").write_text("\n".join(text))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-history", action="store_true"); p.add_argument("--make-card", action="store_true")
    p.add_argument("--verify-prepared", action="store_true"); p.add_argument("--prepared", type=Path)
    p.add_argument("--study", type=Path); p.add_argument("--card", type=Path); p.add_argument("--trusted-sha256")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.verify_history:
        if args.make_card or args.study or args.card or args.trusted_sha256 or args.output or args.verify_prepared or args.prepared:
            raise ValueError("history check reads only completed prior authority")
        inputs = Inputs(); h = history(inputs); inputs.recheck()
        print(json.dumps({"status": "VERIFIED_HISTORICAL_ENGINEERING_ONLY", "attempts": 2, "original_statuses": ["INVALID", "INVALID"],
                          "paid_wall_seconds": h["paid_wall_seconds"], "reserved_seconds": 0, "cumulative_cap_seconds": CAP,
                          "remaining_seconds": LIMIT, "verified_input_files": len(inputs.files), "models": 0, "draws": 0,
                          "scorers": 0, "tests_run_by_publisher": 0, **FLAGS}))
        return 0
    if args.verify_prepared:
        if args.make_card or args.study or args.card or args.output or args.prepared is None or args.trusted_sha256 is None:
            raise ValueError("only a root-pinned prepared packet may be verified")
        record = pin(args.prepared)
        if record["sha256"] != args.trusted_sha256:
            raise ValueError("root-attested preparation SHA required")
        inputs = Inputs(); value = inputs.json(record); selected_contract = contract_from_packet(value)
        _, past = packet(value, inputs, selected_contract); inputs.recheck()
        print(json.dumps({"status": "VERIFIED_PREPARATION_ONLY", "source_commit": COMMIT, "parent_source_digest": PARENT_DIGEST,
                          "derived_source_digest": value["source"]["digest"], "source_file_count": len(value["source"]["files"]),
                          "verified_input_files": len(inputs.files), "control_version": selected_contract["version"],
                          "prior_engineering_paid_seconds": past["paid_wall_seconds"], "current_tests_executed": 0,
                          "models": 0, "draws": 0, "scorers": 0, **FLAGS}))
        return 0
    if args.prepared:
        raise ValueError("preparation pin is not a terminal study or receipt")
    if args.output is None:
        raise ValueError("new card/publication output required")
    if args.make_card:
        if args.study is None or args.card or args.trusted_sha256:
            raise ValueError("root pins only an explicitly terminal study")
        print(json.dumps(make_card(args.study, args.output)))
    else:
        if args.card is None or args.trusted_sha256 is None or args.study:
            raise ValueError("explicit root-trusted terminal card required")
        value = publish(args.card, args.trusted_sha256, args.output)
        print(json.dumps({"status": value["status"], "current_controls": value["current_controls"], "cost": value["cost"], **FLAGS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
