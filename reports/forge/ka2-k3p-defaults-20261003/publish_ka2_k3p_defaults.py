"""Draw-free publication of a separately certified named KA2/K3P defaults pair.

The trusted certification SHA attests root's explicit CPU-only combine_studies
verification (including capacity sampler replay). This publisher only checks
retained bytes and recorded grades; it never calls that verifier, restores a
model, samples, scores arrays, acquires leases or executes a training child.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

CARD_SCHEMA = "particlegan_ka2_k3p_defaults_publication_certification_v1"
RESULT_SCHEMA = "particlegan_ka2_k3p_defaults_publication_v1"
CAPACITY_SCHEMA = "particlegan_ka2_k3p_capacity_v1"
RUNNER = "reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py"
SELF = "reports/forge/ka2-k3p-defaults-20261003/publish_ka2_k3p_defaults.py"
FAMILIES = ("ka2", "k3p")
OLD_FAMILIES = ("atlas", "e22")
BINDER = "reports/forge/ka2-k3p-defaults-20261003/bind_capacity.py"
PROTOCOL = "reports/forge/ka2-k3p-defaults-20261003/protocol.py"
PREVIOUS = "reports/forge/generator-step-20261003/run_generator_step.py"
DELEGATED = "reports/forge/critic-balance-20261003/run_critic_balance.py"
OVERRIDES = {"lr": .006375, "prior_lr_mult": 1.0, "d_lr_mult": 1.0}
EARLIER_SCIENTIFIC = {"atlas": 27.0030500178691, "e22": 27.3557217749767}
PREVIOUS_SCIENTIFIC = {"atlas": 28.03618986881338, "e22": 26.841381517937407}
PRIOR_SCIENTIFIC = {"atlas": 55.03923988668248, "e22": 54.19710329291411}
PRIOR_ENGINEERING = 4.757908704923466
EARLIER_TOTAL = 59.116680497769266
PRIOR_PAID = 113.99425188452005
CAMPAIGN_CAP = 15360.
FAMILY_REMAINING = {"ka2": 7620.202851408394, "k3p": 7625.802896707086}
PAIR_REMAINING = 15246.00574811548
SLOT_PREDECESSOR = {"ka2": "atlas", "k3p": "e22"}
SLOT_PRIOR = {"ka2": 59.797148591605946, "k3p": 54.19710329291411}
DISCOVERY = {"configs/forge/tasks/ring16_acquisition.json":
             "e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1"}
HEX = re.compile(r"[0-9a-f]{64}")
VERIFICATION = {"cpu_only": True, "ordinary_training_updates": 0,
                "capacity_sampler_replay": True, "numeric_trace_rescore": True}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha(path), "bytes": path.stat().st_size}


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: finite nonnegative number required")
    return value


def equal_cost(left, right, label):
    if not math.isclose(number(left, label), number(right, label), rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"{label}: cost differs from durable accounting")


def relative_path(value):
    path = Path(value)
    if not isinstance(value, str) or path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("source/artifact path must be relative and stay inside its declared root")
    return path


class Inputs:
    """Every consumed file is pinned and rechecked after exporting."""
    def __init__(self):
        self.files = {}

    def add(self, pin):
        if (not isinstance(pin, dict) or not {"path", "sha256"} <= set(pin)
                or not isinstance(pin["sha256"], str) or not HEX.fullmatch(pin["sha256"])):
            raise ValueError("artifact requires an exact path and SHA256")
        path = Path(pin["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("pinned file missing, relative or symlinked")
        key = str(path.resolve())
        if key not in self.files:
            self.files[key] = binding(path)
        actual = self.files[key]
        if (actual["sha256"] != pin["sha256"] or
                ("bytes" in pin and (type(pin["bytes"]) is not int or actual["bytes"] != pin["bytes"]))):
            raise ValueError("artifact hash/size drift")
        return path

    def json(self, pin):
        return read(self.add(pin))

    def recheck(self):
        for pin in self.files.values():
            if binding(pin["path"]) != pin:
                raise ValueError("immutable raw inputs changed during publication")


def git_blob(root, commit, relative):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full source commit required")
    relative_path(relative)
    result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("committed source blob unavailable; obtain the frozen source history")
    return hashlib.sha256(result.stdout).hexdigest()


def source_identity(inputs, root, source):
    root = Path(root).resolve()
    files = source.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("complete certifier source manifest required")
    for name, sha in {**files, **source.get("discovery_inputs_sha256", {})}.items():
        path = root / relative_path(name)
        inputs.add({"path": str(path), "sha256": sha})
        if git_blob(root, source["commit"], name) != sha:
            raise ValueError("source commit does not contain its declared bytes")
    return {"commit": source["commit"], "manifest_sha256": digest(files),
            "files": len(files), "discovery_inputs_sha256": source.get("discovery_inputs_sha256", {})}


def snapshot(inputs, manifest, source):
    if (not isinstance(manifest, dict) or manifest.get("origin_commit") != source["commit"]
            or not isinstance(manifest.get("files"), dict) or not manifest["files"]
            or manifest.get("digest") != digest(manifest["files"])):
        raise ValueError("frozen execution snapshot identity mismatch")
    root = Path(manifest["snapshot_path"]).resolve()
    expected = {key: value for key, value in manifest.items() if key != "snapshot_path"}
    metadata = root / "forge-source.json"
    if inputs.json(binding(metadata)) != expected:
        raise ValueError("snapshot metadata differs from supervisor declaration")
    for name, sha in manifest["files"].items():
        path = root / relative_path(name)
        if not path.resolve().is_relative_to(root):
            raise ValueError("snapshot file escapes source root")
        inputs.add({"path": str(path), "sha256": sha})
    source_suffixes = {".py", ".json", ".toml", ".yaml", ".yml", ".sh"}
    unexpected = {str(path.relative_to(root)) for path in root.rglob("*")
                  if path.is_file() and path.suffix in source_suffixes
                  and "__pycache__" not in path.parts and path.name != "forge-source.json"} - set(manifest["files"])
    if unexpected:
        raise ValueError("undeclared files appeared in frozen execution snapshot")
    for name, sha in {**source["files_sha256"], **source.get("discovery_inputs_sha256", {})}.items():
        if manifest["files"].get(name) != sha:
            raise ValueError("scientific source absent from exact execution snapshot")
    return {"digest": manifest["digest"], "origin_commit": manifest["origin_commit"],
            "files": len(manifest["files"]), "snapshot_path": str(root)}


class Oracle:
    """Only existing pure flags/accounting predicates; no recertification calls."""
    def __init__(self, root):
        root = Path(root).resolve()
        if any(name == "benchmarks" or name.startswith("benchmarks.") for name in sys.modules):
            origins = [Path(module.__file__).resolve() for name, module in sys.modules.items()
                       if name.startswith("benchmarks.") and getattr(module, "__file__", None)]
            if any(not path.is_relative_to(root) for path in origins):
                raise ValueError("publisher process already imported another scientific checkout")
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("ka2_k3p_defaults_publication_oracle", root / RUNNER)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.contract, self.search, _ = self.runner.modules()
        from benchmarks.toy_audit import api_publish
        self.api_publish = api_publish
        if Path(api_publish.__file__).resolve() != root / "benchmarks/toy_audit/api_publish.py":
            raise ValueError("pure grade verifier came from another checkout")

    def select(self, packet):
        return self.runner.select_results(packet)

    def costs(self, packet):
        self.runner.verify_costs(packet)

    def durable(self, packet, trial, row):
        self.runner.durable_cost(packet, trial, row)

    def complete(self, directory):
        return self.api_publish.verify_run(directory)

    def hold(self, receipt):
        return self.search.acquisition_hold(receipt)

    def partial(self, path, packet, trial, row):
        return self.runner.outcome(path, packet, trial, row, row.get("child_returncode"))

    def recipe(self, case, family):
        return self.runner.protocol().resolved_recipe(case, family, OVERRIDES)

    def law(self, case, family):
        return self.runner.protocol().family_law(case, family)


def bind_certification(combined_path, scientific_root):
    """Bind root's already completed verification; this executes no certifier."""
    combined = read(combined_path)
    return {"schema": CARD_SCHEMA, "combined": binding(combined_path),
            "certifier": {"root": str(Path(scientific_root).resolve()),
                          **deepcopy(combined["source"]), "function": "combine_studies"},
            "verification": dict(VERIFICATION),
            "prior_carryover": deepcopy(combined["spec"].get("prior_carryover"))}


def check_archives(inputs, packet, *, families=FAMILIES):
    archives = packet.get("family_archives")
    if (not isinstance(archives, list) or len(archives) != 2
            or {row.get("family") for row in archives} != set(families)):
        raise ValueError("one separately certified study per family required")
    result = {}
    for pin in archives:
        family = pin["family"]
        archive = inputs.json(pin)
        if archive.get("executed_family") != family or archive.get("lane_runtime") != pin.get("runtime"):
            raise ValueError("family archive/runtime identity mismatch")
        for key in ("spec", "spec_sha256", "source", "case_definitions", "capacity_preflight", "runtime_contract"):
            if archive.get(key) != packet.get(key):
                raise ValueError("combined certification pools different source/protocol cohorts")
        actual = next((trial for trial in archive["trials"] if trial["family"] == family), None)
        expected = next(trial for trial in packet["trials"] if trial["family"] == family)
        if actual != expected or actual["status"] == "RUNNING":
            raise ValueError("combined trial differs from its finished family archive")
        for trial in archive["trials"]:
            if trial["family"] != family and (trial.get("paid_wall_seconds", 0.) or any(
                    row.get("attempt_key") or row.get("receipt_path") for row in trial["cases"])):
                raise ValueError("family archive contains another family's ordinary work")
        result[family] = archive
    for key in ("spent_seconds", "measured_paid_seconds", "unmeasured_interrupt_reservation_seconds"):
        equal_cost(packet[key], sum(value[key] for value in result.values()), key)
    runtime_fields = ("python", "torch", "cuda", "device", "cuda_device_model", "torch_threads")
    runtimes = [{key: value["lane_runtime"].get(key) for key in runtime_fields}
                for value in result.values() if value.get("lane_runtime") is not None]
    if len(runtimes) == 2 and runtimes[0] != runtimes[1]:
        raise ValueError("cannot publish a joint comparison across different execution runtimes")
    return result


def artifacts(inputs, root, values):
    result = {}
    for name, pin in values.items():
        path = Path(pin["path"]) if "path" in pin else Path(root) / relative_path(name)
        result[name] = binding(inputs.add({**pin, "path": str(path.resolve())}))
    return result


def retained_bindings(inputs, value):
    """Check explicitly pinned construction inputs without opening state files."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            inputs.add(value)
        for name, path in value.items():
            if name.endswith("_path") and name[:-5] + "_sha256" in value:
                inputs.add({"path": path, "sha256": value[name[:-5] + "_sha256"]})
        for child in value.values():
            retained_bindings(inputs, child)
    elif isinstance(value, list):
        for child in value:
            retained_bindings(inputs, child)


def capacity(inputs, packet, root, oracle):
    declared = packet["spec"]["representation_card"]
    path = Path(declared["path"])
    if not path.is_absolute():
        path = Path(root) / path
    card = inputs.json({**declared, "path": str(path)})
    expected = {(family, row["id"]) for family in FAMILIES for row in packet["spec"]["cases"]}
    records = card.get("records", [])
    if (card.get("schema") != CAPACITY_SCHEMA or card.get("status") != "COMPLETE"
            or card.get("required_records") != 16 or len(records) != 16
            or card.get("requested_cells") != [{"family": family, "case_id": row["id"]}
                                               for family in FAMILIES for row in packet["spec"]["cases"]]
            or {(r.get("family"), r.get("case_id")) for r in records} != expected
            or card.get("ordinary_training_updates") != 0 or card.get("fitting_updates") != 0
            or card.get("ordinary_qualification_credit") is not False):
        raise ValueError("capacity packet has incomplete or forged zero-update denominator")
    result = {}
    for record in records:
        key = record["family"] + "/" + record["case_id"]
        case = packet["case_definitions"][record["case_id"]]
        expected_recipe = oracle.recipe(case, record["family"])
        expected_law = oracle.law(case, record["family"])
        if (record != packet["capacity_preflight"].get(key)
                or record.get("schema") != CAPACITY_SCHEMA or record.get("case") != case
                or type(record.get("seed")) is not int or record["seed"] != 24002
                or digest(record.get("recipe_overrides")) != digest(OVERRIDES)
                or digest(record.get("requested_recipe_overrides")) != digest(OVERRIDES)
                or digest(record.get("resolved_recipe")) != digest(expected_recipe)
                or digest(record.get("family_law")) != digest(expected_law)
                or record.get("bindings", {}).get("case_sha256") != digest(case)
                or record.get("status") not in {"SUPPORTED", "UNRESOLVED", "BLOCKED"}
                or record.get("ordinary_training_updates") != 0 or record.get("fitting_updates") != 0
                or record.get("ordinary_qualification_credit") is not False):
            raise ValueError("capacity record changed or grants learning credit")
        captured_source = record.get("source", {})
        if (captured_source.get("binder_sha256") != packet["source"]["files_sha256"][BINDER]
                or captured_source.get("new_test_files_sha256", {}).get(PROTOCOL) != packet["source"]["files_sha256"][PROTOCOL]
                or not isinstance(captured_source.get("files_sha256"), dict) or not captured_source["files_sha256"]):
            raise ValueError("capacity binder/Recipe adapter/protected source bindings differ")
        source_identity(inputs, root, {"commit": captured_source["reproducer_commit"],
            "files_sha256": {**captured_source["files_sha256"], **captured_source["new_test_files_sha256"],
                             BINDER: captured_source["binder_sha256"]}})
        pins = artifacts(inputs, path.parent, record.get("artifacts", {}))
        observations = record.get("observations")
        if record["status"] in {"SUPPORTED", "UNRESOLVED"} and set(pins) != {"state", "samples"}:
            raise ValueError("complete capacity witness needs its state and sampled arrays")
        if record["status"] == "BLOCKED":
            if observations != [] or pins or record.get("capacity_credit") is not False:
                raise ValueError("blocked capacity cannot claim a witness or learning result")
            error_pin = binding(inputs.add(record["error_artifact"]))
            error = record.get("error")
            if (not isinstance(error, dict) or set(error) != {"stage", "type", "message", "traceback"}
                    or not all(isinstance(value, str) and value for value in error.values())
                    or read(error_pin["path"]) != error):
                raise ValueError("blocked capacity needs its exact preparation error")
            available = artifacts(inputs, path.parent, record.get("available_artifacts", {}))
        else:
            error_pin, error, available = None, None, {}
            if not isinstance(observations, list) or len(observations) != 1:
                raise ValueError("one full-count zero-update capacity observation required")
            observation = observations[0]
            expected_pass = record["status"] == "SUPPORTED"
            if (type(observation.get("completed_steps")) is not int or observation["completed_steps"] != 0
                    or type(observation.get("samples")) is not int or observation["samples"] != case["eval_samples"]
                    or type(observation.get("evaluation_seed")) is not int or observation["evaluation_seed"] != 34002
                    or observation.get("passed") is not expected_pass
                    or not isinstance(observation.get("failed_bounds"), list)
                    or bool(observation["failed_bounds"]) == expected_pass
                    or not isinstance(observation.get("metrics"), dict) or not observation["metrics"]
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in observation["metrics"].values())):
                raise ValueError("capacity original status/count/seed/finite metrics contradict its zero-update scope")
        retained_bindings(inputs, record.get("construction_inputs", {}))
        result[key] = {"status": record["status"], "case_sha256": record["bindings"]["case_sha256"],
                       "recipe_sha256": digest(record["resolved_recipe"]), "artifacts": pins,
                       "family_law": expected_law,
                       "source": {"reproducer_commit": captured_source["reproducer_commit"],
                                  "scientific_base_commit": captured_source.get("scientific_base_commit"),
                                  "manifest_sha256": digest(captured_source["files_sha256"]),
                                  "files": len(captured_source["files_sha256"]),
                                  "new_test_files_sha256": captured_source["new_test_files_sha256"],
                                  "binder_sha256": captured_source["binder_sha256"]},
                       "error": error, "error_artifact": error_pin, "available_artifacts": available,
                       "ordinary_training_updates": 0, "observations": [
                           {name: observation.get(name) for name in ("completed_steps", "samples", "evaluation_seed", "passed", "failed_bounds", "metrics")}
                           for observation in record.get("observations", [])]}
    if set(packet["capacity_preflight"]) != set(result):
        raise ValueError("capacity preflight includes unknown or missing cells")
    return result


def supervised(inputs, oracle, packet, trial, row):
    if not row.get("attempt_key"):
        if row.get("paid_wall_seconds", 0.) or row.get("unmeasured_interrupt_reserved_seconds", 0.):
            raise ValueError("charged row lacks durable attempt identity")
        return {}
    oracle.durable(packet, trial, row)
    directory = Path(packet["coordinator"]["queue_root"]) / "policy/attempts" / row["attempt_key"]
    result = {}
    for name in ("supervisor-request.json", "supervisor-terminal.json"):
        path = directory / name
        if path.is_file():
            result[name] = binding(inputs.add(binding(path)))
        elif not (row.get("unmeasured_interrupt_reserved_seconds") == row["timeout_seconds"] + packet["spec"]["export_grace_seconds"]
                  and not row.get("paid_wall_seconds", 0.)):
            raise ValueError("missing durable supervision evidence")
    if row.get("log_path"):
        result["log"] = binding(inputs.add(binding(row["log_path"])))
    return result


def receipt_projection(inputs, oracle, packet, archive, trial, row):
    if row["status"] == "RUNNING":
        raise ValueError("live attempts must reach a stable boundary before publication")
    if not row.get("receipt_path"):
        if (row["status"] in {"PASS", "FAIL"} or row.get("full_protocol_complete")
                or row.get("original_gate") is not None or row.get("study_gate") is not None):
            raise ValueError("scientific result lacks a bound original public receipt")
        return {"completed_updates": None, "final_metrics": None, "media": None}
    path = inputs.add({"path": row["receipt_path"], "sha256": row["receipt_sha256"]})
    raw = read(path)
    case = packet["case_definitions"][row["id"]]
    if (raw.get("case") != case or raw.get("seed") != packet["spec"]["seed"]
            or raw.get("requested_recipe_overrides") != trial["recipe_overrides"]
            or raw.get("source", {}).get("commit") != packet["source"]["commit"]
            or any(raw.get("source", {}).get("files_sha256", {}).get(name) != sha
                   for name, sha in packet["source"]["files_sha256"].items())):
        raise ValueError("public receipt case/seed/recipe/source attribution drift")
    if raw["status"] == "COMPLETE":
        oracle.complete(path.parent)  # Flags, cadence, artifacts and array health; no scorer/forward.
        hold = oracle.hold(raw)
        status = raw["verdict"] if not raw["passed"] else hold["status"]
        expected = {"status": status, "original_gate": raw["verdict"], "study_gate": hold["status"],
                    "full_protocol_complete": True, "acquisition_hold": hold, "recipe": raw["recipe"],
                    "runtime": raw["runtime"], "elapsed_seconds": raw["elapsed_seconds"],
                    "final_metrics": raw["observations"][-1]["metrics"], "artifacts": raw["artifacts"]}
        if (row.get("child_returncode") != (0 if raw["passed"] else 1)
                or digest(raw["recipe"]) != row["resolved_recipe_sha256"]
                or raw["protocol"]["updates"] != case["default_steps"]
                or raw["protocol"]["evaluation_samples"] != case["eval_samples"]
                or raw["protocol"]["wall_cap_seconds"] != row["timeout_seconds"]
                or raw["protocol"]["media_frames"] != packet["spec"]["frames"]
                or number(raw["elapsed_seconds"], "acquisition time") > row["timeout_seconds"]
                or row.get("paid_wall_seconds", 0.) < raw["elapsed_seconds"]
                or any(raw["runtime"].get(key) != value for key, value in archive["lane_runtime"].items())):
            raise ValueError("complete receipt differs from fixed runtime/Recipe/protocol/clock")
    else:
        expected = oracle.partial(path.parent, archive, trial, row)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("saved original/study verdict or receipt metadata contradicts raw observations")
    pins = artifacts(inputs, path.parent, raw.get("artifacts", {}))
    media = pins.get("goal.gif")
    if media:
        from PIL import Image
        with Image.open(media["path"]) as gif:
            if gif.n_frames != raw["gif_frames"]:
                raise ValueError("goal GIF frame count differs from actual public receipt")
        media["frames"] = raw["gif_frames"]
    return {"receipt": binding(path), "completed_updates": raw.get("completed_updates", 0),
            "final_metrics": raw.get("observations", [{}])[-1].get("metrics") if raw.get("observations") else None,
            "artifacts": pins, "media": media}


def engineering(inputs, oracle, packet, card):
    carryover = packet["spec"].get("engineering_carryover")
    if card.get("engineering_carryover") != carryover:
        raise ValueError("certification engineering overhead differs from preregistered carryover")
    if carryover is None:
        return [], 0.
    pins = {name: binding(inputs.add(pin)) for name, pin in carryover["artifacts"].items()}
    raw = read(pins["study"]["path"])
    trial = next(t for t in raw["trials"] if t["family"] == carryover["family"])
    oracle.durable(raw, trial, trial["cases"][0])
    if (trial["status"] != "ERROR" or raw.get("executed_family") != carryover["family"]
            or any(row.get("receipt_path") or row.get("full_protocol_complete") or row["status"] in {"PASS", "FAIL"}
                   for t in raw["trials"] for row in t["cases"])):
        raise ValueError("setup overhead falsely carries scientific credit")
    paid = sum(number(r.get("paid_wall_seconds", 0.), "engineering paid") for t in raw["trials"] for r in t["cases"])
    equal_cost(paid, carryover["paid_seconds"], "engineering paid carryover")
    equal_cost(paid, raw["spent_seconds"], "engineering charged")
    if raw["unmeasured_interrupt_reservation_seconds"] != 0:
        raise ValueError("engineering cost cannot hide additional reservation")
    terminal = read(pins["terminal"]["path"])
    request = read(pins["request"]["path"])
    if (terminal.get("attempt_status") != "completed" or terminal.get("child_returncode") != 1
            or terminal.get("token") != request.get("token") or request.get("source") != raw["execution_source"]):
        raise ValueError("startup error is not bound to its durable completed engineering attempt")
    equal_cost(terminal.get("paid_wall_seconds"), paid, "engineering supervisor cost")
    original_snapshot = snapshot(inputs, raw["execution_source"], raw["source"])
    return [{"family": carryover["family"], "study_id": raw["study_id"], "status": "ERROR",
             "claim": carryover["claim"], "ordinary_training_updates": 0,
             "paid_seconds": paid, "conservative_reserved_seconds": 0., "artifacts": pins,
             "source_commit": raw["source"]["commit"], "execution_snapshot": original_snapshot}], paid


def predecessor(inputs, oracle, pins, *, label, scientific_paid):
    """Retain a closed old pair as immutable source-bound negative evidence."""
    required = {"combined.json", "certification.json", "publication-v1/results.json", "atlas/study.json", "e22/study.json"}
    if set(pins) != required:
        raise ValueError("complete prior publication/combined/certification/study bindings required")
    old = read(pins["combined.json"]["path"])
    old_card = read(pins["certification.json"]["path"])
    report = read(pins["publication-v1/results.json"]["path"])
    schema = "particlegan_" + label + "_publication"
    if (old_card.get("schema") != schema + "_certification_v1"
            or old_card.get("verification") != VERIFICATION or old_card.get("combined") != pins["combined.json"]
            or report.get("schema") != schema + "_v1"
            or report.get("certification") != pins["certification.json"] or report.get("combined") != pins["combined.json"]
            or report["costs"]["ordinary_paid_seconds"] != sum(scientific_paid.values())
            or report["costs"]["ordinary_reserved_seconds"] != 0.):
        raise ValueError("prior trusted certification/result does not bind its separate scientific costs")
    certifier = old_card["certifier"]
    old_source = {key: value for key, value in certifier.items() if key not in {"root", "function"}}
    if old_source != old["source"] or certifier.get("function") != "combine_studies":
        raise ValueError("prior source certification mismatch")
    source_pin = source_identity(inputs, certifier["root"], old_source)
    archives = check_archives(inputs, old, families=OLD_FAMILIES)
    scientific_records = []
    for family in OLD_FAMILIES:
        archived_pin = next(row for row in old["family_archives"] if row["family"] == family)
        if any(archived_pin.get(key) != pins[family + "/study.json"][key] for key in ("path", "sha256")):
            raise ValueError("prior scientific family archive binding differs")
        archive = archives[family]
        trial = next(t for t in old["trials"] if t["family"] == family)
        row = trial["cases"][0]
        if (trial["status"] != "FAIL" or row["status"] != "FAIL" or row.get("original_gate") != "FAIL"
                or row.get("study_gate") != "FAIL" or row.get("full_protocol_complete") is not True
                or row.get("paid_wall_seconds") != scientific_paid[family]
                or archive["measured_paid_seconds"] != scientific_paid[family]
                or archive["spent_seconds"] != scientific_paid[family]
                or archive["unmeasured_interrupt_reservation_seconds"] != 0.
                or any(r["status"] != "UNKNOWN" or r.get("attempt_key") or r.get("receipt_path") for r in trial["cases"][1:])):
            raise ValueError("prior negative science or seven unreached cells changed")
        evidence = receipt_projection(inputs, oracle, old, archive, trial, row)
        supervisors = supervised(inputs, oracle, archive, trial, row)
        if evidence["completed_updates"] != 600:
            raise ValueError("prior full600 scientific boundary missing")
        frozen = snapshot(inputs, archive["execution_source"], archive["source"])
        scientific_records.append({"cohort": label, "family": family, "status": "FAIL",
            "original_gate": row["original_gate"], "study_gate": row["study_gate"], "full_protocol_complete": True,
            "ordinary_training_updates": evidence["completed_updates"], "new_cell_credit": False,
            "paid_seconds": scientific_paid[family], "source_commit": old_source["commit"],
            "execution_snapshot": frozen, "supervision": supervisors, "receipt": evidence["receipt"],
            "artifacts": evidence["artifacts"], "study": pins[family + "/study.json"]})
    return old, old_card, report, {"scientific": scientific_records,
            "source": {"root": certifier["root"], **source_pin}, "publication_artifacts": pins}


def prior(inputs, oracle, packet, card):
    """Cost-only slot inheritance; four old negative runs and one setup ERROR."""
    carry = packet["spec"].get("prior_carryover")
    if (not isinstance(carry, dict) or card.get("prior_carryover") != carry
            or carry.get("schema") != "named_family_previous_debit_v1"
            or carry.get("engineering_paid_seconds") != PRIOR_ENGINEERING
            or carry.get("scientific_paid_seconds") != PRIOR_SCIENTIFIC
            or carry.get("total_paid_seconds") != PRIOR_PAID
            or carry.get("slot_predecessor") != SLOT_PREDECESSOR
            or carry.get("slot_prior_paid_seconds") != SLOT_PRIOR):
        raise ValueError("fixed prior science/engineering debit or cost-only family slots missing or changed")
    latest_pins = {name: binding(inputs.add(pin)) for name, pin in carry["artifacts"].items()}
    latest, latest_card, report, previous = predecessor(inputs, oracle, latest_pins,
        label="generator_step", scientific_paid=PREVIOUS_SCIENTIFIC)
    earlier = latest["spec"].get("prior_carryover")
    if (not isinstance(earlier, dict) or latest_card.get("prior_carryover") != earlier
            or earlier.get("schema") != "fixed_contrast_previous_debit_v1"
            or earlier.get("scientific_paid_seconds") != EARLIER_SCIENTIFIC
            or earlier.get("total_paid_seconds") != EARLIER_TOTAL
            or report["costs"]["prior_scientific_paid_seconds"] != sum(EARLIER_SCIENTIFIC.values())
            or report["costs"]["prior_engineering_paid_seconds"] != PRIOR_ENGINEERING
            or report["costs"]["prior_total_paid_seconds"] != EARLIER_TOTAL
            or report["costs"]["combined_charged_seconds"] != PRIOR_PAID):
        raise ValueError("latest prior publication does not preserve its nested one-time debit")
    earlier_pins = {name: binding(inputs.add(pin)) for name, pin in earlier["artifacts"].items()}
    old, old_card, old_report, first = predecessor(inputs, oracle, earlier_pins,
        label="critic_balance", scientific_paid=EARLIER_SCIENTIFIC)
    expected_engineering = earlier.get("engineering", {})
    original_engineering = old["spec"].get("engineering_carryover", {})
    if (expected_engineering.get("family") != "atlas"
            or expected_engineering.get("paid_seconds") != PRIOR_ENGINEERING
            or old_card.get("engineering_carryover") != original_engineering
            or any(expected_engineering.get(key) != original_engineering.get(key)
                   for key in ("family", "paid_seconds", "artifacts"))
            or old_report["costs"]["engineering_paid_seconds"] != PRIOR_ENGINEERING
            or old_report["costs"]["combined_charged_seconds"] != EARLIER_TOTAL):
        raise ValueError("prior engineering source/cost differs from its original retained cohort")
    engineering_records, engineering_paid = engineering(inputs, oracle, old, old_card)
    science_total = sum(PREVIOUS_SCIENTIFIC.values()) + sum(EARLIER_SCIENTIFIC.values())
    equal_cost(science_total, sum(PRIOR_SCIENTIFIC.values()), "cumulative prior science")
    equal_cost(engineering_paid + science_total, PRIOR_PAID, "prior total debit")
    return {"engineering": engineering_records, "scientific": first["scientific"] + previous["scientific"],
            "source": previous["source"], "sources": [first["source"], previous["source"]],
            "publication_artifacts": latest_pins, "nested_publication_artifacts": earlier_pins,
            "cost_only_slots": SLOT_PREDECESSOR, "paid_seconds": PRIOR_PAID}, PRIOR_PAID


def markdown(results):
    lines = ["# Named KA2/K3P defaults contrast", "", "Two unchanged family configurations; eight required public-API cases each. "
             "Capacity is a zero-update necessary witness, not learned convergence. Unreached cells remain UNKNOWN. "
             "This provisional screen grants no public defaults, global family winner or speed ranking.", "",
             "Family-owned law: fast-only serving, no DV12 controller or latent perturbation, "
             "AMSGrad false, named scheduled rates, fixed output sigma warmed from zero to 0.029 over 20% of each full host horizon. "
             "Particle rows are read directly. The original GIF's inherited generic policy caption does not change this law.", "",
             "| Family | Case / question | Capacity | Execution / added study | Original gate | First-window hold | Goal GIF |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in results["cases"]:
        hold = row.get("acquisition_hold") or {}
        later = f"confirm {hold.get('acquired_step', 'unavailable')}; later {hold.get('hold_passed', 'unavailable')}/{hold.get('hold_checks', 'unavailable')} passing (at least 5 required)"
        media = row.get("media")
        link = f"[original GIF]({media['path']})" if media else "unavailable"
        question = row["question"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['family']} | `{row['id']}`: {question} | {row['capacity']['status']} | "
                     f"{row['status']} / {row.get('study_gate') or 'unavailable'} | {row.get('original_gate') or 'unavailable'} | {later} | {link} |")
    costs = results["costs"]
    lines += ["", f"Measured child time: {costs['ordinary_paid_seconds']:.6f}s; conservative interruption reserve: "
              f"{costs['ordinary_reserved_seconds']:.6f}s; prior science counted once: {costs['prior_scientific_paid_seconds']:.6f}s; "
              f"prior engineering counted once: {costs['prior_engineering_paid_seconds']:.6f}s. "
              f"Combined charge: {costs['combined_charged_seconds']:.6f}s within the original {costs['combined_cap_seconds']:.6f}s ceiling.", "",
              "The KA2/K3P bookkeeping slots inherit prior Atlas/E22 costs only. All four prior scientific failures and the separate "
              "startup ERROR retain their original sources and grades; they provide no new capacity, learned or ordinary-cell credit.", "",
              "Paid time covers durable supervised child intervals; CPU capacity replay, parent certification/publication and queue wait "
              "are separate. No FLOPs or fair time-to-convergence comparison is inferred.", "",
              "Every GIF keeps its original terminal/sustained verdict. The adjacent added study grade requires the first five "
              "consecutive primary PASS observations, at least five later checks, and every later check passing. "
              "An original PASS can therefore have an added FAIL or INCOMPLETE.", "",
              "Native scope: 24 post-update observations of 20,000 noisy served outputs and five terminal checks; "
              "clean outputs are separate diagnostics. This does not supply Atlas19's independent 100,000-output certification.", "",
              "Verification: root separately replayed the CPU capacity sampler and certified retained numeric traces. "
              "This export consumed those SHA-bound certifications and copied GIFs; zero optimizer updates, model restores, "
              "sampling calls or rescoring occur in publication.", ""]
    return "\n".join(lines)


def publish(card_path, trusted_sha256, output, *, publisher_root, oracle=None):
    inputs = Inputs()
    card = inputs.json({"path": str(Path(card_path).resolve()), "sha256": trusted_sha256})
    if card.get("schema") != CARD_SCHEMA or card.get("verification") != VERIFICATION:
        raise ValueError("explicit CPU replay certification and trusted hash required")
    packet = inputs.json(card["combined"])
    certifier = card["certifier"]
    science = {key: value for key, value in certifier.items() if key not in {"root", "function"}}
    if certifier.get("function") != "combine_studies" or science != packet["source"]:
        raise ValueError("certifier is not bound to the combined source cohort")
    if (digest(packet["spec"].get("recipe_overrides")) != digest(OVERRIDES)
            or packet["spec"].get("family_budget_seconds") != FAMILY_REMAINING
            or packet["spec"].get("budget_seconds") != PAIR_REMAINING
            or packet["spec"].get("campaign_cap_seconds") != CAMPAIGN_CAP
            or science.get("discovery_inputs_sha256") != DISCOVERY
            or not {RUNNER, BINDER, PROTOCOL, PREVIOUS, DELEGATED} <= set(science.get("files_sha256", {}))):
        raise ValueError("fixed new trio/debit/cap or runner/binder/delegated/discovery binding changed")
    source_pin = source_identity(inputs, certifier["root"], science)
    own_root = Path(publisher_root).resolve()
    own_commit = subprocess.run(["git", "-C", str(own_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    own_pin = inputs.add(binding(Path(__file__).resolve()))
    if (file_sha(own_pin) != git_blob(own_root, own_commit, SELF)
            or not (own_root / SELF).is_file() or file_sha(own_root / SELF) != file_sha(own_pin)):
        raise ValueError("exporter bytes must be committed and equal their public source blob")
    oracle = Oracle(certifier["root"]) if oracle is None else oracle
    if packet.get("spec_sha256") != digest(packet["spec"]) or packet.get("selection") != oracle.select(packet):
        raise ValueError("forged configuration denominator/qualification selection")
    oracle.costs(packet)
    archives = check_archives(inputs, packet)
    snapshots = {family: snapshot(inputs, raw["execution_source"], raw["source"])
                 for family, raw in archives.items() if raw.get("execution_source")}
    capacities = capacity(inputs, packet, certifier["root"], oracle)
    previous, prior_paid = prior(inputs, oracle, packet, card)
    ordinary_paid = number(packet["measured_paid_seconds"], "ordinary paid")
    ordinary_reserved = number(packet["unmeasured_interrupt_reservation_seconds"], "ordinary reserved")
    equal_cost(packet["spent_seconds"], ordinary_paid + ordinary_reserved, "ordinary charge")
    family_caps = packet["spec"]["family_budget_seconds"]
    if (family_caps != packet["family_paid_budget_seconds"] or
            not math.isclose(sum(family_caps.values()), packet["spec"]["budget_seconds"], abs_tol=1e-8)):
        raise ValueError("family/pair remaining quotas differ")
    for family in FAMILIES:
        if archives[family]["spent_seconds"] > family_caps[family] + 1e-8:
            raise ValueError("family spent beyond unchanged remaining allowance")
    candidate_cap = number(packet["spec"]["candidate_budget_seconds"], "candidate cap")
    equal_cost(sum(family_caps.values()) + prior_paid, candidate_cap * 2, "overhead debit")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("publication requires a new output directory")
    protected_roots = [Path(certifier["root"]).resolve(), own_root,
                       *(Path(value["root"]).resolve() for value in previous["sources"]),
                       *(Path(value["snapshot_path"]).resolve() for value in snapshots.values()),
                       *(Path(pin["path"]).parent.resolve() for pin in packet["family_archives"]),
                       *(Path(row["study"]["path"]).parent.resolve() for row in previous["scientific"]),
                       *(Path(row["execution_snapshot"]["snapshot_path"]).resolve()
                         for row in previous["scientific"] + previous["engineering"])]
    if any(output.is_relative_to(root) for root in protected_roots):
        raise ValueError("publication output must stay outside scientific and raw evidence trees")
    rows = []
    pending_media = []
    for family in FAMILIES:
        archive = archives[family]
        trial = next(t for t in packet["trials"] if t["family"] == family)
        if any(capacities[family + "/" + row["id"]]["status"] != "SUPPORTED" for row in trial["cases"]):
            if (trial["status"] != "BLOCKED" or any(row["status"] != "BLOCKED"
                    or row.get("attempt_key") or row.get("receipt_path") for row in trial["cases"])):
                raise ValueError("negative capacity family cannot receive ordinary acquisition credit")
        for row in trial["cases"]:
            case = packet["case_definitions"][row["id"]]
            expected_recipe = oracle.recipe(case, family)
            if row["resolved_recipe_sha256"] != digest(expected_recipe):
                raise ValueError("recorded recipe hash omits the exact named factory horizon or tuple")
            evidence = receipt_projection(inputs, oracle, packet, archive, trial, row)
            supervisors = supervised(inputs, oracle, archive, trial, row)
            item = {"family": family, "config_id": trial["id"], "id": row["id"], "tier": row["tier"],
                    "question": case["goal"], "title": case["title"], "status": row["status"],
                    "original_gate": row.get("original_gate"), "study_gate": row.get("study_gate"),
                    "full_protocol_complete": row["full_protocol_complete"], "acquisition_hold": row.get("acquisition_hold"),
                    "reason": row.get("reason"), "capacity": capacities[family + "/" + row["id"]],
                    "requirements": {key: deepcopy(case.get(key)) for key in
                                     ("default_steps", "eval_samples", "batch_size", "evaluation_observations", "terminal_observations", "thresholds", "sampling")},
                    "case_sha256": row["case_sha256"], "recipe_overrides": trial["recipe_overrides"],
                    "resolved_recipe_sha256": row["resolved_recipe_sha256"], "runtime": archive.get("lane_runtime"),
                    "family_law": oracle.law(case, family),
                    "supervision": supervisors, "paid_seconds": row.get("paid_wall_seconds", 0.),
                    "conservative_reserved_seconds": row.get("unmeasured_interrupt_reserved_seconds", 0.), **evidence}
            if item["media"]:
                target = Path("media") / family / row["id"] / "goal.gif"
                pending_media.append((item["media"]["path"], target, deepcopy(item["media"])))
                item["media"]["original_path"] = item["media"]["path"]
                item["media"]["path"] = str(target)
                hold = item["acquisition_hold"] or {}
                item["media"]["caption"] = (f"Original gate: {item['original_gate'] or 'unavailable'}; added first-window hold: "
                    f"{item['study_gate'] or 'unavailable'}; confirm step {hold.get('acquired_step', 'unavailable')}; "
                    f"later {hold.get('hold_passed', 'unavailable')}/{hold.get('hold_checks', 'unavailable')} passing; at least 5 required, every later primary check must pass.")
            rows.append(item)
    results = {"schema": RESULT_SCHEMA, "study_id": packet["study_id"], "goal": packet["goal"],
               "selection": packet["selection"], "required_configurations": 2, "required_cases_per_configuration": 8,
               "required_cells": 16, "source": source_pin, "execution_snapshots": snapshots,
               "publisher": {"commit": own_commit, "git_path": SELF, "sha256": file_sha(own_pin)},
               "certification": binding(card_path), "combined": deepcopy(card["combined"]),
               "verification": {"root_certification": VERIFICATION, "publication_model_restores": 0,
                                "publication_draws": 0, "publication_rescores": 0, "publication_optimizer_updates": 0},
               "runtime_cohorts": {family: archives[family].get("lane_runtime") for family in FAMILIES},
               "counts": {"execution": dict(Counter(r["status"] for r in rows)),
                          "capacity": dict(Counter(r["capacity"]["status"] for r in rows)),
                          "original_gates": dict(Counter(r["original_gate"] or "UNAVAILABLE" for r in rows)),
                          "study_gates": dict(Counter(r["study_gate"] or "UNAVAILABLE" for r in rows)),
                          "goal_gifs": len(pending_media)},
               "costs": {"ordinary_paid_seconds": ordinary_paid, "ordinary_reserved_seconds": ordinary_reserved,
                         "prior_scientific_paid_seconds": sum(PRIOR_SCIENTIFIC.values()),
                         "prior_engineering_paid_seconds": PRIOR_ENGINEERING, "prior_total_paid_seconds": prior_paid,
                         "combined_charged_seconds": packet["spent_seconds"] + prior_paid,
                         "combined_cap_seconds": candidate_cap * 2, "remaining_cohort_cap_seconds": packet["spec"]["budget_seconds"],
                         "family_remaining_caps_seconds": family_caps, "clock": packet["paid_cost_scope"]},
               "prior_cohorts": previous, "speed_winner": None, "default_adoption": False,
               "family_law_scope": "Fast-only named KA2/K3P, no DV12; fixed-noise warmup and full-host schedules; cost-only prior slot inheritance",
               "scope": packet["scope"], "native_scope": packet["native_scope"], "cases": rows}
    # Complete all validations before creating any publication output.
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for source, target, pin in pending_media:
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha(destination) != pin["sha256"] or destination.stat().st_size != pin["bytes"]:
            raise ValueError("copied original media bytes differ")
    index = {"schema": "ka2_k3p_defaults_publication_input_index_v1", "files": list(inputs.files.values())}
    (output / "input-index.json").write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n")
    results["input_index"] = {**binding(output / "input-index.json"), "files": len(inputs.files)}
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "README.md").write_text(markdown(results))
    inputs.recheck()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    card = sub.add_parser("bind-certification", help="bind a separately completed root verification; does not run it")
    card.add_argument("--combined", type=Path, required=True)
    card.add_argument("--scientific-root", type=Path, required=True)
    card.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("publish")
    export.add_argument("--certification", type=Path, required=True)
    export.add_argument("--certification-sha256", required=True)
    export.add_argument("--publisher-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "bind-certification":
        if args.output.exists():
            raise ValueError("certification card requires a new path")
        value = bind_certification(args.combined, args.scientific_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(binding(args.output), sort_keys=True))
    else:
        result = publish(args.certification, args.certification_sha256, args.output, publisher_root=args.publisher_root)
        print(json.dumps({"counts": result["counts"], "costs": result["costs"], "selection": result["selection"]}, sort_keys=True))
    return 0  # Successful evidence publication, independent of scientific PASS/FAIL.


if __name__ == "__main__":
    raise SystemExit(main())
