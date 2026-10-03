"""One preregistered KA2/K3P defaults study; unchanged public API training loops.

Planning is read-only. Only an explicit run command initializes shared admission
and executes children. Two whole configurations retain all sixteen required
cells, including unreached cells. This is a provisional family-owned-law screen,
with no default adoption, cross-cohort credit or speed winner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RELATIVE = "reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py"
CAPACITY_RELATIVE = "reports/forge/ka2-k3p-defaults-20261003/bind_capacity.py"
DELEGATED_RELATIVE = "reports/forge/critic-balance-20261003/run_critic_balance.py"
SCHEMA = "particlegan_ka2_k3p_defaults_singleton_v1"
FAMILIES = ("ka2", "k3p")
OVERRIDES = {"lr": .006375, "prior_lr_mult": 1.0, "d_lr_mult": 1.0}
CASE_ROWS = (
    ("image-develop-img_intensity2-source-transpose12", 1, 180.),
    ("api-vector-two-broad", 1, 180.),
    ("api-grid100", 2, 2100.), ("api-rotated100", 2, 2100.),
    ("api-staggered100", 2, 2100.),
    ("api-vector-unequal-mass", 2, 180.), ("api-vector-anisotropic", 2, 180.),
    ("image-develop-img_bars4-source-transpose12", 2, 180.),
)
# Full discover() metadata: host, sampling, gates, horizons and scoring cadence.
CASE_SHA256 = (
    "e9fa47de6adbb31f083c0333b0c7c41352d2b890b16539d47839d3f9720c5dfb",
    "e293cf19f2f235b30807f9782ec65ceea3693f78e3459b9f7eb117179168c8fc",
    "93ca5686f7358c701c58b34798b8e1f784a339c958a0fe8e1c9f90af4867f279",
    "6e06e5ba492950d3f8063a27f82bf3d51e75574148235495cc1b331b7fb30754",
    "9ba09451032b3741df8a30c936bd81cc31038866dd2f4d0eb92966e48b292f48",
    "e29a36b2461a986a28ca950df869a4926f258b2bfca32bd46bc8b2188aebf32e",
    "2ec941011f8398672ce85b1d4df41c6e22dce2e0f98354a75f1d9a6794d4e1b6",
    "d3145a02dd5e98155b006bf3e99899003707fe8fcf62318ba55fcd2fc6374b4d",
)
TERMINAL = {"PASS", "FAIL", "INCOMPLETE", "BLOCKED", "ERROR", "INVALID"}
STATUSES = TERMINAL | {"UNKNOWN", "RUNNING"}
DISCOVERY_INPUTS = {"configs/forge/tasks/ring16_acquisition.json":
                    "e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1"}
PRIOR_ENGINEERING_PAID_SECONDS = 4.757908704923466
PRIOR_SCIENTIFIC_PAID_SECONDS = {"atlas": 55.03923988668248, "e22": 54.19710329291411}
PRIOR_TOTAL_PAID_SECONDS = 113.99425188452005
PRIOR_SOURCE_COMMIT = "488b792e2fb875894f017cf7043420f2bb66190f"
CAMPAIGN_CAP_SECONDS = 15360.
FAMILY_REMAINING_SECONDS = {"ka2": 7620.202851408394, "k3p": 7625.802896707086}
PAIR_REMAINING_SECONDS = 15246.00574811548
PRIOR_ARTIFACTS = {'combined.json': {'path': '/ml2/hypergan/forge-generator-step-20261003/combined.json',
                   'sha256': '2de9d080f8fdacdbb504f7fda6f7ee8ab2ba2dce00a454b117210dcc02a17b23',
                   'bytes': 1345994},
 'certification.json': {'path': '/ml2/hypergan/forge-generator-step-20261003/certification.json',
                        'sha256': '6ba725612752bfaabf1fecd2b3b2bed6f59d150364092b322337f6e6a029cb6f',
                        'bytes': 11733},
 'publication-v1/results.json': {'path': '/ml2/hypergan/forge-generator-step-20261003/publication-v1/results.json',
                                 'sha256': 'a9da2ae3af04d50873cc7f0199eedd34a0f2ba768efc86ada866a40d293c6c63',
                                 'bytes': 92866},
 'atlas/study.json': {'path': '/ml2/hypergan/forge-generator-step-20261003/atlas/study.json',
                      'sha256': '02ed35c689ae1a6f5df11bab6e42b2a07bb389d6ea3db146f3357e226411e2f7',
                      'bytes': 1339184},
 'e22/study.json': {'path': '/ml2/hypergan/forge-generator-step-20261003/e22/study.json',
                    'sha256': '6b0fcfd09df7d5b197cefc608a8ae59ef6583a8179e12685a6e117e9acaae5fa',
                    'bytes': 1338911}}
PROTOCOL_RELATIVE = "reports/forge/ka2-k3p-defaults-20261003/protocol.py"
PREVIOUS_RELATIVE = "reports/forge/generator-step-20261003/run_generator_step.py"
SLOT_PREDECESSOR = {"ka2": "atlas", "k3p": "e22"}



def prior_runner():
    """Unchanged generic accounting/resource helpers; no globals are patched."""
    name = "ka2_k3p_defaults_prior_orchestration"
    if name not in sys.modules:
        path = ROOT / DELEGATED_RELATIVE
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmarks.toy_audit import api_contract, api_family_search, api_run
    return api_contract, api_family_search, api_run


def load_bound(name, relative):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def protocol():
    return load_bound("ka2_k3p_defaults_protocol", PROTOCOL_RELATIVE)


def previous_study():
    return load_bound("ka2_k3p_previous_generator_study", PREVIOUS_RELATIVE)


def capacity_module():
    name = "ka2_k3p_defaults_capacity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / CAPACITY_RELATIVE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def digest(value):
    return modules()[1].digest(value)


def number(value, label):
    return prior_runner().number(value, label)


def default_spec(card_path, card_sha256, *, identifier="ka2-k3p-named-defaults-20261003-v1"):
    """The entire finite protocol; no arbitrary grids or per-toy knob winners."""
    return {
        "schema": SCHEMA, "id": identifier, "families": list(FAMILIES),
        "recipe_overrides": dict(OVERRIDES), "seed": 24002,
        "cases": [{"id": name, "tier": tier, "timeout_seconds": cap, "case_sha256": sha}
                  for (name, tier, cap), sha in zip(CASE_ROWS, CASE_SHA256)],
        "representation_card": {"path": str(card_path), "sha256": card_sha256},
        "candidate_budget_seconds": 7680.,
        "family_budget_seconds": dict(FAMILY_REMAINING_SECONDS),
        "budget_seconds": PAIR_REMAINING_SECONDS,
        "prior_carryover": prior_carryover(),
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "export_grace_seconds": 60., "frames": 9,
        "backend": "cuda", "speed_ranking": False, "default_adoption": False,
        "stability": {"confirmation_checks": 5, "post_confirmation_hold_checks": 5,
                      "first_window_only": True, "all_subsequent_primary_checks": True},
        "resources": {"host_memory_mb": 2048},
        "admission": {"physical_gpu": 1, "minimum_free_mib": 12288., "maximum_temperature_c": 82.,
                      "cuda_memory_fraction": .2, "torch_threads": 1, "exclusive": True},
    }


def validate_spec(spec, cases):
    contract, search, _ = modules()
    card = spec.get("representation_card")
    if (not isinstance(card, dict) or set(card) != {"path", "sha256"}
            or not isinstance(card["path"], str) or not card["path"]
            or not isinstance(card["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", card["sha256"])
            or not isinstance(spec.get("id"), str) or not contract.CASE_ID.fullmatch(spec["id"])):
        raise ValueError("an exact capacity card path/hash and study ID are required")
    expected = default_spec(card["path"], card["sha256"], identifier=spec["id"])
    if digest(spec) != digest(expected):
        raise ValueError("singleton knobs, cases, gates, caps, resources and hold protocol are fixed")
    if [(name, tier) for name, tier, _ in CASE_ROWS] != list(search.DEFAULT_CASES):
        raise ValueError("original required case order/tiers changed")
    for item in expected["cases"]:
        case = cases.get(item["id"])
        if case is None or digest(case) != item["case_sha256"]:
            raise ValueError("original required host/gate/horizon/sampling metadata changed")
        seed = case.get("protocol_seed", 24002)
        if type(seed) is not int or seed != spec["seed"]:
            raise ValueError("case seed differs from the public CLI's frozen protocol seed")
        contract.validate_recipe_overrides(case, "ka2", OVERRIDES)
        contract.validate_recipe_overrides(case, "k3p", OVERRIDES)
    return expected


def trial_id(family):
    return family + "--" + digest({"family": family, "overrides": OVERRIDES})


def source(cases):
    _, search, api = modules()
    value = search._source(cases)
    value["files_sha256"].update({path: api.file_hash(ROOT / path)
                                  for path in (RELATIVE, CAPACITY_RELATIVE, DELEGATED_RELATIVE, PROTOCOL_RELATIVE, PREVIOUS_RELATIVE)})
    # Public receipt source_identity binds Python files. Bind discovery data
    # separately and enforce it in the source snapshot/durable supervisor.
    if any(api.file_hash(ROOT / path) != sha for path, sha in DISCOVERY_INPUTS.items()):
        raise ValueError("original public discovery data changed")
    value["discovery_inputs_sha256"] = dict(DISCOVERY_INPUTS)
    return value


def freeze_execution_source(root, queue_root, expected):
    return prior_runner().freeze_execution_source(root, queue_root, expected)


def prior_carryover():
    return {"schema": "named_family_previous_debit_v1",
            "engineering_paid_seconds": PRIOR_ENGINEERING_PAID_SECONDS,
            "scientific_paid_seconds": dict(PRIOR_SCIENTIFIC_PAID_SECONDS),
            "total_paid_seconds": PRIOR_TOTAL_PAID_SECONDS,
            "slot_predecessor": dict(SLOT_PREDECESSOR),
            "slot_prior_paid_seconds": {"ka2": 59.797148591605946, "k3p": 54.19710329291411},
            "artifacts": deepcopy(PRIOR_ARTIFACTS),
            "claim": "Prior Atlas/E22 scientific failures and separate bootstrap ERROR retained; slot mapping transfers only paid costs, never grades"}


def verify_prior_carryover():
    """Bind latest/nested retained grades and durable costs without old science."""
    _, search, api = modules()
    from benchmarks.toy_audit import api_publish
    old = previous_study()
    old.verify_prior_carryover()  # Original setup + earlier two scientific FAILs.
    for pin in PRIOR_ARTIFACTS.values():
        path = Path(pin["path"])
        if not path.is_file() or path.stat().st_size != pin["bytes"] or api.file_hash(path) != pin["sha256"]:
            raise ValueError("prior science/publication artifact unavailable or changed")
    packet = json.loads(Path(PRIOR_ARTIFACTS["combined.json"]["path"]).read_text())
    card = json.loads(Path(PRIOR_ARTIFACTS["certification.json"]["path"]).read_text())
    report = json.loads(Path(PRIOR_ARTIFACTS["publication-v1/results.json"]["path"]).read_text())
    if (packet["source"]["commit"] != PRIOR_SOURCE_COMMIT
            or card["combined"] != PRIOR_ARTIFACTS["combined.json"]
            or report["certification"] != PRIOR_ARTIFACTS["certification.json"]
            or report["combined"] != PRIOR_ARTIFACTS["combined.json"]
            or report["costs"]["ordinary_paid_seconds"] != 54.87757138675079
            or report["costs"]["prior_scientific_paid_seconds"] != 54.3587717928458
            or report["costs"]["prior_engineering_paid_seconds"] != PRIOR_ENGINEERING_PAID_SECONDS
            or report["costs"]["prior_total_paid_seconds"] != old.PRIOR_TOTAL_PAID_SECONDS
            or report["costs"]["combined_charged_seconds"] != PRIOR_TOTAL_PAID_SECONDS
            or report["costs"]["ordinary_reserved_seconds"] != 0.):
        raise ValueError("prior source/certification/publication or one-time cost debit changed")
    old.verify_costs(packet)
    for family, paid in (("atlas", 28.03618986881338), ("e22", 26.841381517937407)):
        archive = json.loads(Path(PRIOR_ARTIFACTS[family + "/study.json"]["path"]).read_text())
        trial = next(t for t in archive["trials"] if t["family"] == family)
        row = trial["cases"][0]
        pin = PRIOR_ARTIFACTS[family + "/study.json"]
        bound = next(p for p in packet["family_archives"] if p["family"] == family)
        if (bound["path"] != pin["path"] or bound["sha256"] != pin["sha256"]
                or archive["executed_family"] != family or archive["source"] != packet["source"]
                or trial != next(t for t in packet["trials"] if t["family"] == family)
                or trial["status"] != "FAIL" or row["status"] != "FAIL"
                or row["original_gate"] != "FAIL" or row["study_gate"] != "FAIL"
                or row["full_protocol_complete"] is not True or row["paid_wall_seconds"] != paid
                or archive["measured_paid_seconds"] != paid or archive["spent_seconds"] != paid
                or archive["unmeasured_interrupt_reservation_seconds"] != 0.
                or any(r["status"] != "UNKNOWN" or r.get("attempt_key") or r.get("receipt_path") for r in trial["cases"][1:])):
            raise ValueError("prior completed science/cost/unknown denominator changed")
        old.durable_cost(archive, trial, row)
        path = Path(row["receipt_path"])
        if api.file_hash(path) != row["receipt_sha256"]:
            raise ValueError("prior public receipt changed")
        raw = api_publish.verify_run(path.parent)  # Pure retained grade/media.
        if (raw["verdict"] != row["original_gate"] or raw["source"]["commit"] != PRIOR_SOURCE_COMMIT
                or raw["completed_updates"] != 600 or raw["recipe"] != row["recipe"]
                or search.acquisition_hold(raw) != row["acquisition_hold"]):
            raise ValueError("prior original or added grade changed")
    return prior_carryover()


def capacity_outcomes(spec, cases):
    _, _, api = modules()
    path = Path(spec["representation_card"]["path"])
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file() or api.file_hash(path) != spec["representation_card"]["sha256"]:
        raise ValueError("capacity card unavailable or changed")
    raw_packet = json.loads(path.read_text())
    records = raw_packet.get("records")
    expected = {(family, name) for family in FAMILIES for name, _, _ in CASE_ROWS}
    if (not isinstance(records, list) or len(records) != 16
            or {(row.get("family"), row.get("case_id")) for row in records} != expected):
        raise ValueError("all sixteen unique current candidate-bound capacity outcomes are required")
    import torch
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        # The maintained parent resolves its CUDA runtime, but the capacity
        # witness is explicitly CPU-only. Recheck in a new source-bound process,
        # never silently reinterpret the proof in an initialized CUDA context.
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
        child = subprocess.run([sys.executable, str(ROOT / RELATIVE), "--verify-capacity",
                                str(path.resolve()), spec["representation_card"]["sha256"]],
                               cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300)
        if child.returncode != 0:
            raise ValueError("CPU-only capacity recertification rejected: " + child.stderr[-2000:])
        verified = json.loads(child.stdout)
        if set(verified) != {family + "/" + name for family, name in expected}:
            raise ValueError("CPU-only capacity recertification omitted a required outcome")
        return verified
    torch.set_num_threads(1)
    verifier = capacity_module()
    if verifier.SHARED_OVERRIDES != OVERRIDES:
        raise ValueError("capacity caller's singleton knobs differ")
    requested = [{"family": family, "case_id": name} for family in FAMILIES for name, _, _ in CASE_ROWS]
    if (raw_packet.get("schema") != verifier.SCHEMA or raw_packet.get("status") != "COMPLETE"
            or raw_packet.get("claim_scope") != verifier.CLAIM_SCOPE or raw_packet.get("required_records") != 16
            or raw_packet.get("requested_cells") != requested or raw_packet.get("ordinary_training_updates") != 0
            or raw_packet.get("fitting_updates") != 0 or raw_packet.get("ordinary_qualification_credit") is not False):
        raise ValueError("complete zero-update capacity packet/header/denominator required")
    records = verifier.verify_packet(raw_packet)["records"]
    if api.file_hash(path) != spec["representation_card"]["sha256"]:
        raise ValueError("capacity packet changed during verification")
    verified = {}
    for record in records:
        family, name = record["family"], record["case_id"]
        value = record  # verify_packet already strictly verified every record.
        if value.get("status") not in {"SUPPORTED", "UNRESOLVED", "BLOCKED"}:
            raise ValueError("unverified/unknown capacity outcomes cannot authenticate a family")
        verified[family + "/" + name] = value
    return verified


def plan_study(spec, *, cases=None):
    """Pure verification; no queue, reservations, models trained or GPU work."""
    contract, search, api = modules()
    import torch
    torch.set_num_threads(1)
    cases = contract.discover() if cases is None else cases
    spec = validate_spec(spec, cases)
    verify_prior_carryover()
    selected = {item["id"]: cases[item["id"]] for item in spec["cases"]}
    proofs = capacity_outcomes(spec, selected)
    trials = []
    for family in FAMILIES:
        blocked = [name for name in selected if proofs[family + "/" + name]["status"] != "SUPPORTED"]
        rows = [{**item, "status": "BLOCKED" if blocked else "UNKNOWN", "original_gate": None,
                 "study_gate": None, "full_protocol_complete": False,
                 "resolved_recipe_sha256": digest(protocol().resolved_recipe(selected[item["id"]], family, OVERRIDES))}
                for item in spec["cases"]]
        trials.append({"id": trial_id(family), "family": family, "recipe_overrides": dict(OVERRIDES),
                       "status": "BLOCKED" if blocked else "UNKNOWN", "capacity_blocked_cases": blocked,
                       "paid_wall_seconds": 0., "cases": rows})
    packet = {"schema": SCHEMA, "study_id": spec["id"], "spec": spec, "spec_sha256": digest(spec),
              "source": source(selected), "case_definitions": api.json_value(selected),
              "capacity_preflight": api.json_value(proofs), "trials": trials,
              "family_paid_budget_seconds": spec["family_budget_seconds"], "spent_seconds": 0.,
              "measured_paid_seconds": 0., "unmeasured_interrupt_reservation_seconds": 0.,
              "runtime_contract": search._runtime("cpu") | {"device": "cuda:0", "backend": "cuda"},
              "candidate_worst_case_reservation_seconds": 7680., "round_worst_case_reservation_seconds": 15360.,
              "goal": "ka2-k3p-family-owned-defaults", "speed_ranking": False, "default_adoption": False,
              "prior_paid_seconds": PRIOR_TOTAL_PAID_SECONDS, "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
              "paid_cost_scope": "New cohort durable supervised child intervals plus conservative interruption reserves; "
                                 "prior science and engineering are separately bound and debited once from remaining quotas; "
                                 "CPU capacity verification, queue wait and parent certification are separate diagnostics",
              "scope": "Named KA2/K3P family-owned fast/scheduled/fixed-noise laws across eight public API cases; distinct from Atlas/E22 law, no old/MoG/default/speed qualification",
              "native_scope": "24 post-update noisy-fast 20k primary checks and five terminal checks; clean diagnostics separate, no independent 100k gate",
              "family_laws": {family: {name: protocol().family_law(case, family) for name, case in selected.items()} for family in FAMILIES},
              "media_scope": "Original public GIF verdict is terminal/sustained; study acquisition/hold grade is paired separately"}
    packet["selection"] = select_results(packet)
    return packet


def select_results(packet):
    """Keep both configurations and all sixteen cases in the denominator."""
    validate_spec(packet["spec"], packet["case_definitions"])
    trials = packet.get("trials", [])
    if (len(trials) != 2 or {row.get("id") for row in trials} != {trial_id(f) for f in FAMILIES}
            or {row.get("family") for row in trials} != set(FAMILIES)):
        raise ValueError("exactly two singleton family configurations are required")
    for trial in trials:
        if (trial.get("recipe_overrides") != OVERRIDES or trial["id"] != trial_id(trial["family"])
                or trial.get("status") not in STATUSES
                or len(trial.get("cases", [])) != 8):
            raise ValueError("changed configuration/status/required denominator")
        prefix = True
        for row, expected in zip(trial["cases"], packet["spec"]["cases"]):
            if any(row.get(key) != value for key, value in expected.items()) or row.get("status") not in STATUSES:
                raise ValueError("changed case/gate/cap/status or required denominator")
            executed = row.get("attempt_key") or row.get("receipt_path") or row.get("paid_wall_seconds", 0.)
            if not prefix and (executed or row["status"] in {"PASS", "RUNNING"}):
                raise ValueError("downstream work admitted before all preceding gates passed")
            prefix = prefix and row["status"] == "PASS"
        if trial["status"] == "PASS" and not all(row["status"] == "PASS" for row in trial["cases"]):
            raise ValueError("whole PASS lacks a required case")
    concluded = all(trial["status"] in TERMINAL for trial in trials)
    complete = all(trial["status"] in {"PASS", "FAIL"} for trial in trials)
    eligible = [trial["id"] for trial in trials if trial["status"] == "PASS"]
    count = lambda t: tuple(sum(row["status"] == "PASS" and row["tier"] == tier for row in t["cases"]) for tier in (1, 2))
    maximum = max(map(count, trials))
    return {"required_configurations": 2, "required_cells": 16, "required_cases_per_config": 8,
            "attempts_concluded": concluded, "comparison_complete": complete,
            "outcome": "pending" if not concluded else "incomplete_comparison" if not complete else
                       "scoped_fully_qualified" if eligible else "best_observed",
            "fully_qualified_ids": sorted(eligible),
            "best_observed_ids": sorted(t["id"] for t in trials if count(t) == maximum) if concluded else [],
            "speed_winner": None, "default_adoption": False}


def validate_lane(device, *, telemetry=None):
    return prior_runner().validate_lane(device, telemetry=telemetry)


def child_command(spec, trial, row, case_root, device):
    _, search, _ = modules()
    command = search.child_command(spec, trial, row, case_root, device)
    return [command[0], "-u", RELATIVE, "--child", *command[3:]]


def child_main(argv):
    """Resource bootstrap only; all training/observations stay in api_run.main."""
    parser = argparse.ArgumentParser()
    for key in ("case", "output", "recipe", "device", "recipe-overrides", "wall-cap-seconds", "frames"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    caps = {name: cap for name, _, cap in CASE_ROWS}
    if (args.case not in caps or args.recipe not in FAMILIES or json.loads(args.recipe_overrides) != OVERRIDES
            or args.device != "cuda:0" or float(args.wall_cap_seconds) != caps[args.case] or args.frames != "9"):
        raise ValueError("child command differs from frozen singleton protocol")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("child must have exactly physical GPU1 visible")
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    contract, _, api = modules()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    # Imports bind both report-owned helpers into the public source receipt;
    # they invoke no plan, verification, replay, draw or model construction.
    capacity_module()
    prior_runner()
    previous_study()
    protocol()
    case = contract.discover()[args.case]
    if digest(case) != CASE_SHA256[list(caps).index(args.case)] or case.get("protocol_seed", 24002) != 24002:
        raise ValueError("child host/protocol metadata changed")
    return api.main(argv)


def media_context(verified):
    return prior_runner().media_context(verified)


def outcome(path, packet, trial, row, returncode):
    contract, search, api = modules()
    receipt_path = Path(path) / "receipt.json"
    if not receipt_path.is_file():
        return {"status": "ERROR", "reason": "child ended without a durable public receipt",
                "original_gate": None, "study_gate": None, "full_protocol_complete": False}
    raw = json.loads(receipt_path.read_text())
    if raw.get("status") == "COMPLETE":
        value = protocol().verify_case(path, packet["case_definitions"][row["id"]], trial["family"], OVERRIDES,
                                   packet["source"], returncode=returncode, runtime=packet["lane_runtime"],
                                   wall_cap_seconds=row["timeout_seconds"], frames=9)
        value["media_context"] = media_context(value)
        return value
    if raw.get("status") not in {"INCOMPLETE", "BLOCKED", "ERROR"}:
        raise ValueError("unknown/forged incomplete public execution status")
    if raw.get("default_protocol_complete") or raw.get("passed"):
        raise ValueError("partial execution cannot claim completed scientific PASS")
    case = packet["case_definitions"][row["id"]]
    receipt_protocol = raw.get("protocol", {})
    metric_steps = contract.evaluation_steps(case["default_steps"], min(case["default_steps"], contract.metric_observations(case)) + 1)
    media_steps = contract.evaluation_steps(case["default_steps"], 9)
    expected_protocol = {"updates": case["default_steps"], "default_updates": case["default_steps"],
                         "evaluation_samples": case["eval_samples"], "default_evaluation_samples": case["eval_samples"],
                         "evaluation_steps": sorted(set(metric_steps) | set(media_steps)),
                         "metric_evaluation_steps": metric_steps, "media_steps": media_steps,
                         "metric_observations": contract.metric_observations(case), "media_frames": 9,
                         "terminal_observations": case.get("terminal_observations", 5),
                         "wall_cap_seconds": row["timeout_seconds"]}
    completed = raw.get("completed_updates")
    if (digest(raw.get("case")) != digest(case) or raw.get("seed") != 24002
            or raw.get("requested_recipe_overrides") != OVERRIDES or receipt_protocol != expected_protocol
            or type(completed) is not int or not 0 <= completed <= case["default_steps"]
            or raw.get("source", {}).get("commit") != packet["source"]["commit"]
            or any(raw.get("source", {}).get("files_sha256", {}).get(name) != sha for name, sha in packet["source"]["files_sha256"].items())
            or any(raw.get("runtime", {}).get(key) != value for key, value in packet["lane_runtime"].items()
                   if key != "cuda_device_model" or raw.get("recipe") is not None)
            or (raw.get("recipe") is not None and raw["recipe"] != protocol().resolved_recipe(case, trial["family"], OVERRIDES))
            or (completed > 0 and raw.get("recipe") is None)):
        raise ValueError("partial execution case/seed/recipe/source/runtime/protocol attribution changed")
    observation_steps = [record.get("step") for record in raw.get("observations", [])]
    prefix = [step for step in expected_protocol["evaluation_steps"] if step <= completed]
    if raw.get("partial_terminal_observation_added") and completed not in prefix:
        prefix.append(completed)
    if observation_steps != prefix and not (not observation_steps and completed == 0) and not (
            raw["status"] == "ERROR" and observation_steps == prefix[:len(observation_steps)]):
        raise ValueError("partial observed states are not the original completed execution prefix")
    for name, binding in raw.get("artifacts", {}).items():
        artifact = Path(path) / name
        if (Path(name).name != name or not artifact.is_file() or api.file_hash(artifact) != binding["sha256"]
                or artifact.stat().st_size != binding["bytes"]):
            raise ValueError("partial public artifact changed")
    return {"status": raw["status"], "reason": "; ".join(raw.get("failed_bounds", [])),
            "original_gate": None, "reported_original_verdict": raw.get("verdict"), "study_gate": None,
            "full_protocol_complete": False, "completed_updates": raw.get("completed_updates", 0),
            "receipt_path": str(receipt_path), "receipt_sha256": api.file_hash(receipt_path),
            "artifacts": raw.get("artifacts", {})}


def verify_costs(packet):
    prior_runner().verify_costs(packet)
    if (packet.get("prior_paid_seconds") != PRIOR_TOTAL_PAID_SECONDS
            or packet.get("campaign_cap_seconds") != CAMPAIGN_CAP_SECONDS
            or packet["spec"].get("campaign_cap_seconds") != CAMPAIGN_CAP_SECONDS
            or packet["spent_seconds"] > packet["spec"]["budget_seconds"]):
        raise ValueError("new/prior debit or shared finite campaign cap changed")


def durable_cost(packet, trial, row):
    return prior_runner().durable_cost(packet, trial, row)


def recertify_archive(packet):
    contract, search, api = modules()
    cases = contract.discover()
    validate_spec(packet["spec"], cases)
    selected = {item["id"]: cases[item["id"]] for item in packet["spec"]["cases"]}
    if (packet.get("schema") != SCHEMA or packet.get("spec_sha256") != digest(packet["spec"])
            or packet["case_definitions"] != api.json_value(selected)
            or packet["source"]["files_sha256"] != source(selected)["files_sha256"]
            or packet["source"].get("discovery_inputs_sha256") != source(selected)["discovery_inputs_sha256"]):
        raise ValueError("archived source/protocol/case identities changed")
    verify_prior_carryover()
    # All 16 outcomes still authenticate the same current card; negative
    # outcomes block their own family only and are never replaced by old credit.
    if capacity_outcomes(packet["spec"], selected) != packet["capacity_preflight"]:
        raise ValueError("archived capacity evidence changed")
    select_results(packet)
    verify_costs(packet)
    for trial in packet["trials"]:
        blocked = [name for name in selected if packet["capacity_preflight"][trial["family"] + "/" + name]["status"] != "SUPPORTED"]
        if trial.get("capacity_blocked_cases") != blocked or (blocked and (
                trial["status"] != "BLOCKED" or any(row["status"] != "BLOCKED" for row in trial["cases"]))):
            raise ValueError("unresolved capacity family cannot claim scientific execution or PASS")
        if trial["family"] != packet["executed_family"]:
            if trial.get("paid_wall_seconds") or any(row.get("attempt_key") for row in trial["cases"]):
                raise ValueError("one family per independent archive")
            continue
        for row in trial["cases"]:
            if row.get("attempt_key") and row["status"] != "RUNNING":
                durable_cost(packet, trial, row)
            if not row.get("receipt_path"):
                if row["status"] in {"PASS", "FAIL"} or row.get("full_protocol_complete"):
                    raise ValueError("scientific status lacks a bound raw public receipt")
                continue
            if not row.get("attempt_key"):
                raise ValueError("public receipt lacks its source-bound shared physical attempt")
            path = Path(row["receipt_path"])
            if not path.is_file() or api.file_hash(path) != row.get("receipt_sha256"):
                raise ValueError("retained raw receipt changed")
            verified = outcome(path.parent, packet, trial, row, row.get("child_returncode"))
            if row.get("paid_wall_seconds", 0.) < verified.get("elapsed_seconds", 0.):
                raise ValueError("paid time is smaller than certified acquisition time")
            for key, value in verified.items():
                if row.get(key) != value:
                    raise ValueError("saved grade/completion/media/case metadata differ from raw evidence")


def human_readout(packet):
    lines = ["# Named KA2/K3P defaults study", "",
             "Two configurations, eight required cases each. Unreached cases remain UNKNOWN. "
             "This provisional named-family fast/scheduled-law screen grants no defaults, fastest-family or Forge MoG credit.", "",
             "Serving law: fast-only, with no DV12 controller or latent perturbation. "
             "Inherited generic policy captions in the original GIF do not change this family-owned law.", "",
             "Native scope: 24 post-update observations of 20,000 served outputs; "
             "five terminal observations. No independent 100,000-output gate.", "",
             f"This cohort measured paid seconds: {packet['measured_paid_seconds']:.6f}; "
             f"conservative interruption reserve: {packet['unmeasured_interrupt_reservation_seconds']:.6f} seconds. "
             f"Charged spend / cohort ceiling: {packet['spent_seconds']:.6f} / {packet['spec']['budget_seconds']:.6f} seconds; "
             f"unused allowance: {packet['spec']['budget_seconds'] - packet['spent_seconds']:.6f} seconds.", "",
             f"Prior science costs: Atlas {PRIOR_SCIENTIFIC_PAID_SECONDS['atlas']:.6f}s and E22 {PRIOR_SCIENTIFIC_PAID_SECONDS['e22']:.6f}s; "
             f"prior bootstrap ERROR: {PRIOR_ENGINEERING_PAID_SECONDS:.6f}s. Total {PRIOR_TOTAL_PAID_SECONDS:.6f}s debited once. "
             "Combined finite contrast ceiling remains 15,360 seconds; KA2/K3P inherit only the named bookkeeping slots, never old grades.", "",
             "Measured paid time covers supervised children; conservative interruption reserves are recorded separately. "
             "CPU capacity verification, queue wait and parent certification are separate diagnostics.", "",
             "| Family | Required case / question | Execution / study | Original gate | First-window hold | Goal GIF |",
             "| --- | --- | --- | --- | --- | --- |"]
    for trial in packet["trials"]:
        for row in trial["cases"]:
            hold = row.get("acquisition_hold", {})
            details = (f"{row.get('study_gate') or 'unavailable'}; confirm step {hold.get('acquired_step', 'unavailable')}; "
                       f"later {hold.get('hold_passed', 'unavailable')}/{hold.get('hold_checks', 'unavailable')} "
                       "passed (at least 5 required; every later primary check must pass)")
            link = "unavailable"
            if row.get("receipt_path") and row.get("artifacts", {}).get("goal.gif"):
                link = f"[original goal GIF]({Path(row['receipt_path']).parent / 'goal.gif'})"
            goal = packet["case_definitions"][row["id"]]["goal"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {trial['family']} | `{row['id']}`: {goal} | {row['status']} | "
                         f"{row.get('original_gate') or 'unavailable'} | {details} | {link} |")
    lines += ["", "The original GIF displays its original terminal/sustained grade. "
              "The first-window hold grade above is an additional study requirement; "
              "an original PASS does not replace a failed or incomplete hold.", ""]
    return "\n".join(lines)


def save(path, packet):
    _, _, api = modules()
    packet["spent_seconds"] = sum(t["paid_wall_seconds"] for t in packet["trials"])
    packet["measured_paid_seconds"] = sum(r.get("paid_wall_seconds", 0.) for t in packet["trials"] for r in t["cases"])
    packet["unmeasured_interrupt_reservation_seconds"] = sum(r.get("unmeasured_interrupt_reserved_seconds", 0.) for t in packet["trials"] for r in t["cases"])
    packet["selection"] = select_results(packet)
    verify_costs(packet)
    from experiments.forge.contracts import atomic_json
    atomic_json(Path(path), api.json_value(packet))
    Path(path).with_name("README.md").write_text(human_readout(packet))


def allow_next(packet, trial):
    return prior_runner().allow_next(packet, trial)


def retain_terminal(packet, trial, row, coordinator, admission):
    """Certify durable completed work or charge an interrupted full allowance."""
    if admission["status"] == "completed":
        result = deepcopy(admission["result"])
        _, _, api = modules()
        if result.get("receipt_path"):
            path = Path(result["receipt_path"])
            if not path.is_file() or api.file_hash(path) != result.get("receipt_sha256"):
                raise ValueError("shared completed public receipt changed; no automatic retry")
            verified = outcome(path.parent, packet, trial, row, result.get("child_returncode"))
            if any(result.get(key) != value for key, value in verified.items()):
                raise ValueError("shared completed grade differs from its unchanged raw evidence")
        elif result.get("status") in {"PASS", "FAIL"} or result.get("full_protocol_complete"):
            raise ValueError("shared scientific result lacks its original public receipt")
        row.update(result)
        durable_cost(packet, trial, row)
    elif admission["status"] == "awaiting_certification":
        terminal, command = admission["terminal"], admission["command"]
        path = Path(command[command.index("--output") + 1]) / row["id"]
        try:
            row.update(outcome(path, packet, trial, row, terminal["child_returncode"]))
        except Exception as error:
            row.update(status="INVALID", reason=f"{type(error).__name__}: {error}")
        row.update(command=command, child_returncode=terminal["child_returncode"], paid_wall_seconds=terminal["paid_wall_seconds"])
        coordinator.complete(row["attempt_key"], deepcopy(row))
    else:
        terminal = admission.get("terminal")
        paid = terminal["paid_wall_seconds"] if terminal else 0.
        allowance = row["timeout_seconds"] + 60.
        expected_charge = max(allowance, paid)
        if admission.get("charged_seconds") != expected_charge:
            raise ValueError("recovered shared charge differs from original conservative allowance")
        row.update(status="ERROR" if terminal and terminal.get("attempt_status") == "error" else "INCOMPLETE",
                   reason="interrupted attempt retained; no automatic retry",
                   paid_wall_seconds=paid, child_returncode=terminal.get("child_returncode") if terminal else None,
                   command=admission.get("command"))
        row.update(recovered_interruption=True,
                   unmeasured_interrupt_reserved_seconds=max(0., expected_charge - paid))
    row["reused_physical_attempt"] = True


def run_owned(packet, output, family, coordinator, study_lease):
    registration = Path(output) / "study.json"
    trial = next(t for t in packet["trials"] if t["family"] == family)
    if trial["status"] != "UNKNOWN":
        return packet
    while True:
        row = allow_next(packet, trial)
        if row is None:
            trial["status"] = "PASS" if all(r["status"] == "PASS" for r in trial["cases"]) else "INCOMPLETE"
            trial["reason"] = "whole protocol complete" if trial["status"] == "PASS" else "next unchanged full allowance unavailable"
            break
        # Check telemetry before creating a paid admission transaction; the
        # coordinator independently owns exclusive shared host/GPU admission.
        try:
            packet["last_admission_telemetry"] = validate_lane("cuda:0")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            packet["coordinator"]["waiting_reason"] = str(error)
            save(registration, packet)
            return packet
        key = coordinator.attempt_key(packet, trial, row)
        row["attempt_key"] = key
        with coordinator.admit(key, packet, row, "cuda:0") as (admission, attempt_lease):
            if admission["status"] == "busy":
                row.pop("attempt_key", None)
                packet["coordinator"]["waiting_reason"] = admission["reason"]
                save(registration, packet)
                return packet
            if admission["status"] in {"completed", "interrupted", "awaiting_certification"}:
                retain_terminal(packet, trial, row, coordinator, admission)
            else:
                case_root = Path(output) / trial["id"] / row["id"]
                log = case_root.parent / (row["id"] + ".log")
                allowance = row["timeout_seconds"] + 60.
                interruption = None
                if case_root.exists() or log.exists():
                    row.update(status="INCOMPLETE", reason="orphan artifacts; unchanged work cannot be retried",
                               paid_wall_seconds=0., unmeasured_interrupt_reserved_seconds=allowance)
                else:
                    log.parent.mkdir(parents=True, exist_ok=True)
                    command = child_command(packet["spec"], trial, row, case_root, "cuda:0")
                    row.update(status="RUNNING", command=command, log_path=str(log), reserved_seconds=allowance)
                    packet["coordinator"].pop("waiting_reason", None)
                    save(registration, packet)
                    started = time.monotonic()
                    try:
                        child = coordinator.launch(command, packet, log, (study_lease, attempt_lease), allowance)
                        row.update(child_returncode=child.returncode, paid_wall_seconds=child.paid_wall_seconds)
                        row.update(outcome(case_root, packet, trial, row, child.returncode))
                    except BaseException as error:
                        terminal_path = coordinator.root / "policy/attempts" / key / "supervisor-terminal.json"
                        terminal = json.loads(terminal_path.read_text()) if terminal_path.is_file() else None
                        request_path = terminal_path.with_name("supervisor-request.json")
                        if terminal is not None and (not request_path.is_file() or
                                json.loads(request_path.read_text()).get("token") != terminal.get("token")):
                            terminal = None
                        row.update(status="INCOMPLETE" if isinstance(error, (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit)) else "INVALID",
                                   reason=f"{type(error).__name__}: {error}",
                                   paid_wall_seconds=terminal["paid_wall_seconds"] if terminal else 0.,
                                   child_returncode=terminal.get("child_returncode") if terminal else None)
                        if terminal is None:
                            row["unmeasured_interrupt_reserved_seconds"] = allowance
                        elif terminal.get("attempt_status") != "completed":
                            row.update(status="ERROR" if terminal.get("attempt_status") == "error" else "INCOMPLETE",
                                       recovered_interruption=True,
                                       unmeasured_interrupt_reserved_seconds=max(0., allowance - row["paid_wall_seconds"]))
                        if not isinstance(error, Exception):
                            interruption = error
                    row["validation_and_parent_seconds"] = max(0., time.monotonic() - started - row["paid_wall_seconds"])
                coordinator.complete(key, deepcopy(row))
                if interruption is not None:
                    trial["paid_wall_seconds"] = sum(r.get("paid_wall_seconds", 0.) + r.get("unmeasured_interrupt_reserved_seconds", 0.) for r in trial["cases"])
                    trial["status"] = row["status"]
                    save(registration, packet)
                    raise interruption
        trial["paid_wall_seconds"] = sum(r.get("paid_wall_seconds", 0.) + r.get("unmeasured_interrupt_reserved_seconds", 0.) for r in trial["cases"])
        if trial["paid_wall_seconds"] > packet["family_paid_budget_seconds"][family]:
            trial["status"] = "INCOMPLETE"
            trial["reason"] = "actual paid time exceeded the frozen family ceiling"
            break
        if row["status"] != "PASS":
            trial["status"] = row["status"]
            break
        save(registration, packet)
    save(registration, packet)
    return packet


def run_study(spec, output, *, family, queue_root=None):
    """Explicit root-owned execution; shared leases and immutable snapshots."""
    if family not in FAMILIES:
        raise ValueError("exact declared family required")
    contract, search, api = modules()
    packet = plan_study(spec)  # Invalid/missing capacity cannot initialize queues.
    selected = next(t for t in packet["trials"] if t["family"] == family)
    if selected["status"] == "BLOCKED":
        packet["executed_family"] = family
        packet["lane_runtime"] = None
        Path(output).mkdir(parents=True, exist_ok=False)
        save(Path(output) / "study.json", packet)
        return packet
    verify_prior_carryover()
    validate_lane("cuda:0")
    import torch
    torch.set_num_threads(1)
    from experiments.forge.__main__ import queue_location
    from experiments.forge.policy_execution import PolicyCoordinator
    from experiments.forge.sources import verify_snapshot
    runtime = search._runtime("cuda:0")
    coordinator = PolicyCoordinator(queue_location(contract.ROOT, queue_root), report_root=contract.ROOT / "reports/forge")
    packet["execution_source"] = freeze_execution_source(contract.ROOT, coordinator.root, packet["source"])
    verify_snapshot(Path(packet["execution_source"]["snapshot_path"]), packet["execution_source"])
    key, canonical = coordinator.register(api.json_value(packet), Path(output).resolve(), family, runtime)
    with coordinator.study_lease(key) as lease:
        if lease is not None:
            coordinator.recover()
            packet = json.loads((canonical / "study.json").read_text())
            recertify_archive(packet)
            run_owned(packet, canonical, family, coordinator, lease)
    return coordinator.publish_attachment(key, canonical, Path(output).resolve())


def combine_studies(paths):
    _, _, api = modules()
    packets = [json.loads(Path(path).read_text()) for path in paths]
    if len(packets) != 2 or {p.get("executed_family") for p in packets} != set(FAMILIES):
        raise ValueError("one independent archive per declared family required")
    for key in ("spec", "spec_sha256", "source", "case_definitions", "capacity_preflight", "runtime_contract"):
        if packets[0].get(key) != packets[1].get(key):
            raise ValueError(f"family {key} cohorts differ")
    for packet in packets:
        recertify_archive(packet)
    from experiments.forge.policy_execution import scientific_runtime
    runtimes = [p["lane_runtime"] for p in packets if p.get("lane_runtime") is not None]
    if len(runtimes) == 2 and scientific_runtime(runtimes[0]) != scientific_runtime(runtimes[1]):
        raise ValueError("cannot compare differing physical hardware/software runtime cohorts")
    result = deepcopy(packets[0])
    result.pop("executed_family", None)
    result.pop("lane_runtime", None)
    result["family_archives"] = [{"family": p["executed_family"], "path": str(Path(path).resolve()),
                                  "sha256": api.file_hash(path), "runtime": p.get("lane_runtime")}
                                 for path, p in zip(paths, packets)]
    result["trials"] = [next(t for t in p["trials"] if t["family"] == p["executed_family"]) for p in packets]
    result["spent_seconds"] = sum(p["spent_seconds"] for p in packets)
    result["measured_paid_seconds"] = sum(p["measured_paid_seconds"] for p in packets)
    result["unmeasured_interrupt_reservation_seconds"] = sum(p["unmeasured_interrupt_reservation_seconds"] for p in packets)
    result["selection"] = select_results(result)
    verify_costs(result)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--child":
        return child_main(argv[1:])
    if argv and argv[0] == "--verify-capacity":
        if len(argv) != 3 or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("internal capacity verification requires a fresh CPU-only process")
        contract, _, api = modules()
        import torch
        torch.set_num_threads(1)
        spec = default_spec(argv[1], argv[2])
        cases = contract.discover()
        validate_spec(spec, cases)
        print(json.dumps(api.json_value(capacity_outcomes(spec, cases)), sort_keys=True, allow_nan=False))
        return 0
    modules()  # Direct invocation outside the checkout needs the root bootstrap.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("spec", "plan", "run", "combine"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--queue-root", type=Path)
    parser.add_argument("--archive", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    _, _, api = modules()
    if args.stage == "spec":
        packet = default_spec(str(args.input.resolve()), api.file_hash(args.input))
    elif args.stage == "combine":
        packet = combine_studies([args.input, *args.archive])
    else:
        spec = json.loads(args.input.read_text())
        if args.stage == "plan":
            packet = plan_study(spec)
        else:
            if args.output is None or args.family is None:
                parser.error("run requires --output and --family")
            packet = run_study(spec, args.output, family=args.family, queue_root=args.queue_root)
    if args.stage != "run" and args.output is not None:
        api.write_json(args.output, packet)
    print(json.dumps(api.json_value(packet), indent=2, sort_keys=True, allow_nan=False))
    if args.stage == "run":
        status = next(t["status"] for t in packet["trials"] if t["family"] == args.family)
        return 0 if status == "PASS" else 2 if status in {"UNKNOWN", "RUNNING"} else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
