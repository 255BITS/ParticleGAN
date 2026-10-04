"""Portable draw-free projection of one root-certified Gaussian terminal cut."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import zipfile

from PIL import Image

ORIGIN = "b1c33b79efaea16fe8c705af0f5c3af31d369fdf"
SOURCE_DIGEST = "889fcdb097e3ed6e45296faf83b92364b41441d28cd3030c5eb8ce2fd127a2bd"
OBSERVER = "benchmarks/toy_audit/gaussian2d_observer.py"
OBSERVER_SHA = "ef607609d377c9747d74a3e490a0092e59f1a4b027f99e23f60a10b014ba0717"
API_RUN = "benchmarks/toy_audit/api_run.py"
API_RUN_SHA = "c3d54398256eb0aba6dd20bb77e6e9a9a71bb29cc5bd4df8b8f9bce1ef5687b9"
A_PROTOCOL = "reports/forge/gaussian2d-current-c6-20261004/protocol.json"
WRAPPER = "observer-controls/gaussian2d-v1/run_supervised.py"
WRAPPER_SHA = "1152522e2cc3215e0e707cf265697f596f3aa19244d905ffe0eea411b5daee97"
CASE = "api_gaussian2d_c6_observer_gpu_v1"
FAMILY = "atlas_gaussian2d_c6_observer"
COHORT = "api_gaussian2d_c6_gpu_v1"
LIMIT = 180
STEPS = [0, *(math.ceil(1000 * i / 24) for i in range(1, 25))]
MEDIA = list(range(0, 1001, 125))
THRESHOLDS = [["sample_count", ">=", 1024], ["mean_error_sigma", "<=", .10],
    ["min_cov_eigen", ">=", .85], ["max_cov_eigen", "<=", 1.15],
    ["radial_ks", "<=", .075], ["max_projection_ks", "<=", .06]]
OWNERS = {"continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
    "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard"}
HOOKS = ["begin_step", "after_critic_step", "after_generator_backward", "after_generator_step", "finish_step"]
NO_CREDIT = dict(ordinary_current_26_slot_credit=False, new_catalog_question=False,
    historical_credit=False, default_adoption=False, speed_ranking=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    require(type(value) is int and value >= 0, "invalid integer evidence count")
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid evidence cost")
    return value


def hex64(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "missing SHA256 evidence")
    return value


class Inputs:
    def __init__(self):
        self.items = {}

    def check(self, label, value, expected=None):
        require(isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"}, "exact file pin required: " + label)
        path = Path(value["path"])
        require(path.is_absolute() and not path.is_symlink() and path.is_file(), "missing/local changed input: " + label)
        require(count(value["bytes"]) == path.stat().st_size and hex64(value["sha256"]) == sha(path), "changed input bytes: " + label)
        require(expected is None or path.resolve() == Path(expected).resolve(), "borrowed input: " + label)
        public = dict(label=label, sha256=value["sha256"], bytes=value["bytes"], availability="LOCAL_ONLY")
        require(label not in self.items or self.items[label] == public, "conflicting input label")
        self.items[label] = public
        return path.resolve()

    def file(self, label, path):
        return self.check(label, pin(path))


def check_source(packet, inputs, source_manifest):
    source = packet["source"]
    require(source == packet["execution_source"], "executed source differs")
    require(source.get("schema_version") == 1 and type(source.get("schema_version")) is int
        and source.get("origin_commit") == ORIGIN and source.get("digest") == SOURCE_DIGEST,
        "wrong scientific origin/source cohort")
    files = source.get("files", {})
    require(files and digest(files) == source["digest"], "bad full source digest")
    snapshot = Path(source["snapshot_path"])
    header = read(inputs.check("source:forge-source.json", source_manifest, snapshot / "forge-source.json"))
    require(header == {key: value for key, value in source.items() if key != "snapshot_path"}, "snapshot origin/header differs")
    for relative, value in files.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and relative == path.as_posix(), "unsafe source path")
        actual = snapshot / path
        require(not actual.is_symlink() and actual.resolve() == snapshot.resolve() / path, "foreign/aliased source path")
        inputs.check("source:" + relative, dict(path=str(actual.resolve()), sha256=hex64(value), bytes=actual.stat().st_size), actual)
    parent = packet["parent_source"]
    additions = {"wrapper": WRAPPER, "protocol": "observer-controls/gaussian2d-v1/protocol.json",
        "proposal": "observer-controls/gaussian2d-v1/proposal.json", "scope": "observer-controls/gaussian2d-v1/PROPOSAL.md"}
    require(parent.get("origin_commit") == ORIGIN and parent.get("schema_version") == 1
        and digest(parent["files"]) == parent.get("digest")
        and set(files) == set(parent["files"]) | set(additions.values())
        and all(files.get(path) == value for path, value in parent["files"].items()), "parent/derived closure differs")
    require(set(packet["inputs"]) == set(additions), "missing external input closure")
    for name, relative in additions.items():
        inputs.check("supervisor-input:" + name, packet["inputs"][name])
        require(files[relative] == packet["inputs"][name]["sha256"], "snapshot/input binding differs")
    require(files.get(OBSERVER) == OBSERVER_SHA and files.get(API_RUN) == API_RUN_SHA
        and files.get(WRAPPER) == WRAPPER_SHA, "observer/hooks/supervisor bytes differ")
    proposal = read(Path(packet["inputs"]["proposal"]["path"]))
    declared = read(snapshot / A_PROTOCOL)
    require(all(digest(value) == digest(declared.get(key)) for key, value in proposal.items()), "full factory declaration differs")
    execution = dict(updates=1000, eval_samples=4096, protocol_seed=24002, heldout_seed=34002,
        metric_steps=STEPS, media_steps=MEDIA, terminal_observations=5)
    require(proposal.get("schema") == "pg_gaussian2d_additional_question_proposal_v1"
        and proposal.get("requested_recipe") == "atlas"
        and proposal.get("requested_recipe_overrides") == {"lr": .0053125, "prior_lr_mult": 1.5}
        and digest(proposal.get("execution")) == digest(execution), "question/seed/cadence/rates differ")
    case, recipe = proposal["case"], proposal["resolved_recipe"]
    require(case.get("id") == "api-gaussian2d" and case.get("legacy_ids") == ["source-family-16"]
        and case.get("default_steps") == 1000 and case.get("batch_size") == 2048
        and case.get("particles") == 20000 and case.get("z_dim") == 2
        and case.get("eval_samples") == 4096 and case.get("thresholds") == THRESHOLDS
        and case.get("law") == dict(kind="normal2d", mean=[1., 1.], covariance=[[.04, 0.], [0., .04]]),
        "original Gaussian target/resources/gates differ")
    require(recipe.get("lr") == .0053125 and recipe.get("prior_lr_mult") == 1.5
        and recipe.get("num_particles") == 20000 and recipe.get("z_dim") == 2
        and recipe.get("batch_size") == 2048 and recipe.get("total_steps", "missing") is None,
        "effective public Recipe differs")
    require(proposal.get("budget") == dict(physical_gpu=0, logical_device="cuda:0", cpu_threads=1,
        cuda_memory_fraction=.2, inclusive_paid_seconds=180, attempts=1, export_grace_seconds=0), "source-owned allowance differs")
    require(all(packet.get(key) is False for key in NO_CREDIT), "borrowed qualification scope")
    require(packet.get("family_paid_budget_seconds") == {FAMILY: 180}
        and packet.get("schema") == "pg_gaussian2d_supervised_packet_v1"
        and packet["spec"].get("id") == CASE
        and packet["spec"].get("paid_cap_seconds") == 180 and packet["spec"].get("export_grace_seconds") == 0
        and packet["spec"].get("frames") == 9 and packet["spec"].get("physical_attempt_limit") == 1
        and packet["spec"].get("retries") == 0 and digest(packet["spec"]) == packet["spec_sha256"], "supervision cap/spec differs")
    lane = packet.get("lane_runtime", {})
    require(lane.get("physical_gpu") == "0" and lane.get("device") == "cuda:0"
        and type(lane.get("torch_threads")) is int and lane["torch_threads"] == 1,
        "fixed physical/logical GPU runtime differs")
    return source, proposal


def imports(value, source):
    require(isinstance(value, dict) and bool(value), "missing actual imported-source evidence")
    for item in value.values():
        require(source["files"].get(item.get("path")) == item.get("sha256"), "foreign imported source")


def metadata(value, packet):
    source = packet["source"]
    require(value.get("schema") == "pg_gaussian2d_copied_source_metadata_v1" and value.get("status") == "PASS_METADATA_ONLY"
        and value.get("source_commit") == ORIGIN and value.get("source_digest") == SOURCE_DIGEST
        and value.get("wrapper_sha256") == WRAPPER_SHA and value.get("proposal_sha256") == packet["inputs"]["proposal"]["sha256"]
        and value.get("runtime") == packet["runtime_contract"], "copied-source metadata prerequisite differs")
    require(value.get("parent_source_files") == len(packet["parent_source"]["files"])
        and value.get("derived_source_files") == len(source["files"])
        and value.get("current_runtime_closure_subset") is True and value.get("copied_external_closure_exact") is True
        and value.get("cuda_initialized") is False and type(value.get("canonical_snapshot_entries")) is int
        and value["canonical_snapshot_entries"] == 1, "metadata source/namespace/CUDA scope differs")
    require(all(count(value.get(key)) == 0 for key in ("model_constructions", "training_updates", "evaluation_draws", "numerical_scorer_calls")), "metadata preflight acquired numerical credit")
    require(hex64(value.get("global_rng_before_sha256")) == value.get("global_rng_after_sha256"), "metadata RNG changed")
    imports(value.get("imported_sources_after"), source)


def purity(value, pairs, *, finalizer=False):
    require(value.get("pure") is True, "observer purity not established")
    for before, after in pairs:
        require(hex64(value.get(before)) == value.get(after), "state/selected/global RNG purity differs")
    if finalizer:
        require(type(value.get("extra_observations")) is int and value["extra_observations"] == 0, "extra finalizer reads")
    else:
        require(value.get("allowed_changes") == "none", "observer allowed an unknown state change")


def optimizer_updates(value, step):
    require(isinstance(value, dict) and set(value) == {"generator", "prior", "discriminator"}, "missing actual optimizer roles")
    for owner in value.values():
        require(count(owner.get("minimum")) == step and count(owner.get("maximum")) == step
            and count(owner.get("parameters")) > 0 and count(owner.get("uninitialized_parameters")) == (owner["parameters"] if step == 0 else 0), "optimizer role/cursor differs")


def execution(value):
    require(value.get("model_devices") == ["cuda:0"] and value.get("floating_dtypes") == ["torch.float32"]
        and value.get("autocast_enabled") is False, "actual device/dtype/autocast differs")


def observation(value, step):
    require(value.get("cohort") == COHORT and value.get("owner") == "particlegan.UpdatePolicy"
        and type(value.get("completed_steps")) is int and value["completed_steps"] == step
        and value.get("sampler") == "particlegan.GANTrainer.sample" and value.get("output_noise") is False
        and value.get("latent_policy") == "actual_selected_public_policy"
        and isinstance(value.get("selected_source"), str) and bool(value["selected_source"])
        and value.get("digest_kind") == "typed_policy_state_v1", "actual selected sampling law differs")
    hex64(value.get("snapshot_sha256"))
    require(value.get("requested_owners") == {key: True for key in OWNERS}
        and value.get("enabled_owners") == {key: True for key in OWNERS}, "owner disabled or unobserved")
    optimizer_updates(value.get("actual_optimizer_updates"), step)
    execution(value.get("execution", {}))
    purity(value.get("purity", {}), [("before_sha256", "after_sha256"),
        ("global_before_sha256", "global_after_sha256"), ("snapshot_before_sha256", "snapshot_after_sha256")])
    require(value["purity"]["snapshot_before_sha256"] == value["snapshot_sha256"], "selected snapshot binding differs")


def check_arrays(path):
    """Read NPY headers only: neither restore arrays nor recompute metrics."""
    expected = {f"step{step}_view0_{role}.npy" for step in STEPS for role in ("target", "samples")}
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == expected and len(archive.namelist()) == len(expected), "retained view array cadence differs")
        for name in archive.namelist():
            with archive.open(name) as stream:
                require(stream.read(6) == b"\x93NUMPY", "missing real NPY header")
                version = stream.read(2)
                require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "unsupported NPY metadata")
                width = 2 if version == b"\x01\x00" else 4
                length = int.from_bytes(stream.read(width), "little")
                require(0 < length <= 65536, "invalid NPY header size")
                header = ast.literal_eval(stream.read(length).decode("utf-8" if version == b"\x03\x00" else "latin1"))
                require(header.get("shape") == (4096, 2) and header.get("fortran_order") is False
                    and header.get("descr") in {"<f4", "<f8", ">f4", ">f8", "=f4", "=f8"}, "original target/sample count or numeric view differs")
                payload = 4096 * 2 * int(header["descr"][-1])
                require(archive.getinfo(name).file_size == 8 + width + length + payload, "truncated retained NPY artifact")


def check_complete(raw, attestation, packet, proposal, output, inputs, *, retained_only=False):
    source = packet["source"]
    require(raw.get("status") == "COMPLETE" and raw.get("verdict") in {"PASS", "FAIL"}
        and type(raw.get("passed")) is bool and raw["passed"] == (raw["verdict"] == "PASS")
        and type(raw.get("completed_updates")) is int and raw["completed_updates"] == 1000
        and raw.get("default_protocol_complete") is True and raw.get("source_unchanged") is True
        and raw.get("historical_results_changed") is False, "original run not fully complete")
    require(digest(raw.get("case")) == digest(proposal["case"]) and digest(raw.get("recipe")) == digest(proposal["resolved_recipe"])
        and raw.get("requested_recipe_overrides") == proposal["requested_recipe_overrides"]
        and type(raw.get("seed")) is int and raw["seed"] == 24002, "original case/Recipe/seed differs")
    expected_protocol = dict(updates=1000, default_updates=1000, evaluation_samples=4096, default_evaluation_samples=4096,
        evaluation_steps=STEPS, metric_evaluation_steps=STEPS, media_steps=MEDIA, metric_observations=24,
        media_frames=9, terminal_observations=5, wall_cap_seconds=180)
    require(digest(raw.get("protocol")) == digest(expected_protocol), "original protocol differs")
    runtime = raw.get("runtime", {})
    require(runtime.get("device") == "cuda:0" and type(runtime.get("torch_threads")) is int and runtime["torch_threads"] == 1
        and runtime.get("cuda_device_model") == "NVIDIA RTX A6000" and runtime.get("python") == packet["runtime_contract"]["python"]
        and runtime.get("torch") == packet["runtime_contract"]["packages"]["torch"], "original actual runtime differs")
    require(raw.get("source", {}).get("commit") == ORIGIN, "raw scientific source differs")
    imports({name: dict(path=name, sha256=value) for name, value in raw["source"].get("files_sha256", {}).items()}, source)
    require(OBSERVER in raw["source"]["files_sha256"] and WRAPPER in raw["source"]["files_sha256"], "new observer/driver missing from raw source")
    bound = raw.get("gaussian_bound_protocol", {})
    require(digest(bound.get("declaration")) == digest(proposal) and bound.get("source") == source
        and bound.get("scientific_status") == "COMPLETE" and bound.get("final_raw_receipt_is_verdict_authority") is True
        and digest(bound.get("admission")) == digest(dict(status="running", device="cuda:0", physical_gpu=0,
            threads=1, memory_fraction=.2, allowance_seconds=180, grace_seconds=0, lease_verified=True, single_attempt=True)), "bound actual admission/declaration differs")
    rows = raw.get("observations", [])
    require([row.get("step") for row in rows] == STEPS, "missing actual 25-read cadence")
    for row in rows:
        require(type(row.get("passed")) is bool and isinstance(row.get("failed_bounds"), list)
            and row["passed"] == (len(row["failed_bounds"]) == 0), "original observation pass/fail evidence differs")
        values = row.get("metrics", {})
        require(all(key in values and type(values[key]) in (int, float) and math.isfinite(values[key]) for key, _, _ in THRESHOLDS), "missing finite original metrics")
        require(values["sample_count"] == 4096 and len(row.get("views", [])) == 1
            and row["views"][0].get("kind") == "scatter", "original count/view metadata differs")
        observation(row.get("policy_observation", {}), row["step"])
    sustained = all(row["passed"] for row in rows[-5:])
    require(raw.get("metric_passed") is rows[-1]["passed"] and raw.get("sustained_metric_passed") is sustained
        and raw["passed"] is sustained, "original last-five grade consistency differs")
    observer = raw.get("policy_observer", {})
    require(observer.get("schema") == "pg_gaussian2d_policy_observer_sidecar_v1" and observer.get("cohort") == COHORT
        and observer.get("case_id") == "api-gaussian2d" and count(observer.get("completed_updates")) == 1000
        and observer.get("policy_protocol_complete") is True and observer.get("pre_export_numerical_status") == "COMPLETE"
        and observer.get("pre_export_numerical_verdict") == raw["verdict"]
        and observer.get("observer_source_sha256") == OBSERVER_SHA
        and observer.get("quality_from_owner_evidence") is False and observer.get("training_or_rescoring_added") is False
        and observer.get("observations") == [row["policy_observation"] for row in rows], "observer source/cadence/grade differs")
    require(observer.get("checkpoint_digest_kind") == "typed_policy_state_v1", "checkpoint state digest kind differs")
    hex64(observer.get("checkpoint_state_sha256"))
    optimizer_updates(observer.get("actual_optimizer_updates"), 1000)
    purity(observer.get("finalizer_purity", {}), [("state_before_sha256", "state_after_sha256"), ("global_before_sha256", "global_after_sha256")], finalizer=True)
    controls = observer.get("controls", {})
    require(controls.get("requested") == {key: True for key in OWNERS} and controls.get("enabled") == {key: True for key in OWNERS}
        and controls.get("requested_owners_bound") is True and controls.get("implementation_observed") is True
        and controls.get("quality_qualification") is False and controls.get("row_semantics") == "independent"
        and count(controls.get("completed_steps")) == 1000 and count(controls.get("row_evidence_observations")) == 1000
        and controls.get("served_source") == rows[-1]["policy_observation"]["selected_source"], "actual owner/control evidence differs")
    lifecycle = controls.get("lifecycle", {})
    require(lifecycle.get("owner") == "particlegan.UpdatePolicy" and lifecycle.get("complete") is True
        and lifecycle.get("start_completed_steps") == 0 and lifecycle.get("end_completed_steps") == 1000
        and lifecycle.get("observed_updates") == 1000 and lifecycle.get("calls") == {key: 1000 for key in HOOKS}
        and lifecycle.get("last_order") == HOOKS and lifecycle.get("pending") == [] and lifecycle.get("order_errors") == 0, "actual ordered public lifecycle differs")
    require(all(count(lifecycle.get(key)) == expected for key, expected in
        dict(start_completed_steps=0, end_completed_steps=1000, observed_updates=1000, order_errors=0).items())
        and all(count(value) == 1000 for value in lifecycle["calls"].values()), "typed lifecycle cursor differs")
    execution(controls.get("execution", {}))
    raw_path = output / "case/receipt.json"
    names = {"goal.gif", "observations.npz", "final-state.pt"}
    require(set(raw.get("artifacts", {})) == names, "incomplete original export")
    if retained_only:
        require(attestation is None, "retained illustration cannot fabricate final attestation")
        identities = {name: dict(path=str(output / "case" / name), **value) for name, value in raw["artifacts"].items()}
    else:
        require(isinstance(attestation, dict) and attestation.get("schema") == "pg_gaussian2d_supervised_observer_receipt_v1"
            and attestation.get("raw_receipt") == pin(raw_path) and attestation.get("original_verdict") == raw["verdict"]
            and attestation.get("source") == source and attestation.get("wrapper_sha256") == WRAPPER_SHA
            and all(attestation.get(key) is False for key in NO_CREDIT), "final supervised export attestation differs")
        imports(attestation.get("imported_sources_after"), source)
        require(set(attestation.get("artifacts", {})) == names, "incomplete final original attestation")
        identities = attestation["artifacts"]
    for name, value in identities.items():
        actual = inputs.check("raw:case/" + name, value, output / "case" / name)
        require(raw["artifacts"][name] == {key: value[key] for key in ("sha256", "bytes")}, "raw/attested artifact differs")
    check_arrays(output / "case/observations.npz")
    with Image.open(output / "case/goal.gif") as image:
        require(image.format == "GIF" and image.n_frames == 9 and type(raw.get("gif_frames")) is int
            and raw["gif_frames"] == 9, "original nine actual GIF frames missing")
        for frame in range(image.n_frames):
            image.seek(frame); image.load()
    return dict(original_status="COMPLETE", original_verdict=raw["verdict"], completed_updates=1000,
        final_metrics=rows[-1]["metrics"], final_failed_bounds=raw["failed_bounds"],
        terminal_five=[dict(step=row["step"], passed=row["passed"], metrics=row["metrics"], failed_bounds=row["failed_bounds"]) for row in rows[-5:]],
        read_steps=STEPS, media_steps=MEDIA, selected_source=controls["served_source"],
        selected_sampler="particlegan.GANTrainer.sample", output_noise=False,
        latent_policy="actual_selected_public_policy", actual_owner_count=8,
        checkpoint_state_sha256=observer["checkpoint_state_sha256"], runtime=runtime)


def verify(card_path, trusted_sha256):
    inputs = Inputs()
    card_path = Path(card_path).resolve()
    require(sha(card_path) == hex64(trusted_sha256), "root terminal card SHA required")
    inputs.file("root:terminal-card", card_path)
    card = read(card_path)
    require(card.get("schema") == "pg_gaussian2d_terminal_card_v1" and card.get("status") == "TERMINAL_IMMUTABLE", "immutable root terminal cut required")
    output = Path(card["output_root"]).resolve()
    expected = dict(packet=output.parent / ("." + output.name + ".gaussian-supervision.json"),
        study=output / "study.json", cost=output / "cost.json", metadata_preflight=output.parent / ("." + output.name + ".gaussian-metadata-check/receipt.json"))
    require(set(card.get("pins", {})) == {*expected, "terminal", "supervisor_request", "source_manifest"}, "exact immutable terminal inputs required")
    paths = {name: inputs.check("terminal-input:" + name, value, expected.get(name)) for name, value in card["pins"].items()}
    packet, study, cost = (read(paths[name]) for name in ("packet", "study", "cost"))
    source, proposal = check_source(packet, inputs, card["pins"]["source_manifest"])
    metadata(read(paths["metadata_preflight"]), packet)
    for key in ("schema", "spec", "spec_sha256", "source", "execution_source", "parent_source", "case_definitions", "runtime_contract", "family_paid_budget_seconds", "inputs"):
        require(study.get(key) == packet.get(key), "study/source/request identity differs: " + key)
    require(study.get("executed_family") == FAMILY and study.get("lane_runtime") == packet["lane_runtime"]
        and study.get("result") == cost and study.get("status") == cost.get("status"), "study terminal result/family differs")
    terminal, supervisor = read(paths["terminal"]), read(paths["supervisor_request"])
    require(cost.get("terminal") == card["pins"]["terminal"] and paths["supervisor_request"] == paths["terminal"].with_name("supervisor-request.json")
        and isinstance(terminal.get("token"), str) and re.fullmatch(r"[a-f0-9]{32}", terminal["token"])
        and terminal.get("token") == supervisor.get("token") and hashlib.sha256(str(terminal.get("token")).encode()).hexdigest() == cost.get("token_sha256"), "durable terminal fencing differs")
    require(supervisor.get("source") == source and number(supervisor.get("deadline_monotonic")) - number(supervisor.get("started_monotonic")) == 180,
        "durable source/inclusive allowance differs")
    fds = supervisor.get("lease_fds", [])
    require(len(fds) == 2 and len(set(fds)) == 2 and all(type(fd) is int and fd >= 0 for fd in fds), "missing inherited leases")
    command = supervisor.get("command", [])
    require(len(command) == 7 and isinstance(command[0], str) and command[1:] == ["-u", str(Path(source["snapshot_path"]) / WRAPPER),
        "--child", str(output / "resolved.json"), "--lease-fd", str(fds[-1])], "actual admitted child command differs")
    paid = number(terminal.get("paid_wall_seconds"))
    reserve = 0. if terminal.get("attempt_status") == "completed" else max(0., LIMIT - paid)
    require(all(number(cost.get(key)) == value for key, value in dict(paid_wall_seconds=paid, unmeasured_interrupt_reserved_seconds=reserve,
        charged_seconds=paid + reserve, overrun_seconds=max(0., paid + reserve - LIMIT)).items())
        and study.get("spent_seconds") == paid + reserve and all(cost.get(key) is False for key in NO_CREDIT), "inclusive measured/reserved accounting differs")
    require(cost.get("status") in {"COMPLETE", "INVALID", "INCOMPLETE", "BUDGET_EXCEEDED"}
        and (cost["status"] == "BUDGET_EXCEEDED") == (paid + reserve > 180), "terminal budget classification differs")
    available = card.get("available_raw", {})
    require(set(available).issubset({"receipt", "attestation"}), "foreign raw receipt inputs")
    raw_paths = dict(receipt=output / "case/receipt.json", attestation=output / "observer-control.json")
    for name, path in raw_paths.items():
        require(path.exists() == (name in available), "root cut omitted or fabricated existing raw receipt")
        if name in available:
            inputs.check("raw:" + name, available[name], path)
    raw = read(raw_paths["receipt"]) if "receipt" in available else None
    require((raw is None and "retained_original_receipt" not in cost)
        or raw is not None and cost.get("retained_original_receipt") == available["receipt"], "retained original receipt binding differs")
    result = dict(schema="pg_gaussian2d_portable_publication_v1", supervised_status=cost["status"],
        original_status=None if raw is None else raw.get("status"), original_verdict=None, accepted_numeric_status="UNAVAILABLE", required_questions=1,
        media=None, actual_full_original_exports=0, retained_illustrative_exports=0, question_id="api-gaussian2d", legacy_ids=["source-family-16"],
        cohort=COHORT, family=FAMILY, source=dict(origin_commit=ORIGIN, digest=SOURCE_DIGEST,
            parent_digest=packet["parent_source"]["digest"], files=len(source["files"]), source_manifest_sha256=card["pins"]["source_manifest"]["sha256"]),
        requested_recipe="atlas", requested_recipe_overrides=proposal["requested_recipe_overrides"], resolved_recipe=proposal["resolved_recipe"],
        question=proposal["case"]["goal"], law=proposal["case"]["law"], thresholds=THRESHOLDS,
        original_resources=dict(particles=20000, batch_size=2048, z_dim=2, evaluation_samples=4096),
        execution=proposal["execution"], costs=dict(paid_wall_seconds=paid, unmeasured_interrupt_reserved_seconds=reserve,
            charged_seconds=paid + reserve, inclusive_ceiling_seconds=180, export_grace_seconds=0,
            overrun_seconds=max(0., paid + reserve - LIMIT), scope="additional_gaussian_question_only_not_named_10500"),
        terminal_card_sha256=trusted_sha256, raw_availability="LOCAL_ONLY", quality_from_owner_evidence=False,
        publisher=dict(file="publish_results.py", sha256=sha(__file__), method="stdlib/Pillow hash/header/receipt projection; no array restoration or numerical rescoring"), **NO_CREDIT)
    if cost["status"] == "COMPLETE":
        require(terminal.get("attempt_status") == "completed" and set(available) == {"receipt", "attestation"}, "missing complete terminal/original attestation")
        verified = check_complete(raw, read(raw_paths["attestation"]), packet, proposal, output, inputs)
        require(terminal.get("child_returncode") == (0 if verified["original_verdict"] == "PASS" else 1)
            and cost.get("original_verdict") == verified["original_verdict"], "numeric verdict/child return differs")
        evidence = dict(raw_receipt=available["receipt"], attestation=available["attestation"], original_status="COMPLETE",
            original_verdict=verified["original_verdict"], media=pin(output / "case/goal.gif"))
        require(cost.get("evidence") == evidence, "driver-certified original export differs")
        result.update(verified, actual_full_original_exports=1, accepted_numeric_status="ACCEPTED_ORIGINAL")
    else:
        require(cost.get("original_verdict") is None or cost["status"] == "BUDGET_EXCEEDED", "incomplete/invalid numeric credit")
        # Do not publish raw exception command strings, private tokens or FDs.
        result["failure_reason"] = ("Inclusive 180-second allowance exceeded; final supervisor attestation is unavailable."
            if cost["status"] == "BUDGET_EXCEEDED" else
            "Incomplete or invalid supervised evidence; accepted numerical verdict is unavailable. See the pinned local original receipt.")
        if raw is not None:
            result["completed_updates"] = raw.get("completed_updates")
            result["retained_raw_failed_bound_count"] = len(raw.get("failed_bounds", []))
            # Hash only. A partial/error artifact is never promoted or copied as
            # full qualifying media, and checkpoint tensors are never restored.
            for name, value in raw.get("artifacts", {}).items():
                require(name in {"goal.gif", "observations.npz", "final-state.pt"}, "foreign raw artifact")
                inputs.check("raw:case/" + name, dict(path=str(raw_paths["receipt"].parent / name), **value))
        review = card.get("root_retained_review")
        if cost["status"] == "BUDGET_EXCEEDED" and review is not None:
            require(set(available) == {"receipt"} and terminal.get("attempt_status") == "timeout"
                and cost.get("original_verdict") is None
                and review == dict(accepted_numerical_verdict=None, decoded_original_goal_frames=9,
                    missing_final_supervisor_attestation=True, new_models_draws_scoring=0,
                    original_full_export_byte_hashes_verified=True, original_raw_verdict="FAIL",
                    qualification_input=False, scope_status="BUDGET_EXCEEDED"), "root retained-illustration scope differs")
            artifacts = card.get("retained_original_artifacts", {})
            require(set(artifacts) == {"goal.gif", "final-state.pt", "observations.npz"}, "root retained original artifact pins missing")
            for name, identity in artifacts.items():
                inputs.check("raw:case/" + name, identity, output / "case" / name)
                require(raw["artifacts"].get(name) == {key: identity[key] for key in ("sha256", "bytes")}, "root/raw retained artifacts differ")
            retained = check_complete(raw, None, packet, proposal, output, inputs, retained_only=True)
            require(retained["original_verdict"] == "FAIL", "root/raw unaccepted original verdict differs")
            result.update(retained_illustrative_exports=1,
                retained_illustration={**retained, "accepted_numeric_status": "UNAVAILABLE", "qualification_input": False,
                    "missing_final_supervisor_attestation": True,
                    "caption": "BUDGET_EXCEEDED / accepted numeric UNAVAILABLE. Byte-original raw FAIL badge is unaccepted; retained illustration only."})
    return result, inputs, output


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def publish(card_path, trusted_sha256, destination):
    result, inputs, output = verify(card_path, trusted_sha256)
    destination = Path(destination)
    require(not destination.exists(), "fresh publication directory required")
    destination.mkdir(parents=True)
    if result["actual_full_original_exports"] or result["retained_illustrative_exports"]:
        (destination / "gifs").mkdir()
        target = destination / "gifs/goal.gif"
        shutil.copyfile(output / "case/goal.gif", target)
        require(sha(target) == sha(output / "case/goal.gif"), "original media copy differs")
        caption = (result["retained_illustration"]["caption"] if result["retained_illustrative_exports"] else
            "Complete joined original training GIF; original numerical " + result["original_verdict"])
        result["media"] = dict(path="gifs/goal.gif", sha256=sha(target), bytes=target.stat().st_size, frames=9, steps=MEDIA,
            byte_original=True, retained_illustration_only=bool(result["retained_illustrative_exports"]), caption=caption)
        write(destination / "gifs/goal-context.json", dict(gif_sha256=sha(target), caption=caption,
            supervised_status=result["supervised_status"], accepted_numeric_status=result["accepted_numeric_status"],
            accepted_numeric_verdict=result["original_verdict"], raw_original_verdict=(result["retained_illustration"]["original_verdict"]
                if result["retained_illustrative_exports"] else result["original_verdict"])))
        context = destination / "gifs/goal-context.json"
        result["media"]["context"] = dict(path="gifs/goal-context.json", sha256=sha(context), bytes=context.stat().st_size)
    index = dict(schema="pg_gaussian2d_publication_input_index_v1", inputs=sorted(inputs.items.values(), key=lambda row: row["label"]),
        source_checkpoint_array_log_files_copied=False, original_goal_gif_copies=result["actual_full_original_exports"] + result["retained_illustrative_exports"],
        raw_availability="LOCAL_ONLY", private_tokens_or_lease_fds_published=False)
    write(destination / "input-index.json", index)
    result["input_index"] = dict(path="input-index.json", sha256=sha(destination / "input-index.json"), inputs=len(index["inputs"]))
    write(destination / "results.json", result)
    proof = dict(schema="pg_gaussian2d_portable_publication_proof_v1", status="VERIFIED_DRAW_FREE",
        trusted_terminal_card_sha256=trusted_sha256, publisher_sha256=sha(__file__), source_digest=SOURCE_DIGEST,
        results_sha256=sha(destination / "results.json"), input_index_sha256=sha(destination / "input-index.json"),
        consumed_inputs=len(index["inputs"]), original_gif_copies=result["actual_full_original_exports"],
        retained_illustration_gif_copies=result["retained_illustrative_exports"],
        models_restored=0, sample_draws=0, updates=0, scorer_calls=0,
        arrays_read="NPY headers only for complete artifacts; no tensor/array restoration or scoring",
        private_tokens_or_lease_fds_published=False)
    write(destination / "verification.json", proof)
    numerical = result["original_verdict"] or "UNAVAILABLE"
    media = (f"**{result['media']['caption']}**\n\n![{result['media']['caption']}](gifs/goal.gif)\n\n"
        "[Bound media context](gifs/goal-context.json)." if result["media"] else "No full joined original goal GIF is published.")
    text = f"""# Gaussian mean, covariance and distribution recovery

Supervised status: **{result['supervised_status']}**. Accepted numerical verdict: **{numerical}**.
This is one existing additional question (`api-gaussian2d`, source-family-16),
separate from the original 26-slot suite and the named 10,500-second diagnostics.
It gives no historical, default or fair-speed credit. Owner evidence is separate
from density recovery. Checkpoint persistence is retained; no new continuation
test or checkpoint-resume claim is made.

The target is `N((1, 1), .04 I)`. The unchanged full protocol requires 1,000
updates, 20,000 prior rows, batch 2,048 and 25 actual reads of 4,096 samples.
The original last five post-update checks are updates 834, 875, 917, 959 and
1,000. Gates are sample count >= 1,024; mean error <= .10 sigma; covariance
eigenvalues in [.85, 1.15]; radial KS <= .075; maximum projected KS <= .06.
The GIF's nine actual states show the target and selected generated distribution
at updates 0, 125, 250, 375, 500, 625, 750, 875 and 1,000. {media}

C6 requests LR .0053125 / prior multiplier 1.5. The complete original Recipe is
in [results.json](results.json); nominal multipliers do not imply measured
displacements. The primary sampler is the public selected policy with output
noise off and actual DV12 latent perturbation retained. Averaged parameters may
be selected; this is neither forced EMA nor unperturbed-atom sampling.

One inclusive 180-second allowance covers startup, construction, training,
observations, original export and attestation, with zero grace and no retry.
Paid: {paid_format(result['costs']['paid_wall_seconds'])} s; unmeasured interruption
reserve: {paid_format(result['costs']['unmeasured_interrupt_reserved_seconds'])} s;
charged: {paid_format(result['costs']['charged_seconds'])} s; overrun:
{paid_format(result['costs']['overrun_seconds'])} s. These costs belong only to
this additional Gaussian attempt. Full numeric FAIL remains FAIL; software or
source INVALID and partial evidence receive no numerical or fabricated-media credit.
If the final attestation is absent after the deadline, a separately bound full
raw illustration can be shown with its unaccepted raw FAIL badge; it does not
substitute for the missing attestation or an accepted numerical result.

Frozen source `{ORIGIN}`, digest `{SOURCE_DIGEST}`. The root's explicit terminal
card, all current source bytes, imported-source/observer/lease/cost joins and
original artifact hashes are verified. This publisher uses only standard library
and Pillow; it draws no samples, restores no model or tensor and calls no scorer.
Raw source/checkpoint/arrays/logs remain **LOCAL_ONLY**; hashes in
[input-index.json](input-index.json) do not claim an uploaded archive.
[Verification](verification.json) binds the copied media and compact report.
"""
    (destination / "README.md").write_text(text)
    return proof


def paid_format(value):
    return format(value, ".12g")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-card", required=True, type=Path)
    parser.add_argument("--trusted-card-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.terminal_card, args.trusted_card_sha256, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
