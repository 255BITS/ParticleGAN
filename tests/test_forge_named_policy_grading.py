"""Synthetic receipt/byte guards, with no model, sampling or convergence credit.

The metrics and checkpoint payloads are deliberately invented software inputs.
The tests exercise contract rejection and retained-byte identity only; a grader
accepting a synthetic curve is never a trained gate or capacity certificate.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.forge.ae_routed_policy_contracts import make_ae_variant
from experiments.forge.conditional_policy_contracts import make_conditional_variant
from experiments.forge.multibank_policy_contracts import make_variant as make_cover_variant
from experiments.forge.routed_policy_contracts import make_unused_variant
from experiments.forge.word_joint_policy_contracts import make_variant as make_word_variant
from experiments.forge.artifacts import manifest_artifacts
from experiments.forge.sampling import (
    AE_ROUTED_RECONSTRUCTION, FIELDS, GENERATED_AND_RECONSTRUCTED, PARAMETER_MEASUREMENT,
    executed_receipt, expected_policy, grade_sampling, task_blockers,
)
from experiments.forge.views import grade_result


ROOT = Path(__file__).resolve().parents[1]
HOSTS = ("trajectory", "residual_student", "unipolar", "mid_scale_identity",
         "unused_token_hold", "cover_leftover", "ae_gan_hold")
HOOKS = ("begin_step", "after_critic_step", "after_generator_backward",
         "after_generator_step", "finish_step")
OWNERS = ("continuous_controller", "stationarity_lr", "row_evidence", "birth_death",
          "learned_output_noise", "selected_averaging", "optimizer_surprise", "reopen_guard")


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="module")
def declarations():
    result = {name: make_conditional_variant(ROOT, name) for name in HOSTS[:4]}
    result.update(unused_token_hold=make_unused_variant(ROOT), cover_leftover=make_cover_variant(ROOT),
                  ae_gan_hold=make_ae_variant(ROOT), five_word_joint=make_word_variant(ROOT))
    return result


def synthetic_receipt(task, root):
    """Portable invented fields matching the *documented* producer interfaces."""
    contract, host = task["execution"]["policy_contract"], task["execution"]["host"]
    family, horizon = task["policy_family"], task["execution"]["steps"]
    metrics = {name: float(bound) for name, _, bound in task["evaluation"]["thresholds"]}
    clocks = [math.ceil(i * horizon / 24) for i in range(1, 25)]
    groups, count_names = ["generator", "table", "noise"], ["generator", "table", "discriminator"]
    models = ["critic", "generator", "router"]
    if host in {"trajectory", "residual_student"}:
        callback, module = "complete_arc_forward", "conditional_policy_adapters"
        models.append("prior")
    elif host in {"unipolar", "mid_scale_identity"}:
        callback, module = "complete_basis_forward", "conditional_policy_adapters"
        groups = ["table", "noise"]
    elif host == "unused_token_hold":
        callback, module = "complete_slot_forward", "routed_policy_adapters"
    elif host == "cover_leftover":
        callback, module = "complete_polar_forward", "multibank_policy_adapters"
        count_names = ["generator", "prior", "discriminator"]
        models.append("prior")
    elif host == "ae_gan_hold":
        callback, module = "complete_ae_forward", "ae_routed_policy_adapters"
        groups = ["generator", "encoder", "table", "noise"]
        count_names = ["generator", "encoder", "prior", "discriminator"]
        models.extend(("encoder", "prior"))
    else:
        assert host == "five_word_joint"
        callback, module = "joint_generation", "word_joint_policy_adapters"
        groups = ["generator", "encoder", "table", "noise"]
        count_names = ["generator", "encoder", "prior", "discriminator"]
        models = ["critic", "generator", "encoder", "prior"]
    resources = task["execution"]["resources"]
    shape = [resources["num_particles"], resources["z_dim"]]
    routing = contract.get("routing", {"row_buffers": [], "sites": []})
    ownership = {"table": {"shape": shape, "dtype": "torch.float32", "parameter": True, "optimizer": 0},
                 "router.log_mass": {"shape": [shape[0]], "dtype": "torch.float32", "parameter": False, "optimizer": None}}
    for name in routing["row_buffers"]:
        ownership["router." + name] = {"shape": [shape[0], 1], "dtype": "torch.int64", "parameter": False, "optimizer": None}
    counter = {"evals": 0, "moves": 0, "splits": 0}  # Owner/clock proof does not promise a structural move.
    config = {"model_forward": True, "sites": deepcopy(routing["sites"]),
        "log_mass_key": "log_mass", "row_buffers": deepcopy(routing["row_buffers"]),
        "row_parameters": [], "routed_geometry": "mass_atoms_v1", "output_error_guard": True,
        "max_context_harm": 0., "max_output_error_increase": 0., "max_output_context_harm": 0.,
        "reservoir_size": 64, "min_observations": 8, "probe_budget": 8, "candidate_budget": 4,
        "min_effect": 1e-6, "improvement_margin": 1e-8, "persistence_threshold": .75, "split_scale": .1}
    policies, purities, named_audits = [], [], []
    for step in clocks:
        observed = {**deepcopy(task["evaluation"]["policy_observation"]), "observed": True,
            "policy_owner": "particlegan.UpdatePolicy", "selected_source": "fast", "controller": "dv12",
            "completed_steps": step, "snapshot_sha256": sha(f"synthetic-selected-{step}"),
            "family": family, "row_policy": "routed_paired", "backend_selection": {"actual_backend": "routed"}}
        if host in HOSTS[:4]:
            observed.update(output_sigma_used=.029 if host in HOSTS[:2] else 0., latent_perturbation_applied=False)
        policies.append(observed)
        prefix = "global" if host == "ae_gan_hold" else "global_rng"
        purities.append({"completed_steps": step, "digest_kind": "typed_policy_state_v1",
            "before_sha256": sha(f"synthetic-training-state-{step}"), "after_sha256": sha(f"synthetic-training-state-{step}"),
            prefix + "_before_sha256": sha(f"synthetic-global-{step}"),
            prefix + "_after_sha256": sha(f"synthetic-global-{step}"), "pure": True})
        named_audits.append({"changed_streams": [], "unintended_streams": [], "unintended_rng_deviations": 0})
    mechanisms = {name: {"requested": False, "enabled": False, "calls": 0, "eligible": 0, "applied": 0}
                  for name in ("critic_anchor", "critic_guard")}
    mechanisms["critic_penalty"] = {"requested": True, "enabled": True, "calls": horizon, "eligible": horizon, "applied": horizon}
    mechanisms["a2"] = {"requested": True, "enabled": True, "calls": horizon, "eligible": 0, "applied": 0,
        "probe": {"status": "PASS", "kind": "synthetic_public_component", "synthetic_state": True,
                  "training_evidence": False, "measurements": {"damping_branch_applied": True}}}
    mechanisms["direct_particle_gain"] = {"requested": True, "enabled": False, "calls": 0, "eligible": 0, "applied": 0,
        "applicable_to_routed_table": False, "host_activation_credit": False,
        "probe": {"status": "PASS", "kind": "synthetic_public_component", "synthetic_state": True,
                  "training_evidence": False, "measurements": {"gain": 2.}}}
    controls = {"schema_version": 1, "cohort": task["task_cohort"], "family": family,
        "row_semantics": "conditional", "row_policy": "routed_paired", "independent_atlas_qualification": False,
        "completed_steps": horizon, "requested": dict.fromkeys(OWNERS, True), "enabled": dict.fromkeys(OWNERS, True),
        "requested_owners_bound": True, "implementation_observed": True, "served_source": "fast",
        "roles": [groups, ["critic"]], "effective_group_lrs": [[.0053125] * len(groups), [.0053125]],
        "row_evidence_observations": horizon,
        "execution": {"model_devices": ["cuda:0"], "floating_dtypes": ["torch.float32"], "autocast_enabled": False},
        "lifecycle": {"owner": "particlegan.UpdatePolicy", "start_completed_steps": 0,
            "end_completed_steps": horizon, "observed_updates": horizon, "calls": dict.fromkeys(HOOKS, horizon),
            "last_order": list(HOOKS), "pending": [], "order_errors": 0, "complete": True},
        "diagnostics": {"controller": {"variant": "dv12", "updates": horizon, "latent_applications": [
            {"radius_min": 0., "radius_mean": .01, "radius_max": .02, "perturbation_rms": .001, "clipped_fraction": .5}]},
            "stationarity_lr": {**{f"g{i}": {"s": 1.} for i in range(len(groups))}, "d0": {"s": 1.}},
            "row_evidence": {"updates": horizon}, "birth_death": {"counters": deepcopy(counter),
                "rows": {"law": "conditional_paired_diagnostic", "counters": {"updates": horizon}}},
            "surprise": {"fires": 0, "armed": False}, "reopen_guard": {"epoch_rebases": 0}},
        "routed_owner": {"schema_version": 1, "owner": "particlegan.routing.RoutedRowControl",
            "config": config, "model_forward": {"callable": True, "module": "experiments.forge." + module, "qualname": callback},
            "model_roles": sorted(models), "table_matches_policy": True, "averaged_table_matches_policy": True,
            "table_shape": shape, "row_ownership": ownership, "fit_fill": 64, "guard_fill": 64,
            "probe_clock": {"observed_updates": horizon, "last_probe_update": 0}, "counters": deepcopy(counter),
            "state_sha256": sha("synthetic-actual-routed-owner"), "state_digest_kind": "typed_policy_state_v1"}}
    if "table_optimizer_semantics" in contract:
        controls["table_optimizer_semantics"] = deepcopy(contract["table_optimizer_semantics"])
    if host == "five_word_joint":
        controls.pop("routed_owner")
        controls.pop("row_policy")
        controls.update(row_semantics="independent", actual_prior_rows=11, canonical_target_words=5,
            joint_atom_code="same_effective_code", output_noise_coordinates="words168_only",
            resource_adaptation=deepcopy(contract["resource_adaptation"]),
            actual_birth_death={"rows": 11, "neighbours": 5, "isolation": True, "reference_half": 6})
        controls["diagnostics"]["birth_death"] = {"k": 5, "counters": deepcopy(counter)}
        direct = mechanisms["direct_particle_gain"]
        direct.pop("applicable_to_routed_table")
        direct["applicable_to_direct_output"] = False
        for observed in policies:
            observed.pop("row_policy")
    directory = root / host
    (directory / "observations").mkdir(parents=True)
    (directory / "state.pt").write_bytes(b"synthetic software checkpoint bytes; never deserialize")
    for step in clocks:
        (directory / "observations" / f"step_{step:06d}.npz").write_bytes(f"synthetic observed goal {step}".encode())
    manifest = manifest_artifacts(directory)
    evidence = {**{key: task["evaluation"][key] for key in FIELDS}, "scoring_weights": "state_selected",
        "observations": [{"step": step, **deepcopy(metrics)} for step in clocks], "live": deepcopy(metrics),
        "policy_observation": deepcopy(policies[-1]), "policy_observations": policies, "policy_purity": purities,
        "rng_audits": named_audits, "policy_controls": controls,
        "guards": {"all_finite": True, "optimizer_updates": dict.fromkeys(count_names, horizon),
            "unintended_rng_deviations": 0, "hooks_exercised": True,
            "mechanism_audit": {"schema_version": 1, "mechanisms": mechanisms}},
        "checkpoint": {"path": "state.pt", "sha256": manifest["files"]["state.pt"]["sha256"],
            "state_sha256": sha("synthetic-complete-state"), "digest_kind": "typed_policy_state_v1"},
        "artifact_root": str(directory), "artifact_manifest": manifest}
    if host == "five_word_joint":
        evidence["host"] = {"family": family, "task_cohort": task["task_cohort"],
            "actual_resources": deepcopy(resources), "canonical_words": deepcopy(task["execution"]["host_definition"]["words"]),
            "resource_adaptation": deepcopy(contract["resource_adaptation"]), "objective": contract["objective"],
            "reconstruction_training_loss": False, "original_capacity_or_qualification_credit": False}
    return {"task_id": task["id"], "gate_status": "PASS", "evidence": evidence, "software_fixture_only": True}


def grade(task, raw):
    before = deepcopy((task, raw))
    result = grade_result(task, raw)
    assert (task, raw) == before
    return result["status"]


@pytest.mark.parametrize("host", HOSTS + ("five_word_joint",))
def test_named_original_bounds_and_five_terminal_checks_survive(host, declarations, tmp_path):
    task = declarations[host]
    raw = synthetic_receipt(task, tmp_path)
    assert not task_blockers(task)
    assert grade(task, raw) == "PASS"  # Invented receipt only.
    name, comparison, bound = task["evaluation"]["thresholds"][0]
    failed = math.nextafter(float(bound), math.inf if comparison == "<=" else -math.inf)
    raw["evidence"]["observations"][-2][name] = failed
    assert grade(task, raw) == "FAIL"
    raw["evidence"]["observations"][-2][name] = bound
    raw["evidence"]["live"][name] = failed
    assert grade(task, raw) == "FAIL"


def test_ae_clean_complete_law_cannot_alias_scheduled_parent(declarations, tmp_path):
    task = declarations["ae_gan_hold"]
    parent = json.loads((ROOT / "configs/forge/tasks/ae_gan_hold.json").read_text())
    assert expected_policy(parent) == executed_receipt(GENERATED_AND_RECONSTRUCTED, eval_output_noise="public_recipe_schedule")
    assert expected_policy(task) == executed_receipt(AE_ROUTED_RECONSTRUCTION, eval_output_noise="clean_complete_routed_function")
    raw = synthetic_receipt(task, tmp_path)
    raw["evidence"].update(expected_policy(parent))
    assert grade(task, raw) == "INVALID"
    parent["evaluation"].update(expected_policy(task))
    assert task_blockers(parent)


@pytest.mark.parametrize("host", HOSTS + ("five_word_joint",))
@pytest.mark.parametrize("field", ["policy_controls", "checkpoint", "artifact_manifest", "rng_audits"])
def test_no_compact_stamp_can_replace_actual_ownership_or_checkpoint(host, field, declarations, tmp_path):
    task = declarations[host]; raw = synthetic_receipt(task, tmp_path)
    del raw["evidence"][field]
    assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("host", HOSTS + ("five_word_joint",))
def test_every_real_named_optimizer_role_is_required(host, declarations, tmp_path):
    task = declarations[host]; raw = synthetic_receipt(task, tmp_path)
    for role in list(raw["evidence"]["guards"]["optimizer_updates"]):
        mutated = deepcopy(raw); del mutated["evidence"]["guards"]["optimizer_updates"][role]
        assert grade(task, mutated) == "INCOMPLETE"
        mutated["evidence"]["guards"]["optimizer_updates"][role] = 0
        assert grade(task, mutated) == "FAIL"
        mutated["evidence"]["guards"]["optimizer_updates"][role] = task["execution"]["steps"] - 1
        assert grade(task, mutated) == "INVALID"


@pytest.mark.parametrize("host", ["unipolar", "ae_gan_hold"])
def test_packed_table_and_encoder_groups_cannot_be_replaced_by_independent_defaults(host, declarations, tmp_path):
    task = declarations[host]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["roles"] = [["generator", "table", "noise"], ["critic"]]
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("row_semantics", "independent"), ("family", "atlas"),
    ("row_policy", "independent"), ("independent_atlas_qualification", True)])
def test_named_receipt_cannot_borrow_independent_family_credit(field, value, declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("model_forward", {"callable": False}), ("model_roles", ["generator", "critic"]),
    ("table_matches_policy", False), ("averaged_table_matches_policy", False), ("table_shape", [12, 4])])
def test_no_missing_complete_function_or_table_can_claim_routed_owner(field, value, declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["routed_owner"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("sites", ["fake_site"]), ("row_buffers", []),
    ("model_forward", False), ("routed_geometry", "independent_atoms"), ("output_error_guard", False),
    ("max_context_harm", 1e-8), ("max_output_error_increase", 1e-8), ("max_output_context_harm", 1e-8),
    ("log_mass_key", "ignored_mass"), ("min_observations", 1), ("candidate_budget", 1),
    ("persistence_threshold", 0.)])
def test_actual_mass_and_protected_context_contracts_cannot_be_weakened(field, value, declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["routed_owner"]["config"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("pool", ["fit_fill", "guard_fill"])
def test_zero_context_owner_is_incomplete(pool, declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["routed_owner"][pool] = 0
    assert grade(task, raw) == "INCOMPLETE"


def test_a_declared_route_without_observed_owner_or_completed_context_clock_is_rejected(declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    owner = raw["evidence"]["policy_controls"].pop("routed_owner")
    assert grade(task, raw) == "INCOMPLETE"
    raw["evidence"]["policy_controls"]["routed_owner"] = owner
    owner["probe_clock"]["observed_updates"] -= 1
    assert grade(task, raw) == "INVALID"


def test_live_routed_owner_requires_its_typed_state_identity(declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["routed_owner"]["state_sha256"] = "a label"
    assert grade(task, raw) == "INCOMPLETE"


def test_full_hook_stamp_cannot_hide_missing_paired_gradient_or_settling_owner(declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["row_evidence_observations"] = 0
    assert grade(task, raw) == "INVALID"
    raw["evidence"]["policy_controls"]["row_evidence_observations"] = task["execution"]["steps"]
    raw["evidence"]["policy_controls"]["diagnostics"]["stationarity_lr"].pop("g1")
    assert grade(task, raw) == "INCOMPLETE"


def test_actual_row_decision_law_is_conditional_not_borrowed_independent_BH(declarations, tmp_path):
    task = declarations["cover_leftover"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["diagnostics"]["birth_death"]["rows"]["law"] = "independent_atom_bh"
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("enabled", True), ("host_activation_credit", True),
    ("applicable_to_routed_table", True), ("requested", False), ("applied", 1)])
def test_scratch_direct_probe_cannot_be_promoted_to_routed_transport_credit(field, value, declarations, tmp_path):
    task = declarations["cover_leftover"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["guards"]["mechanism_audit"]["mechanisms"]["direct_particle_gain"][field] = value
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("owner", OWNERS)
def test_all_requested_public_policy_owners_must_be_enabled(owner, declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["enabled"][owner] = False
    assert grade(task, raw) == "BLOCKED"


def test_missing_or_substituted_latent_damping_owner_cannot_pass(declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["guards"]["mechanism_audit"]["mechanisms"]["a2"]["calls"] = 0
    assert grade(task, raw) == "BLOCKED"


@pytest.mark.parametrize("applications", [0, 600, [], [{"only_a_label": "dv12"}], [
    {"radius_min": 0., "radius_mean": .01, "radius_max": .02, "perturbation_rms": math.nan, "clipped_fraction": .5}]])
def test_actual_dv12_application_is_a_bounded_read_not_an_invented_counter(applications, declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_controls"]["diagnostics"]["controller"]["latent_applications"] = applications
    # NaN fixtures are compared with a structural copy outside the pure grader.
    assert grade_result(task, raw)["status"] == "BLOCKED"


@pytest.mark.parametrize("host,prefix", [("ae_gan_hold", "global"), ("trajectory", "global_rng")])
def test_separate_global_rng_proof_is_required_and_cannot_borrow_state_purity(host, prefix, declarations, tmp_path):
    task = declarations[host]; raw = synthetic_receipt(task, tmp_path)
    del raw["evidence"]["policy_purity"][0][prefix + "_before_sha256"]
    assert grade(task, raw) == "INCOMPLETE"
    raw["evidence"]["policy_purity"][0][prefix + "_before_sha256"] = sha("different-global-state")
    assert grade(task, raw) == "INVALID"


def test_named_rng_changed_stream_is_not_exempted_by_zero_aggregate(declarations, tmp_path):
    task = declarations["cover_leftover"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["rng_audits"][0]["unintended_streams"] = ["train/prior"]
    assert grade(task, raw) == "INVALID"


def test_changed_training_stream_cannot_be_labeled_an_allowed_evaluation_draw(declarations, tmp_path):
    task = declarations["trajectory"]; raw = synthetic_receipt(task, tmp_path)
    changed = raw["evidence"]["rng_audits"][0]["changed_streams"]
    changed.append(json.dumps(["prior", "latent", "indices", "cuda:0"], separators=(",", ":")))
    assert grade(task, raw) == "INVALID"
    changed[0] = json.dumps(["eval", "sampler", "samples", "cuda:0"], separators=(",", ":"))
    assert grade(task, raw) == "PASS"  # Synthetic isolated evaluation stream only.


@pytest.mark.parametrize("host,field,value", [("trajectory", "latent_perturbation_applied", True),
    ("unipolar", "output_sigma_used", .029), ("residual_student", "output_sigma_used", math.inf)])
def test_actual_conditional_read_cannot_contradict_its_noise_or_enumeration_law(host, field, value, declarations, tmp_path):
    task = declarations[host]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_observations"][-1][field] = value
    raw["evidence"]["policy_observation"] = deepcopy(raw["evidence"]["policy_observations"][-1])
    assert grade_result(task, raw)["status"] == ("INCOMPLETE" if math.isinf(value) else "INVALID")


@pytest.mark.parametrize("field", ["policy_purity", "policy_observations", "observations"])
def test_original_checkpoint_cadence_cannot_drop_early_frames_or_terminal_state(field, declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"][field].pop(0)
    assert grade(task, raw) in {"INCOMPLETE", "INVALID"}


def test_short_complete_lifecycle_is_not_a_full_original_budget(declarations, tmp_path):
    task = declarations["unused_token_hold"]; raw = synthetic_receipt(task, tmp_path)
    controls = raw["evidence"]["policy_controls"]
    controls["completed_steps"] -= 1
    controls["lifecycle"].update(end_completed_steps=199, observed_updates=199, calls=dict.fromkeys(HOOKS, 199))
    assert grade(task, raw) == "INCOMPLETE"


@pytest.mark.parametrize("mutation", ["checkpoint_hash", "path_escape", "changed_checkpoint", "changed_npz", "missing_view"])
def test_full_checkpoint_and_each_retained_goal_view_are_hash_bound(mutation, declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    evidence = raw["evidence"]; directory = Path(evidence["artifact_root"])
    if mutation == "checkpoint_hash":
        evidence["checkpoint"]["sha256"] = sha("unbound-checkpoint")
    elif mutation == "path_escape":
        evidence["checkpoint"]["path"] = "../state.pt"
    elif mutation == "changed_checkpoint":
        (directory / "state.pt").write_bytes(b"modified synthetic checkpoint")
    elif mutation == "changed_npz":
        next((directory / "observations").glob("*.npz")).write_bytes(b"modified synthetic goal view")
    else:
        evidence["artifact_manifest"]["files"].pop(next(name for name in evidence["artifact_manifest"]["files"] if name.endswith(".npz")))
    assert grade(task, raw) == ("INCOMPLETE" if mutation == "missing_view" else "INVALID")


@pytest.mark.parametrize("name", ["state.pt", "observations/step_000011.npz"])
def test_zero_byte_artifacts_cannot_claim_complete_checkpoint_or_goal_view(name, declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    evidence = raw["evidence"]; directory = Path(evidence["artifact_root"])
    (directory / name).write_bytes(b"")
    evidence["artifact_manifest"] = manifest_artifacts(directory)
    evidence["checkpoint"]["sha256"] = evidence["artifact_manifest"]["files"]["state.pt"]["sha256"]
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("field,value", [("family", "atlas_routed"), ("row_policy", "independent"),
    ("controller", "dv11"), ("output_noise", True), ("diagnostic_credit", True), ("selected_source", "forced_ema")])
def test_measurement_cannot_borrow_other_family_noise_or_diagnostic(field, value, declarations, tmp_path):
    task = declarations["ae_gan_hold"]; raw = synthetic_receipt(task, tmp_path)
    raw["evidence"]["policy_observations"][-1][field] = value
    raw["evidence"]["policy_observation"] = deepcopy(raw["evidence"]["policy_observations"][-1])
    assert grade(task, raw) == "INVALID"


@pytest.mark.parametrize("mutation", ["unknown", "word", "missing_contract", "missing_cohort", "missing_sampling_version"])
def test_unknown_parked_and_partial_policy_claims_fail_closed(mutation, declarations, tmp_path):
    task = deepcopy(declarations["unused_token_hold"]); raw = synthetic_receipt(task, tmp_path)
    if mutation in {"unknown", "word"}:
        task["task_cohort"] = "unknown_cohort" if mutation == "unknown" else "word_joint_policy_v1"
        task["execution"]["policy_contract"]["cohort"] = task["task_cohort"]
    elif mutation == "missing_contract":
        del task["execution"]["policy_contract"]
    elif mutation == "missing_cohort":
        del task["task_cohort"]
    else:
        del task["evaluation"]["sampling_contract_version"]
    assert grade(task, raw) == "INVALID"


def test_original_parameter_task_does_not_require_new_owners_or_checkpoint():
    parent = json.loads((ROOT / "configs/forge/tasks/unused_token_hold.json").read_text())
    assert expected_policy(parent) == executed_receipt(PARAMETER_MEASUREMENT, eval_output_noise="not_applied_to_measurement")
    observed = expected_policy(parent)
    assert grade_sampling(parent, observed) is None


@pytest.mark.parametrize('field,value', [
    ('actual_prior_rows', 5), ('actual_prior_rows', 6), ('actual_prior_rows', 10),
    ('canonical_target_words', 11), ('joint_atom_code', 'old_split_raw_code'),
    ('output_noise_coordinates', 'whole170Djoint'), ('independent_atlas_qualification', True),
    ('family', 'atlas'), ('row_semantics', 'conditional')])
def test_word_min11_cannot_borrow_old_resources_or_split_code_and_noise_laws(field, value, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    raw['evidence']['policy_controls'][field] = value
    assert grade(task, raw) == 'INVALID'


@pytest.mark.parametrize('field,value', [('rows', 5), ('neighbours', 4), ('reference_half', 5), ('isolation', False)])
def test_word_actual_independent_birth_death_isolation_owner_must_be_eligible(field, value, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    raw['evidence']['policy_controls']['actual_birth_death'][field] = value
    assert grade(task, raw) == 'INVALID'


@pytest.mark.parametrize('mutation', ['encoder_group', 'encoder_count', 'encoder_stationarity', 'routed_owner', 'direct_output_credit'])
def test_free_word_encoder_and_independent_joint_prior_cannot_alias_routed_or_no_encoder_hosts(mutation, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    evidence = raw['evidence']; controls = evidence['policy_controls']
    if mutation == 'encoder_group': controls['roles'][0].remove('encoder')
    elif mutation == 'encoder_count': del evidence['guards']['optimizer_updates']['encoder']
    elif mutation == 'encoder_stationarity': del controls['diagnostics']['stationarity_lr']['g1']
    elif mutation == 'routed_owner': controls['routed_owner'] = {'owner': 'particlegan.routing.RoutedRowControl'}
    else: evidence['guards']['mechanism_audit']['mechanisms']['direct_particle_gain']['host_activation_credit'] = True
    assert grade(task, raw) == ('INCOMPLETE' if mutation in {'encoder_count', 'encoder_stationarity'} else 'INVALID')


@pytest.mark.parametrize('field,value', [('canonical_words', ['apple']), ('actual_resources', {'num_particles': 5}),
    ('reconstruction_training_loss', True), ('original_capacity_or_qualification_credit', True),
    ('objective', 'joint_RpGAN_plus_new_reconstruction_MSE'), ('family', 'atlas')])
def test_word_joint_objective_target_and_resource_provenance_are_required(field, value, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    raw['evidence']['host'][field] = value
    assert grade(task, raw) == 'INVALID'


@pytest.mark.parametrize('owner', OWNERS)
def test_word_every_requested_policy_owner_must_actually_exist(owner, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    raw['evidence']['policy_controls']['enabled'][owner] = False
    assert grade(task, raw) == 'BLOCKED'


@pytest.mark.parametrize('mutation', ['family', 'missing_family', 'noise', 'diagnostic', 'free_inverse'])
def test_word_selected_observation_binds_free_inverse_and_same_family(mutation, declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    observed = raw['evidence']['policy_observations'][-1]
    if mutation == 'family': observed['family'] = 'atlas_word_joint'
    elif mutation == 'missing_family': del observed['family']
    elif mutation == 'noise': observed['output_noise'] = True
    elif mutation == 'diagnostic': observed['selected_source'] = 'forced_ema'
    else: observed['reconstruction'] = 'nearest_particle_or_reconstruction_MSE_training'
    raw['evidence']['policy_observation'] = deepcopy(observed)
    assert grade(task, raw) == ('INCOMPLETE' if mutation == 'missing_family' else 'INVALID')


def test_word_complete_public_hooks_need_full20001_and24_actual_pure_clocks(declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    assert task['execution']['steps'] == 20001
    assert len(raw['evidence']['policy_observations']) == 24
    assert raw['evidence']['policy_observations'][-1]['completed_steps'] == 20001
    mutated = deepcopy(raw)
    controls = mutated['evidence']['policy_controls']
    controls['completed_steps'] = 20000
    controls['lifecycle'].update(end_completed_steps=20000, observed_updates=20000, calls=dict.fromkeys(HOOKS, 20000))
    assert grade(task, mutated) == 'INCOMPLETE'
    for field in ['policy_purity', 'policy_observations', 'rng_audits']:
        mutated = deepcopy(raw); mutated['evidence'][field].pop(0)
        assert grade(task, mutated) in {'INCOMPLETE', 'INVALID'}


def test_word_observer_needs_separate_global_named_stream_and_complete_checkpoint_proof(declarations, tmp_path):
    task = declarations['five_word_joint']; raw = synthetic_receipt(task, tmp_path)
    mutated = deepcopy(raw); del mutated['evidence']['policy_purity'][0]['global_rng_before_sha256']
    assert grade(task, mutated) == 'INCOMPLETE'
    mutated = deepcopy(raw); mutated['evidence']['policy_purity'][0]['global_rng_after_sha256'] = sha('changed-global')
    assert grade(task, mutated) == 'INVALID'
    mutated = deepcopy(raw)
    mutated['evidence']['rng_audits'][0]['changed_streams'] = [json.dumps(['prior', 'latent', 'indices', 'cuda:0'])]
    assert grade(task, mutated) == 'INVALID'
    (Path(raw['evidence']['artifact_root']) / 'state.pt').write_bytes(b'changed software complete state')
    assert grade(task, raw) == 'INVALID'
