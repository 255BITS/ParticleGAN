"""A named independent joint-cloud law with the original free word encoder.

The complete generated atom is (G(z_effective), z_effective).  This changes
the old callback's split-code DV12 law explicitly, while preserving its clean
BiGAN function, model degrees, objective terms and numerical question.  No
declaration grants original, independent-Atlas or learned qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .contracts import stable_hash
from .policy_declaration_sources import DECLARATION_SOURCES

COHORT = "word_joint_policy_min11_v1"
FAMILY = "atlas_word_joint_min11"
PARENT_ID = "five_word_joint_acquisition"
TASK_ID = PARENT_ID + "_" + COHORT
SHARED_OVERRIDES = {"lr": .0053125, "prior_lr_mult": 1.5}
HOST_RESOURCES = {"num_particles": 11, "z_dim": 2, "batch_size": 256}
ORIGINAL_RESOURCES = {"num_particles": 5, "z_dim": 2, "batch_size": 256}
RESOURCE_ADAPTATION = {"schema_version": 1, "owner": "task", "original": ORIGINAL_RESOURCES,
    "actual": HOST_RESOURCES, "minimum_complete_public_owners": 11,
    "reason": "Original N5 fails public k>=4; N6 through N10 fail isolation leave-one-out reference-half guard. N11 has k5<6.",
    "eligibility_source": "particlegan/birth_death.py:ParticleBirthDeath.__init__",
    "isolation": True, "controls_disabled": False, "numerical_equivalence_to_parent": False,
    "historical_capacity_or_qualification_credit": False,
    "comparisons": "Resource-matched N11 only; no N5 or N6 numerical-equivalence claim."}
HOST_SOURCE = "benchmarks/toy_audit/api_images.py"
THRESHOLDS = [["sample_count", ">=", 1024], ["quality_fraction", ">=", .95],
              ["modes", "==", 5], ["mass_tv", "<=", .1],
              ["reconstruction_exact", "==", 1],
              ["minimum_reconstruction_token_probability", ">=", .9]]
SOURCES = tuple(sorted(DECLARATION_SOURCES | {HOST_SOURCE, "benchmarks/toy_audit/definition_quality.py",
    "benchmarks/transfer_suite/protocol.py", "benchmarks/locked_shared/observation.py",
    "benchmarks/locked_shared/baseline.py", "particlegan/policy.py", "particlegan/recipes.py",
    "particlegan/training.py", "particlegan/init.py", "particlegan/particle_prior.py",
    "particlegan/vicreg_loss.py", "particlegan/birth_death.py", "experiments/forge/rng.py",
    "experiments/forge/policy_adapters.py", "experiments/forge/mechanisms.py",
    "experiments/forge/routed_policy_adapters.py",
    "experiments/forge/api.py", "experiments/forge/boundaries.py", "experiments/forge/policy_cohorts.py",
    "experiments/forge/sampling.py", "experiments/forge/artifacts.py",
    "experiments/forge/word_joint_policy_contracts.py", "experiments/forge/word_joint_policy_adapters.py"}))


def observation():
    return dict(schema_version=1, weight_selector="state_selected", sampler="ServedModel.generate",
        output_noise=False, latent_policy="actual_selected_public_policy", row_selection="uniform_eleven_actual_prior_rows",
        eval_streams="forge-rng-v1_isolated", diagnostic_weights=[], diagnostic_credit=False,
        generated_output="word_coordinates_of_complete_joint_atom",
        reconstruction="selected_free_encoder_on_all_five_known_canonical_inputs_then_same_served_generator",
        unseen_word_generalization_claim=False)


def contract(sources):
    return dict(schema_version=1, cohort=COHORT, family=FAMILY, owner="particlegan.UpdatePolicy",
        lifecycle="ordered_public_update", execution_path="public_components", row_policy="independent",
        row_semantics="independent", schedule="preserve_resolved_recipe_none", external_limit="task.execution.steps",
        execution_device="cuda", precision="preserve_original_fp32_no_autocast", cpu_controls_scope="structural_only",
        numerical_equivalence_to_parent=False, controls="resolved_recipe_requested_enabled_eligible_applied",
        table_owner="prior.z", encoder_owner="auxiliary_original_free_continuous_WordEncoder",
        encoder_mode="none", encoder_optimizer_role="encoder", prior_law="eleven_uniform_raw_learned_2D_rows_for_five_words",
        resource_adaptation=deepcopy(RESOURCE_ADAPTATION), original_five_row_task_status="BLOCKED",
        generation_callback="joint_generation(model,z_effective)=concat(model(z_effective),z_effective)",
        training_latent_policy="actual_DV12_same_effective_code_for_word_generator_and_joint_critic",
        original_split_code_law="old_host_word_generator_receives_perturbed_code_but_joint_critic_receives_raw_code",
        latent_law_change="explicit_complete_joint_atom_for_independent_row_evidence_and_birth_death",
        training_output_noise="public_learned_sigma_on168_word_coordinates_only_code2_coordinates_untouched",
        prior_requirement=dict(kind="particle_cloud", learnable=True, sigma=0., standardize=False,
                               actual_rows=11, canonical_target_words=5),
        evaluation_output_noise=False, free_reconstruction="G(E(word))_under_selected_public_latent_sampling",
        objective="original_joint_RpGAN_real(word,E(word))_fake(G(z),z)_plus_Recipe.prior_reg_times_spread",
        reconstruction_training_loss=False, reconstruction_eval_is_training_signal=False,
        protected_contexts="no_fabricated_RoutedBatch_guard_or_row_to_word_pairing",
        direct_response_scope="latent_table_damping_only_not_direct_output_particle_response",
        original_scaffold="G2_64_128_168_E168_128_64_2_D170_256_128_1_softmax28x6",
        initializer="parent_named_deterministic_orthogonal_original_classes",
        clean_function_and_gradient_equivalence="matched_N11_clean_joint_function_only_not_DV12_on",
        learned_or_Atlas_qualification_reuse=False,
        checkpoint="complete_public_policy_models_optimizers_streams_and_cursor", sources=deepcopy(sources))


def _execution_hash(task):
    return stable_hash({key: task.get(key) for key in
        ("schema_version", "adapter", "execution", "requires_capabilities", "dependencies")})


def _variant(parent, pin, sources):
    task = deepcopy(parent)
    task.update(id=TASK_ID, task_cohort=COHORT, policy_family=FAMILY, policy_parent=deepcopy(pin))
    task["execution"].update(resources=deepcopy(HOST_RESOURCES), device="cuda", policy_contract=contract(sources),
        resource_adaptation=deepcopy(RESOURCE_ADAPTATION),
        policy_recipe_overrides={"encoder_mode": "none", "row_policy": "independent"},
        policy_recipe_overrides_provenance=dict(family=FAMILY, original_free_encoder=True,
            complete_joint_atom=True, same_code_DV12=True, words_only_output_noise=True,
            historical_split_code_numerical_equivalence=False, evidence_reuse=False),
        schedule_policy="public_policy_stationarity_no_declared_horizon")
    task["execution"]["prior"]["exception_reason"] = (
        "Explicit min11 independent joint-cloud variant: eleven actual raw learned rows for the unchanged five-word question; "
        "the original N5 and N6 isolation-invalid variant remain BLOCKED, with no historical capacity or qualification credit.")
    task["resources"].update(device="cuda", gpus=1, gpu_memory_mb=2048, allow_cpu=False)
    task["evaluation"].update(scoring_weights="state_selected", policy_observation=observation())
    task["requires_capabilities"] = ["public_components", "checkpoint", "named_rng", "served_sampling",
        "policy_controls", "policy_serving", "learned_locations", "particle_cloud"]
    return task


def make_variant(root):
    root = Path(root).resolve()
    raw = (root / f"configs/forge/tasks/{PARENT_ID}.json").read_bytes()
    parent = json.loads(raw)
    sources = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in SOURCES}
    pin = dict(id=PARENT_ID, task_sha256=hashlib.sha256(raw).hexdigest(),
        execution_fingerprint=_execution_hash(parent), evaluation_fingerprint=stable_hash(parent["evaluation"]))
    return _variant(parent, pin, sources)


def validate_task(task, *, root=None):
    if (task.get("id") != TASK_ID or task.get("task_cohort") != COHORT or task.get("policy_family") != FAMILY):
        raise ValueError("explicit atlas_word_joint_min11 task required")
    sources = task.get("execution", {}).get("policy_contract", {}).get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("complete word-joint source binding required")
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
           for h in sources.values()):
        raise ValueError("invalid word-joint source hash")
    pin = task.get("policy_parent")
    if (not isinstance(pin, dict) or set(pin) != {"id", "task_sha256", "execution_fingerprint", "evaluation_fingerprint"}
            or pin.get("id") != PARENT_ID or any(not isinstance(pin[k], str) or len(pin[k]) != 64
                or any(c not in "0123456789abcdef" for c in pin[k])
                for k in ("task_sha256", "execution_fingerprint", "evaluation_fingerprint"))):
        raise ValueError("complete original word-parent binding required")
    if root is not None and stable_hash(task) != stable_hash(make_variant(root)):
        raise ValueError("word-joint original source/parent/objective/gates binding drift")
    e, v = task["execution"], task["evaluation"]
    if (stable_hash(e.get("resources")) != stable_hash(HOST_RESOURCES)
            or stable_hash(e.get("resource_adaptation")) != stable_hash(RESOURCE_ADAPTATION)
            or e.get("steps") != 20001 or e.get("original_schedule_horizon") != 20000
            or e.get("produces_state") is not False
            or stable_hash(e["policy_contract"]) != stable_hash(contract(sources))
            or e.get("policy_recipe_overrides") != {"encoder_mode": "none", "row_policy": "independent"}
            or v.get("policy_observation") != observation() or v.get("scoring_weights") != "state_selected"
            or v.get("observations") != 24 or v.get("minimum_stable_checks") != 5
            or v.get("thresholds") != THRESHOLDS or v.get("eval_samples") != 1024):
        raise ValueError("original word-joint scientific contract changed")
    return deepcopy(e["policy_contract"])


def validate_word_observation(task):
    validate_task(task)
    return deepcopy(task["evaluation"]["policy_observation"])


def word_recipe_overrides(task):
    validate_task(task)
    return deepcopy(task["execution"]["policy_recipe_overrides"])


def resolved_recipe(candidate, task):
    from particlegan import get_recipe
    validate_task(task)
    if (candidate.get("task_cohort") != COHORT or candidate.get("recipe_preset") != "atlas"
            or candidate.get("recipe_overrides") != SHARED_OVERRIDES):
        raise ValueError("min11 word joint needs its explicit family and fixed shared C6 pair")
    return get_recipe("atlas", **SHARED_OVERRIDES, **HOST_RESOURCES, encoder_mode="none",
                      row_policy="independent", prior_kind="particles", sigma_rel=0., standardize=False)


def blockers(task, recipe):
    try:
        expected = resolved_recipe(dict(task_cohort=COHORT, recipe_preset="atlas", recipe_overrides=SHARED_OVERRIDES), task)
        if stable_hash(recipe.to_dict() if hasattr(recipe, "to_dict") else recipe) != stable_hash(expected.to_dict()):
            raise ValueError("word-joint resolved Recipe drift")
        return []
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        return [str(error)]
