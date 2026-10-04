"""Private common-26 display proposal. Numerical acceptance is unimplemented.

This pure projection consumes supplied declarations, never evidence files. Old
row scores cannot become fresh scores. Installing this module is not authorized.
"""
from copy import deepcopy
import hashlib
import json
import re

SCHEMA = "pg_fresh_common26_comparison_projection_v3"
FAMILIES = (
    ("r1r2", "R1/R2"), ("bcap", "BCap"), ("k3p", "K3P"),
    ("ka2", "KA2"), ("e22", "E22"), ("atlas", "Full Atlas"),
    ("release07-gan-v3-mog", "GAN v3 release 0.7 (MoG)"),
    ("release07-gan-v3-cloud", "GAN v3 release 0.7 (cloud)"),
    ("k3p-no-anchor", "K3P without critic anchor"),
    ("k3p-no-penalty", "K3P without critic penalty"),
    ("k3p-no-a2", "K3P without A2"),
    ("k3p-no-training-noise", "K3P without training output noise"),
)
ATLAS_CONFIG = {
    "path": "configs/100gaussians/atlas.json",
    "sha256": "a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4",
}
TASKS_BY_TIER = {
    "1": ("two_pole", "unused_token_hold", "ae_gan_hold", "ring16_acquisition",
          "five_word_joint_acquisition"),
    "2": ("trajectory", "residual_student", "unipolar", "cover_leftover",
          "mid_scale_identity", "mode_hold", "vector_two_broad", "vector_unequal_mass",
          "vector_unequal_width", "vector_anisotropic", "vector_overlap", "vector_spiral",
          "img_stripes2", "img_bars4", "img_blobs4", "img_intensity2", "grid100",
          "rotated100", "staggered100"),
    "3": ("ring_hold", "ring_extension"),
}
UNSCORED = {"required": 26, "passed": None, "status": "NOT_RUN",
            "execution_status": "NOT_RUN", "scientific_status": "UNKNOWN",
            "accepted_record": None}
_CHOICE_KEYS = {"candidate_id", "configuration_id", "candidate_revision",
                "declaration_sha256", "recipe_sha256", "source_digest",
                "audit_reference"}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value):
    return (isinstance(value, str) and bool(value.strip())
            and not any(ord(char) < 32 for char in value))


def _audit_path(value):
    # Source declarations/audits must be portable, never this projection's output.
    return (_text(value) and "\\" not in value and ":" not in value
            and all(part not in {"", ".", ".."} for part in value.split("/"))
            and value.split("/")[-1].casefold() not in {
                "technique-inventory.json", "technique-inventory.md"})


def _choice(family, value):
    if value is None:
        return None
    keys = _CHOICE_KEYS | ({"base_config"} if family == "atlas" else set())
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("chosen configuration requires exact display-identity fields")
    if not all(_text(value[key]) for key in ("candidate_id", "configuration_id")):
        raise ValueError("chosen configuration ids must be nonempty strings")
    if not all(_sha(value[key]) for key in ("declaration_sha256", "recipe_sha256")):
        raise ValueError("chosen configuration requires declaration and Recipe hashes")
    if not all(_sha(value[key]) for key in ("candidate_revision", "source_digest")):
        raise ValueError("a chosen configuration must be fully frozen; pending choices are null")
    reference = value["audit_reference"]
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}
            or not _audit_path(reference["path"]) or not _sha(reference["sha256"])
            or type(reference["bytes"]) is not int or reference["bytes"] <= 0):
        raise ValueError("chosen configuration requires a portable noncircular audit-reference pin")
    if family == "atlas" and value["base_config"] != ATLAS_CONFIG:
        raise ValueError("Full Atlas requires the complete requested original configuration")
    return deepcopy(value)


def _digest(value):
    wire = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(wire.encode()).hexdigest()


def _representation(row, catalogs, choice):
    unknown = {"kind": "UNKNOWN", "basis": "UNKNOWN"}
    if choice is None or choice["candidate_revision"] is None or choice["source_digest"] is None:
        return unknown
    bindings = row.get("bindings", {})
    if not isinstance(bindings, dict):
        return unknown
    identity = {"candidate_id": row.get("candidate_id"),
                "configuration_id": row.get("configuration_id", bindings.get("configuration_id")),
                "candidate_revision": row.get("candidate_revision"),
                "recipe_sha256": bindings.get("recipe_sha256"),
                "source_digest": bindings.get("source_digest")}
    if any(choice[key] != value for key, value in identity.items()):
        return unknown
    references, contracts = bindings.get("task_contracts"), catalogs.get("task_contracts", {})
    required_tasks = {task for tasks in TASKS_BY_TIER.values() for task in tasks}
    if (not isinstance(references, dict) or set(references) != required_tasks
            or not isinstance(contracts, dict)):
        return unknown
    kinds = set()
    for digest in references.values():
        contract = contracts.get(digest) if _sha(digest) else None
        if not isinstance(contract, dict) or _digest(contract) != digest:
            return unknown
        prior = contract.get("prior")
        kind = {"mog": "MoG", "particle_cloud": "Particles", "particles": "Particles"}.get(
            prior.get("kind") if isinstance(prior, dict) else None)
        if kind is None:
            return unknown
        kinds.add(kind)
    label = "MoG / Particles" if len(kinds) == 2 else next(iter(kinds))
    return {"kind": label, "basis": "DECLARED"}


def _refuse_fresh_claim(value):
    if not isinstance(value, dict) or set(value) != set(UNSCORED):
        raise ValueError("fresh common-26 acceptance is unimplemented")
    # Equality alone would admit False as 0 and arbitrary false-like fields.
    if (value["passed"] is not None or value["accepted_record"] is not None
            or type(value["required"]) is not int or value != UNSCORED):
        raise ValueError("historical or manual numbers cannot supply fresh common-26 credit")


def project_common26(scientific_rows, catalogs, *, chosen_configurations=None):
    """Return twelve unscored slots; metadata never authorizes fresh credit.

    A choice is a syntactic display identity awaiting root audit, not a receipt.
    Declared representation is shown only for exactly matching chosen identities
    and complete content-bound task-prior declarations. Atlas's Particles label
    describes its requested full configuration, not an executed common cohort.
    """
    family_ids = {family for family, _ in FAMILIES}
    if not isinstance(scientific_rows, list) or not isinstance(catalogs, dict):
        raise ValueError("projection requires supplied row and catalog metadata")
    expected_tiers = {tier: list(tasks) for tier, tasks in TASKS_BY_TIER.items()}
    if (catalogs.get("view") != "discriminator_stability"
            or type(catalogs.get("view_revision")) is not int or catalogs["view_revision"] != 3
            or catalogs.get("tier_requirements") != expected_tiers):
        raise ValueError("projection requires the unchanged ordered revision-3 common-26 view")
    # Cached display payloads are never accepted as evidence or an upgrade path.
    cached = catalogs.get("common26_display")
    if cached is not None:
        cache_keys = {"schema", "view", "view_revision", "required", "tier_required",
                      "numerical_acceptance", "rows"}
        if (not isinstance(cached, dict) or cached.get("schema") != SCHEMA
                or set(cached) != cache_keys or cached.get("view") != "discriminator_stability"
                or type(cached.get("view_revision")) is not int or cached["view_revision"] != 3
                or type(cached.get("required")) is not int or cached["required"] != 26
                or cached.get("tier_required") != {"1": 5, "2": 19, "3": 2}
                or cached.get("numerical_acceptance") != "UNIMPLEMENTED"
                or not isinstance(cached.get("rows"), list) or len(cached["rows"]) != 12):
            raise ValueError("cached common-26 display cannot authorize numerical acceptance")
        row_keys = {"family_id", "label", "chosen_configuration", "requested_configuration",
                    "representation", "fresh_common", "eligibility", "evidence_scope"}
        for (family, _), item in zip(FAMILIES, cached["rows"]):
            if (not isinstance(item, dict) or set(item) != row_keys
                    or item.get("family_id") != family):
                raise ValueError("cached common-26 rows must be unscored")
            _refuse_fresh_claim(item.get("fresh_common"))
    indexed = {}
    for row in scientific_rows:
        if (not isinstance(row, dict) or not isinstance(row.get("trainer_family"), str)
                or row["trainer_family"] not in family_ids):
            raise ValueError("projection accepts only registered main family rows")
        if "fresh_common" in row:
            _refuse_fresh_claim(row["fresh_common"])
        family = row["trainer_family"]
        if family in indexed:
            raise ValueError("one whole configuration slot per family is required")
        indexed[family] = row
    if set(indexed) != family_ids:
        raise ValueError("all twelve family slots, including Atlas, are required")
    choices = {} if chosen_configurations is None else chosen_configurations
    if not isinstance(choices, dict) or set(choices) - family_ids:
        raise ValueError("chosen configurations must be keyed by registered families")
    rows = []
    for family, label in FAMILIES:
        choice = _choice(family, choices.get(family))
        atlas = family == "atlas"
        rows.append({
            "family_id": family, "label": label, "chosen_configuration": choice,
            "requested_configuration": deepcopy(ATLAS_CONFIG) if atlas else None,
            "representation": ({"kind": "Particles", "basis": "DECLARED"} if atlas
                               else _representation(indexed[family], catalogs, choice)),
            "fresh_common": deepcopy(UNSCORED),
            "eligibility": {"status": "BLOCKED" if atlas else "UNKNOWN", "reasons": (
                ["Current common tasks lack the requested ordered policy lifecycle and serving contracts."]
                if atlas else ["Chosen configuration source eligibility awaits root audit."])},
            "evidence_scope": "display_only",
        })
    return {"schema": SCHEMA, "view": "discriminator_stability", "view_revision": 3,
            "required": 26, "tier_required": {"1": 5, "2": 19, "3": 2},
            "numerical_acceptance": "UNIMPLEMENTED", "rows": rows}
