"""Explicit image host profiles around the shared public construction API.

Unprofiled cards retain their declared raw host. The opt-in published profile
binds architecture/resources/data to the archived passing host declaration; it
never imports a PASS or the historical optimizer into the current formulation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .contracts import canonical, identifier

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "published_residual16"
PROFILE_REVISION = 1
PROFILE_SOURCE = "benchmarks/transfer_suite/plans/default_comparison.json"
PROFILE_SHA256 = "2e62a935dd702c9851165354cb580d18ffd4c47b72e0568562d0f50256028883"

# Resource/data/model fields are consumed by the image adapter/factory. The
# remaining legacy optimizer fields are retained only as host provenance;
# Recipe supplies every current formulation setting, as on existing raw cards.
HOST_FIELDS = (
    "family", "split", "pattern", "architecture", "width", "z_dim", "steps",
    "modes", "batch_size", "particles", "lr_g", "lr_d", "adam_betas", "noise_std",
    "gradient_penalty", "penalty_coeff", "kappa", "prior_weight", "ema_decay",
)
ARCHITECTURES = frozenset({"transpose", "residual_upsample", "mean_discriminator", "uniform_generator"})
PATTERN_MODES = {"stripes2": 2, "bars4": 4, "blobs4": 4, "intensity2": 2, "bars8": 8}


def profile_declaration() -> dict:
    """Return the complete versioned opt-in, suitable for a frozen task card."""
    return {"id": PROFILE_ID, "revision": PROFILE_REVISION,
            "source": {"path": PROFILE_SOURCE, "sha256": PROFILE_SHA256}}


def _published_row(task, root):
    declaration = task["execution"]["image_profile"]
    if canonical(declaration) != canonical(profile_declaration()):
        raise ValueError("unsupported or mismatched image profile declaration")
    path = Path(root) / PROFILE_SOURCE
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != PROFILE_SHA256:
        raise ValueError("published image profile source changed; declare a new profile revision")
    rows = [row for row in json.loads(data) if row.get("spec", {}).get("name") == task["execution"].get("host")
            and row.get("spec", {}).get("runner") == "image"]
    if len(rows) != 1 or rows[0].get("architecture") != "residual16":
        raise ValueError("image host is not uniquely declared by the published residual16 profile")
    return rows[0]


def _validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != set(HOST_FIELDS):
        raise ValueError("image host card has missing or unsupported fields; architecture options cannot be ignored")
    if spec["architecture"] not in ARCHITECTURES:
        raise ValueError("unsupported image architecture")
    if spec["pattern"] not in PATTERN_MODES or spec["modes"] != PATTERN_MODES[spec["pattern"]]:
        raise ValueError("image pattern and mode count differ")
    for key in ("width", "z_dim", "steps", "modes", "batch_size", "particles"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"image {key} must be a positive integer")
    if type(spec["noise_std"]) not in (int, float) or not math.isfinite(spec["noise_std"]) or spec["noise_std"] < 0:
        raise ValueError("image data noise must be finite and nonnegative")


def _validate_profile_measurement(task, spec, published):
    execution, evaluation = task["execution"], task["evaluation"]
    if execution.get("steps") != spec["steps"]:
        raise ValueError("image execution budget differs from its published host")
    prior = execution.get("prior", {})
    from .priors import recipe_owned_prior
    if (prior.get("kind") != "particle_cloud" or prior.get("sigma") != 0
            or prior.get("standardize") is not False or (not recipe_owned_prior(task) and prior.get("learnable") is not True)
            or not isinstance(prior.get("exception_reason"), str) or not prior["exception_reason"].strip()):
        raise ValueError("published finite image host requires its explicit learned sigma-zero cloud exception")
    thresholds = published["thresholds"]
    fixed = {"kind": "transfer_sustained", "scoring_weights": "live",
             "observations": thresholds["observations"],
             "minimum_stable_checks": thresholds["minimum_stable_checks"],
             "measurement": thresholds,
             "thresholds": [["modes", ">=", thresholds["modes"]], ["hq", ">=", thresholds["hq_min"]]]}
    from .sampling import ENUMERATED_PRIOR_CLEAN, executed_receipt
    fixed.update(executed_receipt(ENUMERATED_PRIOR_CLEAN, eval_output_noise="clean"))
    if any(canonical(evaluation.get(key)) != canonical(value) for key, value in fixed.items()):
        raise ValueError("image profile task must retain its frozen live enumeration, budget and measurement gates")


def resolve_image_spec(task: dict, *, root: Path | str | None = None) -> dict:
    """Validate one task and return its explicit data/resource/model card.

    No runtime inheritance: a profiled task must already contain the complete
    resolved declaration. Unprofiled raw cards are copied without modification.
    """
    if task.get("adapter") != "transfer_image":
        raise ValueError("image profiles apply only to transfer_image tasks")
    execution = task["execution"]
    spec = deepcopy(execution["host_definition"])
    _validate_spec(spec)
    if "image_profile" not in execution:
        return spec
    row = _published_row(task, ROOT if root is None else root)
    expected = {key: deepcopy(row["spec"][key]) for key in HOST_FIELDS}
    if canonical(spec) != canonical(expected):
        raise ValueError("resolved image host card differs from its declared published profile")
    _validate_profile_measurement(task, spec, row["spec"])
    return spec


def image_profile_blockers(task: dict, *, root: Path | str | None = None) -> list[str]:
    try:
        resolve_image_spec(task, root=root)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"{task.get('id', '<task>')}: {error}"]
    return []


def task_from_profile(base_task: dict, task_id: str, *, root: Path | str | None = None) -> dict:
    """Materialize a new frozen card once; never mutate/inherit a task at runtime."""
    identifier(task_id, "image profile task id")
    if task_id == base_task.get("id"):
        raise ValueError("a host-profile variant needs a distinct task id")
    task = deepcopy(base_task)
    task["id"] = task_id
    task["execution"]["image_profile"] = profile_declaration()
    row = _published_row(task, ROOT if root is None else root)
    task["execution"]["host_definition"] = {key: deepcopy(row["spec"][key]) for key in HOST_FIELDS}
    resolve_image_spec(task, root=root)
    return task


def build_image_models(context, spec: dict):
    """Build exactly the resolved architecture through named public init streams.

    The caller resolves once, then uses the same card for model construction,
    target generation and host resources. No optimizer or training loop lives here.
    """
    _validate_spec(spec)
    from benchmarks.transfer_suite.image_tasks import Generator, Discriminator
    generator = context.construct(lambda: Generator(spec), component="generator").to(context.device)
    discriminator = context.construct(lambda: Discriminator(spec), component="discriminator").to(context.device)
    return generator, discriminator
