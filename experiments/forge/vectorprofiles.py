"""Explicit published vector architectures with current Forge prior/init policy.

These profiles bind architecture only. They neither replay historical results nor
restore the historical finite prior, location init_std=.5, constructor draw order,
optimizer settings, or finite-atom evaluation exemptions. Current task data,
resources, MoG and full-component gates remain declared by the materialized card;
FormulationContext still owns named construction and initialization streams.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch.nn as nn

from .contracts import canonical, identifier

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "published_vector_discriminators"
PROFILE_REVISION = 1
PLAN_SOURCE = "benchmarks/transfer_suite/plans/default_comparison.json"
LEADING_SOURCE = "reports/transfer_suite/unadjusted/leading_profile.json"
PROFILE_SOURCES = {
    PLAN_SOURCE: "2e62a935dd702c9851165354cb580d18ffd4c47b72e0568562d0f50256028883",
    LEADING_SOURCE: "38327c94dd8c1ad2729ab07a5866191d2c4ec05b3acb841498d5abd786d691dd",
}
ARCHITECTURE_FIELDS = frozenset({"hidden", "layers", "fourier", "d_hidden", "d_layers", "research_discriminator"})
# Legacy optimizer values are provenance only; the current Recipe still owns
# optimizer settings. Unknown options must never silently disappear in a factory.
COMMON_FIELDS = frozenset({"hidden", "layers", "fourier", "z_dim", "particles", "batch", "steps",
    "lr", "d_lr_mult", "prior_lr_mult", "prior_reg", "betas", "ema_decay", "reg_arm", "reg_coeff",
    "reg_kappa", "d_every", "g_every", "family", "split", "kind"})
TARGET_FIELDS = {
    "gaussian_mixture": frozenset({"means", "covariances", "masses", "identifiable"}),
    "spiral": frozenset({"turns", "radius_min", "radius_max", "noise"}),
}
OPTIONAL_FIELDS = frozenset({"d_hidden", "d_layers", "research_discriminator"})
# Declared point-mass tanh hosts (zero_tanh_smoke and zero_tanh_rail_escape).
# output_activation is part of that card; it is not a gaussian_mixture or spiral
# profile and must not be rewritten into one.
TANH_HOST_FIELDS = frozenset({"hidden", "layers", "fourier", "z_dim", "particles", "batch", "steps", "output_activation"})
# The adapter attaches gate thresholds after the card is validated. They are not
# part of the host declaration and must not be required to recognize the card.
TANH_HOST_RUNTIME_FIELDS = frozenset({"thresholds"})
CRITIC_FIELDS = {
    "shared_batch_feature_v1": frozenset({"name", "implementation", "width", "layers", "fourier",
        "trunk_normalization", "softplus_beta", "feature", "placement", "kernel_scales", "eps",
        "batch_dependence", "batch_statistics"}),
    "shared_critic_v1": frozenset({"activation", "beta", "branch_hidden", "branch_scale", "features",
        "fourier", "frequency_scale", "hidden", "implementation", "layers", "name", "normalization",
        "output_scale", "quadratic_scale", "raw_linear_skip", "residual", "residual_scale"}),
}


def profile_declaration() -> dict:
    return {"id": PROFILE_ID, "revision": PROFILE_REVISION, "scope": "architecture_only",
            "prior_initialization": "current_formulation_named_streams",
            "historical_parity": False, "sources": dict(PROFILE_SOURCES)}


def profile_source_files(task: dict) -> dict[str, str]:
    """Exact extra snapshot files; raw cards need none. No local fallback paths."""
    declaration = task["execution"].get("vector_profile")
    if "vector_profile" not in task["execution"]:
        return {}
    if canonical(declaration) != canonical(profile_declaration()):
        raise ValueError("unsupported or mismatched vector profile declaration")
    return dict(PROFILE_SOURCES)


def _read_sources(task, root):
    documents = {}
    for name, expected in profile_source_files(task).items():
        data = (Path(root) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"published vector profile source changed: {name}; declare a new revision")
        documents[name] = json.loads(data)
    return documents


def _architecture(spec):
    return {key: deepcopy(value) for key, value in spec.items() if key in ARCHITECTURE_FIELDS}


def _published_architecture(task, root):
    documents = _read_sources(task, root)
    rows = [row for row in documents[PLAN_SOURCE]
            if row.get("spec", {}).get("runner") == "vector"
            and row["spec"].get("name") == task["execution"].get("host")]
    if len(rows) != 1:
        raise ValueError("vector host is not uniquely declared by the published profile")
    spec = deepcopy(rows[0]["spec"])
    card = documents[LEADING_SOURCE]["discriminators"].get(spec["name"])
    if card is not None:
        # This is the same discriminator-only layering as declared_spec(), with
        # no call to effective_spec() that could import historical recipe fields.
        from benchmarks.transfer_suite.shared_variants import architecture_spec
        width = card["width"] if card["implementation"] == "shared_batch_feature_v1" else card["hidden"]
        spec = architecture_spec(spec, {"name": card["name"], "overrides": {
            "d_hidden": width, "d_layers": card["layers"], "fourier": card["fourier"],
            "research_discriminator": deepcopy(card)}})
    return _architecture(spec)


def _declared_tanh_host(spec):
    return isinstance(spec, dict) and spec.get("output_activation") == "tanh" and set(spec) == TANH_HOST_FIELDS


def is_tanh_point_mass_host(spec) -> bool:
    """Origin point-mass tanh card, including keys attached only for scoring.

    The card has ``output_activation: tanh`` and no ``kind``. Gaussian and
    spiral hosts keep ``kind`` and are not rewritten into this law.
    """
    if not isinstance(spec, dict) or spec.get("output_activation") != "tanh" or "kind" in spec:
        return False
    return set(spec) - TANH_HOST_RUNTIME_FIELDS == TANH_HOST_FIELDS


class TanhPointMassGenerator(nn.Module):
    """MLP whose range is tanh, for the declared origin point-mass host.

    ``layers`` counts Linear maps. The zero_tanh_smoke card's ``layers: 2`` is
    Linear(z, hidden), LeakyReLU(0.2), Linear(hidden, 2), Tanh. That is the
    2-layer tanh head, not an extra hidden block on top of SimpleMLPGenerator.
    """

    def __init__(self, z_dim: int, hidden: int, layers: int, out_dim: int = 2) -> None:
        super().__init__()
        if type(layers) is not int or layers < 1:
            raise ValueError("tanh host layers must be a positive integer")
        blocks = []
        in_dim = z_dim
        for _ in range(layers - 1):
            blocks.append(nn.Linear(in_dim, hidden))
            blocks.append(nn.LeakyReLU(0.2, inplace=False))
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, out_dim))
        blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        return self.net(z)


def _validate_spec(spec):
    if _declared_tanh_host(spec):
        for key in ("hidden", "layers", "z_dim", "particles", "batch", "steps", "fourier"):
            if type(spec[key]) is not int or spec[key] < (0 if key == "fourier" else 1):
                raise ValueError(f"invalid vector dimension: {key}")
        return
    if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str) or spec["kind"] not in TARGET_FIELDS:
        raise ValueError("unsupported vector target declaration")
    required = COMMON_FIELDS | TARGET_FIELDS[spec["kind"]]
    if not required <= set(spec) or set(spec) - required - OPTIONAL_FIELDS:
        raise ValueError("vector host card has missing or unsupported fields; architecture options cannot be ignored")
    for key in ("hidden", "layers", "z_dim", "particles", "batch", "steps", "d_hidden", "d_layers", "fourier"):
        if key in spec and (type(spec[key]) is not int or spec[key] < (0 if key == "fourier" else 1)):
            raise ValueError(f"invalid vector dimension: {key}")
    if spec["kind"] == "gaussian_mixture":
        dimension = len(spec["means"][0])
        if dimension not in (1, 2) or any(len(mean) != dimension for mean in spec["means"]):
            raise ValueError("Gaussian hosts require consistent one- or two-dimensional means")
        if len(spec["covariances"]) != len(spec["means"]) or any(
                len(covariance) != dimension or any(len(row) != dimension for row in covariance)
                for covariance in spec["covariances"]):
            raise ValueError("Gaussian covariance dimensions differ from the means")
        if dimension == 1 and "research_discriminator" in spec:
            raise ValueError("one-dimensional hosts require the declared MLP critic")
    if "research_discriminator" not in spec:
        return
    card = spec["research_discriminator"]
    if not isinstance(card, dict) or not isinstance(card.get("implementation"), str):
        raise ValueError("unsupported vector discriminator card")
    implementation = card["implementation"]
    if implementation not in CRITIC_FIELDS or set(card) != CRITIC_FIELDS[implementation]:
        raise ValueError("unsupported vector discriminator schema; options cannot be ignored")
    width = card["width"] if implementation == "shared_batch_feature_v1" else card["hidden"]
    if (spec.get("d_hidden"), spec.get("d_layers"), spec["fourier"]) != (width, card["layers"], card["fourier"]):
        raise ValueError("vector discriminator dimensions differ from its explicit card")


def resolve_vector_spec(task: dict, *, root: Path | str | None = None) -> dict:
    """Return an independently copied, explicit host card without inheritance."""
    if task.get("adapter") != "transfer_vector":
        raise ValueError("vector profiles apply only to transfer_vector tasks")
    spec = deepcopy(task["execution"]["host_definition"])
    _validate_spec(spec)
    if "vector_profile" not in task["execution"]:
        if "research_discriminator" in spec:
            raise ValueError("research_discriminator requires an explicit supported vector profile")
        return spec
    expected = _published_architecture(task, ROOT if root is None else root)
    if canonical(_architecture(spec)) != canonical(expected):
        raise ValueError("resolved vector architecture differs from its declared published profile")
    return spec


def vector_profile_blockers(task: dict, *, root: Path | str | None = None) -> list[str]:
    try:
        resolve_vector_spec(task, root=root)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return [f"{task.get('id', '<task>')}: {error}"]
    return []


def task_from_profile(base_task: dict, task_id: str, *, root: Path | str | None = None) -> dict:
    """Materialize architecture changes only; keep the current prior and gates."""
    identifier(task_id, "vector profile task id")
    if task_id == base_task.get("id"):
        raise ValueError("a host-profile variant needs a distinct task id")
    # Validate the base too, so materialization cannot hide unknown options.
    resolve_vector_spec(base_task, root=root)
    task = deepcopy(base_task)
    task["id"] = task_id
    task["execution"]["vector_profile"] = profile_declaration()
    expected = _published_architecture(task, ROOT if root is None else root)
    old = task["execution"]["host_definition"]
    if any(expected[key] != old[key] for key in ("hidden", "layers")):
        raise ValueError("published vector profile would change current generator dimensions")
    task["execution"]["host_definition"] = {
        **{key: value for key, value in old.items() if key not in ARCHITECTURE_FIELDS}, **expected}
    resolve_vector_spec(task, root=root)
    return task


def build_vector_models(context, spec: dict):
    """Construct a resolved host through shared factories and named init streams.

    This only constructs networks. Context initialization, MoG location scale,
    optimizer settings, and all training remain on the current public API path.
    """
    _validate_spec(spec)
    if context.recipe.z_dim != spec["z_dim"]:
        raise ValueError("vector latent dimension differs from its formulation context")
    from lib.toy_models import SimpleMLPGenerator
    from benchmarks.transfer_suite.public_default_verification import vector_discriminator
    if _declared_tanh_host(spec):
        # Point mass in R^2. This card has no kind and no means.
        dimension = 2
        generator = context.construct(lambda: TanhPointMassGenerator(
            context.recipe.z_dim, spec["hidden"], spec["layers"], dimension), component="generator").to(context.device)
    else:
        dimension = len(spec["means"][0]) if spec["kind"] == "gaussian_mixture" else 2
        generator = context.construct(lambda: SimpleMLPGenerator(
            context.recipe.z_dim, spec["hidden"], spec["layers"], dimension), component="generator").to(context.device)
    if dimension == 1:
        from lib.toy_models import SimpleMLPDiscriminator
        discriminator = context.construct(lambda: SimpleMLPDiscriminator(
            1, spec.get("d_hidden", spec["hidden"]), spec.get("d_layers", spec["layers"]),
            spec["fourier"]), component="discriminator").to(context.device)
    else:
        discriminator = context.construct(lambda: vector_discriminator(
            spec, spec.get("research_discriminator")), component="discriminator").to(context.device)
    return generator, discriminator
