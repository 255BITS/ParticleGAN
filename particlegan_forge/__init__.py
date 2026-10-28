"""Namespaced Forge & Tier API exports (`particlegan_forge`).

A thin, wheel-shipped facade over the repository's (unpackaged)
``experiments/forge`` semantics.  It reads/writes only
``configs/particlegan_forge`` and ``reports/particlegan_forge``.
"""
from .contracts import (
    GATES,
    HOLD_REASONS,
    IMPORTANCES,
    REMOVAL_REASONS,
    atomic_json,
    atomic_text,
    canonical,
    file_hash,
    file_lock,
    finite_number,
    has_nonfinite,
    identifier,
    json_safe,
    read_json,
    stable_hash,
    utc_now,
)
from .gif import render_training_gif
from .rng import NamedStreams, RNG_VERSION, STREAM_NAMES, deterministic_scope
from .tiers import (
    NAMESPACE,
    AttemptTimeout,
    CandidateSpec,
    ForgeRegistry,
    ForgeWorkspace,
    MetricBound,
    TaskSpec,
    TierAssignment,
    TierSpec,
    TierView,
    grade_sustained_observations,
    resolve_device,
)

__all__ = [
    "AttemptTimeout",
    "NAMESPACE",
    "CandidateSpec",
    "ForgeRegistry",
    "ForgeWorkspace",
    "GATES",
    "HOLD_REASONS",
    "IMPORTANCES",
    "MetricBound",
    "NamedStreams",
    "REMOVAL_REASONS",
    "RNG_VERSION",
    "STREAM_NAMES",
    "TaskSpec",
    "TierAssignment",
    "TierSpec",
    "TierView",
    "atomic_json",
    "atomic_text",
    "canonical",
    "deterministic_scope",
    "file_hash",
    "file_lock",
    "finite_number",
    "grade_sustained_observations",
    "has_nonfinite",
    "identifier",
    "json_safe",
    "read_json",
    "render_training_gif",
    "resolve_device",
    "stable_hash",
    "utc_now",
]
