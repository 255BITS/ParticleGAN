"""Test-only proof of frozen source identity across the restoration API rename.

This does not admit changed code under an old task or publication. The callers
must still assert that the actual source-bound preflight rejects current bytes.
Historical execution requires its original checkout, or a separately qualified
new cohort. No scientific declaration or recorded digest is rewritten here.
"""
from hashlib import sha256
from pathlib import Path


RESTORATION_MODULES = frozenset({
    "benchmarks/toy_audit/api_model_ae.py",
    "benchmarks/toy_audit/api_model_ddgan.py",
    "benchmarks/toy_audit/api_model_vae.py",
    "experiments/forge/model_toy_publication.py",
})


def frozen_restoration_source(root, relative, expected):
    """Return exact frozen bytes, accepting only the known API spelling change."""
    current = (Path(root) / relative).read_bytes()
    if sha256(current).hexdigest() == expected:
        return current
    assert relative in RESTORATION_MODULES, relative
    frozen = current.replace(b"Recipe.from_dict(", b"Recipe(**")
    assert frozen != current, relative
    assert sha256(frozen).hexdigest() == expected, relative
    return frozen


def model_source_blockers(root, task):
    """Expected current preflight refusal; frozen source hashes remain intact."""
    for relative, expected in task["evaluation"].get("sources", {}).items():
        frozen_restoration_source(root, relative, expected)
        if sha256((Path(root) / relative).read_bytes()).hexdigest() != expected:
            return [f"model source binding differs: {relative}"]
    return []
