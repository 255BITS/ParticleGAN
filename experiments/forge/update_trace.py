"""Retain scalar public step returns after the complete applied update.

This is an observer: it performs no forwards, draws or parameter writes. Bulk
JSONL stays in the run artifact tree; only its hash enters compact evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .contracts import file_hash


def _scalars(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item() if value.numel() == 1 else None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _scalars(item) for key, item in value.items()
                if not isinstance(item, torch.Tensor) or item.numel() == 1}
    if isinstance(value, (tuple, list)):
        return [_scalars(item) for item in value]
    return None


def append_update(output, values):
    """Append a scalar-only return, after trainer.step has finished its policy."""
    path = Path(output) / "public-updates.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(_scalars(values), allow_nan=False, sort_keys=True) + "\n")


def receipt(output):
    path = Path(output) / "public-updates.jsonl"
    if not path.is_file():
        return None
    with path.open() as stream:
        rows = sum(1 for _ in stream)
    return dict(path=path.name, path_base="adapter_output", sha256=file_hash(path), rows=rows,
                scope="scalar_public_step_returns_after_complete_update",
                observer_additional_forwards=0, observer_additional_draws=0,
                movement_scope="only movement statistics explicitly returned by the trainer; "
                "optimizer post-hook diagnostics may precede displacement policies")
