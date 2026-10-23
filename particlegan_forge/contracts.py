"""Atomic file contracts, locking, hashing, and identifier validation for particlegan_forge."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

SCHEMA_VERSION = 1
GATES = frozenset({"PASS", "FAIL", "INCOMPLETE", "INVALID", "NOT_RUN", "BLOCKED"})
IMPORTANCES = frozenset({"required", "held", "ranking", "diagnostic"})
HOLD_REASONS = frozenset({"no_global_recipe", "unsolved", "held_prerequisite"})
REMOVAL_REASONS = frozenset({"phase", "duplicate", "superseded", "ill_posed"})


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    """Recursively sanitize data structures for strict JSON serialization.

    Converts non-finite floats (NaN, +Inf, -Inf) to None (serializes to JSON null).
    Converts tuples to lists.
    Converts Path objects to strings.
    Converts tensors/numpy arrays/scalars via .tolist() or .item() when available.
    Preserves types (bool, int, finite float, str, None).
    """
    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach().cpu()
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, RuntimeError, TypeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Path):
        return str(value)
    return value


def has_nonfinite(value) -> bool:
    """Recursively check whether any float is NaN or Inf."""
    if hasattr(value, "detach") and callable(value.detach):
        try:
            value = value.detach().cpu()
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, RuntimeError, TypeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return has_nonfinite(value.tolist())
        except Exception:
            pass
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(has_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_nonfinite(v) for v in value)
    return False


def atomic_json(path: Path | str, value) -> None:
    atomic_text(path, json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_text(path: Path | str, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def file_lock(path: Path | str, *, blocking: bool = True):
    """Kernel-backed flock ownership across processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"another coordinator owns {path}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identifier(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@+-]{0,159}", value):
        raise ValueError(f"{label} must be a simple nonempty name (no directories): {value!r}")
    return value


def finite_number(value, label: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)
