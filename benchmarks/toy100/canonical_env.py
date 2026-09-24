"""Canonical toy100 continuous-learning environment.

Source ``scripts/toy100_env.sh`` before a scoring process. The shell sets the
ISA pins, one thread, and ``MKL_CBWR``. This module refuses to score unless
that environment is active, then pins PyTorch threads, deterministic
algorithms, and seed 0.

``MKL_CBWR=AVX2,STRICT`` is requested first. This oneMKL build keeps that
mode on Intel dispatch and replaces it on the non-Intel path, so the shell
falls back to ``COMPATIBLE``, which both paths keep. Public ``mkl_cbwr_get``
is not exported by the PyTorch 2.14 CPU wheel
(MKL is linked into ``libtorch_cpu.so``). ``mkl_serv_cbwr_get(-1)`` returns
the same ``mkl_cbwr`` word: low 16 bits are the branch, bit 0x10000 is STRICT.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import sys

# Branch ids confirmed against this wheel's mkl_serv_cbwr_get, not guessed
# from an older header. STRICT is bit 16 of the full word (disassembly of
# get_mkl_cbwr_from_env ORs 0x10000 when the env value ends in ",STRICT").
STRICT_BIT = 0x10000
BRANCH_NAMES = {
    1: "AUTO",
    3: "COMPATIBLE",
    8: "SSE4_2",
    9: "AVX",
    10: "AVX2",
}
AVX2_STRICT = 10 | STRICT_BIT
COMPATIBLE = 3
REQUESTED_MODES = ("AVX2,STRICT", "COMPATIBLE")

PINS = {
    "TOY100_CANONICAL_ENV": "1",
    "TOY100_SEED": "0",
    "ATEN_CPU_CAPABILITY": "avx2",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "DNNL_MAX_CPU_ISA": "AVX2",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
}

_APPLIED = False
_RECEIPT = None


def decode_cbwr(raw: int | None) -> str | None:
    if raw is None:
        return None
    branch = BRANCH_NAMES.get(raw & 0xFFFF, f"raw:{raw & 0xFFFF}")
    if raw & STRICT_BIT:
        return f"{branch},STRICT"
    return branch


def expected_cbwr_raw(mode: str) -> int | None:
    if mode == "AVX2,STRICT":
        return AVX2_STRICT
    if mode == "COMPATIBLE":
        return COMPATIBLE
    return None


def _libtorch():
    import torch
    path = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
    return ctypes.CDLL(str(path))


def read_cbwr_raw() -> int | None:
    """Effective CNR word, or None if the getter cannot be called."""
    try:
        lib = _libtorch()
        getter = lib.mkl_serv_cbwr_get
        getter.argtypes = [ctypes.c_int]
        getter.restype = ctypes.c_int
        return int(getter(-1))
    except (AttributeError, OSError):
        return None


def _mkl_version(lib) -> str | None:
    try:
        fn = lib.MKL_Get_Version_String
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
        buf = ctypes.create_string_buffer(256)
        fn(buf, 256)
        text = buf.value.decode(errors="replace").strip()
        return text or None
    except (AttributeError, OSError):
        return None


def _cpu() -> dict:
    vendor = model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("vendor_id") and vendor is None:
                vendor = line.split(":", 1)[1].strip()
            elif line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            if vendor and model:
                break
    except OSError:
        pass
    return {"vendor": vendor, "model": model, "machine": platform.machine()}


def _thread_counts(lib) -> dict:
    import torch
    mkl_threads = None
    omp_threads = None
    try:
        lib.mkl_get_max_threads.restype = ctypes.c_int
        mkl_threads = int(lib.mkl_get_max_threads())
    except (AttributeError, OSError):
        pass
    try:
        gomp = ctypes.CDLL(str(Path(torch.__file__).resolve().parent / "lib" / "libgomp.so.1"))
        gomp.omp_get_max_threads.restype = ctypes.c_int
        omp_threads = int(gomp.omp_get_max_threads())
    except (AttributeError, OSError):
        pass
    return {
        "torch": int(torch.get_num_threads()),
        "interop": int(torch.get_num_interop_threads()),
        "omp": omp_threads,
        "omp_env": os.environ.get("OMP_NUM_THREADS"),
        "mkl": mkl_threads,
        "mkl_env": os.environ.get("MKL_NUM_THREADS"),
    }


def env_problems() -> list[str]:
    problems = []
    if os.environ.get("TOY100_CANONICAL_ENV") != "1":
        problems.append("TOY100_CANONICAL_ENV is not 1 (source scripts/toy100_env.sh)")
    for key, expected in PINS.items():
        if key == "TOY100_CANONICAL_ENV":
            continue
        if os.environ.get(key) != expected:
            problems.append(f"{key}={os.environ.get(key)!r}, need {expected!r}")
    mode = os.environ.get("MKL_CBWR")
    selected = os.environ.get("TOY100_CBWR_MODE")
    if mode not in REQUESTED_MODES:
        problems.append(f"MKL_CBWR={mode!r}, need one of {REQUESTED_MODES}")
    elif selected != mode:
        problems.append(f"TOY100_CBWR_MODE={selected!r} does not match MKL_CBWR={mode!r}")
    if not os.environ.get("TOY100_CBWR_REASON"):
        problems.append("TOY100_CBWR_REASON is empty")
    return problems


def _refuse(problems: list[str]) -> None:
    message = "REFUSING TO SCORE: canonical toy100 env is not active. " + "; ".join(problems)
    message += ". Source scripts/toy100_env.sh in this process before the gate. No score was written."
    print(message, file=sys.stderr)
    raise SystemExit(2)


def apply_runtime() -> None:
    """Pin threads, deterministic algorithms, and seed 0. Call once, before work."""
    global _APPLIED
    if _APPLIED:
        return
    import random
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The inter-op pool starts on first use. A second call is ignored.
        pass
    torch.use_deterministic_algorithms(True)
    seed = int(os.environ.get("TOY100_SEED", "0"))
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    _APPLIED = True


def collect_receipt() -> dict:
    import torch
    lib = _libtorch()
    raw = read_cbwr_raw()
    version = _mkl_version(lib)
    if version is None:
        version = "unavailable; MKL_CBWR=" + str(os.environ.get("MKL_CBWR"))
    show = torch.__config__.show().replace("\n", " ")
    found = {}
    for part in show.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().split()[-1]
        if key in {"BLAS_INFO", "USE_MKL", "TORCH_VERSION", "COMMIT_SHA"}:
            found[key] = value.strip().split()[0]
    build = " ".join(f"{key}={found[key]}" for key in
                     ("TORCH_VERSION", "COMMIT_SHA", "BLAS_INFO", "USE_MKL") if key in found)
    return {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_build": build,
        "cpu": _cpu(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkl_version": version,
        "cbwr_env": os.environ.get("MKL_CBWR"),
        "cbwr_effective": decode_cbwr(raw),
        "cbwr_raw": raw,
        "cbwr_getter": "mkl_serv_cbwr_get(-1)",
        "cbwr_reason": os.environ.get("TOY100_CBWR_REASON"),
        "isa": {
            "ATEN_CPU_CAPABILITY": os.environ.get("ATEN_CPU_CAPABILITY"),
            "MKL_ENABLE_INSTRUCTIONS": os.environ.get("MKL_ENABLE_INSTRUCTIONS"),
            "ONEDNN_MAX_CPU_ISA": os.environ.get("ONEDNN_MAX_CPU_ISA"),
            "DNNL_MAX_CPU_ISA": os.environ.get("DNNL_MAX_CPU_ISA"),
        },
        "threads": _thread_counts(lib),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "seed": int(os.environ.get("TOY100_SEED", "0")),
    }


def harness_receipt() -> dict:
    if _RECEIPT is None:
        raise RuntimeError("canonical env was not activated; refuse scoring instead of continuing")
    return _RECEIPT


def require_canonical_env() -> dict:
    """Exit 2 unless the shell entry point is active and MKL kept the selected mode."""
    global _RECEIPT
    problems = env_problems()
    if problems:
        _refuse(problems)
    import torch  # noqa: F401  (loads MKL under the pins)
    raw = read_cbwr_raw()
    expected = expected_cbwr_raw(os.environ["MKL_CBWR"])
    if raw != expected:
        _refuse([
            f"MKL_CBWR={os.environ['MKL_CBWR']} but effective mode is "
            f"{decode_cbwr(raw)} raw {raw}; the requested mode did not stick"
        ])
    apply_runtime()
    _RECEIPT = collect_receipt()
    return _RECEIPT


def select_cbwr() -> tuple[str, str]:
    """Choose the CBWR mode both vendor paths can keep.

    The process must be started with ``MKL_CBWR=AVX2,STRICT``. AVX2|STRICT
    sticks on Intel dispatch in oneMKL 2024.2 and does not stick when
    ``mkl_serv_intel_cpu_true`` is false (effective raw 65538, branch 2 with
    the STRICT bit). Scoring that mode would make Intel and AMD runs diverge,
    so the harness falls back to COMPATIBLE, which both paths keep as raw 3.
    """
    import torch  # noqa: F401
    raw = read_cbwr_raw()
    if raw == AVX2_STRICT:
        reason = (
            "AVX2,STRICT is effective on this CPU "
            f"(mkl_serv_cbwr_get(-1)={raw}) but oneMKL does not keep it on the "
            "non-Intel dispatch path (effective raw 65538). "
            "Fell back to COMPATIBLE so both vendors use one mode."
        )
    else:
        reason = (
            "AVX2,STRICT unsupported on this CPU: "
            f"mkl_serv_cbwr_get(-1)={raw} ({decode_cbwr(raw)}), not {AVX2_STRICT}. "
            "Fell back to COMPATIBLE."
        )
    return "COMPATIBLE", reason


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--select-cbwr"]:
        mode, reason = select_cbwr()
        print(mode)
        print(reason)
        return 0
    if args == ["--check"]:
        receipt = require_canonical_env()
        import json
        print(json.dumps(receipt, sort_keys=True))
        return 0
    print("usage: python -m benchmarks.toy100.canonical_env --select-cbwr|--check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
