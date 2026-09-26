"""Compare a hid_q baseline trajectory with 1e-7 twins. Prints a growth table."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GROUPS = ("D", "G", "prior", "ema_G", "ema_prior", "ema_critic", "latent_anchor")
PHASES = (
    ("steps_1_20", 1, 20),
    ("steps_21_80", 21, 80),
    ("full_lr_81_720", 81, 720),
    ("anneal_721_1200", 721, 1200),
)


def _load(path: Path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _series(base, twin):
    names = [name for name in GROUPS if name in base and name in twin and base[name].shape == twin[name].shape]
    out = {}
    for name in names:
        delta = twin[name].astype(np.float64) - base[name].astype(np.float64)
        out[name] = np.sqrt((delta * delta).sum(axis=1))
    joint = np.sqrt(sum(out[name] ** 2 for name in ("D", "G", "prior") if name in out))
    out["joint"] = joint
    return out


def _geo(ratios):
    ratios = np.asarray(ratios, dtype=np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return None
    return float(np.exp(np.mean(np.log(ratios))))


def summarize(base_dir: Path, twin_dir: Path) -> dict:
    base, twin = _load(base_dir / "traj.npz"), _load(twin_dir / "traj.npz")
    series = _series(base, twin)
    joint = series["joint"]
    rows = []
    for name, start, end in PHASES:
        window = joint[start:end + 1]
        prev = joint[start - 1:end]
        ratios = window / np.maximum(prev, 1e-30)
        # Linear regime: stop counting once the joint gap is already O(1e-2).
        live = prev < 1e-2
        rows.append({
            "phase": name,
            "geo_growth": _geo(ratios[live]),
            "max_growth": float(np.max(ratios)) if ratios.size else None,
            "end_l2": float(joint[min(end, len(joint) - 1)]),
        })
    shares = {}
    for name in ("D", "G", "prior", "ema_G", "ema_prior"):
        if name not in series:
            continue
        peak = int(np.argmax(series[name]))
        shares[name] = {
            "final_l2": float(series[name][-1]),
            "max_l2": float(series[name].max()),
            "max_step": int(peak),
        }
    crossed = {}
    for level in (1e-5, 1e-3, 1e-1):
        hit = np.where(joint >= level)[0]
        crossed[f"{level:.0e}"] = int(hit[0]) if hit.size else None
    probe = twin_dir / "probe" / "result.json"
    status = None
    if probe.exists():
        payload = json.loads(probe.read_text())
        status = {"status": payload.get("status"), "live": (payload.get("result") or {}).get("live")}
    return {
        "twin": twin_dir.name,
        "init_l2": float(joint[0]),
        "final_l2": float(joint[-1]),
        "crossed": crossed,
        "phases": rows,
        "groups": shares,
        "outcome": status,
    }


def main():
    root = Path("/tmp/k3p-diag/sens")
    base = root / "base"
    twins = [path for path in sorted(root.iterdir()) if path.is_dir() and path != base and (path / "traj.npz").exists()]
    report = [summarize(base, twin) for twin in twins]
    text = json.dumps(report, indent=2)
    out = Path("/tmp/k3p-diag/sens/divergence.json")
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
