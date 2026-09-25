"""Shared behavior from the pinned application, without formulation stamp scoring.

These checks do not train candidate GAN configurations. They are required once for
full suite completion and are displayed separately from the candidate ranking.
"""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .reference import COMMIT


def numeric(values, requirements):
    from .baseline import score_metrics
    cells = score_metrics(values, requirements)
    return {"status": "PASS" if all(c["status"] == "PASS" for c in cells) else "FAIL",
            "metrics": cells, "summary": "; ".join(f"{c['metric']}={c['value']:.6g} ({c['op']} {c['threshold']})" if c['value'] is not None else f"{c['metric']} missing" for c in cells)}


def combine(rows):
    return {"status": "PASS" if rows and all(r["status"] == "PASS" for r in rows.values()) else "FAIL",
            "rows": rows, "summary": "; ".join(f"{key}: {value['status']}" for key, value in rows.items())}


def orbit():
    from conceptmod.toys import orbit_hold as host
    _, direction, speed, radius = host._rollout(host.OrbitRecipe(), host.ClosedLoopRadialHead())
    return numeric(dict(direction_cos=direction, speed_rel=speed, radius_rel=radius),
                   [("direction_cos", ">=", host.DIRECTION_COS_MIN), ("speed_rel", "<=", host.SPEED_REL_MAX), ("radius_rel", "<=", host.RADIUS_REL_MAX)])


def erase_keep():
    from conceptmod.backends import load_backend
    from conceptmod.toys import erase_keep_backend as host
    raw = host.score_backend(load_backend("cpu", device="cpu", lora_rank=4, seed=0), "locked", name="cpu", seed=0)
    result = numeric(raw, [("teacher_leak", "<=", host.LEAK_RATIO_MAX), ("u_kept", ">=", host.U_KEPT_MIN),
                           ("pole_rel_err", "<=", host.POLE_REL_ERR_MAX), ("same_dir", "<=", host.SAME_DIR_MAX)])
    result["raw"] = raw
    return result


def late_collapse():
    from conceptmod.toys import late_collapse as host
    curve = host.synthetic_curve()
    selected = host.select_val_gate(curve)
    rows = {"validation": asdict(selected), "last": asdict(host.select_last_step(curve)),
            "best_train": asdict(host.select_best_train(curve))}
    result = numeric({"selected_val_error": selected.val_error, "selected_collapsed": int(selected.collapsed),
                      "last_val_error": rows["last"]["val_error"], "best_train_val_error": rows["best_train"]["val_error"]},
                     [("selected_val_error", "<=", host.VAL_GATE_MAX), ("selected_collapsed", "<=", 0),
                      ("last_val_error", ">=", host.VAL_GATE_MAX), ("best_train_val_error", ">=", host.VAL_GATE_MAX)])
    result["raw"] = rows
    return result


def keep_critic():
    from conceptmod.toys import keep_critic as host
    raw = host.run_keep()
    return numeric({"weight_delta": raw.weight_delta}, [("weight_delta", "<=", 0.0)])


def lm_target():
    from conceptmod.toys import lm_target as host
    # Direct training avoids the run_arm formulation-refusal wrapper.
    torch.manual_seed(host.SEED)
    student = host.NeuResidual()
    optimizer = torch.optim.Adam(student.parameters(), lr=host.LR, betas=host.BETAS)
    for _ in range(host.STEPS):
        optimizer.zero_grad(set_to_none=True)
        host.arm_loss("trajectory", student).backward()
        optimizer.step()
    raw = host.score_residual("trajectory", student)
    return numeric(raw, [("expr_gain", ">=", host.EXPR_GAIN_MIN), ("struct_hold", ">=", host.STRUCT_HOLD_MIN),
                         ("identity_mse", "<=", host.IDENTITY_MSE_MAX)])


def field_lift():
    from conceptmod.toys import field_lift as host
    rows = {}
    for name in ("plane", "tilted"):
        frame = host._frame_of(host.LiftArm(name, "lift", frame=name))
        raw = host._geometry(host.pushforward(frame), frame, host.guarded_odd(frame))
        rows[name] = numeric(raw, [("u_kept", ">=", host.U_KEPT_MIN), ("content_kept", ">=", host.CONTENT_KEPT_MIN),
                                   ("leak_ratio", "<=", host.LEAK_RATIO_MAX), ("pole_rel_err", "<=", host.POLE_REL_ERR_MAX),
                                   ("same_dir", "<=", host.SAME_DIR_MAX), ("lyric_abs", "<=", 1e-6)])
    return combine(rows)


def path_suffix():
    from conceptmod.toys import path_suffix_lora as host
    rows = {}
    for name, targets in (("suffix", host.LOCKED_SUFFIXES), ("regex", host.LOCKED_REGEX)):
        raw = asdict(host.probe(host.Arm(name, targets)))
        rows[name] = numeric(raw, [("coverage", ">=", 1.0), ("self_abs", ">=", host.GRAD_ABS_MIN),
                                   ("cross_abs", ">=", host.GRAD_ABS_MIN)])
        rows[name]["raw"] = raw
    return combine(rows)


def dsl_macro():
    from conceptmod.toys import dsl_macro_expand as host
    # Parser/expansion semantics only. Never call the GAN stamp/game scorer.
    rows = [host.run_arm(arm) for arm in host.demo_arms()]
    ok = rows[0].won and all(not row.won for row in rows[1:])
    return {"status": "PASS" if ok else "FAIL", "raw": [asdict(row) for row in rows],
            "summary": f"documented expansion {'PASS' if rows[0].won else 'FAIL'}; {sum(not r.won for r in rows[1:])}/{len(rows)-1} incorrect expansions rejected"}


def dsl_phrases():
    from conceptmod.toys import dsl_phrase_jobs as host
    rows, cache = {}, {}
    for arm in host.documented_arms():
        # Keep actual phrase/geometry checks; discard the source's adv_stamp gate.
        result = host.score_arm(arm, cache)
        raw = asdict(result)
        raw.pop("adv", None)
        raw.pop("adv_ok", None)
        raw["reasons"] = [reason for reason in result.reasons if reason != "adv_stamp"]
        raw["passed"] = result.geometric == "right" and not raw["reasons"]
        rows[arm.name] = {"status": "PASS" if raw["passed"] else "FAIL", "raw": raw}
    return combine(rows)


def dsl_geometry():
    from conceptmod import analysis_2d as host
    rows = {}
    for spec in host.method_specs():
        if spec["name"] not in ("write", "erase_esd", "erase_esd_freeze", "exaggerate"):
            continue
        raw = host.run_method(**spec)
        rows[spec["name"]] = {"status": "PASS" if raw.verdict == "right" else "FAIL",
                              "geometry": raw.geometry, "before": asdict(raw.before), "after": asdict(raw.after),
                              "summary": raw.note}
    return combine(rows)


CHECKS = {"orbit_hold": orbit, "erase_keep_backend": erase_keep, "late_collapse": late_collapse,
          "keep_critic": keep_critic, "lm_target": lm_target, "field_lift": field_lift,
          "path_suffix_lora": path_suffix, "dsl_macro_expand": dsl_macro,
          "dsl_phrase_jobs": dsl_phrases, "dsl_game_geometry": dsl_geometry}


def run_shared(reference: Path, report, save):
    root = reference.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"reference must be conceptmod {COMMIT}, got {revision}")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "conceptmod"], text=True).strip():
        raise ValueError("reference conceptmod source has local changes")
    environment = {"conceptmod_commit": revision,
                   **{key: importlib.metadata.version(key) for key in ("peft", "transformers")}}
    if report.get("shared_environment") not in (None, environment):
        raise ValueError("shared-check dependencies changed; use a separate output directory")
    report["shared_environment"] = environment
    sys.path.insert(0, str(root))
    try:
        for name, run in CHECKS.items():
            if report["shared"].get(name, {}).get("status") == "PASS":
                continue
            print(f"START shared={name}", flush=True)
            start = time.monotonic()
            try:
                result = run()
                json.dumps(result, allow_nan=False)
            except Exception as exc:
                result = {"status": "ERROR", "summary": f"{type(exc).__name__}: {exc}"}
            result["seconds"] = time.monotonic() - start
            report["shared"][name] = result
            save()
            print(f"DONE shared={name} {result['status']} {result['summary']}", flush=True)
    finally:
        sys.path.remove(str(root))
