"""Asset-free public-API caption benchmark crossing native penalty phases.

CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_late_phase --run --out runs/caption-late-phase-v1
Exit 0=terminal numerical PASS, 1=completed FAIL, 2=incomplete/error.
Only the fixed training horizon changes from the frozen scalar-moment fixture.
"""
import time
STARTED = time.monotonic()
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from examples import e22_routed_caption_frozen_stats as stats

base, common, units, flow = stats.base, stats.common, stats.units, stats.flow
TASK = "routed_caption_late_phase_v1"
STEPS, SECONDS = 2048, 900
ENDPOINTS = (512, 768, 800, 1024, 1536, 2048)
MEDIA_STEPS = (0, 256, 512, 768, 800, 1024, 1536, 2048)
ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_caption_late_phase_v1.json"
SOURCES = ("examples/e22_routed_caption_late_phase.py", "examples/render_e22_routed_caption_late_phase.py",
           "tests/test_e22_routed_caption_late_phase.py", "tests/test_e22_routed_caption_late_phase_renderer.py",
           *stats.SOURCES)
THRESHOLDS = {**units.THRESHOLDS, "live_denominator": STEPS - 1}


def budget():
    if time.monotonic() - STARTED > SECONDS:
        raise TimeoutError("late-phase startup-through-final-write900s exceeded")


def source_identity():
    return {name: base.sha(ROOT / name) for name in SOURCES}


def scalar_observation(value, path="", missing=None):
    """Read-only JSON copy; unsupported/nonfinite scalar placeholders stay null."""
    missing = [] if missing is None else missing
    if isinstance(value, dict):
        return {str(k): scalar_observation(v, f"{path}.{k}", missing) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar_observation(v, f"{path}[{i}]", missing) for i, v in enumerate(value)]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("phase telemetry must contain scalars, not tensor dumps")
        value = value.detach().item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value): return value
        missing.append(path); return None
    missing.append(path); return None


def phase_observation(policy):
    """Never invokes a penalty/diagnostic hook or advances a native clock."""
    record = getattr(policy.opt_d, "record", None)
    missing = []
    raw_stats = getattr(policy.penalty, "last_stats", None)
    if not isinstance(raw_stats, dict): missing.append(".penalty_last_stats.unavailable")
    result = scalar_observation({"observation_timing": "after_public_finish_step; sigma/LRs are post-update values",
        "availability": {"penalty_last_stats": isinstance(raw_stats, dict),
            "critic_calls": getattr(record, "calls", None) is not None,
            "critic_observed_steps": getattr(record, "observed_steps", None) is not None},
        "penalty_last_stats": raw_stats if isinstance(raw_stats, dict) else {},
        "critic_calls": getattr(record, "calls", None),
        "critic_observed_steps": getattr(record, "observed_steps", None),
        "output_sigma": policy.output_sigma(),
        "critic_group_lrs": [group["lr"] for group in policy.opt_d.param_groups]}, missing=missing)
    result["null_placeholder_paths"] = missing
    return result


def add_phase(summary, step, observation):
    phase = observation["penalty_last_stats"].get("phase")
    key = "unavailable" if phase is None else str(phase)
    summary["phase_counts"][key] = summary["phase_counts"].get(key, 0) + 1
    if phase == "blend" and summary["first_blend_step"] is None:
        summary["first_blend_step"] = step
    summary["last_observation"] = observation


def scientific_gate(ordinary, shared, candidate, zero, *, bank_updates, query_updates, C_norms, particle_Up_norms):
    """The inherited numerical criteria, with the actual 2047 live denominator."""
    keys = {str(i) for i in range(6)}
    metrics = (ordinary, shared, candidate, zero)
    if any(set(v["by_source"]) != keys for v in metrics):
        raise ValueError("all six source metrics are mandatory")
    if not all(math.isfinite(v) and v >= 0 for m in metrics for v in (m["rmse"], *m["by_source"].values())):
        raise ValueError("finite nonnegative physical accuracy required")
    if any(type(v) is not int or not 0 <= v <= STEPS - 1 for v in (bank_updates, query_updates)):
        raise ValueError("live counts must cover actual updates2..2048")
    def compare(reference):
        checks = {"aggregate_accuracy_improved": candidate["rmse"] <= reference["rmse"] * (1 - base.RELATIVE_MARGIN),
            "no_source_harmed": all(candidate["by_source"][k] <= reference["by_source"][k] + base.SOURCE_HARM for k in keys)}
        return {"pass": all(checks.values()), **checks,
            "relative_accuracy_improvement": None if reference["rmse"] == 0 else 1 - candidate["rmse"] / reference["rmse"]}
    comparisons = {"versus_ordinary": compare(ordinary), "versus_shared": compare(shared)}
    checks = {"aggregate_code_benefit": zero["rmse"] >= candidate["rmse"] * (1 + base.RELATIVE_MARGIN),
        "code_beneficial_each_source": all(zero["by_source"][k] > candidate["by_source"][k] for k in keys),
        "bank_live": bank_updates / (STEPS - 1) >= base.LIVE_FRACTION,
        "query_live": query_updates / (STEPS - 1) >= base.LIVE_FRACTION,
        "C_live_all_six_sites": set(C_norms) == set(base.SITES) and all(math.isfinite(v) and v > 0 for v in C_norms.values()),
        "particle_Up_live_all_six_sites": set(particle_Up_norms) == set(base.SITES) and all(math.isfinite(v) and v > 0 for v in particle_Up_norms.values())}
    return {"pass": all(checks.values()) and all(v["pass"] for v in comparisons.values()), **comparisons, **checks,
        "relative_code_benefit": None if candidate["rmse"] == 0 else zero["rmse"] / candidate["rmse"] - 1,
        "relative_margin": base.RELATIVE_MARGIN, "per_source_harm_tolerance": base.SOURCE_HARM,
        "live_fraction_threshold": base.LIVE_FRACTION, "live_denominator": STEPS - 1}


def controls():
    metric = lambda v: {"rmse": v, "by_source": {str(i): v for i in range(6)}}
    live = {"bank_updates": STEPS - 1, "query_updates": STEPS - 1,
            "C_norms": {s: .1 for s in base.SITES}, "particle_Up_norms": {s: .1 for s in base.SITES}}
    if not scientific_gate(metric(1), metric(1), metric(.99), metric(1), **live)["pass"]:
        raise AssertionError("positive terminal oracle rejected")
    bad = metric(.99); bad["by_source"]["5"] = 1 + 2 * base.SOURCE_HARM
    examples = ((metric(1), metric(.98), metric(.99), metric(1)),
                (metric(1), metric(1), metric(1), metric(1)),
                (metric(1), metric(1), bad, metric(1.01)),
                (metric(1), metric(1), metric(.99), metric(.99)))
    if any(scientific_gate(*e, **live)["pass"] for e in examples):
        raise AssertionError("losing-control/no-change/source-harm/useless-code control passed")
    if scientific_gate(metric(1), metric(1), metric(.99), metric(1), **{**live, "bank_updates": 511})["pass"]:
        raise AssertionError("512-era live count incorrectly passed2048")
    return {**base.scorer_controls(), "both_controls_source_code_and_2047_live_oracles": True}


def run(data, out):
    quality, curves, physical_curves, witnesses, media, events, phases, trace_hashes = {}, {}, {}, {}, {}, {}, {}, {}
    common_stream = None; live = {"bank": 0, "query": 0}; cnorm, unorm = {}, {}
    camera = data["test"]["context"][list(base.MEDIA_INDICES)]
    for arm in common.ARMS:
        loop = common.make_loop(arm, data); p = loop.policy; stream = hashlib.sha256()
        media[arm] = {"0": base.observe(loop, camera).cpu()}; curves[arm] = {}; physical_curves[arm] = {}
        phase = {"phase_counts": {}, "first_blend_step": None, "last_observation": None}
        learned_names = {n for n, v in p.G.named_parameters() if v.requires_grad}
        frozen = lambda: base.digest({n: v for n, v in p.G.state_dict().items() if n not in learned_names})
        frozen_sha = frozen(); proposal_events = accepted_rows = accepted_proposals = 0; loss_ema = None
        for step in range(1, STEPS + 1):
            row = base.update(loop); telemetry = phase_observation(p); add_phase(phase, step, telemetry)
            stream.update(json.dumps({k: row[k] for k in ("step", "batch_indices", "paired_bases", "data_rng", "paired_rng", "penalty_globals")}, sort_keys=True).encode())
            if row["move"] is not None:
                proposal_events += 1; accepted_rows += int(row["move"].get("moves", 0)); accepted_proposals += int(row["move"].get("accepted", False))
            if arm == common.UNTIED and step > 1:
                live["bank"] += int(row["bank_live"]); live["query"] += int(row["query_live"])
            with (out / f"{arm}.jsonl").open("a") as handle:
                handle.write(json.dumps({**row, "phase_observation": telemetry}, allow_nan=False) + "\n")
            loss_ema = row["loss_g"] if loss_ema is None else .98 * loss_ema + .02 * row["loss_g"]
            if step % 64 == 0 or step in ENDPOINTS:
                print(json.dumps({"arm": arm, "step": step, "steps": STEPS, "native_G_loss": row["loss_g"],
                    "native_G_loss_EMA": loss_ema, "native_D_loss": row["loss_d_game"], "phase": telemetry,
                    "seconds": time.monotonic() - STARTED}, allow_nan=False), flush=True)
            if step in ENDPOINTS:
                prediction = base.observe(loop, data["test"]["context"])
                physical_curves[arm][str(step)] = prediction.cpu()
                curves[arm][str(step)] = base.accuracy(prediction, data["test"]["source_ids"])
                print(json.dumps({"arm": arm, "offline_TEST_step": step, "accuracy": curves[arm][str(step)]}, allow_nan=False), flush=True)
            if step in MEDIA_STEPS: media[arm][str(step)] = base.observe(loop, camera).cpu()
            budget()
        base.learned_finite(p)
        if frozen_sha != frozen(): raise AssertionError("frozen host/teacher/caption owners changed")
        if common_stream is None: common_stream = stream.hexdigest()
        elif stream.hexdigest() != common_stream: raise AssertionError("external data/Gaussian/native-penalty streams differ")
        state = base.checkpoint(loop); base.restore(loop, state)
        repeat = base.observe(loop, data["test"]["context"]).cpu()
        if base.digest(base.checkpoint(loop)) != base.digest(state) or not torch.equal(repeat, physical_curves[arm][str(STEPS)]):
            raise AssertionError("terminal public restore/clean repeat differs")
        quality[arm] = curves[arm][str(STEPS)]
        witnesses[arm] = {"steps": p.completed_steps, "recipe": p.recipe.to_dict(),
            "native_state_digest": base.digest(state["native"]), "public_terminal_restore_output_exact": True,
            "learned_weights_gradients_moments_finite": True, "frozen_student_teacher_values_unchanged": True}
        events[arm] = {"proposal_events_including_skips": proposal_events, "accepted_row_moves": accepted_rows, "accepted_proposals": accepted_proposals}
        phase["actual_blend_observed"] = phase["phase_counts"].get("blend", 0) > 0
        phases[arm] = phase
        trace_hashes[arm] = base.sha(out / f"{arm}.jsonl")
        if arm == common.UNTIED:
            zero = base.observe(loop, data["test"]["context"], zero_code=True).cpu()
            quality["zero_code"] = base.accuracy(zero, data["test"]["source_ids"])
            cnorm = {b.site: float(b.bridge.weight[:, data["geometry"].rank:].detach().norm()) for b in p.G.branches()}
            unorm = {b.site: float(b.particle_up.weight.detach().norm()) for b in p.G.branches()}
        del loop, p, state
        budget()
    gate = scientific_gate(*(quality[arm] for arm in common.ARMS), quality["zero_code"],
        bank_updates=live["bank"], query_updates=live["query"], C_norms=cnorm, particle_Up_norms=unorm)
    endpoint = out / "endpoint-residuals.pt"; curve_file = out / "curve-residuals.pt"; frame_file = out / "observed-media.pt"
    binding = {"source_ids": data["test"]["source_ids"], "test_context_digest": base.digest(data["test"]["context"]), "target_digest": base.digest(data["test"]["targets"])}
    torch.save({"physical_residuals": {**{arm: physical_curves[arm][str(STEPS)] for arm in common.ARMS}, "zero_code": zero}, **binding}, endpoint)
    torch.save({"physical_residuals": physical_curves, "steps": ENDPOINTS, **binding}, curve_file)
    torch.save({"actual_residuals": media, "target_residual": torch.zeros_like(media[common.ARMS[0]]["0"]),
        "steps": MEDIA_STEPS, "indices": base.MEDIA_INDICES, "source_ids": [data["test"]["source_ids"][i] for i in base.MEDIA_INDICES],
        "context_digest": base.digest(camera), "capture_native_state_rng_diagnostics_unchanged": True}, frame_file)
    budget()
    return {"task": TASK, "complete": True, "scientific_status": "PASS" if gate["pass"] else "FAIL", "gate": gate,
        "accuracy": quality, "curves": curves, "arms": witnesses, "phase_summary": phases,
        "phase_coverage_observed_all_arms": all(p["actual_blend_observed"] for p in phases.values()),
        "phase_coverage_scope": "Read-only actual execution evidence, not a historical-phase predicate in the performance gate. Future API schedules may differ.",
        "quality_updates": 3 * STEPS, "replay_updates": 0, "live": {**live, "denominator": STEPS - 1},
        "C_norms": cnorm, "particle_Up_norms": unorm, "population_events": events, "trace_sha256": trace_hashes,
        "matched_external_data_Gaussian_native_penalty_streams": True, "matched_stream_digest": common_stream,
        "data_digest": data["digest"], "counts": common.counts(data["geometry"]), "geometry": asdict(data["geometry"]),
        "media_steps": list(MEDIA_STEPS), "endpoint_steps": list(ENDPOINTS),
        "endpoint_residual_sha256": base.sha(endpoint), "curve_residual_sha256": base.sha(curve_file), "observed_media_sha256": base.sha(frame_file)}


def validate_card(card, sources):
    expected = {"task": TASK, "arms": list(common.ARMS), "geometry": asdict(base.FULL), "steps": STEPS,
        "seconds": SECONDS, "endpoint_steps": list(ENDPOINTS), "media_steps": list(MEDIA_STEPS),
        "media_indices": list(base.MEDIA_INDICES), "accuracy_thresholds": THRESHOLDS, "sources": sources,
        "flow": flow.LAW, "normalization": units.NORMALIZATION, "frozen_calibration": stats.RULE,
        "profile_sha256": base.sha(stats.PROFILE)}
    if any(card.get(k) != v for k, v in expected.items()): raise ValueError("fixed late-phase protocol/source/data/gate differs")
    stats.read_profile()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--protocol", type=Path, default=CARD); args = parser.parse_args()
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    sources = source_identity(); native = base.package_identity()
    out = device = entry = report = error = card_sha = None; code = 2
    try:
        card = json.loads(args.protocol.read_text()); card_sha = base.sha(args.protocol); validate_card(card, sources)
        if args.out.exists(): raise ValueError("fresh exclusive output required")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available(): raise ValueError("physicalGPU0 requires CUDA_VISIBLE_DEVICES=0")
        if base.backend_flags() != card["precision_backend"]: raise ValueError("declared precision backend differs")
        out = units.fresh_output(args.out); device = torch.device("cuda:0"); entry = base.global_rng(device)
        oracles = controls(); data, evidence = stats.make_data(base.FULL, device); budget()
        inputs = out / "late-phase-inputs.pt"
        torch.save({"flow": data["flow"], "contexts": {k: data[k]["context"].cpu() for k, _, _ in flow.POOLS},
            "frozen": data["frozen"], "fit_baseline": data["fit_baseline"].cpu(), "scale": data["scale"].cpu(), "evidence": evidence}, inputs)
        print(json.dumps({"task": TASK, "initial_frozen_stats": evidence}, allow_nan=False), flush=True)
        initial = common.preflight(data); budget(); report = run(data, out); budget()
        if source_identity() != sources or base.package_identity() != native or base.sha(args.protocol) != card_sha or base.backend_flags() != card["precision_backend"]:
            raise AssertionError("source/profile/card/API/backend changed within run")
        report.update(execution_helper_task=common.TASK, data_helper_task=stats.TASK, frozen_stats=evidence,
            profile_sha256=base.sha(stats.PROFILE), late_phase_input_sha256=base.sha(inputs), initial_prerequisite=initial,
            scorer_oracles_and_destructive_controls=oracles, source_identity=sources, protocol_sha256=card_sha,
            imported_package=native, imported_package_unchanged=True, precision_backend=base.backend_flags(),
            scope="Only fixed horizon changes to2048. Actual penalty phase telemetry is descriptive; terminal raw accuracy alone sets the numerical gate. No actual-caption/full-Supra qualification or unique phase-causality claim.")
        code = 0 if report["gate"]["pass"] else 1
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        import traceback; traceback.print_exc()
    finally:
        if entry is not None:
            base.set_global_rng(entry, device)
            if base.digest(base.global_rng(device)) != base.digest(entry): error = {"type": "AssertionError", "message": "caller RNG restoration differs"}
        torch.set_num_threads(threads); elapsed = time.monotonic() - STARTED
        if elapsed > SECONDS: error = {"type": "TimeoutError", "message": "startup/cleanup900s exceeded"}
        if error is not None: code = 2
        receipt = {"task": TASK, "complete": error is None and report is not None,
            "scientific_status": None if error or report is None else report["scientific_status"], "error": error,
            "seconds": elapsed, "limit_seconds": SECONDS, "source_identity": sources, "protocol_sha256": card_sha,
            "imported_package": native, "caller_CPU_CUDA_RNG_restored": entry is not None and base.digest(base.global_rng(device)) == base.digest(entry)}
        if out is not None:
            if report is not None:
                report["seconds"] = elapsed; (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                receipt["report_sha256"] = base.sha(out / "report.json")
            path = out / "completion.json"; path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
            if time.monotonic() - STARTED > SECONDS:
                receipt.update(complete=False, error={"type": "TimeoutError", "message": "final writes900s exceeded"}, seconds=time.monotonic() - STARTED)
                path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n"); code = 2
        print(json.dumps({"completion": receipt, "scientific_gate": None if report is None else report["gate"]}, allow_nan=False), flush=True)
    return code


if __name__ == "__main__": raise SystemExit(main())
