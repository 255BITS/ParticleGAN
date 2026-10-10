"""A fixed FIT-standard-deviation normalization variant of the caption toy.

CUDA_VISIBLE_DEVICES=0 python -m examples.e22_routed_caption_normalization \
    --run --out runs/caption-normalization-v1
Exit0=completed scientific PASS,1=completed FAIL,2=incomplete/error.
The immutable imported helper executes the three public-API native game loops.
"""
import time
STARTED = time.monotonic()
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from examples import e22_routed_caption_untied as common

base = common.base
TASK = "routed_caption_fit_std_v1"
EPSILON = 1e-8
SECONDS = 300
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_caption_normalization_v1.json"
SOURCES = (
    "examples/e22_routed_caption_normalization.py",
    "examples/render_e22_routed_caption_normalization.py",
    "tests/test_e22_routed_caption_normalization.py",
    "examples/e22_routed_caption_accuracy.py",
    "examples/e22_routed_caption_untied.py",
    "examples/render_e22_routed_caption_untied.py",
)
THRESHOLDS = {"relative_aggregate_improvement_gte": .001, "source_harm_lte": 1e-6,
              "relative_code_benefit_gte": .001, "strict_code_benefit_each_source": True,
              "bank_router_live_fraction_gte": .9, "live_denominator": 511,
              "all_six_particle_Up_and_C_nonzero": True}
NORMALIZATION = {"statistic": "untrained FIT residual coordinate std, correction=1, flattened contexts and tokens",
                 "epsilon": EPSILON, "reject_std_lte_epsilon": True,
                 "previous_floor": .04, "shared_across_all_arms_and_D_EMA_features": True}


def budget():
    if time.monotonic() - STARTED > SECONDS:
        raise TimeoutError("normalization startup-through-final-write300s budget exceeded")


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: base.sha(root / name) for name in SOURCES}


def data_digest(data):
    return base.digest({k: asdict(v) if isinstance(v, base.Geometry) else v
                        for k, v in data.items() if k != "digest"})


def power_decomposition(value):
    """Physical or normalized token-mean + centered squared power, in F64."""
    x = value.detach().double()
    mean = x.mean(1, keepdim=True)
    power = float(x.square().mean())
    mean_power = float(mean.square().mean())
    return {"power": power, "rms": power ** .5, "token_mean_power": mean_power,
            "token_centered_power": float((x - mean).square().mean()),
            "token_mean_power_fraction": None if power == 0 else mean_power / power,
            "global_coordinate_mean_power": float(x.mean((0, 1)).square().mean())}


def normalize_data(original):
    """Change only scale and its derived digest; reject degenerate FIT units."""
    residual = original["fit_baseline"]
    if (residual.ndim != 3 or residual.shape[-1] != original["geometry"].output
        or residual.shape[0] != len(original["fit"]["source_ids"])
        or residual.shape[:2] != original["fit"]["context"].shape[:2]
        or not residual.is_floating_point() or not bool(torch.isfinite(residual).all())):
        raise ValueError("finite untrained FIT residuals with declared context/token/coordinate law required")
    if data_digest(original) != original["digest"]:
        raise ValueError("original generated data digest differs")
    raw = residual.flatten(0, 1).std(0, correction=1)
    if not bool(torch.isfinite(raw).all()) or bool((raw <= EPSILON).any()):
        raise ValueError("zero/near-zero/nonfinite FIT coordinate std rejected before science")
    old = raw.clamp_min(.04)
    if not torch.equal(original["scale"], old):
        raise ValueError("inherited fixed .04 floor law differs")
    data = {**original, "scale": raw.clamp_min(EPSILON).detach().clone()}
    data["digest"] = data_digest(data)
    fixed = {k: v for k, v in original.items() if k not in ("scale", "digest")}
    if data_digest(fixed) != data_digest({k: v for k, v in data.items() if k not in ("scale", "digest")}):
        raise AssertionError("a factor besides normalization scale changed")
    evidence = {"law": NORMALIZATION, "original_data_digest": original["digest"],
                "data_digest": data["digest"], "other_data_fields_digest": data_digest(fixed),
                "raw_fit_std": raw.detach().cpu().tolist(), "scale": data["scale"].cpu().tolist(),
                "previous_floor_scale": old.cpu().tolist(),
                "epsilon_floor_fraction": float((raw < EPSILON).float().mean()),
                "previous_floor_fraction": float((raw < .04).float().mean()),
                "fit_contexts": len(residual), "fit_tokens": residual.shape[1],
                "fit_source_ids": list(original["fit"]["source_ids"]),
                "raw_initial_fit": power_decomposition(residual),
                "normalized_initial_fit": power_decomposition(residual / data["scale"]),
                "previous_floor_normalized_initial_fit": power_decomposition(residual / old),
                "frozen_before_any_native_updates": True, "only_scale_and_derived_digest_changed": True}
    return data, evidence


def validate_card(card, sources):
    expected = {"task": TASK, "execution_helper_task": common.TASK, "arms": list(common.ARMS),
                "geometry": asdict(base.FULL), "steps": common.STEPS, "seconds": SECONDS,
                "media_steps": list(base.MEDIA_STEPS), "media_indices": list(base.MEDIA_INDICES),
                "normalization": NORMALIZATION, "accuracy_thresholds": THRESHOLDS, "sources": sources}
    if any(card.get(k) != value for k, value in expected.items()):
        raise ValueError("fixed normalization/data/geometry/gate/source protocol differs")


def fresh_output(path):
    if path.exists():
        raise ValueError("fresh exclusive output directory required")
    path.mkdir(parents=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CARD)
    args = parser.parse_args()
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    sources = source_identity(); native = base.package_identity()
    out = device = entry = report = error = card_sha = None; code = 2
    try:
        card = json.loads(args.protocol.read_text()); card_sha = base.sha(args.protocol)
        validate_card(card, sources)
        if args.out.exists(): raise ValueError("fresh exclusive output directory required")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
            raise ValueError("physicalGPU0 requires CUDA_VISIBLE_DEVICES=0")
        if base.backend_flags() != card["precision_backend"]:
            raise ValueError("declared precision backend differs")
        out = fresh_output(args.out); device = torch.device("cuda:0"); entry = base.global_rng(device)
        common.controls()
        original = base.make_data(base.FULL, device)
        data, normalization = normalize_data(original); del original
        inputs = out / "normalization-inputs.pt"
        torch.save({"fit_baseline": data["fit_baseline"].cpu(), "scale": data["scale"].cpu(),
                    "source_ids": data["fit"]["source_ids"],
                    "fit_context_digest": base.digest(data["fit"]["context"]),
                    "normalization": normalization}, inputs)
        print(json.dumps({"task": TASK, "initial_normalization": normalization}, allow_nan=False), flush=True)
        initial = common.preflight(data); budget()
        # Explicit immutable public-API helper call: no globals, native methods,
        # private namespaces or training loops are replaced/copied.
        executed = common.run(data, out); budget()
        if (source_identity() != sources or base.package_identity() != native
            or base.sha(args.protocol) != card_sha or base.backend_flags() != card["precision_backend"]):
            raise AssertionError("source/card/API/backend changed within run")
        report = {**executed, "task": TASK, "execution_helper_task": executed["task"],
                  "normalization": normalization, "normalization_input_sha256": base.sha(inputs),
                  "initial_prerequisite": initial, "source_identity": sources,
                  "protocol_sha256": card_sha, "imported_package": native,
                  "imported_package_unchanged": True, "precision_backend": base.backend_flags(),
                  "scope": "Only frozen residual normalization differs from the generated PR240 task. PASS rejects normalization alone as sufficient to remove the win in this one fixed generated fixture; FAIL identifies the failed numeric bounds, not a unique full-Supra cause. Raw TEST accuracy is offline; no output objective/guard/selection."}
        code = 0 if report["gate"]["pass"] else 1
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        import traceback; traceback.print_exc()
    finally:
        if entry is not None:
            base.set_global_rng(entry, device)
            if base.digest(base.global_rng(device)) != base.digest(entry):
                error = {"type": "AssertionError", "message": "caller CPU/CUDA RNG restoration differs"}
        torch.set_num_threads(threads)
        elapsed = time.monotonic() - STARTED
        if elapsed > SECONDS: error = {"type": "TimeoutError", "message": "startup/cleanup300s exceeded"}
        if error is not None: code = 2
        completion = {"task": TASK, "complete": error is None and report is not None,
                      "scientific_status": None if error or report is None else report["scientific_status"],
                      "error": error, "seconds": elapsed, "limit_seconds": SECONDS,
                      "source_identity": sources, "protocol_sha256": card_sha, "imported_package": native,
                      "caller_CPU_CUDA_RNG_restored": entry is not None and base.digest(base.global_rng(device)) == base.digest(entry)}
        if out is not None:
            if report is not None:
                report["seconds"] = elapsed
                (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                completion["report_sha256"] = base.sha(out / "report.json")
            path = out / "completion.json"
            path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
            if time.monotonic() - STARTED > SECONDS:
                completion.update(complete=False, error={"type": "TimeoutError", "message": "final writes300s exceeded"}, seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n"); code = 2
        print(json.dumps({"completion": completion, "scientific_gate": None if report is None else report["gate"]}, allow_nan=False), flush=True)
    return code


if __name__ == "__main__": raise SystemExit(main())
