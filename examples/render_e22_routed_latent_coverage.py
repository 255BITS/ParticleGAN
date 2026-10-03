"""Verify and render saved actual latent-coverage training observations on CPU.

CUDA_VISIBLE_DEVICES='' python -m examples.render_e22_routed_latent_coverage \
    --observed RUN/observed-training.pt --producer-report RUN/report.json \
    --producer-completion RUN/completion.json --out NEW_MEDIA_DIRECTORY

Optional dependencies: Torch, NumPy and Pillow >= 10.1. No model is constructed,
no ParticleGAN API is called, and no training is executed. The saved-array
verification authenticates TEST metrics; the producer's complete scientific
verdict is retained separately, rather than inferred from an attractive GIF.
"""

from __future__ import annotations

import time

STARTED = time.monotonic()

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


TASK = "routed_latent_coverage_v2"
ARMS = ("two_seed", "eight_seed")
MEDIA_STEPS = (0, 128, 256, 512, 800, 1024)
QUALITY_STEPS = (896, 928, 960, 992, 1024)
MEDIA_INDICES = (0, 40, 80, 120, 160, 200)
TEST_SEEDS = (39001, 39002)
FIT_SEEDS = (7063, 7064)
JUDGES = ("two_seed_final", "eight_seed_final")
SITES = ("input", "edit")
THRESHOLDS = {"baseline_test_fit_mse_ratio": 1.10, "relative_rmse_gain": .001,
              "source_harm": 1e-6, "seed_harm": 1e-6, "gap_fraction": .9,
              "common_game_gain": 1e-6, "code_gain": 1e-6, "live_fraction": .9}
LIMIT_SECONDS = 60.0
IDENTITY_ATOL = 1e-12
CHECKS = 0


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def budget(limit: float) -> None:
    if time.monotonic() - STARTED > limit:
        raise TimeoutError("complete CPU verification/media budget exceeded")


def require(value: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not value:
        raise ValueError(message)


def close(observed: object, expected: float, where: str) -> None:
    require(isinstance(observed, (int, float)) and not isinstance(observed, bool),
            f"numeric metric required: {where}")
    value = float(observed)
    require(math.isfinite(value) and abs(value - expected) <= IDENTITY_ATOL * max(1.0, abs(expected)),
            f"saved-array metric differs: {where}")


def ordered_mean(values: list[float]) -> float:
    require(bool(values), "nonempty fixed observation group required")
    return math.fsum(values) / len(values)


def verify_provenance(protocol: Path, source_root: Path, report: dict, completion: dict,
                      limit: float) -> dict:
    """Rehash producer/card/native files without importing any executed source."""
    require(protocol.is_file(), "actual frozen protocol file required")
    protocol_sha = sha256(protocol)
    require(protocol_sha == report.get("protocol_sha256") == completion.get("protocol_sha256"),
            "actual frozen card hash differs from producer/completion")
    card = json.loads(protocol.read_text())
    require(card.get("status") == "FROZEN_PRE_EXECUTION" and card.get("approved_for_execution") is True,
            "frozen root-reviewed protocol required")
    source = report.get("source_identity")
    require(isinstance(source, dict) and bool(source) and source == card.get("sources") ==
            completion.get("source_identity"), "producer/card/completion source map differs")
    root = source_root.resolve()
    actual_source = {}
    for name, expected in source.items():
        require(isinstance(name, str) and isinstance(expected, str) and len(expected) == 64,
                "named source hash entries required")
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), "source must remain within declared checkout")
        actual_source[name] = sha256(path)
        require(actual_source[name] == expected, "actual producer/renderer/fixture source hash differs")
        budget(limit)
    require("examples/e22_routed_latent_coverage.py" in source and
            "examples/render_e22_routed_latent_coverage.py" in source and
            "examples/e22_routed_caption_accuracy.py" in source,
            "complete declared producer/renderer/public-loop source closure required")
    native = report.get("native_package")
    require(isinstance(native, dict) and native == completion.get("native_package") and
            isinstance(native.get("imported_package"), str) and
            card.get("native_package") == {"python_sha256": native.get("python_sha256")},
            "native byte identity or producer/completion import provenance differs")
    declared_native_files = card.get("native_python_files")
    require(isinstance(declared_native_files, dict) and bool(declared_native_files) and
            declared_native_files == report.get("native_python_files") == completion.get("native_python_files"),
            "complete relative native source maps differ")
    native_root = root / "particlegan"
    require(native_root.is_dir(), "native sources under declared source-root required")
    digest = hashlib.sha256()
    native_files = {}
    for path in sorted(native_root.rglob("*.py")):
        name = str(Path("particlegan") / path.relative_to(native_root))
        digest.update(name.encode())
        per_file = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                per_file.update(block)
        native_files[name] = per_file.hexdigest()
        budget(limit)
    require(bool(native_files) and digest.hexdigest() == native.get("python_sha256"),
            "actual native sorted-relative-path/raw-bytes source digest differs")
    require(native_files == declared_native_files, "actual relative native Python file identities differ")
    law = card["law"]
    require(law.get("task") == TASK and tuple(law.get("arms", ())) == ARMS and law.get("steps") == 1024 and law.get("seconds") == 300 and
            tuple(law.get("score_steps", ())) == QUALITY_STEPS and tuple(law.get("media_steps", ())) == MEDIA_STEPS and
            tuple(law.get("media_indices", ())) == MEDIA_INDICES and law.get("thresholds") == THRESHOLDS,
            "actual card task/budget/endpoint/scientific-threshold law differs")
    return {"protocol_path": str(protocol.resolve()), "protocol_sha256": protocol_sha,
            "source_root": str(root), "source_identity": actual_source,
            "native_package": native, "native_python_files": native_files,
            "native_digest_law": "SHA256 of sorted UTF8 particlegan-relative paths followed by raw file bytes"}


def integer_list(value: object, where: str, torch) -> list[int]:
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and value.ndim == 1,
                f"CPU vector required: {where}")
        require(value.dtype in (torch.int32, torch.int64), f"integer vector required: {where}")
        value = value.tolist()
    require(isinstance(value, (list, tuple)), f"sequence required: {where}")
    require(all(isinstance(v, int) and not isinstance(v, bool) for v in value),
            f"integer values required: {where}")
    return list(value)


def finite_residual(value: object, shape: tuple[int, ...], where: str, torch) -> None:
    require(isinstance(value, torch.Tensor), f"tensor required: {where}")
    require(value.device.type == "cpu" and value.dtype == torch.float32,
            f"CPU float32 physical residual required: {where}")
    require(tuple(value.shape) == shape and bool(torch.isfinite(value).all()),
            f"finite declared residual shape required: {where}")


def metrics(residual, sources: list[int], seeds: list[int], seed_values=TEST_SEEDS) -> dict:
    # F64 within each physical context, then ordered Python averaging across it.
    per_context = residual.double().square().flatten(1).mean(1).tolist()
    mse = ordered_mean(per_context)
    return {
        "count": len(per_context), "mse": mse, "rmse": math.sqrt(mse),
        "by_source": {str(s): math.sqrt(ordered_mean([v for i, v in enumerate(per_context)
                                                      if sources[i] == s])) for s in range(6)},
        "by_seed": {str(s): math.sqrt(ordered_mean([v for i, v in enumerate(per_context)
                                                    if seeds[i] == s])) for s in seed_values},
    }


def verify_metrics(saved: dict, actual: dict, where: str) -> int:
    require(isinstance(saved, dict) and saved.get("count") == actual["count"],
            f"saved fixed context count differs: {where}")
    checks = 1
    for key in ("mse", "rmse"):
        close(saved[key], actual[key], f"{where}.{key}")
        checks += 1
    for group in ("by_source", "by_seed"):
        require(isinstance(saved.get(group), dict) and set(saved[group]) == set(actual[group]),
                f"complete fixed metric grouping required: {where}.{group}")
        checks += 1
        for key, expected in actual[group].items():
            value = saved[group][key]
            # Reports can attach subgroup counts without changing the RMS law.
            close(value.get("rmse") if isinstance(value, dict) else value, expected,
                  f"{where}.{group}.{key}")
            checks += 1
    return checks


def verify_saved(data: dict, report: dict, torch, limit: float) -> dict:
    checks_before = CHECKS
    require(isinstance(data, dict) and data.get("task") == TASK, "raw task differs")
    require(tuple(data.get("arms", ())) == ARMS, "raw arm order differs")
    require(tuple(data.get("media_steps", ())) == MEDIA_STEPS, "actual media steps differ")
    require(tuple(data.get("media_indices", ())) == MEDIA_INDICES, "fixed camera indices differ")
    require(set(data.get("media", {})) == set(ARMS), "exact two media arms required")
    require(set(data.get("quality", {})) == set(ARMS), "exact two quality arms required")
    require(set(data.get("endpoint_residuals", {})) == set(ARMS),
            "actual full TEST residuals required for metric verification")
    require(set(data.get("fit_endpoint_residuals", {})) == set(ARMS) and
            set(data.get("fit_quality", {})) == set(ARMS),
            "actual common original-FIT residuals and metrics required")
    require(data.get("protocol_sha256") == report.get("protocol_sha256") and
            isinstance(data.get("protocol_sha256"), str) and len(data["protocol_sha256"]) == 64,
            "raw/producer protocol identity differs or is absent")
    require(isinstance(data.get("source_identity"), dict) and bool(data["source_identity"]) and
            data["source_identity"] == report.get("source_identity"),
            "raw/producer source identity differs or is absent")
    require(isinstance(data.get("evidence"), dict) and bool(data["evidence"]) and
            data["evidence"] == report.get("data_evidence"),
            "raw/producer generated input lineage differs or is absent")

    sources = integer_list(data["test_source_ids"], "test_source_ids", torch)
    seeds = integer_list(data["test_seeds"], "test_seeds", torch)
    require(len(sources) == len(seeds) == 240, "full TEST240 metadata required")
    require(set(sources) == set(range(6)) and all(sources.count(s) == 40 for s in range(6)),
            "six equally weighted TEST sources required")
    require(set(seeds) == set(TEST_SEEDS) and all(seeds.count(s) == 120 for s in TEST_SEEDS),
            "two equally weighted held-out TEST seeds required")
    for source in range(6):
        require(all(sum(s == source and k == seed for s, k in zip(sources, seeds)) == 20
                    for seed in TEST_SEEDS), "source/TEST seed stratification differs")
    require([sources[i] for i in MEDIA_INDICES] == list(range(6)) and
            [seeds[i] for i in MEDIA_INDICES] == [39001] * 6,
            "camera must show one fixed held-out seed across the six sources")

    paths = data.get("test_paths")
    if isinstance(paths, torch.Tensor):
        paths = integer_list(paths, "test_paths", torch)
    require(isinstance(paths, (list, tuple)) and len(paths) == 240,
            "full TEST path metadata required")
    require(set(paths) in ({"neutral", "positive"}, {0, 1}), "declared two TEST caption paths required")
    neutral = "neutral" if isinstance(paths[0], str) else 0
    require(all(paths[i] == neutral for i in MEDIA_INDICES), "fixed neutral camera path required")
    times = data.get("test_times")
    if isinstance(times, torch.Tensor):
        require(times.device.type == "cpu" and times.ndim == 1, "CPU time metadata required")
        times = times.tolist()
    require(isinstance(times, (list, tuple)) and len(times) == 240, "full TEST time metadata required")
    time_indices = []
    for t in times:
        require(isinstance(t, (float, int)) and math.isfinite(float(t)), "finite TEST times required")
        index = round(float(t) * 10)
        require(0 <= index <= 9 and abs(float(t) - index / 10) <= 1e-7,
                "ten declared recorded Euler times required")
        time_indices.append(index)
    require(all(time_indices[i] == 0 for i in MEDIA_INDICES), "fixed first-time camera required")
    require(len(set(zip(sources, seeds, paths, time_indices))) == 240,
            "TEST must contain each source/seed/path/time exactly once")

    fit_sources = integer_list(data["fit_source_ids"], "fit_source_ids", torch)
    fit_seeds = integer_list(data["fit_seeds"], "fit_seeds", torch)
    require(len(fit_sources) == len(fit_seeds) == 240, "common original FIT240 metadata required")
    require(set(fit_sources) == set(range(6)) and set(fit_seeds) == set(FIT_SEEDS),
            "original FIT source/seed law differs")
    require(all(sum(s == source and k == seed for s, k in zip(fit_sources, fit_seeds)) == 20
                for source in range(6) for seed in FIT_SEEDS), "original FIT stratification differs")
    qualified_quality = {}
    qualified_fit_quality = {}
    for arm in ARMS:
        media = data["media"][arm]
        endpoint = data["endpoint_residuals"][arm]
        quality = data["quality"][arm]
        require(set(media) == {str(s) for s in MEDIA_STEPS}, "all actual media steps required")
        require(set(endpoint) == set(quality) == {str(s) for s in QUALITY_STEPS},
                "all five actual score endpoints required")
        require(set(report["quality"][arm]) == {str(s) for s in QUALITY_STEPS},
                "producer endpoint quality keys differ")
        qualified_quality[arm] = {}
        qualified_fit_quality[arm] = {}
        require(set(data["fit_endpoint_residuals"][arm]) == set(data["fit_quality"][arm]) ==
                set(report["fit_quality"][arm]) == {str(s) for s in QUALITY_STEPS},
                "all five common original-FIT score endpoints required")
        for step in MEDIA_STEPS:
            finite_residual(media[str(step)], (6, 16, 16), f"media.{arm}.{step}", torch)
        for step in QUALITY_STEPS:
            key = str(step)
            finite_residual(endpoint[key], (240, 16, 16), f"endpoint.{arm}.{step}", torch)
            actual = metrics(endpoint[key], sources, seeds)
            verify_metrics(quality[key], actual, f"raw.{arm}.{step}")
            verify_metrics(report["quality"][arm][key], actual, f"report.{arm}.{step}")
            qualified_quality[arm][key] = actual
            fit_value = data["fit_endpoint_residuals"][arm][key]
            finite_residual(fit_value, (240, 16, 16), f"FIT.{arm}.{step}", torch)
            actual_fit = metrics(fit_value, fit_sources, fit_seeds, FIT_SEEDS)
            verify_metrics(data["fit_quality"][arm][key], actual_fit, f"raw.FIT.{arm}.{step}")
            verify_metrics(report["fit_quality"][arm][key], actual_fit, f"report.FIT.{arm}.{step}")
            qualified_fit_quality[arm][key] = actual_fit
            budget(limit)
        # The final camera is a literal subset of the actual final TEST tensor.
        require(torch.equal(media["1024"], endpoint["1024"][list(MEDIA_INDICES)]),
                "final camera differs from actual scored TEST residuals")
        if "media_metrics" in data:
            require(set(data["media_metrics"][arm]) == {str(s) for s in MEDIA_STEPS},
                    "every declared camera metric required")
            for step in MEDIA_STEPS:
                value = media[str(step)]
                actual_mse = ordered_mean(value.double().square().mean(dim=(1, 2)).tolist())
                saved = data["media_metrics"][arm][str(step)]
                require(saved.get("count") == 6, "camera metric count must remain six")
                close(saved["mse"], actual_mse, f"camera.{arm}.{step}.mse")
                close(saved["rmse"], math.sqrt(actual_mse), f"camera.{arm}.{step}.rmse")
    require(torch.equal(data["media"][ARMS[0]]["0"], data["media"][ARMS[1]]["0"]),
            "actual common initial camera predictions differ")
    if "target_residual" in data:
        finite_residual(data["target_residual"], (6, 16, 16), "target_residual", torch)
        require(int(data["target_residual"].count_nonzero()) == 0, "paired residual target must be zero")

    endpoint_accuracy_gates = {}
    for step in QUALITY_STEPS:
        baseline = qualified_quality[ARMS[0]][str(step)]
        candidate = qualified_quality[ARMS[1]][str(step)]
        endpoint_accuracy_gates[str(step)] = {
            "relative_TEST_RMSE_lte_0_999": candidate["rmse"] <= .999 * baseline["rmse"],
            "source_harm_lte_1e_minus6": all(candidate["by_source"][str(s)] <=
                                             baseline["by_source"][str(s)] + 1e-6 for s in range(6)),
            "seed_harm_lte_1e_minus6": all(candidate["by_seed"][str(s)] <=
                                           baseline["by_seed"][str(s)] + 1e-6 for s in TEST_SEEDS),
        }
    zero = data["zero_code_residual"]
    finite_residual(zero, (240, 16, 16), "expanded final zero-code residual", torch)
    zero_quality = metrics(zero, sources, seeds)
    verify_metrics(data["zero_code_quality"], zero_quality, "raw.zero_code")
    verify_metrics(report["zero_code_quality"], zero_quality, "report.zero_code")
    require(set(data.get("common_games", {})) == set(JUDGES) == set(report.get("common_games", {})),
            "both common frozen terminal judges required")
    common_games = {}
    common_game_gates = {}
    for judge in JUDGES:
        raw_games = data["common_games"][judge]
        require(set(raw_games) == set(ARMS) | {"eight_seed_zero_code"}, "frozen judge arm families differ")
        common_games[judge] = {}
        for arm in (*ARMS, "eight_seed_zero_code"):
            steps = (1024,) if arm == "eight_seed_zero_code" else QUALITY_STEPS
            require(set(raw_games[arm]) == set(report["common_games"][judge][arm]) == {str(s) for s in steps},
                    "frozen judge actual score endpoints differ")
            common_games[judge][arm] = {}
            for step in steps:
                value = raw_games[arm][str(step)]
                require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and
                        value.dtype in (torch.float32, torch.float64) and tuple(value.shape) == (240, 4) and
                        bool(torch.isfinite(value).all()), "finite retained per-context/per-panel games required")
                mean = ordered_mean(value.double().mean(dim=1).tolist())
                saved = report["common_games"][judge][arm][str(step)]
                close(saved.get("mean") if isinstance(saved, dict) else saved, mean,
                      f"common_games.{judge}.{arm}.{step}")
                common_games[judge][arm][str(step)] = mean
                budget(limit)
        common_game_gates[judge] = {str(s): common_games[judge][ARMS[0]][str(s)] -
                                                   common_games[judge][ARMS[1]][str(s)] > 1e-6
                                          for s in QUALITY_STEPS}

    baseline_test = qualified_quality[ARMS[0]]["1024"]
    candidate_test = qualified_quality[ARMS[1]]["1024"]
    baseline_fit = qualified_fit_quality[ARMS[0]]["1024"]
    candidate_fit = qualified_fit_quality[ARMS[1]]["1024"]
    baseline_gap = baseline_test["mse"] - baseline_fit["mse"]
    candidate_gap = candidate_test["mse"] - candidate_fit["mse"]
    array_gates = {
        "phenotype_TEST_MSE_gte_1_10_originalFIT_MSE": baseline_fit["mse"] > 0 and
            baseline_test["mse"] >= 1.10 * baseline_fit["mse"],
        "TEST_accuracy_each_endpoint": endpoint_accuracy_gates,
        "positive_baseline_gap_and_candidate_gap_lte_0_90": baseline_gap > 0 and candidate_gap <= .90 * baseline_gap,
        "common_game_improvement_each_endpoint": common_game_gates,
        "final_zero_code_RMSE_harm_gt_1e_minus6": zero_quality["rmse"] - candidate_test["rmse"] > 1e-6,
        "final_zero_code_source_harm_strict_all_six": all(zero_quality["by_source"][str(s)] >
                                                               candidate_test["by_source"][str(s)] for s in range(6)),
        "final_zero_code_game_harm_gt_1e_minus6_both_judges": all(
            common_games[j]["eight_seed_zero_code"]["1024"] - common_games[j][ARMS[1]]["1024"] > 1e-6
            for j in JUDGES),
    }
    def all_bools(value):
        return all(all_bools(v) for v in value.values()) if isinstance(value, dict) else value is True
    array_gate_pass = all_bools(array_gates)
    retention = data.get("retention")
    require(isinstance(retention, dict) and set(retention) == set(ARMS) and retention == report.get("retention"),
            "raw/producer complete native retention witnesses differ")
    for arm in ARMS:
        witness = retention[arm]
        require(witness.get("eligible_updates") == 1023, "fixed post-initial live denominator differs")
        for key in ("bank_live", "dense_rows_live", "query_live"):
            require(isinstance(witness.get(key), int) and not isinstance(witness[key], bool) and
                    0 <= witness[key] <= 1023, "finite bounded empirical native live counts required")
        for key in ("bank_changed", "router_changed", "finite"):
            require(isinstance(witness.get(key), bool), "typed native owner witness required")
        for key in ("C_norms", "Up_norms"):
            require(isinstance(witness.get(key), dict) and set(witness[key]) == set(SITES),
                    "both actual particle site norm witnesses required")
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                        for v in witness[key].values()), "finite nonnegative particle site norms required")
        require(isinstance(witness.get("KA2_phase"), dict) and
                {"799", "800", "1024"} <= set(witness["KA2_phase"]), "native transition witnesses required")
    retained = all(w["bank_live"] / 1023 >= .9 and w["dense_rows_live"] / 1023 >= .9 and
                   w["query_live"] / 1023 >= .9 and w["bank_changed"] and w["router_changed"] and w["finite"] and
                   all(v > 0 for v in (*w["C_norms"].values(), *w["Up_norms"].values()))
                   for w in retention.values())
    native_phase = all(w["KA2_phase"]["799"] == "a" and w["KA2_phase"]["800"] == "blend" and
                       w["KA2_phase"]["1024"] == "blend" for w in retention.values())
    full_gate_checks = {
        "baseline_fit_test_phenotype": array_gates["phenotype_TEST_MSE_gte_1_10_originalFIT_MSE"],
        "held_RMSE_improved_all_five": all(v["relative_TEST_RMSE_lte_0_999"] for v in endpoint_accuracy_gates.values()),
        "no_source_harmed_all_five": all(v["source_harm_lte_1e_minus6"] for v in endpoint_accuracy_gates.values()),
        "no_TEST_seed_harmed_all_five": all(v["seed_harm_lte_1e_minus6"] for v in endpoint_accuracy_gates.values()),
        "common_originalFIT_gap_reduced": array_gates["positive_baseline_gap_and_candidate_gap_lte_0_90"],
        "both_common_games_improved_all_five": all(all(v.values()) for v in common_game_gates.values()),
        "code_RMSE_beneficial": array_gates["final_zero_code_RMSE_harm_gt_1e_minus6"],
        "code_RMSE_beneficial_each_source": array_gates["final_zero_code_source_harm_strict_all_six"],
        "code_game_beneficial_both_judges": array_gates["final_zero_code_game_harm_gt_1e_minus6_both_judges"],
        "particles_retained_both_arms": retained,
        "native_KA2_phase_qualified": native_phase,
    }
    require(report.get("gate", {}).get("thresholds") == THRESHOLDS, "producer gate thresholds differ from fixed law")
    require(report["gate"].get("checks") == full_gate_checks, "producer gate booleans differ from recomputed saved evidence")
    require(set(report["gate"].get("failed", ())) == {k for k, v in full_gate_checks.items() if not v},
            "producer failed gate list differs from recomputed saved evidence")
    gate_pass = all(full_gate_checks.values())
    require(report["gate"].get("pass") == gate_pass and
            report["scientific_status"] == ("PASS" if gate_pass else "FAIL"),
            "producer scientific status contradicts recomputed numerical gate")
    require(report.get("gate", {}).get("pass") == (report["scientific_status"] == "PASS"),
            "producer status differs from its declared complete gate")
    return {"checks": CHECKS - checks_before, "check_count_law": "Each executed require() assertion counts once, including close() prerequisites.",
            "quality": qualified_quality, "fit_quality": qualified_fit_quality,
            "zero_code_quality": zero_quality, "common_games": common_games,
            "baseline_final_TEST_minus_originalFIT_MSE": baseline_gap,
            "candidate_final_TEST_minus_originalFIT_MSE": candidate_gap,
            "recomputed_saved_array_gates": array_gates, "saved_array_gates_pass": array_gate_pass,
            "recomputed_complete_gate_checks": full_gate_checks, "recomputed_complete_gate_pass": gate_pass,
            "producer_retention_witness": retention,
            "verified_scope": "Saved original-FIT/TEST/camera metrics, common-game means, gate arithmetic and bound native counters; no new native state qualification.",
            "not_verified_here": ["Game-score model provenance beyond producer hash binding",
                                  "learned owner/gradient/moment finiteness and live-state prerequisites",
                                  "native exact recovery and source/sampling/software prerequisites"]}


def token_maps(residual):
    return residual.double().square().mean(-1).sqrt().reshape(6, 4, 4).numpy()


def curve_image(quality: dict, fit_quality: dict, *, step: int, scientific_status: str, Image, ImageDraw, ImageFont):
    width, height = 1170, 450
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    draw.text((18, 10), "Common original-FIT240 and held-out TEST240 RMSE | native game-only training", fill="#111111", font=font)
    draw.text((18, 36), "Dots are actual saved score endpoints; no interpolated scores or model reruns.", fill="#444444", font=small)
    left, right, top, bottom = 94, 1110, 83, 343
    values = [cohort[a][str(s)]["rmse"] for cohort in (quality, fit_quality) for a in ARMS for s in QUALITY_STEPS]
    lower, upper = min(values), max(values)
    margin = max((upper - lower) * .18, upper * .002, 1e-8)
    lower, upper = max(0.0, lower - margin), upper + margin
    x = lambda s: left + (s - 880) / (1024 - 880) * (right - left)
    y = lambda v: bottom - (v - lower) / (upper - lower) * (bottom - top)
    for tick in range(5):
        value = lower + (upper - lower) * tick / 4
        yy = y(value)
        draw.line((left, yy, right, yy), fill="#dddddd", width=1)
        draw.text((8, yy - 8), f"{value:.6f}", fill="#555555", font=small)
    draw.line((left, top, left, bottom, right, bottom), fill="#777777", width=2)
    for score_step in QUALITY_STEPS:
        xx = x(score_step)
        draw.text((xx - 18, bottom + 8), str(score_step), fill="#555555", font=small)
    for arm, color, label in zip(ARMS, ("#ad4b14", "#2165ad"), ("Two TRAIN seeds", "Eight TRAIN seeds")):
        shown = [(s, quality[arm][str(s)]["rmse"]) for s in QUALITY_STEPS if s <= step]
        for score_step, value in shown:
            xx, yy = x(score_step), y(value)
            draw.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=color, outline=color)
        for score_step in QUALITY_STEPS:
            if score_step <= step:
                xx, yy = x(score_step), y(fit_quality[arm][str(score_step)]["rmse"])
                draw.rectangle((xx - 4, yy - 4, xx + 4, yy + 4), fill="white", outline=color, width=2)
        legend_y = 389 if arm == ARMS[0] else 414
        draw.ellipse((18, legend_y, 26, legend_y + 8), fill=color)
        draw.text((34, legend_y - 4), label + " | dot: TEST, square: original FIT", fill=color, font=small)
    if step < QUALITY_STEPS[0]:
        draw.text((left + 90, top + 110), "First mandatory score is at update 896; none measured yet.", fill="#555555", font=small)
    label = (f"Producer gate: {scientific_status}; native-state prerequisites not regraded here."
             if step == 1024 else f"Frame {step}; later score points not shown yet.")
    draw.text((490, 397), label, fill="#333333", font=small)
    return image


def goal_frame(data: dict, quality: dict, fit_quality: dict, *, step: int, vmax: float, scientific_status: str,
               np, Image, ImageDraw, ImageFont):
    image = Image.new("RGB", (1170, 1120), "white")
    draw = ImageDraw.Draw(image)
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    draw.text((18, 12), f"Latent trajectory coverage | native update {step}/1024 per arm", fill="#111111", font=font)
    draw.text((18, 41), "Particles retained; native game-only optimization. Desired physical paired residual = 0.", fill="#333333", font=small)
    draw.text((18, 64), "Same six TEST queries: seed 39001, neutral path, t=0. Lower patch RMS error is better.", fill="#333333", font=small)
    rows = [("Target = zero", np.zeros((6, 4, 4))),
            ("Two TRAIN seeds", token_maps(data["media"][ARMS[0]][str(step)])),
            ("Eight TRAIN seeds", token_maps(data["media"][ARMS[1]][str(step)]))]
    for row, (label, maps) in enumerate(rows):
        yy = 125 + row * 169
        draw.text((14, yy + 63), label, fill="#111111", font=small)
        for source, values in enumerate(maps):
            xx = 165 + source * 164
            intensity = np.clip(values / vmax, 0, 1)
            rgb = np.stack((np.full_like(intensity, 255), 255 * (1 - intensity),
                            255 * (1 - intensity)), axis=-1).astype(np.uint8)
            image.paste(Image.fromarray(rgb).resize((150, 150), Image.Resampling.NEAREST), (xx, yy))
            draw.rectangle((xx, yy, xx + 150, yy + 150), outline="#bbbbbb", width=1)
            if row == 0:
                draw.text((xx + 39, 99), f"Source {source}", fill="#333333", font=small)
    draw.text((18, 644), f"Fixed scale from initial observations only: white=0; red >= {vmax:.6f}. No image interpolation.", fill="#333333", font=small)
    plot = curve_image(quality, fit_quality, step=step, scientific_status=scientific_status,
                       Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont)
    image.paste(plot, (0, 670))
    return image


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--producer-report", type=Path, required=True)
    parser.add_argument("--producer-completion", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] /
                        "docs/e22_routed_latent_coverage_v2.json")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-seconds", type=float, default=LIMIT_SECONDS)
    args = parser.parse_args()
    out = None
    torch = None
    old_threads = None
    entry_rng = None
    receipt = None
    error = None
    result = 2
    bindings = {}
    source = None
    limit_seconds = LIMIT_SECONDS
    try:
        require(math.isfinite(args.limit_seconds) and 0 < args.limit_seconds <= LIMIT_SECONDS,
                "separate CPU media allowance must be positive and at most 60 seconds")
        limit_seconds = args.limit_seconds
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                "CPU renderer requires CUDA_VISIBLE_DEVICES='' before import")
        require(not args.out.exists(), "fresh media directory required; existing results are preserved")
        require(all(p.is_file() for p in (args.observed, args.producer_report, args.producer_completion)),
                "existing completed producer artifacts required")
        require(args.out.resolve() not in {p.resolve() for p in
                                          (args.observed, args.producer_report, args.producer_completion)},
                "media output cannot replace an input")
        source = sha256(Path(__file__))
        bindings = {"observed_training_sha256": sha256(args.observed),
                    "producer_report_sha256": sha256(args.producer_report),
                    "producer_completion_sha256": sha256(args.producer_completion)}
        report = json.loads(args.producer_report.read_text())
        completion = json.loads(args.producer_completion.read_text())
        require(report.get("complete") is True and report.get("task") == TASK and
                report.get("scientific_status") in ("PASS", "FAIL"), "completed fixed producer task required")
        require(completion.get("complete") is True and completion.get("report_sha256") ==
                bindings["producer_report_sha256"] and completion.get("scientific_status") == report["scientific_status"],
                "completed report/scientific-status binding required")
        require(report.get("observed_training_sha256") == bindings["observed_training_sha256"],
                "actual observation raw hash differs from completed producer report")
        require(tuple(report.get("media_steps", ())) == MEDIA_STEPS and
                tuple(report.get("media_indices", ())) == MEDIA_INDICES,
                "producer fixed camera law differs")
        require(set(report.get("arms", {})) == set(ARMS) and set(report.get("quality", {})) == set(ARMS),
                "producer exact two-arm comparison required")
        provenance = verify_provenance(args.protocol, args.source_root, report, completion, limit_seconds)
        budget(limit_seconds)
        args.out.mkdir(parents=True, exist_ok=False)
        out = args.out

        # Only root executes this lazy import/load; the source author uses AST QA.
        import torch as torch_module
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        torch = torch_module
        require(not torch.cuda.is_initialized(), "CPU renderer must enter without CUDA initialized")
        entry_rng = torch.get_rng_state().clone()
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        data = torch.load(args.observed, map_location="cpu", weights_only=True)
        verified = verify_saved(data, report, torch, limit_seconds)
        require(not torch.cuda.is_initialized(), "saved-array verification initialized CUDA")
        vmax = max(float(token_maps(data["media"][arm]["0"]).max()) for arm in ARMS)
        require(math.isfinite(vmax) and vmax > 0, "nondegenerate common initial color limit required")
        frames = []
        for step in MEDIA_STEPS:
            frames.append(goal_frame(data, verified["quality"], verified["fit_quality"], step=step, vmax=vmax,
                                     scientific_status=report["scientific_status"], np=np,
                                     Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont))
            budget(limit_seconds)
        artifacts = {"goal.gif": out / "goal.gif", "goal-final.png": out / "goal-final.png",
                     "quality-points.png": out / "quality-points.png"}
        with artifacts["goal.gif"].open("xb") as handle:
            frames[0].save(handle, format="GIF", save_all=True, append_images=frames[1:],
                           duration=[900, 900, 900, 900, 900, 1800], loop=0, disposal=2)
        with artifacts["goal-final.png"].open("xb") as handle:
            frames[-1].save(handle, format="PNG")
        plot = curve_image(verified["quality"], verified["fit_quality"], step=1024, scientific_status=report["scientific_status"],
                           Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont)
        with artifacts["quality-points.png"].open("xb") as handle:
            plot.save(handle, format="PNG")
        write_json_new(out / "saved-metric-review.json", {
            "task": TASK, "complete": True, "qualification": "PASS",
            "producer_scientific_status_unchanged": report["scientific_status"],
            "science_regraded": False, "numerical_gate_recomputed": True, "native_state_reexecuted": False,
            **bindings, **verified,
            "verified_provenance": provenance,
            "native_updates": 0, "optimizer_steps": 0, "model_forwards": 0, "ParticleGAN_API_calls": 0,
        })
        budget(limit_seconds)
        require(sha256(Path(__file__)) == source, "renderer source changed during execution")
        for path, key in ((args.observed, "observed_training_sha256"),
                          (args.producer_report, "producer_report_sha256"),
                          (args.producer_completion, "producer_completion_sha256")):
            require(sha256(path) == bindings[key], "bound producer artifact changed during execution")
        require(verify_provenance(args.protocol, args.source_root, report, completion, limit_seconds) == provenance,
                "actual card/producer/source/native provenance changed within saved review")
        require(torch.equal(entry_rng, torch.get_rng_state()), "renderer changed caller CPU RNG")
        receipt = {
            "task": TASK, "complete": True, "qualification": "PASS", **bindings,
            "renderer_source_sha256": source,
            "verified_provenance": provenance,
            "artifacts": {name: sha256(path) for name, path in artifacts.items()},
            "saved_metric_review_sha256": sha256(out / "saved-metric-review.json"),
            "actual_media_steps": list(MEDIA_STEPS), "actual_quality_steps": list(QUALITY_STEPS),
            "fixed_media_indices": list(MEDIA_INDICES), "fixed_initial_RMS_color_max": vmax,
            "score_interpolation": False, "image_interpolation": False,
            "producer_scientific_status_unchanged": report["scientific_status"], "science_regraded": False,
            "numerical_gate_recomputed": True, "native_state_reexecuted": False,
            "saved_array_metric_checks": verified["checks"],
            "native_updates": 0, "optimizer_steps": 0, "model_forwards": 0, "ParticleGAN_API_calls": 0,
            "caller_CPU_RNG_unchanged": True, "CUDA_initialized": False,
            "seconds": time.monotonic() - STARTED, "limit_seconds": limit_seconds,
        }
        result = 0
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        if torch is not None:
            if old_threads is not None:
                torch.set_num_threads(old_threads)
            if entry_rng is not None and not torch.equal(entry_rng, torch.get_rng_state()):
                error = {"type": "AssertionError", "message": "CPU RNG changed during renderer cleanup"}
            if torch.cuda.is_initialized():
                error = {"type": "AssertionError", "message": "CPU renderer initialized CUDA"}
        if time.monotonic() - STARTED > limit_seconds:
            error = {"type": "TimeoutError", "message": "complete CPU renderer allowance exceeded"}
        if error is not None:
            receipt = {"task": TASK, "complete": False, "qualification": "FAIL", **bindings,
                       "renderer_source_sha256": source, "error": error,
                       "seconds": time.monotonic() - STARTED, "limit_seconds": limit_seconds}
            result = 2
        if out is not None:
            path = out / "media-completion.json"
            try:
                write_json_new(path, receipt)
            except BaseException as caught:
                print(json.dumps({"complete": False, "error": {"type": type(caught).__name__,
                                                                "message": str(caught)}}, allow_nan=False), flush=True)
                return 2
            if time.monotonic() - STARTED > limit_seconds:
                # Only this invocation's new completion is amended to reject its overrun.
                receipt.update(complete=False, qualification="FAIL", error={"type": "TimeoutError",
                               "message": "final media receipt serialization exceeded CPU allowance"},
                               seconds=time.monotonic() - STARTED)
                path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
                result = 2
        print(json.dumps(receipt, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
