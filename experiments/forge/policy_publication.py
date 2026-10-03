"""Publish the single policy-family goal board from certified combined archives.

The frozen coordinator certifies metrics and model state before combination.
This publisher checks unchanged packet/receipt/artifact identities and projects
those verdicts without API calls, training, sample generation or gate rescoring.
Pass counts rank complete configurations within one source/spec/runtime cohort;
they never pool individual cases across configurations or source revisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

from .policy_family_readout import project as project_lane


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "particlegan_policy_family_search_v1"
GOAL = "policy-family-defaults"
STATUSES = {"UNKNOWN", "RUNNING", "PASS", "FAIL", "INCOMPLETE", "BLOCKED", "ERROR", "INVALID"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    return json.loads(Path(path).read_bytes(), parse_constant=nonfinite)


def brief_diagnostic(diagnostic):
    return {key: deepcopy(value) for key, value in diagnostic.items()
            if key != "recorded_primary_observations"}


def _verify_case(row, trial, packet, case):
    """Check recorded grade/protocol bindings, without invoking its scorer."""
    receipt = read_json(row["receipt_path"])
    protocol = receipt["protocol"]
    if (receipt["case"] != case or receipt["seed"] != packet["spec"]["seed"]
            or receipt.get("requested_recipe_overrides") != trial["recipe_overrides"]
            or digest(receipt["recipe"]) != row["resolved_recipe_sha256"]
            or protocol["updates"] != case["default_steps"]
            or protocol["default_updates"] != case["default_steps"]
            or protocol["evaluation_samples"] != case["eval_samples"]
            or protocol["default_evaluation_samples"] != case["eval_samples"]
            or protocol["wall_cap_seconds"] != row["timeout_seconds"]
            or protocol["media_frames"] != packet["spec"]["frames"]
            or receipt.get("source_unchanged") is not True):
        raise ValueError("original case, seed, Recipe, sampling resources or source binding changed")
    diagnostic = row["diagnostic"]
    thresholds = case.get("thresholds", {})
    expected_count = case.get("evaluation_observations", thresholds.get("observations", 24)
                              if isinstance(thresholds, dict) else 24)
    steps = protocol["metric_evaluation_steps"]
    if (diagnostic["primary_checks"] != expected_count or steps != sorted(set(steps))
            or steps[0] != 0 or steps[-1] != case["default_steps"]):
        raise ValueError("original complete primary cadence changed")
    sustained = diagnostic["terminal_passing_suffix"] >= protocol["terminal_observations"]
    if (receipt["sustained_metric_passed"] is not sustained or receipt["passed"] is not sustained
            or receipt["verdict"] != ("PASS" if sustained else "FAIL")):
        raise ValueError("original recorded terminal gate disagrees with its verdict stream")
    hold = row["acquisition_hold"]
    window = diagnostic["first_acquisition_window"]
    if window is None:
        expected_hold = "FAIL"
    elif diagnostic["first_post_acquisition_failure"] is not None:
        expected_hold = "FAIL"
    elif diagnostic["post_acquisition_checks"] < 5:
        expected_hold = "INCOMPLETE"
    else:
        expected_hold = "PASS"
    if hold["status"] != expected_hold or row["study_gate"] != expected_hold:
        raise ValueError("separate recorded first-acquisition retention gate changed")
    if window is not None:
        acquired = next(observation for observation in receipt["observations"] if observation["step"] == window[-1])
        if (hold["acquired_step"] != window[-1] or hold["acquired_seconds"] != acquired["elapsed_seconds"]
                or hold["hold_checks"] != diagnostic["post_acquisition_checks"]
                or hold["hold_passed"] != diagnostic["post_acquisition_passed"]):
            raise ValueError("recorded hold or acquisition timing differs from original observations")
    if row.get("child_returncode") != (0 if receipt["passed"] else 1):
        raise ValueError("recorded child exit disagrees with original gate")
    paid = row.get("paid_wall_seconds")
    if (isinstance(paid, bool) or not isinstance(paid, (int, float)) or not math.isfinite(paid)
            or paid < receipt["elapsed_seconds"] or row["elapsed_seconds"] != receipt["elapsed_seconds"]):
        raise ValueError("paid cost or acquisition time differs from the bound original receipt")
    overall = receipt["verdict"] if not receipt["passed"] else expected_hold
    if row["status"] != overall:
        raise ValueError("case status differs from its preserved original and study gates")
    gif = Path(row["bound_artifacts"]["goal.gif"]["path"])
    with Image.open(gif) as image:
        frame_count = image.n_frames
    if frame_count != receipt["gif_frames"] or frame_count != len(protocol["media_steps"]):
        raise ValueError("original GIF decoded frames differ from actual media schedule")
    return frame_count


def load_combined(path):
    path = Path(path).resolve()
    packet = read_json(path)
    if (packet.get("schema") != SCHEMA or "executed_family" in packet
            or digest(packet["spec"]) != packet["spec_sha256"]
            or packet["spec"]["families"] != ["atlas", "e22"]
            or packet["spec"].get("speed_ranking") is not False
            or packet["spec"].get("default_adoption") is not False):
        raise ValueError("source-bound certified two-family combined archive required")
    stability = {"confirmation_checks": 5, "post_confirmation_hold_checks": 5,
                 "first_window_only": True, "all_subsequent_primary_checks": True}
    if packet["spec"].get("stability") != stability:
        raise ValueError("unknown first-acquisition/retention contract")
    expected = {(family, lr, rate) for family, lr, rate in product(
        packet["spec"]["families"], packet["spec"]["grid"]["lr"], packet["spec"]["grid"]["prior_lr_mult"])}
    actual = {(trial["family"], trial["recipe_overrides"]["lr"], trial["recipe_overrides"]["prior_lr_mult"])
              for trial in packet["trials"]}
    if len(packet["trials"]) != len(expected) or actual != expected:
        raise ValueError("all declared complete configurations must remain in the denominator")
    archives = packet.get("family_archives", [])
    if len(archives) != 2 or {archive["family"] for archive in archives} != {"atlas", "e22"}:
        raise ValueError("exactly one original bound archive per family required")
    lanes, raw_trials = {}, {}
    for archive in archives:
        original = Path(archive["path"])
        if file_hash(original) != archive["sha256"]:
            raise ValueError("original family archive changed after certified combination")
        raw = read_json(original)
        if (raw["executed_family"] != archive["family"] or raw["lane_runtime"] != archive["runtime"]
                or any(raw[key] != packet[key] for key in
                       ("spec_sha256", "spec", "source", "case_definitions", "capacity_preflight", "runtime_contract"))):
            raise ValueError("source/spec/capacity/runtime cohorts cannot be pooled")
        lanes[archive["family"]] = project_lane(original)
        raw_trials.update({trial["id"]: trial for trial in raw["trials"] if trial["family"] == archive["family"]})
    declared_cases = [(case["id"], case["tier"]) for case in packet["spec"]["cases"]]
    denominator = dict(Counter(tier for _, tier in declared_cases))
    trials = []
    for original in packet["trials"]:
        if raw_trials.get(original["id"]) != original:
            raise ValueError("combined trial differs from its unchanged original family packet")
        knobs = original["recipe_overrides"]
        if (set(knobs) != {"lr", "prior_lr_mult"} or original["id"] != original["family"] + "--" +
                digest({"family": original["family"], "overrides": knobs})):
            raise ValueError("whole-configuration content identity changed")
        lane_trial = next(trial for trial in lanes[original["family"]]["trials"] if trial["id"] == original["id"])
        if [(row["id"], row["tier"]) for row in lane_trial["cases"]] != declared_cases:
            raise ValueError("required case roster changed")
        cases = []
        for row in lane_trial["cases"]:
            case = packet["case_definitions"][row["id"]]
            if row["status"] not in STATUSES or row["case_sha256"] != digest(case):
                raise ValueError("case definition or recorded status changed")
            if ((row["status"] in {"PASS", "FAIL"} or row.get("original_gate") == "PASS"
                 or row.get("full_protocol_complete")) and not row.get("diagnostic")):
                raise ValueError("recorded scientific gate lacks an unchanged full-budget receipt")
            item = {key: deepcopy(row.get(key)) for key in
                    ("id", "tier", "status", "case_sha256", "resolved_recipe_sha256", "original_gate", "study_gate",
                     "full_protocol_complete", "acquisition_hold", "paid_wall_seconds", "elapsed_seconds", "reason",
                     "receipt_path", "receipt_sha256", "final_metrics", "original_failed_bounds")}
            item["title"] = case["title"]
            item["goal"] = case["goal"]
            item["actual_gif"] = None
            for field in ("evidence_source", "evidence_runtime"):
                if field in row:
                    item[field] = deepcopy(row[field])
            if row.get("diagnostic"):
                frames = _verify_case(row, original, packet, case)
                item.update(diagnostic=brief_diagnostic(row["diagnostic"]), recipe=deepcopy(row["recipe"]),
                            sampling=deepcopy(row["original_sampling"]), protocol=deepcopy(row["protocol"]),
                            raw_artifacts=deepcopy(row["bound_artifacts"]), runtime=deepcopy(row["runtime"]),
                            actual_gif={**row["bound_artifacts"]["goal.gif"], "frames": frames,
                                        "media_steps": row["protocol"]["media_steps"]})
            cases.append(item)
        counts = {tier: {"study_passed": sum(row["status"] == "PASS" and row["tier"] == tier for row in cases),
                         "original_passed": sum(row["original_gate"] == "PASS" and row["tier"] == tier for row in cases),
                         "required": total} for tier, total in denominator.items()}
        eligible = original["status"] == "PASS" and all(row["status"] == "PASS" for row in cases)
        if original["status"] == "PASS" and not eligible:
            raise ValueError("a passing complete configuration lacks required passing cases")
        trials.append({"id": original["id"], "family": original["family"], "recipe_overrides": deepcopy(knobs),
                       "status": original["status"], "reason": original.get("reason"),
                       "fully_qualified": eligible, "counts": counts, "required_cases": len(cases),
                       "measured_cases": sum(bool(row.get("diagnostic")) for row in cases),
                       "unknown_cases": sum(row["status"] == "UNKNOWN" for row in cases),
                       "paid_wall_seconds": original.get("paid_wall_seconds", 0.), "cases": cases})
    eligible_ids = sorted(trial["id"] for trial in trials if trial["fully_qualified"])
    if packet["selection"]["fully_qualified_ids"] != eligible_ids:
        raise ValueError("combined whole-configuration eligibility changed")
    tier_order = sorted(denominator)
    ranking = lambda trial: (*(-trial["counts"][tier]["study_passed"] for tier in tier_order), trial["id"])
    family_best, family_ties = {}, {}
    for family in packet["spec"]["families"]:
        ranked = sorted((trial for trial in trials if trial["family"] == family), key=ranking)
        measured = [trial for trial in ranked if trial["measured_cases"]]
        family_best[family] = measured[0]["id"] if measured else None
        family_ties[family] = [trial["id"] for trial in measured
                               if ranking(trial)[:-1] == ranking(measured[0])[:-1]] if measured else []
    cohort_id = packet["spec"]["id"] + "--" + packet["source"]["commit"][:12]
    return {"id": cohort_id, "study_id": packet["spec"]["id"],
            "combined_archive": {"path": str(path), "sha256": file_hash(path)},
            "upstream_certification": "frozen api_family_search.combine_studies; publisher verifies recorded bindings",
            "source": deepcopy(packet["source"]), "spec_sha256": packet["spec_sha256"],
            "runtime_contract": deepcopy(packet["runtime_contract"]), "family_archives": deepcopy(archives),
            "capacity_preflight_sha256": digest(packet["capacity_preflight"]),
            "declared_configs": len(trials), "required_case_denominators": denominator,
            "required_cases_per_config": len(declared_cases),
            "measured_paid_seconds": packet["measured_paid_seconds"],
            "unmeasured_interrupt_reservation_seconds": packet["unmeasured_interrupt_reservation_seconds"],
            "selection": deepcopy(packet["selection"]), "per_family_best_observed": family_best,
            "per_family_best_observed_ties": family_ties,
            "trials": sorted(trials, key=lambda trial: (trial["family"], *ranking(trial)))}


def publisher_source():
    source = Path(__file__).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    files = {path.relative_to(ROOT).as_posix(): file_hash(path)
             for path in (source, source.with_name("policy_family_readout.py"))}
    committed_at_head = True
    for path, expected in files.items():
        committed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False)
        committed_at_head &= committed.returncode == 0 and hashlib.sha256(committed.stdout).hexdigest() == expected
    return {"path": source.relative_to(ROOT).as_posix(), "sha256": file_hash(source),
            "files_sha256": files, "repository_head": commit,
            "publisher_committed_at_head": committed_at_head}


def review_projection_media(row, output, relative):
    """Show the retained KS value/bound without changing any observation."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmarks.toy_audit import api_reframe, api_run
    raw = Path(row["receipt_path"]).parent
    receipt = read_json(raw / "receipt.json")
    bounds = [bound for bound in receipt["case"]["thresholds"]
              if isinstance(bound, list) and len(bound) == 3 and bound[0] == "projection_ks"]
    if len(bounds) != 1 or bounds[0][1] != "<=":
        raise ValueError("one original numeric projection-KS upper bound required")
    before = api_reframe._raw_identity(raw)
    records = api_reframe.reconstruct_media(raw, receipt)
    numeric_identity = api_reframe._observation_identity(records)
    order = ("projection_ks", "hq", "mass_tv", "sw1_normalized", "component_covariance_error", "max_component_spill")
    for record in records:
        if "projection_ks" not in record["metrics"]:
            raise ValueError("every retained goal frame must contain its original projection KS")
        values = record["metrics"]
        record["metrics"] = {**{name: values[name] for name in order if name in values}, **values}
    # Sorted metadata identity ignores dictionary display order but binds every
    # exact value, verdict, view label and tensor byte. No scorer is invoked.
    if api_reframe._observation_identity(records) != numeric_identity:
        raise ValueError("display ordering altered retained observations")
    case = deepcopy(receipt["case"])
    annotation = f"Original shape bound: projection_ks <= {bounds[0][2]:g}."
    case["goal"] = case["goal"] + " " + annotation
    renderer = api_reframe.renderer_source()
    source = publisher_source()
    destination = output / relative
    temporary = destination.with_name(destination.stem + "-rendering.gif")
    try:
        result = api_run.render_gif(case, records, temporary,
                                   full_budget=receipt["default_protocol_complete"],
                                   requested_steps=receipt["protocol"]["updates"], final_verdict=receipt["verdict"])
        with Image.open(temporary) as gif:
            frames = gif.n_frames
        if (frames != len(receipt["protocol"]["media_steps"])
                or result.get("numeric_observations_changed") is not False
                or result.get("default_verdict_displayed") != receipt["verdict"]
                or api_reframe._observation_identity(records) != numeric_identity
                or api_reframe._raw_identity(raw) != before
                or api_reframe.renderer_source() != renderer or publisher_source() != source):
            raise ValueError("media-only review changed source, observations, raw artifacts or verdict")
        temporary.replace(destination)
        sidecar_relative = relative.with_suffix(".media-review.json")
        review = {"schema": "policy_family_metric_media_review_v1", "raw_receipt": str(raw / "receipt.json"),
                  "raw_receipt_sha256": before["receipt.json"]["sha256"],
                  "raw_artifacts": deepcopy(receipt["artifacts"]), "raw_files_unchanged": True,
                  "training_source_commit": receipt["source"]["commit"],
                  "renderer_source": renderer, "publisher_source": source,
                  "original_case_sha256": digest(receipt["case"]), "goal_annotation": annotation,
                  "numeric_observation_sha256": numeric_identity, "numeric_observations_changed": False,
                  "display_metric_order": list(order), "original_gate": receipt["verdict"],
                  "study_gate": row["study_gate"], "media_steps": receipt["protocol"]["media_steps"],
                  "training_or_rescoring": False, "render_annotations": result,
                  "reviewed_gif": {"relative_path": relative.as_posix(), "sha256": file_hash(destination),
                                   "bytes": destination.stat().st_size, "frames": frames}}
        (output / sidecar_relative).write_text(json.dumps(review, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return {**review["reviewed_gif"], "media_steps": review["media_steps"],
                "review_sidecar": {"relative_path": sidecar_relative.as_posix(),
                                   "sha256": file_hash(output / sidecar_relative)}}
    finally:
        temporary.unlink(missing_ok=True)


def copy_representative_media(cohort, output, *, review_projection=False, all_media=False):
    """Copy unchanged selected/all observed GIFs, marking display representatives."""
    media = []
    for family, trial_id in cohort["per_family_best_observed"].items():
        if trial_id is None:
            continue
        best = next(trial for trial in cohort["trials"] if trial["id"] == trial_id)
        observed = [row for row in best["cases"] if row["actual_gif"]]
        representatives = {observed[0]["id"], observed[-1]["id"]} if observed else set()
        selected = [(trial, row) for trial in cohort["trials"] if trial["family"] == family
                    for row in trial["cases"] if row["actual_gif"] and
                    (all_media or (trial["id"] == trial_id and row["id"] in representatives))]
        for trial, row in selected:
            raw = row["actual_gif"]
            relative = Path("policy-family-media") / cohort["id"] / family / (trial["id"].split("--")[1][:12] + "--" + row["id"] + ".gif")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if file_hash(raw["path"]) != raw["sha256"]:
                raise ValueError("original representative GIF changed")
            shutil.copyfile(raw["path"], destination)
            if file_hash(destination) != raw["sha256"]:
                raise ValueError("copied representative GIF differs from original")
            published = {"relative_path": relative.as_posix(), "sha256": raw["sha256"], "bytes": raw["bytes"],
                         "frames": raw["frames"], "media_steps": raw["media_steps"], "source_path": raw["path"],
                         "family": family, "trial_id": trial["id"], "case_id": row["id"],
                         "original_gate_displayed": row["original_gate"], "study_gate": row["study_gate"],
                         "representative": trial["id"] == trial_id and row["id"] in representatives,
                         "reviewed": False, "numeric_observations_changed": False}
            if review_projection and "projection_ks" in row["final_metrics"]:
                reviewed_relative = relative.with_name(relative.stem + "-reviewed.gif")
                reviewed = review_projection_media(row, output, reviewed_relative)
                published["original_gif"] = {key: published[key] for key in
                                             ("relative_path", "sha256", "bytes", "frames", "media_steps")}
                published.update(reviewed, reviewed=True)
            row["published_gif"] = deepcopy(published)
            media.append(published)
    cohort["representative_media"] = media
    cohort["published_media_scope"] = "all actual observed runs" if all_media else "best-observed display representatives"


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def reason(row):
    diagnostic = row.get("diagnostic")
    if diagnostic:
        broken = diagnostic["first_post_acquisition_failure"]
        bounds = broken["failed_bounds"] if broken else row.get("original_failed_bounds", [])
        prefix = f"hold break at {broken['step']}: " if broken else ""
        return prefix + ("; ".join(bounds) or row["acquisition_hold"]["reason"])
    return row.get("reason") or "unmeasured; retained in the required denominator"


def markdown(board):
    current = board["current_cohort_id"]
    lines = ["# Policy-family goal board", "",
             "This is the primary leaderboard for **policy-family-defaults**. It ranks whole shared-knob "
             "configurations within each frozen family/source/spec/runtime lane. The public selected/served "
             "particle-cloud questions retain their original host adaptations; this is separate from "
             "the Forge clean-MoG inventory.", "",
             f"Current cohort: `{current}`. Source cohorts remain separate; individual passing cases "
             "never combine into a configuration. UNKNOWN stays unmeasured. This screen supplies "
             "**no shipping-default adoption or speed winner**.", "",
             "The original toy gate requires its terminal passing suffix. The added study gate uses "
             "the first five consecutive primary successes, then at least five uninterrupted hold "
             "checks and every subsequent primary check. An original PASS may therefore have a "
             "FAIL or INCOMPLETE study verdict. GIF badges display the original toy verdict.", ""]
    lines += ["Matching small-host numerical traces do not establish Atlas/E22 algorithmic or complete "
              "policy-state equivalence. Reference-kNN smoke results do not qualify Atlas feature-cell "
              "behavior; unmeasured native quality cases remain in the denominator.", ""]
    for cohort in reversed(board["cohorts"]):
        lines += [f"## {'Current' if cohort['id'] == current else 'Retained'} cohort `{cohort['id']}`", "",
                  f"{cohort['declared_configs']} declared configurations; each retains "
                  f"{cohort['required_cases_per_config']} required cases. Paid child execution "
                  f"{cohort['measured_paid_seconds']:.3f}s. Whole-config outcome: "
                  f"**{cohort['selection']['outcome']}**; fully qualified: "
                  f"**{len(cohort['selection']['fully_qualified_ids'])}**. "
                  "Partial or interrupted evidence supplies no additional qualification.", "",
                  "| Family / config | LR / prior rate | Study smoke | Study quality | Original passes | Measured / required | Whole-config status | First non-pass |",
                  "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |"]
        for trial in cohort["trials"]:
            counts = trial["counts"]
            smoke = counts.get(1, counts.get("1", {}))
            quality = counts.get(2, counts.get("2", {}))
            first = next((row for row in trial["cases"] if row["status"] not in {"PASS", "UNKNOWN"}), None)
            why = f"{first['id']}: {reason(first)}" if first else trial.get("reason") or "all observed cases pass; see required denominator"
            values = [f"{trial['family']} / `{trial['id'].split('--')[1][:12]}`",
                      f"{trial['recipe_overrides']['lr']} / {trial['recipe_overrides']['prior_lr_mult']}",
                      f"{smoke.get('study_passed', 0)}/{smoke.get('required', 0)}",
                      f"{quality.get('study_passed', 0)}/{quality.get('required', 0)}",
                      sum(count['original_passed'] for count in counts.values()),
                      f"{trial['measured_cases']}/{trial['required_cases']}", trial["status"], why]
            lines.append("| " + " | ".join(escape(value) for value in values) + " |")
        lines += ["", "Per-family best observed display (pass counts, then content ID; not a default or speed selection): " +
                  "; ".join(f"{family} `{trial_id.split('--')[1][:12]}`" if trial_id else f"{family} unmeasured"
                            for family, trial_id in cohort["per_family_best_observed"].items()) + ".", "",
                  "| Observed case / config | Original gate | Study gate | Acquisition / hold passed | Terminal metrics | Actual GIF |",
                  "| --- | --- | --- | --- | --- | --- |"]
        tied = {family: identities for family, identities in cohort["per_family_best_observed_ties"].items()
                if len(identities) > 1}
        if tied:
            lines[-2:-2] = ["Pass-count ties: " + "; ".join(
                family + " " + ", ".join(f"`{identity.split('--')[1][:12]}`" for identity in identities)
                for family, identities in tied.items()) +
                ". The displayed content ID only breaks a presentation tie; it supplies no measured quality advantage.", ""]
        for trial in cohort["trials"]:
            for row in trial["cases"]:
                if not row.get("diagnostic"):
                    continue
                metrics = row["final_metrics"]
                names = ("hq", "modes", "finite_template_tv", "distribution_tv", "mass_tv", "projection_ks",
                         "acc_center_rms_sigma", "acc_cov_rel_frob", "acc_radial_ks")
                compact = "; ".join(f"{name}={metrics[name]:.6g}" for name in names if name in metrics)
                hold = row["acquisition_hold"]
                acquired = f"{hold.get('acquired_step', 'none')} / {hold.get('hold_passed', 0)}/{hold.get('hold_checks', 0)}"
                gif = row.get("published_gif")
                link = f"[actual {gif['frames']}-frame GIF]({gif['relative_path']})" if gif else "bound raw GIF in JSON"
                values = [f"{trial['family']} `{trial['id'].split('--')[1][:12]}` / {row['id']}",
                          row["original_gate"], row["study_gate"], acquired, compact, link]
                lines.append("| " + " | ".join(escape(value) for value in values) + " |")
        for media in cohort["representative_media"]:
            if not media["representative"]:
                continue
            lines += ["", f"{media['family']} `{media['trial_id'].split('--')[1][:12]}` / `{media['case_id']}`: "
                      f"original **{media['original_gate_displayed']}**, study **{media['study_gate']}**. " +
                      ("Reviewed retained observations: original numeric values and verdict unchanged; "
                       "projection KS and its original bound are visible." if media["reviewed"] else
                       "Copied unchanged from actual retained observations."), "",
                      f"![{media['case_id']} actual training]({media['relative_path']})"]
            if media["reviewed"]:
                lines += ["", f"[Unchanged original GIF]({media['original_gif']['relative_path']}) · "
                          f"[Separate media-review provenance]({media['review_sidecar']['relative_path']})."]
        lines += ["", f"Frozen source `{cohort['source']['commit']}`; spec SHA256 `{cohort['spec_sha256']}`; "
                  f"combined archive SHA256 `{cohort['combined_archive']['sha256']}`. "
                  "Exact recipes, priors/serving law, runtimes, source manifests, case receipts, artifact "
                  "hashes and all unknown required cells are retained in the JSON.", ""]
    lines += ["[Complete compact goal inventory](policy-family-inventory.json). Detailed case readouts "
              "are explanatory and do not create another primary leaderboard.", ""]
    return "\n".join(lines)


def publish(paths, output, *, review_projection=False, all_media=False):
    if not paths:
        raise ValueError("at least one certified combined archive required")
    cohorts = [load_combined(path) for path in paths]
    if len({cohort["id"] for cohort in cohorts}) != len(cohorts):
        raise ValueError("duplicate source/spec cohort inputs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        copy_representative_media(cohort, output, review_projection=review_projection, all_media=all_media)
    board = {"schema": "policy_family_goal_inventory_v1", "goal": GOAL, "primary_goal_board": True,
             "generated_by": publisher_source(), "training_updates": 0, "metric_rescoring": False,
             "current_cohort_id": cohorts[-1]["id"], "default_adoption": False, "speed_winner": None,
             "ranking": "within each family/source/spec/runtime lane: studyPASS counts by tier, then complete config content ID",
             "cross_cohort_case_pooling": False, "cohorts": cohorts}
    encoded = json.dumps(board, sort_keys=True, indent=2, allow_nan=False) + "\n"
    rendered = markdown(board)
    (output / "policy-family-inventory.json").write_text(encoded)
    (output / "policy-family-inventory.md").write_text(rendered)
    return board


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, action="append", required=True,
                        help="certified combined archive; oldest to current, repeat to retain cohorts")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/forge")
    parser.add_argument("--review-projection", action="store_true",
                        help="media-only retained-array review exposing the original projection KS value/bound")
    parser.add_argument("--all-media", action="store_true",
                        help="copy every reached run's original GIF and link it in the observed-case table")
    args = parser.parse_args(argv)
    board = publish(args.combined, args.output, review_projection=args.review_projection, all_media=args.all_media)
    print(json.dumps({"goal": GOAL, "cohorts": len(board["cohorts"]),
                      "current": board["current_cohort_id"], "training_updates": 0,
                      "published_runs": sum(len(cohort["representative_media"]) for cohort in board["cohorts"]),
                      "published_gifs": sum(1 + media["reviewed"] for cohort in board["cohorts"]
                                            for media in cohort["representative_media"])}))


if __name__ == "__main__":
    main()
