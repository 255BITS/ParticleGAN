"""Markdown and offline HTML views of the same sparse evidence projection."""
import html
import json
from posixpath import relpath
from pathlib import Path
from collections import Counter

from .test_performance import cell_status


def render_discovery_navigation(root, page):
    """Shared inventory navigation over byte-pinned scoped outcomes only."""
    from .test_performance import Inputs, SUPPLEMENTAL, portable
    inputs = Inputs(root)
    if not (inputs.root / SUPPLEMENTAL).is_file():
        return ""
    registry = inputs.read(SUPPLEMENTAL)
    if registry.get("schema") != "forge_test_display_registry_v1" or registry.get("qualification_input") is not False:
        raise ValueError("test discoveries: invalid display registry")
    full = prefix = reconciled = astra = None
    confidence_packets = {}
    confidence_ids = {'pr399-confidence-lr-tier1-6-20261013': 'screen', 'pr399-confidence-lr-own-retention-2-20261013': 'retention', 'pr399-gaussian-finite-cap-1-20261013': 'finite_cap', 'pr399-critic-orientation-2-20261013': 'orientation', 'pr399-signed-confidence-2-20261013': 'signed', 'pr399-positive-evidence-2-20261013': 'positive'}
    for pin in registry["publications"]:
        document = inputs.read(pin["path"], sha256=pin["sha256"])
        summary = document.get("discovery_summary", {})
        if summary.get("id") not in {"nonatlas-cell-moment-seed0-20261012", "nonatlas-finite-realization-prefix64", "nonatlas-reconciled-204-20261013", "astra-cumulative-24-20261013"} | set(confidence_ids):
            continue
        if (document.get("schema") != "forge_test_display_evidence_v1" or document.get("qualification_input") is not False
                or document.get("qualification_reuse") is not False or summary.get("ordinary_qualification") is not False
                or summary.get("promotion") is not False or portable(document) != document):
            raise ValueError("test discoveries: scoped evidence cannot promote or leak local references")
        if summary["id"] in confidence_ids:
            key = confidence_ids[summary["id"]]
            if key in confidence_packets:
                raise ValueError("test discoveries: duplicate confidence summary")
            confidence_packets[key] = pin, document
        elif summary["id"] == "astra-cumulative-24-20261013":
            if astra is not None:
                raise ValueError("test discoveries: duplicate Astra cumulative summary")
            astra = pin, document
        elif summary["id"] == "nonatlas-reconciled-204-20261013":
            if reconciled is not None:
                raise ValueError("test discoveries: duplicate reconciled summary")
            reconciled = pin, document
        elif summary["id"] == "nonatlas-cell-moment-seed0-20261012":
            if full is not None:
                raise ValueError("test discoveries: duplicate full-run summary")
            for round in summary["rounds"]:
                cells = [cell for cell in document["cells"] if cell["result"]["round"] == round["id"]]
                counts = Counter(("control" if cell["result"]["mode"] == "per_kernel" else "aggregate", cell["status"])
                                 for cell in cells)
                if (len(cells) != round["full_runs"] or any(cell["full_protocol_complete"] is not True for cell in cells)
                        or counts["aggregate", "PASS"] != round["aggregate_passed"]
                        or counts["control", "PASS"] != round["control_passed"]
                        or sum(count for (kind, _), count in counts.items() if kind == "aggregate") != round["aggregate_total"]
                        or sum(count for (kind, _), count in counts.items() if kind == "control") != round["control_total"]
                        or abs(sum(cell["result"]["actual_Popen_to_exit_seconds"] for cell in cells)
                               - round["Popen_to_exit_seconds"]) > 1e-7):
                    raise ValueError("test discoveries: summary differs from original cells")
            if (summary["full_run_count"] != len(document["cells"])
                    or sum(round["full_runs"] for round in summary["rounds"]) != len(document["cells"])
                    or any(summary[field] != sum(round[field] for round in summary["rounds"])
                           for field in ("aggregate_passed", "aggregate_total", "control_passed", "control_total"))
                    or abs(summary["full_round_charge_seconds"]
                           - sum(round["Popen_to_exit_seconds"] for round in summary["rounds"])) > 1e-7):
                raise ValueError("test discoveries: overall summary differs from original rounds")
            full = pin, summary
        else:
            if len(document["cells"]) != summary["prefix_cells"] or any(cell["status"] != "INCOMPLETE"
                    or cell["full_protocol_complete"] is not False for cell in document["cells"]) or abs(
                    sum(cell["result"]["actual_Popen_to_exit_seconds"] for cell in document["cells"])
                    - summary["paired_charge_seconds"]) > 1e-7:
                raise ValueError("test discoveries: prefix cannot become full quality evidence")
            prefix = pin, summary
    if full is None:
        return ""
    if prefix and abs(prefix[1]["combined_disjoint_charge_seconds"]
                      - full[1]["full_round_charge_seconds"] - prefix[1]["paired_charge_seconds"]) > 1e-7:
        raise ValueError("test discoveries: disjoint charge differs from original scopes")
    base = Path(page).parent
    def link(name):
        return relpath(str(Path(root) / name), str(base))
    if reconciled:
        text = _render_reconciled(reconciled, full, prefix, registry, inputs, link)
        if astra:
            text += _render_astra(astra, reconciled, link)
        text += _render_confidence_packets(confidence_packets, link)
        return text
    pin, summary = full
    lines = ["", "## Scoped non-Atlas discoveries", "",
             f"[Verified complete recipes, exact protocols and all outcomes]({link(pin['path'])}): "
             f"**{summary['full_run_count']} unique full runs** in three separately scoped seed-0 rounds. "
             f"Aggregate outcomes are **{summary['aggregate_passed']}/{summary['aggregate_total']} PASS**; "
             f"the **{summary['control_total']} unique controls** are **{summary['control_passed']}/{summary['control_total']} PASS**. "
             "These are execution-ledger counts, not a pooled score, independent control repetitions or an ordinary whole-suite winner.", "",
             "| Registered round / task IDs | Executed source | Aggregate PASS / total | Control PASS / total | Inclusive child seconds |",
             "| --- | --- | ---: | ---: | ---: |"]
    for round in summary["rounds"]:
        lines.append(f"| {_text(round['id'])}<br>{', '.join('`'+name+'`' for name in round['case_ids'])} | "
                     f"`{round['source_digest'][:12]}` · seed 0 | {round['aggregate_passed']}/{round['aggregate_total']} | "
                     f"{round['control_passed']}/{round['control_total']} | {round['Popen_to_exit_seconds']:.9f} |")
    lines += ["", "All four aggregate objectives pass the anisotropic, unequal-width, unequal-mass, broad, overlap and spiral "
              "scoped cases in the first two rounds. Acquisition preserves **Bures ring FAIL**. The Gaussian acquisition "
              "smoke uses any scheduled full pass plus an independent same-state draw; ring executes 1,600 updates with "
              "96 observations and its original 400-update schedule horizon. Published-discriminator width is a distinct "
              "task ID. Metric regressions remain visible; no objective is declared universally superior.", ""]
    if prefix:
        pp, ps = prefix
        lines += [f"[Paired finite-realization mechanism prefix]({link(pp['path'])}): two 64-update arms, both **INCOMPLETE** "
                  "against the original 7,000-update staggered100 quality gates. "
                  f"Prefix charge **{ps['paired_charge_seconds']:.9f} seconds**; disjoint charge including the three full rounds "
                  f"**{ps['combined_disjoint_charge_seconds']:.9f} seconds**. No prefix PASS is imported into full quality.", ""]
    lines += ["Preparation/audit errors and exclusions are retained in the pinned receipts. Original source, generated arrays "
              "and checkpoints remain local-only; individual consumed CUDA packets were not separately retained. "
              "These discoveries change display coverage only. Selected family rows, ordinary qualification and defaults remain unchanged.", ""]
    return "\n".join(lines) + "\n" + _render_confidence_packets(confidence_packets, link)


def _render_reconciled(reconciled, original, prefix, registry, inputs, link):
    """Reconcile ledger summaries against every registered original cell."""
    pin, document = reconciled
    summary = document["discovery_summary"]
    cells = [cell for entry in registry["publications"]
             for cell in inputs.read(entry["path"], sha256=entry["sha256"])["cells"]
             if cell["id"].startswith("nonatlas-")]
    full = [cell for cell in cells if cell["full_protocol_complete"] is True]
    errors = [cell for cell in document["cells"] if cell["status"] == "ERROR"]
    known_attempts = {cell["result"]["attempt_id"] for cell in full}
    scopes = {entry["id"] for entry in summary["rounds"]}
    if (len(full) != summary["full_run_count"] or len(known_attempts) != len(full)
            or len({cell["candidate_id"] for cell in full}) != summary["unique_recipe_ids"]
            or dict(Counter(cell["status"] for cell in full)) != summary["full_run_status_counts"]
            or summary["global_survivor"] is not False
            or {cell["result"]["round"] for cell in full} != scopes
            or len(errors) != summary["startup_error_count"]):
        raise ValueError("test discoveries: reconciled ledger differs from registered cells")
    conditions = {key: value for entry in registry["publications"]
                  for key, value in inputs.read(entry["path"], sha256=entry["sha256"]).get("condition_contracts", {}).items()}
    identities = {(cell["candidate_id"], cell["candidate_revision"], conditions[cell["conditions_sha256"]]["source_digest"])
                  for cell in full}
    sources = {identity[2] for identity in identities}
    if len(identities) != summary["source_bound_revisions"] or len(sources) != summary["source_cohorts"]:
        raise ValueError("test discoveries: source or revision coverage differs from cells")
    for round in summary["rounds"]:
        rows = [cell for cell in full if cell["result"]["round"] == round["id"]]
        if (len(rows) != round["full_runs"] or dict(Counter(cell["status"] for cell in rows)) != round["counts"]
                or sorted({conditions[cell["conditions_sha256"]]["source_digest"] for cell in rows}) != round["source_digests"]
                or abs(sum(cell["result"]["actual_Popen_to_exit_seconds"] for cell in rows)
                       - round["Popen_to_exit_seconds"]) > 1e-7):
            raise ValueError("test discoveries: round counts or charge differ from cells")
    charge = sum(cell["result"]["actual_Popen_to_exit_seconds"] for cell in full)
    failed_charge = sum(cell["result"]["actual_Popen_to_exit_seconds"] for cell in errors)
    if (abs(charge - summary["full_round_charge_seconds"]) > 1e-7
            or abs(failed_charge - summary["startup_error_charge_seconds"]) > 1e-7
            or prefix is None or abs(charge + failed_charge + prefix[1]["paired_charge_seconds"]
                                    - summary["total_charged_attempt_wall_seconds"]) > 1e-7):
        raise ValueError("test discoveries: reconciled disjoint charge differs from cells")
    counts = summary["full_run_status_counts"]
    followup = summary["cpu_reference_followup"]
    cpu = followup["reference_summary"]
    if (cpu["global_qualification"] is not False or cpu["new_training_updates"] != 0 or cpu["new_GPU_children"] != 0
            or cpu["primary_decision"] != followup["reference_ablation"]["decision"]
            or cpu["secondary_posthoc_decision"] != followup["quota_confirmation"]["decision"]
            or abs(cpu["timed_CPU_wall_seconds"] - sum(followup[key]["CPU_wall_seconds"]
                   for key in ("reference_ablation", "quota_confirmation", "quota_receipts"))) > 1e-7):
        raise ValueError("test discoveries: CPU diagnostic summary differs from original receipts")
    lines = ["", "## Scoped non-Atlas discoveries", "",
             f"[Verified reconciled receipts, complete recipes and audit gaps]({link(pin['path'])}): "
             f"**{len(full)} full runs** across **{summary['unique_recipe_ids']} recipe IDs**, "
             f"**{len(identities)} source-bound revisions** and **{len(sources)} source cohorts**. "
             f"Recorded outcomes are **{counts.get('PASS', 0)} PASS / {counts.get('FAIL', 0)} FAIL**. "
             "These are disjoint ledger counts across exact protocols; **no global survivor or pooled score** is established.", "",
             "**Evidence limits:** six mode-hold transfers lack saved sample panels; raw archives and full-run CUDA packets "
             "are not publicly available. Their original logged-metric FAIL grades remain explicit.", "",
             "<details>", "<summary>Open the 204-run cohort table, exact clocks and audit limitations</summary>", "",
             "| Round | Full PASS / FAIL | Source cohorts | Inclusive attempt wall seconds | Audit |",
             "| --- | ---: | --- | ---: | --- |"]
    for round in summary["rounds"]:
        lines.append(f"| {_text(round['id'])} | {round['counts'].get('PASS', 0)} / {round['counts'].get('FAIL', 0)} | "
                     f"{', '.join('`'+source[:12]+'`' for source in round['source_digests'])} | "
                     f"{round['Popen_to_exit_seconds']:.9f} | {_text(round['audit_status'])} |")
    lines += ["", "The original 45 geometry/transfer/acquisition runs remain indexed once: aggregate 35/36 PASS and "
              "five of nine fresh controls PASS, with Bures ring FAIL retained. All cumulative representatives, all 12 memory "
              "corrections and all 12 mixed-role recipes fail full Gaussian stability; all 12 quadrature recipes fail mode retention. "
              "Gradient attenuation passes five of 12 full Gaussian runs, but all five passing recipes fail essential transfer gates. "
              "Forward, reverse and moment transfer cohorts pass 2/12, 3/12 and 3/12; these cannot fill one another's protocols.", "",
              "Gaussian continuations retain the complete 6,000-step clock and their own pinned 1,000-step producers: "
              "only 5,000 new updates and the new attempt charge are counted. Complete recipes expose objective, reference policy, "
              "cubature, optimizer, G/D/prior role assignments and signal attenuation. Published/base architectures and source revisions remain separate.", "",
              f"**{len(errors)} pre-runtime ERROR cells** cost **{failed_charge:.9f} seconds**, with zero training updates. "
              "Three CPU-nonfinite SGDA rejections, semantic dedup exclusions, preparation/audit failures and the CPU gradient "
              "probe remain separate diagnostics. The probe's REJECT decision supplies no optimizer or feasibility PASS.", "",
              f"The amended CPU reference ablation keeps primary **{cpu['primary_decision']}** and secondary posthoc quota "
              f"**{cpu['secondary_posthoc_decision']}** separate, "
              "with original registrations and transparent quota/directional-measurement errata. These add zero trained rows "
              f"and no qualified recipe or feasibility witness; {cpu['timed_CPU_wall_seconds']:.9f} timed CPU seconds "
              "remain separate from attempt charges.", "",
              f"[Two paired64 cells]({link(prefix[0]['path'])}) remain **INCOMPLETE** against original 7,000-step staggered100 gates. "
              f"Full-run wall charge **{charge:.9f} seconds**; prefix **{prefix[1]['paired_charge_seconds']:.9f} seconds**; "
              f"disjoint full/prefix/failed-supervisor total **{summary['total_charged_attempt_wall_seconds']:.9f} seconds**. "
              "CPU preparation/probe costs are recorded separately; no exclusive CUDA-time or cross-protocol speed ranking is claimed.", "",
              "**Six mode-hold transfer jobs have unavailable saved sample panels**. Their cells explicitly retain original "
              "logged-metric gate regrades; score-history GIFs are not sample trajectories. The other 384 saved sample observations "
              "per transfer cohort regrade exactly in the supplied audits. Full-run CUDA packets and contemporaneous causal "
              "gradient/reference histories were not retained. Source/artifact hashes were verified; bulk archives remain local-only.", "",
              "Display coverage changes only. Private prototype implementations are not public API releases or adopted defaults; "
              "selected family rows, qualification, experiments and training code remain unchanged.", "", "</details>", "", ""]
    return "\n".join(lines)


def _render_astra(astra, previous, link):
    """Keep the cumulative packet and paid controls separate from the earlier ledger."""
    pin, document = astra
    summary = document["discovery_summary"]
    cells = document["cells"]
    old = previous[1]["discovery_summary"]
    candidates = [c for c in cells if c["result"]["round"].startswith("astra-") and c["result"]["role"] == "candidate"]
    controls = [c for c in cells if c["result"]["round"].startswith("astra-") and c["result"]["role"] == "control"]
    attempts = {c["result"]["attempt_id"] for c in cells}
    predictor_evals = sum(c["result"]["additional_temporary_predictor_evaluations"] for c in cells)
    updates = sum(c["result"]["new_optimizer_updates"] for c in cells)
    queue = sum(c["result"]["queue_charged_seconds"] for c in cells)
    wall = sum(c["result"]["actual_Popen_to_exit_seconds"] for c in cells)
    sources = {document["condition_contracts"][c["conditions_sha256"]]["source_digest"] for c in cells}
    identities = {(c["candidate_id"], c["candidate_revision"], document["condition_contracts"][c["conditions_sha256"]]["source_digest"])
                  for c in cells}
    if (len(cells) != summary["full_runs"] or len(attempts) != len(cells)
            or not all(c["full_protocol_complete"] is True for c in cells)
            or dict(Counter(c["status"] for c in cells)) != summary["status_counts"]
            or len(candidates) != summary["candidate_hypotheses"] or len(controls) != summary["fresh_astra_controls"]
            or any(c["status"] != "FAIL" for c in candidates)
            or set(summary["interim_attempts_reused_once"]) - attempts
            or predictor_evals != summary["additional_temporary_predictor_evaluations"]
            or updates != summary["new_updates"]
            or len(sources) != summary["source_cohorts"] or len(identities) != summary["source_bound_revisions"]
            or summary["family_full_runs"] != old["full_run_count"] + len(cells)
            or summary["family_new_updates"] != old["new_full_updates"] + updates
            or abs(queue - summary["all_24_queue_charge_seconds"]) > 1e-7
            or abs(wall - summary["all_24_supervisor_wall_seconds"]) > 1e-7
            or summary["global_survivor"] is not False):
        raise ValueError("test discoveries: Astra cumulative counts or costs differ from original cells")
    for round in summary["rounds"]:
        rows = [c for c in cells if c["result"]["round"] == round["id"]]
        if (len(rows) != round["runs"] or dict(Counter(c["status"] for c in rows)) != round["status_counts"]
                or abs(sum(c["result"]["actual_Popen_to_exit_seconds"] for c in rows) - round["supervisor_wall_seconds"]) > 1e-7
                or abs(sum(c["result"]["queue_charged_seconds"] for c in rows) - round["queue_charge_seconds"]) > 1e-7):
            raise ValueError("test discoveries: Astra round differs from original cells")
    lines = ["", "## Quota-transition and Astra portfolio follow-up", "",
             f"[Verified cumulative receipts and original submetrics]({link(pin['path'])}): **{len(cells)} additional full runs**, "
             f"{summary['predecessor_runs']} predecessor quota runs plus {summary['astra_runs']} Astra runs. "
             f"The six interim Astra receipts occur once. Family execution ledger: **{summary['family_full_runs']} full runs** "
             f"and **{summary['family_new_updates']:,} new accepted updates**; no global survivor.", "",
             "**Finding:** all twelve Astra hypotheses fail full unequal-mass gates. Three endpoint covariance submetrics "
             "pass, but their 1/5 terminal suffixes still fail the original sustained requirement.", "",
             "<details>", "<summary>Open the 24-run follow-up: controls, predictors and separate cost clocks</summary>", "",
             "| Round | Full PASS / FAIL | Source | Queue seconds | Supervisor wall seconds |",
             "| --- | ---: | --- | ---: | ---: |"]
    for round in summary["rounds"]:
        lines.append(f"| {_text(round['id'])} | {round['status_counts'].get('PASS',0)} / {round['status_counts'].get('FAIL',0)} | "
                     f"{', '.join('`'+s[:12]+'`' for s in round['source_digests'])} | {round['queue_charge_seconds']:.9f} | {round['supervisor_wall_seconds']:.9f} |")
    lines += ["", "All **12 Astra hypotheses FAIL** the unchanged full unequal-mass gate. Four fresh source-bound controls "
              "are separate paid runs. Swap, slow covariance and lookahead pass the final covariance submetric, but each "
              "ends with a one-check suffix against five required checks; their full verdict stays FAIL. "
              "The predecessor minimum-change recipe passes Gaussian acquisition/stability and mode retention, but fails unequal mass.", "",
              f"Added work: **{updates:,} accepted updates** plus **{predictor_evals:,} temporary predictor evaluations**. "
              "Lookahead's temporary predictor is not another accepted optimizer update; its work is already included in both clocks. "
              "Original horizons, budgets, quotas, assignment/transport semantics, reference policies and covariance-memory changes remain bound.", "",
              f"The 24-run queue charge is **{queue:.9f} seconds**; inclusive supervisor wall charge is **{wall:.9f} seconds**. "
              "These clocks are distinct and are not added together. Inherited budgets and earlier charges are not counted again. "
              "Preparation errors and predecessor failed-audit CPU costs remain diagnostic exclusions; unavailable preparation elapsed times stay null.", "",
              f"{len(sources)} additional source cohorts and all saved-output/state/trace pins were verified. Fractional transport uses "
              "row-argmax labels with exact soft capacities; integer assignment counts are not substituted for those labels. "
              "Controls' model equality is certified by the supplied audits. Raw arrays, tensor states and traces stay local-only; "
              "the import runs no research code, training or metric rescore and grants no qualification.", "", "</details>", "", ""]
    return "\n".join(lines)


def _render_confidence_packets(packets, link):
    if not packets:
        return ""
    if set(packets) != {"screen", "retention", "finite_cap", "orientation", "signed", "positive"}:
        raise ValueError("test discoveries: complete confidence publication requires all six packets")
    return (_render_confidence_followup(packets["screen"], packets["retention"], link)
            + _render_corrective_gaussian(packets["finite_cap"], packets["orientation"], link)
            + _render_response_variants(packets["signed"], packets["positive"], link))


def _render_confidence_followup(confidence, retention, link):
    """Report the complete six-task screen and its own retention failure separately."""
    if confidence is None or retention is None:
        raise ValueError("test discoveries: confidence-LR screen and retention receipts must be indexed together")
    for item, count, statuses, screen_pass in (
            (confidence, 6, {"PASS": 6}, True),
            (retention, 2, {"PASS": 1, "FAIL": 1}, False)):
        document = item[1]
        cells = document["cells"]
        summary = document["discovery_summary"]
        if (len(cells) != count or summary["attempts"] != count
                or dict(Counter(c["status"] for c in cells)) != statuses
                or summary["status_counts"] != statuses
                or summary["original_six_task_tier1_screen_pass"] is not screen_pass
                or summary["global_survivor"] is not False
                or len({c["result"]["attempt_id"] for c in cells}) != count
                or any(c["full_protocol_complete"] is not True for c in cells)
                or summary["source_packet_sha256"] != document["original_evidence"]["packet_sha256"]):
            raise ValueError("test discoveries: confidence-LR packet differs from original cells")
    all_cells = confidence[1]["cells"] + retention[1]["cells"]
    sources = {item[1]["condition_contracts"][c["conditions_sha256"]]["source_digest"]
               for item in (confidence, retention) for c in item[1]["cells"]}
    if (len({c["result"]["attempt_id"] for c in all_cells}) != 8
            or len(sources) != 1
            or len({c["candidate_revision"] for c in all_cells}) != 1):
        raise ValueError("test discoveries: confidence-LR attempts or source bindings overlap")
    return "\n".join(["", "## Confidence-LR correction: six-task screen and own retention", "",
        f"[Six original required Tier 1 attempts]({link(confidence[0]['path'])}) are **6/6 PASS** with complete actual "
        "optimizer histories. The positive confidence-scaled learning-rate response preserves generator, encoder "
        "and prior updates. This passes only the frozen six-task screening view; earlier hard-EMA and control "
        "attempts retain their separate source identities and are not relabelled as current-source controls.", "",
        f"[Two own-checkpoint continuations]({link(retention[0]['path'])}) keep the contrast: "
        "original word hold **PASS 25/25**, but Gaussian retention **FAIL** with 8/72 stationary checks, "
        "0/24 shifted-hold checks and failed reacquisition. An acquisition pass is not sustained retention. "
        "The measured recipe has no broader Tier 2, full Martyn, robustness or release qualification.", "",
        "The cumulative packet now has **74 unique attempts**: 66 earlier identities and eight new confidence-LR runs, "
        "**32 PASS / 35 FAIL / 7 INCOMPLETE**. The earlier rows and failed gates are byte-unchanged. "
        "This is a source-scoped attempt ledger, not a pooled score or an addition to the 249-run non-Atlas total. "
        "Exact cumulative updates remain unavailable because the older timed-out word control has only a lower bound. "
        "No default, qualification or release promotion follows.", ""])


def _render_corrective_gaussian(finite_cap, orientation, link):
    """Keep a failed acquisition and a failed own retention as separate outcomes."""
    if finite_cap is None or orientation is None:
        raise ValueError("test discoveries: both corrective Gaussian packets are required")
    for item, count, statuses in ((finite_cap, 1, {"FAIL": 1}),
                                  (orientation, 2, {"PASS": 1, "FAIL": 1})):
        document = item[1]
        cells = document["cells"]
        summary = document["discovery_summary"]
        if (len(cells) != count or summary["attempts"] != count
                or dict(Counter(c["status"] for c in cells)) != statuses
                or summary["status_counts"] != statuses
                or summary["global_survivor"] is not False
                or summary["original_six_task_tier1_screen_pass"] is not False
                or len({c["result"]["attempt_id"] for c in cells}) != count
                or any(c["full_protocol_complete"] is not True for c in cells)
                or summary["source_packet_sha256"] != document["original_evidence"]["packet_sha256"]):
            raise ValueError("test discoveries: corrective Gaussian cells differ from original packet")
    blocked = finite_cap[1]["discovery_summary"].get("unstarted_retention", {})
    if (blocked.get("status") != "BLOCKED_UNSTARTED" or blocked.get("attempts") != 0
            or blocked.get("new_updates") != 0 or blocked.get("charges") != 0
            or {c["task_id"] for c in finite_cap[1]["cells"]} != {"gaussian1d_smoke"}
            or {c["task_id"] for c in orientation[1]["cells"]} != {"gaussian1d_smoke", "gaussian1d_stability"}
            or len({c["result"]["attempt_id"] for item in (finite_cap, orientation)
                    for c in item[1]["cells"]}) != 3):
        raise ValueError("test discoveries: unstarted retention was promoted or correction attempts overlap")
    return "\n".join(["", "## Later Gaussian corrections: one failed acquisition, one failed retention", "",
        f"[Finite critic cap]({link(finite_cap[0]['path'])}) completes the original Gaussian acquisition task "
        "but **FAILS** its any-scheduled-pass-with-independent-confirmation gate. Own Gaussian retention is "
        "**BLOCKED/UNSTARTED**: zero attempts, updates and charge; there is no synthetic retention result.", "",
        f"[Critic orientation correction]({link(orientation[0]['path'])}) **PASSes** original Gaussian acquisition "
        "at its distinct source/revision, then **FAILS** the original own-state retention continuation: "
        "5/72 stationary, 1/24 shifted and 0/48 frozen checks, with failed reacquisition. Its remaining five "
        "required Tier 1 tasks were not run at this source. It cannot inherit the earlier confidence-LR "
        "six-task screen as current-source qualification.", "",
        "The cumulative ledger has **77 unique attempts: 33 PASS / 37 FAIL / 7 INCOMPLETE**. All earlier "
        "74 rows retain their exact grades and identities; these three attempts are added once. Neither correction "
        "establishes Gaussian retention, full six-task superiority, broad Tier 2 or release eligibility.", ""])


def _render_response_variants(signed, positive, link):
    """Preserve two acquisition passes and their separate full retention failures."""
    if signed is None or positive is None:
        raise ValueError("test discoveries: both response-variant packets are required")
    sources = set()
    revisions = set()
    attempts = set()
    for item, stationary, shifted in ((signed, 7, 0), (positive, 14, 2)):
        document = item[1]
        cells = document["cells"]
        summary = document["discovery_summary"]
        checks = summary.get("retention_checks", {})
        if (len(cells) != 2 or summary["attempts"] != 2
                or dict(Counter(c["status"] for c in cells)) != {"PASS": 1, "FAIL": 1}
                or summary["status_counts"] != {"PASS": 1, "FAIL": 1}
                or summary["global_survivor"] is not False
                or summary["original_six_task_tier1_screen_pass"] is not False
                or checks != {"stationary_passes": stationary, "stationary_checks": 72,
                              "shift_hold_passes": shifted, "shift_hold_checks": 24,
                              "frozen_passes": 0, "frozen_checks": 48, "reacquisition_status": "FAIL"}
                or {c["task_id"] for c in cells} != {"gaussian1d_smoke", "gaussian1d_stability"}
                or any(c["full_protocol_complete"] is not True for c in cells)
                or summary["source_packet_sha256"] != document["original_evidence"]["packet_sha256"]):
            raise ValueError("test discoveries: response-variant packet differs from original cells")
        sources.update(document["condition_contracts"][c["conditions_sha256"]]["source_digest"] for c in cells)
        revisions.update(c["candidate_revision"] for c in cells)
        attempts.update(c["result"]["attempt_id"] for c in cells)
    if len(sources) != 2 or len(revisions) != 2 or len(attempts) != 4:
        raise ValueError("test discoveries: response variants merged source identities or attempts")
    return "\n".join(["", "## Signed and positive-evidence responses: retention remains unsolved", "",
        f"[Signed-confidence response]({link(signed[0]['path'])}) **PASSes** original Gaussian acquisition, "
        "then **FAILs** own-checkpoint retention: 7/72 stationary, 0/24 shifted, 0/48 frozen and deadline "
        "reacquisition FAIL. The signed gap is a confidence proxy, not a restoring-gradient certificate.", "",
        f"[Positive-evidence response]({link(positive[0]['path'])}) also **PASSes** acquisition and **FAILs** "
        "full retention: 14/72 stationary, 2/24 shifted, 0/48 frozen and deadline reacquisition FAIL. "
        "Its final mean, width ratio and CDF KS meet numerical endpoint bounds, which do not replace the "
        "unchanged sustained gate. Neutral zero-gain proposals still advance real optimizer moment histories.", "",
        "These are two distinct source/revision cohorts; neither inherits the earlier six-task screen. "
        "The cumulative packet has **81 unique attempts: 35 PASS / 39 FAIL / 7 INCOMPLETE**, retaining all "
        "77 earlier rows exactly and adding four attempts once. No Gaussian-retention winner, full Martyn "
        "superiority, default or release qualification follows.", ""])


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def protocol_label(test):
    conditions = test["conditions"]
    if conditions.get("no_recorded_protocol"):
        return "unmeasured declaration"
    seed = conditions.get("seed", (conditions.get("protocol") or {}).get("seed", "unbound"))
    steps = (conditions.get("budget") or {}).get("steps", (test.get("contract") or {}).get("steps", "unbound"))
    return f"seed {seed} · {steps} updates · source {str(conditions.get('source_digest', conditions.get('source_identity', 'unbound')))[:10]}"


def selected_tests(report):
    ids = {recipe["id"] for recipe in report["recipes"] if recipe["selected"]}
    return [test for test in report["tests"] if test["lane"] == "Unmeasured declaration"
            or ids.intersection(test["cells"])]


def campaign_overview(report):
    """Display counts of already-indexed receipts; never a comparable suite score."""
    rows = [r for r in report["records"].values() if r.get("cell_id", "").startswith(("nonatlas-", "astra-"))
            and r.get("full_protocol_complete") is True]
    counts = Counter(r["status"] for r in rows)
    return {"full_runs": len(rows), "passes": counts["PASS"], "fails": counts["FAIL"]}


def diagnostic_group(receipt):
    if "pr399-confidence-lr-" in receipt:
        return "Confidence LR: acquisition and own retention"
    if any(name in receipt for name in ("pr399-gaussian-finite-cap-", "pr399-critic-orientation-", "pr399-signed-confidence-", "pr399-positive-evidence-")):
        return "Gaussian confidence corrections and retention"
    name = Path(receipt or "").name
    if name.startswith("astra-"):
        return "Quota transition and Astra portfolio"
    if name.startswith("nonatlas-finite"):
        return "Bounded mechanism prefixes"
    if name.startswith("nonatlas-"):
        return "Cell objectives, memory and transfer"
    if name.startswith("g-angle-"):
        return "G-angle diagnostics"
    if name.startswith("atlas-"):
        return "Atlas diagnostics"
    return "Other scoped evidence"


def render_markdown(report):
    recipes = [recipe for recipe in report["recipes"] if recipe["selected"]]
    tests = sorted(selected_tests(report), key=lambda test: (test["selected_coverage"]["optimizer_passes"], test["task_id"], test["id"]))
    overview = campaign_overview(report)
    astra_count = sum(r.get("cell_id", "").startswith("astra-") and r.get("result", {}).get("role") == "candidate"
                      and r.get("result", {}).get("round", "").startswith("astra-") for r in report["records"].values())
    lines = ["# Which recipes pass each test?", "",
             "[Generate/open the sortable offline matrix](../../docs/forge-test-performance.md) · "
             "[Family leaderboard](technique-inventory.md)", "",
             "One row is **one test under one exact protocol**. One column is a complete recipe, source and runtime identity. "
             "Different seeds, budgets, sampling laws, initializers and original gates stay separate.", "",
             "[How to read a row](#how-to-read-a-row) · [Selected-recipe matrix](#selected-recipe-matrix) · "
             "[Research cohorts](#research-cohorts) · [Limitations](#limitations)", "",
             "## Findings at a glance", "",
             f"- Latest indexed research: **{overview['full_runs']} full runs**, with **{overview['passes']} PASS / {overview['fails']} FAIL** "
             "under their separate original protocols. **No global winner** is established; these are ledger counts, not a pooled score.",
             ("- All twelve Astra hypotheses fail the full unequal-mass gate. Three final covariance submetrics pass, but their 1/5 "
              "terminal suffixes still fail the original sustained requirement." if astra_count == 12 else
              "- A passing endpoint submetric does not override a failed full or sustained gate."),
             "- Zero passing recipes is an evidence gap to investigate, not proof that a test is impossible. "
             "Oracle and representational witnesses are counted separately; none is currently registered.", "",
             "## How to read a row", "",
             "**4/5 passing recipes** means four of the five matching recipe identities have a compatible PASS receipt. "
             "It does not mean four independent seeds, and it does not borrow passes from another protocol. "
             "Open the cell for its original gate and evidence; open a column for the whole recipe.", "",
             "| Cell | Meaning |", "| --- | --- |",
             "| PASS | The original full scientific gate passes. |",
             "| FAIL | The original full gate fails, even if an endpoint submetric passes. |",
             "| NOT RUN | No compatible indexed run exists; this is not a failed experiment. |",
             "| ERROR | Execution failed, including failures before training. |",
             "| INCOMPLETE | Evidence stops before the full required protocol. |",
             "| BLOCKED / INVALID | Original admission or contract failure; reasons remain in the receipt. |",
             "| MIXED | Compatible receipts disagree; every outcome remains visible. |", "",
             "## Selected-recipe matrix", "",
             f"Indexed: **{len(report['tests'])} test/protocol rows**, **{len(report['recipes'])} complete recipe/source/runtime identities**; "
             f"below: **{len(tests)} rows**, **{len(recipes)} selected identities**. PASS counts recipes with any compatible passing receipt; MIXED keeps all disagreeing outcomes. Counts are recipe cells, not independent runs or summed view requirements.", "",
             "<details>", f"<summary>Open all {len(tests)} selected-protocol rows and their exact recipe identities</summary>", "",
             "| Recipe | Full identity |", "| --- | --- |"]
    for index, recipe in enumerate(recipes, 1):
        identity = recipe["identity"]
        lines.append(f"| R{index}: {_text(recipe['label'])} | `{_text(identity.get('candidate_id', 'API'))}` · recipe `{str(identity.get('bindings', {}).get('recipe_sha256', 'unbound'))[:12]}` · [full identity](technique-inventory.json) |")
    lines += ["", "| Test / exact protocol | Definition | PASS / recipes | Witnesses | " + " | ".join(f"R{i}" for i in range(1, len(recipes) + 1)) + " |",
              "| --- | --- | ---: | --- | " + " | ".join("---" for _ in recipes) + " |"]
    for test in tests:
        count = test["selected_coverage"]
        witnesses = ("NO INDEXED WITNESS" if count["no_indexed_witness"] else
                     f"optimizer {count['optimizer_passes']}, oracle {count['oracle_passes']}, representation {count['representational_passes']}")
        cells = []
        for recipe in recipes:
            records = [report["records"][key] for key in test["cells"].get(recipe["id"], [])]
            status = cell_status(records)
            targets = list(dict.fromkeys(record.get("receipt") or record.get("publication") for record in records))
            targets = [target for target in targets if target]
            cells.append(" · ".join(f"[{status if i == 0 else 'receipt ' + str(i+1)}]({relpath(target, 'reports/forge')})"
                                    for i, target in enumerate(targets)) or status)
        definition = (f"[{_text(test['definition_match'])}]({relpath(test['definition'], 'reports/forge')})"
                      if test.get('definition') else _text(test['definition_match']))
        lines.append(f"| {_text(test['task_id'])}<br>{_text(protocol_label(test))}<br>`{test['id'][:12]}` | {definition} | {count['optimizer_passes']}/{len(recipes)} | {witnesses} | " + " | ".join(cells) + " |")
    lines += ["", "</details>", "", "## Research cohorts", "",
              "Use the offline view's cohort menu to inspect one study at a time. Each row below keeps its original "
              "source, seed, budget and gate; these diagnostic passes grant no suite qualification."]
    diagnostics = [test for test in report["tests"] if test["lane"] == "Scoped diagnostic"]
    if diagnostics:
        groups = {}
        recipe_map = {recipe["id"]: recipe for recipe in report["recipes"]}
        for test in diagnostics:
            for recipe_id, keys in test["cells"].items():
                for key in keys:
                    record = report["records"][key]
                    conditions = test["conditions"]
                    group = diagnostic_group(record["receipt"])
                    groups.setdefault(group, []).append(f"| {_text(test['task_id'])} | {_text(recipe_map[recipe_id]['label'])} | "
                                 f"{conditions['seed']} | {conditions['budget']['steps']} | {record['status']} | "
                                 f"[full recipe, gates and original pins]({relpath(record['receipt'], 'reports/forge')}) |")
        for group, rows in groups.items():
            lines += ["", "<details>", f"<summary>{group} · {len(rows)} evidence records</summary>", "",
                      "| Test | Complete recipe | Seed | Declared updates | Original result | Evidence |",
                      "| --- | --- | ---: | ---: | --- | --- |", *rows, "", "</details>"]
    lines += ["", "## Limitations", "",
              "**No global winner or qualification change.** Six mode-hold transfers lack saved sample panels and retain "
              "logged-metric FAIL regrades. Bulk source, samples, tensors and traces remain local-only. Full-run CUDA packets "
              "and contemporaneous causal histories were not retained. Missing preparation timings stay unknown, not zero.", "",
              "NO INDEXED WITNESS means no compatible passing optimizer, oracle or representational evidence is registered "
              "for that row; unindexed evidence remains unknown. A row with no graded result is different from a row of complete FAILs.", "",
              "## Generate and verify", "", "Regenerate with `python -m experiments.forge.test_performance` or the existing "
              "`python reports/forge/regenerate_technique_inventory.py` workflow. "
              "`python -m experiments.forge.test_performance --check` verifies deterministic generated bytes without writing or training.", "",
              "Witnesses require an explicit byte-pinned `test-witnesses.json` registration with the exact row conditions, type, numerical "
              "verdict pointer and scope. API variants never satisfy a Forge row by matching a name. "
              "See [report contract and evidence gaps](../../docs/forge-test-performance.md).", ""]
    return "\n".join(lines)


def render_html(report):
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    overview = campaign_overview(report)
    overview_text = (f"{overview['full_runs']} full research runs · {overview['passes']} PASS / {overview['fails']} FAIL "
                     "under separate protocols. No global winner.")
    witness_note = ("Oracle and representational witnesses are counted separately; see each row." if any(t["witnesses"] for t in report["tests"])
                    else "No oracle or representational witness is currently registered.")
    return HTML.replace("__DATA__", data).replace("__DIGEST__", html.escape(report["input_digest"])).replace("__OVERVIEW__", overview_text).replace("__WITNESS_NOTE__", witness_note)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParticleGAN · Performance by test</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182a35;background:#f3f6f8;font-size:14px}*{box-sizing:border-box}body{margin:0}main{max-width:1700px;margin:auto;padding:32px}a{color:#14577d}h1{font-size:32px;margin:8px 0 12px;letter-spacing:-.8px}h2{font-size:20px}p{line-height:1.6;margin:8px 0}.eyebrow{color:#516a7a;letter-spacing:1.7px;text-transform:uppercase;font-size:12px;font-weight:700}.intro{max-width:1080px}.note{background:#fff7e4;border-left:4px solid #bd8424;padding:12px 16px;margin:20px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;padding:18px;background:white;border:1px solid #d6e0e6;border-radius:8px;margin-top:20px}label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#48606f;font-weight:650}select,input,button{font:inherit;border:1px solid #b7c9d5;border-radius:5px;background:#fff;padding:9px;color:#182a35}input{min-width:205px}button{cursor:pointer}button:hover{background:#edf4f8}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #1684c3;outline-offset:2px}button:disabled{opacity:.45;cursor:default}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.stat{background:white;border:1px solid #d6e0e6;border-radius:8px;padding:14px 20px;min-width:165px}.stat b{font-size:25px;display:block}.stat span{font-size:12px;color:#526b7a}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.badge{font-size:11px;font-weight:750;border-radius:4px;padding:5px 7px;display:inline-block;white-space:nowrap}.PASS{color:#11603f;background:#daf1e5}.FAIL{color:#962f37;background:#fce4e4}.ERROR{color:#8d451a;background:#ffe4d0}.INCOMPLETE{color:#745405;background:#fff0bd}.NOT_RUN,.BLOCKED,.INVALID,.UNASSESSED,.UNCONFIRMED{color:#52626c;background:#eaf0f3}.MIXED{color:#654187;background:#efdef9}.pager{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.pager span{font-size:12px;color:#526b7a}.table-wrap{max-height:650px;overflow:auto;border:1px solid #c5d4dd;border-radius:8px;background:white}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}th,td{padding:10px;border-bottom:1px solid #dbe5eb;border-right:1px solid #e9eff3;vertical-align:top;min-width:96px}thead th{position:sticky;top:0;z-index:3;background:#eaf1f5;text-align:left;height:105px}th small,td small{display:block;color:#597180;font-weight:400;line-height:1.5}tbody th{position:sticky;left:0;z-index:2;min-width:250px;max-width:290px;background:#fff;text-align:left}thead th:first-child{left:0;z-index:4;min-width:250px}tr.zero th{border-left:4px solid #bd8424}td .badge{border:0}td .detail{padding:4px;background:transparent;border:0;text-align:left}thead button{padding:0;border:0;background:transparent;text-align:left;max-width:145px;font-weight:700;overflow-wrap:anywhere}tbody th button{padding:0;border:0;background:transparent;font-weight:700;text-align:left;overflow-wrap:anywhere}.empty{padding:35px;text-align:center;color:#526b7a}footer{font-size:12px;color:#597180;margin-top:24px;overflow-wrap:anywhere}dialog{border:1px solid #bed0dc;border-radius:10px;padding:24px;max-width:950px;width:calc(100% - 36px);max-height:88vh;box-shadow:0 12px 90px #182a3544}dialog::backdrop{background:#182a3577}.close{float:right}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f9;border-radius:5px;padding:14px;font-size:12px;line-height:1.5}details{margin:12px 0}summary{cursor:pointer;font-weight:650;padding:8px 0}.links{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.warning{color:#945817}noscript{display:block;padding:20px;background:#fff0bd}@media(max-width:650px){main{padding:18px}h1{font-size:27px}.stat{min-width:140px;flex:1}tbody th,thead th:first-child{min-width:190px;max-width:210px}.controls label{width:100%}input{width:100%}}
/* Reader-facing layout; the evidence payload and comparison keys stay unchanged. */
main{max-width:1480px}h1{font-size:36px}.intro{max-width:1000px}.links{font-size:13px}
.overview{display:grid;grid-template-columns:1.5fr 1fr;gap:16px;margin:22px 0}.panel{background:#fff;border:1px solid #d6e0e6;border-radius:10px;padding:18px 20px}.panel h2{margin:0 0 8px;font-size:17px}.panel p{font-size:14px}.panel ul{margin:10px 0 0;padding-left:20px;line-height:1.6}.panel li{margin:4px 0}.panel.limitations{background:#fff9ec;border-color:#e3cf9f}.overview-line{font-weight:650}.jump-links{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.jump-links button{font-size:13px;padding:8px 12px}.guide{background:#fff;border:1px solid #d6e0e6;border-radius:8px;padding:10px 15px;margin:12px 0}.guide summary{padding:5px 0}.guide p{max-width:1050px}.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.controls input{width:100%;min-width:0}.stats{margin:16px 0}.stat{padding:12px 18px;min-width:180px}.legend{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.legend-item{padding:10px 12px;background:#fff;border:1px solid #d6e0e6;border-radius:7px}.legend-item small{display:block;margin-top:6px;font-size:12px;line-height:1.4;color:#526b7a}.legend-other{margin:10px 0 16px;font-size:12px;color:#526b7a}.table-wrap{max-height:620px}table{font-size:13px}thead th{height:105px;min-width:145px}thead th:first-child,tbody th{min-width:235px;max-width:270px}.exact-id{font-family:ui-monospace,monospace;font-size:10px;overflow-wrap:anywhere}.recipe-meta summary{font-size:11px;color:#14577d;font-weight:500;padding:3px 0}.recipe-meta{margin:6px 0}.recipe-meta small{font-size:11px}.coverage-state{font-weight:650;margin-bottom:6px}.pass-count{font-weight:700;font-size:16px}.protocol-detail{padding:10px 0;font-size:12px}.definition-note{font-size:11px;color:#657784}.evidence-summary{background:#f5f8fa;border:1px solid #d9e4ea;border-radius:8px;padding:12px 16px;margin:12px 0}.evidence-summary p{margin:3px 0}.metric-table{font-size:12px;margin:12px 0;table-layout:fixed}.metric-table th,.metric-table td{min-width:0;padding:8px}.metric-table th{position:static;height:auto;background:#eef4f7}.metric-table code{font-size:11px}.summary-note{font-size:12px;color:#526b7a}dialog h2{margin:4px 0 8px}dialog .links{margin:8px 0}dialog details{margin:8px 0}.warning{background:#fff5df;border-left:3px solid #bd8424;padding:9px 12px}
@media(max-width:800px){.overview{grid-template-columns:1fr}.legend{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:650px){h1{font-size:28px}.overview{gap:10px;margin:16px 0}.panel{padding:14px}.controls{grid-template-columns:repeat(2,minmax(0,1fr));padding:14px;gap:12px}.controls label{width:auto}.controls label:first-child,.controls label:nth-child(3),.controls label:nth-child(4){grid-column:1/-1}.legend{grid-template-columns:repeat(2,minmax(0,1fr))}.legend-item{padding:9px}.stat{min-width:140px;padding:11px 14px}.table-wrap{max-height:560px}thead th:first-child,tbody th{min-width:170px;max-width:200px}dialog{padding:16px}.metric-table{font-size:11px}.jump-links button{flex:1}.exact-id{font-size:9px}}
</style></head><body><main>
<div class="eyebrow">ParticleGAN · Forge research evidence</div>
<h1>Which recipes pass each test?</h1>
<div class="intro"><p>See which <strong>complete recipes</strong> pass, fail or lack evidence. Each row is one test under one exact protocol; different seeds, budgets, sources and sampling laws stay separate.</p>
<div class="links"><a href="technique-inventory.md">Family leaderboard</a><a href="test-performance.md">Readable Markdown matrix</a><a href="test-performance.json">Projection JSON</a><a href="../../docs/forge-test-performance.md">Contract &amp; evidence gaps</a></div></div>
<div class="overview">
<section class="panel"><h2>What the indexed research shows</h2><p class="overview-line">__OVERVIEW__</p><p>Individual successes do not establish a recipe that wins the whole suite. A passing endpoint metric can still end in a full FAIL when sustained checks are missing.</p><div class="jump-links"><button id="jumpResearch">Browse research cohorts</button><button id="jumpZero">Find tests with no PASS</button><button id="jumpErrors">Find execution errors</button></div></section>
<aside class="panel limitations"><h2>Read the limits first</h2><p><strong>Zero optimizer passes is a smell, not proof of impossibility.</strong> __WITNESS_NOTE__ Some sample panels and causal histories are unavailable; raw archives remain local-only. Each cell exposes its gaps. These rows grant no qualification.</p></aside>
</div>
<details class="guide" id="rowGuide"><summary>How to read a row: passing, failed and missing evidence</summary><p><strong>4 / 5 passing recipes</strong> means four of the five matching recipe identities have at least one compatible PASS receipt. It does not mean four independent seeds. <strong>No graded result</strong> means there is no complete PASS/FAIL evidence; <strong>failures only</strong> means completed gates failed. Click a cell for the original gate, completion, costs and provenance. Click a recipe name for its full configuration.</p><p>A final covariance PASS with a <strong>1/5 terminal suffix</strong> is still a full FAIL. NOT RUN means no compatible indexed result, not that an experiment failed. MIXED keeps conflicting receipts visible and counts a recipe as passing if any compatible receipt passes.</p></details>
<noscript>Open <a href="test-performance.md">the Markdown matrix</a> for a readable view without JavaScript.</noscript>
<div class="controls" id="explore">
<label>Evidence scope<select id="scope"><option value="selected">Selected Forge recipes</option><option value="diagnostics">Research diagnostics</option><option value="forge">All measured Forge history</option><option value="api">Public-API protocols</option><option value="all">All indexed evidence</option></select></label>
<label>Research cohort<select id="cohort"><option value="all">All cohorts</option><option value="cell">Cell objectives / memory / transfer</option><option value="astra">Quota transition / Astra</option><option value="atlas">Atlas</option><option value="g-angle">G-angle</option><option value="finite">Bounded mechanism prefixes</option></select></label>
<label>Find a test / protocol<input id="search" type="search" placeholder="e.g. unequal masses or source ID"></label>
<label>Find a recipe / source<input id="recipe" type="search" placeholder="e.g. lookahead, dualnorm or revision"></label>
<label>Evidence filter<select id="filter"><option value="all">All tests</option><option value="zero">Zero optimizer passes</option><option value="witness">No indexed witness</option><option value="error">Execution errors</option><option value="incomplete">Incomplete protocols</option></select></label>
<label>Definition<select id="definition"><option value="all">All original protocols</option><option value="matches">Matches current declaration</option><option value="CHANGED">Changed since execution</option></select></label>
<label>Sort tests<select id="sort"><option value="passes">Fewest passing recipes</option><option value="name">Test name</option><option value="coverage">Least graded coverage</option><option value="errors">Most execution errors</option></select></label>
</div>
<div class="stats" id="stats" aria-live="polite"></div>
<div class="legend" id="legend"></div>
<div class="legend-other"><strong>BLOCKED / INVALID:</strong> original admission or contract failure. <strong>MIXED:</strong> compatible receipts disagree; all remain visible. Other recorded statuses remain explicit in the receipt.</div>
<div class="pager"><button id="prevCols">Previous recipes</button><button id="nextCols">Next recipes</button><span id="columns"></span></div>
<div class="table-wrap"><table id="matrix"><caption id="caption" style="text-align:left;padding:12px">Each row is one exact protocol. Counts cover every matching recipe, including columns on another page. Click a result for its evidence.</caption><thead></thead><tbody></tbody></table><div class="empty" id="empty" hidden>No compatible rows match these filters. Clear a search or choose another scope/cohort.</div></div>
<div class="pager"><button id="prevRows">Previous tests</button><button id="nextRows">Next tests</button><span id="rows"></span></div>
<footer>Read-only projection · no training or regrading · input digest <code>__DIGEST__</code><br>NOT RUN means no compatible indexed result. BLOCKED/INVALID retain original admission or contract reasons. MIXED retains all disagreeing receipts.</footer>
</main><dialog id="detail"><button class="close" id="close">Close</button><div id="detailBody"></div></dialog>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';
const report=JSON.parse(document.getElementById('evidence').textContent), byRecipe=new Map(report.recipes.map(r=>[r.id,r])), byTest=new Map(report.tests.map(t=>[t.id,t]));
const $=id=>document.getElementById(id), scope=$('scope'), cohort=$('cohort'), search=$('search'), recipeSearch=$('recipe'), filter=$('filter'), definition=$('definition'), sort=$('sort');
let rowPage=0,colPage=0; const rowSize=60,colSize=12;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function records(test,id){return (test.cells[id]||[]).map(key=>report.records[key]);}
function status(test,id){const s=[...new Set(records(test,id).map(r=>r.status))];return s.length===1?s[0]:s.length?'MIXED':'NOT RUN';}
function badge(s){return el('span',s,'badge '+s.replaceAll(' ','_'));}
function counts(test,recipes){const c={passing:0,graded:0};recipes.forEach(r=>{const s=status(test,r.id),list=records(test,r.id);c[s]=(c[s]||0)+1;if(list.some(x=>x.status==='PASS'))c.passing++;if(list.some(x=>x.status==='PASS'||x.status==='FAIL'))c.graded++;});return c;}
function protocolLabel(t){const c=t.conditions;if(c.no_recorded_protocol)return 'No recorded execution protocol';const seed=c.seed??c.protocol?.seed??'unbound',steps=c.budget?.steps??t.contract?.steps??'unbound';return 'Seed '+seed+' · '+steps+' updates · source '+String(c.source_digest||c.source_identity||'unbound').slice(0,10);}
function hasWitness(test){return test.witnesses.some(w=>w.status==='PASS');}
const taskNames={vector_anisotropic:'Anisotropic components',vector_unequal_width:'Unequal component widths',vector_unequal_width_published:'Unequal widths · published architecture',vector_unequal_mass:'Unequal component masses',vector_overlap:'Overlapping components',vector_two_broad:'Two broad components',vector_spiral:'Spiral distribution',gaussian1d_smoke:'Gaussian acquisition',gaussian1d_stability:'Gaussian stability',mode_hold:'Mode retention',ring16_acquisition:'16-mode ring acquisition',staggered100:'Staggered 100-mode distribution'};
const recipeNames={'cell-extragradient-map-v1':'Temporary lookahead','cell-slow-covariance-v1':'Slow covariance memory','cell-quota-aggregate-swap-v1':'Aggregate-loss swap','cell-fractional-capacity-forward-kl-g-v1':'Fractional capacities','cell-quota-minimum-change-forward-kl-g-v1':'Minimum-change quotas','cell-quota-equal-shape-v1':'Equal-cell shape weighting','cell-worst-shape-v1':'Worst-cell shape weighting','cell-projected-cells-v1':'Projected cell gradients','cell-minimum-switch-v1':'Minimum assignment switches','cell-conditional-rbf-v1':'Conditional RBF shape','cell-conditional-sliced-v1':'Conditional sliced shape','cell-conditional-fourth-v1':'Conditional fourth moments','cell-quota-prior-signal-v1':'Generator + prior attenuation'};
function taskName(t){if(taskNames[t.task_id])return taskNames[t.task_id];const prefix='recipe-prior-refactor-legacy-';if(t.task_id.startsWith(prefix)){const base=t.task_id.slice(prefix.length);return (taskNames[base]||base.replaceAll('_',' '))+' · legacy prior fixture';}return t.task_id.replaceAll('_',' ').replaceAll('-',' ');}
function recipeName(r){if(recipeNames[r.identity?.candidate_id])return recipeNames[r.identity.candidate_id];const v=r.recipe?.declared_global_recipe||{},objective={aggregate_forward_kl:'Forward KL',aggregate_reverse_kl:'Reverse KL',aggregate_bures:'Bures',aggregate_moment:'Quadratic moments',per_kernel:'Per-kernel moments'}[v.cell_gaussian_objective];if(v.optimizer_role_families&&objective){const names={adam:'Adam',nsgda_global:'NSGDA'},roles=v.optimizer_role_families;return objective+' · G '+(names[roles.generator]||roles.generator)+' / D '+(names[roles.critic]||roles.critic)+' / P '+(names[roles.prior]||roles.prior);}if(r.label.startsWith('cell-gradient-signal-')&&objective)return objective+' · '+(v.optimizer_signal_roles||[]).map(x=>x==='generator'?'G':x==='prior'?'P':x).join('+')+' attenuation';if(r.label.startsWith('cell-memory-')&&objective)return objective+' · '+v.cell_reference_policy;return r.label;}
function cohortOf(rec){const p=rec.receipt||'';return p.includes('astra-cumulative')?'astra':p.includes('nonatlas-finite')?'finite':p.includes('nonatlas-')?'cell':p.includes('atlas-location')?'atlas':p.includes('g-angle-')?'g-angle':'other';}
function cohortMatches(t,id){return cohort.value==='all'||records(t,id).some(rec=>cohortOf(rec)===cohort.value);}
function matchesQuery(q,value,friendly){q=q.trim();const text=JSON.stringify(value).toLowerCase();if(!q||text.includes(q))return true;const words=q.match(/[a-z0-9_]+/g)||[],labels=(friendly||'').toLowerCase().match(/[a-z0-9_]+/g)||[];return words.length>1&&words.every(word=>labels.includes(word));}
function jsonBlock(title,value,parent){const d=el('details'),summary=el('summary',title);d.append(summary,el('pre',JSON.stringify(value,null,2)));parent.append(d);}
function recipeHints(r){const v=r.recipe?.declared_global_recipe||r.recipe?.native_recipe||r.recipe?.complete_recipe||r.recipe||{},parts=[v.optimizer_family||r.lane];for(const [key,label] of [['cell_gaussian_objective','objective'],['cell_reference_policy','reference'],['cell_cubature_rule','cubature'],['cell_quota_policy','quota'],['cell_transport_policy','transport'],['cell_distribution_shape','shape'],['cell_covariance_memory','covariance memory'],['cell_assignment_refinement','assignment']])if(v[key]&&!['none','unchanged'].includes(v[key]))parts.push(label+': '+v[key]);if(v.cell_shape_weighting&&v.cell_shape_weighting!=='mass')parts.push('shape weighting: '+v.cell_shape_weighting);if(v.cell_gradient_projection)parts.push('projected cell gradients');if(v.cell_extragradient_map)parts.push('temporary predictor: 1 / accepted step');if(v.optimizer_role_families)parts.push('roles: '+Object.entries(v.optimizer_role_families).map(([k,v])=>k+'='+v).join(', '));if(v.optimizer_signal_roles?.length)parts.push('attenuated: '+v.optimizer_signal_roles.join(', '));return parts;}
function link(label,path,parent){if(!path||path.startsWith('/')||path.includes('..')||path.includes(':')||path.includes('\\'))return;const a=el('a',label);a.href='../../'+path;parent.append(a);}
function terminalNote(rec){const e=rec.result?.original_grade?.evaluator_result,c=e?.convergence,m=Array.isArray(e?.metrics)?e.metrics.find(x=>x.metric==='component_covariance_error'):null;return rec.status==='FAIL'&&m?.status==='PASS'&&c&&c.passing_suffix<c.minimum_stable_checks?'Final covariance submetric passes; terminal suffix '+c.passing_suffix+'/'+c.minimum_stable_checks+'. Full result: FAIL.':null;}
function showRecipe(id){const r=byRecipe.get(id);if(!r)return;const body=$('detailBody');body.replaceChildren(el('h2',recipeName(r)),el('p',r.label,'exact-id'),el('p','Whole recipe, executed source and runtime. This identity stays the same across test rows.'));const hints=el('div',undefined,'evidence-summary');recipeHints(r).forEach(h=>hints.append(el('p',h)));body.append(hints);jsonBlock('Full recipe',r.recipe,body);jsonBlock('Recipe identity and executed source',r.identity,body);$('detail').showModal();}
function metricSummary(rec,parent){const result=rec.result||{},e=result.original_grade?.evaluator_result,box=el('div',undefined,'evidence-summary');box.append(el('p',rec.full_protocol_complete===true?'Full recorded protocol completed':rec.status==='ERROR'?'Execution failed; no full scientific grade claimed':'Completion follows the original receipt'));
const steps=result.completed_steps??result.completed_updates;if(steps!==undefined)box.append(el('p','Completed clock: '+steps+' updates'+(result.new_optimizer_updates!==undefined?' · new work: '+result.new_optimizer_updates:'')+(result.reused_prefix_steps?' · restored prefix: '+result.reused_prefix_steps:'')));
if(result.additional_temporary_predictor_evaluations)box.append(el('p','Temporary predictor evaluations: '+result.additional_temporary_predictor_evaluations+' (included in costs; separate from accepted updates)'));
if(result.queue_charged_seconds!==undefined){const p=el('p','Queue charge: '+result.queue_charged_seconds.toFixed(9)+' s');p.title='Exact receipt value: '+result.queue_charged_seconds;box.append(p);}
if(result.actual_Popen_to_exit_seconds!==undefined){const p=el('p','Inclusive supervisor wall charge: '+result.actual_Popen_to_exit_seconds.toFixed(9)+' s');p.title='Exact receipt value: '+result.actual_Popen_to_exit_seconds;box.append(p);}
if(e?.convergence)box.append(el('p','Terminal passing suffix: '+e.convergence.passing_suffix+' / '+e.convergence.minimum_stable_checks+' required checks'));
parent.append(box);if(!Array.isArray(e?.metrics))return;const table=el('table',undefined,'metric-table'),head=el('tr');['Endpoint metric','Observed','Original gate','Result'].forEach(x=>head.append(el('th',x)));table.append(head);e.metrics.forEach(m=>{const tr=el('tr');tr.append(el('td',m.metric.replaceAll('_',' ')),el('td',typeof m.value==='number'?Number(m.value.toPrecision(6)).toString():String(m.value)),el('td',m.op+' '+m.threshold));const td=el('td');td.append(badge(m.status));tr.append(td);table.append(tr);});parent.append(table,el('p','Observed values are rounded here for reading. Exact values and full sustained gates remain in the original receipt below.','summary-note'));}
function showCell(testId,recipeId){const t=byTest.get(testId),r=byRecipe.get(recipeId);if(!t)return;const body=$('detailBody');body.replaceChildren(el('h2',taskName(t)),el('p',t.task_id,'exact-id'),el('p',r?recipeName(r)+' · '+status(t,r.id):'Exact test protocol'),el('p',protocolLabel(t),'protocol-detail'));
const links=el('div',undefined,'links');link('Current declaration (may differ)',t.definition,links);body.append(links);body.append(el('p','Definition: '+t.definition_match+'. Original recorded conditions remain authoritative.'));
if(r){jsonBlock('Complete recipe',r.recipe,body);jsonBlock('Recipe/source/runtime identity',r.identity,body);const list=records(t,r.id);if(!list.length)body.append(el('p','NOT RUN — no compatible indexed evidence for this recipe and exact protocol.'));
list.forEach((rec,i)=>{body.append(el('h3','Evidence '+(i+1)+' · '+rec.status));const ls=el('div',undefined,'links');link('Pinned compact receipt',rec.receipt,ls);link('Original publication',rec.publication,ls);body.append(ls);const note=terminalNote(rec);if(note)body.append(el('p',note,'warning'));if(rec.result?.audit_scope?.sample_panels_available===false)body.append(el('p','Audit gap: saved sample panels unavailable. This full gate retains the original logged-metric regrade; score-history GIFs are not sample trajectories.','warning'));metricSummary(rec,body);jsonBlock('Original gates, metrics, completion, errors and receipt hashes',rec,body);const catalog=report.evidence_catalogs?.[rec.original_evidence?.catalog_sha256];if(catalog)jsonBlock('Pinned source catalogs, receipt hashes and audit limitations',catalog,body);});}
jsonBlock('Exact protocol, seed/RNG and source bindings',t.conditions,body);jsonBlock('Recorded task contract (original evaluator fingerprint and budget)',t.contract,body);
jsonBlock('Oracle and representational feasibility witnesses',t.witnesses,body);body.append(el('p','No indexed witness is an evidence gap. It is not proof of impossibility. Oracle/representation passes cannot replace an optimizer run.'));$('detail').showModal();}
function render(){
const q=search.value.toLowerCase(),rq=recipeSearch.value.toLowerCase();let rs=report.recipes.filter(r=>(scope.value==='selected'?r.selected:scope.value==='diagnostics'?report.tests.some(t=>t.lane==='Scoped diagnostic'&&t.cells[r.id]):scope.value==='forge'?r.lane==='Forge':scope.value==='api'?r.lane==='API':true)&&matchesQuery(rq,[r,recipeName(r)],recipeName(r))&&(cohort.value==='all'||report.tests.some(t=>cohortMatches(t,r.id))));
let ts=report.tests.filter(t=>(scope.value==='selected'?t.lane==='Unmeasured declaration'||rs.some(r=>t.cells[r.id]):scope.value==='diagnostics'?t.lane==='Scoped diagnostic'&&rs.some(r=>t.cells[r.id]):scope.value==='forge'?t.lane!=='API'&&(t.lane==='Unmeasured declaration'||rs.some(r=>t.cells[r.id])):scope.value==='api'?t.lane==='API':true)&&matchesQuery(q,[t.task_id,taskName(t),t.id,t.conditions],taskName(t)+' '+protocolLabel(t))&&(definition.value==='all'||t.definition_match===definition.value)&&(cohort.value==='all'||rs.some(r=>cohortMatches(t,r.id)))).map(t=>({t,c:counts(t,rs)}));
ts=ts.filter(({t,c})=>filter.value==='zero'?!c.passing:filter.value==='witness'?!c.passing&&!hasWitness(t):filter.value==='error'?(c.ERROR||0)>0:filter.value==='incomplete'?(c.INCOMPLETE||0)>0:true);
ts.sort((a,b)=>{let d=sort.value==='passes'?a.c.passing-b.c.passing:sort.value==='coverage'?a.c.graded-b.c.graded:sort.value==='errors'?(b.c.ERROR||0)-(a.c.ERROR||0):0;return d||a.t.task_id.localeCompare(b.t.task_id)||a.t.id.localeCompare(b.t.id);});
rowPage=Math.min(rowPage,Math.max(0,Math.ceil(ts.length/rowSize)-1));colPage=Math.min(colPage,Math.max(0,Math.ceil(rs.length/colSize)-1));
const stats=$('stats');stats.replaceChildren();for(const [n,label] of [[ts.length,'test / protocol rows'],[rs.length,'recipe identities'],[ts.filter(x=>!x.c.passing).length,'zero optimizer passes'],[ts.filter(x=>!x.c.passing&&!hasWitness(x.t)).length,'no indexed witness']]){const div=el('div',undefined,'stat');div.append(el('b',n),el('span',label));stats.append(div);}
const cols=rs.slice(colPage*colSize,(colPage+1)*colSize),thead=$('matrix').querySelector('thead'),tbody=$('matrix').querySelector('tbody');thead.replaceChildren();tbody.replaceChildren();const head=el('tr');head.append(el('th','Test / exact protocol'),el('th','PASS / recipes'),el('th','Coverage & witnesses'));
cols.forEach(r=>{const th=el('th'),btn=el('button',recipeName(r));btn.title='Open complete recipe '+r.label+' · '+r.id;btn.onclick=()=>showRecipe(r.id);th.append(btn,el('small',r.identity?.candidate_id||r.label,'exact-id'));const hints=el('details',undefined,'recipe-meta');hints.append(el('summary','Recipe details · '+r.id.slice(0,8)));recipeHints(r).forEach(hint=>hints.append(el('small',hint)));th.append(hints);head.append(th);});thead.append(head);
ts.slice(rowPage*rowSize,(rowPage+1)*rowSize).forEach(({t,c})=>{const tr=el('tr',undefined,c.passing?'':'zero'),th=el('th'),btn=el('button',taskName(t));btn.onclick=()=>showCell(t.id);th.append(btn,el('small',t.task_id,'exact-id'),el('small',t.lane+' · '+t.id.slice(0,12)),el('small',protocolLabel(t)),el('small','Definition: '+t.definition_match,'definition-note'));tr.append(th,el('td',c.passing+' / '+rs.length,'pass-count'));const cov=el('td');cov.append(el('small',c.passing?'Passing evidence':c.graded?'Failures only':'No graded result','coverage-state'),el('small',c.graded+' graded · '+(c.ERROR||0)+' errors · '+(c.INCOMPLETE||0)+' incomplete'),el('small',(c['NOT RUN']||0)+' not run · '+(c.BLOCKED||0)+' blocked · '+(c.INVALID||0)+' invalid'));
const witnesses=t.witnesses.filter(w=>w.status==='PASS');cov.append(el('small','Oracle '+witnesses.filter(w=>w.kind==='oracle').length+' · representation '+witnesses.filter(w=>w.kind==='representational').length));if(!c.passing&&!witnesses.length)cov.append(el('small','NO INDEXED WITNESS','warning'));tr.append(cov);
cols.forEach(r=>{const td=el('td'),btn=el('button',undefined,'detail');btn.setAttribute('aria-label',t.task_id+' · '+r.label+' · '+status(t,r.id)+' · evidence');btn.append(badge(status(t,r.id)));btn.onclick=()=>showCell(t.id,r.id);td.append(btn);tr.append(td);});tbody.append(tr);});
$('empty').hidden=ts.length>0;$('columns').textContent='Recipes '+(rs.length?colPage*colSize+1:0)+'–'+Math.min(rs.length,(colPage+1)*colSize)+' of '+rs.length+'; counts include all '+rs.length+' matching identities.';$('rows').textContent='Tests '+(ts.length?rowPage*rowSize+1:0)+'–'+Math.min(ts.length,(rowPage+1)*rowSize)+' of '+ts.length;
$('prevCols').disabled=colPage===0;$('nextCols').disabled=(colPage+1)*colSize>=rs.length;$('prevRows').disabled=rowPage===0;$('nextRows').disabled=(rowPage+1)*rowSize>=ts.length;
}
for(const [s,meaning] of [['PASS','Original full gate passes'],['FAIL','Original full gate fails'],['NOT RUN','No compatible indexed run'],['ERROR','Execution failed'],['INCOMPLETE','Full protocol not completed']]){const item=el('div',undefined,'legend-item');item.append(badge(s),el('small',meaning));$('legend').append(item);}
[scope,cohort,search,recipeSearch,filter,definition,sort].forEach(control=>control.addEventListener(control.tagName==='INPUT'?'input':'change',()=>{if(control===scope&&scope.value==='selected')cohort.value='all';if(control===cohort&&cohort.value!=='all')scope.value='diagnostics';rowPage=0;colPage=0;render();}));
function navigate(s,c,f){scope.value=s;cohort.value=c;filter.value=f;search.value='';recipeSearch.value='';definition.value='all';sort.value='passes';rowPage=0;colPage=0;render();$('explore').scrollIntoView({block:'start'});}
$('jumpResearch').onclick=()=>navigate('diagnostics','all','all');$('jumpZero').onclick=()=>navigate('all','all','zero');$('jumpErrors').onclick=()=>navigate('forge','all','error');
$('prevCols').onclick=()=>{colPage--;render();};$('nextCols').onclick=()=>{colPage++;render();};$('prevRows').onclick=()=>{rowPage--;render();};$('nextRows').onclick=()=>{rowPage++;render();};$('close').onclick=()=>$('detail').close();
function openHash(){const hash=location.hash.slice(1);if(hash.startsWith('recipe-'))showRecipe(hash.slice(7));else if(hash.startsWith('cell-')){const ids=hash.slice(5).split('-');if(ids.length===2)showCell(ids[0],ids[1]);}}
render();openHash();window.addEventListener('hashchange',openHash);
</script></body></html>
'''
