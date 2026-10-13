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
    full = prefix = reconciled = None
    for pin in registry["publications"]:
        document = inputs.read(pin["path"], sha256=pin["sha256"])
        summary = document.get("discovery_summary", {})
        if summary.get("id") not in {"nonatlas-cell-moment-seed0-20261012", "nonatlas-finite-realization-prefix64", "nonatlas-reconciled-204-20261013"}:
            continue
        if (document.get("schema") != "forge_test_display_evidence_v1" or document.get("qualification_input") is not False
                or document.get("qualification_reuse") is not False or summary.get("ordinary_qualification") is not False
                or summary.get("promotion") is not False or portable(document) != document):
            raise ValueError("test discoveries: scoped evidence cannot promote or leak local references")
        if summary["id"] == "nonatlas-reconciled-204-20261013":
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
        return _render_reconciled(reconciled, full, prefix, registry, inputs, link)
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
    return "\n".join(lines) + "\n"


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
              "selected family rows, qualification, experiments and training code remain unchanged.", "", ""]
    return "\n".join(lines)


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


def render_markdown(report):
    recipes = [recipe for recipe in report["recipes"] if recipe["selected"]]
    tests = sorted(selected_tests(report), key=lambda test: (test["selected_coverage"]["optimizer_passes"], test["task_id"], test["id"]))
    lines = ["# Performance by test and complete recipe", "",
             "[Generate/open the sortable offline matrix](../../docs/forge-test-performance.md) · "
             "[Family leaderboard](technique-inventory.md)", "",
             "The table below shows selected recipes. The offline view also searches measured Forge history and public-API protocols. "
             "Each row binds one exact test and protocol, including seeds/RNG, source, runtime, initialization, sampling, budgets and original gates. "
             "Changed definitions remain separate; these counts do not rank techniques or select a different recipe for every task.", "",
             "**Zero optimizer passes is a smell, not proof of impossibility.** Oracle and representational witnesses are separate "
             "from optimizer passes. NO INDEXED WITNESS means none is registered for these exact conditions; unindexed evidence remains unknown.", "",
             "PASS and FAIL retain the original complete gate; ERROR is execution failure (including pre-training errors); "
             "INCOMPLETE lacks the full protocol; NOT RUN has no compatible indexed result. BLOCKED, INVALID and other recorded statuses "
             "remain explicit. MIXED exposes conflicting outcomes instead of picking a favorable one. Click any cell for receipts and metrics.", "",
             f"Indexed: **{len(report['tests'])} test/protocol rows**, **{len(report['recipes'])} complete recipe/source/runtime identities**; "
             f"below: **{len(tests)} rows**, **{len(recipes)} selected identities**. PASS counts recipes with any compatible passing receipt; MIXED keeps all disagreeing outcomes. Counts are recipe cells, not independent runs or summed view requirements.", "",
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
    diagnostics = [test for test in report["tests"] if test["lane"] == "Scoped diagnostic"]
    if diagnostics:
        lines += ["", "## Recent scoped diagnostics", "",
                  "These complete recipes keep their original sources, seeds, gates and budgets. They do not fill another "
                  "row or establish a suite winner. The startup error remains independent of the later Atlas recovery.", "",
                  "| Test | Complete recipe | Seed | Declared updates | Original result | Evidence |",
                  "| --- | --- | ---: | ---: | --- | --- |"]
        recipe_map = {recipe["id"]: recipe for recipe in report["recipes"]}
        for test in diagnostics:
            for recipe_id, keys in test["cells"].items():
                for key in keys:
                    record = report["records"][key]
                    conditions = test["conditions"]
                    lines.append(f"| {_text(test['task_id'])} | {_text(recipe_map[recipe_id]['label'])} | "
                                 f"{conditions['seed']} | {conditions['budget']['steps']} | {record['status']} | "
                                 f"[full recipe, gates and original pins]({relpath(record['receipt'], 'reports/forge')}) |")
    lines += ["", "Regenerate with `python -m experiments.forge.test_performance` or the existing "
              "`python reports/forge/regenerate_technique_inventory.py` workflow. "
              "`python -m experiments.forge.test_performance --check` verifies deterministic generated bytes without writing or training.", "",
              "Witnesses require an explicit byte-pinned `test-witnesses.json` registration with the exact row conditions, type, numerical "
              "verdict pointer and scope. API variants never satisfy a Forge row by matching a name. "
              "See [report contract and evidence gaps](../../docs/forge-test-performance.md).", ""]
    return "\n".join(lines)


def render_html(report):
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return HTML.replace("__DATA__", data).replace("__DIGEST__", html.escape(report["input_digest"]))


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParticleGAN · Performance by test</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182a35;background:#f3f6f8;font-size:14px}*{box-sizing:border-box}body{margin:0}main{max-width:1700px;margin:auto;padding:32px}a{color:#14577d}h1{font-size:32px;margin:8px 0 12px;letter-spacing:-.8px}h2{font-size:20px}p{line-height:1.6;margin:8px 0}.eyebrow{color:#516a7a;letter-spacing:1.7px;text-transform:uppercase;font-size:12px;font-weight:700}.intro{max-width:1080px}.note{background:#fff7e4;border-left:4px solid #bd8424;padding:12px 16px;margin:20px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;padding:18px;background:white;border:1px solid #d6e0e6;border-radius:8px;margin-top:20px}label{display:flex;flex-direction:column;gap:6px;font-size:12px;color:#48606f;font-weight:650}select,input,button{font:inherit;border:1px solid #b7c9d5;border-radius:5px;background:#fff;padding:9px;color:#182a35}input{min-width:205px}button{cursor:pointer}button:hover{background:#edf4f8}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #1684c3;outline-offset:2px}button:disabled{opacity:.45;cursor:default}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.stat{background:white;border:1px solid #d6e0e6;border-radius:8px;padding:14px 20px;min-width:165px}.stat b{font-size:25px;display:block}.stat span{font-size:12px;color:#526b7a}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.badge{font-size:11px;font-weight:750;border-radius:4px;padding:5px 7px;display:inline-block;white-space:nowrap}.PASS{color:#11603f;background:#daf1e5}.FAIL{color:#962f37;background:#fce4e4}.ERROR{color:#8d451a;background:#ffe4d0}.INCOMPLETE{color:#745405;background:#fff0bd}.NOT_RUN,.BLOCKED,.INVALID,.UNASSESSED,.UNCONFIRMED{color:#52626c;background:#eaf0f3}.MIXED{color:#654187;background:#efdef9}.pager{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.pager span{font-size:12px;color:#526b7a}.table-wrap{max-height:650px;overflow:auto;border:1px solid #c5d4dd;border-radius:8px;background:white}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}th,td{padding:10px;border-bottom:1px solid #dbe5eb;border-right:1px solid #e9eff3;vertical-align:top;min-width:96px}thead th{position:sticky;top:0;z-index:3;background:#eaf1f5;text-align:left;height:105px}th small,td small{display:block;color:#597180;font-weight:400;line-height:1.5}tbody th{position:sticky;left:0;z-index:2;min-width:250px;max-width:290px;background:#fff;text-align:left}thead th:first-child{left:0;z-index:4;min-width:250px}tr.zero th{border-left:4px solid #bd8424}td .badge{border:0}td .detail{padding:4px;background:transparent;border:0;text-align:left}thead button{padding:0;border:0;background:transparent;text-align:left;max-width:145px;font-weight:700;overflow-wrap:anywhere}tbody th button{padding:0;border:0;background:transparent;font-weight:700;text-align:left;overflow-wrap:anywhere}.empty{padding:35px;text-align:center;color:#526b7a}footer{font-size:12px;color:#597180;margin-top:24px;overflow-wrap:anywhere}dialog{border:1px solid #bed0dc;border-radius:10px;padding:24px;max-width:950px;width:calc(100% - 36px);max-height:88vh;box-shadow:0 12px 90px #182a3544}dialog::backdrop{background:#182a3577}.close{float:right}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f9;border-radius:5px;padding:14px;font-size:12px;line-height:1.5}details{margin:12px 0}summary{cursor:pointer;font-weight:650;padding:8px 0}.links{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}.warning{color:#945817}noscript{display:block;padding:20px;background:#fff0bd}@media(max-width:650px){main{padding:18px}h1{font-size:27px}.stat{min-width:140px;flex:1}tbody th,thead th:first-child{min-width:190px;max-width:210px}.controls label{width:100%}input{width:100%}}
</style></head><body><main>
<div class="eyebrow">ParticleGAN · Forge research evidence</div>
<h1>Which recipes pass each test?</h1>
<div class="intro"><p>Explore complete recipes under <strong>exact recorded conditions</strong>. Each row is one test and protocol; seeds, source, runtime, original gates, budgets and sampling stay bound. Click a status for the evidence, or a recipe for its full configuration.</p>
<div class="links"><a href="technique-inventory.md">Family leaderboard</a><a href="test-performance.md">Readable Markdown matrix</a><a href="test-performance.json">Projection JSON</a><a href="../../docs/forge-test-performance.md">Contract &amp; evidence gaps</a></div></div>
<div class="note"><strong>A test with zero optimizer passes is a smell, not proof of impossibility.</strong> Oracle and representational witnesses count separately. “No indexed witness” means no compatible witness is registered here; unindexed evidence remains unknown. These cells grant no qualification and cannot be combined into a suite winner.</div>
<noscript>Open <a href="test-performance.md">the Markdown matrix</a> for a readable view without JavaScript.</noscript>
<div class="controls">
<label>Evidence scope<select id="scope"><option value="selected">Selected Forge recipes</option><option value="forge">All measured Forge history</option><option value="api">Public-API protocols</option><option value="all">All indexed evidence</option></select></label>
<label>Test name / protocol<input id="search" type="search" placeholder="e.g. vector_anisotropic"></label>
<label>Recipe / source / optimizer<input id="recipe" type="search" placeholder="e.g. dualnorm, revision hash"></label>
<label>Evidence filter<select id="filter"><option value="all">All tests</option><option value="zero">Zero optimizer passes</option><option value="witness">No indexed witness</option><option value="error">Execution errors</option><option value="incomplete">Incomplete protocols</option></select></label>
<label>Definition<select id="definition"><option value="all">All original protocols</option><option value="matches">Matches current declaration</option><option value="CHANGED">Changed since execution</option></select></label>
<label>Sort tests<select id="sort"><option value="passes">Fewest passing recipes</option><option value="name">Test name</option><option value="coverage">Least graded coverage</option><option value="errors">Most execution errors</option></select></label>
</div>
<div class="stats" id="stats" aria-live="polite"></div>
<div class="legend" id="legend"></div>
<div class="pager"><button id="prevCols">Previous recipes</button><button id="nextCols">Next recipes</button><span id="columns"></span></div>
<div class="table-wrap"><table id="matrix"><caption id="caption" style="text-align:left;padding:12px">PASS counts recipes with at least one compatible passing receipt. MIXED retains both passes and failures. Counts include recipes outside the displayed column page.</caption><thead></thead><tbody></tbody></table><div class="empty" id="empty" hidden>No compatible rows match these filters.</div></div>
<div class="pager"><button id="prevRows">Previous tests</button><button id="nextRows">Next tests</button><span id="rows"></span></div>
<footer>Read-only projection · no training or regrading · input digest <code>__DIGEST__</code><br>NOT RUN means no compatible indexed result. BLOCKED/INVALID retain original admission or contract reasons. MIXED retains all disagreeing receipts.</footer>
</main><dialog id="detail"><button class="close" id="close">Close</button><div id="detailBody"></div></dialog>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';
const report=JSON.parse(document.getElementById('evidence').textContent), byRecipe=new Map(report.recipes.map(r=>[r.id,r])), byTest=new Map(report.tests.map(t=>[t.id,t]));
const $=id=>document.getElementById(id), scope=$('scope'), search=$('search'), recipeSearch=$('recipe'), filter=$('filter'), definition=$('definition'), sort=$('sort');
let rowPage=0,colPage=0; const rowSize=60,colSize=12;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function records(test,id){return (test.cells[id]||[]).map(key=>report.records[key]);}
function status(test,id){const s=[...new Set(records(test,id).map(r=>r.status))];return s.length===1?s[0]:s.length?'MIXED':'NOT RUN';}
function badge(s){return el('span',s,'badge '+s.replaceAll(' ','_'));}
function counts(test,recipes){const c={passing:0,graded:0};recipes.forEach(r=>{const s=status(test,r.id),list=records(test,r.id);c[s]=(c[s]||0)+1;if(list.some(x=>x.status==='PASS'))c.passing++;if(list.some(x=>x.status==='PASS'||x.status==='FAIL'))c.graded++;});return c;}
function protocolLabel(t){const c=t.conditions;if(c.no_recorded_protocol)return 'No recorded execution protocol';const seed=c.seed??c.protocol?.seed??'unbound',steps=c.budget?.steps??t.contract?.steps??'unbound';return 'Seed '+seed+' · '+steps+' updates · source '+String(c.source_digest||c.source_identity||'unbound').slice(0,10);}
function hasWitness(test){return test.witnesses.some(w=>w.status==='PASS');}
function jsonBlock(title,value,parent){const d=el('details'),summary=el('summary',title);d.append(summary,el('pre',JSON.stringify(value,null,2)));parent.append(d);}
function recipeHints(r){const v=r.recipe?.declared_global_recipe||r.recipe?.native_recipe||r.recipe?.complete_recipe||r.recipe||{},parts=[v.optimizer_family||r.lane];for(const [key,label] of [['cell_gaussian_objective','objective'],['cell_reference_policy','reference'],['cell_cubature_rule','cubature']])if(v[key])parts.push(label+': '+v[key]);if(v.optimizer_role_families)parts.push('roles: '+Object.entries(v.optimizer_role_families).map(([k,v])=>k+'='+v).join(', '));if(v.optimizer_signal_roles?.length)parts.push('attenuated: '+v.optimizer_signal_roles.join(', '));return parts;}
function link(label,path,parent){if(!path||path.startsWith('/')||path.includes('..')||path.includes(':')||path.includes('\\'))return;const a=el('a',label);a.href='../../'+path;parent.append(a);}
function showRecipe(id){const r=byRecipe.get(id);if(!r)return;const body=$('detailBody');body.replaceChildren(el('h2',r.label),el('p','This whole recipe/source/runtime identity is preserved across test rows. No test-specific winner is selected.'));jsonBlock('Full recipe',r.recipe,body);jsonBlock('Recipe identity and executed source',r.identity,body);$('detail').showModal();}
function showCell(testId,recipeId){const t=byTest.get(testId),r=byRecipe.get(recipeId);if(!t)return;const body=$('detailBody');body.replaceChildren(el('h2',t.task_id),el('p',r?r.label+' · '+status(t,r.id):'Exact test protocol'));
const links=el('div',undefined,'links');link('Current declaration (may differ)',t.definition,links);body.append(links);body.append(el('p','Definition: '+t.definition_match+'. Original recorded conditions remain authoritative.'));
if(r){jsonBlock('Complete recipe',r.recipe,body);jsonBlock('Recipe/source/runtime identity',r.identity,body);const list=records(t,r.id);if(!list.length)body.append(el('p','NOT RUN — no compatible indexed evidence for this recipe and exact protocol.'));
list.forEach((rec,i)=>{body.append(el('h3','Evidence '+(i+1)+' · '+rec.status));const ls=el('div',undefined,'links');link('Pinned compact receipt',rec.receipt,ls);link('Original publication',rec.publication,ls);body.append(ls);if(rec.result?.audit_scope?.sample_panels_available===false)body.append(el('p','Audit gap: saved sample panels unavailable. This full gate retains the original logged-metric regrade; score-history GIFs are not sample trajectories.','warning'));jsonBlock('Original gates, metrics, completion, errors and receipt hashes',rec,body);const catalog=report.evidence_catalogs?.[rec.original_evidence?.catalog_sha256];if(catalog)jsonBlock('Pinned source catalogs, receipt hashes and audit limitations',catalog,body);});}
jsonBlock('Exact protocol, seed/RNG and source bindings',t.conditions,body);jsonBlock('Recorded task contract (original evaluator fingerprint and budget)',t.contract,body);
jsonBlock('Oracle and representational feasibility witnesses',t.witnesses,body);body.append(el('p','No indexed witness is an evidence gap. It is not proof of impossibility. Oracle/representation passes cannot replace an optimizer run.'));$('detail').showModal();}
function render(){
const q=search.value.toLowerCase(),rq=recipeSearch.value.toLowerCase();let rs=report.recipes.filter(r=>(scope.value==='selected'?r.selected:scope.value==='forge'?r.lane==='Forge':scope.value==='api'?r.lane==='API':true)&&(!rq||JSON.stringify(r).toLowerCase().includes(rq)));
let ts=report.tests.filter(t=>(scope.value==='selected'?t.lane==='Unmeasured declaration'||rs.some(r=>t.cells[r.id]):scope.value==='forge'?t.lane!=='API'&&(t.lane==='Unmeasured declaration'||rs.some(r=>t.cells[r.id])):scope.value==='api'?t.lane==='API':true)&&(!q||JSON.stringify([t.task_id,t.id,t.conditions]).toLowerCase().includes(q))&&(definition.value==='all'||t.definition_match===definition.value)).map(t=>({t,c:counts(t,rs)}));
ts=ts.filter(({t,c})=>filter.value==='zero'?!c.passing:filter.value==='witness'?!c.passing&&!hasWitness(t):filter.value==='error'?(c.ERROR||0)>0:filter.value==='incomplete'?(c.INCOMPLETE||0)>0:true);
ts.sort((a,b)=>{let d=sort.value==='passes'?a.c.passing-b.c.passing:sort.value==='coverage'?a.c.graded-b.c.graded:sort.value==='errors'?(b.c.ERROR||0)-(a.c.ERROR||0):0;return d||a.t.task_id.localeCompare(b.t.task_id)||a.t.id.localeCompare(b.t.id);});
rowPage=Math.min(rowPage,Math.max(0,Math.ceil(ts.length/rowSize)-1));colPage=Math.min(colPage,Math.max(0,Math.ceil(rs.length/colSize)-1));
const stats=$('stats');stats.replaceChildren();for(const [n,label] of [[ts.length,'test / protocol rows'],[rs.length,'recipe identities'],[ts.filter(x=>!x.c.passing).length,'zero optimizer passes'],[ts.filter(x=>!x.c.passing&&!hasWitness(x.t)).length,'no indexed witness']]){const div=el('div',undefined,'stat');div.append(el('b',n),el('span',label));stats.append(div);}
const cols=rs.slice(colPage*colSize,(colPage+1)*colSize),thead=$('matrix').querySelector('thead'),tbody=$('matrix').querySelector('tbody');thead.replaceChildren();tbody.replaceChildren();const head=el('tr');head.append(el('th','Test / exact protocol'),el('th','PASS / recipes'),el('th','Coverage & witnesses'));
cols.forEach(r=>{const th=el('th'),btn=el('button',r.label);btn.title='Open complete recipe '+r.id;btn.onclick=()=>showRecipe(r.id);th.append(btn,el('small',r.id.slice(0,10)));recipeHints(r).forEach(hint=>th.append(el('small',hint)));head.append(th);});thead.append(head);
ts.slice(rowPage*rowSize,(rowPage+1)*rowSize).forEach(({t,c})=>{const tr=el('tr',undefined,c.passing?'':'zero'),th=el('th'),btn=el('button',t.task_id);btn.onclick=()=>showCell(t.id);th.append(btn,el('small',t.lane+' · '+t.id.slice(0,12)),el('small',protocolLabel(t)),el('small','Definition: '+t.definition_match));tr.append(th,el('td',c.passing+' / '+rs.length));const cov=el('td');cov.append(el('small',c.graded+' graded · '+(c.ERROR||0)+' errors · '+(c.INCOMPLETE||0)+' incomplete'),el('small',(c['NOT RUN']||0)+' not run · '+(c.BLOCKED||0)+' blocked · '+(c.INVALID||0)+' invalid'));
const witnesses=t.witnesses.filter(w=>w.status==='PASS');cov.append(el('small','Oracle '+witnesses.filter(w=>w.kind==='oracle').length+' · representation '+witnesses.filter(w=>w.kind==='representational').length));if(!c.passing&&!witnesses.length)cov.append(el('small','NO INDEXED WITNESS','warning'));tr.append(cov);
cols.forEach(r=>{const td=el('td'),btn=el('button',undefined,'detail');btn.setAttribute('aria-label',t.task_id+' · '+r.label+' · '+status(t,r.id)+' · evidence');btn.append(badge(status(t,r.id)));btn.onclick=()=>showCell(t.id,r.id);td.append(btn);tr.append(td);});tbody.append(tr);});
$('empty').hidden=ts.length>0;$('columns').textContent='Recipes '+(rs.length?colPage*colSize+1:0)+'–'+Math.min(rs.length,(colPage+1)*colSize)+' of '+rs.length+'; counts include all '+rs.length+' matching identities.';$('rows').textContent='Tests '+(ts.length?rowPage*rowSize+1:0)+'–'+Math.min(ts.length,(rowPage+1)*rowSize)+' of '+ts.length;
$('prevCols').disabled=colPage===0;$('nextCols').disabled=(colPage+1)*colSize>=rs.length;$('prevRows').disabled=rowPage===0;$('nextRows').disabled=(rowPage+1)*rowSize>=ts.length;
}
['PASS','FAIL','ERROR','INCOMPLETE','NOT RUN','BLOCKED','INVALID','MIXED'].forEach(s=>$('legend').append(badge(s)));
[scope,search,recipeSearch,filter,definition,sort].forEach(control=>control.addEventListener(control.tagName==='INPUT'?'input':'change',()=>{rowPage=0;colPage=0;render();}));
$('prevCols').onclick=()=>{colPage--;render();};$('nextCols').onclick=()=>{colPage++;render();};$('prevRows').onclick=()=>{rowPage--;render();};$('nextRows').onclick=()=>{rowPage++;render();};$('close').onclick=()=>$('detail').close();
function openHash(){const hash=location.hash.slice(1);if(hash.startsWith('recipe-'))showRecipe(hash.slice(7));else if(hash.startsWith('cell-')){const ids=hash.slice(5).split('-');if(ids.length===2)showCell(ids[0],ids[1]);}}
render();openHash();window.addEventListener('hashchange',openHash);
</script></body></html>
'''
