"""Preserve PR399's original research gates without adopting EMA-idle."""
from collections import Counter, defaultdict
from pathlib import Path
from posixpath import relpath

from .contracts import stable_hash
from .test_performance import Inputs, SUPPLEMENTAL


PINS = {'reports/forge/test-performance-evidence/pr399-retained-successor-nonatlas-21.json': 'ade708511a345262c07ab562fed340edf963917d2a30ececfeed584ecb2baca0', 'reports/forge/test-performance-evidence/pr399-retained-atlas-seed0-diagnostics-3.json': '6cc3cd8bbcfb8306c57065c70f67de70e9899132ad92fc6fae5e495ef13c9301', 'reports/forge/test-performance-evidence/pr399-retained-pr399-paired-tier1-12.json': 'e85096d198e735640807bed50b981032522d099ad878aef9614f7d5d72f48378', 'reports/forge/test-performance-evidence/pr399-retained-atlas-dynamics-6.json': 'fe2420f19398069d8e2b9b66152eb6c5100ad173fdd47635aeda9195afe340a7'}


def publications(root):
    """Validate byte pins and source identities before displaying archived scores."""
    inputs = Inputs(root)
    registry = inputs.read(SUPPLEMENTAL)
    indexed = {p["path"]: p["sha256"] for p in registry["publications"]}
    packets = []
    attempts = set()
    for path, sha in PINS.items():
        if indexed.get(path) != sha:
            raise ValueError("PR399 evidence missing from display index")
        document = inputs.read(path, sha256=sha)
        if any(document.get(key) is not False for key in ("qualification_input", "qualification_reuse")):
            raise ValueError("PR399 archival display cannot grant qualification")
        expected = document["discovery_summary"]["status_counts"]
        if dict(Counter(c["status"] for c in document["cells"])) != expected:
            raise ValueError("PR399 recorded gate counts changed")
        for cell in document["cells"]:
            condition = document["condition_contracts"][cell["conditions_sha256"]]
            recipe = document["recipe_contracts"][cell["recipe_sha256"]]
            result = cell["result"]
            attempt = result["attempt_id"]
            if (attempt in attempts or stable_hash(recipe) != cell["recipe_sha256"]
                    or stable_hash(condition) != cell["conditions_sha256"]
                    or condition["archived_conditions_sha256"] != cell["archived_conditions_sha256"]
                    or condition["task"]["id"] != cell["task_id"]
                    or condition["seed"] != 0 or not condition["source_digest"]
                    or not cell["archived_cell_sha256"]):
                raise ValueError("PR399 duplicate attempt or altered scientific binding")
            attempts.add(attempt)
            grade = result.get("original_grade")
            if grade and grade.get("gate_status", grade.get("status")) != cell["status"]:
                raise ValueError("PR399 original gate changed")
            if cell["status"] in {"PASS", "FAIL"} and not cell["full_protocol_complete"]:
                raise ValueError("PR399 partial output cannot pass a task")
        packets.append((path, document))
    if len(attempts) != 42:
        raise ValueError("PR399 must retain all 42 distinct attempts")
    return packets


def tier_rows(root, page, requirements):
    from .research_tiers import _row
    rows = defaultdict(list)
    for path, document in publications(root):
        if "-atlas-" in path:  # Explicit policy/alternate-host diagnostics are untiered.
            continue
        groups = defaultdict(dict)
        labels = {}
        for cell in document["cells"]:
            condition = document["condition_contracts"][cell["conditions_sha256"]]
            recipe = document["recipe_contracts"][cell["recipe_sha256"]]
            family = "atlas" if recipe["recipe_preset"] == "atlas" else "bcap-pure"
            identity = (family, cell["candidate_id"], cell["candidate_revision"],
                        cell["recipe_sha256"], condition["source_digest"], condition["seed"])
            if cell["task_id"] in groups[identity]:
                raise ValueError("PR399 research row cannot pick a best attempt")
            groups[identity][cell["task_id"]] = cell["status"]
            labels[identity] = cell["label"]
        for identity, tasks in groups.items():
            family, _, _, _, source, seed = identity
            target = f"reports/forge/families/{family}.md#pr399-retained-research"
            label = f"{labels[identity]}; seed {seed}; {source[:8]}"
            from .research_attempts import scope_id
            rows[family].append(_row(root, page, label, target, tasks, requirements,
                                     scope=scope_id(source, identity[2]) if seed == 0 else None))
    return dict(rows)


def family_lines(root, page, family):
    if family not in {"atlas", "bcap-pure"}:
        return []
    def link(label, path):
        return f"[{label}]({relpath(str(Path(root) / path), str(Path(page).parent))})"
    lines = ['<a name="pr399-retained-research"></a>', "", "## PR399 retained research", "",
             "Each tier row keeps its original recipe, source and seed. The task-name projection into "
             "the recorded revision 9 tiers is archival research; it grants no current qualification. "
             "Final metrics and observation checks never override the original full task gate.", ""]
    if family == "bcap-pure":
        lines += ["The EMA-idle proposal is retired: **4/6 Tier 1** against **5/6** for the matched control. "
                  "Candidate Gaussian and word gates remain INCOMPLETE: 946/1,000 and 12,613/20,001 actual "
                  "generator/prior updates. The control word timed out; its terminal update count is unknown. "
                  "Ring and unused-token have zero skipped updates and identical metrics, so these establish no idle win. "
                  "No idle hook or runtime dependency is adopted.", "",
                  "The 21 successor attempts contain **9 PASS / 12 FAIL** under their separate source and recipe "
                  "identities. Gaussian history and bank variants retain their acquisition/retention wins; none supplies "
                  "one recipe passing the whole gauntlet. The prior 228-run ledger is not recounted or pooled.", "",
                  "Paired cost: **2,052.616064 queue seconds**, **2,072.815230 inclusive supervisor seconds**, zero paid retries. "
                  "These are overlapping clocks. A6000 co-tenancy prevents a controlled speed claim. Owner source "
                  "`706726813d2693d506a2ede3f2471852adc5f8a6b3a8057cc50bf7dc644b0966` and executed FD-forwarding "
                  "closure `27ab4cf9c9de35487f5dca60dc7c0329ddf89977d37f91e547e517de97a7bf06` retain separate identities.", ""]
    else:
        lines += ["The seed-0 80-update diagnostics retain **1 PASS / 2 FAIL**. The separate dynamics-only "
                  "ablation retains **1 PASS / 1 FAIL / 4 INCOMPLETE**. Saved ring samples are diagnostics; "
                  "they cannot repair missing lifecycle/optimizer metadata or grant a full task pass. "
                  "The earlier cumulative 66-attempt packet is not pooled into a qualification score.", ""]
    for path, document in publications(root):
        cells = [c for c in document["cells"] if
                 (document["recipe_contracts"][c["recipe_sha256"]]["recipe_preset"] == "atlas") == (family == "atlas")]
        if not cells:
            continue
        archive = document["archived_publication"]
        summary = document["discovery_summary"]
        lines += [f"### {summary['id']}", "",
                  f"{link('Compact evidence', path)} · [immutable original]({archive['url']}) · "
                  f"original blob `{archive['git_blob']}`, SHA-256 `{archive['sha256']}`. "
                  "Full original contracts and per-observation diagnostics remain in that archive; compact receipts retain "
                  "their exact hashes and all final grades, costs and artifact pins.", "",
                  "| Recipe / source | Task | Original gate | Completed lifecycle / optimizer updates | Final metrics / observations | Queue seconds |",
                  "| --- | --- | --- | --- | --- | ---: |"]
        for cell in cells:
            result = cell["result"]
            condition = document["condition_contracts"][cell["conditions_sha256"]]
            steps = result.get("completed_steps", result.get("completed_lifecycle_steps"))
            updates = result.get("actual_optimizer_updates", result.get("raw_cost", {}).get("optimizer_updates"))
            progress = f"{steps if steps is not None else 'unknown'} / {updates if updates is not None else 'unknown'}"
            metrics = result.get("final_metrics")
            if metrics is None:
                diag = result.get("saved_sample_diagnostic") or {}
                metrics = {k: diag[k] for k in ("regraded_observations", "passing", "suffix", "qualification") if k in diag}
            selected = {k: v for k, v in (metrics or {}).items() if
                        k in {"mean_abs", "grad_med", "mean_error_sigma", "std_ratio", "cdf_ks", "mass_tv",
                              "sw1_normalized", "covariance_error", "min_mass_ratio", "stationary_passing",
                              "shift_passing", "regraded_observations", "passing", "suffix", "qualification"}}
            shown = "; ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k,v in selected.items())
            if not shown:
                shown = "Full final metrics and original grade in compact evidence"
            queue = result.get("queue_seconds")
            charge = f"{queue:.6f}" if isinstance(queue, (int,float)) else "not recorded"
            lines.append(f"| `{cell['label']}` / `{condition['source_digest'][:8]}` | `{cell['task_id']}` | "
                         f"{cell['status']} | {progress} | {shown} | {charge} |")
        lines.append("")
    return lines
