"""Regroup existing evidence by formulation, retaining architecture trials."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.transfer_suite.formulations import architecture_cell
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent
SOLVE = SUITE / "solvability"


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def trial(path, label, *, spec=None):
    value = read(path)
    result = value.get("result", value)
    effective = deepcopy(spec or value.get("effective_spec", value["spec"]))
    effective["schedule"] = value["policy"]["schedule"]
    # The old test name encoded its penalty. Its fixed-formulation counterpart
    # has the same target/architecture/budget and is the nominal ring condition.
    if effective["name"] == "stress_r1_r2":
        effective["name"] = "stress_nominal_ring"
    return dict(label=label, spec=effective, result=result,
                artifact=str(path.relative_to(SUITE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def build():
    declared = read(SUITE / "study/manifest.json")
    originals = {s["name"]: s for s in declared["tasks"] if s["split"] == "development"}
    dynamics = read(SOLVE / "dynamics/episodes.json.gz")["episodes"]
    rows = []
    for coefficient in (3, 10):
        row = dict(name=f"rp_logistic_bcap{coefficient}", penalty_coefficient=coefficient,
                   fixed=dict(loss="logistic", mode="rp", penalty="b_cap", kappa=1.25,
                              prior_regularization=.05, particle_l2=0.), cases={}, required={})
        for name, original in originals.items():
            if original["tier"] == "diagnostic":
                continue
            if original["tier"] == "required":
                path = (SUITE / "study/episodes" / f"cosine__{name}.json.gz" if coefficient == 3
                        else SOLVE / "required_cap10/episodes" / f"cap10__{name}.json.gz")
                value = read(path)
                verdict = test_verdict(original, value["result"])
                row["required"][name] = dict(verdict=verdict, artifact=str(path.relative_to(SUITE)),
                                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                continue
            runner = original["runner"]
            variants = []
            if name == "stress_r1_r2":
                variants.append(trial(ROOT / "nominal_ring/episodes" / f"bcap{coefficient}_nominal_ring__{name}.json.gz",
                                      f"bcap{coefficient}, nominal ring"))
            elif runner == "image":
                cards = [("stage1", "baseline"), ("stage1", "residual16"), ("stage1", "transpose24"),
                         ("stage2", "residual12"), ("stage2", "transpose16")] if coefficient == 3 else [("cross", "residual16_cap10")]
                for stage, card in cards:
                    variants.append(trial(SOLVE / "images" / stage / "episodes" / f"{card}__{name}.json.gz", card))
            elif coefficient == 3:
                path = SUITE / "study/episodes" / f"cosine__{name}.json.gz"
                variants.append(trial(path, "original architecture", spec=original))
            elif runner == "vector":
                stage = "screen" if name in ("vector_unequal_mass", "vector_unequal_width", "vector_overlap") else "regressions"
                variants.append(trial(SOLVE / "vectors" / stage / "episodes" / f"cap10__{name}.json.gz", "cap10 original architecture"))
            else:
                value = next(v for v in dynamics if v["candidate"]["name"] == "cap10" and v["spec"]["name"] == name)
                # dynamics/scripts/search.py fixes cosine for every archived run.
                effective = value["spec"] | dict(schedule="cosine")
                variants.append(dict(label="cap10 original architecture", spec=effective, result=value["result"],
                                     artifact="solvability/dynamics/episodes.json.gz", episode_index=value["episode_index"],
                                     sha256=hashlib.sha256((SOLVE / "dynamics/episodes.json.gz").read_bytes()).hexdigest()))
            cell = architecture_cell(variants, runner)
            assert cell["formulation"] == dict(loss="logistic", mode="rp", penalty="b_cap", coefficient=coefficient,
                                               kappa=1.25, particle_l2=0., prior_regularization=.05, prior_kind="particles")
            for stored, variant in zip(cell["trials"], variants):
                stored.update({k: variant[k] for k in ("sha256", "episode_index") if k in variant})
            canonical_name = variants[0]["spec"]["name"]
            row["cases"][canonical_name] = dict(runner=runner, **cell)
        row["required_passes"] = sum(v["verdict"]["passed"] for v in row["required"].values())
        row["practical_passes"] = sum(v["supported"] for v in row["cases"].values())
        row["eligible"] = row["required_passes"] == len(row["required"]) == 9
        row["domains"] = {domain: dict(passed=sum(c["supported"] for c in row["cases"].values() if c["runner"] == domain),
                                      total=sum(c["runner"] == domain for c in row["cases"].values()))
                          for domain in ("vector", "stress", "image")}
        rows.append(row)
    assert [r["practical_passes"] for r in rows] == [7, 7]
    report = dict(version="formulation-architecture-v1", rows=rows,
                  rule="One formulation entry; architecture trials stay inside each case. Target, formulation, training settings and resource budget must match within an architecture cell. Every attempt remains visible.",
                  nominal_ring="Replaces the formulation-changing R1+R2 condition with its b_cap counterpart for b_cap entries; historical R1+R2 results remain separate.",
                  training_scope="Existing declared host recipes are retained, including different host learning rates and Adam betas. No claim of one universal numerical optimizer preset.",
                  source_sha256={str(p.relative_to(SUITE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), SUITE.parents[1] / "benchmarks/transfer_suite/formulations.py"]})
    (ROOT / "leaderboard.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Formulation × architecture results", "",
             "One entry per formulation. PASS means at least one listed architecture sustains every live metric; "
             "each case counts once. Architecture failures remain visible. Architecture cells cannot mix numerical "
             "formulations, optimizer settings, data or training budgets.", ""]
    for row in rows:
        lines += [f"## {row['name']}", "", f"Required: {row['required_passes']}/9. Practical: {row['practical_passes']}/16. "
                  f"Eligible on required tests: {row['eligible']}.", "",
                  "| Problem / condition | Supported | Architecture observations |", "| --- | --- | --- |"]
        for name, cell in row["cases"].items():
            values = "; ".join(f"{t['label']}: {t['verdict']['status']}" for t in cell["trials"])
            lines.append(f"| {name} | {cell['status']} | {values} |")
        lines.append("")
    lines += ["The image variants include changes to G as well as D. Discriminator width changes alone, generator changes, "
              "and their combination are explicit in the machine-readable architecture records. These are inspected "
              "development cases with one initialization; architecture support is not a fresh-transfer result.", "",
              "[Exact axes, metrics and source artifacts](leaderboard.json)."]
    (ROOT / "MATRIX.md").write_text("\n".join(lines) + "\n")
    print([(r["name"], r["required_passes"], r["practical_passes"]) for r in rows])


if __name__ == "__main__":
    build()
