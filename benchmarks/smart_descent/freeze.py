"""Freeze one nonzero challenger using development data, before transfer."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.locked_shared import baseline
from . import study


def run(sources, output):
    if output.exists():
        raise FileExistsError("use a new evaluation directory")
    rows, provenance, reports = [], [], {}
    for source in sources:
        payload = (source / "search.json").read_bytes()
        report = json.loads(payload)
        reports[source] = report
        if "selected" not in report:
            raise ValueError(f"search is not complete: {source}")
        provenance.append(dict(source=str(source), sha256=hashlib.sha256(payload).hexdigest(),
                               selected=report["selected"]))
        for row in report["rows"]:
            nonzero = any(x != 0 for role in row["policy"]["weights"] for action in role for x in action)
            summary = study.row_summary(row)
            if nonzero and summary["bounds"] == 29 and summary["stable"] == 9:
                rows.append((source, row, summary))
    if not rows:
        raise RuntimeError("no nonzero policy passes and sustains every development toy")
    source, best, summary = min(rows, key=lambda value: value[2]["mean_confirmation_fraction"])
    selected_report = reports[source]
    fitting = selected_report["protocol"]["source_sha256"]
    current = study.fingerprint()["source_sha256"]
    numerical = {name: digest for name, digest in fitting.items()
                 if name.startswith(("particlegan/", "benchmarks/locked_shared/")) or
                 name in ("benchmarks/smart_descent/controller.py", "benchmarks/smart_descent/study.py",
                          "benchmarks/learned_lr_evaluation.py")}
    for name, digest in numerical.items():
        if current.get(name) != digest:
            raise RuntimeError(f"numerical source changed after fitting: {name}")
    transfer = selected_report["fresh_transfer"]
    if any(report["fresh_transfer"] != transfer for report in reports.values()):
        raise ValueError("candidate searches must share the same predeclared transfer cases")
    frozen = dict(policy=best["policy"], selected=best["name"], summary=summary,
                  fresh_transfer=transfer, selection="fastest nonzero 29/29, 9/9 sustained development challenger; cosine remains eligible as overall winner",
                  eligible_count=len(rows), searches=provenance, selected_search=str(source),
                  fitting_source_sha256=fitting, numerical_source_sha256=numerical,
                  evaluation_source_sha256=current)
    output.mkdir(parents=True)
    baseline.write_json(output / "frozen.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
