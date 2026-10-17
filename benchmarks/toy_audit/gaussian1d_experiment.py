"""Validate the scalar scorer or publish saved evidence; neither action trains."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .api_contract import discover
from .api_gaussian1d import CASE_ID, list_cases
from .api_publish import verify_run
from .api_run import file_hash, json_value, write_json
from .api_vectors import _bounds
from .gaussian1d_quality import sample_target, score_samples


def controls():
    case = list_cases()[0]
    spec = case["spec"]
    n = case["eval_samples"]
    oracle = sample_target(spec, n, torch.Generator().manual_seed(78013))
    samples = {
        "independent_target": oracle,
        "collapsed_width": torch.full((n, 1), 2.),
        "wrong_location": oracle + .5,
        "doubled_width": 2. + 2. * (oracle - 2.),
        "same_moment_two_atoms": torch.tensor([1.5, 2.5]).repeat(n // 2)[:, None],
        "same_moment_uniform": (2. + 3.**.5 * .5 * torch.linspace(-1., 1., n))[:, None],
        "nonfinite": torch.full((n, 1), float("nan")),
        "undersized": oracle[:128],
    }
    rows = []
    for name, values in samples.items():
        metrics = score_samples(values, spec)
        failures = _bounds(metrics, case["thresholds"])
        passed, expected = not failures, name == "independent_target"
        rows.append(dict(control=name, metrics=metrics, failed_bounds=failures,
                         passed=passed, expected_pass=expected, control_passed=passed == expected))
    return dict(schema="gaussian1d-scorer-controls-v1", training_updates=0,
                evaluation_seed=78013, case_id=CASE_ID, thresholds=case["thresholds"],
                passed=all(row["control_passed"] for row in rows), controls=rows)


def publish(raw, output):
    raw, output = Path(raw), Path(output)
    receipt = verify_run(raw)  # Rederive complete protocol and verify actual media.
    case = receipt["case"]
    if case != json_value(discover()[CASE_ID]) or receipt["seed"] != case["protocol_seed"]:
        raise ValueError("saved execution differs from the frozen scalar declaration")
    metric_steps = receipt["protocol"]["metric_evaluation_steps"]
    scored = [o for o in receipt["observations"] if o["step"] > 0 and o["step"] in metric_steps]
    source_key = hashlib.sha256(json.dumps(receipt["source"], sort_keys=True).encode()).hexdigest()
    row = dict(
        id=CASE_ID, recipe=receipt["recipe"], runtime=receipt["runtime"],
        protocol=receipt["protocol"], seed=receipt["seed"], api_components=receipt["api_components"],
        initialization=case["initialization"], prior_options=case["prior_options"],
        sampling=case["sampling"], source_identity=source_key,
        completed_updates=receipt["completed_updates"],
        default_protocol_complete=receipt["default_protocol_complete"],
        metric_passed=receipt["metric_passed"], sustained_metric_passed=receipt["sustained_metric_passed"],
        verdict=receipt["verdict"], failed_bounds=receipt["failed_bounds"],
        initial_metrics=receipt["observations"][0]["metrics"],
        final_metrics=receipt["observations"][-1]["metrics"],
        passing_checks=sum(o["passed"] for o in scored), total_checks=len(scored),
        terminal_metrics=[{k:o[k] for k in ("step", "metrics", "passed", "failed_bounds")}
                          for o in scored[-case["terminal_observations"]:]],
        execution_elapsed_seconds=receipt["elapsed_seconds"],
        gif="goal.gif", gif_sha256=receipt["artifacts"]["goal.gif"]["sha256"],
        frames=receipt["gif_frames"], raw_receipt_sha256=file_hash(raw / "receipt.json"),
        raw_artifacts=receipt["artifacts"], qualification_input=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "goal.gif"
    if target.exists() and file_hash(target) != row["gif_sha256"]:
        raise ValueError("refusing to overwrite a different training GIF")
    shutil.copyfile(raw / "goal.gif", target)
    write_json(output / "results.json", dict(
        schema="particlegan_api_toy_supplement_v1", cases=[case], runs=[row],
        readouts=[dict(id=CASE_ID, goal=case["goal"], scope=case["scope"],
                       execution_status=receipt["status"], verdict=receipt["verdict"],
                       completed_updates=receipt["completed_updates"], default_updates=case["default_steps"],
                       failed_bounds=receipt["failed_bounds"], gif=row["gif"],
                       source_commit=receipt["source"]["commit"])],
        source_identities={source_key:receipt["source"]},
        training_or_rescoring_by_publication=False, historical_receipts_changed=False))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("controls", "publish"))
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.action == "controls":
        report = controls()
        write_json(args.output, report)
        print(json.dumps(dict(passed=report["passed"], controls=len(report["controls"]), training_updates=0)))
        return 0 if report["passed"] else 1
    if args.raw is None:
        parser.error("publish requires --raw")
    row = publish(args.raw, args.output)
    print(json.dumps(dict(verdict=row["verdict"], completed_updates=row["completed_updates"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
