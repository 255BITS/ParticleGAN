"""Numerical ring16 oracle/destructive controls; this command never trains."""
from copy import deepcopy
import argparse
import json

import torch

from .api_ring16 import list_cases
from .api_vectors import _target, score_case


def run_controls():
    case = list_cases()[0]
    n, step = case["eval_samples"], case["default_steps"]

    def draw(changed):
        return _target(changed, n, torch.Generator().manual_seed(731), step)

    target = draw(case)
    centers = torch.tensor(case["spec"]["means"])
    missing = deepcopy(case)
    missing["spec"]["masses"] = [1/15]*15 + [0.]
    biased = deepcopy(case)
    biased["spec"]["masses"] = [.55] + [.45/15]*15
    narrow, broad = deepcopy(case), deepcopy(case)
    narrow["spec"]["covariances"] = [[[.0001, 0.], [0., .0001]]]*16
    broad["spec"]["covariances"] = [[[.25, 0.], [0., .25]]]*16
    angles = torch.arange(n)*(2*torch.pi/n)
    samples = {
        "independent_target": target,
        "missing_cluster": draw(missing),
        "biased_mass": draw(biased),
        "centers_only": centers[torch.arange(n) % 16],
        "collapsed_width": draw(narrow),
        "inflated_width": draw(broad),
        "continuous_circle": 3*torch.stack([angles.cos(), angles.sin()], 1),
        "wrong_location": target + torch.tensor([1., 0.]),
        "nonfinite": target.clone(),
    }
    samples["nonfinite"][0, 0] = float("nan")
    rows = []
    for name, points in samples.items():
        try:
            result = score_case(case, points, step)
        except ValueError as error:
            result = dict(passed=False, metrics={"nonfinite_output_values": 1},
                          failed_bounds=[str(error)])
        expected = name == "independent_target"
        rows.append(dict(control=name, expected_pass=expected,
                         control_passed=result["passed"] == expected, **result))
    return dict(schema="ring16-acquisition-scorer-controls-v1", training_updates=0,
                target_seed=731, case_id=case["id"], thresholds=case["thresholds"],
                passed=all(row["control_passed"] for row in rows), controls=rows)


def main(argv=None):
    from .api_run import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    options = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = run_controls()
    if options.output:
        write_json(options.output, report)
    print(json.dumps({"passed": report["passed"], "controls": len(report["controls"]),
                      "training_updates": 0}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
