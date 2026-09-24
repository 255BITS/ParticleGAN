"""Read-only audit of the frozen pre-G anchor saved-state gate and one-state fit.

This neither trains a host nor grades a new candidate. Its inputs are the
already recorded 44-step replay and update-2401 parameter-motion capture.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch
from torch.func import functional_call, jacrev


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def check_gate(root, gate):
    summary = read_json(gate / "summary.json")
    assert summary["status"] == "PASS" and summary["warm_eligible"]
    declaration = summary["declaration"]
    for name, expected in declaration["sources"].items():
        archived = (gate / "source" / name).read_bytes()
        assert digest(archived) == expected, name
        assert (root / name).read_bytes() == archived, name

    reference = read_json(root / "reports/toy100/continuous-evidence/"
                          "pr84-stationary-failure-diagnosis/diagnosis.json.gz")
    reference_rows = {row["step"]: row for row in reference["rows"]}
    total = 0
    branches = []
    for branch in summary["branches"]:
        start, end = branch["start"], branch["end"]
        original = branch["variants"]["original"]
        trial = branch["variants"]["reallocation"]
        n = end - start + 1
        assert len(original["points"]) == len(trial["points"]) == n
        assert len(trial["dynamics"]["corrections"]) == n
        assert original["rates"] == trial["rates"]
        assert original["moment_steps"] == trial["moment_steps"] == {"d": [end], "g": [end]}
        assert original["rng_final_sha256"] == trial["rng_final_sha256"]
        assert original["noise"] == trial["noise"]
        assert trial["dynamics"]["method"] == declaration["method"]
        assert (trial["dynamics"]["source_sha256"] ==
                declaration["sources"]["reports/toy100/sample_anchor_prestart_candidate.py"])
        assert trial["dynamics"]["correction_rng_checks"] == n
        assert trial["dynamics"]["correction_owner_checks"] == n
        assert trial["dynamics"]["native_batch_checks"] == 3 * n
        for offset, (control, test, correction) in enumerate(zip(
                original["points"], trial["points"], trial["dynamics"]["corrections"])):
            step = start + offset
            assert control["step"] == test["step"] == correction["step"] == step
            assert control["support"] == reference_rows[step]["stages"]["bounded_joint"]
            assert original["dynamics"]["records"][offset] == dict(
                reference_rows[step]["stages"]["record"], outer_step=offset + 1)
            assert test["grade"]["modes"] == 8 and test["grade"]["hq"] >= .9
            assert correction["fit"]["status"] == "CONVERGED"
            assert correction["selected"] == "joint_fit" and not correction["nonconverged_fit_rested"]
            assert correction["final_cost"] <= correction["pre_cost"] + 1e-11
            d_rate, g_rate = trial["rates"][2*offset:2*offset+2]
            assert d_rate == {"step": step, "role": "d", "rates": [.00425]}
            assert g_rate == {"step": step, "role": "g_prior", "rates": [.00425, .0085]}
        jacobians = sum(len(row["fit"]["records"]) for row in trial["dynamics"]["corrections"])
        trials = sum(len(record["trials"]) for row in trial["dynamics"]["corrections"]
                     for record in row["fit"]["records"])
        assert trial["dynamics"]["additional_joint_output_jacobians"] == jacobians
        assert trial["dynamics"]["additional_nonlinear_output_trials"] == trials
        assert trial["local_gate"]["checks"] == trial["local_gate"]["passing_checks"] == n
        total += n
        branches.append(dict(start=start, end=end, checks=n,
                             min_hq=trial["local_gate"]["min_hq"],
                             original_passing=original["local_gate"]["passing_checks"],
                             jacobians=jacobians, nonlinear_trials=trials))
    assert total == 44
    return dict(checks=total, branches=branches,
                method=declaration["method"], summary_sha256=digest((gate / "summary.json").read_bytes()))


def check_motion(root, motion):
    manifest = read_json(motion / "manifest.json")
    for name, expected in manifest["files"].items():
        data = (motion / name).read_bytes()
        assert digest(data) == expected["stored_sha256"], name
        assert len(data) == expected["stored_bytes"], name
        if name.endswith(".gz"):
            raw = gzip.decompress(data)
            assert digest(raw) == expected["original_sha256"], name
            assert len(raw) == expected["original_bytes"], name

    capture = torch.load(io.BytesIO(gzip.decompress((motion / "captured-parameters.pt.gz").read_bytes())),
                         weights_only=True, map_location="cpu")
    reported = read_json(motion / "result.json.gz")
    sys.path.insert(0, str(root))
    from benchmarks.locked_shared.mlp import SimpleMLPGenerator
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.joint_output_pullback import fit_output_targets

    torch.set_num_threads(1)
    generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
    with torch.no_grad():
        for parameter, value in zip(generator.parameters(), capture["pre"][:-1]):
            parameter.copy_(value)
    prior = torch.nn.Parameter(capture["pre"][-1].clone())
    names = list(dict(generator.named_parameters()))

    def output(parameters, z):
        return functional_call(generator, dict(zip(names, parameters)), (z,))

    derivatives = jacrev(output, argnums=(0, 1))(tuple(generator.parameters()), prior)
    target = torch.tensor(capture["correction"]["mm"]["target"], dtype=prior.dtype)
    jacobian = torch.cat([v.reshape(target.numel(), -1) for v in
                          [*derivatives[0], derivatives[1]]], dim=1).double().detach()
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    active = singular > 1e-6 * singular.max()
    basis = vh[active]
    assert int(active.sum()) == reported["pre_jacobian"]["rank"] == 24

    def vector(values):
        return torch.cat([p.detach().double().flatten() for p in values])

    initial = vector(capture["pre"])

    def geometry(values):
        delta = vector(values) - initial
        null = delta - basis.T @ (basis @ delta)
        return dict(total_norm=float(delta.norm()),
                    linear_output_norm=float((jacobian @ delta).norm()),
                    null_norm=float(null.norm()),
                    null_fraction=float(null.norm() / delta.norm()) if delta.norm() else 0.)

    for label, values in (("native", capture["native"]),
                          ("post_native_fit", capture["accepted"])):
        for key, value in geometry(values).items():
            assert abs(value - reported[label][key]) < 1e-8, (label, key)
    fit = fit_output_targets(generator, prior, target)
    assert fit["status"] == reported["fit"]["status"] == "CONVERGED"
    assert abs(fit["final_max_row_error"] - reported["fit"]["final_max_row_error"]) < 1e-9
    values = [p.detach() for p in generator.parameters()] + [prior.detach()]
    for key, value in geometry(values).items():
        assert abs(value - reported["pre_start_fit"][key]) < 1e-8, key
    assert reported["pre_start_grade"]["modes"] == 8
    assert reported["pre_start_grade"]["hq"] == 1.0
    assert reported["one_update_receipt"]["updates"] == 1
    return dict(input_state_sha256=read_json(motion / "declaration.json.gz")["input_state_sha256"],
                native=reported["native"], post_native_fit=reported["post_native_fit"],
                pre_start_fit=reported["pre_start_fit"],
                target_max_error=fit["final_max_row_error"],
                result_sha256=digest((motion / "result.json.gz").read_bytes()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    gate = root / "artifacts/continuous-learning/round6/sample-anchor-prestart-saved44"
    motion = root / "reports/toy100/continuous-evidence/round6-anchor-parameter-motion"
    result = dict(status="PASS", scope="read-only archived evidence audit",
                  gate=check_gate(root, gate), motion=check_motion(root, motion))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], checks=result["gate"]["checks"],
                          null_fraction={k: result["motion"][k]["null_fraction"]
                                         for k in ("native", "post_native_fit", "pre_start_fit")})))


if __name__ == "__main__":
    main()
