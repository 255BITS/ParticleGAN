"""Abort each exact warm/hold replay at its 200th active pre-EMA boundary.

This passive diagnostic retains both full snapshots. It does not continue
through the 1,400-step candidate hold or edit any frozen training source.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys


class Captured(Exception):
    pass


def differences(a, b, prefix=""):
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if torch.equal(a, b) else [dict(path=prefix, kind="tensor", a_shape=list(a.shape), b_shape=list(b.shape))]
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in a.keys() | b.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                result.append(dict(path=path, kind="missing_key"))
            else:
                result.extend(differences(a[key], b[key], path))
        return result
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [dict(path=prefix, kind="length", a=len(a), b=len(b))]
        return [row for index, (left, right) in enumerate(zip(a, b))
                for row in differences(left, right, f"{prefix}[{index}]")]
    return [] if a == b else [dict(path=prefix, kind="value", a=a, b=b)]


def run(root, archive, output, horizon):
    import torch
    from benchmarks.toy100.continuous_probe import run_probe
    from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, training_state_sha256
    from reports.toy100 import allocation_continuous_probe as driver
    from reports.toy100.pr84_critic_refinement_capture import snapshot, _sha
    declaration = json.loads((archive / "declaration.json").read_text())
    expected = json.loads((archive / "forks/candidate.json").read_text())
    driver.verify_sources(declaration["source"], root=root, archive=archive / "source")
    factory, method, _ = driver.load_factory(declaration["factory"])
    if method != declaration["method"]:
        raise RuntimeError("candidate factory method differs")
    captured = {}
    with factory(task="mode_hold", start_step=1000, correction=True) as (recorder, source), ExitStack() as stack:
        def at_warm(state):
            if training_state_sha256(state) != expected["warm_state_sha256"]:
                raise RuntimeError("fresh prefix does not match the original warm state")
            stack.enter_context(constant_rate_context(state))
            def accounting(calls, outer):
                state["declare_optimizer_accounting"](calls=calls+horizon-outer, moment_updates=horizon)
                if outer % 50 == 0:
                    print(json.dumps(dict(event="CAPTURE_PROGRESS", horizon=horizon, update=1000+outer)), flush=True)
                if outer == 200:
                    value = snapshot(recorder._local)
                    digest = _sha(value)
                    if digest != expected["dynamics_receipt"]["first200_full_state_sha256"]:
                        raise RuntimeError("passive boundary capture differs from archived raw hash")
                    for key in ("records", "corrections"):
                        if driver.untimed(getattr(recorder, key)) != driver.untimed(expected["dynamics_receipt"][key][:200]):
                            raise RuntimeError(f"passive replay changed {key}")
                    captured.update(state=value, sha256=digest)
                    raise Captured()
            recorder.accounting = accounting
        try:
            run_probe(json.loads((root / driver.CONFIG).read_text()), mode="scheduled", steps=horizon,
                      diagnostic_every=50, dense_after=1000, dense_until=horizon,
                      checkpoint_hook_step=1000, checkpoint_hook=at_warm)
        except Captured:
            pass
        else:
            raise RuntimeError("capture did not stop at update1200")
    path = output / f"snapshot-horizon{horizon}.pt"
    torch.save(captured["state"], path)
    return captured["state"], dict(horizon=horizon, raw_snapshot_sha256=captured["sha256"],
        original_hash_exact=True, first200_records_and_corrections_exact=True, stopped_before_ema_at_update=1200,
        snapshot_file=path.name, snapshot_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        declaration_sha256=hashlib.sha256((archive / "declaration.json").read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "observer.py").write_bytes(Path(__file__).read_bytes())
    warm, a = run(args.root, args.warm, args.output, 1200)
    hold, b = run(args.root, args.hold, args.output, 2400)
    diff = sorted(differences(warm, hold), key=lambda row: row["path"])
    predicted = {"noise_policy._counts.output_eval_calls": 6,
                 "noise_policy._counts.output_eval_elements": 24648}
    exact = len(diff) == 2 and all(row["kind"] == "value" and row["path"] in predicted
              and row["a"]-row["b"] == predicted[row["path"]] for row in diff)
    value = dict(status="EXACT_PREDICTED_EVALUATION_COUNTER_DIFFERENCE" if exact else "OTHER_DIFFERENCE",
        shared_gate_eligible=False, warm=a, hold=b, differences=diff,
        native_extra_warm_evaluation_steps=[1050, 1100, 1150],
        per_measure_output_calls=2, per_measure_output_elements=2*(4096+12),
        explanation="native live_curve tail depends on recipe.steps; evaluation restores training RNG",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(value, indent=2)+"\n")
    print(json.dumps(value), flush=True)
    if not exact:
        raise RuntimeError("snapshot differs beyond the predicted native evaluation counters")


if __name__ == "__main__":
    main()
