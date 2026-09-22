"""Frozen comparison with a real fixed-schedule control and readable tables."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch

from benchmarks.locked_shared import baseline
from benchmarks.learned_lr_evaluation import FixedSchedule, control_host_schedules
from . import study


class FixedControl(FixedSchedule):
    """No feature extraction or learned policy work in the baseline arm."""
    def __init__(self, policy, total_steps, *, ablation="none"):
        super().__init__(total_steps, cosine=policy["schedule"] == "cosine")
        self.controller_seconds = 0.

    def regularization_scale(self, role):
        return 1.


def fixed_toy(name, card):
    controller = FixedControl(card, baseline.BUDGETS[name])
    started = time.perf_counter()
    try:
        with control_host_schedules(controller) as timing:
            result = baseline.run_toy(name, study.BASE)
        result["controller_seconds"] = timing["seconds"]
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc())
    result.update(seconds=time.perf_counter() - started, actions=controller.trace)
    return result


def render(report, output):
    lines = ["# Smart descent v2 — frozen comparison", "",
             f"The nine familiar toys are development data. {len(report['frozen']['fresh_transfer'])} transfer task/architecture combinations were declared "
             "before search and first evaluated after the selected policy was frozen. All training uses seed 0. "
             "Live weights determine success; EMA is separate.", "",
             "| Controller | Live bounds | Sustained toys | Mean confirmation / budget | Ring modes / HQ | Ring confirmation | Sum confirmation seconds | Full suite seconds | Controller seconds |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in report["rows"]:
        summary = study.row_summary(row)
        ring = row["toys"].get("mode_hold", {})
        live = ring.get("live", {})
        overhead = sum(t.get("controller_seconds", 0) for t in row["toys"].values())
        confirmation = (sum(t["convergence"]["confirmed_seconds"] for t in row["toys"].values())
                        if summary["stable"] == 9 else None)
        confirmation_text = f"{confirmation:.2f}" if confirmation is not None else "—"
        lines.append(f"| {row['name']} | {summary['bounds']}/29 | {summary['stable']}/9 | {summary['mean_confirmation_fraction']:.3f} | "
                     f"{live.get('modes', '—')}/8 / {live.get('hq', 0):.2%} | "
                     f"{ring.get('convergence', {}).get('confirmed_step') or '—'} | {confirmation_text} | {summary['seconds']:.2f} | {overhead:.2f} |")
    lines += ["", "Confirmation needs five consecutive final passing observations in the complete 24-point curve. "
              "The ring requires 8/8 modes and HQ≥90%. Mean confirmation fractions use 2 for a non-converged toy; "
              "they are meaningful for comparing speed after checking every toy passed.", "",
              "| Fresh transfer task | Controller | Live modes / total | Live HQ | Confirmed step | EMA modes / HQ | Seconds |",
              "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for spec in report["frozen"]["fresh_transfer"]:
        count = spec.get("modes", spec.get("side", 0) ** 2)
        for row in report["rows"]:
            result = row["transfer"].get(spec["name"], {})
            live, ema = result.get("live", {}), result.get("ema", {})
            lines.append(f"| {spec['name']} | {row['name']} | {live.get('modes', '—')}/{count} | {live.get('hq', 0):.2%} | "
                         f"{result.get('convergence', {}).get('confirmed_step') or '—'} | "
                         f"{ema.get('modes', '—')} / {ema.get('hq', 0):.2%} | {result.get('seconds', 0):.2f} |")
    lines += ["", "The cosine arm performs no gradient feature extraction. All other arms retain observation overhead, "
              "including ablations. Times are single CPU observations with setup and metric evaluation; concurrency and system load can affect them. "
              "Controller seconds count feature/action work, except the fixed arm records the full lightweight host callback. "
              "Sum confirmation seconds adds each host's measured time to its fifth final passing observation, including setup/measurement; "
              "later observations were still run to check stability. No early-stop rule is implemented. "
              "No wall-time speedup is established by update counts alone.", "",
              "`bias_only` zeroes four feedback inputs, preserving learned constant offsets and the selected base schedule. "
              "`lr_only` suppresses regularization actions; `reg_only` suppresses LR actions. "
              "`constant_feedback` keeps the frozen coefficients but removes cosine; it is not a refit for that setting.", "",
              "Full curves, actions, errors, exact configs and source hashes: [evaluation.json](evaluation.json).", ""]
    (output / "EVALUATION.md").write_text("\n".join(lines))


def run(output, reference=None):
    torch.set_num_threads(1)
    frozen = json.loads((output / "frozen.json").read_text())
    if (output / "evaluation.json").exists():
        raise FileExistsError("evaluation already exists")
    root = Path(__file__).resolve().parents[2]
    expected_sources = frozen.get("numerical_source_sha256", frozen.get("source_sha256", {}))
    if not expected_sources:
        raise ValueError("frozen policy requires numerical source hashes")
    for name, digest in expected_sources.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"numerical source changed before evaluation: {name}")
    zero = study.policy(torch.zeros(2, 2, 5))
    if frozen["policy"]["schedule"] == "constant":
        cards = [("cosine", zero, "none"), ("constant", {**zero, "schedule": "constant"}, "none"),
                 ("feedback", frozen["policy"], "none"), ("bias_only", frozen["policy"], "bias_only")]
    else:
        cards = [("cosine", zero, "none"), ("feedback", frozen["policy"], "none"),
                 ("bias_only", frozen["policy"], "bias_only"), ("lr_only", frozen["policy"], "lr_only"),
                 ("reg_only", frozen["policy"], "reg_only"),
                 ("constant_feedback", {**frozen["policy"], "schedule": "constant"}, "none")]
    report = dict(protocol=study.fingerprint(), frozen=frozen, rows=[], shared={})
    def save():
        baseline.write_json(output / "evaluation.json", report)
        render(report, output)
    for name, card, ablation in cards:
        row = dict(name=name, policy=card, ablation=ablation, config=study.asdict(study.BASE), toys={}, transfer={})
        report["rows"].append(row)
        print(f"START evaluation {name}", flush=True)
        for toy in baseline.BUDGETS:
            row["toys"][toy] = fixed_toy(toy, card) if name in ("cosine", "constant") else study.run_toy(toy, card, ablation=ablation)
            save()
        for spec in frozen["fresh_transfer"]:
            with ExitStack() as stack:
                if name in ("cosine", "constant"):
                    stack.enter_context(patch.object(study, "GradientFeedback", FixedControl))
                row["transfer"][spec["name"]] = study.transfer_episode(spec, card, ablation=ablation)
            save()
            result = row["transfer"][spec["name"]]
            print(json.dumps(dict(event="TRANSFER", controller=name, task=spec["name"], live=result.get("live"),
                                  convergence=result.get("convergence"), error=result.get("error"))), flush=True)
        print(json.dumps(dict(event="EVALUATED", name=name, **study.row_summary(row))), flush=True)
    if reference:
        from benchmarks.locked_shared.shared_checks import run_shared
        run_shared(reference, report, save)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run(args.output, args.reference)


if __name__ == "__main__":
    main()
