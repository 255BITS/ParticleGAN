"""Read the completed grid100 JSON evidence without importing Torch."""
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / "validation/screens/runs/grid100"
HARNESS = Path("/ml2/hypergan/lrfree-20260926/harness")
paths = [RUN / name for name in ("result.json", "execution-receipt.json", "metrics.jsonl",
         "native100-diagnostics.jsonl", "rates.jsonl", "native-noisy/verdict.json",
         "native-noisy/config.json")]
paths += [HARNESS / "screen.py", HARNESS / "native100_diagnostics.py",
          HARNESS / "native100_score.py", HARNESS / "tasks/native100_fixture.json",
          ROOT / "pkg-CB64-RA2/particlegan/training.py", Path(__file__)]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
start_hashes = {str(path): sha(path) for path in paths}
result = json.loads((RUN / "result.json").read_text())
execution = json.loads((RUN / "execution-receipt.json").read_text())
verdict = json.loads((RUN / "native-noisy/verdict.json").read_text())
metrics = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines()]
sidecars = {value["step"]: value for value in
            (json.loads(line) for line in (RUN / "native100-diagnostics.jsonl").read_text().splitlines())}
fixture = json.loads((HARNESS / "tasks/native100_fixture.json").read_text())
frozen_repo = Path(fixture["frozen_repo"])
for filename, expected in verdict["sources"].items():
    assert sha(frozen_repo / filename) == expected, filename
    start_hashes[str(frozen_repo / filename)] = expected
assert execution["source_integrity_before"] == execution["source_integrity_after"]
assert execution["source_integrity_after"]["status"] == "VALID"
assert sha(RUN / "result.json") == execution["result_sha256"]


def failed_thresholds(value):
    return [dict(metric=name, value=value[name], operator=operator, bound=bound)
            for name, operator, bound in result["thresholds"]
            if name in value and not (value[name] >= bound if operator == ">=" else value[name] <= bound)]


keys = ("precision", "mass_tv", "max_cov_eig_ratio", "acc_center_rms_sigma",
        "acc_abs_cov_trace_bias", "acc_radial_ks")
by_step = {value["step"]: value for value in metrics}
terminal = []
for check in verdict["accuracy"]["terminal_checks"]:
    step = check["step"]
    value = by_step[step]
    previous = metrics[metrics.index(value)-1]
    prior = value["diag"]["lr_settle"]["g1"]
    birth_death = value["diag"]["birth_death"]
    affine = sidecars[step]["affine_motion_v1"]
    terminal.append(dict(step=step, passed=check["passed"],
                         live={name: value[name] for name in keys},
                         ema={name: value["ema"][name] for name in keys},
                         failed_thresholds=failed_thresholds(value),
                         ema_failed_thresholds=failed_thresholds(value["ema"]),
                         prior_scale=prior["s"], prior_block=prior["b"],
                         prior_last_decision=prior["last"],
                         excluded_rows=prior["excluded_rows"], rebases=prior["counts"]["rebases"],
                         lrs=value["lr"], output_sigma=value["output_sigma"],
                         ordinary_moves_in_previous_250_steps=birth_death["counters"]["moves"]
                             - previous["diag"]["birth_death"]["counters"]["moves"],
                         isolation_moves_in_previous_250_steps=birth_death["counters"]["iso_moves"]
                             - previous["diag"]["birth_death"]["counters"]["iso_moves"],
                         iso_flags_at_check=birth_death["last"]["iso_flagged"],
                         affine_motion={name: affine.get(name) for name in
                            ("rms_prior_sigma", "rms_generator_sigma", "rms_total_sigma",
                             "rms_mode_mean_sigma", "rms_within_mode_sigma", "lag1_lineage_valid",
                             "lag1_row_motion_cosine", "lag1_mean_row_cosine")}))
assert [value["passed"] for value in terminal] == [False, False, False, True, True]
assert result["passing_checks"] == sum(bool(value["pass"]) for value in metrics) == 3
assert result["native"]["coverage_stable_checks"] == 2
timing = []
for value in metrics:
    if value["step"] < 4750:
        continue
    previous = metrics[metrics.index(value)-1]
    birth_death = value["diag"]["birth_death"]
    timing.append(dict(step=value["step"], seconds=value["seconds"],
                       interval_seconds=value["seconds"]-previous["seconds"],
                       interval_steps=value["step"]-previous["step"],
                       last_birth_death_seconds=birth_death["last"]["eval_seconds"],
                       last_work=birth_death["last"]["work"]))
receipt = dict(scope="Completed saved grid100 JSON evidence only; no new trajectory or gate",
               status="ANALYZED", verdict="FAIL", observations=len(metrics),
               passing_observation_steps=[value["step"] for value in metrics if value["pass"]],
               terminal=terminal, terminal_pass_count=sum(value["passed"] for value in terminal),
               required_terminal_pass_count=verdict["coverage"]["required_stable_checks"],
               holdout=verdict["accuracy"]["holdout_metrics"], final=result["final"],
               timing=timing, full_train_seconds=result["train_seconds"],
               source_sha256=start_hashes,
               source_integrity_validated_before_and_after=True,
               new_trajectories=0, new_seeds=0, optimizer_updates=0,
               torch_imported=False, cuda_context_created=False,
               inference="Late prior stationarity changes the served pair to EMA, explaining the terminal improvement; saved intervals also show residual within-mode row motion. Causal attribution of transient wall-time variation is unavailable from these JSONs.")
assert start_hashes == {name: sha(Path(name)) for name in start_hashes}
(HERE / "grid100-analysis.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({name: receipt[name] for name in ("status", "verdict", "observations", "passing_observation_steps",
                 "terminal_pass_count", "required_terminal_pass_count", "full_train_seconds", "torch_imported",
                 "cuda_context_created")}))
