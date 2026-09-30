"""Read frozen saved states; exercise copied CPU objects without optimization."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from population_certificate import population_test_class

torch.set_num_threads(1)
ROOT = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929")
OUT = Path(__file__).resolve().parent
RA4 = ROOT / "validation-ra4/learned/training/toy/CB64-RA4"
E22 = Path("/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training/toy/E22")
PKG = ROOT / "pkg-CB64-RA4/particlegan"
files = [PKG / name for name in ("continuous.py", "training.py", "row_evidence.py", "recipes.py")]
files += [base / "config.json" for base in (RA4, E22)]
files += [p for base in (RA4, E22) for p in sorted(base.glob("checkpoint-*.pt"))]
files += [OUT / "population_certificate.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def equal(a, b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all()) \
            if a.is_floating_point() else torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


before = {str(p): sha(p) for p in files}
rng_before = torch.get_rng_state().clone()
continuous = module(PKG / "continuous.py", "frozen_ra4_continuous")
row_mod = module(PKG / "row_evidence.py", "frozen_ra4_row_evidence")
Prototype = population_test_class(continuous.SequentialSettleTest)
saved = {}
traces = []
real_equal = []
for name, base in (("RA4", RA4), ("E22", E22)):
    previous_w = None
    copied_lower_bound = torch.zeros(1024, dtype=torch.bool)
    for step in (0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000):
        checkpoint = torch.load(base / f"checkpoint-{step:04d}.pt", map_location="cpu", weights_only=False)
        saved[name, step] = checkpoint
        state, record = checkpoint["trainer"], checkpoint["record"]
        tester, evidence = state["lr_settle"][0][1], state["row_evidence"]
        neff = evidence["W"].square() / evidence["S"].clamp_min(1e-30)
        counts = lambda pairs: [] if not pairs else [int(torch.isfinite(p).sum()) for p in pairs]
        finite = torch.stack(tester["r_b"]) if tester["r_b"] else torch.empty(0, 1024)
        known_copy = torch.zeros_like(copied_lower_bound)
        if step >= 750:
            if previous_w is not None:
                known_copy = evidence["W"] < previous_w
                copied_lower_bound |= known_copy
            previous_w = evidence["W"]
        bd = state["birth_death"]
        traces.append(dict(run=name, step=step, metrics=record["metrics"],
            data_position=checkpoint["data_position"], s=tester["s"], b=tester["b"],
            ema_rate=tester["s"] / (4 * tester["b"]), last_decisive=tester["last_decisive"],
            last=tester["last"], rebases=tester["counts"]["rebases"],
            finite_b_rows=counts(tester["r_b"]), finite_2b_rows=counts(tester["r_2b"]),
            finite_b_at_least_two=int((torch.isfinite(finite).sum(0) >= 2).sum()),
            copied_rows_lower_bound_interval=int(known_copy.sum()),
            copied_rows_lower_bound_since750=int(copied_lower_bound.sum()),
            neff_min=float(neff.min()), neff_median=float(neff.median()), neff_max=float(neff.max()),
            mature_rows=int((neff >= 384).sum()), row_flags=int(evidence["flag"].sum()),
            evidence_valid=evidence["valid"], evidence_counters=evidence["counters"],
            cumulative_moves=bd["counters"]["moves"], latest_moves=bd["last"].get("moves"),
            latest_move_step=bd["last"].get("step"), output_sigma=record["diagnostics"]["output_sigma"],
            generator_scale=state["lr_settle"][0][0]["s"], controller=record["diagnostics"]["controller"]))
for step in (0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000):
    a, b = saved["RA4", step], saved["E22", step]
    ra, rb = a["trainer"]["birth_death"]["reservoir"], b["trainer"]["birth_death"]["reservoir"]
    same = ra is rb if ra is None or rb is None else torch.equal(ra, rb)
    assert same and a["data_position"] == b["data_position"] == 256 * step
    real_equal.append(dict(step=step, reservoir_equal=same, data_position=a["data_position"]))

checks = {}
# Reproduce the actual original behavior on a saved table clone.
old = saved["RA4", 750]["trainer"]["lr_settle"][0][1]
param = torch.nn.Parameter(saved["RA4", 750]["trainer"]["models"]["prior"]["z"].clone())
copied = (saved["RA4", 1000]["trainer"]["row_evidence"]["W"] <
          saved["RA4", 750]["trainer"]["row_evidence"]["W"]).nonzero().flatten()
assert len(copied) == 398
original = continuous.SequentialSettleTest(early_stationary_only=True, final_table=continuous._T9875)
original.load_state_dict(old, param.numel())
original.rebase([param], copied)
assert original.last_decisive == -1 and original.s == .5 and original.b == 16
checks["original_stale_verdict_reproduced"] = dict(rows=398, last_decisive=original.last_decisive,
    s=original.s, b=original.b, tau=original.tau, training_changed=False)

# This best-case bootstrap is mechanical review only; production rejects an
# old law instead of inventing historical participation.
proto = Prototype(param, early_stationary_only=True, final_table=continuous._T9875)
proto.__dict__.update(deepcopy(old))
proto.stationary_rows.fill_(True)
proto.population_active = True
proto.stationary_undo_s = 1.
old_tau, old_b = proto.tau, proto.b
proto.rebase([param], copied)
assert proto.last_decisive == 0 and proto.s == 1. and proto.b == old_b and proto.tau == old_tau
assert proto.population_expiries == 1
proto.rebase([param], copied)
assert proto.population_expiries == 1 and proto.s == 1.
checks["saved_copy_lower_bound_expires_once"] = dict(rows=len(copied), surviving_rows=int(proto.stationary_rows.sum()),
    s=proto.s, b=proto.b, tau=proto.tau, release_rule="undo one accepted descent, not a DRIFT verdict")

table = torch.nn.Parameter(torch.zeros(20, 2))
t = Prototype(table, early_stationary_only=True, final_table=continuous._T9875)
t.begin([table]); t.b = 8.; t.tau = .75
def negative_window(test, missing=None):
    test.r_b = [torch.full((20,), -.2 - .01 * i) for i in range(12)]
    test.r_2b = [torch.full((20,), -.3 - .01 * i) for i in range(6)]
    if missing is not None:
        for pair in test.r_b + test.r_2b:
            pair[missing] = float("nan")
    vb, v2 = test._scale_values()
    return test._final_verdict(vb), test._final_verdict(v2)
sb, s2 = negative_window(t)
assert t._conclude(sb, s2, t.b, 1, False) == "stationary"
assert t.s == .5 and t.population_active
# Keep an unfinished new window across the copies, while invalidating only
# the copied rows in its block/pair history.
t.r_b = [torch.arange(20, dtype=torch.float32)]
t.r_2b = [torch.arange(20, dtype=torch.float32) + .25]
t.blocks = [torch.arange(40, dtype=torch.float32)]
t.blocks_in_window = 1
kept = deepcopy((t.r_b, t.r_2b, t.blocks))
kept_tau, kept_b = t.tau, t.b
# Exactly Q is permitted; cumulative distinct replacement over Q revokes.
t.rebase([table], torch.tensor([0]))
assert t.population_active and t.s == .5
t.rebase([table], torch.tensor([0]))
assert t.population_active and t.s == .5
t.rebase([table], torch.tensor([1]))
assert not t.population_active and t.s == 1. and t.population_expiries == 1
checks["Q_boundary_and_unique_spending"] = dict(population=20, allowed_rows=1, expires_at_distinct_rows=2)
assert t.tau == kept_tau and t.b == kept_b and t.blocks_in_window == 1
assert torch.equal(t.r_b[0][2:], kept[0][0][2:]) and torch.equal(t.r_2b[0][2:], kept[1][0][2:])
assert torch.equal(t.blocks[0].view(20, 2)[2:], kept[2][0].view(20, 2)[2:])
assert torch.isnan(t.r_b[0][:2]).all() and torch.isnan(t.blocks[0].view(20, 2)[:2]).all()
checks["active_window_preserved"] = dict(tau=kept_tau, b=kept_b, pairs=1, quads=1,
    untouched_rows_bitwise_equal=True, moved_rows_invalidated=True)
sb, s2 = negative_window(t, torch.tensor([0, 1]))
before_s, before_b = t.s, t.b
assert t._conclude(sb, s2, t.b, 2, False) == "inconclusive"
assert t.s == before_s and t.b == 2 * before_b and t.population_coverage_rejections == 1
checks["partial_population_cannot_anneal"] = dict(participants=18, required=19, longer_scale_search=True)
sb, s2 = negative_window(t)
assert t._conclude(sb, s2, t.b, 3, False) == "stationary" and t.s == .5
assert t.population_active and t.stationary_rows.all()
snapshot = t.state_dict()
resumed = Prototype(table, early_stationary_only=True, final_table=continuous._T9875)
resumed.load_state_dict(snapshot, table.numel())
assert equal(t.state_dict(), resumed.state_dict())
t.rebase([table], torch.tensor([3, 4])); resumed.rebase([table], torch.tensor([3, 4]))
assert equal(t.state_dict(), resumed.state_dict())
guard = resumed.state_dict()
try:
    resumed.load_state_dict(old, param.numel())
    raise AssertionError("old law accepted")
except ValueError:
    pass
assert equal(guard, resumed.state_dict())
bad = deepcopy(snapshot); bad["stationary_rows"] = bad["stationary_rows"].float()
try:
    resumed.load_state_dict(bad, table.numel())
    raise AssertionError("bad mask accepted")
except ValueError:
    pass
assert equal(guard, resumed.state_dict())
checks["checkpoint_and_continuation"] = dict(exact=True, old_law_atomic_rejection=True,
    bad_mask_atomic_rejection=True, added_tensor_elements=20, added_tensor_dtype="bool")

ev = row_mod.RowEvidence(param, null="scaled")
ev.load_state_dict(saved["RA4", 2000]["trainer"]["row_evidence"])
ev._test()
assert ev.valid and not ev.flag.any()
assert (2. - ev.lam) / ev.lam == 99. and 3 * ev.d == 384
checks["row_gate_impossible"] = dict(d=ev.d, window=1 / ev.lam, neff_cap=99., required_neff=384,
    final_valid=ev.valid, tested_rows=0, flagged_rows=0,
    finite_horizon_note="128 draws among1024rows gives about235 touched steps/row, below384 even without forgetting")
assert torch.equal(rng_before, torch.get_rng_state())
assert before == {str(p): sha(p) for p in files}
checks["frozen_sources_inputs_rng_unchanged"] = dict(pass_=True, cuda_initialized=torch.cuda.is_initialized(),
    optimizer_updates=0, seeds=0)
assert not torch.cuda.is_initialized()

# Save a complete-record snapshot from the existing, still-running native job.
native_path = ROOT / "validation-ra4/screens/runs/grid100/native100-diagnostics.jsonl"
native = []
for line in native_path.read_text().splitlines():
    try:
        v = json.loads(line)
    except json.JSONDecodeError:
        continue
    native.append(dict(step=v["step"], output_sigma=v["output_sigma"], lr=v["lr"], settle=v["settle"],
        prior_motion=v["prior_motion"], affine_motion=v["affine_motion_v1"],
        clean_live={k:v["clouds"]["clean"]["live"][k] for k in ("precision", "trace_ratio", "center_offset_sigma")},
        noisy_live={k:v["clouds"]["noisy"]["live"][k] for k in ("precision", "trace_ratio", "center_offset_sigma")},
        copies=v["birth_death"]["counters"]["moves"], latest_moves=v["birth_death"]["last"].get("moves")))

OUT.mkdir(exist_ok=True, parents=True)
for name, value in (("traces.json", traces), ("held-real-checks.json", real_equal), ("grid-snapshot.json", native)):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
receipt = dict(status="PASS", scope="saved CPU mechanical diagnostics; proposed scheduler law only",
    checks=checks, reviewed_hashes=before, toy_quality_passed=False, native_quality_final=False,
    production_recommendation="Count-certified replacements invalidate population continuity; quality effect of one-descent undo remains untested.",
    limitations=["Participation coverage does not certify each row stationary.",
        "No historical CUDA action replay or gradient trajectory was executed.",
        "A table-only release can increase prior motion; it cannot be claimed to fix the observed quality failure.",
        "Count tests retain the frozen per-comparison conditional law, with no repeated adaptive guarantee.",
        "Serving/noise/EMA formulas and all frozen gates are unchanged by this prototype."])
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=True) + "\n")
print(json.dumps(dict(status="PASS", checks=list(checks), latest_grid_snapshot_step=native[-1]["step"],
    receipt=str(OUT / "receipt.json")), indent=2))
