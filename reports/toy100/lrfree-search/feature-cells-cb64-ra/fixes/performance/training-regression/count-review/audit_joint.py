"""Independent CPU audit of the overlapping count law and joint action ledger."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
from types import ModuleType
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
JOINT = ROOT / "integration/review/training-regression/joint-count"
V4 = ROOT / "integration/review/training-regression/pkg-count-recovery"
TWO = ROOT / "integration/review/training-regression/support-count/pkg-support-count"
paths = [Path(__file__), JOINT / "pkg-joint-count/particlegan/feature_cells.py",
         JOINT / "inputs.pt", JOINT / "prepare-inputs.json", JOINT / "PROTOCOL.md",
         V4 / "particlegan/feature_cells.py", TWO / "particlegan/feature_cells.py"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
before = {str(p): sha(p) for p in paths}


def package(name, path):
    holder = ModuleType(name)
    holder.__path__ = [str(path / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells")


joint = package("independent_joint", JOINT / "pkg-joint-count")
v4 = package("independent_joint_v4", V4)
two = package("independent_joint_two", TWO)
inputs = torch.load(JOINT / "inputs.pt", map_location="cpu", weights_only=False)
cases = inputs["cases"]
fallback_rng = cases["saved_toy_1000"]["planning_rng"]
checks = []


def checked(name, **kw):
    checks.append(dict(name=name, status="PASS", **kw))


def stream(value=None):
    return torch.Generator(device="cpu").set_state(
        fallback_rng if value is None or value.get("planning_rng") is None else value["planning_rng"])


def restore(module, value):
    result = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    result.__dict__.update(deepcopy(value["snapshot"]))
    return result


# Independently mutate odd rows while preserving even rows and the saved stream.
real = cases["saved_toy_1000"]["real_features"][:64].clone()
old_rng, new_rng, changed_rng = stream(), stream(), stream()
old = v4.FeatureCellSnapshot.fit(real, generator=old_rng, cells=6, rank=3, chunk=16)
new = joint.FeatureCellSnapshot.fit(real, generator=new_rng, cells=6, rank=3, chunk=16)
changed_real = real.clone()
changed_real[1::2] = changed_real[1::2].flip(0) * 13. + 11.
changed = joint.FeatureCellSnapshot.fit(changed_real, generator=changed_rng, cells=6, rank=3, chunk=16)
geometry = ("mean", "scale", "basis", "centers", "cell_scale", "reference_counts",
            "real_representatives", "real_representative_rows")
for name in geometry:
    assert torch.equal(getattr(old, name), getattr(new, name)), name
    assert torch.equal(getattr(new, name), getattr(changed, name)), name
assert torch.equal(new.count_boundary, changed.count_boundary)
assert torch.equal(new.reference_category_counts, changed.reference_category_counts)
assert new.count_partition == changed.count_partition
assert torch.equal(old_rng.get_state(), new_rng.get_state())
assert torch.equal(new_rng.get_state(), changed_rng.get_state())
for a, b in zip(old.support(cases["saved_toy_1000"]["q"][:64]),
                new.support(cases["saved_toy_1000"]["q"][:64])):
    assert torch.equal(a, b)
boundary = new.count_boundary.clone()
new.cell_comparison(cases["saved_toy_1000"]["q"][:64] * 19. + 7.)
assert torch.equal(boundary, new.count_boundary)
checked("even_only_boundary_original_geometry_support_and_rng", cells=new.cells,
        fitted_rows=new.count_partition["fitted_rows"], ordinal=new.count_partition["ordinal"])


def exact_p(real_count, fake_count, real_rows, fake_rows):
    total = real_count + fake_count
    low, high = max(0, total - fake_rows), min(total, real_rows)
    weights = [math.comb(real_rows, j) * math.comb(fake_rows, total - j)
               for j in range(low, high + 1)]
    observed = weights[real_count - low]
    return Fraction(sum(w for w in weights if w <= observed), math.comb(real_rows + fake_rows, total))


# The coarse K family overlaps its 2K refinement. Enumerate the joint sample
# allocation, rather than pretending the concatenated counts form one sample.
nulls = []
pooled = (6, 6, 4, 0)
for real_rows, fake_rows in ((8, 8), (5, 11)):
    mass, rejection = Fraction(0), Fraction(0)
    allocations, max_error = 0, 0.
    for allocation in itertools.product(*(range(n + 1) for n in pooled)):
        if sum(allocation) != real_rows:
            continue
        allocations += 1
        probability = Fraction(math.prod(math.comb(n, x) for n, x in zip(pooled, allocation)),
                               math.comb(real_rows + fake_rows, real_rows))
        mass += probability
        rc = torch.tensor(allocation, dtype=torch.long)
        fc = torch.tensor(pooled, dtype=torch.long) - rc
        coarse_rc, coarse_fc = rc.reshape(2, 2).sum(1), fc.reshape(2, 2).sum(1)
        got = torch.cat((joint.conditional_count_pvalues(coarse_rc, coarse_fc, real_rows, fake_rows),
                         joint.conditional_count_pvalues(rc, fc, real_rows, fake_rows)))
        exact = [exact_p(int(r), int(f), real_rows, fake_rows)
                 for r, f in zip(torch.cat((coarse_rc, rc)), torch.cat((coarse_fc, fc)))]
        max_error = max(max_error, max(abs(float(a) - float(b)) for a, b in zip(got, exact)))
        reject = any(p <= Fraction(1, 120) for p in exact)
        assert bool((got <= .05 / 6).any()) == reject
        if reject:
            rejection += probability
        assert int(rc.sum()) == int(coarse_rc.sum()) == real_rows
        assert int(fc.sum()) == int(coarse_fc.sum()) == fake_rows
    assert mass == 1 and rejection <= Fraction(1, 20) and max_error < 2e-12
    nulls.append(dict(real_rows=real_rows, fake_rows=fake_rows, allocations=allocations,
                      pooled_categories=list(pooled), hypotheses=6,
                      rejection_probability=float(rejection), rejection_probability_exact=str(rejection),
                      max_pvalue_error=max_error))
checked("exact_overlapping_K_plus_2K_null_with_unequal_sample_sizes_and_empty_bin", nulls=nulls)


def comparison_contract(snap, value, cmp):
    k = snap.cells
    assert cmp["multiplicity"] == 3 * k == len(cmp["pvalues"])
    assert cmp["cutoff"] == .05 / (3 * k) and cmp["family_sizes"] == (k, 2 * k)
    for family, size in ((cmp["mass"], k), (cmp["support"], 2 * k)):
        assert family["real_counts"].shape == family["fake_counts"].shape == (size,)
        assert int(family["real_counts"].sum()) == snap.calibration_rows
        assert int(family["fake_counts"].sum()) == len(value["fake_features"])
        p = joint.conditional_count_pvalues(family["real_counts"], family["fake_counts"],
                                           snap.calibration_rows, len(value["fake_features"]))
        difference = family["fake_counts"].double() / len(value["fake_features"])
        difference -= family["real_counts"].double() / snap.calibration_rows
        assert torch.equal(p, family["pvalues"]) and torch.equal(difference, family["difference"])
        assert torch.equal(family["excess"], (p <= cmp["cutoff"]) & (difference > 0))
        assert torch.equal(family["deficit"], (p <= cmp["cutoff"]) & (difference < 0))


def target_reference(snap, n):
    counts = snap.reference_counts + snap.real_calibration_counts
    denominator = int(counts.sum())
    targets = [n * int(c) // denominator for c in counts]
    order = sorted(range(len(targets)), key=lambda i: (-(n * int(counts[i]) % denominator), i))
    for i in order[:n - sum(targets)]:
        targets[i] += 1
    return torch.tensor(targets, dtype=torch.long)


def check_plan(name, value):
    snap = restore(joint, value)
    flags, pvalues = value["flags"].clone(), value["pvalues"].clone()
    cmp = snap.cell_comparison(value["fake_features"])
    comparison_contract(snap, value, cmp)
    rng = stream(value)
    child, parent, detail = snap.ordinary_transport(value["q"], flags, cmp, generator=rng, pvalues=pvalues)
    n, k = len(value["q"]), snap.cells
    ids, categories = detail["query_cell_ids"], detail["query_category_ids"]
    count = lambda rows: torch.bincount(ids[rows], minlength=k)
    assert len(child) == len(parent) <= math.floor(.05 * n)
    assert len(torch.unique(child)) == len(child) and len(torch.unique(parent)) == len(parent)
    assert not set(child.tolist()) & set(parent.tolist())
    assert not bool(flags[parent].any()) and bool((pvalues[parent] > .05).all())
    target = target_reference(snap, n)
    assert torch.equal(target, detail["target_counts"])
    clean = count((~flags).nonzero().flatten())
    after = clean - count(child[~flags[child]]) + count(parent)
    assert torch.equal(after, detail["planned_supported_counts"])
    assert bool((after <= torch.maximum(clean, target)).all())
    assert bool((snap._group_counts(after) <= torch.maximum(snap._group_counts(clean), snap._group_counts(target))).all())
    mc, mp = detail["mass_children"], detail["mass_parents"]
    sc, sp = detail["support_children"], detail["support_parents"]
    # The mass phase must be exactly v4 at the corrected evidence and stream.
    baseline, baseline_rng = restore(v4, value), stream(value)
    bc, bp, bd = baseline.ordinary_transport(value["q"], flags, cmp["mass"],
                                            generator=baseline_rng, pvalues=pvalues)
    assert torch.equal(mc, bc) and torch.equal(mp, bp)
    for field in ("death_allocation", "birth_allocation", "target_counts", "planned_supported_counts"):
        assert torch.equal(detail["mass_phase"][field], bd[field]), field
    assert bool(cmp["mass"]["excess"][ids[mc]].all())
    assert bool(cmp["mass"]["deficit"][ids[mp]].all())
    if detail["support_phase"]["ran"]:
        phase = detail["support_phase"]
        assert bool(flags[sc].all()) and bool((categories[sc] % 2 == 1).all())
        assert bool((categories[sp] % 2 == 0).all())
        assert bool(cmp["support"]["excess"][categories[sc]].all())
        assert bool(cmp["support"]["deficit"][categories[sp]].all())
        assert not (set(sc.tolist()) | set(sp.tolist())) & (set(mc.tolist()) | set(mp.tolist()))
        raw_death = (n * cmp["support"]["difference"][1::2].clamp_min(0) + 1e-10).floor().long()
        raw_death *= cmp["support"]["excess"][1::2]
        raw_birth = (n * (-cmp["support"]["difference"][0::2]).clamp_min(0) + 1e-10).floor().long()
        raw_birth *= cmp["support"]["deficit"][0::2]
        spent_death = torch.bincount(categories[mc], minlength=2*k).reshape(k, 2)[:, 1]
        spent_birth = torch.bincount(categories[mp], minlength=2*k).reshape(k, 2)[:, 0]
        for direction, raw, spent, allocation in (("death", raw_death, spent_death, count(sc)),
                                                  ("birth", raw_birth, spent_birth, count(sp))):
            assert torch.equal(phase[f"raw_certified_{direction}_capacity"], raw)
            assert torch.equal(phase[f"spent_certified_{direction}_capacity"], spent)
            assert torch.equal(phase[f"residual_certified_{direction}_capacity"], (raw-spent).clamp_min(0))
            assert bool((allocation <= (raw-spent).clamp_min(0)).all())
    ordinary_rng = rng.get_state().clone()
    iso_child, iso_parent, iso = snap.select_parents(value["q"], flags, ordinary_children=child,
                                                  ordinary_parents=parent, generator=rng, pvalues=pvalues)
    assert iso["guard_passed"] == (0 < int(flags.sum()) <= .05*n)
    if iso["guard_passed"]:
        assert torch.equal(iso["kept_counts"], after)
        assert bool(flags[iso_child].all()) and not bool(flags[iso_parent].any())
        assert bool((pvalues[iso_parent] > .05).all())
        all_child, all_parent = torch.cat((child, iso_child)), torch.cat((parent, iso_parent))
        assert len(torch.unique(all_child)) == len(all_child)
        assert len(torch.unique(all_parent)) == len(all_parent)
        assert not set(all_child.tolist()) & set(all_parent.tolist())
        final = after + count(iso_parent)
        assert bool((snap._group_counts(final) <= torch.maximum(snap._group_counts(clean), snap._group_counts(target))).all())
    else:
        assert len(iso_child) == len(iso_parent) == 0
    # A zero joint budget must bind both phases; it is not a second budget.
    limited = restore(joint, value)
    zc, zp, zd = limited.ordinary_transport(value["q"], flags, cmp, generator=stream(value),
                                          pvalues=pvalues, max_moves=0)
    assert len(zc) == len(zp) == zd["mass_moves"] == zd["support_moves"] == 0
    return dict(name=name, mass_moves=len(mc), support_moves=len(sc), ordinary_moves=len(child),
                isolation_moves=len(iso_child), flags=int(flags.sum()),
                discoveries=int((cmp["pvalues"] <= cmp["cutoff"]).sum()),
                zero_budget_enforced=True, v4_mass_phase_exact=True,
                support_certificate_reservations_valid=True, supported_ledger_exact=True,
                count_sample_scope=value.get("count_sample_scope", "saved emitted replay")), detail, snap


case_rows = []
for name, value in cases.items():
    result, _, _ = check_plan(name, value)
    case_rows.append(result)
assert next(r for r in case_rows if r["name"] == "supported_mass_imbalance")["mass_moves"] > 0
assert next(r for r in case_rows if r["name"] == "supported_balanced_control")["ordinary_moves"] == 0
checked("saved_actions_corrected_v4_mass_phase_exact_targets_shared_rows_and_supported_ledger", cases=case_rows)


# A specified two-cell stress case spends the complete left refined birth
# certificate in mass transport, while left parents and vacancies remain.
def cloud(center, rows, width):
    return torch.stack((torch.linspace(center-width, center+width, rows, dtype=torch.float64),
                        torch.zeros(rows, dtype=torch.float64)), 1)


even = torch.cat((cloud(-3., 154, .04), cloud(3., 358, .04)))
odd = torch.cat((cloud(-3., 26, .017), cloud(3., 486, .017)))
real = torch.empty((1024, 2), dtype=torch.float64)
real[0::2], real[1::2] = even, odd
fit_rng = stream()
stress = joint.FeatureCellSnapshot.fit(real, generator=fit_rng, cells=2, rank=2, chunk=256)
left, right, outside = (torch.tensor([point], dtype=torch.float64) for point in ((-3., 0.), (3., 0.), (6., 0.)))
q = torch.cat((left.expand(64, -1), right.expand(256, -1), outside.expand(704, -1))).clone()
fake = torch.cat((left.expand(20, -1), right.expand(400, -1), outside.expand(604, -1))).clone()
flags, pvalues, _ = stress.support(q)
assert int(flags.sum()) == 704 and bool((pvalues[:320] > .05).all())
stress.cache_queries(q)
value = dict(snapshot=deepcopy(vars(stress)), q=q, flags=flags, pvalues=pvalues,
             fake_features=fake, planning_rng=fit_rng.get_state(),
             count_sample_scope="specified deterministic planner contract, not a quality trajectory")
result, detail, snap = check_plan("exhausted_refined_birth_certificate", value)
left_cell = int(snap.assign(left)[0][0])
phase = detail["support_phase"]
assert detail["mass_moves"] == 32 and detail["support_moves"] == 19
assert int(phase["raw_certified_birth_capacity"][left_cell]) == 32
assert int(phase["spent_certified_birth_capacity"][left_cell]) == 32
assert int(phase["residual_certified_birth_capacity"][left_cell]) == 0
assert int(phase["birth_allocation"][left_cell]) == 0
assert int(phase["eligible_parent_counts"][left_cell]) == 32
assert int(phase["clean_vacancies"][left_cell]) == 84
checked("mass_exhausts_refined_certificate_with_unused_parents_and_vacancy",
        case=result, left_cell=left_cell, raw_birth_capacity=32, mass_spent_birth_capacity=32,
        support_left_births=0, unused_left_parents=32, remaining_left_vacancy=84)


constant = torch.ones((6, 2), dtype=torch.float64)
degenerate = joint.FeatureCellSnapshot.fit(constant, generator=stream(), cells=4, rank=2, chunk=2)
cmp = degenerate.cell_comparison(constant[:2])
assert degenerate.cells == 3 and cmp["multiplicity"] == 9 and len(cmp["pvalues"]) == 9
assert not degenerate.valid_metric and float(degenerate.count_boundary) == 0.
assert bool((cmp["pvalues"] == 1).all())
for family in (cmp["mass"], cmp["support"]):
    assert not bool((family["excess"] | family["deficit"]).any())
checked("degenerate_ties_empty_categories_actual_K_controls_multiplicity", cells=3, hypotheses=9)

after = {str(p): sha(p) for p in paths}
assert after == before, "reviewed source or fixed inputs changed during audit"
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", checks=checks, source_sha256=before,
               cuda_initialized=False, new_seeds=0, optimizer_updates=0,
               scope="Partition, conditional null and action contracts; no learned quality claim")
(HERE / "joint-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
