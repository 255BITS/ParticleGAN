"""Independent finite reference and partition isolation checks, CPU only."""
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from fractions import Fraction

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "integration/review/training-regression/support-count/pkg-support-count"
V4 = ROOT / "integration/review/training-regression/pkg-count-recovery"
INPUT = ROOT / "integration/review/training-regression/snapshot-1000.pt"
source_paths = [Path(__file__), SOURCE / "particlegan/feature_cells.py",
                V4 / "particlegan/feature_cells.py", INPUT]
start_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths}


def package(name, path):
    holder = ModuleType(name)
    holder.__path__ = [str(path / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells")


proposal = package("review_count_proposal", SOURCE)
v4 = package("review_count_v4", V4)
payload = torch.load(INPUT, map_location="cpu", weights_only=False)
checks = []


def checked(name, **values):
    checks.append(dict(name=name, status="PASS", **values))


def stream():
    result = torch.Generator(device="cpu")
    result.set_state(payload["planning_rng"])
    return result


# A specified real-feature prefix is enough to test dependency isolation.
# Both fits reuse the exact saved generator byte state; no seed is generated.
real = payload["real_features"][:64].clone()
old_stream, new_stream, changed_stream = stream(), stream(), stream()
old = v4.FeatureCellSnapshot.fit(real, generator=old_stream, cells=6, rank=3, chunk=16)
new = proposal.FeatureCellSnapshot.fit(real, generator=new_stream, cells=6, rank=3, chunk=16)
changed_real = real.clone()
changed_real[1::2] = changed_real[1::2].flip(0) * 13. + 11.
changed = proposal.FeatureCellSnapshot.fit(changed_real, generator=changed_stream,
                                         cells=6, rank=3, chunk=16)
geometry = ("mean", "scale", "basis", "centers", "cell_scale",
            "reference_counts", "real_representatives", "real_representative_rows")
for name in geometry:
    assert torch.equal(getattr(old, name), getattr(new, name)), name
    assert torch.equal(getattr(new, name), getattr(changed, name)), name
assert torch.equal(new.count_boundary, changed.count_boundary)
assert torch.equal(new.reference_category_counts, changed.reference_category_counts)
assert new.count_partition == changed.count_partition
assert torch.equal(old_stream.get_state(), new_stream.get_state())
assert torch.equal(new_stream.get_state(), changed_stream.get_state())
for left, right in zip(old.support(payload["q"][:64]), new.support(payload["q"][:64])):
    assert torch.equal(left, right), "the original support law changed"
query = payload["q"][:64]
frozen_boundary = new.count_boundary.clone()
comparison = new.cell_comparison(query)
new.cell_comparison(query * 19. + 7.)
assert torch.equal(new.count_boundary, frozen_boundary)
assert comparison["multiplicity"] == 2 * new.cells == len(comparison["pvalues"])
assert comparison["cutoff"] == proposal.Q / (2 * new.cells)
assert int(new.reference_category_counts.sum()) == len(real[0::2])
assert int(new.real_calibration_category_counts.sum()) == len(real[1::2])
checked("odd_and_fake_boundary_isolation_original_geometry_support_and_rng",
        fitted_rows=new.count_partition["fitted_rows"],
        ordinal=new.count_partition["ordinal"], boundary=float(new.count_boundary),
        actual_multiplicity=comparison["multiplicity"])

# Exact arithmetic enumerates one pooled 4-category null, including a zero bin.
# This is independent of Torch's log-factorial implementation.
pooled = (6, 6, 4, 0)
real_rows = fake_rows = 8
denominator = math.comb(real_rows + fake_rows, real_rows)
family_probability = Fraction(0)
mass = Fraction(0)
allocations = 0
max_error = 0.
for allocation in itertools.product(*(range(total + 1) for total in pooled)):
    if sum(allocation) != real_rows:
        continue
    allocations += 1
    probability = Fraction(math.prod(math.comb(total, value)
                                    for total, value in zip(pooled, allocation)), denominator)
    mass += probability
    real_counts = torch.tensor(allocation, dtype=torch.long)
    fake_counts = torch.tensor(pooled, dtype=torch.long) - real_counts
    actual = proposal.conditional_count_pvalues(real_counts, fake_counts, real_rows, fake_rows)
    exact = []
    for total, observed in zip(pooled, allocation):
        low, high = max(0, total - fake_rows), min(total, real_rows)
        weights = [math.comb(real_rows, j) * math.comb(fake_rows, total - j)
                   for j in range(low, high + 1)]
        exact.append(Fraction(sum(weight for weight in weights
                                  if weight <= weights[observed - low]),
                              math.comb(real_rows + fake_rows, total)))
    max_error = max(max_error, max(abs(float(want) - float(got))
                                   for want, got in zip(exact, actual)))
    assert max_error < 2e-12
    assert float(actual[-1]) == 1.
    exact_rejected = any(value <= Fraction(1, 80) for value in exact)
    assert bool((actual <= .05 / len(pooled)).any()) == exact_rejected
    if exact_rejected:
        family_probability += probability
assert mass == 1
assert family_probability <= Fraction(1, 20)
checked("exact_conditional_family_reference_with_retained_empty_bin",
        allocations=allocations, pooled_counts=list(pooled),
        family_rejection_probability=float(family_probability),
        family_rejection_probability_exact=str(family_probability),
        max_pvalue_error=max_error, actual_multiplicity=len(pooled))

# Every transformed even score is tied at zero in the degenerate fixture.
constant = torch.ones((6, 2), dtype=torch.float64)
degenerate = proposal.FeatureCellSnapshot.fit(constant, generator=stream(),
                                             cells=4, rank=2, chunk=2)
assert not degenerate.valid_metric and float(degenerate.count_boundary) == 0.
categories = degenerate.count_categories(torch.tensor([[1., 1.], [9., -9.]], dtype=torch.float64))
assert categories.tolist() == [0, 0]
empty_comparison = degenerate.cell_comparison(constant[:2])
assert len(empty_comparison["pvalues"]) == 2 * degenerate.cells == 6
assert bool((empty_comparison["pvalues"] == 1).all())
assert not bool((empty_comparison["excess"] | empty_comparison["deficit"]).any())
assert degenerate.reference_category_counts.tolist() == [3, 0, 0, 0, 0, 0]
checked("minimum_input_ties_empty_cells_and_degenerate_discoveries",
        cells=degenerate.cells, categories=2 * degenerate.cells,
        reference_category_counts=degenerate.reference_category_counts.tolist())

stop_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source_paths}
assert stop_hashes == start_hashes, "a reviewed source changed during the checks"
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", checks=checks, source_sha256=start_hashes,
               cuda_initialized=False, optimizer_updates=0, new_seeds=0,
               scope="Finite partition/count contracts; no action response or learned quality rerun")
(HERE / "count-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
