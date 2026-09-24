"""Two-bank free-output falsifier for persistent data-only support groups."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
from reports.toy100.sample_group_anchor import output_mm_step
from reports.toy100.sample_group_memory import PersistentGroupMemory


SOURCES = (
    "reports/toy100/sample_group_memory.py",
    "reports/toy100/sample_group_memory_filter.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/coverage_fixed_eval.py",
    "benchmarks/locked_shared/mode_hold.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = args.input.read_bytes()
    source = json.loads(raw)
    assert source["scope"].startswith("free-output conditional-minibatch diagnostic")
    points = torch.tensor(source["points"], dtype=torch.float32)
    real_full = torch.tensor(source["branches"]["ordinary"]["real128"], dtype=torch.float32)
    real_omit = torch.tensor(source["branches"]["component0_absent"]["real128"], dtype=torch.float32)
    indices, noise = fixed_draw(2401, points)
    grade = lambda support: score_support(support.float(), indices, noise, mode_hold.ring_means())
    assert grade(points)["modes"] == 8 and grade(points)["hq"] == 1.0
    assert source["branches"]["component0_absent"]["grade"]["modes"] == 7

    rng_before = torch.random.get_rng_state().clone()
    established = PersistentGroupMemory()
    first_centers, first = established.observe(real_full)
    saved_center = first_centers.clone()
    remembered, second = established.observe(real_omit)
    assert len(first_centers) == len(remembered) == 8
    assert second["current_groups"] == 7 and not second["added"]
    assert len(second["absent_remembered"]) == 1
    absent_index = second["absent_remembered"][0]
    assert torch.equal(remembered[absent_index], saved_center[absent_index])
    target = torch.tensor(output_mm_step(points.double(), remembered)["target"])
    retained_grade = grade(target)
    assert retained_grade["modes"] == 8 and retained_grade["hq"] >= .9

    bootstrap = PersistentGroupMemory()
    seven_centers, seven = bootstrap.observe(real_omit)
    assert len(seven_centers) == 7
    seven_points = torch.tensor(output_mm_step(points.double(), seven_centers)["target"])
    seven_grade = grade(seven_points)
    assert seven_grade["modes"] == 7
    serialized = bootstrap.state_dict()
    restored = PersistentGroupMemory()
    restored.load_state_dict(serialized)
    assert torch.equal(restored.centers(), bootstrap.centers())
    discovered, next_row = restored.observe(real_full)
    assert len(discovered) == 8 and len(next_row["added"]) == 1
    assert len(next_row["absent_remembered"]) == 0
    first_discovery_step = torch.tensor(output_mm_step(seven_points.double(), discovered)["target"])
    first_discovery_grade = grade(first_discovery_step)
    # The first active quadratic can land an anchor between old and new
    # centers. The next MM step on the same frozen data field resolves it.
    acquired = torch.tensor(output_mm_step(first_discovery_step.double(), discovered)["target"])
    acquired_grade = grade(acquired)
    assert first_discovery_grade["modes"] == 7
    assert acquired_grade["modes"] == 8 and acquired_grade["hq"] >= .9

    exact_support = torch.cat((remembered, remembered[:4]))
    stationary = torch.tensor(output_mm_step(exact_support, remembered)["target"], dtype=torch.float64)
    assert float((stationary-exact_support).abs().max()) < 1e-12
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    report = dict(scope="two archived banks, free-output only; no native GAN training",
        input_file_sha256=hashlib.sha256(raw).hexdigest(),
        input_state_sha256=source["input_state_sha256"],
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        established_then_omitted=dict(first=first, second=second,
            absent_centroid_bitwise_preserved=True, baseline_absent_grade=source["branches"]["component0_absent"]["grade"],
            retained_grade=retained_grade),
        initially_incomplete_then_discovered=dict(first=seven, second=next_row,
            before_grade=seven_grade, first_mm_grade=first_discovery_grade,
            second_mm_grade=acquired_grade,
            serialized_state_exact=True),
        exact_cached_support_rest=True, torch_rng_unchanged=True,
        method="cached_group_sample_sums_and_counts; nearest match within half cached minimum intercenter separation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(known_then_omitted=retained_grade,
                          incomplete_then_discovered=dict(before=seven_grade,
                            first_mm=first_discovery_grade,second_mm=acquired_grade)),indent=2))


if __name__ == "__main__":
    main()
