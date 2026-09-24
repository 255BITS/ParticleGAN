"""Four predeclared free-output checks for RMS-overlap support memory."""

import argparse
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
from reports.toy100.sample_group_dispersion_memory import DispersionGroupMemory


SOURCES = (
    "reports/toy100/sample_group_dispersion_memory.py",
    "reports/toy100/sample_group_dispersion_filter.py",
    "reports/toy100/sample_group_memory.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/coverage_fixed_eval.py",
    "benchmarks/locked_shared/mode_hold.py",
)


def one_mm(points, centers):
    return torch.tensor(output_mm_step(points.double(), centers)["target"], dtype=torch.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    raw = args.input.read_bytes()
    source = json.loads(raw)
    points = torch.tensor(source["points"], dtype=torch.float64)
    ordinary = torch.tensor(source["branches"]["ordinary"]["real128"], dtype=torch.float32)
    omitted = torch.tensor(source["branches"]["component0_absent"]["real128"], dtype=torch.float32)
    indices, noise = fixed_draw(2401, points.float())
    grade = lambda value: score_support(value.float(), indices, noise, mode_hold.ring_means())
    rng = torch.random.get_rng_state().clone()

    # Previously observed mode must not be deleted by a single incomplete bank.
    known = DispersionGroupMemory()
    eight, initial = known.observe(ordinary)
    old = eight.clone()
    eight_again, missing = known.observe(omitted)
    assert len(eight) == len(eight_again) == 8 and missing["current_groups"] == 7
    assert not missing["added"] and len(missing["absent_remembered"]) == 1
    absent = missing["absent_remembered"][0]
    assert torch.equal(old[absent], eight_again[absent])
    retained = grade(one_mm(points, eight_again))
    assert retained["modes"] == 8 and retained["hq"] >= .9

    # Incomplete discovery must later add a new group, retaining existing ones.
    incomplete = DispersionGroupMemory()
    seven, initial_incomplete = incomplete.observe(omitted)
    seven_support = one_mm(points, seven)
    assert grade(seven_support)["modes"] == 7
    snapshot = incomplete.state_dict()
    restored = DispersionGroupMemory()
    restored.load_state_dict(snapshot)
    assert torch.equal(restored.centers(), incomplete.centers())
    discovered, discovery = restored.observe(ordinary)
    assert len(discovered) == 8 and len(discovery["added"]) == 1
    first_acquisition = one_mm(seven_support, discovered)
    second_acquisition = one_mm(first_acquisition, discovered)
    assert grade(first_acquisition)["modes"] == 7
    assert grade(second_acquisition)["modes"] == 8 and grade(second_acquisition)["hq"] >= .9

    # An incomplete cache at 0,4 must not absorb a genuinely new center at 1.
    pre = dict(sums=[torch.tensor([0.,0.],dtype=torch.float64),
                     torch.tensor([16.,0.],dtype=torch.float64)],
               counts=[4,4], squared_norm_sums=[torch.tensor(0.,dtype=torch.float64),
                                                torch.tensor(64.,dtype=torch.float64)])
    two_group_bank = torch.tensor([[0.,0.]]*4 + [[1.,0.]]*4, dtype=torch.float32)
    old_rule = PersistentGroupMemory()
    old_rule.load_state_dict(dict(sums=pre["sums"], counts=pre["counts"]))
    _, old_result = old_rule.observe(two_group_bank)
    assert old_result["remembered_groups"] == 2 and not old_result["added"]
    revised = DispersionGroupMemory()
    revised.load_state_dict(pre)
    three, new_result = revised.observe(two_group_bank)
    assert len(three) == 3 and len(new_result["added"]) == 1
    assert torch.equal(three[2], torch.tensor([1.,0.],dtype=torch.float64))

    # The same new group must be discoverable when the cache has only one group.
    singleton = DispersionGroupMemory()
    singleton.load_state_dict(dict(sums=pre["sums"][:1], counts=pre["counts"][:1],
                                   squared_norm_sums=pre["squared_norm_sums"][:1]))
    two, singleton_result = singleton.observe(two_group_bank)
    assert len(two) == 2 and len(singleton_result["added"]) == 1
    assert singleton_result["half_cached_separation"] is None

    # Exact cached-centroid supports are stationary under the free-output MM.
    stationary_support = torch.cat((eight_again, eight_again[:4]))
    stationary = one_mm(stationary_support, eight_again)
    assert float((stationary-stationary_support).abs().max()) < 1e-12
    assert torch.equal(torch.random.get_rng_state(), rng)
    invalid = restored.state_dict()
    invalid["counts"][0] = True
    try:
        DispersionGroupMemory().load_state_dict(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("bool count was accepted")

    result = dict(scope="four free-output/synthetic association checks; no host training",
        input_file_sha256=hashlib.sha256(raw).hexdigest(),
        input_state_sha256=source["input_state_sha256"],
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        established_then_omitted=dict(initial=initial, omitted=missing,
            absent_centroid_bitwise_preserved=True, retained_grade=retained,
            current_bank_only_grade=source["branches"]["component0_absent"]["grade"]),
        incomplete_then_discovered=dict(initial=initial_incomplete, discovered=discovery,
            initial_grade=grade(seven_support), first_mm_grade=grade(first_acquisition),
            second_mm_grade=grade(second_acquisition), serialized_state_exact=True),
        incomplete_cache_new_center=dict(old_half_separation_rule=old_result,
            revised_rms_overlap_rule=new_result, revised_centers=three.tolist()),
        single_cached_group_new_center=dict(result=singleton_result, centers=two.tolist()),
        exact_cached_support_rest=True, torch_rng_unchanged=True,
        malformed_bool_count_rejected=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', omission=retained['modes'],
        rediscovery=grade(second_acquisition)['modes'],
        counterexample=len(three),singleton=len(two))))


if __name__ == "__main__":
    main()
