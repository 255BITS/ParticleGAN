"""One conditioned scarce-group bank from an archived passing-state data stream.

This tests a single false-birth mechanism in the frozen RMS-overlap rule.
Ground-truth component labels define the conditional bank and grade only;
the memory receives unlabeled coordinates. No neural or optimizer step runs.
"""

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
from reports.toy100.sample_anchor_free1200 import initial_support
from reports.toy100.sample_group_anchor import mst_groups, output_mm_step
from reports.toy100.sample_group_dispersion_memory import DispersionGroupMemory
from reports.toy100.pr84_critic_refinement_capture import _sha


SOURCES = (
    "reports/toy100/sample_group_singleton_filter.py",
    "reports/toy100/sample_group_dispersion_memory.py",
    "reports/toy100/sample_group_anchor.py",
    "reports/toy100/sample_anchor_free1200.py",
    "reports/toy100/coverage_fixed_eval.py",
    "benchmarks/locked_shared/mode_hold.py",
)


def grade(points, index, noise, means):
    return score_support(points.float(), index, noise, means)


def counts(points, means):
    return torch.bincount(torch.cdist(points.float(), means).argmin(1), minlength=len(means)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state_bytes = args.state.read_bytes()
    saved = torch.load(args.state, weights_only=True, map_location="cpu")
    previous_bytes = args.previous.read_bytes()
    previous = json.loads(previous_bytes)
    assert hashlib.sha256(state_bytes).hexdigest() == previous["input_file_sha256"]
    assert _sha(saved) == previous["input_state_sha256"]
    points = initial_support(saved).double()
    means = mode_hold.ring_means()
    index, noise = fixed_draw(2401, points.float())
    stream = torch.Generator().set_state(saved["rng"]["data"])
    ordinary = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, stream)
    assert torch.equal(ordinary, torch.tensor(previous["branches"]["ordinary"]["real128"]))
    torch_rng_before = torch.random.get_rng_state().clone()
    memory = DispersionGroupMemory()
    full_centers, initial = memory.observe(ordinary)
    assert len(full_centers) == 8
    # The component index is diagnostic only, after the memory has consumed
    # the unlabeled full bank. It never enters observe() or output_mm_step().
    cached_mode = int(torch.cdist(full_centers, means[:1].double()).argmin())
    cached_rms = memory.radii()[cached_mode]
    position = int(torch.randint(0, 128, (1,), generator=stream))
    labels = torch.randint(1, 8, (127,), generator=stream)
    labels = torch.cat((labels[:position], torch.zeros(1,dtype=torch.long), labels[position:]))
    attempts = 0
    while True:
        attempts += 1
        perturbation = torch.randn(128, 2, generator=stream)
        scarce = means[labels] + mode_hold.SIGMA * perturbation
        deviation = float(torch.linalg.vector_norm(scarce[position].double()-full_centers[cached_mode]))
        if deviation > cached_rms:
            break
        if attempts > 100:
            raise RuntimeError("unexpectedly many conditional singleton-tail draws")
    current_centers, current_groups = mst_groups(scarce)
    tentative_target = torch.tensor(output_mm_step(points, full_centers)["target"],dtype=torch.float64)
    after, update = memory.observe(scarce)
    ordinary_target = torch.tensor(output_mm_step(points, current_centers)["target"],dtype=torch.float64)
    memory_target = torch.tensor(output_mm_step(points, after)["target"],dtype=torch.float64)
    result = dict(scope="one exact saved-stream full bank then one conditioned singleton-tail bank; free output only",
        state_file_sha256=hashlib.sha256(state_bytes).hexdigest(), input_state_sha256=_sha(saved),
        previous_result_sha256=hashlib.sha256(previous_bytes).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        conditioning=dict(singleton_component=0,singleton_position=position,
            singleton_label_event_probability=128*(1/8)*(7/8)**127,
            tail_condition="singleton coordinate outside its cached group's empirical RMS disk",
            accepted_noise_bank_attempt=attempts,
            singleton_distance_to_cached=deviation,cached_rms=cached_rms),
        initial=dict(cache=initial, grade=grade(points,index,noise,means),
                     nearest_mode_counts=counts(points,means)),
        current_bank=dict(mst_groups=current_groups["n_groups"],
                          singleton_group_index=int(torch.cdist(current_centers,scarce[position:position+1].double()).argmin()),
                          data_rng_after_sha256=hashlib.sha256(stream.get_state().numpy().tobytes()).hexdigest()),
        memory_update=update,
        false_birth=len(after)>len(full_centers),
        tentative_unconfirmed_target=dict(grade=grade(tentative_target,index,noise,means),
                                         nearest_mode_counts=counts(tentative_target,means)),
        current_bank_only_target=dict(grade=grade(ordinary_target,index,noise,means),
                                      nearest_mode_counts=counts(ordinary_target,means)),
        memory_target=dict(grade=grade(memory_target,index,noise,means),
                           nearest_mode_counts=counts(memory_target,means),
                           target_occupied_cached_groups=len(set(output_mm_step(points,after)["target_occupied_groups"]))),
        torch_rng_unchanged=torch.equal(torch.random.get_rng_state(),torch_rng_before))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(false_birth=result["false_birth"],
        current_groups=result["current_bank"]["mst_groups"],
        cache_after=update["remembered_groups"],
        initial=result["initial"]["grade"]["modes"],
        unconfirmed=result["tentative_unconfirmed_target"]["grade"]["modes"],
        current_bank_only=result["current_bank_only_target"]["grade"]["modes"],
        memory_target=result["memory_target"]["grade"]["modes"],
        singleton_distance=deviation,cached_rms=cached_rms)))


if __name__ == "__main__":
    main()
