"""Isolated high-z block-size probe for the exact PR217 nearest search.

Run with CUDA_VISIBLE_DEVICES=1 /tmp/pr38-default-env/bin/python
reports/pr217-nearest-block-bench.py. This monkeypatches only this process's
_NEAREST_FLOOR; the candidate package and PR checkouts are not edited.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import torch


HARNESS = Path('/ml2/hypergan/lrfree-20260926')
PACKAGE = HARNESS / 'candidates/pr217-dv12-exact/package'
OUT = HARNESS / 'reports/pr217-nearest-block-bench.json'
sys.path.insert(0, str(PACKAGE))
from particlegan import particle_prior  # noqa: E402


def block_layout(rows, count, z_dim, floor_elems):
    budget = min(particle_prior._NEAREST_BLOCK['cuda'],
                 max(rows * min(count, 4096), floor_elems))
    pairs = max(1, budget // (z_dim + 1))
    step = min(count, max(pairs // rows, __import__('math').isqrt(pairs), 1))
    query_step = max(1, pairs // step)
    return dict(budget_elements=budget, budget_mib_fp32=budget * 4 / 2**20,
                center_step=step, row_step=query_step,
                center_blocks=-(-count // step), row_blocks=-(-rows // query_step))


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    rows, count, z_dim = 64, 65536, 128
    rng = torch.Generator(device=device).manual_seed(217155)
    table = torch.randn((count, z_dim), generator=rng, device=device)
    idx = torch.randint(count, (rows,), generator=rng, device=device)
    latent = table[idx].contiguous()
    variants = [1 << 20, 1 << 22, 1 << 23, 1 << 24]
    original_floor = particle_prior._NEAREST_FLOOR
    samples = {str(v): [] for v in variants}
    reference = None
    correctness = {}

    try:
        for floor in variants:
            particle_prior._NEAREST_FLOOR = floor
            result = particle_prior._nearest_other(latent, table)
            torch.cuda.synchronize()
            if reference is None:
                reference = result.clone()
            diff = (result - reference).abs()
            correctness[str(floor)] = dict(
                bitwise_equal=bool(torch.equal(result, reference)),
                unequal_rows=int((result != reference).sum().item()),
                max_abs_difference=float(diff.max().item()),
                max_ulp_difference=int((result.view(torch.int32) -
                                        reference.view(torch.int32)).abs().max().item()),
            )

        # Each order contains all arms, to reduce sensitivity to shared load.
        for cycle in range(4):
            order = variants[cycle:] + variants[:cycle]
            for floor in order:
                particle_prior._NEAREST_FLOOR = floor
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                start.record()
                result = particle_prior._nearest_other(latent, table)
                end.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000
                samples[str(floor)].append(dict(
                    cycle=cycle, cuda_event_ms=start.elapsed_time(end), wall_ms=wall_ms,
                    peak_extra_allocated_mib=(torch.cuda.max_memory_allocated(device) - baseline) / 2**20,
                    output_checksum=float(result.sum().item()),
                ))
    finally:
        particle_prior._NEAREST_FLOOR = original_floor

    summary = {}
    for floor in variants:
        key = str(floor)
        summary[key] = dict(layout=block_layout(rows, count, z_dim, floor),
                            cuda_event_median_ms=statistics.median(x['cuda_event_ms'] for x in samples[key]),
                            wall_median_ms=statistics.median(x['wall_ms'] for x in samples[key]),
                            peak_extra_allocated_mib=max(x['peak_extra_allocated_mib'] for x in samples[key]),
                            correctness=correctness[key])
    payload = dict(source_commit='261fcfd1decba067b9c95123abf21e046dc32127',
                   package_path=str(PACKAGE), device=torch.cuda.get_device_name(device),
                   torch_version=torch.__version__, shape=dict(rows=rows, count=count, z_dim=z_dim),
                   dtype='float32', same_process=True, candidate_package_unchanged=True,
                   notes='The floor is changed only in this Python process. Existing z<=32 behavior is unchanged by the proposed conditional rule.',
                   summary=summary, samples=samples)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(path=str(OUT), summary=summary), indent=2))


if __name__ == '__main__':
    main()
