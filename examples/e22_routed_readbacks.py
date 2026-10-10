"""Profile synchronization as a shared bank is reused at many routing sites.

python -u examples/e22_routed_readbacks.py --device cuda --sites 71 --output /tmp/readbacks.json

This is a small overhead fixture, not a Supra speed or output-quality benchmark.
It keeps full E22 row controls enabled, with checkpointed probe_interval=1000,
so short runs measure updates between structural evaluations. Frozen BF16 hosts
and FP32 trainable owners follow the ordinary two-site example. Every subsequent
site uses hidden inputs changed by the previous site's mixed particle codes.
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch

import e22_routed_sites as sites_example


def many_site_forward(models, context, candidate, routing):
    generator, encoder, router = (models[name] for name in ("generator", "encoder", "router"))
    hidden = generator.first_host(context.bfloat16()).float() + .01 * encoder(context)
    for index in range(generator.routing_site_count):
        query = router.first_query if index % 2 == 0 else router.second_query
        logits = query(hidden) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
        codes = routing.mix(f"site_{index}", logits)
        hidden = hidden + .02 * generator.first_adapter(torch.cat((hidden.tanh(), codes), dim=-1))
    return generator.second(hidden, codes)


def make_loop(*, device="cpu", sites=71, tokens=8, particles=128, z_dim=4, batch_size=4):
    if type(sites) is not int or sites < 1:
        raise ValueError("sites must be a positive integer")
    loop = sites_example.make_loop(device=device, mode="full", initialization="api",
                                   tokens=tokens, particles=particles, z_dim=z_dim,
                                   batch_size=batch_size, probe_interval=1000)
    policy = loop.policy
    policy.G.routing_site_count = policy.ema_G.routing_site_count = sites
    policy.routed_control.spec.model_forward = many_site_forward
    policy.routed_control.spec.sites = tuple(f"site_{index}" for index in range(sites))
    loop.config["routing_sites"] = sites
    with torch.no_grad():
        context = loop.test_context[:batch_size]
        clean = policy.routed_generate(context, sigma=0, perturb=False)
        loop.initial_rmse = float((clean - loop.test_targets[:batch_size]).square().mean().sqrt())
    return loop


def forward_pair(loop):
    """One no-grad critic forward and one differentiable generator forward."""
    policy, context = loop.policy, loop.fit_context[:loop.policy.recipe.batch_size]
    with torch.no_grad():
        first = policy.routed_generate(context, sigma=0)
    second = policy.routed_generate(context, sigma=0)
    return first, second


def scalar_extractions(call):
    # CPU activities include the host aten operation initiating each device
    # scalar readback. Profiling is deliberately outside timing measurements.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = call()
    return sum(event.count for event in profile.key_averages()
               if event.key == "aten::_local_scalar_dense"), result


def benchmark(*, device="cpu", sites=71, tokens=8, particles=128, z_dim=4, batch_size=4,
              steps=20, warmup_steps=3, state_output=None):
    if type(steps) is not int or steps < 1 or type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps nonnegative")
    if steps >= 1000 or warmup_steps >= 1000:
        raise ValueError("use fewer than 1000 updates to measure the non-probe hot path")
    loop = make_loop(device=device, sites=sites, tokens=tokens, particles=particles,
                     z_dim=z_dim, batch_size=batch_size)
    policy = loop.policy
    # Exercise kernels without advancing the measured training trajectory.
    sites_example.warmup(loop, warmup_steps)
    initial = sites_example.checkpoint(loop)
    counts = {}
    counts["two_generator_forwards"], outputs = scalar_extractions(lambda: forward_pair(loop))
    del outputs
    sites_example.restore(loop, initial)
    counts["complete_update"], _ = scalar_extractions(lambda: sites_example.update(loop))
    sites_example.restore(loop, initial)
    elapsed, trace = 0., []
    for _ in range(steps):
        sites_example.synchronize(policy.device)
        start = time.perf_counter()
        trace.append(sites_example.update(loop))
        sites_example.synchronize(policy.device)
        elapsed += time.perf_counter() - start
    state = sites_example.checkpoint(loop)
    if state_output is not None:
        torch.save({"state": state, "trace": trace}, state_output)
    return {"device": str(policy.device), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(policy.device) if policy.device.type == "cuda" else None,
            "sites": sites, "tokens": tokens, "particles": particles, "z_dim": z_dim,
            "batch_size": batch_size, "steps": steps, "warmup_steps": warmup_steps,
            "seconds": elapsed, "milliseconds_per_update": 1000 * elapsed / steps,
            "host_scalar_extractions": counts,
            "probe_interval": policy.routed_control.spec.probe_interval,
            "structural_evaluations": policy.routed_control.counters["evals"],
            "last_dv12_applications": state["policy"]["controller"]["latent_applications"],
            "precision": "frozen BF16 hosts; FP32 trainable owners/table",
            "scope": "small shared-bank overhead fixture; complete_update includes caller logging; "
                     "timing includes synchronized updates but excludes profiling, serving, checkpoint writes and warmup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sites", type=int, default=71)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    torch.set_num_threads(1)
    # Match the saved ambient streams between separate before/after processes.
    # Task/data/DV12 streams and public initializer keys are fixed by make_loop.
    torch.manual_seed(1729)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1729)
    with torch.autograd.set_multithreading_enabled(False):
        result = benchmark(**args)
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
