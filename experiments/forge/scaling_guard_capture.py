"""One public checkpoint continuation to inspect actual failed polar offsets.

This is a separately declared numerical diagnostic, not a latency measurement or
quality result. It restores the original public state without rewriting metadata,
extends the public external execution allowance by one update, and records the
fast solver's two guards before its existing exact-SVD fallback changes outputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.toy_audit.reproducibility import reproducible_execution
from .api import FormulationContext
from .contracts import atomic_json, canonical, file_hash
from .scaling_benchmark import DCGANCritic, DCGANGenerator, _finite_model, _render_frame
from .state import state_digest


SCHEMA = "forge_polar_guard_capture_v1"


def declare(root, output, checkpoint, *, source_commit, reservation_seconds=240):
    root, output, checkpoint = Path(root).resolve(), Path(output).resolve(), Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    completed = state["context"]["trainer"]["completed_steps"]
    if completed != state["spec"]["warmup_updates"] + state["spec"]["measured_updates"]:
        raise ValueError("capture requires a completed original benchmark checkpoint")
    files = sorted(set((root / "particlegan").rglob("*.py")) |
                   set((root / "experiments/forge").glob("*.py")) |
                   {root / "benchmarks/toy_audit/reproducibility.py"})
    declaration = dict(schema=SCHEMA, source_commit=source_commit,
                       source_composition="Declared Git baseline plus added guard-capture harness; complete source hashes bind actual composition.",
                       source_files_sha256={str(p.relative_to(root)): file_hash(p) for p in files},
                       checkpoint=dict(path=str(checkpoint), sha256=file_hash(checkpoint)),
                       original_spec=state["spec"], original_recipe=state["context"]["recipe"],
                       original_completed_steps=completed, extra_updates=1, max_total_updates=completed+1,
                       budget=dict(reservation_seconds=reservation_seconds, paid_retries=0),
                       scope="Actual failed-offset guard inspection at one public continuation update; no latency, quality, qualification or promotion claim.",
                       numerical_gate="finite completed extra update, changed learned G/D, full checkpointed streams; actual guards recorded before fallback; source and original checkpoint immutable")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "declaration.json"
    if path.exists() and json.loads(path.read_text()) != declaration:
        raise ValueError("immutable capture declaration differs")
    atomic_json(path, declaration)
    return declaration


@reproducible_execution
def run(*, root, output, device="cuda:0", seed=0):
    import particlegan.optim.dualnorm as dualnorm
    root, output = Path(root).resolve(), Path(output).resolve()
    declaration = json.loads((output / "declaration.json").read_text())
    if declaration["schema"] != SCHEMA or (output / "result.json").exists():
        raise ValueError("capture requires one supported unused declaration")
    for relative, expected in declaration["source_files_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"frozen capture source changed: {relative}")
    checkpoint = Path(declaration["checkpoint"]["path"])
    if file_hash(checkpoint) != declaration["checkpoint"]["sha256"]:
        raise ValueError("original checkpoint changed")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    spec = saved["spec"]
    context = FormulationContext(recipe_overrides=saved["context"]["recipe"], prior=spec["prior"],
                                 prior_contract="recipe_owned_v1", seed=seed, device=device)
    g = context.construct(lambda: DCGANGenerator(spec["image_size"], spec["latent_dim"],
                          spec["base_channels"], spec["max_channels"]), component="generator").to(device)
    d = context.construct(lambda: DCGANCritic(spec["image_size"], spec["base_channels"],
                          spec["max_channels"]), component="discriminator").to(device)
    trainer = context.build_trainer(g, d, max_steps=declaration["original_completed_steps"])
    context.load_state_dict(saved["context"])
    if trainer.completed_steps != declaration["original_completed_steps"]:
        raise ValueError("public checkpoint did not restore original cursor")
    initial = {name: state_digest(dict(model.named_parameters())) for name, model in (("G", g), ("D", d))}
    # This public method changes only the external execution allowance. The
    # original recipe, schedule horizon and saved checkpoint remain untouched.
    trainer.extend_execution(declaration["max_total_updates"])
    eval_stream = context.streams.generator("eval", component="guard_capture", purpose="before_after")
    target_stream = context.streams.generator("eval", component="guard_target", purpose="before_after")
    shape = (spec["batch_size"], 3, spec["image_size"], spec["image_size"])
    target = torch.rand((4, *shape[1:]), generator=target_stream, device=device).mul_(2).sub_(1)
    with torch.no_grad():
        before = trainer.sample(4, generator=eval_stream, ema=False, output_noise=False)
    frames = [_render_frame(before, target, trainer.completed_steps, 1.)]
    current, records, bundles = {}, [], []
    original_polar, original_solve = dualnorm._newton_schulz_polar, dualnorm._PolarGraphCache.solve
    def polar(matrices, **kwargs):
        pointer = matrices.untyped_storage().data_ptr()
        owners = [(role, name, parameter, optimizer) for role, model, optimizer in
                  (("critic", d, trainer.opt_d), ("generator", g, trainer.opt_g))
                  for name, parameter in model.named_parameters()
                  if parameter.grad is not None and parameter.grad.untyped_storage().data_ptr() == pointer]
        if len(owners) != 1:
            raise ValueError("cannot bind convolution batch to one actual learned parameter")
        role, name, parameter, optimizer = owners[0]
        group = next(group for group in optimizer.param_groups if any(p is parameter for p in group["params"]))
        current.update(role=role, parameter=name, metadata=copy.deepcopy(group["dualnorm_convolution"]))
        try:
            return original_polar(matrices, **kwargs)
        finally:
            current.clear()
    def solve(cache, oriented, smoothing):
        result, failed = original_solve(cache, oriented, smoothing)
        entry = next(entry for entry in cache.entries.values() if entry.result.data_ptr() == result.data_ptr())
        with torch.no_grad():
            orthogonality = (entry.identity - result.mT @ result - entry.lower.mT @ entry.lower).abs().sum(-1).amax(-1)
            denominator = entry.regularizer.flatten() * result.norm(dim=(-2, -1))
            relation = ((entry.regularizer * result - entry.scaled @ entry.lower).norm(dim=(-2, -1)) /
                        denominator.clamp_min(torch.finfo(result.dtype).tiny))
        failed_cpu = failed.detach().cpu()
        indices = failed_cpu.nonzero(as_tuple=False).flatten().tolist()
        metadata = current["metadata"]
        kh, kw = metadata["kernel_size"]
        row = dict(call=len(records), **copy.deepcopy(current), shape=list(oriented.shape),
                   smoothing=smoothing, failed_indices=indices,
                   failed_offsets=[dict(group=i//(kh*kw), row=(i//kw)%kh, column=i%kw) for i in indices],
                   failed_iterations=32 if indices else None,
                   orthogonality=orthogonality.detach().cpu().tolist(), relation=relation.detach().cpu().tolist(),
                   interpretation="failed call reaches the actual maximum32 iterations; successful stopping iteration is not exposed")
        records.append(row)
        if indices:
            bundles.append(dict(**row, matrices=oriented.detach().cpu(), failed=failed_cpu,
                                result_before_fallback=result.detach().cpu(),
                                lower=entry.lower.detach().cpu(), scaled=entry.scaled.detach().cpu(),
                                regularizer=entry.regularizer.detach().cpu()))
        print(canonical(dict(event="polar_guard", call=row["call"], role=row["role"], parameter=row["parameter"],
                             shape=row["shape"], failed=len(indices),
                             orthogonality_max=max(row["orthogonality"]), relation_max=max(row["relation"]))), flush=True)
        return result, failed
    real = torch.rand(shape, generator=context.streams.generator("data", component="critic", purpose="training"), device=device).mul_(2).sub_(1)
    real_g = torch.rand(shape, generator=context.streams.generator("data", component="generator", purpose="training"), device=device).mul_(2).sub_(1)
    with patch.object(dualnorm, "_newton_schulz_polar", polar), patch.object(dualnorm._PolarGraphCache, "solve", solve):
        losses = trainer.step(real, generator_real=real_g, collect_stats=False)
    finite = all(bool(torch.isfinite(x)) for x in losses.values() if isinstance(x, torch.Tensor)) and _finite_model(g,d,trainer.prior)
    final = {name: state_digest(dict(model.named_parameters())) for name, model in (("G", g), ("D", d))}
    with torch.no_grad():
        after = trainer.sample(4, generator=eval_stream, ema=False, output_noise=False)
    frames.append(_render_frame(after, target, trainer.completed_steps, float(finite)))
    frames[0].save(output/"training.gif", save_all=True, append_images=frames[1:],duration=700,loop=0,optimize=False)
    torch.save(bundles, output/"failed-matrix-batches.pt")
    torch.save(dict(schema=SCHEMA, context=context.state_dict(), completed_steps=trainer.completed_steps,
                    original_checkpoint=declaration["checkpoint"]), output/"state.pt")
    with (output/"guards.jsonl").open("x") as stream:
        for row in records:
            stream.write(canonical(row)+"\n")
    result=dict(schema=SCHEMA, declaration_sha256=file_hash(output/"declaration.json"),
                original_completed_steps=declaration["original_completed_steps"],completed_steps=trainer.completed_steps,
                finite=finite,initial_parameters_sha256=initial,final_parameters_sha256=final,
                changed={name:initial[name]!=final[name] for name in initial},
                failed_count=sum(len(row["failed_indices"]) for row in records),calls=len(records),
                scope=declaration["scope"], guards=records,
                artifacts={name:dict(path=str(output/name),sha256=file_hash(output/name)) for name in
                           ("failed-matrix-batches.pt","guards.jsonl","state.pt","training.gif")})
    result["passed"]=finite and trainer.completed_steps==declaration["max_total_updates"] and all(result["changed"].values())
    atomic_json(output/"result.json",result)
    print(canonical(dict(event="capture_complete",passed=result["passed"],failed=result["failed_count"],output=str(output))),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("declare","run"))
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.action=="declare":
        if not args.checkpoint or not args.source_commit:
            parser.error("declare requires --checkpoint and --source-commit")
        declare(args.root,args.output,args.checkpoint,source_commit=args.source_commit)
    else:
        result=run(root=args.root,output=args.output,device=args.device,seed=0)
        raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()
