"""Pure actual-state observations for the unmodified transition source.

The trajectory trainer has no CPU path. This observer never ports its loop.
Transition panels retain all original test contexts, five ticks, 512 rows per
context and the source's MoG sampling/chunk order. Original endpoint evaluation
remains in the original trainer. Added observations confer no qualification.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch


def digest(value):
    h = hashlib.sha256()

    def add(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().contiguous().cpu()
            h.update(str((tuple(y.shape), y.dtype)).encode())
            h.update(y.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                add(str(key)); add(x[key])
        elif isinstance(x, (tuple, list)):
            for part in x:
                add(part)
        else:
            h.update(repr(x).encode())
        h.update(b"\0")
    add(value)
    return h.hexdigest()


def owner_state(local):
    """Include every nested optimizer, model, gradient and named RNG stream."""
    owners, seen = {}, set()

    def walk(name, value):
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer, torch.Generator, torch.Tensor)):
            if id(value) in seen:
                return
            seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            owners[name] = dict(state=value.state_dict(),
                               modes={k: m.training for k, m in value.named_modules()},
                               gradients={k: (p.requires_grad, p.grad) for k, p in value.named_parameters()})
        elif isinstance(value, torch.optim.Optimizer):
            owners[name] = value.state_dict()
        elif isinstance(value, torch.Generator):
            owners[name] = value.get_state()
        elif isinstance(value, torch.Tensor):
            owners[name] = (value, value.requires_grad, value.grad if value.is_leaf else None)
        elif isinstance(value, dict):
            for key, part in value.items():
                walk(f"{name}/{key}", part)
        elif isinstance(value, (tuple, list)):
            for i, part in enumerate(value):
                walk(f"{name}/{i}", part)
    for name, value in local.items():
        walk(name, value)
    return digest(dict(owners=owners, global_rng=torch.get_rng_state(), threads=torch.get_num_threads()))


class TransitionCapture:
    def __init__(self, module, output):
        self.module, self.output = module, output
        self.rows, self.arrays, self.seconds = [], {}, 0.

    def observe(self, step, local):
        started = time.perf_counter()
        before = owner_state(local)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            arrays, metrics = self.measure(local)
        assert owner_state(local) == before, "observation changed a training owner or RNG stream"
        prefix = f"step_{step:06d}"
        for name, tensor in arrays.items():
            self.arrays[f"{prefix}__{name}"] = tensor.detach().cpu().numpy().copy()
        row = dict(step=step, metrics=metrics, array_prefix=prefix,
                   owner_state_sha256=before, complete_owner_and_rng_pure=True)
        self.rows.append(row)
        with (self.output / "observations.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        self.seconds += time.perf_counter() - started
        print(json.dumps(dict(event="route_source_observation", step=step,
                              live_consistency=metrics["live_prior"]["consistency_mean"],
                              ema_consistency=metrics["ema_prior"]["consistency_mean"])), flush=True)

    def measure(self, local):
        m, toy, scaler, cfg = self.module, local["toy"], local["scaler"], local["cfg"]
        ticks = sorted(set(round(f * (toy.length - 2)) for f in (0, .25, .5, .75, 1)))
        cc, gg = toy.contexts("test")
        c0, geom0 = cc.repeat_interleave(len(ticks)), gg.repeat_interleave(len(ticks), 0)
        t0 = torch.tensor(ticks).repeat(len(cc))
        groups = torch.arange(len(c0)).repeat_interleave(cfg["eval_per_context"])
        c, geom, tick = c0[groups], geom0[groups], t0[groups]
        context = toy.condition(geom, tick)
        real = toy.sample(c, geom, tick, torch.Generator().manual_seed(99001))
        arrays = dict(c=c, geom=geom, tick=tick, group=groups, reference=real)
        metrics = {}
        for label, g, prior, e in (("live", local["g"], local["prior"], local["e"]),
                                    ("ema", local["ema_g"], local["ema_prior"], local["ema_e"])):
            rng = torch.Generator().manual_seed(99000)
            chunks = []
            for j in range(0, len(c), 256):
                sl = slice(j, j + 256)
                z, _ = prior.sample(len(c[sl]), rng)
                chunks.append(scaler.inverse(g(z, c[sl], context[sl])))
            prediction = torch.cat(chunks)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite actual source predictions")
            arrays[label + "_prior"] = prediction
            metrics[label + "_prior"] = {k: v for k, v in m.metrics(prediction, real, scaler, groups).items()
                                          if k != "contexts"}
            if e is not None:
                chunks = []
                normalized_real = scaler(real)
                for j in range(0, len(c), 256):
                    sl = slice(j, j + 256)
                    encoded = m.encoded_transition(e, g, prior, normalized_real[sl, :4], c[sl], context[sl])[0]
                    chunks.append(scaler.inverse(encoded))
                paired = torch.cat(chunks)
                arrays[label + "_encoded"] = paired
                metrics[label + "_encoded"] = dict(
                    next_state_mse=float((paired[:, 4:] - real[:, 4:]).square().mean()),
                    state_action_mse=float((paired[:, :4] - real[:, :4]).square().mean()))
        metrics["panel"] = dict(split="test", contexts=len(c0), rows=len(c), ticks=ticks,
                                rows_per_context=cfg["eval_per_context"], latent_seed=99000, reference_seed=99001,
                                latent_law="original MoG prior including its Gaussian component noise",
                                original_endpoint_sampling_preserved=True)
        return arrays, metrics

    def save(self):
        if self.arrays:
            np.savez_compressed(self.output / "observations.npz", **self.arrays)
