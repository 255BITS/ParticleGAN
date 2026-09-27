"""Initialization receipt shared by the simple-critic workers (ring and toy100).

``receipt`` records, for the networks as they stand before the first update:
  initialization         the recipe's mode (``batch_feature_zero`` = the package default)
  external_init_hook     particlegan.initialization._external_init (must be None: no registry hook)
  param_sha256           sha256 over (name, dtype, shape, bytes) of every parameter of G, D, prior
  ema_D_equals_D         the EMA critic (when the worker has one) holds D's initial weights
  old_init_param_sha256  the same hashes for an ``initialization=None`` build with the same seed
                         and construction order (the pre-#194 random init)
  differs_from_old_init  per network; ``differs_from_old_init_all`` is their conjunction
  tensors_equal_to_old_init  parameters the new init leaves as drawn (should be empty on the ring)
  public_path_sha256_match   live hashes equal a fresh build through the public recipe path
                         (make_prior + make_optimizers), i.e. the worker applied init exactly as
                         GANTrainer / recipe.make_optimizers do
The old/public builds run on CPU inside ``torch.random.fork_rng`` so the run's RNG is untouched.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import torch

SEED = 0


def _tensors(module):
    return {name: p.detach().cpu().contiguous() for name, p in module.named_parameters()}


def param_sha256(module) -> str:
    out = hashlib.sha256()
    for name, t in _tensors(module).items():
        out.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        out.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return out.hexdigest()


def ring_build(recipe, G_cls, D_cls, mode_hold, seed=SEED):
    """The ring workers' construction order (seeded G, D, then recipe prior) + the public
    recipe path's init (make_optimizers with an EMA critic), on CPU, RNG-neutral."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        G = G_cls(recipe.z_dim, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
        D = D_cls(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
        prior = recipe.make_prior()
        recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D), foreach=False, fused=False)
    return {"G": G, "D": D, "prior": prior}


def old_recipe(recipe):
    return dataclasses.replace(recipe, initialization=None)


def receipt(recipe, live, *, old, public=None, ema_D=None, applied_via=""):
    """live/old/public: {"G": module, "D": module, "prior": module}."""
    from particlegan import initialization as _init
    live_sha = {k: param_sha256(m) for k, m in live.items()}
    old_sha = {k: param_sha256(m) for k, m in old.items()}
    equal = []
    for k in live:
        a, b = _tensors(live[k]), _tensors(old[k])
        equal += [f"{k}.{n}" for n in a if n in b and torch.equal(a[n], b[n])]
    differs = {k: live_sha[k] != old_sha[k] for k in live}
    ema_equal = None
    if ema_D is not None:
        d, e = _tensors(live["D"]), _tensors(ema_D)
        ema_equal = d.keys() == e.keys() and all(torch.equal(d[n], e[n]) for n in d)
    return {
        "initialization": recipe.initialization,
        "external_init_hook": getattr(_init, "_external_init", None),
        "applied_via": applied_via,
        "param_sha256": live_sha,
        "ema_D_equals_D": ema_equal,
        "old_init_param_sha256": old_sha,
        "differs_from_old_init": differs,
        "differs_from_old_init_all": all(differs.values()),
        "tensors_equal_to_old_init": equal,
        "public_path_sha256_match": None if public is None else {
            k: live_sha[k] == param_sha256(public[k]) for k in live},
    }


def ring_receipt(recipe, G, D, prior, G_cls, D_cls, mode_hold, *, ema_D=None, applied_via=""):
    """Receipt for the ring workers (all build SimpleMLP G/D with seed 0, then the recipe prior)."""
    return receipt(recipe, {"G": G, "D": D, "prior": prior},
                   old=ring_build(old_recipe(recipe), G_cls, D_cls, mode_hold),
                   public=ring_build(recipe, G_cls, D_cls, mode_hold), ema_D=ema_D, applied_via=applied_via)


def summary_line(r) -> str:
    """One log line: '# init batch_feature_zero differs_old=G,D,prior public_match=... ema=...'."""
    diff = ",".join(k for k, v in r["differs_from_old_init"].items() if v) or "none"
    pub = r["public_path_sha256_match"]
    pub = "n/a" if pub is None else ",".join(k for k, v in pub.items() if v) or "none"
    return (f"# init={r['initialization']} hook={r['external_init_hook']} differs_from_old={diff} "
            f"public_path_match={pub} ema_D_equals_D={r['ema_D_equals_D']} "
            f"sha G={r['param_sha256']['G'][:12]} D={r['param_sha256']['D'][:12]} "
            f"prior={r['param_sha256']['prior'][:12]}\n")
