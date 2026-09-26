"""``qr_bz_pq`` from ``ortho-init-k3p`` (``w=qr,s=std,b=zero,p=qmc``).

``install`` records each parameter's declared ``uniform_`` / ``normal_``
distribution and patches ``Adam.__init__`` to rewrite that optimizer's
parameters. The per-parameter key is ``_key(oi, pi, shape)``: optimizer
creation index, parameter index in that optimizer, then
``sha256(repr(parts))`` first 8 bytes little-endian, shifted right by 1.

``ParticlePrior.z`` is an R2 row-as-point draw (``qmc_draw``). 1-D module
biases are zero. Every other recorded parameter is
``semi_orthogonal(key, rows, cols, "normal")``: splitmix64, ``ndtri``,
float64 CPU QR with a positive diagonal, scaled by ``rms * sqrt(max(rows, cols))``.

Defaults kept from that branch: ``io=same``, ``src=normal``. Omit ``install``
and the PyTorch init is unchanged. Learning rates, losses, and clips are not
touched.
"""
from __future__ import annotations

import atexit
import gc
import hashlib
import json
import math
import os

import numpy as np
import torch
from torch import nn

VARIANT = "qr_bz_pq"
# Same merged spec as ortho_init.py for _NAMED['qr_bz_pq'].
SPEC = dict(w="qr", s="std", b="zero", p="qmc", io="same", src="normal")
TAG = "_ortho_tag"
LOG: list[dict] = []
_opt_counter = [0]
_PATCHED = False
_ORIG_ADAM = None
_DUMPED = False


def _hash_u01(key, n):
    with np.errstate(over="ignore"):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(key)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def _key(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") >> 1


def _source(key, rows, cols, kind):
    u = torch.from_numpy(_hash_u01(key, rows * cols)).reshape(rows, cols)
    return torch.special.ndtri(u) if kind == "normal" else 2 * u - 1


def _hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64, device="cpu")
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def semi_orthogonal(key, rows, cols, kind="normal", variant=None):
    """[rows, cols] float64 with orthonormal columns (rows>=cols) or orthonormal rows (rows<cols)."""
    flip = rows < cols
    r, c = (cols, rows) if flip else (rows, cols)
    if (variant or SPEC["w"]) == "hadamard":
        n = 1 << (r - 1).bit_length()
        s_r = torch.from_numpy(np.where(_hash_u01(key, n) < 0.5, -1.0, 1.0))
        s_c = torch.from_numpy(np.where(_hash_u01(key + 1, n) < 0.5, -1.0, 1.0))
        q = (s_r[:, None] * _hadamard(n) * s_c[None, :])[:r, :c] / math.sqrt(n)
        q, _ = torch.linalg.qr(q)
    elif (variant or SPEC["w"]) == "eye":
        q = torch.eye(r, c, dtype=torch.float64, device="cpu")
    elif (variant or SPEC["w"]) == "dct":
        k = torch.arange(r, dtype=torch.float64, device="cpu")
        basis = torch.cos(math.pi / r * (k[None, :] + 0.5) * k[:, None]) * math.sqrt(2.0 / r)
        basis[0] /= math.sqrt(2.0)
        cols_pick = torch.from_numpy(np.argsort(_hash_u01(key, r), kind="stable")[:c].copy())
        signs = torch.from_numpy(np.where(_hash_u01(key + 7, r) < 0.5, -1.0, 1.0))
        q = (basis.T * signs[None, :])[:, cols_pick]
    else:
        q, rr = torch.linalg.qr(_source(key, r, c, kind))
        d = torch.sign(torch.diagonal(rr))
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = q * d
    return q.T.contiguous() if flip else q


def fixedgen_draw(key, shape, tag):
    g = torch.Generator(device="cpu").manual_seed(key)
    t = torch.empty(tuple(shape), dtype=torch.float64, device="cpu")
    kind, a, b = tag
    return t.uniform_(a, b, generator=g) if kind == "uniform" else t.normal_(a, b, generator=g)


def _r2(n, d):
    """Roberts R2 low-discrepancy sequence in [0,1)^d (generalised golden ratio); no RNG at all."""
    phi = 2.0
    for _ in range(64):
        phi = (1 + phi) ** (1.0 / (d + 1))
    alpha = np.array([(1 / phi) ** (j + 1) for j in range(d)])
    return np.mod(0.5 + np.arange(1, n + 1)[:, None] * alpha[None, :], 1.0)


def qmc_draw(key, shape, tag, rows_as_points=False):
    """Quasi-random sample of the declared distribution. Particle clouds: each row is one R2 point in d dims.
    Weights: a 1-D R2 sequence over all elements, placed by a hashed permutation (avoids lattice structure in rows)."""
    shape = tuple(shape)
    kind, a, b = tag
    if rows_as_points:
        u = _r2(shape[0], int(np.prod(shape[1:])))
    else:
        n = int(np.prod(shape)) if len(shape) else 1
        u = _r2(n, 1)[:, 0][np.argsort(_hash_u01(key, n), kind="stable")]
    u = torch.from_numpy(np.ascontiguousarray(u)).reshape(shape)
    return a + (b - a) * u if kind == "uniform" else a + b * torch.special.ndtri(u)


def pattern_bias(key, shape, tag):
    n = int(np.prod(shape)) if len(shape) else 1
    _, mean, std = _declared_rms_mean_std(tag)
    u = torch.from_numpy(_hash_u01(key, n)) * 2 - 1
    if n > 1:
        u = (u - u.mean()) / u.std(unbiased=False)
    return (mean + std * u).reshape(tuple(shape))


def _declared_rms_mean_std(tag):
    kind, a, b = tag
    if kind == "uniform":
        return math.sqrt((a * a + a * b + b * b) / 3), (a + b) / 2, (b - a) / math.sqrt(12)
    return math.sqrt(a * a + b * b), a, b


def install(name: str = VARIANT) -> str:
    """Activate ``qr_bz_pq``. Construction order (optimizer index) restarts at zero."""
    if name != VARIANT:
        raise ValueError(f"init must be {VARIANT}")
    _ensure_patched()
    _opt_counter[0] = 0
    LOG.clear()
    path = os.environ.get("K3P_QR_BZ_PQ_DUMP")
    if path:
        atexit.register(_dump_atexit, path)
    print(json.dumps({"event": "det_init", "name": name, "spec": SPEC}), flush=True)
    return name


def _ensure_patched() -> None:
    global _PATCHED, _ORIG_ADAM
    if _PATCHED:
        return
    for name in ("uniform_", "normal_"):
        orig = getattr(torch.Tensor, name)

        def wrap(self, *args, _orig=orig, _name=name, **kw):
            out = _orig(self, *args, **kw)
            if isinstance(self, nn.Parameter):
                if _name == "uniform_":
                    lo = args[0] if len(args) > 0 else kw.get("from", 0.0)
                    hi = args[1] if len(args) > 1 else kw.get("to", 1.0)
                    tag = ("uniform", float(lo), float(hi))
                else:
                    m = args[0] if len(args) > 0 else kw.get("mean", 0.0)
                    s = args[1] if len(args) > 1 else kw.get("std", 1.0)
                    tag = ("normal", float(m), float(s))
                setattr(self, TAG, (tag, self._version))
            return out

        setattr(torch.Tensor, name, wrap)

    from particlegan.particle_prior import ParticlePrior
    _ORIG_ADAM = torch.optim.Adam.__init__

    def adam_init(opt, params, *args, **kw):
        _ORIG_ADAM(opt, params, *args, **kw)
        apply(opt, ParticlePrior)

    torch.optim.Adam.__init__ = adam_init
    _PATCHED = True


def _owners():
    own = {}
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Module):
                for n, p in obj.named_parameters(recurse=False):
                    own.setdefault(id(p), (obj, n))
        except Exception:
            pass
    return own


def _is_identity_or_const(v):
    f = v.detach().cpu().flatten()
    if bool((f == f[0]).all()):
        return True
    if v.ndim == 2 and v.shape[0] == v.shape[1]:
        return bool(torch.equal(v.detach().cpu(), torch.eye(v.shape[0], dtype=v.dtype, device="cpu")))
    return False


@torch.no_grad()
def apply(opt, particle_cls):
    oi = _opt_counter[0]
    _opt_counter[0] += 1
    own = _owners()
    params = [p for g in opt.param_groups for p in g["params"]]
    for pi, p in enumerate(params):
        owner, pname = own.get(id(p), (None, None))
        oname = type(owner).__name__ if owner is not None else None
        tagged = getattr(p, TAG, None)
        fresh = tagged is not None and tagged[1] == p._version
        key = _key(oi, pi, tuple(p.shape))
        rule = None
        if tagged is None:
            rule = "keep_deterministic" if _is_identity_or_const(p) else "UNTAGGED_RANDOM_KEPT"
        elif not fresh:
            rule = "keep_host_set"
        if rule is None:
            flag = ""
            if isinstance(owner, particle_cls) and pname == "z" and SPEC["p"] == "qmc":
                new = qmc_draw(key, p.shape, tagged[0], rows_as_points=True)
                rule = "particle_qmc"
            elif isinstance(owner, particle_cls) and pname == "z" and SPEC["p"] == "fixedgen":
                new = fixedgen_draw(key, p.shape, tagged[0])
                rule = "particle_fixedgen"
            elif isinstance(owner, particle_cls) and pname == "z" and tagged is not None:
                _, mean, std = _declared_rms_mean_std(tagged[0])
                n, d = p.shape
                q = semi_orthogonal(key, n, d, tagged[0][0], variant="qr")
                new = mean + std * math.sqrt(n) * q
                rule = f"particle_{tagged[0][0]}_whitened" + flag
            elif pname == "bias" and oname is not None and p.ndim == 1:
                if SPEC["b"] == "pattern":
                    new = pattern_bias(key, p.shape, tagged[0])
                    rule = "bias_pattern"
                elif SPEC["b"] == "qmc":
                    new = qmc_draw(key, p.shape, tagged[0])
                    rule = "bias_qmc"
                elif SPEC["b"] == "fixedgen":
                    new = fixedgen_draw(key, p.shape, tagged[0])
                    rule = "bias_fixedgen"
                else:
                    new = torch.zeros(tuple(p.shape), dtype=torch.float64, device="cpu")
                    rule = "bias_zero" + flag
            elif SPEC["w"] == "fixedgen":
                new = fixedgen_draw(key, p.shape, tagged[0])
                rule = "fixedgen"
            elif SPEC["w"] == "qmc":
                new = qmc_draw(key, p.shape, tagged[0])
                rule = "qmc"
            elif SPEC["io"] == "qmc" and (p.ndim < 2 or min(p.shape[0], int(np.prod(p.shape[1:]))) <= 4):
                new = qmc_draw(key, p.shape, tagged[0])
                rule = "io_qmc"
            elif SPEC["io"] == "fixedgen" and (p.ndim < 2 or min(p.shape[0], int(np.prod(p.shape[1:]))) <= 4):
                new = fixedgen_draw(key, p.shape, tagged[0])
                rule = "io_fixedgen"
            elif tagged is not None:
                rms = _declared_rms_mean_std(tagged[0])[0]
                rows = p.shape[0] if p.ndim >= 1 else 1
                cols = int(np.prod(p.shape[1:])) if p.ndim >= 2 else (p.shape[0] if p.ndim == 1 else 1)
                if p.ndim == 1:
                    rows = 1
                q = semi_orthogonal(key, rows, cols, SPEC["src"])
                if SPEC["s"] == "unit":
                    new = q.reshape(p.shape)
                    rule = "orthogonal_gain1"
                elif SPEC["s"] == "spectral" and min(rows, cols) > 1:
                    sigma = _declared_rms_mean_std(tagged[0])[2] * (math.sqrt(rows) + math.sqrt(cols))
                    new = (q * sigma).reshape(p.shape)
                    rule = "orthogonal_spectral_matched"
                else:
                    new = (q * rms * math.sqrt(max(rows, cols))).reshape(p.shape)
                    rule = "orthogonal_declared_rms" + flag
            if new is not None:
                p.copy_(new.to(dtype=p.dtype, device=p.device))
        v = p.detach().cpu().contiguous()
        LOG.append(dict(opt=oi, idx=pi, owner=oname, name=pname, shape=list(p.shape), rule=rule,
                        declared=None if tagged is None else list(tagged[0]),
                        sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()))


def all_params_sha256() -> str:
    return hashlib.sha256("".join(r["sha256"] for r in LOG).encode()).hexdigest()


def dump(path) -> str:
    all_sha = all_params_sha256()
    with open(path, "w") as fh:
        json.dump(dict(variant=VARIANT, spec=SPEC, all_params_sha256=all_sha, n=len(LOG), params=LOG), fh, indent=1)
    return all_sha


def _dump_atexit(path) -> None:
    global _DUMPED
    if _DUMPED or not LOG:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dump(path)
    _DUMPED = True
