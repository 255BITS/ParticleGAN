"""Deterministic orthogonal initialization for every learned parameter handed to torch.optim.Adam.

Seed-independent: every value is a closed-form function of (optimizer index, parameter index, shape, declared init
distribution). The source matrix is a splitmix64 counter hash (numpy uint64), mapped to N(0,1) with ndtri or to
U(-1,1), then orthogonalized by float64 CPU QR with sign fix (variant "qr"), or taken from a Sylvester-Hadamard
matrix with hashed row/column signs (variant "hadamard"). The torch RNG is never consulted.

Rules (per parameter, at optimizer construction):
  * weight-like (ndim>=2, flattened to [out, prod(rest)]): semi-orthogonal, scaled so its element RMS equals the RMS of the
    distribution the recipe itself declared for it (captured analytically from the uniform_/normal_ call, e.g.
    kaiming_uniform bound -> bound/sqrt(3)). Only the scale is inherited: structure becomes orthogonal and deterministic.
  * ParticlePrior.z [N, d]: orthonormal columns * sqrt(N) * declared std + declared mean (uniform-hash for uniform_
    priors, gaussian-hash for normal_ priors): a decorrelated, exactly whitened cloud of the same scale.
  * nn.Linear/Conv/... bias: zeros (standard orthogonal companion).
  * other RNG-initialised 1-D parameters: a deterministic hash vector at the declared RMS (1 x n orthogonal row).
  * parameters that were never RNG-initialised (constants, identity) or were rewritten by the host after their draw
    (two_pole's stored host critic, the native identity generator): kept as-is; they are host definitions, and the
    two-seed hash check proves they are seed-independent.
"""
import gc, hashlib, json, math, os
import numpy as np
import torch
from torch import nn

VARIANT = os.environ.get('K3P_ORTHO_VARIANT', 'qr')
TAG = '_ortho_tag'
LOG = []
_opt_counter = [0]
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _hash_u01(key, n):
    with np.errstate(over='ignore'):
        x = (np.arange(n, dtype=np.uint64) + np.uint64(key)) * np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def _key(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') >> 1


def _source(key, rows, cols, kind):
    u = torch.from_numpy(_hash_u01(key, rows * cols)).reshape(rows, cols)
    return torch.special.ndtri(u) if kind == 'normal' else 2 * u - 1


def _hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64, device='cpu')
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def semi_orthogonal(key, rows, cols, kind='normal', variant=None):
    """[rows, cols] float64 with orthonormal columns (rows>=cols) or orthonormal rows (rows<cols)."""
    flip = rows < cols
    r, c = (cols, rows) if flip else (rows, cols)
    if (variant or VARIANT) == 'hadamard':
        n = 1 << (r - 1).bit_length()
        s_r = torch.from_numpy(np.where(_hash_u01(key, n) < 0.5, -1.0, 1.0))
        s_c = torch.from_numpy(np.where(_hash_u01(key + 1, n) < 0.5, -1.0, 1.0))
        q = (s_r[:, None] * _hadamard(n) * s_c[None, :])[:r, :c] / math.sqrt(n)
        q, _ = torch.linalg.qr(q)  # exact re-orthonormalisation of the truncated block (deterministic)
    else:
        q, rr = torch.linalg.qr(_source(key, r, c, kind))
        d = torch.sign(torch.diagonal(rr)); d = torch.where(d == 0, torch.ones_like(d), d)
        q = q * d
    return q.T.contiguous() if flip else q


def _declared_rms_mean_std(tag):
    kind, a, b = tag
    if kind == 'uniform':
        return math.sqrt((a * a + a * b + b * b) / 3), (a + b) / 2, (b - a) / math.sqrt(12)
    return math.sqrt(a * a + b * b), a, b


def install():
    for name in ('uniform_', 'normal_'):
        orig = getattr(torch.Tensor, name)

        def wrap(self, *args, _orig=orig, _name=name, **kw):
            out = _orig(self, *args, **kw)
            if isinstance(self, nn.Parameter):
                if _name == 'uniform_':
                    lo = args[0] if len(args) > 0 else kw.get('from', 0.0); hi = args[1] if len(args) > 1 else kw.get('to', 1.0)
                    tag = ('uniform', float(lo), float(hi))
                else:
                    m = args[0] if len(args) > 0 else kw.get('mean', 0.0); s = args[1] if len(args) > 1 else kw.get('std', 1.0)
                    tag = ('normal', float(m), float(s))
                setattr(self, TAG, (tag, self._version))
            return out
        setattr(torch.Tensor, name, wrap)

    from particlegan.particle_prior import ParticlePrior
    orig_init = torch.optim.Adam.__init__

    def adam_init(opt, params, *args, **kw):
        orig_init(opt, params, *args, **kw)
        apply(opt, ParticlePrior)
    torch.optim.Adam.__init__ = adam_init


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
        return bool(torch.equal(v.detach().cpu(), torch.eye(v.shape[0], dtype=v.dtype, device='cpu')))
    return False


@torch.no_grad()
def apply(opt, particle_cls):
    oi = _opt_counter[0]; _opt_counter[0] += 1
    own = _owners()
    params = [p for g in opt.param_groups for p in g['params']]
    for pi, p in enumerate(params):
        owner, pname = own.get(id(p), (None, None))
        oname = type(owner).__name__ if owner is not None else None
        tagged = getattr(p, TAG, None)
        fresh = tagged is not None and tagged[1] == p._version
        key = _key(oi, pi, tuple(p.shape))
        rule = None
        if tagged is None:
            rule = 'keep_deterministic' if _is_identity_or_const(p) else 'UNTAGGED_RANDOM_KEPT'
        elif not fresh:
            # the host rewrote the value after its RNG draw (e.g. two_pole copies stored host weights, the native
            # affine generator is set to identity): a deterministic host definition, kept as-is.
            rule = 'keep_host_set'
        if rule is None:
            flag = ''
            if isinstance(owner, particle_cls) and pname == 'z' and tagged is not None:
                _, mean, std = _declared_rms_mean_std(tagged[0])
                n, d = p.shape
                q = semi_orthogonal(key, n, d, tagged[0][0], variant='qr')  # a Hadamard cloud would sit on 2^d corner points
                new = mean + std * math.sqrt(n) * q
                rule = f'particle_{tagged[0][0]}_whitened' + flag
            elif pname == 'bias' and oname is not None and p.ndim == 1:
                new = torch.zeros(tuple(p.shape), dtype=torch.float64, device='cpu'); rule = 'bias_zero' + flag
            elif tagged is not None:
                rms = _declared_rms_mean_std(tagged[0])[0]
                rows = p.shape[0] if p.ndim >= 1 else 1
                cols = int(np.prod(p.shape[1:])) if p.ndim >= 2 else (p.shape[0] if p.ndim == 1 else 1)
                if p.ndim == 1: rows = 1
                q = semi_orthogonal(key, rows, cols)
                new = (q * rms * math.sqrt(max(rows, cols))).reshape(p.shape)
                rule = 'orthogonal_declared_rms' + flag
            if new is not None:
                p.copy_(new.to(dtype=p.dtype, device=p.device))
        v = p.detach().cpu().contiguous()
        LOG.append(dict(opt=oi, idx=pi, owner=oname, name=pname, shape=list(p.shape), rule=rule,
                        declared=None if tagged is None else list(tagged[0]),
                        sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()))


def dump(path):
    all_sha = hashlib.sha256(''.join(r['sha256'] for r in LOG).encode()).hexdigest()
    json.dump(dict(variant=VARIANT, all_params_sha256=all_sha, n=len(LOG), params=LOG), open(path, 'w'), indent=1)
    return all_sha
