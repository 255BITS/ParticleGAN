"""Task 5: adversarial cases for pkg-E17's isolation (flag ON).  Each case prints one outcome line (no exception / exception type, invariants: finite table, moves counted, parents legal).
usage: python adversarial_E17.py [case-substring ...]   (CPU, 2 threads)"""
import sys, json, math, traceback, hashlib, torch, torch.nn as nn
import os; torch.set_num_threads(int(os.environ.get("NT", 2)))
sys.path.insert(0, '/ml2/hypergan/gan-attempts/seqtest-20260928/tests')
from fixture import OVERRIDES
E14 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E14'; E17 = '/ml2/hypergan/gan-attempts/noout-20260928/pkg-E17'
def fresh():
    for k in [k for k in sys.modules if k == 'particlegan' or k.startswith('particlegan.')]: del sys.modules[k]
    sys.path[:] = [p for p in sys.path if 'pkg-' not in p]

def real(i, n=64):
    g = torch.Generator().manual_seed(1000 + i); k = torch.randint(0, 8, (n,), generator=g); ang = k.float() * math.pi / 4
    return torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .03 * torch.randn(n, 2, generator=g)

def mk(N, zdim=2, flag=True, D=None, pkg=E17, seed=0, dtype=torch.float32, **extra):
    fresh(); sys.path.insert(0, pkg)
    import particlegan as package
    from particlegan.particle_prior import ParticlePrior
    ov = json.load(open(OVERRIDES)); ov.update(num_particles=N, z_dim=zdim, batch_size=64, birth_death_space='critic', row_evidence_gate=True, table_release_rule='anchor', reopen_signal='none')
    if flag: ov['birth_death_isolation'] = True
    ov.update(extra)
    recipe = package.get_recipe(**ov); torch.manual_seed(seed)
    prior = ParticlePrior(N, zdim)
    with torch.no_grad(): prior.z.uniform_(-5, 5)
    G = nn.Linear(zdim, 2)
    with torch.no_grad(): G.weight.zero_(); G.weight[0, 0] = 1; G.weight[1, 1] = 1; G.bias.zero_()
    D = D if D is not None else nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    if dtype != torch.float32: G, D, prior = G.to(dtype), D.to(dtype), prior.to(dtype)
    t = package.GANTrainer(recipe, G, D, prior=prior, seed=seed, optimizer_options={'foreach': False, 'fused': False})
    torch.manual_seed(seed)
    return t

def ring(t, N, n_stray, zdim=2, seed=0, dup=None, extra_sd=.02):
    """8 modes on a circle of radius 3 in the first two latent coordinates (G projects onto them), strays on an inner ring; other coordinates N(0, extra_sd^2)"""
    gg = torch.Generator().manual_seed(seed)
    mode = torch.randint(0, 8, (N,), generator=gg); ang = mode.float() * math.pi / 4
    z = torch.zeros(N, zdim); z[:, :2] = torch.stack([3 * ang.cos(), 3 * ang.sin()], 1) + .02 * torch.randn(N, 2, generator=gg)
    if zdim > 2: z[:, 2:] = extra_sd * torch.randn(N, zdim - 2, generator=gg)
    a = torch.rand(n_stray, generator=gg) * 2 * math.pi; r = 1.0 + torch.rand(n_stray, generator=gg)
    if n_stray: z[:n_stray, :2] = torch.stack([r * a.cos(), r * a.sin()], 1)
    if dup is not None: z = dup(z)
    with torch.no_grad():
        t.G.weight.zero_(); t.G.weight[0, 0] = 1; t.G.weight[1, 1] = 1; t.G.bias.zero_()          # the trainer's initialisation / warm-up moved G: table placed in data coordinates again
        t.prior.z.copy_(z.to(t.prior.z.dtype)); t.ema_prior.z.copy_(z.to(t.prior.z.dtype))
    bd = t.birth_death
    bd.fill, bd.cursor, bd.rows_since_eval = 0, 0, 0
    dt = t.prior.z.dtype
    bd.observe_real(torch.cat([real(1000 + i) for i in range(math.ceil(N / 64))])[:N].to(dt))
    return z, mode

def warm(t, steps=3, dt=torch.float32):
    for i in range(steps): t.step(real(i).to(dt))

def iso_counts(bd): return {k: v for k, v in bd.counters.items() if k.startswith('iso')}
def stats_after(t, bd, n_stray, z0=None):
    mv = bd.moved_rows
    return dict(z_finite=bool(torch.isfinite(t.prior.z).all()), ema_finite=bool(torch.isfinite(t.ema_prior.z).all()), moved=None if mv is None else len(mv),
                strays_moved=None if mv is None else int(torch.isin(torch.arange(n_stray), mv).sum()), counters=iso_counts(bd))

CASES = {}
def case(f): CASES[f.__name__] = f; return f

@case
def tiny_N():
    out = {}
    for N in (5, 6, 8, 20, 32, 200):
        try:
            res = {}
            for flag in (True, False):
                t = mk(N, flag=flag)
                for i in range(120): t.step(real(i))
                h = hashlib.sha256()
                for m in (t.G, t.D, t.prior, t.ema_G, t.ema_prior):
                    for p in m.parameters(): h.update(p.detach().numpy().tobytes())
                res[flag] = (h.hexdigest()[:12], iso_counts(t.birth_death) if flag else None, t.birth_death.k, t.birth_death.counters['evals'])
            out[N] = f'runs 120 steps; k={res[True][2]}, evals {res[True][3]}, iso {res[True][1]}; flag on == flag off: {res[True][0] == res[False][0]}'
        except Exception as e:
            out[N] = f'{type(e).__name__}: {str(e)[:90]}'
    return out

@case
def tiny_N_statistic():
    """the smallest number of tied rows BH can flag and the smallest N at which the guard can ever let isolation act"""
    out = {}
    Q = .05
    for N in (20, 32, 200, 400, 799, 800, 1000, 2000, 20000, 200000):
        n2 = N // 2; pmin = 1. / (1 + n2)
        cmin = math.ceil(N * pmin / Q - 1e-12)                       # rows tied at p_min needed for BH at level Q over N rows
        out[N] = f'|R2|={n2}: BH flags nothing below {cmin} rows tied at p_min; guard allows at most {int(Q * N)} -> can act: {cmin <= Q * N}'
    return out

@case
def duplicate_rows():
    out = {}
    for name, dup, ns in (('500 rows copies of 5 distinct rows', lambda z: torch.cat([z[:1500], z[:5].repeat(100, 1)]), 60),
                          ('whole table collapsed to one point', lambda z: z[1500:1501].repeat(2000, 1), 0),
                          ('60 strays sitting on 3 distinct positions (20 copies each)', lambda z: torch.cat([z[:3].repeat_interleave(20, 0), z[60:]]), 60),
                          ('every row duplicated once (1000 distinct)', lambda z: z[:1000].repeat_interleave(2, 0), 30)):
        try:
            t = mk(2000); warm(t); z0, mode = ring(t, 2000, ns, dup=dup)
            last = t.birth_death.maybe_apply(t, .03)
            s = stats_after(t, t.birth_death, ns)
            out[name] = f"no exception; iso_flagged {last.get('iso_flagged')}, iso_moves {last.get('iso_moves')}, moves {last.get('moves')}, skip {last.get('skip')}; {s}"
        except Exception as e:
            out[name] = f'{type(e).__name__}: {str(e)[:120]}'
    return out

@case
def nonfinite():
    out = {}
    # (a)/(b) real rows with NaN / inf enter the reservoir
    for label, bad in (('NaN', float('nan')), ('inf', float('inf'))):
        try:
            t = mk(2000); warm(t); ring(t, 2000, 60); bd = t.birth_death
            with torch.no_grad(): bd.reservoir[5, 0] = bad
            last = bd.maybe_apply(t, .03)
            out[f'reservoir with one {label} row'] = f"no exception; skip={last.get('skip')}, iso_flagged {last.get('iso_flagged')}, moves {last.get('moves')}, dim_skips {bd.counters['dim_skips']}; z finite {bool(torch.isfinite(t.prior.z).all())}"
        except Exception as e:
            out[f'reservoir with one {label} row'] = f'{type(e).__name__}: {str(e)[:120]}'
    # (c) table rows with NaN latents (their G(z) and features are NaN)
    for n_nan in (5, 60, 150):
        try:
            t = mk(2000); warm(t); z0, mode = ring(t, 2000, 0); bd = t.birth_death
            with torch.no_grad(): t.prior.z[100:100 + n_nan] = float('nan')
            last = bd.maybe_apply(t, .03)
            mv = bd.moved_rows
            healthy_nan = int((~torch.isfinite(t.prior.z[:100])).any(1).sum() + (~torch.isfinite(t.prior.z[300:])).any(1).sum())
            out[f'{n_nan} table rows with NaN latents'] = (f"no exception; iso_flagged {last.get('iso_flagged')}, iso_moves {last.get('iso_moves')}, moved {None if mv is None else len(mv)}, NaN rows left in table "
                                                              f"{int((~torch.isfinite(t.prior.z)).any(1).sum())}, healthy rows turned NaN {healthy_nan}, skip={last.get('skip')}")
        except Exception as e:
            out[f'{n_nan} table rows with NaN latents'] = f'{type(e).__name__}: {str(e)[:120]}\n' + traceback.format_exc().splitlines()[-3]
    # (c2) 60 strays to re-draw and a few NaN rows that stay unflagged (fewer than the 40 tied rows BH needs): does the NaN poison the parent choice?
    for n_nan in (0, 5):
        try:
            t = mk(2000); warm(t); z0, mode = ring(t, 2000, 60); bd = t.birth_death
            with torch.no_grad(): t.prior.z[100:100 + n_nan] = float('nan')
            seen = []; orig = bd._move
            def spy(trainer, child, parent, seen=seen, orig=orig): seen.append((child.clone(), parent.clone())); return orig(trainer, child, parent)
            bd._move = spy
            last = bd.maybe_apply(t, .03); child, parent = seen[-1] if seen else (torch.zeros(0, dtype=torch.long),) * 2
            out[f'60 strays + {n_nan} NaN rows'] = (f"iso_flagged {last.get('iso_flagged')}, moved {len(child)}, distinct parents {len(parent.unique())}, parent index min {int(parent.min()) if len(parent) else None}, "
                                                    f"share of the most frequent parent {float(parent.bincount().max()) / max(1, len(parent)):.2f}")
        except Exception as e:
            out[f'60 strays + {n_nan} NaN rows'] = f'{type(e).__name__}: {str(e)[:120]}'
    # (d) a dead critic: every feature identical
    try:
        D = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        with torch.no_grad():
            for m in D:
                if isinstance(m, nn.Linear): m.weight.zero_(); m.bias.zero_()
            D[2].bias.fill_(1.)
        t = mk(2000, D=D); warm(t, 1); z0, mode = ring(t, 2000, 60); bd = t.birth_death
        last = bd.maybe_apply(t, .03)
        out['dead critic (constant features)'] = f"no exception; skip={last.get('skip')}, iso_flagged {last.get('iso_flagged')}, moves {last.get('moves')}"
    except Exception as e:
        out['dead critic (constant features)'] = f'{type(e).__name__}: {str(e)[:120]}'
    return out

@case
def refused_head():
    out = {}
    class Raw(nn.Module):
        def __init__(s): super().__init__(); s.h = nn.Linear(2, 1)
        def forward(s, x): return s.h(x)
    class Skip(nn.Module):
        def __init__(s): super().__init__(); s.a = nn.Linear(2, 16); s.head = nn.Linear(18, 1)
        def forward(s, x): return s.head(torch.cat([torch.relu(s.a(x)), x], 1))
    for name, D in (('head reads the raw sample', Raw()), ('head reads features ++ raw sample', Skip())):
        res = {}
        for pkg in (E17, E14):
            try:
                t = mk(200, D=type(D)(), pkg=pkg, flag=(pkg == E17)); steps = 0
                for i in range(20): t.step(real(i)); steps += 1
                res[pkg[-3:]] = f'ran {steps} steps without error (evals {t.birth_death.counters["evals"]})'
            except Exception as e:
                res[pkg[-3:]] = f'{type(e).__name__} after {steps} completed steps (trainer.completed_steps {t.completed_steps}): {str(e)[:70]}'
        out[name] = res
    return out

@case
def z_dim():
    out = {}
    for zd in (4, 64):
        try:
            t = mk(2000, zdim=zd); warm(t); z0, mode = ring(t, 2000, 60, zdim=zd); bd = t.birth_death
            seen = []; orig = bd._move
            def spy(trainer, child, parent, seen=seen, orig=orig): seen.append((child.clone(), parent.clone())); return orig(trainer, child, parent)
            bd._move = spy
            zpre = t.prior.z.detach().clone()
            last = bd.maybe_apply(t, .03); s = stats_after(t, bd, 60)
            child, parent = seen[-1] if seen else (torch.zeros(0, dtype=torch.long),) * 2
            zc = t.prior.z.detach()[child][:, :2]; centre_err = float(((zc.norm(dim=1) - 3).abs()).max()) if len(child) else float('nan')
            # ball size: fraction of the unflagged table inside twice the nearest unflagged distance, for the moved rows (measured in the full latent space)
            keep = torch.ones(2000, dtype=torch.bool); keep[child] = False
            d = torch.cdist(zpre[child], zpre[keep]); frac = ((d <= 2 * d.min(1, keepdim=True).values).double().mean(1))
            out[f'z_dim {zd}'] = (f"no exception; flagged {last.get('iso_flagged')}, moved {s['moved']}, strays moved {s['strays_moved']}/60; re-drawn rows: max |radius-3| of the new position {centre_err:.3f}; "
                                  f"ball = {float(frac.median()):.3f} of the unflagged table (median over moved rows)")
        except Exception as e:
            out[f'z_dim {zd}'] = f'{type(e).__name__}: {str(e)[:120]}'
    return out

@case
def ball_dimension():
    """how local is the ball (radius = 2 x nearest unflagged row) for an isotropic table z ~ N(0, I_d)?  median fraction of the table inside the ball for random rows"""
    out = {}
    g = torch.Generator().manual_seed(3)
    for d in (2, 4, 8, 16, 64):
        z = torch.randn(20000, d, generator=g); rows = torch.arange(500)
        dist = torch.cdist(z[rows], z[500:])
        frac = (dist <= 2 * dist.min(1, keepdim=True).values).double().mean(1)
        out[f'd={d}'] = f'median fraction of the unflagged table inside the ball {float(frac.median()):.4f} (min {float(frac.min()):.4f}, max {float(frac.max()):.4f})'
    return out

@case
def flagged_all():
    out = {}
    t = mk(2000); warm(t); ring(t, 2000, 60); bd = t.birth_death
    bd._isolated = lambda q, R, k, fast: torch.ones(len(q), dtype=torch.bool)
    z0 = t.prior.z.detach().clone(); last = bd.maybe_apply(t, .03)
    out['everything flagged, guard on'] = f"iso_flagged {last.get('iso_flagged')}, iso_moves {last.get('iso_moves')}, table unchanged {torch.equal(z0, t.prior.z.detach())}, counters {iso_counts(bd)}"
    t = mk(2000); warm(t); ring(t, 2000, 60); bd = t.birth_death
    bd._isolated = lambda q, R, k, fast: torch.ones(len(q), dtype=torch.bool); bd.Q = 2.
    try:
        z0 = t.prior.z.detach().clone(); last = bd.maybe_apply(t, .03)
        out['everything flagged, guard lifted (Q=2)'] = f"no exception; iso_moves {last.get('iso_moves')}, keep set empty -> nothing re-drawn by isolation; ordinary moves {last.get('moves')}; z finite {bool(torch.isfinite(t.prior.z).all())}"
    except Exception as e:
        out['everything flagged, guard lifted (Q=2)'] = f'{type(e).__name__}: {str(e)[:120]}'
    return out

@case
def dtype_float64():
    out = {}
    try:
        t = mk(2000, dtype=torch.float64); warm(t, 3, torch.float64); z0, mode = ring(t, 2000, 60); bd = t.birth_death
        last = bd.maybe_apply(t, .03); s = stats_after(t, bd, 60)
        out['float64 trainer'] = f"no exception; flagged {last.get('iso_flagged')}, moved {s['moved']}, strays moved {s['strays_moved']}/60, z dtype {t.prior.z.dtype}"
    except Exception as e:
        out['float64 trainer'] = f'{type(e).__name__}: {str(e)[:150]}'
    return out

@case
def odd_N_and_width1():
    out = {}
    try:
        t = mk(2001); warm(t); ring(t, 2001, 60); bd = t.birth_death; last = bd.maybe_apply(t, .03); s = stats_after(t, bd, 60)
        out['N = 2001 (odd reservoir split)'] = f"no exception; flagged {last.get('iso_flagged')}, moved {s['moved']}, strays moved {s['strays_moved']}/60"
    except Exception as e:
        out['N = 2001 (odd reservoir split)'] = f'{type(e).__name__}: {str(e)[:150]}'
    try:
        D = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.ReLU(), nn.Linear(1, 1))
        t = mk(2000, D=D); warm(t); ring(t, 2000, 60); bd = t.birth_death; last = bd.maybe_apply(t, .03); s = stats_after(t, bd, 60)
        out['critic feature width 1'] = f"no exception; feature width {sum(m.in_features for m in bd._heads)}, flagged {last.get('iso_flagged')}, moved {s['moved']}, skip={last.get('skip')}"
    except Exception as e:
        out['critic feature width 1'] = f'{type(e).__name__}: {str(e)[:150]}'
    return out

if __name__ == '__main__':
    sel = sys.argv[1:]
    for name, f in CASES.items():
        if sel and not any(s in name for s in sel): continue
        print(f'=== {name}', flush=True)
        try:
            for k, v in f().items(): print(f'  {k}: {v}', flush=True)
        except Exception:
            print('  CASE CRASHED:', traceback.format_exc()[-400:], flush=True)
