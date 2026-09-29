"""2-D families with regions and planted strays for the local-scale review (Q3). Each family:
   draw(n, g) -> x [n,2], region code [n]  (regions are defined from the generating coordinates, analysis only)
   stray(m, t, g) -> [m,2] points displaced by t * sperp along the normal from a random support point (None if not defined)
   names: region names; sperp: the thin-direction noise sd (unit of the stray distance)."""
import numpy as np
def rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

class Iso100:
    name = 'iso100 (100 isotropic modes, sd .03)'; sperp = .03
    names = ['r<1', '1<=r<2', '2<=r<3', '3<=r<4', 'r>=4']
    def __init__(self):
        co = np.arange(10) - 4.5; self.c = np.stack(np.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)
    def draw(self, n, g):
        c = self.c[g.integers(0, 100, n)]; e = g.standard_normal((n, 2)); r = np.linalg.norm(e, axis=1)
        return c + self.sperp * e, np.digitize(r, [1, 2, 3, 4])
    def stray(self, m, t, g):
        th = g.uniform(0, 2 * np.pi, m); return self.c[g.integers(0, 100, m)] + t * self.sperp * np.stack([np.cos(th), np.sin(th)], 1)

class CoreHalo:
    """dense narrow core (sd .005, mass .25) inside a broad halo (sd .15, mass .25) + 4 ordinary components (sd .05, mass .125 each)."""
    name = 'corehalo (core sd .005 inside halo sd .15, density ratio ~1000)'; sperp = .005
    names = ['core r<3sc', 'core edge 3-6sc', 'halo r<.03', 'halo .03-.15', 'halo .15-.3', 'halo >.3', 'other comps']
    def __init__(self):
        self.other = np.array([[3., 0.], [-3., 0.], [0., 3.], [0., -3.]])
    def draw(self, n, g):
        u = g.random(n); x = np.zeros((n, 2)); reg = np.zeros(n, int)
        core = u < .25; halo = (u >= .25) & (u < .5); oth = u >= .5
        x[core] = .005 * g.standard_normal((core.sum(), 2)); x[halo] = .15 * g.standard_normal((halo.sum(), 2))
        x[oth] = self.other[g.integers(0, 4, oth.sum())] + .05 * g.standard_normal((oth.sum(), 2))
        r = np.linalg.norm(x, axis=1)
        reg[core] = np.where(r[core] < .015, 0, 1); reg[halo] = np.digitize(r[halo], [.03, .15, .3]) + 2; reg[oth] = 6
        return x, reg
    def stray(self, m, t, g): return None

class Thin:
    """6 thin Gaussians sd_along .3 x sd_across .01 (aspect 30), random orientations, centres 2 apart."""
    name = 'thin (6 Gaussians .3 x .01, aspect 30)'; sperp = .01
    names = ['core |u|<1,|v|<1', 'sides |v|>=2', 'ends |u|>=2', 'corner']
    def __init__(self, seed=5):
        g = np.random.default_rng(seed); self.c = np.array([[i * 2., j * 2.] for i in range(3) for j in range(2)]); self.a = g.uniform(0, np.pi, 6)
    def draw(self, n, g):
        k = g.integers(0, 6, n); u = g.standard_normal(n); v = g.standard_normal(n); x = np.zeros((n, 2))
        for j in range(6):
            s = k == j; R = rot(self.a[j]); x[s] = self.c[j] + (R @ np.stack([.3 * u[s], .01 * v[s]])).T
        reg = np.where((np.abs(u) < 1) & (np.abs(v) < 1), 0, np.where((np.abs(v) >= 2) & (np.abs(u) < 2), 1, np.where((np.abs(u) >= 2) & (np.abs(v) < 2), 2, 3)))
        # rows in none of the four boxes (1<=|u|<2 or 1<=|v|<2 mixed) -> region 3 'corner/other'
        return x, reg
    def stray(self, m, t, g):
        k = g.integers(0, 6, m); u = np.clip(g.standard_normal(m), -1.5, 1.5); sign = g.choice([-1., 1.], m); x = np.zeros((m, 2))
        for j in range(6):
            s = k == j; R = rot(self.a[j]); x[s] = self.c[j] + (R @ np.stack([.3 * u[s], sign[s] * t * .01])).T
        return x

class Ring:
    name = 'ring (radius 1, radial sd .02)'; sperp = .02
    names = ['|dr|<1', '1<=|dr|<2', '2<=|dr|<3', '|dr|>=3']
    def draw(self, n, g):
        th = g.uniform(0, 2 * np.pi, n); dr = g.standard_normal(n); r = 1 + self.sperp * dr
        return np.stack([r * np.cos(th), r * np.sin(th)], 1), np.digitize(np.abs(dr), [1, 2, 3])
    def stray(self, m, t, g):
        th = g.uniform(0, 2 * np.pi, m); r = 1 + g.choice([-1., 1.], m) * t * self.sperp
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)

class Spiral:
    """Archimedean spiral r = .2 + .08 theta, theta in [0, 6 pi] uniform (arclength density falls ~8x from the centre to the end), isotropic noise sd .02."""
    name = 'spiral (3 turns, spacing .5, isotropic noise sd .02, density gradient ~8x along the curve)'; sperp = .02
    names = ['inner third, |e|<2', 'middle third, |e|<2', 'outer third, |e|<2', '|e|>=2 any']
    def pt(self, th):
        r = .2 + .08 * th; return np.stack([r * np.cos(th), r * np.sin(th)], 1)
    def normal(self, th):
        r = .2 + .08 * th; rp = .08
        tx = rp * np.cos(th) - r * np.sin(th); ty = rp * np.sin(th) + r * np.cos(th); nrm = np.hypot(tx, ty)
        return np.stack([-ty / nrm, tx / nrm], 1)
    def draw(self, n, g):
        th = g.uniform(0, 6 * np.pi, n); e = g.standard_normal((n, 2)); x = self.pt(th) + self.sperp * e
        reg = np.where(np.linalg.norm(e, axis=1) >= 2, 3, np.digitize(th, [2 * np.pi, 4 * np.pi]))
        return x, reg
    def stray(self, m, t, g):
        th = g.uniform(0, 6 * np.pi, m); return self.pt(th) + g.choice([-1., 1.], m)[:, None] * t * self.sperp * self.normal(th)

ALL = [Iso100, CoreHalo, Thin, Ring, Spiral]
