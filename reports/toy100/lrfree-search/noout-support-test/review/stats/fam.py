"""Synthetic families used by the E17 review. A family draws real data points, 'clean' table rows (the deconvolved law: what a generator with
output noise sigma_out would have as clean centres so that clean + sigma_out * eps has exactly the real law) and applies a feature map.
All sampling is done with numpy Generators so that runs are reproducible per (family, seed)."""
import math, numpy as np, torch

SD = 0.03
def lattice(n=10, spacing=1.0):
    co = (np.arange(n) - (n - 1) / 2) * spacing
    return np.stack(np.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)

class Mixture:
    """Isotropic Gaussian mixture: centres [M, d], per-component sd [M] (per coordinate), masses [M]. Optionally features(x) applies a fixed map."""
    def __init__(self, centres, sds, masses=None, feature=None, name=''):
        self.c = np.asarray(centres, dtype=np.float64); self.M, self.d = self.c.shape
        self.s = np.broadcast_to(np.asarray(sds, dtype=np.float64), (self.M,)).copy()
        self.m = np.full(self.M, 1. / self.M) if masses is None else np.asarray(masses, dtype=np.float64) / np.sum(masses)
        self.feature = feature; self.name = name
    def labels(self, n, g):
        return g.choice(self.M, size=n, p=self.m)
    def real(self, n, g):
        lab = self.labels(n, g)
        return self.c[lab] + self.s[lab, None] * g.standard_normal((n, self.d)), lab
    def clean(self, n, g, sigma):
        """clean rows: component centre + N(0, s^2 - sigma^2) (sigma must be <= s); + sigma * eps gives the real law exactly"""
        lab = self.labels(n, g)
        sc = np.sqrt(np.maximum(self.s[lab] ** 2 - sigma ** 2, 0.))
        return self.c[lab] + sc[:, None] * g.standard_normal((n, self.d)), lab
    def feat(self, x):
        if self.feature is None:
            return np.asarray(x, dtype=np.float64)
        with torch.no_grad():
            return self.feature(torch.as_tensor(x, dtype=torch.float32)).double().numpy()

def grid100():
    return Mixture(lattice(10), SD, name='grid100 (100 modes, sd .03, equal mass)')

def unequal_mixed(seed=7):
    """13 components in [-4, 4]^2 with masses .32 ... .002 (smallest = 40 rows of 20,000) and sd .03 ... .15 (all >= sigma_out = .029)."""
    g = np.random.default_rng(seed); cs = []
    while len(cs) < 13:
        c = g.uniform(-4, 4, 2)
        if all(np.linalg.norm(c - o) > 1.2 for o in cs):
            cs.append(c)
    masses = np.array([.32, .2, .14, .1, .07, .05, .04, .03, .02, .013, .008, .004, .002])
    sds = np.array([.03, .05, .08, .12, .03, .06, .10, .04, .15, .03, .07, .05, .03])
    return Mixture(np.array(cs), sds, masses, name='unequal_mixed (13 comps, masses .32-.002, sd .03-.15)')

def highdim(d, h=256, M=50, seed=3, depth=2, sd=SD, centre_scale=2.0):
    """M equal-mass isotropic Gaussians in R^d (sd per coordinate) pushed through a random ReLU network with `depth` layers of width h."""
    from simlib import relu_map
    g = np.random.default_rng(seed)
    f = relu_map(d, h, seed=seed, depth=depth)
    return Mixture(centre_scale * g.standard_normal((M, d)), sd, feature=f, name=f'gauss{d}D->ReLU{h}x{depth} ({M} comps, sd {sd}/coord)')
