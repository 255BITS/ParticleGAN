"""Q6: the action (ball rule) and mass bias. Static analysis in the table's own latent space (the test is not involved: every stray is assumed flagged,
every table row is assumed unflagged -> the parent distribution the shipped `_isolation_pick` arithmetic would produce, computed exactly per stray).
Rules: BALL = uniform over unflagged rows within 2x the distance of the nearest unflagged row (shipped E17); NEAR1 = the nearest row; KNN10 = uniform over the 10 nearest (E16);
UNIF = uniform over all unflagged rows (E15).
A. 100-mode lattice, masses log-uniform 1:10 (0.2%-2%), strays uniform in the plane >= 6 sigma from every mode: inflow to mode k / its mass share (1 = mass neutral).
B. Two modes A, B (spacing 1) with n_A, n_B rows: probability that a stray at fraction x of the way from A to B is re-drawn into A.
C. Latent dimension: share of the table inside the ball as a function of z_dim for (i) iid Gaussian z and (ii) 100 tight clusters in z.
D. Strays close to a mode (4-8 sigma): probability the parent is in the mode the stray came from."""
import numpy as np, torch
torch.set_num_threads(2)
g = np.random.default_rng(11); tg = torch.Generator().manual_seed(11)
SD = .03
def parents(zs, zt, rule):
    """returns for each stray the probability vector is too big; return sampled parent index (one sample per stray) and the inclusion mask share"""
    d = torch.cdist(torch.as_tensor(zs, dtype=torch.float64), torch.as_tensor(zt, dtype=torch.float64))
    dmin = d.min(1, keepdim=True).values
    if rule == 'BALL': w = (d <= 2 * dmin).double()
    elif rule == 'NEAR1': w = (d <= dmin).double()
    elif rule == 'KNN10':
        w = torch.zeros_like(d); w.scatter_(1, d.topk(10, 1, largest=False).indices, 1.)
    else: w = torch.ones_like(d)
    return w / w.sum(1, keepdim=True), w.sum(1)
print('== A. inflow / mass share per mode; 100-mode lattice, table N=20000 rows at the mode centres (+ jitter of the mode sd), 3000 uniform strays >= 6 (own) sigma from every mode. inflow_k = mean over strays of P(parent in mode k); ratio = inflow_k / mass share of k')
co = np.arange(10) - 4.5; C = np.stack(np.meshgrid(co, co, indexing='ij'), -1).reshape(-1, 2)
for label, mass, sd in (('equal masses, equal sd .03', np.ones(100), np.full(100, SD)),
                        ('masses log-uniform 1:10, sd .03', np.exp(g.uniform(0, np.log(10), 100)), np.full(100, SD)),
                        ('equal masses, sd log-uniform .02-.10', np.ones(100), np.exp(g.uniform(np.log(.02), np.log(.10), 100)))):
    mass = mass / mass.sum(); N = 20000
    lab = g.choice(100, N, p=mass); zt = C[lab] + sd[lab, None] * g.standard_normal((N, 2))
    zs = []
    while len(zs) < 3000:
        p = g.uniform(-5.5, 5.5, 2)
        if (np.linalg.norm(p - C, axis=1) / sd).min() >= 6: zs.append(p)
    zs = np.array(zs)
    print(f'  variant: {label}')
    for rule in ('BALL', 'NEAR1', 'KNN10', 'UNIF'):
        P, cover = parents(zs, zt, rule)
        flux = np.zeros(100)
        for k in range(100): flux[k] = P[:, torch.as_tensor(lab == k)].sum(1).mean()
        rho = flux / mass
        near_mode = (np.linalg.norm(zs[:, None, :] - C[None], axis=2) / sd[None]).argmin(1)
        pn = np.array([P[i, torch.as_tensor(lab == near_mode[i])].sum().item() for i in range(len(zs))])
        dens = mass / sd ** 2; dens = dens / dens.mean()
        print(f'    {rule:6s}: inflow/mass share: min {rho.min():.2f} p5 {np.percentile(rho, 5):.2f} median {np.median(rho):.2f} p95 {np.percentile(rho, 95):.2f} max {rho.max():.2f} | corr(rho, mass) {np.corrcoef(rho, mass)[0, 1]:+.2f} corr(rho, density m/sd^2) {np.corrcoef(rho, dens)[0, 1]:+.2f} | mean rows in the ball {cover.mean():.0f} | P(parent in the stray\'s nearest mode) {pn.mean():.3f}')
print('\n== B. two modes A, B spacing 1.0 (33 sigma), stray at fraction x of the way; P(re-drawn into A) under BALL; rows n_A, n_B (sd .03 jitter)')
for nA, nB in ((1000, 1000), (1000, 100), (1000, 40), (5000, 200)):
    zt = np.concatenate([np.array([0., 0.]) + SD * g.standard_normal((nA, 2)), np.array([1., 0.]) + SD * g.standard_normal((nB, 2))]); isA = np.arange(nA + nB) < nA
    row = []
    for x in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
        zs = np.array([[x, 0.01]] * 1); P, cover = parents(zs, zt, 'BALL'); row.append(P[0, torch.as_tensor(isA)].sum().item())
    print(f'  n_A={nA:5d} n_B={nB:5d}: x=.1..:  ' + ' '.join(f'{v:5.2f}' for v in row) + '   (pure nearest-mode would give 1 for x<.5, 0 for x>.5)')
print('\n== C. share of the unflagged table inside the ball vs latent dimension z_dim (N=20000; stray = a table row displaced by 6 sigma of its cluster)')
for dz in (2, 4, 8, 16, 32, 64, 128):
    N = 20000
    z_iid = g.standard_normal((N, dz)); zs_iid = z_iid[:200] + .0  # a stray drawn like any row
    centres = g.standard_normal((100, dz)) * 3; lab = g.integers(0, 100, N); z_cl = centres[lab] + .03 * g.standard_normal((N, dz))
    zs_cl = centres[g.integers(0, 100, 200)] + .18 * (lambda u: u / np.linalg.norm(u, axis=1, keepdims=True))(g.standard_normal((200, dz)))
    out = []
    for name, zt, zs in (('iid N(0,I) z', z_iid[200:], zs_iid), ('100 tight clusters', z_cl, zs_cl)):
        P, cover = parents(zs, zt, 'BALL'); out.append(f'{name}: ball holds {cover.mean() / len(zt):.3f} of the rows (median {np.median(cover.numpy()) / len(zt):.3f})')
    print(f'  z_dim={dz:4d} | ' + ' | '.join(out))
print('\n== D. strays 4-8 sigma from a mode centre: P(parent in that mode) and mean parent distance to that centre (sigma units), 100 equal modes')
N = 20000; lab = g.integers(0, 100, N); zt = C[lab] + SD * g.standard_normal((N, 2))
for r in (4, 5, 6, 8):
    k = g.integers(0, 100, 500); th = g.uniform(0, 6.283, 500); zs = C[k] + r * SD * np.stack([np.cos(th), np.sin(th)], 1)
    P, cover = parents(zs, zt, 'BALL'); own = np.array([P[i, torch.as_tensor(lab == k[i])].sum().item() for i in range(500)])
    dpar = []
    for i in range(500):
        pr = P[i].numpy(); j = g.choice(N, p=pr / pr.sum()); dpar.append(np.linalg.norm(zt[j] - C[k[i]]) / SD)
    print(f'  stray at {r} sigma: P(parent in own mode) {own.mean():.3f} | rows in ball {cover.mean():.0f} | parent distance to own centre: mean {np.mean(dpar):.2f} sigma, share >= 3 sigma {np.mean(np.array(dpar) >= 3):.3f}')
