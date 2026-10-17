"""How often does an IDEAL finite table (rows drawn i.i.d. from the true mixture, plus the declared output noise) pass vector_unequal_mass?
Same scorer formulas as harness/hosts/vector_host.py::score_samples (component_covariance_error etc.); CPU only, analysis of the test, not of a candidate."""
import torch, math, sys
torch.manual_seed(0)
means = torch.tensor([[-1.5,-1.5],[-1.5,1.5],[1.5,-1.5],[1.5,1.5]])
masses = torch.tensor([.55,.30,.13,.02]); cov = torch.eye(2)*0.0324; N = 256; S = 4096; sig = 0.029
L = torch.linalg.cholesky(cov); inv = torch.linalg.inv(cov)
def score(fake):
    assign = torch.cdist(fake, means).argmin(1)
    errs = []
    for k in range(4):
        p = fake[assign == k]
        if len(p) < 10: errs.append(1.0); continue
        x = p - p.mean(0); e = x.T @ x / len(p)
        errs.append(float((e - cov).norm() / cov.norm()))
    return sum(errs) / 4, errs, torch.bincount(assign, minlength=4)
def run(mode, trials=3000):
    ok = 0; cnts = []; e4 = []
    for t in range(trials):
        if mode == 'iid':
            comp = torch.multinomial(masses, N, replacement=True)
        else:   # stratified: exactly proportional counts (largest remainder)
            c = torch.floor(masses * N).long(); r = N - int(c.sum()); c[3] += r; comp = torch.repeat_interleave(torch.arange(4), c)
        rows = means[comp] + (torch.randn(N, 2) @ L.T)
        idx = torch.randint(0, N, (S,))
        fake = rows[idx] + sig * torch.randn(S, 2)
        m, errs, counts = score(fake)
        ok += m <= 0.85; cnts.append(int((comp == 3).sum())); e4.append(errs[3])
    cnts = torch.tensor(cnts, dtype=torch.float); e4 = torch.tensor(e4)
    print(f'{mode:11s} P(component_covariance_error <= .85) = {ok/trials:.3f} | rows in the 2% component: mean {cnts.mean():.1f} min {int(cnts.min())} | its own error: median {e4.median():.2f} p90 {e4.quantile(.9):.2f}')
run('iid'); run('stratified')
