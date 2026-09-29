"""Peak memory of the stale-site test of maybe_apply (the three lines below are copied verbatim from birth_death.py) as a function of N and the number of sites S = 2 (ordinary + isolation moves)."""
import sys, torch, json
torch.set_num_threads(1)
N, S = int(sys.argv[1]), int(sys.argv[2])
anchor = torch.randn(N, 4); radius = torch.rand(N) * .1; n = torch.ones(N, dtype=torch.long); sites = torch.randn(S, 4)
def status(key):
    for line in open('/proc/self/status'):
        if line.startswith(key): return int(line.split()[1]) / 1024.
open('/proc/self/clear_refs', 'w').write('5'); base = status('VmRSS:')
dist = torch.cdist(anchor, sites, compute_mode="use_mm_for_euclid_dist")
stale = (n > 0) & (dist <= radius[:, None]).any(1)
print(json.dumps(dict(N=N, sites=S, extra_mb=round(status('VmHWM:') - base), model_5NS_mb=round(5 * N * S / 2 ** 20), stale=int(stale.sum()))))
