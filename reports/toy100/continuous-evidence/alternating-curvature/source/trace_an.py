import json,torch,sys
for name in ('adam','v6','v8'):
    d=json.load(open(f'/tmp/cgd/trace-{name}.json'));X=torch.tensor(d['trace']);M=torch.tensor(d['means'])
    near=torch.cdist(X,M).min(-1).values            # T x 12
    step=(X[1:]-X[:-1]).norm(dim=-1)                # T-1 x 12
    K=10
    net=(X[K:]-X[:-K]).norm(dim=-1)                  # T-K x 12
    path=torch.stack([step[t:t+K].sum(0) for t in range(len(step)-K+1)])  # aligned: window ending t+K
    eff=(net/path.clamp_min(1e-12))
    print(name)
    for a in range(0,1200,100):
        b=a+100; e=eff[max(a-K,0):b-K].mean().item() if b-K>0 else float('nan')
        print(f' {a:4d}-{b:4d} near_mean {near[a:b].mean():.3f} frac_in_hq {(near[a:b]<.21).float().mean():.2f} motion {step[a:b].mean():.4f} directedness {e:.3f}')
