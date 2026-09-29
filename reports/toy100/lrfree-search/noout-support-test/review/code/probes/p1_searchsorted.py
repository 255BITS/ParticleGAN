import torch
torch.set_num_threads(1)
null = torch.tensor([1., 2., 2., 3., 5.], dtype=torch.float64)
s = torch.tensor([0.5, 2., 2.5, 5., 6.], dtype=torch.float64)
idx = torch.searchsorted(null, s)          # default side='left'
print('searchsorted(left)  ', idx.tolist(), ' -> #(null >= s) =', (len(null) - idx).tolist(), ' brute:', [(null >= x).sum().item() for x in s])
idx_r = torch.searchsorted(null, s, right=True)
print('searchsorted(right) ', idx_r.tolist(), ' -> #(null >  s) =', (len(null) - idx_r).tolist(), ' brute:', [(null > x).sum().item() for x in s])
# NaN behaviour
nn_ = torch.tensor([1., 2., 3., float('nan'), float('nan')], dtype=torch.float64)
print('null with NaN tail, s=[0.5, 10, nan]:', torch.searchsorted(nn_, torch.tensor([0.5, 10., float('nan')], dtype=torch.float64)).tolist())
print('sort puts NaN last:', torch.tensor([3., float('nan'), 1.]).sort().values.tolist())
# lower median equivalence
x = torch.randn(1000, 10, dtype=torch.float64)
a = x.median(dim=1).values; b = x.sort(dim=1).values[:, (10 - 1) // 2]
print('torch.median(dim) == sort[(k-1)//2] (k=10):', torch.equal(a, b), ' k=9:', torch.equal(x[:, :9].median(dim=1).values, x[:, :9].sort(dim=1).values[:, (9 - 1) // 2]))
# argmax tie -> first index
t = torch.tensor([[0.3, 0.9, 0.9, -1.]]); print('argmax tie ->', t.argmax(1).item())
# 0.05 * N exactness
for N in (20, 32, 200, 800, 2000, 20000, 200000):
    print(N, repr(0.05 * N), 'n_flag<=QN boundary int:', int(0.05 * N))
