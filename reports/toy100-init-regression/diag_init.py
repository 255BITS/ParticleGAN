"""Step 1: old (xavier, random) vs new direct-API vs registry-hook D at construction, toy100 ref config.

Prints per-layer weight RMS / spectral norm and D output + input-gradient statistics on real and generated
samples. Usage: PYTHONPATH=<worktree> python diag_init.py [problem]
"""
import dataclasses, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import benchmarks.toy100.train as T
from benchmarks.toy100.problems import sample_real

problem = sys.argv[1] if len(sys.argv) > 1 else "grid100"
manifest = T.load_config(ROOT / "configs/toy100/constraints_simple_regularization.json")
from benchmarks.toy100.config import resolve_problem_config
cfg = resolve_problem_config(manifest, problem, steps=None, device="cuda")
resolved, recipe = T.resolve_config(cfg)


def build(mode):
    r = dataclasses.replace(recipe, initialization="batch_feature_zero" if mode == "direct" else None)
    return T.make_trainer(resolved, r)


def stats(tr, tag):
    D = tr.D.model if hasattr(tr.D, "model") else tr.D
    lin = [m for m in D.modules() if isinstance(m, torch.nn.Linear)]
    row = {"tag": tag}
    for i, m in enumerate(lin):
        w = m.weight.detach().double()
        row[f"L{i}"] = f"rms={w.pow(2).mean().sqrt():.4f} sn={torch.linalg.matrix_norm(w, 2):.3f} b={m.bias.detach().abs().max():.2g}"
    gen = torch.Generator(device="cuda").manual_seed(5)
    real = sample_real(problem, 4096, device=torch.device("cuda"), generator=gen)
    with torch.no_grad():
        z, _ = tr.prior.sample(4096, generator=gen)
        fake = tr.G(z)
    for name, x in (("real", real), ("fake", fake)):
        x = x.detach().requires_grad_(True)
        out = D(x)
        g = torch.autograd.grad(out.sum(), x)[0].norm(dim=1)
        row[name] = (f"D mean={out.mean():+.4f} std={out.std():.4f} | grad mean={g.mean():.4f} "
                     f"p50={g.median():.4f} max={g.max():.4f}")
    row["x_std"] = f"real={real.std(0).tolist()} fake={fake.std(0).tolist()}"
    return row


rows = [stats(build("old"), "old(xavier)"), stats(build("direct"), "new(direct API)")]
from particlegan.init_registry import use_init
use_init("batch_feature_zero")
rows.append(stats(build("old"), "hook(registry)"))
for r in rows:
    print("==", r.pop("tag"))
    for k, v in r.items():
        print(f"  {k}: {v}")
