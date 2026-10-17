"""CPU inference only: matched kernels on immutable final CUDA checkpoints."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parent
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
RUNS = Path('/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training')
OLD_PKG = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929/pkg-CB64-RA')
sys.path.insert(0, str(PREV))
from models_metrics import networks, Evaluator, encode
sys.path.insert(0, str(ROOT/'pkg-CB64-RA2'))
from particlegan.continuous import DataDriftController


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moments(x, weights=None):
    x = x.double().numpy()
    if weights is None:
        return x.mean(0), np.cov(x, rowvar=False)
    weights = weights.double().numpy()
    weights = weights/weights.sum()
    mean = (x*weights[:, None]).sum(0)
    centered = x-mean
    cov = (centered*weights[:, None]).T@centered / (1-(weights*weights).sum())
    return mean, cov


def fd_from_moments(a, b):
    ma, ca = a; mb, cb = b
    w, v = np.linalg.eigh(ca)
    root = (v*np.sqrt(w.clip(0))[None, :])@v.T
    middle = root@cb@root
    return float(max(0., np.sum((ma-mb)**2)+np.trace(ca)+np.trace(cb)
                     -2*np.sqrt(np.linalg.eigvalsh((middle+middle.T)/2).clip(0)).sum()))


@torch.no_grad()
def radii(x):
    return torch.cat([torch.cdist(b, x).topk(6, largest=False).values[:, -1]
                      for b in x.split(256)])


class Scores:
    @torch.no_grad()
    def __init__(self):
        self.evaluator = Evaluator().eval()
        self.evaluator.load_state_dict(torch.load(PREV/'evaluator.pt', map_location='cpu', weights_only=True)['model'])
        train = MNIST(PREV/'data', train=True, download=False)
        test = MNIST(PREV/'data', train=False, download=False)
        _, train_f = encode(self.evaluator, train.data[:5000].float().unsqueeze(1)/127.5-1)
        mean, std = train_f.mean(0), train_f.std(0)
        self.active = std > float(std.max())*1e-6
        self.mean, self.std = mean[self.active], std[self.active]
        _, test_f = encode(self.evaluator, test.data[:5000].float().unsqueeze(1)/127.5-1)
        self.ref = (test_f[:, self.active]-self.mean)/self.std
        self.ref_labels = test.targets[:2048]
        self.class_mass = torch.bincount(test.targets, minlength=10).double()/len(test)
        self.ref_moments = moments(self.ref)
        self.ref_radius = radii(self.ref[:2048])

    @torch.no_grad()
    def __call__(self, images, manifold=True):
        p, f = encode(self.evaluator, images)
        f = (f[:, self.active]-self.mean)/self.std
        confidence, classes = p.max(1)
        counts = torch.bincount(classes, minlength=10)
        mass = counts.double()/len(classes)
        # Diagnostic only: equalize fake classifier mass to real class mass.
        # Missing fake classes are disclosed; no oracle labels enter the model.
        weights = (self.class_mass/counts.clamp_min(1))[classes]
        result = dict(class_mass_tv=float((mass-self.class_mass).abs().sum()/2),
                      class_mass=mass.tolist(), mean_confidence=float(confidence.mean()),
                      confident_fraction=float((confidence >= .9).float().mean()),
                      active_fd=fd_from_moments(self.ref_moments, moments(f)),
                      class_mass_equalized_fd=fd_from_moments(self.ref_moments, moments(f, weights)),
                      missing_classes=(counts == 0).nonzero().flatten().tolist(),
                      feature_total_variance=float(f.var(0).sum()))
        within = []
        for c in range(10):
            selected = f[classes == c]
            within.append(float(selected.var(0).sum()) if len(selected) > 1 else None)
        result['within_class_feature_variance'] = within
        if manifold:
            fake = f[:2048]
            fake_radius = radii(fake)
            precision = torch.cat([(torch.cdist(b, self.ref[:2048]) <= self.ref_radius[None]).any(1)
                                   for b in fake.split(256)])
            recall = torch.cat([(torch.cdist(b, fake) <= fake_radius[None]).any(1)
                                for b in self.ref[:2048].split(256)])
            result.update(active_precision=float(precision.float().mean()),
                          active_recall=float(recall.float().mean()),
                          recall_by_real_class=[float(recall[self.ref_labels == c].float().mean()) for c in range(10)])
        return result


def fixed_displacement(latent, noise):
    delta = .025*noise
    return delta*(.05/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)[:, None]


def exact_displacement(latent, noise, prior, bandwidth):
    delta = bandwidth*noise
    nearest = latent.new_full((len(latent),), float('inf'))
    for centers in prior.z.detach().split(max(1, 2048//latent.shape[1])):
        distance = (latent[:, None]-centers[None]).square().sum(-1)
        distance.masked_fill_(distance == 0, float('inf'))
        nearest = torch.minimum(nearest, distance.min(1).values)
    nearest = nearest.sqrt()
    radius = torch.where(torch.isfinite(nearest), .5*nearest, torch.zeros_like(nearest))
    return delta*(radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)[:, None]


@torch.no_grad()
def sample(model, prior, bandwidth, sigma, kernel, proposed=None):
    stream = torch.Generator().manual_seed(314259)
    images, deltas, ids_all = [], [], []
    for _ in range(16):
        ids = torch.randint(len(prior.z), (256,), generator=stream)
        latent = prior.z[ids]
        # Consume identical latent noise even for the zero-jitter diagnostic.
        noise = torch.randn(latent.shape, generator=stream)
        if kernel == 'zero':
            delta = torch.zeros_like(latent)
        elif kernel == 'fixed':
            delta = fixed_displacement(latent, noise)
        elif kernel == 'exact_dv12':
            delta = exact_displacement(latent, noise, prior, bandwidth)
        elif kernel == 'proposed':
            delta = proposed.displacement(latent, prior, bandwidth, noise)
        else:
            raise ValueError(kernel)
        raw = model(latent+delta)
        images.append(raw+sigma*torch.randn(raw.shape, generator=stream))
        deltas.append(delta); ids_all.append(ids)
    delta = torch.cat(deltas)
    return torch.cat(images), dict(delta_rms=float(delta.square().mean().sqrt()),
                                   delta_norm_mean=float(delta.norm(dim=1).mean()),
                                   delta_norm_max=float(delta.norm(dim=1).max()),
                                   row_ids_sha256=hashlib.sha256(torch.cat(ids_all).numpy().tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-proposed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(314159)
    torch.use_deterministic_algorithms(True)
    proposed = None
    if args.include_proposed:
        from particlegan.feature_cells import BoundedLatentGeometry
        proposed = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
    start = time.perf_counter()
    source_paths = [Path(__file__), ROOT/'PROTOCOL.md', PREV/'models_metrics.py', PREV/'evaluator.pt']
    source_paths += sorted((ROOT/'pkg-CB64-RA2'/'particlegan').glob('*.py'))
    frozen = {str(p): sha(p) for p in source_paths}
    results = dict(scope='CPU saved-checkpoint inference; matched CPU draws, no training or GPU quality verdict',
                   seed=314259, samples=4096, batch=256, source_sha256=frozen, cases=[])
    if proposed:
        results['proposed_kernel'] = 'bounded_local_dv12'
    scores = Scores()
    print(json.dumps(dict(event='evaluator_ready', active_dimensions=int(scores.active.sum()))), flush=True)
    for variant in ('E22', 'CB64-RA'):
        path = RUNS/'mnist'/variant/'checkpoint-2000.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        state = saved['trainer']
        stationary = state['lr_settle'][0][1]['last_decisive'] == -1
        serve = stationary and state['recipe']['serve_average'] > 0
        g_name, p_name = ('ema_G', 'ema_prior') if serve else ('G', 'prior')
        model, _ = networks('mnist')
        model.load_state_dict(state['models'][g_name]); model.eval()
        prior = SimpleNamespace(z=state['models'][p_name]['z'])
        bandwidth = state['controller']['latent_bandwidth']
        sigma = saved['record']['diagnostics']['output_sigma']
        meta = dict(variant=variant, checkpoint_sha256=sha(path), served_model=g_name,
                    served_prior=p_name, bandwidth_mean=float(bandwidth.mean()), output_sigma=sigma)
        clean = torch.cat([model(z) for z in prior.z.split(256)])
        case = dict(**meta, kernel='clean_centers', metrics=scores(clean, manifold=False))
        results['cases'].append(case)
        print(json.dumps(dict(event='case', **case)), flush=True)
        kernels = ['zero', 'fixed', 'exact_dv12']+(['proposed'] if proposed else [])
        for kernel in kernels:
            images, detail = sample(model, prior, bandwidth, sigma, kernel, proposed)
            case = dict(**meta, kernel=kernel, draw_receipt=detail, metrics=scores(images))
            results['cases'].append(case)
            print(json.dumps(dict(event='case', **case)), flush=True)
    results.update(seconds=time.perf_counter()-start, cuda_initialized=torch.cuda.is_initialized(),
                   sources_unchanged=all(sha(p) == expected for p, expected in frozen.items()))
    assert not results['cuda_initialized'] and results['sources_unchanged']
    target = ROOT/('saved-proposed-results.json' if args.include_proposed else 'saved-baseline-results.json')
    target.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(dict(event='complete', seconds=results['seconds'], path=str(target))), flush=True)


if __name__ == '__main__':
    main()
