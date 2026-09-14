"""FID protocol: torch-fidelity Inception, uint8 RGB, CIFAR-10 train (50k)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import fid_statistics_to_metric
from torchvision.utils import save_image

PROTOCOL = {'dataset': 'CIFAR-10', 'split': 'train', 'real_samples': 50000,
            'extractor': 'torch-fidelity-0.3.0-inception-v3-compat-2048',
            'weights': 'weights-inception-2015-12-05-6726825d.pth',
            'precision': 'float32 extractor (TF32 disabled), float64 statistics',
            'preprocess': 'RGB uint8; generated clamp[-1,1], round((x+1)*127.5); TF-compatible bilinear resize inside extractor'}


def uint8_images(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class FIDEvaluator:
    def __init__(self, real, cache_dir, batch_size=128):
        self.model = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).cuda().eval()
        self.batch_size = batch_size
        key = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f'cifar10-train-{key}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Real cache is prepared before concurrent experiments are launched.
        if path.exists():
            with np.load(path) as f:
                self.real = {'mu': f['mu'], 'sigma': f['sigma']}
        else:
            self.real = self.statistics(real)
            tmp = path.with_suffix('.tmp.npz')
            np.savez(tmp, **self.real)
            tmp.replace(path)
        self.cache_path = str(path)

    @torch.no_grad()
    def statistics(self, images):
        features = []
        # FID always uses full FP32 even when training enables TF32.
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            for x in images.split(self.batch_size):
                features.append(self.model(x.cuda())[0].cpu().numpy())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
        features = np.concatenate(features).astype(np.float64)
        return {'mu': features.mean(0), 'sigma': np.cov(features, rowvar=False)}

    def __call__(self, images):
        return float(fid_statistics_to_metric(self.statistics(images), self.real, verbose=False)['frechet_inception_distance'])


def save_grid(images, path):
    # Each row is one requested CIFAR class, in torchvision's standard order.
    save_image(images, path, nrow=10, normalize=True, value_range=(-1, 1))
