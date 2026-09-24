"""Vector discriminators for ParticleGAN trainers."""
import math

import torch
from torch import nn


class _SmoothFourierMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden, fourier, beta):
        super().__init__()
        self.register_buffer('freqs', torch.pi * (2. ** torch.arange(fourier, dtype=torch.float32)))
        width = in_dim + 2 * fourier * in_dim
        layers = []
        for _ in range(n_hidden):
            layers.extend((nn.Linear(width, hidden_dim), nn.Softplus(beta=beta)))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        phase = (x.unsqueeze(-1) * self.freqs).flatten(1)
        features = torch.cat([x, phase.sin(), phase.cos()], dim=1)
        return self.net(features).squeeze(-1)


class LinearSkipDiscriminator(nn.Module):
    """Smooth Fourier MLP plus an initially zero, learned raw-input linear score.

    Defaults reproduce the 2D rare-mixture winner (96x2, Softplus beta5, two
    Fourier bands, 10,467 parameters). Inputs are flat [batch, in_dim] vectors.
    This is an optional reference architecture: callers can supply any critic
    to GANTrainer. Architecture support across the suite uses different critics.
    """

    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=2, fourier=2, beta=5.):
        super().__init__()
        for name, value in (('in_dim', in_dim), ('hidden_dim', hidden_dim), ('n_hidden', n_hidden)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(fourier) is not int or fourier < 0:
            raise ValueError('fourier must be a nonnegative integer')
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError('beta must be finite and positive')
        self.main = _SmoothFourierMLP(in_dim, hidden_dim, n_hidden, fourier, beta)
        self.skip = nn.Linear(in_dim, 1, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        return self.main(x).reshape(-1) + self.skip(x).squeeze(-1)


class _CenteredFeatures(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=-1, keepdim=True)


class BatchDistanceDiscriminator(nn.Module):
    """Smooth vector critic with differentiable current-batch distance features.

    Defaults reproduce the shared-c6 2D rare-mixture witness: three width-96
    hidden layers, Softplus beta 6, scales (.1, .25, .5, 1), and 19,013
    trainable parameters. Each sample gets four local weighted squared-distance
    features from the other samples in this call. The scores are permutation
    equivariant and depend on batch composition in both train and eval modes.
    No batch statistics are stored, and all feature paths remain differentiable.

    Coordinate units matter for ``scales``. ``GANTrainer`` still accepts any
    caller-supplied critic; this class does not change other host networks.
    """

    def __init__(self, in_dim=2, hidden_dim=96, n_hidden=3, *,
                 scales=(.1, .25, .5, 1.), beta=6., eps=1e-5):
        super().__init__()
        for name, value in (('in_dim', in_dim), ('hidden_dim', hidden_dim), ('n_hidden', n_hidden)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(scales, (str, bytes)):
            raise ValueError('scales must be a nonempty sequence of positive finite numbers')
        try:
            declared_scales = tuple(scales)
            widths = tuple(float(scale) for scale in declared_scales)
        except (TypeError, ValueError) as error:
            raise ValueError('scales must be a nonempty sequence of positive finite numbers') from error
        if (not widths or any(isinstance(scale, bool) for scale in declared_scales)
                or any(not math.isfinite(scale) or scale <= 0 for scale in widths)):
            raise ValueError('scales must be a nonempty sequence of positive finite numbers')
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
            raise ValueError('beta must be finite and positive')
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be finite and positive')

        self.in_dim = in_dim
        self.eps = eps
        self.register_buffer('scales', torch.tensor(widths))
        self.layers = nn.ModuleList(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim)
                                    for i in range(n_hidden))
        self.normalizers = nn.ModuleList(_CenteredFeatures() for _ in range(n_hidden))
        self.activation = nn.Softplus(beta=beta)
        self.head = nn.Linear(hidden_dim+len(widths), 1)

    def pairwise_features(self, x):
        """Smooth local mean-square distance at each fixed kernel scale."""
        delta = x[:, None, :] - x[None, :, :]
        d2 = delta.square().sum(-1)
        scales2 = self.scales.square()
        kernels = torch.exp(-d2[..., None] / (2*scales2))
        n = len(x)
        offdiag = (1-torch.eye(n, device=x.device, dtype=x.dtype))[..., None]
        kernels = kernels*offdiag
        weighted = (kernels*d2[..., None]).sum(dim=1)
        return weighted/(kernels.sum(dim=1)+self.eps)/scales2

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_dim or len(x) == 0:
            raise ValueError(f'expected a nonempty [batch, {self.in_dim}] input')
        batch_feature = self.pairwise_features(x)
        for linear, normalizer in zip(self.layers, self.normalizers):
            x = self.activation(normalizer(linear(x)))
        return self.head(torch.cat((x, batch_feature), dim=-1)).squeeze(-1)
