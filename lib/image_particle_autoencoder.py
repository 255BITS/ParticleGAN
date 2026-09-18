"""Direct CIFAR decoder and hard particle routing, without image side inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from lib.image_ddgan import ResBlock
from lib.image_moonshots import PretrainedFeatureDiscriminator


def route(query, means, sigma, offset, temperature):
    """Hard forward; global soft query gradient; selected-row mean gradient.

    Mean squared distance keeps the temperature in per-coordinate units.
    The matrix identity avoids allocating a [batch, particles, dimensions] array.
    """
    fixed = means.detach()
    distances = (query.square().sum(1, keepdim=True) + fixed.square().sum(1)[None]
                 - 2 * query @ fixed.T) / query.shape[1]
    ids = distances.argmin(1)
    soft = (-distances / temperature).softmax(1)
    proxy = soft @ fixed
    center = means[ids] + (proxy - proxy.detach())
    bounded = 3 * torch.tanh(offset / 3)
    return center + sigma * bounded, ids, bounded, soft


class DirectGenerator(nn.Module):
    def __init__(self, z_dim=64, width=32):
        super().__init__()
        self.input = nn.Linear(z_dim, 4 * width * 4 * 4)
        self.embed = nn.Sequential(nn.Linear(z_dim, 4 * width), nn.LeakyReLU(.2))
        self.blocks = nn.ModuleList([ResBlock(a, b, 4 * width) for a, b in
                                    [(4 * width, 4 * width), (4 * width, 2 * width),
                                     (2 * width, width)]])
        self.output = nn.Conv2d(width, 3, 3, padding=1)
        self.width = width

    def forward(self, z):
        h = self.input(z).reshape(-1, 4 * self.width, 4, 4)
        e = self.embed(z)
        for block in self.blocks:
            h = block(F.interpolate(h, scale_factor=2, mode='nearest'), e)
        return self.output(F.leaky_relu(h, .2)).tanh()


class ImageRoutingEncoder(nn.Module):
    def __init__(self, z_dim=64, width=32):
        super().__init__()
        layers = []
        for a, b in [(3, width), (width, 2 * width), (2 * width, 4 * width)]:
            layers += [nn.Conv2d(a, b, 4, stride=2, padding=1),
                       nn.GroupNorm(8, b), nn.LeakyReLU(.2)]
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.query = nn.Linear(4 * width * 4 * 4, z_dim)
        self.offset = nn.Linear(4 * width * 4 * 4, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x, means, sigma, temperature):
        h = self.features(x)
        q = self.query(h)
        # No toy-specific spatial skip. Fix query scale without batch statistics.
        q = F.layer_norm(q, (q.shape[1],))
        return route(q, means, sigma, self.offset(h), temperature)


class PretrainedImageRoutingEncoder(nn.Module):
    """Frozen ImageNet ResNet18 spatial features with learned AE routing heads."""
    def __init__(self, z_dim=64):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3)
        self.query = nn.Linear(256 * 4 * 4, z_dim)
        self.offset = nn.Linear(256 * 4 * 4, z_dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.pretrained_metadata = {'weights': str(weights), 'input_size': 64,
                                    'output': 'layer3 spatial 256x4x4', 'frozen': True}
        self.features.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.features.requires_grad_(False)
        return self

    def forward(self, x, means, sigma, temperature):
        with torch.no_grad():
            x = F.interpolate(x, size=64, mode='bilinear', align_corners=False)
            h = self.features((x * .5 + .5 - self.mean) / self.std).flatten(1)
        q = self.query(h)
        q = F.layer_norm(q, (q.shape[1],))
        return route(q, means, sigma, self.offset(h), temperature)


def build_encoder(cfg):
    kind = cfg.get('encoder_backbone', 'scratch')
    if kind == 'scratch':
        return ImageRoutingEncoder(cfg['z_dim'], cfg['width'])
    if kind == 'pretrained_resnet18':
        return PretrainedImageRoutingEncoder(cfg['z_dim'])
    raise ValueError(f'unknown encoder_backbone: {kind}')


class DirectDiscriminator(nn.Module):
    """Reuse the CIFAR feature critic with one scalar head and constant context.

    No labels/timesteps/noisy image information: the context is always zero.
    Frozen ResNet18 weights and its fixed BN statistics stay frozen in D steps.
    """
    def __init__(self, width=32):
        super().__init__()
        self.critic = PretrainedFeatureDiscriminator({
            'd_width': width, 'd_norm': 'group', 'd_mode': 'ucd', 'classes': 1,
            'alpha_bar': [1., 0.], 'ucd_target': 'time_class',
            'd_backbone': 'pretrained_resnet18'})
        self.register_buffer('context', torch.zeros(1, 3, 32, 32))
        self._context_features = None

    def requires_grad_(self, requires_grad=True):
        self.critic.requires_grad_(requires_grad)
        return self

    def forward(self, x):
        if self._context_features is None:
            self._context_features = self.critic.condition_features(self.context)
        features = [v.expand(len(x), -1, -1, -1) for v in self._context_features]
        labels = torch.zeros(len(x), device=x.device, dtype=torch.long)
        return self.critic(x, labels, self.context.expand(len(x), -1, -1, -1),
                           torch.ones_like(labels), condition_features=features)[0]
