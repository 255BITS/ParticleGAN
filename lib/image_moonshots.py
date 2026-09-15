"""Architecture-only CIFAR moonshots; shared DDGAN/ParticleGAN training is unchanged."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from lib.image_ddgan import ImageGenerator, ImageDiscriminator


class FlatBlock(nn.Module):
    def __init__(self, width, heads, depth):
        super().__init__()
        self.heads = heads
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.condition = nn.Linear(width, 6 * width)
        self.local = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width),
                                   nn.GELU(), nn.Conv2d(width, width, 1))
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.gains = nn.Parameter(torch.full((3, width), 1 / math.sqrt(3 * depth)))
        # Small initial modulation keeps label/time/particle gradients active.
        nn.init.normal_(self.condition.weight, std=.01)
        nn.init.zeros_(self.condition.bias)

    def forward(self, x, e):
        mods = self.condition(F.silu(e)).chunk(6, -1)
        for i in range(3):
            h = self.norms[i](x) * (1 + mods[2*i][:, None]) + mods[2*i+1][:, None]
            if i == 0:
                h = self.local(h.transpose(1, 2).reshape(len(x), -1, 16, 16)).flatten(2).transpose(1, 2)
            elif i == 1:
                b, n, w = h.shape
                q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
                h = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w))
            else:
                h = self.mlp(h)
            x = x + h * self.gains[i]
        return x


class FlatParticleGenerator(nn.Module):
    """256 constant-width tokens, no pooling/merging/compressed image latent."""
    def __init__(self, cfg):
        super().__init__()
        w = cfg['g_width']
        self.embed = nn.Sequential(nn.Linear(cfg['z_dim'], w), nn.SiLU(), nn.Linear(w, w))
        self.cls = nn.Embedding(cfg['classes'], w)
        self.time = nn.Embedding(len(cfg['alpha_bar']), w)
        # Pixel unshuffle is a bijection: 3x32x32 -> 12x16x16, then expand.
        self.input = nn.Conv2d(12, w, 1)
        self.spatial = nn.Linear(cfg['z_dim'], 16 * 16 * cfg['spatial_channels'])
        self.spatial_proj = nn.Conv2d(cfg['spatial_channels'], w, 1)
        self.position = nn.Parameter(torch.randn(1, 256, w) * .02)
        self.blocks = nn.ModuleList([FlatBlock(w, cfg['g_heads'], cfg['g_depth']) for _ in range(cfg['g_depth'])])
        self.norm = nn.LayerNorm(w)
        self.output = nn.Conv2d(w, 12, 3, padding=1)

    def forward(self, z, c, xt, t):
        e = self.embed(z) + self.cls(c) + self.time(t)
        h = self.input(F.pixel_unshuffle(xt, 2))
        h = h + self.spatial_proj(self.spatial(z).reshape(len(z), -1, 16, 16))
        h = h.flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            h = block(h, e)
        h = self.norm(h).transpose(1, 2).reshape(len(z), -1, 16, 16)
        return F.pixel_shuffle(self.output(h), 2).tanh()


class PretrainedFeatureDiscriminator(nn.Module):
    """Pixel critic plus a frozen ResNet feature branch, one combined UCD score.

    No auxiliary loss. Candidate gradients pass through the frozen extractor,
    including the double backward needed for candidate-only bcap.
    """
    def __init__(self, cfg):
        super().__init__()
        from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
        if cfg.get('d_backbone', 'pretrained_resnet18') == 'pretrained_resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1
            net = resnet34(weights=weights)
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1
            net = resnet18(weights=weights)
        self.features = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1),
            net.layer2, net.layer3])
        # Avoid in-place activations in the second-order candidate gradient graph.
        for m in self.features.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.pixel = ImageDiscriminator(cfg)
        heads = cfg['classes'] * (len(cfg['alpha_bar']) - 1)
        self.project = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * ch, 64, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(.2),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(.2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, heads))
            for ch in (64, 128, 256)])
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.features.eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, value in self.features.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.pretrained_metadata = {'weights': str(weights),
                                    'feature_state_sha256': digest.hexdigest(),
                                    'input_size': 64, 'stages': ['layer1', 'layer2', 'layer3']}

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        self.features.requires_grad_(False)
        return self

    def ucd_labels(self, c, t):
        return self.pixel.ucd_labels(c, t)

    @torch.no_grad()
    def condition_features(self, xt):
        """Batch-local frozen conditioning features; never cache candidate features."""
        h = (F.interpolate(xt, size=64, mode='bilinear', align_corners=False) * .5 + .5 - self.mean) / self.std
        result = []
        for block in self.features:
            h = block(h)
            result.append(h)
        return result

    def forward(self, x, c, xt, t, condition_features=None):
        logits = self.pixel(x, c, xt, t)[1]
        # No clamping: diffusion states can exceed the image range.
        h = (F.interpolate(torch.cat([x, xt]) if condition_features is None else x, size=64, mode='bilinear', align_corners=False) * .5 + .5 - self.mean) / self.std
        feature_logits = []
        for i, (block, head) in enumerate(zip(self.features, self.project)):
            h = block(h)
            a, b = h.chunk(2) if condition_features is None else (h, condition_features[i])
            feature_logits.append(head(torch.cat([a, b], 1)))
        logits = (logits + sum(feature_logits) / math.sqrt(3)) / math.sqrt(2)
        return logits.gather(1, self.ucd_labels(c, t)[:, None]).squeeze(1), logits


def build_models(cfg):
    if cfg['architecture'] == 'ncsnpp':
        from lib.image_ncsnpp import NCSNppParticleGenerator
        g = NCSNppParticleGenerator(cfg)
    else:
        g = FlatParticleGenerator(cfg) if cfg['architecture'] == 'flat_hybrid' else ImageGenerator(cfg)
    d = PretrainedFeatureDiscriminator(cfg) if cfg.get('d_backbone', 'pixel') in ('pretrained_resnet18', 'pretrained_resnet34') else ImageDiscriminator(cfg)
    return g, d
