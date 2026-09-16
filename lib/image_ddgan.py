"""Small residual U-Net and batch-independent convolutional UCD critic."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import ucd_labels


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, normalize=True, affine_condition=True):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin) if normalize else nn.Identity()
        self.n2 = nn.GroupNorm(min(8, cout), cout) if normalize else nn.Identity()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.affine = normalize and affine_condition
        self.cond = nn.Linear(emb, 2 * cout if self.affine else cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.normalize = normalize

    def forward(self, x, e):
        h = self.c1(F.leaky_relu(self.n1(x), .2))
        q = self.cond(F.leaky_relu(e, .2))[:, :, None, None]
        h = self.n2(h)
        if self.affine:
            scale, shift = q.chunk(2, 1)
            h = h * (1 + scale) + shift
        else:
            h = h + q
        h = self.c2(F.leaky_relu(h, .2))
        return (self.skip(x) + h) / math.sqrt(2)


class SpatialAttention(nn.Module):
    """Residual spatial self-attention, initially an identity mapping."""
    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            b, 3, self.heads, c // self.heads, h * w).unbind(1)
        y = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        y = y.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.project(y)


class ImageGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w, e = cfg['g_width'], cfg['g_width'] * 4
        self.embed = nn.Sequential(nn.Linear(cfg['z_dim'], e), nn.LeakyReLU(.2), nn.Linear(e, e))
        self.cls = nn.Embedding(cfg['classes'], e)
        self.time = nn.Embedding(len(cfg['alpha_bar']), e)
        self.input = nn.Conv2d(3, w, 3, padding=1)
        self.enc = nn.ModuleList([ResBlock(w, w, e), ResBlock(w, w*2, e), ResBlock(w*2, w*4, e)])
        self.mid = ResBlock(w*4, w*4, e)
        self.dec = nn.ModuleList([ResBlock(w*8, w*2, e), ResBlock(w*4, w, e), ResBlock(w*2, w, e)])
        self.output = nn.Conv2d(w, 3, 3, padding=1)
        resolutions = cfg.get('g_attn_resolutions', [])
        # Optional modules do not change existing convolution or D initialization.
        # Models are constructed on CPU before the trainer moves them to CUDA.
        with torch.random.fork_rng(devices=[]):
            self.enc_attention = nn.ModuleList([
                SpatialAttention(ch, cfg['g_heads']) if res in resolutions else nn.Identity()
                for ch, res in [(w, 32), (w*2, 16), (w*4, 8)]])
            self.dec_attention = nn.ModuleList([
                SpatialAttention(ch, cfg['g_heads']) if res in resolutions else nn.Identity()
                for ch, res in [(w*2, 8), (w, 16), (w, 32)]])

    def forward(self, z, c, xt, t):
        e = self.embed(z) + self.cls(c) + self.time(t)
        h, skips = self.input(xt), []
        for block, attention in zip(self.enc, self.enc_attention):
            h = attention(block(h, e))
            skips.append(h)
            h = F.avg_pool2d(h, 2)
        h = self.mid(h, e)
        for block, attention, skip in zip(self.dec, self.dec_attention, reversed(skips)):
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = attention(block(torch.cat([h, skip], 1), e))
        return self.output(F.leaky_relu(h, .2)).tanh()


class ImageDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        w, e = cfg['d_width'], cfg['d_width'] * 4
        self.mode = cfg['d_mode']
        self.classes = cfg['classes']
        self.ucd_target = cfg.get('ucd_target', 'class')
        self.joint_ucd = self.ucd_target == 'time_class'
        if self.ucd_target not in ('class', 'time_class') or (self.joint_ucd and self.mode != 'ucd'):
            raise ValueError('time_class requires a UCD discriminator')
        self.emb_dim = e
        self.time = None if self.joint_ucd else nn.Embedding(len(cfg['alpha_bar']), e)
        self.cls = nn.Embedding(cfg['classes'], e) if self.mode == 'concat' else None
        self.input = nn.Conv2d(6, w, 3, padding=1)
        normalize = cfg.get('d_norm', 'none') == 'group'
        self.blocks = nn.ModuleList([ResBlock(a, b, e, normalize, affine_condition=False)
                                     for a, b in [(w, w*2), (w*2, w*4), (w*4, w*4)]])
        heads = self.classes * (len(cfg['alpha_bar']) - 1 if self.joint_ucd else 1)
        self.output = nn.Linear(w*4*4*4, heads if self.mode == 'ucd' else 1)

    def ucd_labels(self, c, t):
        return ucd_labels(c, t, num_classes=self.classes, target=self.ucd_target, validate_args=False)

    def forward(self, x, c, xt, t):
        e = x.new_zeros(len(x), self.emb_dim) if self.joint_ucd else self.time(t)
        if self.cls is not None:
            e = e + self.cls(c)
        h = self.input(torch.cat([x, xt], 1))
        for block in self.blocks:
            h = F.avg_pool2d(block(h, e), 2)
        logits = self.output(F.leaky_relu(h, .2).flatten(1))
        score = logits.gather(1, self.ucd_labels(c, t)[:, None]).squeeze(1) if self.mode == 'ucd' else logits[:, 0]
        return score, logits


@torch.no_grad()
def sample_images(g, prior, schedule, c, rng):
    xt = torch.randn((len(c), 3, 32, 32), device=c.device, generator=rng)
    for step in range(schedule.steps, 0, -1):
        t = torch.full_like(c, step)
        clean = g(prior.sample(len(c), rng)[0], c, xt, t)
        eta = torch.randn(xt.shape, device=xt.device, generator=rng)
        xt = schedule.reverse(clean, xt, t, eta)
    return xt


@torch.no_grad()
def update_ema(target, source, decay):
    for a, b in zip(target.parameters(), source.parameters()):
        a.lerp_(b, 1 - decay)
    for a, b in zip(target.buffers(), source.buffers()):
        a.copy_(b)
