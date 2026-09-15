"""Optional frozen/trainable Anima blocks transplanted into a CIFAR U-Net.

Architecture reference: Comfy-Org/ComfyUI comfy/ldm/cosmos/predict2.py and
comfy/ldm/cosmos/position_embedding.py. This implements only the selected
blocks' tensor operations, with native PyTorch; no ComfyUI runtime, VAE, text
encoder, original prediction head, flow sampler, or diffusion loss is used.
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lib.image_ddgan import ImageGenerator

DONOR_DIMS = (2048, 1024, 16, 256)  # width, cross-attention width, heads, AdaLN rank


def validate_anima(cfg):
    indices = cfg['anima_blocks']
    if (not isinstance(indices, list) or not indices or
            any(type(i) is not int or not 0 <= i < 28 for i in indices) or
            indices != sorted(set(indices))):
        raise ValueError('Anima blocks must be unique increasing indices in [0, 27]')
    digest = cfg['anima_weights_sha256']
    if (not cfg['anima_weights'] or not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Anima needs a local tensor bundle and its SHA256')
    if cfg['anima_init'] not in ('pretrained', 'random') or cfg['anima_dtype'] not in ('float32', 'bfloat16'):
        raise ValueError('Invalid Anima initialization or dtype')
    if type(cfg.get('anima_trainable', False)) is not bool:
        raise ValueError('anima_trainable must be a boolean')
    if type(cfg['anima_context_tokens']) is not int or cfg['anima_context_tokens'] < 2:
        raise ValueError('Anima context needs at least two tokens for cross-attention')
    if cfg['channels_last']:
        raise ValueError('Anima transplant currently requires contiguous tensor layout')


def image_rope(side, head_dim):
    """Cosmos T=1 rotary coordinates, split-half channel convention."""
    spatial = head_dim // 6 * 2
    temporal = head_dim - 2 * spatial
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
    frequency = 10000. ** (-torch.arange(0, spatial, 2).float() / spatial)
    angles = torch.cat([torch.zeros(side * side, temporal // 2),
                        yy.flatten()[:, None] * frequency,
                        xx.flatten()[:, None] * frequency], -1)
    return angles.cos()[None, None], angles.sin()[None, None]


def rotate(x, cos, sin):
    a, b = x.float().chunk(2, -1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1).to(x.dtype)


class DonorAttention(nn.Module):
    def __init__(self, width, context, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(context, width, bias=False)
        self.v_proj = nn.Linear(context, width, bias=False)
        self.output_proj = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // heads, eps=1e-6)

    def forward(self, x, context, rope=None):
        def heads(y):
            return y.reshape(len(y), -1, self.heads, y.shape[-1] // self.heads).transpose(1, 2)
        q = self.q_norm(heads(self.q_proj(x)).to(self.q_norm.weight.dtype))
        k = self.k_norm(heads(self.k_proj(context)).to(self.k_norm.weight.dtype))
        v = heads(self.v_proj(context))
        # FP32 master norm parameters may promote Q/K under mixed precision.
        # SDPA requires Q, K and V to have the same dtype.
        q, k = q.to(v.dtype), k.to(v.dtype)
        if rope is not None:
            q, k = rotate(q, *rope), rotate(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.output_proj(y.transpose(1, 2).reshape_as(x))


class DonorMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer1 = nn.Linear(width, 4 * width, bias=False)
        self.layer2 = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        return self.layer2(F.gelu(self.layer1(x)))


class DonorBlock(nn.Module):
    stages = ('self_attn', 'cross_attn', 'mlp')

    def __init__(self, width, context, heads, rank):
        super().__init__()
        self.self_attn = DonorAttention(width, width, heads)
        self.cross_attn = DonorAttention(width, context, heads)
        self.mlp = DonorMLP(width)
        for stage in self.stages:
            setattr(self, 'adaln_modulation_' + stage, nn.Sequential(
                nn.SiLU(), nn.Linear(width, rank, bias=False), nn.Linear(rank, 3 * width, bias=False)))

    def forward(self, x, context, mods, rope):
        # Residual arithmetic and layer normalization stay float32. Only donor
        # matrix multiplies/attention use autocast, including input backward.
        for i, stage in enumerate(self.stages):
            shift, scale, gate = mods[:, i].chunk(3, -1)
            h = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)
            h = h * (1 + scale[:, None]) + shift[:, None]
            if stage == 'self_attn':
                y = self.self_attn(h, h, rope)
            elif stage == 'cross_attn':
                y = self.cross_attn(h, context)
            else:
                y = self.mlp(h)
            x = x.float() + gate[:, None] * y.float()
        return x


class DonorTime(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear_1 = nn.Linear(width, width, bias=False)
        self.linear_2 = nn.Linear(width, 3 * width, bias=False)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class AnimaDonor(nn.Module):
    def __init__(self, indices, width=2048, context=1024, heads=16, rank=256):
        super().__init__()
        self.width, self.heads = width, heads
        self.blocks = nn.ModuleDict({str(i): DonorBlock(width, context, heads, rank) for i in indices})
        self.t_embedder = nn.Sequential(nn.Identity(), DonorTime(width))
        self.t_embedding_norm = nn.RMSNorm(width, eps=1e-6)

    @torch.no_grad()
    def prepare(self, alpha_bar, dtype, trainable=False):
        self.trainable = trainable
        self.compute_dtype = dtype
        # Match signal/noise ratio for the donor time embedding only. The DDGAN
        # corruption/reverse transition still uses the original alpha_bar.
        a = torch.tensor(alpha_bar).sqrt()
        s = (1 - torch.tensor(alpha_bar)).sqrt()
        donor_t = s / (a + s)
        freq = torch.exp(-math.log(10000) * torch.arange(self.width // 2) / (self.width // 2))
        phase = donor_t[:, None] * freq
        self.register_buffer('time_features', torch.cat([phase.cos(), phase.sin()], -1), persistent=False)
        # Trainable donor weights and Adam/EMA state stay float32: directly
        # updating BF16 weights would round away small updates. Only the matrix
        # operations use BF16 autocast. Frozen donors retain their old storage.
        if not trainable:
            table = self.modulations()
            self.to(dtype=dtype)
            self.register_buffer('modulation_table', table, persistent=False)
        cos, sin = image_rope(8, self.width // self.heads)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.train(trainable).requires_grad_(trainable)

    def modulations(self):
        # Recompute from current weights, including EMA weights, when trainable.
        # Only the five fixed sinusoidal inputs are shared across minibatches;
        # no learned timestep result is cached or detached.
        with torch.autocast(self.time_features.device.type, enabled=False):
            shared = self.t_embedder[1](self.time_features)
            e = self.t_embedding_norm(self.time_features)
            return torch.stack([torch.stack([
                getattr(block, 'adaln_modulation_' + stage)(e) + shared
                for stage in block.stages], 1) for block in self.blocks.values()])

    def forward(self, x, context, t):
        table = self.modulations() if self.trainable else self.modulation_table
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=self.compute_dtype == torch.bfloat16):
            for i, block in enumerate(self.blocks.values()):
                x = block(x, context, table[i, t], (self.rope_cos, self.rope_sin))
        return x


# Compatibility for callers constructing the original frozen donor in tests.
FrozenAnima = AnimaDonor


class AnimaTransplantGenerator(ImageGenerator):
    """Existing U-Net plus a zero-start donor branch at its 8x8 grid."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.context_tokens = cfg['anima_context_tokens']
        self.donor_trainable = cfg.get('anima_trainable', False)
        width, self.context_width, heads, rank = DONOR_DIMS
        # Keep baseline U-Net and D initialization identical across both arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'] + 801)
            self.transplant_in = nn.Linear(4 * cfg['g_width'], width)
            self.transplant_norm = nn.LayerNorm(width)
            self.transplant_out = nn.Linear(width, 4 * cfg['g_width'])
            nn.init.zeros_(self.transplant_out.weight)
            nn.init.zeros_(self.transplant_out.bias)
            self.context_z = nn.Linear(cfg['z_dim'], self.context_tokens * self.context_width)
            self.context_c = nn.Embedding(cfg['classes'], self.context_tokens * self.context_width)
            self.context_norm = nn.LayerNorm(self.context_width)
            self.donor = AnimaDonor(cfg['anima_blocks'], width, self.context_width, heads, rank)
            path = Path(cfg['anima_weights'])
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != cfg['anima_weights_sha256']:
                raise ValueError('Anima bundle hash mismatch')
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            if bundle['metadata']['blocks'] != cfg['anima_blocks']:
                raise ValueError('Anima bundle block selection mismatch')
            if cfg['anima_init'] == 'pretrained':
                self.donor.load_state_dict(bundle['tensors'], strict=True)
            self.pretrained_metadata = {**bundle['metadata'], 'bundle_sha256': digest,
                                       'initialization': cfg['anima_init'], 'frozen': not self.donor_trainable,
                                       'parameter_dtype': 'float32' if self.donor_trainable else cfg['anima_dtype'],
                                       'cached_modulation': not self.donor_trainable,
                                       'dtype': cfg['anima_dtype'], 'insertion_resolution': 8,
                                       'donor_time': 'sqrt(1-alpha_bar)/(sqrt(alpha_bar)+sqrt(1-alpha_bar))'}
            dtype = torch.bfloat16 if cfg['anima_dtype'] == 'bfloat16' else torch.float32
            self.donor.prepare(cfg['alpha_bar'], dtype, self.donor_trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self.donor_trainable:
            self.donor.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        if not self.donor_trainable:
            self.donor.requires_grad_(False)
        return self

    def transform_encoded(self, h, z, c, t):
        x = self.transplant_in(h.flatten(2).transpose(1, 2))
        context = (self.context_z(z) + self.context_c(c)).reshape(len(z), self.context_tokens, self.context_width)
        y = self.donor(x, self.context_norm(context), t)
        y = self.transplant_out(self.transplant_norm(y.float()))
        return h + y.transpose(1, 2).reshape_as(h)
