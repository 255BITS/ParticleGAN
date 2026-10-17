"""Portable fixed-endpoint caption accuracy regression through public APIs.

The source is self-contained: Torch plus ParticleGAN, no pretrained assets.
--run compares two fixed512-update BF16 arms. Exit0=scientific PASS,
exit1=completed FAIL, exit2=incomplete. Output accuracy is offline only.
This generated one-block task does not establish a full-Supra improvement.
"""
import time
STARTED = time.monotonic()

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import particlegan
from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe, init


TASK = "routed_caption_accuracy_v1"
ARMS = ("ordinary_BF16", "particle_BF16")
SITES = ("ctx_proj", "block.self_attn.qkv", "block.self_attn.proj",
         "block.cross_attn.q", "block.cross_attn.kv", "block.cross_attn.proj")
N, Z, B, STEPS, SECONDS, SIGMA = 128, 4, 4, 512, 300, .125
RELATIVE_MARGIN, SOURCE_HARM, LIVE_FRACTION = .001, 1e-6, .9
MEDIA_STEPS = (0, 64, 128, 256, 384, 512)
MEDIA_INDICES = (0, 8, 16, 24, 32, 40, 1, 9)
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_caption_accuracy_v1.json"


@dataclass(frozen=True)
class Geometry:
    width: int = 576
    text: int = 768
    rank: int = 16
    tokens: int = 256
    length: int = 128
    heads: int = 9
    output: int = 16
    frequency: int = 256


FULL = Geometry()
# Explicit software-only host; never a convergence cohort or science CLI option.
SMALL = Geometry(width=12, text=16, rank=2, tokens=4, length=5, heads=3, output=4, frequency=8)
MASK_LENGTHS = (1, 22, 29, 38, 23, 33, 24, 40, 47, 53, 47, 51, 43)

# Fixed CPU retained-buffer statistics; generated inputs need no external assets.
ACTUAL_CAPTION_IDS = (0, 4, 5, 15, 16, 17, 18, 3, 13, 14, 12, 2, 11)
CAPTION_GRAM = (
    (0.866142995541493, -0.01866677206874441, -0.0794452880390014, -0.14687445569027785, 0.00437249023152603, -0.13019560817410109, -0.055286632781124556, -0.05873392649908602, -0.13339377701033628, -0.16561246061894752, -0.08077227456598636, -0.13897591752938468, -0.1083003600899986),
    (-0.01866677206874441, 1.7026675187325397, 1.466341628972238, 1.3051350900829406, 1.254152877814096, 1.450268808569098, 1.487167485489843, 1.3171858162225063, 1.387562904921972, 1.2673702861073146, 1.2337855219838723, 1.306375055532947, 1.30035142734316),
    (-0.0794452880390014, 1.466341628972238, 1.8177579521475473, 1.5657769676713809, 1.4248009803146797, 1.6739301756984628, 1.6813661917050966, 1.3478038114752664, 1.7266243573768294, 1.5513224848681393, 1.438460965777198, 1.5652451639711071, 1.5600914407752153),
    (-0.14687445569027785, 1.3051350900829406, 1.5657769676713809, 1.8557196117178616, 1.3273800970952816, 1.614167272626273, 1.5402857977531719, 1.3191844967713968, 1.5851102151261287, 1.7675628409008384, 1.3706905569178969, 1.5101484009734532, 1.5024152784501734),
    (0.00437249023152603, 1.254152877814096, 1.4248009803146797, 1.3273800970952816, 1.5737906638722559, 1.362525270428468, 1.3974844757150506, 1.287396506559771, 1.4181331592528719, 1.3238960161805133, 1.3732304374043973, 1.343694520177032, 1.3686262718091304),
    (-0.13019560817410109, 1.450268808569098, 1.6739301756984628, 1.614167272626273, 1.362525270428468, 1.8929805181301984, 1.688636113783253, 1.3151387610556058, 1.6103423075602719, 1.5806613271122285, 1.396807052041996, 1.6459780469132443, 1.5399750354424606),
    (-0.055286632781124556, 1.487167485489842, 1.6813661917050964, 1.5402857977531719, 1.397484475715051, 1.6886361137832522, 1.8583657153619613, 1.2830079505936252, 1.5309915768105857, 1.4561382474561473, 1.3240903257303704, 1.4486928594799864, 1.5123590814588164),
    (-0.05873392649908602, 1.3171858162225063, 1.3478038114752664, 1.3191844967713968, 1.287396506559771, 1.3151387610556058, 1.2830079505936247, 1.8108696972760772, 1.7096411620295906, 1.5876452063607382, 1.7027191705198754, 1.5875994849467279, 1.7280228462002893),
    (-0.13339377701033628, 1.387562904921972, 1.7266243573768294, 1.5851102151261287, 1.4181331592528719, 1.6103423075602719, 1.530991576810586, 1.7096411620295906, 2.0718619402286476, 1.8103546090199227, 1.8046617670148797, 1.8248352008700253, 1.881590462706881),
    (-0.16561246061894752, 1.2673702861073146, 1.5513224848681393, 1.7675628409008384, 1.3238960161805133, 1.5806613271122285, 1.4561382474561482, 1.5876452063607382, 1.8103546090199227, 1.94741045920978, 1.6791261187567463, 1.725350255650253, 1.779977021150395),
    (-0.08077227456598636, 1.2337855219838723, 1.438460965777198, 1.3706905569178969, 1.3732304374043973, 1.396807052041996, 1.3240903257303704, 1.7027191705198754, 1.8046617670148797, 1.6791261187567463, 1.9318939840758904, 1.6995289873124153, 1.8288747886728096),
    (-0.13897591752938468, 1.306375055532947, 1.5652451639711071, 1.5101484009734532, 1.343694520177032, 1.6459780469132443, 1.4486928594799857, 1.5875994849467279, 1.8248352008700253, 1.725350255650253, 1.6995289873124153, 1.891598709416895, 1.7687352018016491),
    (-0.1083003600899986, 1.30035142734316, 1.5600914407752153, 1.5024152784501734, 1.3686262718091304, 1.5399750354424606, 1.5123590814588157, 1.7280228462002893, 1.881590462706881, 1.779977021150395, 1.8288747886728096, 1.7687352018016491, 1.9475420696280412),
)
# Per-caption valid-centered RMS, pad mean L2, pad-centered RMS, pad/valid cosine.
CAPTION_STATS = (
    (0.0, 2.3012397602778885, 0.0, 0.02944421338012041),
    (0.09949586923298814, 3.2982930785392433, 0.06976896541200041, 0.26037052096785207),
    (0.10402082073198161, 3.3064359678670368, 0.07129644513289513, 0.27735953887279624),
    (0.10512405791501468, 3.3555227193288424, 0.0689369926976539, 0.22143219578482287),
    (0.10030598069191841, 3.3443697244425246, 0.06887159201834485, 0.1876659651318226),
    (0.10616226856318742, 3.4001558680088078, 0.07921539841059762, 0.25551107115373956),
    (0.09891206031645852, 3.39427108162082, 0.07373631794128634, 0.2381440271971107),
    (0.10636902100212679, 3.6474756341909775, 0.0725480176754635, 0.15586421075341048),
    (0.10851222954791333, 3.4420420585697387, 0.0730371674978047, 0.21534637675585105),
    (0.10747949654000774, 3.326133815046198, 0.0768590708958912, 0.13622469398361892),
    (0.10815907699821267, 3.529544296396443, 0.06493697571883071, 0.17395683815753898),
    (0.10775338803036696, 3.4335947004374248, 0.07811856134970492, 0.17889211395837648),
    (0.10450089420743149, 3.4485470660372695, 0.07984475505834403, 0.1714680195670066),
)
TEXT_PROVENANCE = {'data_sha256': 'fc18f90e1422a4a03287e368071cd71cab0d38c4e6cba1337c001b2d82598afc', 'masked_gram_receipt_sha256': 'a38cc36267294b513f48a917a9435d3f98e28f183674af9e7bc93f0f7db6a595', 'token_scale_receipt_sha256': '2b12e2381ca6a882cceeae53ef6206ff157d06b788b8faf3ba163468ea1b2c50', 'scope': 'CPU retained-buffer descriptive statistics only; actual GPU census failed before host forward; no activation/convergence qualification.'}


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x): h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else: h.update(json.dumps(x, sort_keys=True).encode())
    add(value)
    return h.hexdigest()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_identity():
    root = Path(particlegan.__file__).resolve().parent; h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(Path("particlegan") / path.relative_to(root)).encode()); h.update(path.read_bytes())
    return {"imported_package": str(root), "python_sha256": h.hexdigest()}


def backend_flags():
    return {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}


def budget():
    if time.monotonic() - STARTED > SECONDS: raise TimeoutError("fixed300s startup-through-final-write budget exceeded")


def initialize(module, role):
    declarations = init.declarations(module)
    streams = {name: torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(
        f"routed-caption-geometry-v1:{role}:{name}".encode()).digest()[:8], "little") % (2**63 - 1))
        for name, p in module.named_parameters() if p.requires_grad and p.numel() and declarations[name] is not init.KEEP}
    init.initialize_(module, method="sample_distributions_v1", parameter_generators=streams)


def global_rng(device):
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None}


def set_global_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None: torch.cuda.set_rng_state(state["cuda"], device)


class TimeEmbedding(nn.Module):
    def __init__(self, g):
        super().__init__(); self.frequency = g.frequency
        self.mlp = nn.Sequential(nn.Linear(g.frequency, g.width), nn.SiLU(), nn.Linear(g.width, g.width))
    def forward(self, t):
        half = self.frequency // 2
        phase = t[:, None].float() * 1000 * torch.exp(-math.log(10000.) * torch.arange(half, device=t.device).float() / half)
        return self.mlp(torch.cat((phase.cos(), phase.sin()), -1))


class Attention(nn.Module):
    def __init__(self, g, cross=False):
        super().__init__(); self.g, self.cross = g, cross
        if cross: self.q, self.kv = nn.Linear(g.width, g.width), nn.Linear(g.width, 2 * g.width)
        else: self.qkv = nn.Linear(g.width, 3 * g.width)
        self.proj = nn.Linear(g.width, g.width)
    def forward(self, x, text=None, mask=None):
        b, t, d = x.shape; h, hd = self.g.heads, d // self.g.heads
        if self.cross:
            q = self.q(x).view(b, t, h, hd).transpose(1, 2)
            kv = self.kv(text).view(b, text.shape[1], 2, h, hd)
            k, v = (kv[:, :, i].transpose(1, 2) for i in range(2))
        else:
            qkv = self.qkv(x).view(b, t, 3, h, hd)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[:, None, None].bool())
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


def modulate(x, shift, scale): return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, g):
        super().__init__(); self.norm1, self.norm_ca, self.norm2 = (nn.LayerNorm(g.width, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.self_attn, self.cross_attn = Attention(g), Attention(g, cross=True)
        self.mlp = nn.Sequential(nn.Linear(g.width, 4 * g.width), nn.GELU(approximate="tanh"), nn.Linear(4 * g.width, g.width))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(g.width, 6 * g.width))
    def forward(self, x, c, text, mask):
        shift, scale, gate, ms, mc, mg = self.adaln(c).chunk(6, -1)
        x = x + gate[:, None] * self.self_attn(modulate(self.norm1(x), shift, scale))
        x = x + self.cross_attn(self.norm_ca(x), text, mask)
        return x + mg[:, None] * self.mlp(modulate(self.norm2(x), ms, mc))


class Final(nn.Module):
    def __init__(self, g):
        super().__init__(); self.norm = nn.LayerNorm(g.width, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(g.width, 2 * g.width)); self.linear = nn.Linear(g.width, g.output)
    def forward(self, x, c):
        shift, scale = self.adaln(c).chunk(2, -1)
        return self.linear(modulate(self.norm(x), shift, scale))


class Backbone(nn.Module):
    def __init__(self, g):
        super().__init__(); self.g = g
        self.x_embed = nn.Linear(g.output, g.width); self.pos_embed = nn.Parameter(torch.zeros(1, g.tokens, g.width))
        self.t_embed = TimeEmbedding(g); self.ctx_proj = nn.Linear(g.text, g.width)
        self.block, self.final = Block(g), Final(g)
    def forward(self, latent, times, text, mask):
        hidden = self.x_embed(latent) + self.pos_embed; cond = self.t_embed(times)
        return self.final(self.block(hidden, cond, self.ctx_proj(text), mask), cond)


# The frozen position parameter has an explicit sampled declaration. Initialize
# before freezing, so sampled AdaLN attention/MLP gates and output signal survive.
init.register(Backbone, {"pos_embed": init.Normal(0., .02)})


class Adapter(nn.Module):
    def __init__(self, base, g, arm):
        super().__init__(); self.base, self.g, self.arm = base, g, arm
        self.down, self.up = nn.Linear(base.in_features, g.rank, bias=False), nn.Linear(g.rank, base.out_features, bias=False)
        if arm != ARMS[0]: self.bridge = nn.Linear(g.rank + Z, g.rank)
        self.site = None; self.route = None
    def forward(self, x):
        base = self.base(x)
        if self.arm == ARMS[0]: return base + self.up(self.down(x))
        router, candidate, routing = self.route
        with torch.autocast(x.device.type, enabled=False):
            query = router.queries[self.site.replace(".", "__")](x.float())
            logits = query @ candidate.table.float().T / math.sqrt(Z)
            b, t = logits.shape[0] // 2, logits.shape[1]
            grouped = logits.reshape(2, b, t, logits.shape[-1]).permute(1, 0, 2, 3)
            mixed = routing.mix(self.site, grouped)
            code = mixed.permute(1, 0, 2, 3).reshape(2 * b, t, Z)
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=True): h = self.down(x.float())
            h = h.float(); m = h + F.linear(h, self.bridge.weight[:, :self.g.rank], self.bridge.bias).tanh()
            m = m + h * F.linear(code.float(), self.bridge.weight[:, self.g.rank:]).tanh()
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=True): delta = self.up(m)
            return (base.float() + delta.float()).to(base.dtype)


def replace(root, path, value):
    pieces = path.split("."); owner = root
    for name in pieces[:-1]: owner = getattr(owner, name)
    setattr(owner, pieces[-1], value)


class Host(nn.Module):
    def __init__(self, data, arm):
        super().__init__(); self.arm, self.g = arm, data["geometry"]
        self.backbone, self.teacher_backbone = Backbone(self.g), Backbone(self.g)
        self.backbone.load_state_dict(data["frozen"]); self.teacher_backbone.load_state_dict(data["frozen"])
        # Actual caption host stores frozen weights/position in FP32 and uses
        # BF16 autocast for linears; the position/residual stream remains FP32.
        self.backbone.requires_grad_(False); self.teacher_backbone.requires_grad_(False)
        self.register_buffer("captions", data["captions"].clone()); self.register_buffer("masks", data["masks"].clone())
        for site in SITES:
            branch = Adapter(self.backbone.get_submodule(site), self.g, arm); branch.site = site
            replace(self.backbone, site, branch)
    def branches(self): return [self.backbone.get_submodule(site) for site in SITES]
    @staticmethod
    def guided(value):
        conditional, unconditional = value.float().chunk(2)
        return unconditional + 3 * (conditional - unconditional)
    def inputs(self, context, teacher=False):
        ids = context[:, 0, self.g.output].long() + (7 if teacher else 1)
        halves = torch.cat((ids, torch.zeros_like(ids)))
        return (torch.cat((context[..., :self.g.output],) * 2), torch.cat((context[:, 0, self.g.output + 1],) * 2), self.captions[halves], self.masks[halves])
    @torch.no_grad()
    def teacher(self, context):
        with torch.autocast(context.device.type, dtype=torch.bfloat16): y = self.teacher_backbone(*self.inputs(context, True))
        return self.guided(y).float()
    def prediction(self, context):
        with torch.autocast(context.device.type, dtype=torch.bfloat16): y = self.backbone(*self.inputs(context))
        return self.guided(y).float()
    def forward(self, context):
        if self.arm != ARMS[0]: raise ValueError("particle host requires public routed forward")
        return self.prediction(context) - self.teacher(context)
    def forward_routed(self, context, router, candidate, routing):
        branches = self.branches()
        if any(b.route is not None for b in branches): raise RuntimeError("nested routed host execution")
        try:
            for branch in branches: branch.route = (router, candidate, routing)
            return self.prediction(context) - self.teacher(context)
        finally:
            for branch in branches: branch.route = None


class Router(nn.Module):
    def __init__(self, g):
        super().__init__(); self.queries = nn.ModuleDict({s.replace(".", "__"): nn.Linear(g.text if s == "ctx_proj" else g.width, Z) for s in SITES})
        self.register_buffer("log_mass", torch.zeros(N))


class Encoder(nn.Module):
    def __init__(self, captions, masks, g):
        super().__init__(); self.g = g
        self.register_buffer("means", (captions * masks[..., None]).sum(1) / masks.sum(1, keepdim=True))
    def condition(self, x):
        return torch.cat((self.means[x[:, 0, self.g.output].long() + 1], x[:, 0, self.g.output + 1:]), -1)


class Critic(nn.Module):
    def __init__(self, scale, g):
        super().__init__(); self.error_input, self.condition_input = nn.Linear(g.output, 48), nn.Linear(g.text + 1, 48, bias=False)
        self.feature_output, self.score = nn.Linear(48, 16), nn.Linear(16, 1)
        self.register_buffer("scale", scale.clone())
    def features(self, x, c): return self.feature_output((self.error_input(x) + self.condition_input(c)[:, None]).tanh()).tanh()
    def forward(self, x, c): return self.score(self.features(x, c).mean(1))


class PenaltyView(nn.Module):
    def __init__(self, critic, g): super().__init__(); self.critic, self.g = critic, g
    def forward(self, x, c): return self.critic(x.reshape(-1, self.g.tokens, self.g.output), c).repeat_interleave(self.g.tokens, 0)


def model_forward(models, context, candidate, routing): return models["generator"].forward_routed(context, models["router"], candidate, routing)
def features(models, context, samples, targets):
    d = models["critic"]
    return d.features((samples - targets) / d.scale, models["encoder"].condition(context)).flatten(1)


def caption_data(g):
    if g == FULL:
        # Fixed orthonormal DCT directions are independent of all adapter/frozen
        # initialization streams. Cholesky preserves the observed pooled Gram.
        axis = torch.arange(g.text, dtype=torch.float64) + .5
        basis = (torch.arange(14, dtype=torch.float64)[:, None] * axis * math.pi / g.text).cos()
        basis *= math.sqrt(2 / g.text); basis[0] /= math.sqrt(2)
        gram = torch.tensor(CAPTION_GRAM, dtype=torch.float64)
        means = torch.linalg.cholesky((gram + gram.T) / 2) @ basis[:13]
        gen = torch.Generator().manual_seed(103); captions = []; masks = []
        def residual(count, scale):
            if count == 1 or scale == 0: return torch.zeros(count, g.text, dtype=torch.float64)
            r = torch.randn(count, g.text, dtype=torch.float64, generator=gen)
            r = r - (r @ basis.T) @ basis
            r -= r.mean(0); return r * (scale / r.square().mean().sqrt())
        for i, (length, stats) in enumerate(zip(MASK_LENGTHS, CAPTION_STATS)):
            valid_rms, pad_norm, pad_rms, cosine = stats
            valid_mean = means[i]; unit = valid_mean / valid_mean.norm()
            pad_mean = pad_norm * (cosine * unit + math.sqrt(1 - cosine**2) * basis[13])
            valid = valid_mean + residual(length, valid_rms)
            padding = pad_mean + residual(g.length - length, pad_rms)
            captions.append(torch.cat((valid, padding)).float()); masks.append(torch.arange(g.length) < length)
        values, mask = torch.stack(captions), torch.stack(masks)
        return values, mask, {"provenance": TEXT_PROVENANCE, "actual_caption_ids": ACTUAL_CAPTION_IDS,
            "construction": "DCT13 pooled-Gram Cholesky; fixed CPU103 mean-zero orthogonal Gaussian token residuals scaled to per-caption centered RMS; padding matches own norm/cosine using shared orthogonal DCT14 direction.",
            "captions_sha256": digest(values), "masks_sha256": digest(mask),
            "limits": "No pretrained weights/token vectors; cross-caption padded Gram, token covariance/directions and actual activation geometry not matched. GPU initial census failed beforeforward."}
    # Reduced software tensors exercise shape/masking only, never convergence.
    gen = torch.Generator().manual_seed(101)
    captions = torch.randn(13, g.length, g.text, generator=gen)
    lengths = torch.tensor([1] + [min(g.length, 2 + i % max(1, g.length - 1)) for i in range(12)])
    return captions, torch.arange(g.length)[None] < lengths[:, None], {"software_only": True}


def make_data(g=FULL, device=torch.device("cpu")):
    captions, masks, text_law = caption_data(g)
    with torch.random.fork_rng(devices=[]): frozen = Backbone(g)
    initialize(frozen, "frozen_backbone"); frozen.requires_grad_(False)
    data = {"geometry": g, "frozen": deepcopy(frozen.state_dict()), "captions": captions, "masks": masks, "text_law": text_law}
    generator = torch.Generator().manual_seed(102)
    for pool, times, draws in (("fit", (.1, .35, .6, .85), (0, 1)), ("guard", (.22, .72), (2,)), ("test", (.18, .43, .68, .93), (3, 4))):
        contexts, ids = [], []
        for source in range(6):
            for t in times:
                for _ in draws:
                    latent = torch.randn(g.tokens, g.output, generator=generator)
                    contexts.append(torch.cat((latent, torch.tensor([source, t]).expand(g.tokens, -1)), -1)); ids.append(source)
        data[pool] = {"context": torch.stack(contexts).to(device), "targets": torch.zeros(len(contexts), g.tokens, g.output, device=device), "source_ids": ids}
    with torch.random.fork_rng(devices=[]): host = Host(data, ARMS[0])
    initialize(host, "generator")
    with torch.no_grad():
        for branch in host.branches(): branch.up.weight.zero_()
        host.to(device)
        data["fit_baseline"] = torch.cat([host(data["fit"]["context"][i:i+B]) for i in range(0, len(data["fit"]["context"]), B)])
        data["scale"] = data["fit_baseline"].flatten(0, 1).std(0).clamp_min(.04)
    data["digest"] = digest({k: asdict(v) if isinstance(v, Geometry) else v for k, v in data.items()})
    return data


@dataclass
class Loop:
    policy: E22Policy
    arm: str
    data: dict
    data_rng: torch.Generator
    paired_rng: torch.Generator
    globals: dict


def make_loop(arm, data, *, software_C_zero=False):
    if arm not in ARMS: raise ValueError("one fixed arm is required")
    particle = arm != ARMS[0]; device = data["fit"]["context"].device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(900)
        if device.type == "cuda": torch.cuda.manual_seed(900)
        g, d, e = Host(data, arm), Critic(data["scale"].cpu(), data["geometry"]), Encoder(data["captions"], data["masks"], data["geometry"])
        r = Router(data["geometry"]) if particle else None
        initialize(g, "generator"); initialize(d, "critic")
        if r is not None: initialize(r, "router")
        with torch.no_grad():
            for branch in g.branches():
                branch.up.weight.zero_()
                if particle:
                    branch.bridge.weight[:, :data["geometry"].rank].zero_(); branch.bridge.bias.zero_()
                    if software_C_zero: branch.bridge.weight[:, data["geometry"].rank:].zero_()
        g.to(device); d.to(device); e.to(device)
        if r is not None: r.to(device)
        opts = dict(num_particles=N, z_dim=Z, batch_size=B, output_noise_std=SIGMA, birth_death_backend="auto" if particle else "knn", reopen_guard="settled")
        if not particle: opts.update(particle_birth_death=False, row_evidence_gate=False, birth_death_feature_scale="none", birth_death_isolation=False)
        recipe = get_recipe("e22_routed" if particle else "e22", **opts)
        prior = recipe.make_prior(); init.deterministic_orthogonal_(prior); table = prior.to(device).z.requires_grad_(particle)
        groups = [{"params": [v for v in g.parameters() if v.requires_grad], "lr": 5e-5}]
        if particle: groups += [{"params": list(r.parameters()), "lr": 5e-5}, {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}]
        opt_g = recipe.make_generator_optimizer(groups, latent_table=table if particle else None, foreach=False)
        opt_d = recipe.make_critic_optimizer(d, ema_critic=deepcopy(d), foreach=False)
        kwargs = {"routed_rows": RoutedRows(model_forward=model_forward, features=features, sites=SITES, probe_interval=100, max_context_harm=0., output_error_guard=False)} if particle else {"row_semantics": "conditional"}
        p = E22Policy(recipe, g, d, table=table, encoder=e, router=r, generator_optimizer=opt_g, critic_optimizer=opt_d,
            roles=[["generator", "router", "table"] if particle else ["generator"], ["critic"]], seed=21, **kwargs)
        p.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
        # Constructor consumption differs by architecture; native penalty draws
        # start from the declared same streams, not incidental creation order.
        torch.manual_seed(900)
        if device.type == "cuda": torch.cuda.manual_seed(900)
        owned_globals = global_rng(device)
    return Loop(p, arm, data, torch.Generator().manual_seed(7), torch.Generator().manual_seed(43), owned_globals)


def generate(p, x): return p.G(x) if p.routed_control is None else p.routed_generate(x, sigma=0, perturb=True)


def update(loop):
    p, data = loop.policy, loop.data; device = data["fit"]["context"].device
    devices = [device.index or 0] if device.type == "cuda" else []
    # Native penalties may use global randomness: each arm owns an identical
    # initial CPU/CUDA stream, restored/captured around every native update.
    with torch.random.fork_rng(devices=devices):
        set_global_rng(loop.globals, device)
        try:
            ids = torch.randint(len(data["fit"]["context"]), (B,), generator=loop.data_rng)
            x, target = (data["fit"][k][ids] for k in ("context", "targets")); p.G.train(); p.D.train(); c = p.encoder.condition(x)
            bases = [torch.randn(target.shape, generator=loop.paired_rng).to(device) for _ in range(2)]
            routed = RoutedBatch(x, target, data["guard"]["context"], data["guard"]["targets"]) if p.routed_control else None
            noise = p.begin_step(target, routed=routed); loss = p.recipe.make_loss()
            with torch.no_grad(): real = noise.output_sigma * bases[0]; fake = real + (generate(p, x) - target) / p.D.scale
            p.observe_critic_pair(real, fake); penalty = p.penalty(PenaltyView(p.D, data["geometry"]), real.flatten(0, 1), fake.flatten(0, 1), c)
            d_game = loss.d_loss(p.D(real, c), p.D(fake, c)); d_total = d_game + penalty
            p.opt_d.zero_grad(set_to_none=True); p.before_critic_backward(); d_total.backward(); p.opt_d.step(); p.after_critic_step()
            flags = [v.requires_grad for v in p.D.parameters()]; p.D.eval()
            try:
                p.D.requires_grad_(False); real = noise.output_sigma * bases[1]
                with torch.no_grad(): reference = p.D(real.detach(), c)
                g_game = loss.g_loss(p.D(real + (generate(p, x) - target) / p.D.scale, c), reference)
                if not bool(torch.isfinite(g_game)) or not bool(torch.isfinite(d_total)): raise FloatingPointError("nonfinite native game")
                p.opt_g.zero_grad(set_to_none=True); p.before_generator_backward(); g_game.backward()
                def live(values): return values is not None and any(v.grad is not None and bool(v.grad.count_nonzero()) for v in values)
                bank_live, query_live = live([p.table]), live(p.router.parameters() if p.router else None)
                p.after_generator_backward(loss_gan=g_game.detach(), loss_critic=d_game.detach()); p.opt_g.step(); p.after_generator_step()
            finally:
                for value, flag in zip(p.D.parameters(), flags): value.requires_grad_(flag)
            event = p.finish_step()
        finally: loop.globals = global_rng(device)
    return {"step": p.completed_steps, "loss_g": float(g_game.detach()), "loss_d_game": float(d_game.detach()), "penalty": float(penalty.detach()),
        "batch_indices": ids.tolist(), "paired_bases": digest(bases), "data_rng": digest(loop.data_rng.get_state()), "paired_rng": digest(loop.paired_rng.get_state()),
        "DV12_rng": digest(p.noise_generator.get_state()), "penalty_globals": digest(loop.globals), "bank_live": bank_live, "query_live": query_live, "move": event}


def checkpoint(loop):
    p = loop.policy; owners = {k: v for k, v in (("G", p.G), ("D", p.D), ("encoder", p.encoder), ("router", p.router)) if v is not None}
    return deepcopy({"arm": loop.arm, "data_digest": loop.data["digest"], "native": p.state_dict(), "globals": loop.globals,
        "data_rng": loop.data_rng.get_state(), "paired_rng": loop.paired_rng.get_state(), "modes": {k: {n: v.training for n, v in o.named_modules()} for k, o in owners.items()},
        "gradients": {k: {n: v.grad for n, v in o.named_parameters()} for k, o in owners.items()}, "table_grad": p.table.grad, "noise_grad": p.log_output_sigma.grad})


def restore(loop, state):
    if state["arm"] != loop.arm or state["data_digest"] != loop.data["digest"]: raise ValueError("checkpoint precision/data law differs")
    p = loop.policy; p.load_state_dict(state["native"]); loop.globals = deepcopy(state["globals"])
    loop.data_rng.set_state(state["data_rng"]); loop.paired_rng.set_state(state["paired_rng"])
    for role, modes in state["modes"].items():
        owner = getattr(p, role)
        for name, module in owner.named_modules(): module.training = modes[name]
        for name, value in owner.named_parameters():
            grad = state["gradients"][role][name]; value.grad = None if grad is None else grad.to(value.device).clone()
    p.table.grad = None if state["table_grad"] is None else state["table_grad"].to(p.table.device).clone()
    p.log_output_sigma.grad = None if state["noise_grad"] is None else state["noise_grad"].to(p.log_output_sigma.device).clone()


@torch.no_grad()
def capture(p, x, zero_code=False):
    before = digest(p.state_dict()); modes = [(m, m.training) for m in p.G.modules()]; values = []
    try:
        p.G.eval()
        for i in range(0, len(x), B):
            batch = x[i:i+B]
            if p.routed_control is None: value = p.G(batch)
            elif zero_code: value = p.routed_control.generate(batch, candidate=p.routed_control.candidate(), perturb_fn=lambda z: torch.zeros_like(z))
            else: value = p.routed_generate(batch, sigma=0, perturb=False)
            values.append(value.clone())
    finally:
        for m, mode in modes: m.training = mode
    if digest(p.state_dict()) != before: raise AssertionError("clean capture mutated native owners/streams")
    return torch.cat(values)




def trainable_counts(g):
    down_up = g.rank * (g.text + 14 * g.width)
    bridge = 6 * (g.rank * (g.rank + Z) + g.rank)
    router = Z * (g.text + 5 * g.width) + 6 * Z
    return {"ordinary_DownUp": down_up, "particle_bridge": bridge, "router": router, "table": N * Z,
            "particle_total": down_up + bridge + router + N * Z, "total_ratio": (down_up + bridge + router + N * Z) / down_up}


def accuracy(residual, source_ids):
    """Unscaled physical prediction-minus-teacher RMSE, offline evaluation only."""
    labels = torch.as_tensor(source_ids, device=residual.device)
    if (residual.ndim != 3 or not len(residual) or not residual.is_floating_point()
        or labels.shape != (len(residual),) or labels.dtype not in (torch.int32, torch.int64)
        or not bool(torch.isfinite(residual).all()) or set(labels.tolist()) != set(range(6))):
        raise ValueError("finite paired residuals and all six integer source IDs are required")
    powers = residual.double().square().flatten(1).mean(1)
    return {"rmse": float(powers.mean().sqrt()),
        "by_source": {str(i): float(powers[labels == i].mean().sqrt()) for i in range(6)},
        "source_counts": {str(i): int((labels == i).sum()) for i in range(6)},
        "contexts": len(residual), "coordinates_per_context": residual[0].numel()}


def scientific_gate(ordinary, particle, zero_code, *, bank_updates, query_updates, C_norms):
    """Future APIs can pass directly; reproducing a historical failure is absent."""
    keys = {str(i) for i in range(6)}
    if any(set(v["by_source"]) != keys for v in (ordinary, particle, zero_code)):
        raise ValueError("all six sources are mandatory")
    numbers = [value for v in (ordinary, particle, zero_code)
               for value in (v["rmse"], *v["by_source"].values())]
    if not all(math.isfinite(v) and v >= 0 for v in numbers):
        raise ValueError("offline physical accuracy must be finite and nonnegative")
    if any(type(v) is not int or not 0 <= v <= STEPS - 1 for v in (bank_updates, query_updates)):
        raise ValueError("native live counts must use updates2..512")
    checks = {
        "aggregate_accuracy_improved": particle["rmse"] <= ordinary["rmse"] * (1 - RELATIVE_MARGIN),
        "no_source_harmed": all(particle["by_source"][k] <= ordinary["by_source"][k] + SOURCE_HARM for k in keys),
        "aggregate_code_benefit": zero_code["rmse"] >= particle["rmse"] * (1 + RELATIVE_MARGIN),
        "code_beneficial_each_source": all(zero_code["by_source"][k] > particle["by_source"][k] for k in keys),
        "bank_live": bank_updates / (STEPS - 1) >= LIVE_FRACTION,
        "query_live": query_updates / (STEPS - 1) >= LIVE_FRACTION,
        "C_live_all_six_sites": set(C_norms) == set(SITES) and all(math.isfinite(v) and v > 0 for v in C_norms.values()),
    }
    return {"pass": all(checks.values()), **checks,
        "relative_accuracy_improvement": None if ordinary["rmse"] == 0 else 1 - particle["rmse"] / ordinary["rmse"],
        "relative_code_benefit": None if particle["rmse"] == 0 else zero_code["rmse"] / particle["rmse"] - 1,
        "relative_margin": RELATIVE_MARGIN, "per_source_harm_tolerance": SOURCE_HARM,
        "live_fraction_threshold": LIVE_FRACTION, "live_denominator": STEPS - 1}


def learned_finite(p):
    """Learned weights/gradients/moments, without validating private monitors."""
    def check(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite learned weight, gradient or optimizer moment")
        if isinstance(value, dict):
            for item in value.values(): check(item)
        elif isinstance(value, (tuple, list)):
            for item in value: check(item)
    for model in (p.G, p.D, p.encoder, p.router, p.ema_G, p.ema_encoder, p.ema_router, p.opt_d.ema_critic):
        if model is not None:
            for parameter in model.parameters(): check(parameter); check(parameter.grad)
    check((p.table, p.table.grad, p.averaged_table, p.log_output_sigma, p.log_output_sigma.grad))
    for optimizer in (p.opt_g, p.opt_d): check(optimizer.state)


def scorer_controls():
    """Analytical oracle and destructive controls, before any training."""
    labels = torch.arange(6)
    zero = accuracy(torch.zeros(6, 2, 4), labels)
    one = accuracy(torch.ones(6, 2, 4), labels)
    if zero["rmse"] != 0 or any(zero["by_source"].values()) or one["rmse"] != 1:
        raise AssertionError("physical RMSE oracle differs")
    def metric(v): return {"rmse": v, "by_source": {str(i): v for i in range(6)}}
    kwargs = {"bank_updates": STEPS - 1, "query_updates": STEPS - 1, "C_norms": {site: .1 for site in SITES}}
    if not scientific_gate(metric(1), metric(.99), metric(1), **kwargs)["pass"]:
        raise AssertionError("positive accuracy/code oracle rejected")
    if scientific_gate(metric(1), metric(1), metric(1), **kwargs)["pass"]:
        raise AssertionError("no-change destructive control passed")
    harmed = metric(.99); harmed["by_source"]["5"] = 1 + 2 * SOURCE_HARM
    if scientific_gate(metric(1), harmed, metric(1.01), **kwargs)["pass"]:
        raise AssertionError("single-source harm destructive control passed")
    try: accuracy(torch.full((6, 2, 4), float("nan")), labels)
    except ValueError: pass
    else: raise AssertionError("nonfinite destructive control passed")
    return {"zero_target_oracle": True, "unit_error_oracle": True, "positive_gate_oracle": True,
            "no_change_harmed_source_and_nonfinite_controls_rejected": True}


def observe(loop, context, *, zero_code=False):
    """Native clean capture plus caller streams, modes and gradient identity."""
    p = loop.policy
    owners = tuple(m for m in (p.G, p.D, p.encoder, p.router, p.ema_G, p.ema_encoder, p.ema_router, p.opt_d.ema_critic) if m is not None)
    def caller():
        return digest({"data_rng": loop.data_rng.get_state(), "paired_rng": loop.paired_rng.get_state(), "owned_globals": loop.globals,
            "modes": [[m.training for m in owner.modules()] for owner in owners],
            "gradients": [[v.grad for v in owner.parameters()] for owner in owners],
            "table_grad": p.table.grad, "noise_grad": p.log_output_sigma.grad,
            "closed_routes": [b.route is None for b in p.G.branches()]})
    before = caller(); result = capture(p, context, zero_code=zero_code)
    if before != caller(): raise AssertionError("observation changed caller streams, modes, gradients or route lifecycle")
    return result


def preflight(data):
    """Zero native updates; reduced instances are software-only."""
    device = data["fit"]["context"].device; before = global_rng(device)
    loops = [make_loop(arm, data) for arm in ARMS]
    ordinary, particle = (loop.policy for loop in loops)
    for left, right in zip(ordinary.G.branches(), particle.G.branches()):
        if not torch.equal(left.down.weight, right.down.weight) or not torch.equal(left.up.weight, right.up.weight):
            raise AssertionError("common named Down/Up initial values differ")
    if digest(ordinary.D.state_dict()) != digest(particle.D.state_dict()): raise AssertionError("initial critics differ")
    if digest(loops[0].globals) != digest(loops[1].globals): raise AssertionError("initial penalty streams differ")
    x = data["test"]["context"][:B]; values = [capture(p, x) for p in (ordinary, particle)]
    if not torch.equal(*values): raise AssertionError("zero-Up initial predictions differ")
    for loop in loops:
        learned_finite(loop.policy)
        for branch in loop.policy.G.branches():
            if branch.up.weight.count_nonzero() or branch.base.weight.requires_grad:
                raise AssertionError("fresh Up or frozen base differs")
            if loop.arm == ARMS[1] and (branch.bridge.weight[:, :data["geometry"].rank].count_nonzero()
                or branch.bridge.bias.count_nonzero() or not branch.bridge.weight[:, data["geometry"].rank:].count_nonzero()):
                raise AssertionError("H/b-zero sampled-C initialization differs")
        # Fresh construction uses initialize_; public restoration itself does not.
        state = checkpoint(loop); recovered = make_loop(loop.arm, data); restore(recovered, state)
        if digest(checkpoint(recovered)) != digest(state) or not torch.equal(capture(recovered.policy, x), capture(loop.policy, x)):
            raise AssertionError("fresh-owner public initial restore differs")
        del recovered
    if digest(before) != digest(global_rng(device)): raise AssertionError("preflight changed global RNG")
    return {"pass": True, "native_updates": 0, "fresh_public_restore_exact": True,
        "initial_DownUp_critics_and_predictions_exact": True, "counts": trainable_counts(data["geometry"]),
        "scope": "software only" if data["geometry"] != FULL else "zero-update full-geometry prerequisite"}


def run(data, out):
    """Fixed horizons: no accuracy enters native updates or a stopping rule."""
    results, predictions, summaries, media = {}, {}, {}, {}
    matched = None; coverage = {"bank": 0, "query": 0}; C_norms = {}
    controls = scorer_controls()
    camera = data["test"]["context"][list(MEDIA_INDICES)]
    for arm in ARMS:
        loop = make_loop(arm, data); p = loop.policy; matched_stream = hashlib.sha256()
        media[arm] = {"0": observe(loop, camera).cpu()}
        frozen = digest({n: v for n, v in p.G.state_dict().items()
                         if not n.endswith(("down.weight", "up.weight", "bridge.weight", "bridge.bias"))})
        loss_ema = None
        for step in range(1, STEPS + 1):
            row = update(loop)
            matched_stream.update(json.dumps({k: row[k] for k in ("step", "batch_indices", "paired_bases", "data_rng", "paired_rng", "penalty_globals")}, sort_keys=True).encode())
            if arm == ARMS[1] and step > 1:
                coverage["bank"] += int(row["bank_live"]); coverage["query"] += int(row["query_live"])
            with (out / f"{arm}.jsonl").open("a") as stream: stream.write(json.dumps(row, allow_nan=False) + "\n")
            loss_ema = row["loss_g"] if loss_ema is None else .98 * loss_ema + .02 * row["loss_g"]
            if step % 64 == 0:
                print(json.dumps({"arm": arm, "step": step, "steps": STEPS, "native_G_loss": row["loss_g"],
                    "native_G_loss_EMA": loss_ema, "native_D_loss": row["loss_d_game"], "seconds": time.monotonic() - STARTED}), flush=True)
            if step in MEDIA_STEPS:
                media[arm][str(step)] = observe(loop, camera).cpu()
            budget()
        learned_finite(p)
        if matched is None: matched = matched_stream.hexdigest()
        elif matched_stream.hexdigest() != matched: raise AssertionError("external batches/paired Gaussian/native penalty streams differ")
        if frozen != digest({n: v for n, v in p.G.state_dict().items()
                             if not n.endswith(("down.weight", "up.weight", "bridge.weight", "bridge.bias"))}):
            raise AssertionError("frozen host/teacher/captions changed")
        # Same terminal endpoint: restoration is an integrity check, never a
        # second selected checkpoint or additional optimization step.
        state = checkpoint(loop); predictions[arm] = observe(loop, data["test"]["context"])
        restore(loop, state)
        if digest(checkpoint(loop)) != digest(state) or not torch.equal(predictions[arm], observe(loop, data["test"]["context"])):
            raise AssertionError("terminal public restore/prediction replay differs")
        results[arm] = accuracy(predictions[arm], data["test"]["source_ids"])
        summaries[arm] = {"steps": p.completed_steps, "recipe": p.recipe.to_dict(), "native_state_digest": digest(state["native"]),
            "caller_and_modes_gradients_digest": digest({k: v for k, v in state.items() if k != "native"}),
            "public_terminal_restore_and_output_exact": True, "learned_weights_gradients_moments_finite": True}
        if arm == ARMS[1]:
            predictions["zero_code"] = observe(loop, data["test"]["context"], zero_code=True)
            results["zero_code"] = accuracy(predictions["zero_code"], data["test"]["source_ids"])
            C_norms = {b.site: float(b.bridge.weight[:, data["geometry"].rank:].detach().norm()) for b in p.G.branches()}
        del loop, p, state
        budget()
    gate = scientific_gate(results[ARMS[0]], results[ARMS[1]], results["zero_code"],
        bank_updates=coverage["bank"], query_updates=coverage["query"], C_norms=C_norms)
    artifact = out / "endpoint-residuals.pt"
    torch.save({"physical_residuals": predictions, "source_ids": data["test"]["source_ids"],
        "test_context_digest": digest(data["test"]["context"]), "target_digest": digest(data["test"]["targets"])}, artifact)
    media_artifact = out / "observed-media.pt"
    torch.save({"actual_residuals": media, "target_residual": torch.zeros_like(media[ARMS[0]]["0"]),
        "steps": MEDIA_STEPS, "indices": MEDIA_INDICES, "rendered_indices": MEDIA_INDICES[:6],
        "source_ids": [data["test"]["source_ids"][i] for i in MEDIA_INDICES],
        "context_digest": digest(camera), "capture_native_state_rng_diagnostics_unchanged": True}, media_artifact)
    budget()
    return {"complete": True, "scientific_status": "PASS" if gate["pass"] else "FAIL", "gate": gate,
        "metric": "terminal512 unscaled physical prediction-minus-teacher RMSE, float64 reduction, offline only",
        "accuracy": results, "arms": summaries, "coverage": {**coverage, "denominator": STEPS - 1}, "C_norms": C_norms,
        "quality_updates": 2 * STEPS, "replay_updates": 0, "endpoint_steps": [STEPS],
        "matched_external_data_Gaussian_and_native_penalty_streams": True, "data_digest": data["digest"],
        "scorer_oracles_and_destructive_controls": controls, "media_steps": list(MEDIA_STEPS),
        "observed_media_sha256": sha(media_artifact), "media_capture_native_state_rng_diagnostics_unchanged": True,
        "geometry": asdict(data["geometry"]), "counts": trainable_counts(data["geometry"]), "text_law": data["text_law"],
        "endpoint_residual_sha256": sha(artifact), "scope": "One generated caption-retargeting task; no pretrained/full-Supra win or precision remedy established."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true"); actions.add_argument("--run", action="store_true")
    parser.add_argument("--protocol", type=Path, default=CARD); parser.add_argument("--out", type=Path)
    args = parser.parse_args(); previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    imported = package_identity(); source_sha = sha(__file__); protocol_sha = None
    device = None; entry = None; out = None; report = None; failure = None; code = 2
    try:
        card = json.loads(args.protocol.read_text()); protocol_sha = sha(args.protocol)
        expected = {"task": TASK, "arms": list(ARMS), "geometry": asdict(FULL), "steps": STEPS, "seconds": SECONDS,
            "media_steps": list(MEDIA_STEPS), "media_indices": list(MEDIA_INDICES),
            "accuracy_thresholds": {"relative_aggregate_improvement_gte": RELATIVE_MARGIN, "source_harm_lte": SOURCE_HARM,
                "relative_code_benefit_gte": RELATIVE_MARGIN, "code_beneficial_every_source": True,
                "bank_router_live_fraction_gte": LIVE_FRACTION, "live_denominator": STEPS - 1}}
        if any(card[k] != v for k, v in expected.items()): raise ValueError("fixed benchmark physics/metric/budget differs")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
            raise ValueError("full benchmark requires physical GPU0 via CUDA_VISIBLE_DEVICES=0")
        if backend_flags() != card["precision_backend"]: raise ValueError("declared precision backend differs")
        if args.run:
            if args.out is None or args.out.exists(): raise ValueError("a fresh --out directory is required")
            args.out.mkdir(parents=True); out = args.out
        device = torch.device("cuda:0"); entry = global_rng(device)
        scorer_controls(); data = make_data(FULL, device); checks = preflight(data); budget()
        if args.preflight:
            report = {"complete": True, "preflight": checks, "quality_updates": 0, "scope": "full-host zero-update prerequisite"}
            code = 0
        else:
            report = run(data, out); code = 0 if report["gate"]["pass"] else 1
        if package_identity() != imported or sha(__file__) != source_sha or sha(args.protocol) != protocol_sha:
            raise ValueError("source/protocol/imported package changed within run")
        if backend_flags() != card["precision_backend"]: raise ValueError("precision backend changed within run")
        report.update(task=TASK, imported_package=imported, source_sha256=source_sha, protocol_sha256=protocol_sha,
            imported_package_unchanged=True, precision_backend=backend_flags())
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        import traceback
        traceback.print_exc(); code = 2
    finally:
        if entry is not None:
            set_global_rng(entry, device)
            if digest(global_rng(device)) != digest(entry): failure = {"type": "AssertionError", "message": "caller RNG restoration differs"}; code = 2
        torch.set_num_threads(previous_threads)
        elapsed = time.monotonic() - STARTED
        if elapsed > SECONDS: failure = {"type": "TimeoutError", "message": "startup-through-cleanup300s budget exceeded"}; code = 2
        completion = {"complete": failure is None, "scientific_status": None if failure or report is None else report.get("scientific_status"),
            "error": failure, "seconds": elapsed, "limit_seconds": SECONDS, "source_sha256": source_sha,
            "protocol_sha256": protocol_sha, "imported_package": imported,
            "caller_CPU_CUDA_RNG_restored": entry is not None and digest(global_rng(device)) == digest(entry)}
        if out is not None:
            if report is not None:
                report["seconds"] = elapsed; (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                completion["report_sha256"] = sha(out / "report.json")
            path = out / "completion.json"; path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
            if time.monotonic() - STARTED > SECONDS:
                completion.update(complete=False, error={"type": "TimeoutError", "message": "final serialization exceeded300s"}, seconds=time.monotonic() - STARTED)
                path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n"); code = 2
        print(json.dumps({"completion": completion, "preflight": report if args.preflight else None,
                          "scientific_gate": None if report is None else report.get("gate")}, allow_nan=False), flush=True)
    return code


if __name__ == "__main__": raise SystemExit(main())
