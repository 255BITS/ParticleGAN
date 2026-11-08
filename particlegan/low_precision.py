"""Software references for Blackwell NVFP4 and MXFP8 linear training.

These emulate quantized GEMM operands, not native execution time or storage.
FP32 master parameters/updates and BF16 GEMM outputs are retained. Higher-order
autograd uses explicit straight-through quantizer derivatives; hardware parity
of those surrogate derivatives is not claimed.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

FORMATS = ("nvfp4", "mxfp8")


def _fp4(values, *, generator=None):
    levels = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    magnitude = values.abs().clamp(max=6.)
    if generator is None:
        boundaries = (levels[:-1] + levels[1:]) / 2
        index = torch.bucketize(magnitude.contiguous(), boundaries)
        # E2M1 round-to-nearest-even, including exact midpoint ties.
        tie = (index < 7) & (magnitude == boundaries[index.clamp(max=6)])
        index = index + (tie & (index.remainder(2) == 1)).long()
    else:
        upper = torch.searchsorted(levels, magnitude.contiguous()).clamp(1, 7)
        lower = upper - 1
        probability = (magnitude - levels[lower]) / (levels[upper] - levels[lower])
        draw = torch.rand(values.shape, device=values.device, generator=generator)
        index = lower + (draw < probability).long()
    return values.sign() * levels[index]


def quantize_dequantize(values, format, *, weight=False, generator=None):
    """Quantize a matrix using E2M1/E4M3 scales or E4M3/E8M0 scales.

    Padding represents zero-filled hardware alignment, not extra parameters.
    NVFP4 weights share a scale per 16x16 tile; all other blocks are rowwise.
    MXFP8 transposes must be independently quantized from the original matrix.
    """
    if format not in FORMATS or values.ndim != 2:
        raise ValueError("expected a matrix and nvfp4 or mxfp8")
    shape = values.shape
    x = values.detach().to(torch.bfloat16).float()
    block = 16 if format == "nvfp4" else 32
    x = F.pad(x, (0, (-shape[1]) % block, 0, (-shape[0]) % block))
    if format == "nvfp4" and weight:
        blocks = x.reshape(x.shape[0] // block, block, x.shape[1] // block, block).permute(0, 2, 1, 3)
        amax = blocks.abs().amax(dim=(-1, -2), keepdim=True)
    else:
        blocks = x.reshape(x.shape[0], -1, block)
        amax = blocks.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(torch.float32).tiny
    if format == "nvfp4":
        global_scale = (x.abs().amax() / (448. * 6.)).clamp_min(tiny)
        local_scale = ((amax / 6.) / global_scale).clamp(max=448.).to(torch.float8_e4m3fn).float()
        scale = (global_scale * local_scale).clamp_min(tiny)
        result = _fp4(blocks / scale, generator=generator) * scale
    else:
        scale = torch.exp2(torch.ceil(torch.log2((amax / 448.).clamp_min(tiny))))
        result = (blocks / scale).clamp(-448., 448.).to(torch.float8_e4m3fn).float() * scale
    if format == "nvfp4" and weight:
        result = result.permute(0, 2, 1, 3).reshape(x.shape)
    else:
        result = result.reshape(x.shape)
    return result[:shape[0], :shape[1]]


def _quantized_proxy(values, format, **options):
    rounded = quantize_dequantize(values, format, **options)
    return values + (rounded - values).detach()


def _bf16(values):
    return values.to(torch.bfloat16).to(values.dtype)


def _rotate_columns(values, hadamard):
    columns = values.shape[-1]
    padded = F.pad(_bf16(values), (0, (-columns) % 16))
    return _bf16((padded.reshape(-1, 16) @ hadamard).reshape(padded.shape))


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, owner):
        ctx.save_for_backward(x, weight)
        ctx.owner, ctx.has_bias = owner, bias is not None
        owner.calls["forward"] += 1
        qx = quantize_dequantize(x, owner.format)
        qw = quantize_dequantize(weight, owner.format, weight=True)
        result = qx @ qw.T
        if bias is not None:
            result = result + _bf16(bias)
        return _bf16(result)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        owner, format = ctx.owner, ctx.owner.format
        options = {"generator": owner.rounding_rng} if format == "nvfp4" else {}
        dx = dw = None
        if ctx.needs_input_grad[0]:
            owner.calls["dgrad"] += 1
            qg = _quantized_proxy(grad, format, **options)
            qw = (_quantized_proxy(weight, format, weight=True) if format == "nvfp4"
                  else _quantized_proxy(weight.T, format).T)
            dx = _bf16(qg @ qw)
        if ctx.needs_input_grad[1]:
            owner.calls["wgrad"] += 1
            # WGRAD's contraction axis is the batch dimension. Rotate both
            # transposed operands identically, then quantize them separately.
            gt, xt = grad.T, x.T
            if format == "nvfp4":
                gt, xt = _rotate_columns(gt, owner.hadamard), _rotate_columns(xt, owner.hadamard)
            qg = _quantized_proxy(gt, format, **options)
            qx = _quantized_proxy(xt, format)
            dw = _bf16(qg @ qx.T)
        db = _bf16(grad.sum(0)) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


class EmulatedLinear(nn.Linear):
    """Linear with the same trainable parameters and quantized FPROP/DGRAD/WGRAD.

    Each NVFP4 layer owns a checkpointed rounding stream. Evaluation has no
    stochastic forwards and cannot consume that stream. The fixed RHT is shared
    across layers. This class is opt-in and never changes existing models.
    """
    @classmethod
    def from_linear(cls, layer, format, *, seed=0, signs=None):
        if format not in FORMATS:
            raise ValueError("unsupported linear precision")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.in_features, result.out_features = layer.in_features, layer.out_features
        result.weight, result.bias = layer.weight, layer.bias
        result.format = format
        result.rounding_rng = torch.Generator(device=layer.weight.device).manual_seed(seed)
        h = torch.ones(1, 1, device=layer.weight.device)
        for _ in range(4):
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        if signs is None:
            signs = 2 * torch.randint(0, 2, (16,), generator=torch.Generator().manual_seed(0)).float() - 1
        result.register_buffer("hadamard", signs.to(h.device)[:, None] * h / 4.)
        result.calls = {"forward": 0, "dgrad": 0, "wgrad": 0}
        result.train(layer.training)
        return result

    def forward(self, x):
        shape = x.shape
        return _Linear.apply(x.reshape(-1, shape[-1]), self.weight, self.bias, self).reshape(*shape[:-1], self.out_features)

    def get_extra_state(self):
        # GANTrainer checks tensor schema before restore. Keep dynamic RNG bytes
        # in a tensor, with a format tag so a cross-precision load still refuses.
        tag = torch.tensor([4 if self.format == "nvfp4" else 8], dtype=torch.uint8)
        return torch.cat((tag, self.rounding_rng.get_state().cpu()))

    def set_extra_state(self, state):
        if not isinstance(state, torch.Tensor) or int(state[0]) != (4 if self.format == "nvfp4" else 8):
            raise ValueError("checkpoint linear precision differs")
        self.rounding_rng.set_state(state[1:].cpu().clone())


def emulate_linear_training_(module, format, *, seed_for=lambda name: 0, signs=None):
    """Replace Linear execution without draws or parameter changes; return root.

    Containers are modified in place. Assign the return value when the root is
    itself a Linear. Shared child modules remain shared.
    """
    if format not in FORMATS:
        raise ValueError("unsupported linear precision")
    for child in module.modules():
        if isinstance(child, nn.Linear) and type(child) is not nn.Linear:
            raise ValueError("linear precision requires unmodified torch.nn.Linear modules")
    converted = {}
    def replace(parent, prefix=""):
        if id(parent) in converted:
            return converted[id(parent)]
        if type(parent) is nn.Linear:
            result = EmulatedLinear.from_linear(parent, format, seed=seed_for(prefix), signs=signs)
        else:
            result = parent
            for name, child in list(parent._modules.items()):
                if child is not None:
                    path = f"{prefix}.{name}" if prefix else name
                    setattr(parent, name, replace(child, path))
        converted[id(parent)] = result
        return result
    return replace(module)
