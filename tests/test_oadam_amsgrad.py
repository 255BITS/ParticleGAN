"""lib/oadam.py: the default path is the vendored algorithm bit for bit; amsgrad keeps a non-shrinking max buffer."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("lib_oadam", Path(__file__).resolve().parents[1] / "lib" / "oadam.py")
oadam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oadam)
OptimisticAdam = oadam.OptimisticAdam


@torch.no_grad()
def reference_step(params, state, lr, betas, eps):
    """The vendored (pre-amsgrad) OptimisticAdam.step body, verbatim."""
    beta1, beta2 = betas
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad
        st = state.setdefault(p, {})
        if not st:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)
            st["prev_step"] = torch.zeros_like(p)
        st["step"] += 1
        t = st["step"]
        m, v = st["exp_avg"], st["exp_avg_sq"]
        prev_step = st["prev_step"]
        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        denom = v.div(1.0 - beta2 ** t).sqrt_().add_(eps)
        cur_step = m.div(1.0 - beta1 ** t).div_(denom)
        p.add_(cur_step, alpha=-2.0 * lr).add_(prev_step, alpha=lr)
        prev_step.copy_(cur_step)


def _bilinear_game(opt_x, opt_y, x, y, A):
    """min_x max_y x^T A y + .1|x|^2 - .1|y|^2: one simultaneous step."""
    loss = x @ A @ y + 0.1 * x.square().sum() - 0.1 * y.square().sum()
    gx, gy = torch.autograd.grad(loss, (x, y))
    x.grad, y.grad = gx, -gy
    opt_x(), opt_y()


def test_default_is_bit_identical_to_the_vendored_algorithm():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(5, 5, generator=g)
    x0, y0 = torch.randn(5, generator=g), torch.randn(5, generator=g)
    for betas in ((0.0, 0.999), (0.9, 0.99)):
        x, y = x0.clone().requires_grad_(True), y0.clone().requires_grad_(True)
        xr, yr = x0.clone().requires_grad_(True), y0.clone().requires_grad_(True)
        ox, oy = OptimisticAdam([x], lr=0.05, betas=betas), OptimisticAdam([y], lr=0.05, betas=betas)
        state = {}
        for _ in range(200):
            _bilinear_game(ox.step, oy.step, x, y, A)
            _bilinear_game(lambda: reference_step([xr], state, 0.05, betas, 1e-8),
                           lambda: reference_step([yr], state, 0.05, betas, 1e-8), xr, yr, A)
            assert torch.equal(x, xr) and torch.equal(y, yr)
        assert "max_exp_avg_sq" not in ox.state[x]


def test_amsgrad_max_buffer_never_shrinks_and_is_the_running_max():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(50))
    opt = OptimisticAdam([p], lr=0.01, betas=(0.0, 0.999), amsgrad=True)
    v_ref, vmax_ref, prev = torch.zeros(50), torch.zeros(50), None
    for t in range(1, 301):
        scale = 10.0 if t < 20 else 0.01  # a spike, then small gradients: plain v decays, vmax must not
        p.grad = scale * torch.randn(50)
        v_ref = 0.999 * v_ref + 0.001 * p.grad.square()
        vmax_ref = torch.maximum(vmax_ref, v_ref)
        opt.step()
        vmax = opt.state[p]["max_exp_avg_sq"]
        assert torch.allclose(vmax, vmax_ref, rtol=1e-6)
        if prev is not None:
            assert (vmax >= prev).all()
        prev = vmax.clone()
    assert (opt.state[p]["exp_avg_sq"] < vmax).all()  # the plain v did shrink below the max


def test_amsgrad_lookback_uses_the_amsgrad_preconditioned_step():
    """p_t = mhat / (sqrt(vmax / bc2) + eps) is torch Adam(amsgrad)'s step direction; update = -2 lr p_t + lr p_{t-1}."""
    torch.manual_seed(1)
    grads = [torch.randn(8) * (5.0 if t == 3 else 1.0) for t in range(10)]
    p = torch.nn.Parameter(torch.zeros(8))
    q = torch.nn.Parameter(torch.zeros(8))
    opt = OptimisticAdam([p], lr=1.0, betas=(0.5, 0.99), amsgrad=True)
    adam = torch.optim.Adam([q], lr=1.0, betas=(0.5, 0.99), amsgrad=True)
    prev = torch.zeros(8)
    for g in grads:
        before, q_before = p.detach().clone(), q.detach().clone()
        p.grad, q.grad = g.clone(), g.clone()
        opt.step()
        adam.step()
        adam_step = q_before - q.detach()
        assert torch.allclose(opt.state[p]["prev_step"], adam_step, rtol=1e-5, atol=1e-7)
        assert torch.allclose(p.detach() - before, -2 * adam_step + prev, rtol=1e-5, atol=1e-6)
        prev = adam_step
