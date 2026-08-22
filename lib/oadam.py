r"""Optimistic Adam: Algorithm 1 of "Training GANs with Optimism".

A drop-in `torch.optim.Optimizer` implementing the update rule of Daskalakis,
Ilyas, Syrgkanis & Zeng, ICLR 2018 (arXiv:1711.00141), for minimax problems
where plain Adam's rotation around the equilibrium stalls or cycles.

    m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    p_t     = (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps)
    theta_t = theta_{t-1} - 2 * lr * p_t + lr * p_{t-1}

i.e. take TWICE the current Adam step and UNDO the previous one.

THE DISTINGUISHING FEATURE, and the reason this exists as its own module: the
optimism correction is applied to the PRECONDITIONED step p, NOT to the raw
gradient. Implementations that circulate online commonly feed 2 g_t - g_{t-1}
into an Adam update instead, or flip the sign of the lookback; both are
different algorithms with different fixed points. See `OptimisticAdam` for the
full derivation and for the two places where the printed algorithm leaves a
choice.

Depends on torch alone.

Vendored unchanged from ParticleGAN-WorldModel/pwm/oadam.py (2026-08-21).
"""

from __future__ import annotations

import torch

__all__ = ["OptimisticAdam"]


class OptimisticAdam(torch.optim.Optimizer):
    r"""Optimistic Adam, Algorithm 1 of Daskalakis, Ilyas, Syrgkanis & Zeng,
    "Training GANs with Optimism", ICLR 2018 (arXiv:1711.00141).

    IMPLEMENTED FROM THE PAPER, DELIBERATELY NOT IMPORTED FROM ANY THIRD-PARTY
    PACKAGE (Martyn 2026-08-21): the implementations that circulate online
    commonly apply the optimism correction to the RAW GRADIENT before the
    moment updates (feeding 2 g_t - g_{t-1} into Adam), or flip the sign of
    the lookback. Both are different algorithms. In Algorithm 1 the correction
    is applied to the PRECONDITIONED step.

    THE EQUATION, exactly as implemented, for each parameter theta and step
    t = 1, 2, ... with g_t the gradient at theta_{t-1}:

        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        mhat_t  = m_t / (1 - beta1^t)          (bias correction)
        vhat_t  = v_t / (1 - beta2^t)
        p_t     = mhat_t / (sqrt(vhat_t) + eps)     <- the PRECONDITIONED step
                                                       (eps OUTSIDE the sqrt,
                                                        torch.optim.Adam's
                                                        placement)
        theta_t = theta_{t-1} - 2 * lr * p_t + lr * p_{t-1}

    i.e. take TWICE the current Adam step and UNDO the previous one; p_{t-1} is
    carried per-parameter in the state as `prev_step`. At t = 1 the moments
    start at zero and there is no previous step, so p_0 == 0 exactly and the
    first update is precisely twice Adam's first update — which is what the
    paper's initialisation (m_0 = v_0 = 0) gives.

    Setting the lookback to zero recovers Adam with a doubled learning rate,
    NOT Adam: the whole content of the method is that the two terms cancel to a
    single Adam step in the limit of a slowly-moving gradient field, while the
    extra half-step of anticipation cancels the ROTATION of a minimax field
    (Daskalakis 2018 Thm; Mescheder 2017 for the rotation diagnosis).

    Two points where the printed algorithm leaves a choice, and what was
    chosen here:
      1. BIAS CORRECTION. The paper's displayed Algorithm 1 writes the moments
         without the hats; its text derives the method as "Adam's update with
         the optimistic step", and Adam means bias-corrected moments. We apply
         the standard Adam bias correction to BOTH terms — the current step and
         the stored lookback (the lookback was itself bias-corrected at the
         step it was computed). This is also the only choice that makes
         swapping this optimizer in a pure change of update rule against an
         Adam baseline, which corrects.
      2. LEARNING RATE ON THE LOOKBACK. `prev_step` stores the preconditioned
         direction p_{t-1} WITHOUT the learning rate, and the current group lr
         multiplies both terms. Identical to the paper for a constant lr and
         the well-behaved choice if one is ever scheduled.

    Supports the subset of the torch.optim.Adam interface a minimax loop
    needs: per-group lr/betas/eps, and nothing else (no weight decay, no
    amsgrad, no fused/foreach kernels, no sparse gradients).
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"invalid lr {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid eps {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("OptimisticAdam does not support sparse "
                                       "gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # p_0 == 0: at t == 1 the lookback term contributes nothing.
                    state["prev_step"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                prev_step = state["prev_step"]

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                # denominator: sqrt(vhat) + eps  (eps OUTSIDE the sqrt)
                denom = v.div(1.0 - beta2 ** t).sqrt_().add_(eps)
                cur_step = m.div(1.0 - beta1 ** t).div_(denom)

                # theta <- theta - 2*lr*p_t + lr*p_{t-1}
                p.add_(cur_step, alpha=-2.0 * lr).add_(prev_step, alpha=lr)
                prev_step.copy_(cur_step)
        return loss
