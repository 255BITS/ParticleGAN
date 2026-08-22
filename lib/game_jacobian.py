"""
Local spectral analysis of the GAN game's update map.

Mescheder et al. (2018) frame GAN convergence as a property of the *update
operator*, not of the loss: write one full alternating round of training as a
map F(theta) on the concatenated parameters of all players, and the equilibrium
is locally attracting iff every eigenvalue of the Jacobian F'(theta) has
modulus < 1. Vanilla GAN dynamics sit exactly on the unit circle (they orbit);
a zero-centered gradient penalty is what pulls the spectrum inside it.

That makes the dominant eigenvalue modulus a direct, per-arm measurement of how
much damping a gradient penalty actually buys — computable without waiting to
see whether a run oscillates.

The map implemented here mirrors the training loop:

    1. one plain gradient-descent step of the D player
       (RpGAN d_loss + the regularizer's penalty),
    2. one plain gradient-descent step of the G + prior player
       (RpGAN g_loss + VICReg on the batch's particles),
       evaluated at the POST-D-step discriminator,

both on a *fixed* real batch and *fixed* particle indices, so F is a
deterministic function of the parameters. Plain GD rather than Adam: we want
the geometry of the game, not the optimizer's own state-space.

Cost: the Jacobian is never formed. Jacobian-vector products come from central
finite differences of F, and the spectrum from a short Arnoldi iteration, so
the whole analysis costs O(krylov_dim) forward/backward passes.
"""

import copy
from typing import Dict, List

import numpy as np
import torch


# =========================
#  Flat parameter plumbing
# =========================

def _flat(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of tensors into one flat 1-D vector."""
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten_into(
    vec: torch.Tensor,
    params: List[torch.Tensor],
    offset: int = 0,
) -> int:
    """Copy slices of `vec` into `params` in order; returns the new offset."""
    for p in params:
        n = p.numel()
        p.copy_(vec[offset : offset + n].view_as(p))
        offset += n
    return offset


# =========================
#  Main entry point
# =========================

def estimate_update_spectrum(
    D: torch.nn.Module,
    G: torch.nn.Module,
    prior: torch.nn.Module,
    gan_loss,
    vic_reg,
    regularizer,
    x_real_fixed: torch.Tensor,
    idx_fixed: torch.Tensor,
    lrs: Dict[str, float],
    krylov_dim: int = 24,
    fd_eps_scale: float = 1e-3,
    lambda_ep: float = 1.0,
    seed: int = 0,
    power_iters: int = 3,
    penalty_step: int = 0,
) -> Dict:
    """
    Estimate the dominant eigenvalues of the alternating-update map's Jacobian.

    Args:
        D, G, prior: the live models. They are deep-copied and cast to float64
            internally, so the caller's models are untouched (and stay float32).
        gan_loss: a lib.gan_loss.GANLoss configured exactly as in training.
        vic_reg: the VICReg-like module applied to the batch's particles.
        regularizer: a lib.grad_regularizers.GradRegularizer. Pass a clone with
            lazy_k=1 so the penalty applies on every step of the map.
        x_real_fixed: (B, 2) fixed real batch. Both half-steps use it (the D
            step as the reals, the G step as the RpGAN reference batch).
        idx_fixed: (B,) fixed particle indices. Only the *unique* rows they
            touch enter theta: untouched rows get exactly zero gradient from a
            fixed batch and would contribute a wall of spurious eigenvalues at
            exactly 1.0, drowning the modes we care about.
        lrs: {'D': .., 'G': .., 'prior': ..} per-player step sizes. Pass the
            base (unannealed) training LRs.
        krylov_dim: number of Arnoldi steps.
        fd_eps_scale: relative finite-difference step,
            h = fd_eps_scale * (||theta|| + 1e-8) / (||v|| + 1e-8).
            LeakyReLU nets are only piecewise smooth, so a large h averages the
            Jacobian over the kinks and mildly *under*-estimates the dominant
            modulus (measured here: ~1.0003 at 3e-2 vs ~1.0016 below 1e-3 on an
            untrained model). The bias is systematic across arms, so leave this
            fixed when comparing them.
        lambda_ep: weight on the VICReg term, matching training.
        seed: seeds both the Arnoldi start vector and the RNG consumed inside
            the penalty (e_interp draws interpolation weights), so F is a
            deterministic function of theta.
        power_iters: extra power iterations run from the dominant Ritz vector
            as an independent check on the dominant modulus.
        penalty_step: the step index handed to `regularizer.penalty`, held
            fixed across every evaluation of F. It only matters for a
            regularizer whose penalty center is annealed, where the analysis
            should see the center in force at the point of the run being
            analysed; pass the end-of-training step for an end-of-training
            spectrum. With a lazy_k=1 clone and no anneal (the historical
            case) the value is irrelevant and 0 reproduces it exactly.

    Returns:
        {
          'dominant_modulus': float,     # max |Ritz value|
          'power_iter_modulus': float,   # ||Jv|| / ||v|| after power_iters
          'top_eigs': [[re, im], ...],   # up to 8 Ritz values by modulus
          'krylov_dim': int,
        }
    """
    device = x_real_fixed.device

    # Work on float64 deep copies: the finite-difference JVPs subtract two
    # nearly-equal vectors, which float32 cannot resolve at h ~ 1e-3 * ||theta||.
    D = copy.deepcopy(D).to(device=device, dtype=torch.float64)
    G = copy.deepcopy(G).to(device=device, dtype=torch.float64)
    prior = copy.deepcopy(prior).to(device=device, dtype=torch.float64)
    for m in (D, G, prior):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(True)

    x_real = x_real_fixed.detach().to(device=device, dtype=torch.float64)
    idx_fixed = idx_fixed.detach().to(device=device, dtype=torch.long)
    uidx = torch.unique(idx_fixed)

    d_params = list(D.parameters())
    g_params = list(G.parameters())
    z_param = prior.z

    lr_d = float(lrs["D"])
    lr_g = float(lrs["G"])
    lr_p = float(lrs["prior"])

    n_d = sum(p.numel() for p in d_params)
    n_g = sum(p.numel() for p in g_params)
    n_z = uidx.numel() * z_param.shape[1]
    n_theta = n_d + n_g + n_z

    def _read_theta() -> torch.Tensor:
        with torch.no_grad():
            return torch.cat(
                [
                    _flat([p.detach() for p in d_params]),
                    _flat([p.detach() for p in g_params]),
                    z_param.detach()[uidx].reshape(-1),
                ]
            )

    def _write_theta(vec: torch.Tensor) -> None:
        with torch.no_grad():
            off = _unflatten_into(vec, d_params, 0)
            off = _unflatten_into(vec, g_params, off)
            z_param[uidx] = vec[off : off + n_z].view(uidx.numel(), -1)

    theta0 = _read_theta().clone()

    # F re-seeds the global RNG on every call (see below); stash the caller's
    # stream so this analysis cannot shift a training run's randomness.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )

    # -------------------------
    #  The update map F
    # -------------------------

    def _F(theta: torch.Tensor) -> torch.Tensor:
        """One alternating round (D step, then G+prior step) as theta -> theta."""
        _write_theta(theta)
        # Any randomness inside the penalty (e_interp's interpolation weights)
        # must be identical across F evaluations or the finite differences
        # measure noise instead of curvature.
        torch.manual_seed(seed)

        # --- 1) D half-step -------------------------------------------------
        with torch.no_grad():
            x_fake = G(z_param[idx_fixed])

        loss_d = gan_loss.d_loss(D(x_real), D(x_fake))
        pen, _ = regularizer.penalty(D, x_real, x_fake, penalty_step)
        loss_d = loss_d + pen

        grads_d = torch.autograd.grad(loss_d, d_params, allow_unused=True)
        d_new = [
            p.detach() - lr_d * (g if g is not None else torch.zeros_like(p))
            for p, g in zip(d_params, grads_d)
        ]
        with torch.no_grad():
            for p, pn in zip(d_params, d_new):
                p.copy_(pn)

        # --- 2) G + prior half-step, at the POST-D-step discriminator -------
        x_fake_g = G(z_param[idx_fixed])
        loss_gan = gan_loss.g_loss(D(x_fake_g), D(x_real))
        loss_g = loss_gan + lambda_ep * vic_reg(z_param[uidx])

        grads_g = torch.autograd.grad(
            loss_g, g_params + [z_param], allow_unused=True
        )
        g_grads, z_grad = grads_g[:-1], grads_g[-1]
        g_new = [
            p.detach() - lr_g * (g if g is not None else torch.zeros_like(p))
            for p, g in zip(g_params, g_grads)
        ]
        z_rows = z_param.detach()[uidx]
        if z_grad is not None:
            z_rows = z_rows - lr_p * z_grad.detach()[uidx]

        out = torch.cat([_flat(d_new), _flat(g_new), z_rows.reshape(-1)])

        _write_theta(theta0)
        return out.detach()

    # -------------------------
    #  Finite-difference JVPs
    # -------------------------

    theta_norm = float(theta0.norm())

    def _jvp(v_np: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(v_np).to(device=device, dtype=torch.float64)
        v_norm = float(v.norm())
        h = fd_eps_scale * (theta_norm + 1e-8) / (v_norm + 1e-8)
        fp = _F(theta0 + h * v)
        fm = _F(theta0 - h * v)
        return ((fp - fm) / (2.0 * h)).cpu().numpy().astype(np.float64)

    # -------------------------
    #  Arnoldi
    # -------------------------

    rng = np.random.default_rng(seed)
    m = int(min(krylov_dim, n_theta - 1))
    if m < 2:
        raise ValueError(f"krylov_dim too small for theta of size {n_theta}")

    V = np.zeros((n_theta, m + 1), dtype=np.float64)
    H = np.zeros((m + 1, m), dtype=np.float64)

    v0 = rng.standard_normal(n_theta)
    V[:, 0] = v0 / np.linalg.norm(v0)

    k_used = m
    for j in range(m):
        w = _jvp(V[:, j])
        # Modified Gram-Schmidt, twice (re-orthogonalization keeps the basis
        # usable despite the finite-difference noise floor).
        for _ in range(2):
            for i in range(j + 1):
                c = float(V[:, i] @ w)
                H[i, j] += c
                w = w - c * V[:, i]
        beta = float(np.linalg.norm(w))
        H[j + 1, j] = beta
        if beta < 1e-14:
            k_used = j + 1
            break
        V[:, j + 1] = w / beta

    Hk = H[:k_used, :k_used]
    ritz = np.linalg.eigvals(Hk)
    order = np.argsort(-np.abs(ritz))
    ritz = ritz[order]

    dominant = float(np.abs(ritz[0]))
    top_eigs = [[float(z.real), float(z.imag)] for z in ritz[:8]]

    # -------------------------
    #  Power-iteration cross-check
    # -------------------------

    evals, evecs = np.linalg.eig(Hk)
    dom_idx = int(np.argmax(np.abs(evals)))
    y = evecs[:, dom_idx]
    v = V[:, :k_used] @ np.real(y)
    if np.linalg.norm(v) < 1e-12:
        v = V[:, :k_used] @ np.imag(y)
    if np.linalg.norm(v) < 1e-12:
        v = V[:, 0]
    v = v / np.linalg.norm(v)

    power_modulus = float("nan")
    for _ in range(max(1, int(power_iters))):
        w = _jvp(v)
        nw = float(np.linalg.norm(w))
        power_modulus = nw
        if nw < 1e-14:
            break
        v = w / nw

    _write_theta(theta0)
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)

    return {
        "dominant_modulus": dominant,
        "power_iter_modulus": power_modulus,
        "top_eigs": top_eigs,
        "krylov_dim": int(k_used),
    }
