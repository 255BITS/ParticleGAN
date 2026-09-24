"""Critic-value spare-particle step. Not a GAN optimizer and not a rest gate.

The missing ring mode is the critic's highest value while the local generator
gradient points away. This rule reads that value gap on the current real
minibatch. When the gap exceeds twice the real-score spread, one spare
particle takes an output step of at most 0.1 toward the softmax centroid of
the high-scoring reals. A particle is eligible when a neighbor lies inside
half its target distance, or when it is already clearly closer to the centroid
than every other particle. No mode centers, no clock, no slope threshold.
"""
import torch


TRUST = 0.1
GAP_SPREADS = 2.0


def select_particle(points, real, fake_scores, real_scores, *, trust=TRUST):
    if points.ndim != 2 or real.ndim != 2 or points.shape[1] != real.shape[1]:
        raise ValueError("expected matching output-space matrices")
    fake_scores = fake_scores.reshape(-1)
    real_scores = real_scores.reshape(-1)
    if fake_scores.shape[0] != points.shape[0] or real_scores.shape[0] != real.shape[0]:
        raise ValueError("score rows must match points")
    gap = real_scores.max() - fake_scores.mean()
    spread = real_scores.std(unbiased=False).clamp_min(1e-6)
    info = dict(fired=False, gap=float(gap), spread=float(spread), trust=float(trust))
    if not torch.isfinite(gap) or float(gap) <= GAP_SPREADS * float(spread):
        return None, info
    cutoff = fake_scores.mean() + GAP_SPREADS * spread
    preferred = real[real_scores >= cutoff]
    scores = real_scores[real_scores >= cutoff]
    if preferred.shape[0] == 0:
        return None, info
    weights = torch.softmax(scores, 0)
    target = (weights.unsqueeze(1) * preferred).sum(0)
    distance = torch.cdist(points, target.view(1, -1)).flatten()
    order = torch.argsort(distance)
    index = int(order[0])
    d1 = distance[index]
    d2 = distance[order[1]] if points.shape[0] > 1 else d1
    others = torch.cdist(points[index:index + 1], points).flatten()
    others[index] = float("inf")
    twin = bool(points.shape[0] > 1 and float(others.min()) < 0.5 * float(d1))
    ahead = bool(points.shape[0] > 1 and float(d1) < 0.75 * float(d2))
    info.update(n_preferred=int(preferred.shape[0]), target=target.detach().cpu().tolist(),
                nearest=index, twin=twin, ahead=ahead, d1=float(d1), d2=float(d2))
    if not twin and not ahead:
        info["blocked"] = True
        return None, info
    info["fired"] = True
    return (index, target.detach()), info


def spare_particle_pull(clean, prior_z, real, fake_scores, real_scores, metric, *, trust=TRUST):
    """Move one prior row through its own Jacobian, or leave z unchanged.

    The step is the minimum Adam-metric solution of an output displacement of
    length at most ``trust``. It is kept only when that particle's nonlinear
    output is strictly closer to the centroid and every other particle stays
    put. Network weights, Adam moments and the global RNG are not touched.
    """
    if metric.shape != prior_z.shape or not torch.isfinite(metric).all() or not bool((metric > 0).all()):
        raise FloatingPointError("invalid prior Adam metric")
    with torch.no_grad():
        base_z = prior_z.detach().clone()
        base_x = clean(base_z).detach()
    index_target, info = select_particle(base_x, real.detach(), fake_scores.detach(),
                                         real_scores.detach(), trust=trust)
    info.update(accepted=False, alpha=0.0, latent_norm=0.0,
                output_error_before=None, output_error_after=None)
    if index_target is None:
        return info
    index, target = index_target
    residual = target - base_x[index]
    error = residual.norm().clamp_min(1e-12)
    info["output_error_before"] = float(error)
    step = residual * min(1.0, float(trust) / float(error))
    one_row = lambda z: clean(z.unsqueeze(0)).squeeze(0)
    with torch.enable_grad():
        jacobian = torch.func.jacfwd(one_row)(base_z[index].detach()).detach()
    if jacobian.shape != (base_x.shape[1], prior_z.shape[1]) or not torch.isfinite(jacobian).all():
        raise FloatingPointError("invalid particle Jacobian")
    j = jacobian.double()
    p = metric[index].double()
    gram = j * p.unsqueeze(0) @ j.transpose(0, 1)
    correction = p * (j.transpose(0, 1) @ torch.linalg.pinv(gram, rtol=1e-6, atol=0.) @ step.double())
    correction = correction.to(prior_z.dtype)
    if not torch.isfinite(correction).all():
        raise FloatingPointError("nonfinite latent correction")
    rng = torch.get_rng_state()
    accepted_alpha = 0.0
    for alpha in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
        with torch.no_grad():
            prior_z.copy_(base_z)
            prior_z[index] = base_z[index] + correction * alpha
            after = clean(prior_z).detach()
        same = torch.allclose(after[torch.arange(len(after)) != index],
                              base_x[torch.arange(len(base_x)) != index], atol=1e-6, rtol=1e-5)
        closer = float((after[index] - target).norm()) < float(error) - 1e-8
        if same and closer and torch.isfinite(after).all():
            accepted_alpha = alpha
            info["output_error_after"] = float((after[index] - target).norm())
            break
    if accepted_alpha == 0.0:
        with torch.no_grad():
            prior_z.copy_(base_z)
        info["output_error_after"] = info["output_error_before"]
    else:
        info.update(accepted=True, alpha=accepted_alpha,
                    latent_norm=float((correction * accepted_alpha).norm()))
    if not torch.equal(torch.get_rng_state(), rng):
        raise RuntimeError("particle pull consumed global RNG")
    return info
