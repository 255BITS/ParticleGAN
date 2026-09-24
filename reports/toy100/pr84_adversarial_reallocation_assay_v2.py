"""Repair V1's selection-baseline shape; preserve its frozen failed bytes.

The host critic returns (B,), while the analytic toy returned (B,1). V1
reshaped only real logits for its baseline and therefore accidentally used
an all-pairs loss on the host. Enumeration and after/heldout evaluation were
already paired. This source epoch fixes the baseline and guards its exact
native equality. No proposal, objective, state, budget or stencil changes.
"""
import torch
import torch.nn.functional as F

from reports.toy100 import pr84_adversarial_reallocation_assay as base

METHOD = base.METHOD + '_v2_paired_baseline'


@torch.no_grad()
def select(critic, fake, batch, candidates, particles, gan):
    if gan.mode != 'rp' or gan.loss_type != 'logistic':
        raise ValueError('the separable enumeration is only the declared Rp logistic G loss')
    if not all(torch.isfinite(x).all() for x in (fake, batch['real'], candidates)):
        raise FloatingPointError('nonfinite selection input')
    real_logits = base.fit.smooth(critic, batch['real']).reshape(-1)
    fake_logits = base.fit.smooth(critic, fake).reshape(-1)
    original_terms = F.softplus(real_logits - fake_logits)
    before = gan.g_loss(fake_logits, real_logits)
    if float(before) != float(base.loss(critic, fake, batch['real'], gan)):
        raise RuntimeError('paired selection baseline differs from the native G objective')
    scores = torch.empty((particles, len(candidates)), dtype=torch.float64)
    for donor in range(particles):
        mask = batch['indices'] == donor
        if not bool(mask.any()):
            scores[donor].fill_(float(before))
            continue
        points = candidates[:, None, :] + batch['sigma'] * batch['noise'][mask][None, :, :]
        logits = base.fit.smooth(critic, points.reshape(-1, points.shape[-1]))
        logits = logits.reshape(len(candidates), -1)
        terms = F.softplus(real_logits[mask][None, :] - logits)
        scores[donor] = (original_terms[~mask].double().sum() + terms.double().sum(1)) / len(fake)
    chosen = int(scores.argmin())
    donor, candidate = divmod(chosen, len(candidates))
    target = candidates[candidate].detach().clone()
    trial = base.replace_fake(fake, batch, donor, target)
    actual = base.loss(critic, trial, batch['real'], gan)
    accepted = bool(torch.isfinite(actual) and actual < before)
    return dict(donor=donor, candidate=candidate, target=target.tolist(),
                selected=accepted, native_loss_before=float(before),
                native_loss_after=float(actual), enumerated_minimum=float(scores.min()),
                enumeration_full_loss_error=abs(float(actual)-float(scores.min())),
                proposals=particles*len(candidates), paired_baseline_exact=True), target, scores


def main():
    original = base.select, base.METHOD, base.SOURCES
    try:
        base.select = select
        base.METHOD = METHOD
        base.SOURCES += ('reports/toy100/pr84_adversarial_reallocation_assay_v2.py',
                         'tests/test_pr84_adversarial_reallocation_assay_v2.py')
        base.main()
    finally:
        base.select, base.METHOD, base.SOURCES = original


if __name__ == '__main__':
    main()
