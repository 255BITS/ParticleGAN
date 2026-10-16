"""One exact-capacity swap selected with the aggregate training objective."""
import torch


@torch.no_grad()
def best_single_swap(mean, covariance, reference, assignment):
    from . import aggregate_loss
    cells, dimension = reference['means'].shape
    n = len(mean)
    info = dict(swap_candidates=0, swap_accepted=False, swap_loss_decrease=0.0)
    if cells < 2:
        return assignment, info
    # Double arithmetic improves proposal ranking; acceptance is recomputed
    # with the exact original centered-covariance implementation and dtype.
    m, cov = mean.detach().double(), covariance.detach().double()
    count = torch.bincount(assignment, minlength=cells).double()
    center = m.new_zeros(cells, dimension).index_add_(0, assignment, m) / count[:, None]
    second = cov + m[:, :, None] * m[:, None, :]
    second_mean = m.new_zeros(cells, dimension, dimension).index_add_(0, assignment, second) / count[:, None, None]
    eye = torch.eye(dimension, device=m.device, dtype=m.dtype)

    def weighted_terms(centers, moments, labels):
        covariance_ = moments - centers[:, :, None] * centers[:, None, :]
        covariance_ = (covariance_ + covariance_.transpose(-1, -2)) / 2
        covariance_ = covariance_ + reference['jitter'].double() * eye
        sign, logdet = torch.linalg.slogdet(covariance_)
        delta = centers - reference['means'].double()[labels]
        inv = reference['inverse'].double()[labels]
        value = (torch.einsum('bi,bij,bj->b', delta, inv, delta)
                 + torch.einsum('bij,bji->b', inv, covariance_)
                 - logdet + reference['logdet'].double()[labels] - dimension) / dimension
        return torch.where(sign > 0, value * count[labels] / n, torch.full_like(value, float('inf')))

    labels = torch.arange(cells, device=m.device)
    base_terms = weighted_terms(center, second_mean, labels)
    i, j = torch.triu_indices(n, n, 1, device=m.device)
    keep = assignment[i] != assignment[j]
    i, j = i[keep], j[keep]
    a, b = assignment[i], assignment[j]
    info['swap_candidates'] = len(i)
    if len(i) == 0:
        return assignment, info
    delta_m, delta_s = m[j] - m[i], second[j] - second[i]
    ca = center[a] + delta_m / count[a, None]
    cb = center[b] - delta_m / count[b, None]
    sa = second_mean[a] + delta_s / count[a, None, None]
    sb = second_mean[b] - delta_s / count[b, None, None]
    change = weighted_terms(ca, sa, a) + weighted_terms(cb, sb, b) - base_terms[a] - base_terms[b]
    # triu order plus first argmin defines lexicographic tie-breaking.
    k = int(torch.argmin(change))
    if not bool(torch.isfinite(change[k])) or float(change[k]) >= 0:
        return assignment, info
    proposal = assignment.clone()
    proposal[i[k]], proposal[j[k]] = assignment[j[k]], assignment[i[k]]
    old_loss = aggregate_loss(mean, covariance, reference, assignment, 'aggregate_forward_kl')[0]
    new_loss = aggregate_loss(mean, covariance, reference, proposal, 'aggregate_forward_kl')[0]
    if bool(torch.isfinite(new_loss)) and float(new_loss) < float(old_loss):
        assert torch.equal(torch.bincount(proposal, minlength=cells), torch.bincount(assignment, minlength=cells))
        info.update(swap_accepted=True, swap_loss_decrease=float(old_loss-new_loss))
        return proposal, info
    return assignment, info
