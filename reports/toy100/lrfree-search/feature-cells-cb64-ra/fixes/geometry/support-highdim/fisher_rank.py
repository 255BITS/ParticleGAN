"""Real-only bounded Fisher geometry with an explicit numerical dictionary rank.

This is a CPU diagnostic implementation.  It imports no fixtures or labels.
Cell assignment remains the original snapshot's assignment.
"""
import torch


@torch.no_grad()
def fit_score(snapshot, reference_features, *, source_dtype=None):
    source_dtype = reference_features.dtype if source_dtype is None else source_dtype
    if not source_dtype.is_floating_point:
        raise ValueError("source_dtype must be a floating dtype")
    x = (reference_features.double()-snapshot.mean)/snapshot.scale
    ids, _ = snapshot.assign(reference_features)
    counts = snapshot.reference_counts
    weights = counts.double()/len(x)
    grand = x.mean(0)
    means = torch.stack([x[ids == c].mean(0) if bool((ids == c).any()) else grand
                         for c in range(snapshot.cells)])
    # This SVD is H by K, where K <= 64.  A reduced QR alone completes
    # rank-deficient cell-mean directions with arbitrary feature directions.
    u, singular, _ = torch.linalg.svd((means-grand).T, full_matrices=False)
    tol = singular.max()*max(means.shape)*torch.finfo(source_dtype).eps
    dictionary_rank = int((singular > tol).sum())
    dictionary = u[:, :dictionary_rank]
    if dictionary_rank:
        y = x@dictionary
        centers = means@dictionary
        residual = y-centers[ids]
        within = residual.T@residual/len(x)
        delta = centers-grand@dictionary
        between = (delta*weights[:, None]).T@delta
        eig, vec = torch.linalg.eigh(within)
        eps = torch.finfo(source_dtype).eps
        # All-zero within covariance occurs when each reference row owns a
        # cell.  Epsilon times the smallest representable variance underflows
        # after whitening.  Bound roundoff from the observed between-cell
        # variance as well; no absolute unit or outcome threshold is used.
        covariance_scale = torch.maximum(eig.max().clamp_min(0),
                                          eps*between.diagonal().sum().clamp_min(0))
        floor = eps*dictionary_rank*covariance_scale
        if not bool(torch.isfinite(floor)) or not bool(floor > 0):
            raise ValueError("non-finite or unidentifiable support covariance")
        whitening = vec/eig.clamp_min(floor).sqrt()[None]
        separation = whitening.T@between@whitening
        signal, basis = torch.linalg.eigh((separation+separation.T)/2)
        signal_tol = signal.max().clamp_min(0)*dictionary_rank*torch.finfo(source_dtype).eps
        rank = min(snapshot.requested_rank, int((signal > signal_tol).sum()))
        transform = dictionary@whitening@basis[:, -rank:] if rank else dictionary[:, :0]
    else:
        eig = signal = x.new_empty(0)
        floor = x.new_zeros(())
        within = x.new_empty((0, 0))
        rank = 0
        transform = dictionary
    z = x@transform
    zcenters = means@transform
    distance = (z-zcenters[ids]).square().sum(1)
    sse = torch.stack([distance[ids == c].sum() for c in range(snapshot.cells)])
    radius = (sse/counts.clamp_min(1)).sqrt()
    positive = radius[(counts > 0) & (radius > 0)]
    global_radius = positive.median() if len(positive) else x.new_tensor(1.)
    scale2 = (sse+4*global_radius.square())/(counts+4.)
    scale2 = scale2.clamp_min(global_radius.square()*1e-6)
    scale2 *= 1+counts.clamp_min(1).double().reciprocal()

    def score(features):
        snapshot._input(features)
        query = ((features.double()-snapshot.mean)/snapshot.scale)@transform
        if not len(query):
            return query.new_empty(0)
        return torch.cat([((b[:, None]-zcenters[None]).square().sum(2)/scale2[None]).min(1).values.sqrt()
                          for b in query.split(snapshot.chunk)])

    metadata = dict(dictionary_rank=dictionary_rank, dictionary_singular_values=singular.tolist(),
                    dictionary_tolerance=float(tol), source_dtype=str(source_dtype), rank=rank,
                    within_eigenvalues=eig.tolist(), within_floor=float(floor),
                    within_floor_rule="eps*D*max(lambda_max_within,eps*trace_between)",
                    generalized_eigenvalues=signal.tolist(),
                    transform_bytes=transform.untyped_storage().nbytes(),
                    covariance_bytes=within.untyped_storage().nbytes(),
                    center_bytes=zcenters.untyped_storage().nbytes())
    return score, metadata
