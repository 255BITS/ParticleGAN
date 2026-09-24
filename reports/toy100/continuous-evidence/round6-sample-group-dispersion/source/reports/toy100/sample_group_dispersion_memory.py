"""Data-only group memory matched by observed within-group RMS overlap.

This is a geometric association rule, not a statistical confidence sequence.
It assumes groups inferred within each bank are meaningful. There is no
configured group count, target geometry, model-output test, or decay clock.
"""

import math

import torch

from reports.toy100.sample_group_anchor import mst_groups


class DispersionGroupMemory:
    def __init__(self):
        self.sums = []
        self.counts = []
        self.squared_norm_sums = []

    def state_dict(self):
        return dict(sums=[x.clone() for x in self.sums],
                    counts=list(self.counts),
                    squared_norm_sums=[x.clone() for x in self.squared_norm_sums])

    def load_state_dict(self, state):
        if set(state) != {"sums", "counts", "squared_norm_sums"}:
            raise ValueError("group memory requires all three sufficient-statistic arrays")
        sums, counts, sq = state["sums"], state["counts"], state["squared_norm_sums"]
        if not (len(sums) == len(counts) == len(sq)):
            raise ValueError("group memory lengths differ")
        for total, count, square in zip(sums, counts, sq):
            if (not isinstance(total, torch.Tensor) or total.shape != (2,)
                    or total.dtype != torch.float64 or not bool(torch.isfinite(total).all())
                    or type(count) is not int or count <= 0
                    or not isinstance(square, torch.Tensor) or square.shape != ()
                    or square.dtype != torch.float64 or not bool(torch.isfinite(square))):
                raise ValueError("group memory has invalid sufficient statistics")
            lower = float(total.square().sum()) / count
            if float(square) + 1e-8 * max(1., lower) < lower:
                raise ValueError("group memory violates nonnegative empirical variance")
        self.sums = [x.detach().clone() for x in sums]
        self.counts = list(counts)
        self.squared_norm_sums = [x.detach().clone() for x in sq]

    def centers(self):
        if not self.sums:
            raise ValueError("group memory has no real observations")
        return torch.stack([total / n for total, n in zip(self.sums, self.counts)])

    def radii(self):
        if not self.sums:
            raise ValueError("group memory has no real observations")
        radii = []
        for total, n, square in zip(self.sums, self.counts, self.squared_norm_sums):
            variance = float(square) / n - float((total / n).square().sum())
            radii.append(math.sqrt(max(0., variance)))
        return radii

    @torch.no_grad()
    def observe(self, real):
        if real.ndim != 2 or real.shape[1] != 2 or not bool(torch.isfinite(real).all()):
            raise ValueError("finite two-dimensional real minibatch required")
        current, grouping = mst_groups(real)
        cached = self.centers() if self.sums else None
        cached_radii = self.radii() if self.sums else None
        half_separation = None
        if cached is not None and len(cached) > 1:
            distances = torch.cdist(cached, cached)
            distances.fill_diagonal_(float("inf"))
            half_separation = float(distances.min()) / 2.

        matched, added = [], []
        for j, members in enumerate(grouping["member_indices"]):
            values = real[members].double()
            total = values.sum(dim=0)
            count = len(members)
            square = values.square().sum()
            current_radius = math.sqrt(max(0., float(square) / count
                                            - float((total / count).square().sum())))
            if cached is not None:
                distances = torch.linalg.vector_norm(cached-current[j], dim=1)
                i = int(distances.argmin())
                overlap_radius = cached_radii[i] + current_radius
                limit = (overlap_radius if half_separation is None
                         else min(overlap_radius, half_separation))
                if float(distances[i]) <= limit:
                    self.sums[i] += total
                    self.counts[i] += count
                    self.squared_norm_sums[i] += square
                    matched.append(dict(current=j, remembered=i,
                        distance=float(distances[i]), current_rms=current_radius,
                        cached_rms=cached_radii[i], acceptance_radius=limit))
                    continue
            else:
                limit = None
            added.append(dict(current=j, remembered=len(self.sums),
                current_rms=current_radius, nearest_cached_distance=(float(distances[i]) if cached is not None else None),
                acceptance_radius=limit))
            self.sums.append(total)
            self.counts.append(count)
            self.squared_norm_sums.append(square)
        unmatched = (list(range(len(cached))) if cached is not None else [])
        unmatched = [i for i in unmatched if i not in {row["remembered"] for row in matched}]
        return self.centers(), dict(current_groups=len(current), remembered_groups=len(self.sums),
            half_cached_separation=half_separation, matched=matched, added=added,
            absent_remembered=unmatched, counts=list(self.counts),
            radii=self.radii(), current_largest_mst_gap=grouping["largest_additive_gap"])
