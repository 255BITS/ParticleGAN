"""Persistent data-only support groups for a fixed, separated target.

The first bank uses the unchanged MST grouping. Later inferred group centers
match cached centers inside half the smallest cached inter-center distance.
Matched groups add their samples to exact running sums; unmatched groups are
new; groups absent from one minibatch remain cached. This has no target labels,
configured group count, generator state, optimizer schedule, or decay clock.

The half-separation premise is observable but not certified for every future
bank. True disappearance cannot be distinguished from a long sampling gap by
this rule. A production adapter must serialize sums/counts with learner state.
"""

import torch

from reports.toy100.sample_group_anchor import mst_groups


class PersistentGroupMemory:
    def __init__(self):
        self.sums = []
        self.counts = []

    def state_dict(self):
        return dict(sums=[value.clone() for value in self.sums], counts=list(self.counts))

    def load_state_dict(self, value):
        sums, counts = value["sums"], value["counts"]
        if (len(sums) != len(counts) or any(type(c) is not int or c <= 0 for c in counts)
                or any(v.shape != (2,) or not bool(torch.isfinite(v).all()) for v in sums)):
            raise ValueError("invalid persistent group state")
        self.sums = [v.detach().double().clone() for v in sums]
        self.counts = [int(c) for c in counts]

    def centers(self):
        if not self.sums:
            raise ValueError("support memory has not observed a real minibatch")
        return torch.stack([total / count for total, count in zip(self.sums, self.counts)])

    @torch.no_grad()
    def observe(self, real):
        if real.ndim != 2 or real.shape[1] != 2 or not bool(torch.isfinite(real).all()):
            raise ValueError("finite two-dimensional real minibatch required")
        current, grouping = mst_groups(real)
        original = self.centers() if self.sums else None
        if original is None:
            radius = None
        elif len(original) > 1:
            distances = torch.cdist(original, original)
            distances.fill_diagonal_(float("inf"))
            radius = float(distances.min()) / 2.
        else:
            # With only one cached group, the new bank's MST gap supplies the
            # only observed separator. This bootstrap is not universally safe.
            radius = float(grouping["cut_threshold"])

        matched = []
        added = []
        for index, members in enumerate(grouping["member_indices"]):
            batch_sum = real[members].double().sum(dim=0)
            if original is not None:
                distances = torch.linalg.vector_norm(original-current[index], dim=1)
                nearest = int(distances.argmin())
                if float(distances[nearest]) < radius:
                    self.sums[nearest] += batch_sum
                    self.counts[nearest] += len(members)
                    matched.append(dict(current=index, remembered=nearest,
                                        distance=float(distances[nearest])))
                    continue
            added.append(dict(current=index, remembered=len(self.sums)))
            self.sums.append(batch_sum)
            self.counts.append(len(members))
        unmatched = (list(range(len(original))) if original is not None else [])
        unmatched = [j for j in unmatched if j not in {row["remembered"] for row in matched}]
        return self.centers(), dict(current_groups=len(current), remembered_groups=len(self.sums),
            matching_radius=radius, matched=matched, added=added,
            absent_remembered=unmatched, counts=list(self.counts),
            current_largest_mst_gap=grouping["largest_additive_gap"])
