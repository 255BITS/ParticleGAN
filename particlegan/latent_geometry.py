"""Prior-derived bounded local DV12 latent geometry for feature-cell copies."""
import weakref

import torch

from .cell_statistics import JITTER_CAP, JITTER_STD, require_positive_ints


class BoundedLatentGeometry:
    """Prior-derived anisotropic width and bounded local DV12 latent radius.

    At most ``neighbors`` candidates come from coordinate sort neighborhoods
    on the ``rank`` most varying latent coordinates. Distances use the full
    latent vector. The radius can exceed the exact nearest-other radius, so
    this is an approximate DV12 kernel, not an exact nearest-neighbor claim.
    Setup is O(rank*N*log(N)); queries use O(neighbors*latent_width) per row.
    Deterministic caches are derived, versioned and never checkpointed.
    Coordinate width is bounded by the local RMS of the nearest ``rank``
    candidates divided by sqrt(2), the pair-difference variance factor. This
    prevents an isotropic norm ball from ignoring narrow local coordinates.
    """
    def __init__(self, *, rank=8, neighbors=64, chunk=256, lineage=None):
        require_positive_ints(rank=rank, neighbors=neighbors, chunk=chunk)
        self.rank, self.neighbors, self.chunk = rank, neighbors, chunk
        self.lineage = lineage
        self.clear()

    def clear(self):
        self._entries = {}
        self.work = dict(builds=0, distance_pairs=0, max_query_rows=0, max_candidates=0,
                         lineage_distance_pairs=0)

    @torch.no_grad()
    def _orders(self, points):
        key = id(points)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is points and entry[1] == points._version:
            return entry[2]
        spread = points.detach().std(0, unbiased=False)
        count = min(self.rank, int((spread > 0).sum()))
        # Cache Python indices after one batched transfer. A CUDA scalar axis
        # would otherwise synchronize every block[:, axis] lookup per chunk.
        axes = spread.argsort(descending=True, stable=True)[:count].cpu().tolist()
        orders = []
        for axis in axes:
            values, rows = points.detach()[:, axis].sort(stable=True)
            unique, counts = torch.unique_consecutive(values, return_counts=True)
            # Skip entire tied-coordinate groups. This avoids losing every
            # candidate when many row copies have exactly identical centers.
            first = counts.cumsum(0)-counts
            orders.append((axis, unique, rows[first]))
        if len(self._entries) >= 2:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = (weakref.ref(points), points._version, orders)
        self.work["builds"] += 1
        return orders

    @torch.no_grad()
    def radius(self, latent, prior, *, rows=None):
        return self._local_geometry(latent, prior, rows=rows)[0]

    @torch.no_grad()
    def _local_geometry(self, latent, prior, *, rows=None):
        points = prior.z
        if self.lineage is not None and rows is not None:
            self.lineage._rows(rows)
            if len(rows) != len(latent) or len(points) != self.lineage.n:
                raise ValueError("latent rows and lineage table must match")
        orders = self._orders(points)
        if not orders:
            return latent.new_zeros(len(latent)), torch.zeros_like(latent)
        output, widths = [], []
        for start in range(0, len(latent), self.chunk):
            block = latent.detach()[start:start+self.chunk]
            distances, point_ids = [], []
            for index, (axis, values, axis_rows) in enumerate(orders):
                count = self.neighbors//len(orders)+(index < self.neighbors % len(orders))
                if not count:
                    continue
                offsets = torch.arange(-(count//2), count-count//2, device=block.device)
                positions = torch.searchsorted(values, block[:, axis].contiguous())[:, None]+offsets
                valid = (positions >= 0) & (positions < len(values))
                candidates = axis_rows[positions.clamp(0, len(values)-1)]
                distance = (block[:, None]-points.detach()[candidates]).square().sum(-1)
                distance.masked_fill_(~valid | (distance == 0), float("inf"))
                distances.append(distance)
                point_ids.append(candidates)
                self.work["distance_pairs"] += len(block)*count
            linked = 0
            if self.lineage is not None and rows is not None:
                candidates = self.lineage.neighbors[rows[start:start+len(block)]]
                distance = (block[:, None]-points.detach()[candidates.clamp_min(0)]).square().sum(-1)
                distance.masked_fill_((candidates < 0) | (distance == 0), float("inf"))
                distances.append(distance)
                point_ids.append(candidates.clamp_min(0))
                linked = self.lineage.degree
                self.work["distance_pairs"] += len(block)*linked
                self.work["lineage_distance_pairs"] += len(block)*linked
            self.work["max_query_rows"] = max(self.work["max_query_rows"], len(block))
            self.work["max_candidates"] = max(self.work["max_candidates"], self.neighbors+linked)
            distances, point_ids = torch.cat(distances, 1), torch.cat(point_ids, 1)
            nearest = distances.min(1).values.sqrt()
            output.append(torch.where(torch.isfinite(nearest), .5*nearest,
                                      torch.zeros_like(nearest)))
            selected = distances.argsort(dim=1, stable=True)[:, :min(self.rank, distances.shape[1])]
            valid = torch.isfinite(distances.gather(1, selected))
            neighbors = points.detach()[point_ids.gather(1, selected)]
            difference2 = (block[:, None]-neighbors).square()*valid[:, :, None]
            # A difference between independent local draws has twice their
            # coordinate variance. Reuse rank as the bounded local sample size.
            widths.append((difference2.sum(1)/(2*valid.sum(1).clamp_min(1)[:, None])).sqrt())
        return torch.cat(output), torch.cat(widths)

    def displacement(self, latent, prior, bandwidth, noise, *, record=None, rows=None):
        radius, local_width = self._local_geometry(latent, prior, rows=rows)
        global_width = torch.as_tensor(bandwidth, device=latent.device, dtype=latent.dtype)
        applied_width = torch.minimum(global_width, local_width)
        delta = applied_width*noise
        fraction = (radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
        delta = delta*fraction[:, None]
        if record is not None:
            kernel = "bounded_local_dv12_lineage" if self.lineage else "bounded_local_dv12"
            record.append(dict(radius_min=float(radius.min()), radius_mean=float(radius.mean()),
                               radius_max=float(radius.max()),
                               perturbation_rms=float(delta.detach().square().mean().sqrt()),
                               clipped_fraction=float((fraction < 1.).float().mean()),
                               global_bandwidth_mean=float(global_width.mean()),
                               local_bandwidth_mean=float(applied_width.mean()),
                               backend="feature_cells", kernel=kernel))
            del record[:-2]
        return delta


def bounded_jitter(latent, noise):
    """Historical fixed kernel, retained for explicit comparison diagnostics."""
    delta = JITTER_STD * noise
    fraction = (JITTER_CAP / delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1.)
    return delta * fraction[:, None]
