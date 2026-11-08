"""Bounded symmetric latent copy lineage shared by the live and EMA tables."""
import torch

from .cell_statistics import PARENT_RESERVOIR


class LatentLineage:
    """Bounded symmetric row-copy links; semantic state shared by live/EMA.

    Overwriting a row removes all its old incident edges. New parent/child
    pairs are inserted newest first. A full endpoint evicts its oldest edge
    symmetrically. Links refer to current row incarnations, not old positions.
    """
    def __init__(self, n, degree, device):
        if (type(n) is not int or n <= 0 or type(degree) is not int
                or not 0 < degree <= min(PARENT_RESERVOIR, n-1)):
            raise ValueError("invalid latent lineage size/degree")
        self.n, self.degree = n, degree
        self.neighbors = torch.full((n, degree), -1, dtype=torch.long, device=device)
        self.work = dict(updates=0, copied_rows=0, reciprocal_slots=0,
                         maximum_reciprocal_slots=0, maximum_parent_rows=0)

    def _rows(self, rows):
        if (not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.dtype != torch.long
                or rows.device != self.neighbors.device
                or bool(((rows < 0) | (rows >= self.n)).any())):
            raise ValueError("invalid latent lineage row indices")

    def validate(self, neighbors):
        if (not isinstance(neighbors, torch.Tensor) or neighbors.shape != self.neighbors.shape
                or neighbors.dtype != torch.long
                or bool(((neighbors < -1) | (neighbors >= self.n)).any())):
            raise ValueError("invalid latent lineage checkpoint shape/indices")
        valid = neighbors >= 0
        rows, slots = valid.nonzero(as_tuple=True)
        partners = neighbors[rows, slots]
        if bool((rows == partners).any()):
            raise ValueError("latent lineage cannot contain self edges")
        ordered = neighbors.masked_fill(~valid, self.n).sort(1).values
        if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < self.n)).any()):
            raise ValueError("latent lineage contains repeated neighbors")
        if len(rows) and not bool((neighbors[partners] == rows[:, None]).any(1).all()):
            raise ValueError("latent lineage must be symmetric")

    def _compact(self, rows):
        old = self.neighbors[rows]
        slots = torch.arange(self.degree, device=old.device)[None].expand_as(old)
        order = torch.where(old >= 0, slots, slots+self.degree).argsort(dim=1, stable=True)
        return old.gather(1, order)

    @torch.no_grad()
    def _remove_reciprocals(self, sources, adjacent):
        # Only known neighbors of touched endpoints are read. Padding uses a
        # masked row0 read; it never writes row0 merely because a slot is empty.
        targets = adjacent.clamp_min(0)
        remove = (self.neighbors[targets] == sources[:, None, None]) & (adjacent >= 0)[:, :, None]
        slots = torch.arange(self.degree, device=adjacent.device)
        flat = targets[:, :, None]*self.degree+slots
        self.neighbors.view(-1)[flat[remove]] = -1
        amount = len(sources)*self.degree*self.degree
        self.work["reciprocal_slots"] += amount
        self.work["maximum_reciprocal_slots"] = max(self.work["maximum_reciprocal_slots"], amount)

    @torch.no_grad()
    def register_copies(self, children, parents):
        self._rows(children)
        self._rows(parents)
        if children.shape != parents.shape or len(torch.unique(children)) != len(children):
            raise ValueError("lineage copies require distinct paired child rows")
        if not len(children):
            return
        self.work["updates"] += 1
        self.work["copied_rows"] += len(children)
        self._remove_reciprocals(children, self.neighbors[children].clone())
        self.neighbors[children] = -1
        # A parent overwritten in this same simultaneous batch refers to its
        # previous incarnation, so no edge to its current row is registered.
        ordered_children = children.sort().values
        positions = torch.searchsorted(ordered_children, parents)
        last = max(0, len(children)-1)
        keep = ((positions == len(children))
                | (ordered_children[positions.clamp_max(last)] != parents))
        children, parents = children[keep], parents[keep]
        if not len(children):
            return
        # Group repeated manual-copy parents in one batch. Later entries in a
        # simultaneous copy batch are newest; keep at most degree children per
        # parent. Internal reaction plans already use distinct parents.
        order = parents.argsort(stable=True)
        unique, counts = torch.unique_consecutive(parents[order], return_counts=True)
        ends = counts.cumsum(0)
        slots = torch.arange(self.degree, device=parents.device)[None]
        locations = ends[:, None]-1-slots
        present = locations >= (ends-counts)[:, None]
        incoming = children[order[locations.clamp_min(0)]].masked_fill(~present, -1)
        room = self.degree-counts.clamp_max(self.degree)
        old = self._compact(unique)
        evicted = old.masked_fill(slots < room[:, None], -1)
        # Truncate both endpoints before re-reading updated rows. This prevents
        # two touched parents from reintroducing an edge one of them evicted.
        self.neighbors[unique] = old.masked_fill(slots >= room[:, None], -1)
        self._remove_reciprocals(unique, evicted)
        old = self._compact(unique)
        used = counts.clamp_max(self.degree)
        old_slots = (slots-used[:, None]).clamp_min(0)
        self.neighbors[unique] = torch.where(slots < used[:, None], incoming,
                                             old.gather(1, old_slots))
        selected = incoming >= 0
        selected_children = incoming[selected]
        selected_parents = unique[:, None].expand_as(incoming)[selected]
        self.neighbors[selected_children, 0] = selected_parents
        self.work["maximum_parent_rows"] = max(self.work["maximum_parent_rows"], len(unique))
