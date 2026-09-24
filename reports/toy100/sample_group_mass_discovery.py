"""Pure-data, held-out mass certificate for one newly rejected support region.

The discovery bank selects a fixed ball outside all existing fixed identity
balls. Only subsequently observed, independent real banks contribute to its
mass estimate. One candidate per absolute proposal ID can use alpha_j =
delta/[j(j+1)]; the per-bank confidence spending is alpha_j/[b(b+1)].
Thus the one-sided Hoeffding/union bound controls false admissions over
countably many adaptively proposed regions and validation-bank checkpoints.

This certifies region probability above one equal particle's mass, not that
the region is a true mixture component or that a neural update can realize it.
"""

from dataclasses import dataclass
import hashlib
import math

import torch

from reports.toy100.sample_group_anchor import mst_groups


DELTA = 0.01


def _real(real):
    if (not isinstance(real, torch.Tensor) or real.ndim != 2 or real.shape[1] != 2
            or real.dtype != torch.float32 or real.device.type != 'cpu'
            or len(real) == 0 or not bool(torch.isfinite(real).all())):
        raise ValueError('expected finite CPU float32 real points in R2')
    return real.detach().contiguous()


def _sha(real):
    return hashlib.sha256(_real(real).numpy().tobytes()).hexdigest()


def _references(references, old_radius):
    if (not isinstance(references, torch.Tensor) or references.ndim != 2
            or references.shape[1] != 2 or references.dtype != torch.float64
            or references.device.type != 'cpu' or len(references) == 0
            or not bool(torch.isfinite(references).all())):
        raise ValueError('expected nonempty frozen float64 identity centers')
    if type(old_radius) is not float or not math.isfinite(old_radius) or old_radius <= 0:
        raise ValueError('expected fixed positive identity radius')
    return references.detach().clone()


@dataclass(frozen=True)
class Region:
    center: tuple[float, float]
    radius: float
    proposal_bank_id: int
    proposal_bank_sha256: str
    reference_sha256: str
    rejected_count: int
    selected_group_count: int

    def __post_init__(self):
        if (type(self.center) is not tuple or len(self.center) != 2
                or any(type(value) is not float or not math.isfinite(value)
                       for value in self.center)
                or type(self.radius) is not float or not math.isfinite(self.radius)
                or self.radius <= 0 or type(self.proposal_bank_id) is not int
                or self.proposal_bank_id < 1
                or any(type(value) is not str or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value)
                       for value in (self.proposal_bank_sha256, self.reference_sha256))
                or type(self.rejected_count) is not int or self.rejected_count < 1
                or type(self.selected_group_count) is not int
                or not 1 <= self.selected_group_count <= self.rejected_count):
            raise ValueError('invalid frozen discovery-region receipt')

    def contains(self, real):
        value = _real(real).double()
        center = torch.tensor(self.center, dtype=torch.float64)
        return (value - center).square().sum(dim=1) <= self.radius ** 2


def propose_region(real, references, old_radius, *, bank_id, reserved=()):
    """Freeze one nonoverlapping candidate ball from one rejected real bank.

    The largest deterministic MST component of rejected points supplies its
    center and empirical maximum radius. The ball is capped at half of its
    clearance from existing identity balls and any caller-supplied pending or
    admitted balls. No old reference, target label, or validation draw is
    used to select its geometry.
    """
    real = _real(real)
    refs = _references(references, old_radius)
    if type(bank_id) is not int or bank_id < 1:
        raise ValueError('proposal bank requires a positive absolute ID')
    if any(type(value) is not Region for value in reserved):
        raise ValueError('reserved regions must be frozen Region receipts')
    reference_sha = hashlib.sha256(refs.numpy().tobytes()).hexdigest()
    nearest = torch.cdist(real.double(), refs).min(dim=1).values
    rejected = real[nearest >= old_radius]
    receipt = dict(bank_id=bank_id, bank_sha256=_sha(real),
                   reference_sha256=reference_sha, bank_size=len(real),
                   rejected_count=len(rejected), old_radius=old_radius)
    if len(rejected) < 4:
        return None, dict(receipt, status='TOO_FEW_REJECTED_TO_GROUP')
    centers, grouping = mst_groups(rejected)
    members = grouping['member_indices']
    selected = min(range(len(members)), key=lambda i: (-len(members[i]), min(members[i])))
    cloud = rejected[members[selected]].double()
    center = centers[selected].double()
    empirical_radius = float(torch.linalg.vector_norm(cloud-center, dim=1).max())
    clearance = float(torch.cdist(center[None], refs).min()) - old_radius
    for region in reserved:
        distance = math.dist(tuple(float(x) for x in center), region.center)
        clearance = min(clearance, distance - region.radius)
    radius = min(empirical_radius, clearance / 2)
    receipt.update(mst_groups=len(members), selected_group=selected,
                   selected_group_count=len(cloud), empirical_radius=empirical_radius,
                   clearance=clearance, radius=radius,
                   center=[float(center[0]), float(center[1])])
    if not math.isfinite(radius) or radius <= 0:
        return None, dict(receipt, status='NO_DISJOINT_POSITIVE_BALL')
    region = Region(tuple(receipt['center']), radius, bank_id, receipt['bank_sha256'],
                    reference_sha, len(rejected), len(cloud))
    if (any(math.dist(region.center, tuple(float(x) for x in ref)) <= region.radius + old_radius
            for ref in refs)
            or any(math.dist(region.center, other.center) <= region.radius + other.radius
                   for other in reserved)):
        raise AssertionError('candidate region intersects frozen old or reserved support')
    return region, dict(receipt, status='REGION_FROZEN')


class AnytimeMassCertificate:
    """One-sided bankwise confidence sequence for a fixed discovery region."""

    def __init__(self, region, *, candidate_index, n_particles, delta=DELTA):
        if type(region) is not Region:
            raise ValueError('mass certificate requires a frozen region')
        if (type(candidate_index) is not int or candidate_index < 1
                or type(n_particles) is not int or n_particles < 1
                or type(delta) is not float or not 0 < delta < 1):
            raise ValueError('invalid candidate allocation, particle count or delta')
        self.region = region
        self.candidate_index = candidate_index
        self.n_particles = n_particles
        self.delta = delta
        self.alpha = delta / (candidate_index * (candidate_index + 1))
        self.banks = 0
        self.samples = 0
        self.hits = 0
        self.last_bank_id = region.proposal_bank_id
        self.last_bank_sha256 = region.proposal_bank_sha256
        self.admitted = False

    def observe(self, real, *, bank_id):
        real = _real(real)
        if self.admitted:
            raise RuntimeError('an admitted certificate is final')
        if type(bank_id) is not int or bank_id != self.last_bank_id + 1:
            raise RuntimeError('validation must use consecutive post-proposal bank IDs')
        digest = _sha(real)
        if digest == self.region.proposal_bank_sha256:
            raise RuntimeError('proposal bank reused as its own held-out validation')
        if digest == self.last_bank_sha256:
            raise RuntimeError('identical consecutive validation bank bytes cannot certify IID mass')
        self.banks += 1
        self.samples += len(real)
        new_hits = int(self.region.contains(real).sum())
        self.hits += new_hits
        self.last_bank_id = bank_id
        self.last_bank_sha256 = digest
        # Check only after a complete new bank. Sum_b 1/[b(b+1)] = 1.
        radius = math.sqrt(math.log(self.banks * (self.banks + 1) / self.alpha)
                           / (2 * self.samples))
        empirical = self.hits / self.samples
        lower = max(0., empirical - radius)
        self.admitted = lower > 1 / self.n_particles
        return dict(status='MASS_SUPPORTED' if self.admitted else 'INSUFFICIENT_MASS_EVIDENCE',
                    bank_id=bank_id, bank_sha256=digest,
                    candidate_index=self.candidate_index, delta=self.delta,
                    candidate_error_allocation=self.alpha,
                    validation_banks=self.banks, validation_samples=self.samples,
                    new_hits=new_hits, hits=self.hits, empirical_mass=empirical,
                    lower_mass_bound=lower, threshold=1 / self.n_particles,
                    admitted=self.admitted)
