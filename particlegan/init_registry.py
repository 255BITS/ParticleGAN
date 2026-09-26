"""One ``--init`` name for every deterministic weight and particle initializer.

The default PyTorch init is unchanged until ``install`` or ``use_init`` is
called. Names come from the screened families:

* A, PR #173: delta-orthogonal and scaled identity, including ``hid_q``
* B, PR #174: Hadamard / DCT / DST / Householder
* C, PR #175: orthogonal LSUV, Sobol, Halton, fixed generator
* D, PR #179: ``hid_q`` x ``qr_pb_pq`` hybrids, including ``qr_pb_pq``
* E, PR #180: structured orthogonal families (Cayley, Fourier, Haar, ...)
* F, PR #181: particle-prior sequences on the ``hid_q`` and ``qr_pb_pq`` arms

``hid_q`` is family A's. Family D and family F reproduce that name; the
registry keeps A's tensors for it. ``qr_pb_pq`` is family D's copy of the
ortho-search init. Family F's other prior names still use family F's arm.
"""
from __future__ import annotations

import hashlib
import os

import torch
from torch import nn

from particlegan import (
    det_init_a,
    det_init_d,
    det_init_f,
    deterministic_init,
    family_e_init,
    structured_init,
)
from particlegan.particle_prior import ParticlePrior

# (family, names, install). First family to claim a name keeps it.
_FAMILIES = (
    ("A", det_init_a.VARIANTS, det_init_a.install),
    ("B", structured_init.NAMES, structured_init.configure),
    ("C", deterministic_init.KINDS, deterministic_init.install),
    ("D", det_init_d.VARIANTS, det_init_d.install),
    ("E", family_e_init.names(), family_e_init.install),
    ("F", det_init_f.VARIANTS, det_init_f.install),
)

_INSTALLERS: dict[str, tuple[str, object]] = {}
for _family, _variant_names, _install in _FAMILIES:
    for _name in _variant_names:
        _INSTALLERS.setdefault(_name, (_family, _install))

NAMES = tuple(_INSTALLERS)


def names() -> tuple[str, ...]:
    return NAMES


def family_of(name: str) -> str:
    try:
        family, _install = _INSTALLERS[name]
    except KeyError:
        raise ValueError(f"unknown init {name!r}; {len(NAMES)} names are registered") from None
    return family


def install(name: str) -> str:
    """Activate one registered init. Construction order restarts inside that family."""
    try:
        _family, installer = _INSTALLERS[name]
    except KeyError:
        raise ValueError(f"unknown init {name!r}; {len(NAMES)} names are registered") from None
    installer(name)
    return name


def use_init(name: str | None) -> str | None:
    """Install ``name``, or ``K3P_INIT`` when ``name`` is empty.

    ``None`` with ``K3P_INIT`` unset leaves the PyTorch init in place.
    """
    chosen = name or os.environ.get("K3P_INIT") or None
    if not chosen:
        return None
    return install(chosen)


def add_init_argument(parser) -> None:
    parser.add_argument(
        "--init",
        default=None,
        help="deterministic weight and particle init name; omit to keep the PyTorch init",
    )


def witness_sha256(prepare=None) -> str:
    """SHA-256 of one fixed CPU init witness. ``install`` must already have run.

    The witness is a stand-in for the K3P modules the hooks rewrite: a small
    generator, critic, rectangular map, convolution, conv-transpose, and
    particle table, then the same two Adam constructions the ring trainer uses.
    """
    torch.manual_seed(0)
    generator = nn.Sequential(nn.Linear(8, 8), nn.LeakyReLU(0.2), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1))
    wide = nn.Linear(7, 11)
    conv = nn.Conv2d(3, 3, 3)
    deconv = nn.ConvTranspose2d(4, 2, 4)
    prior = ParticlePrior(5, 2, init_std=0.25, generator=torch.Generator(device="cpu").manual_seed(9))
    if prepare is not None:
        prepare(generator, critic)
    elif deterministic_init.active_kind() == "ortho_lsuv":
        deterministic_init.prepare_modules(generator, critic)
    torch.optim.Adam(list(generator.parameters()) + list(prior.parameters()), lr=1e-3)
    torch.optim.Adam(
        list(critic.parameters()) + list(wide.parameters()) + list(conv.parameters()) + list(deconv.parameters()),
        lr=1e-3,
    )
    hasher = hashlib.sha256()
    for module in (generator, critic, wide, conv, deconv, prior):
        for param in module.parameters():
            value = param.detach().cpu().contiguous()
            hasher.update(str(tuple(value.shape)).encode())
            hasher.update(str(value.dtype).encode())
            hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()
