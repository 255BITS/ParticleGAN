"""Proposed reachable teacher-span transfer law; no training entry point.

Keep PR227's fixture and public native API intact. Rotate only the ordinary
teacher's down row spaces away from both students' common initial down basis.
The fixed rho is a weight-space observation from the full Supra comparison,
not an estimate of its frozen-caption target span or a causal quality result.
"""
from copy import deepcopy

import torch

if __package__:
    from . import e22_routed_convergence as baseline
    from . import e22_routed_convergence_neutral as neutral
else:
    import e22_routed_convergence as baseline
    import e22_routed_convergence_neutral as neutral


RHO = 0.027625366713810866
AUDIT_SHA256 = "bfd38b84d558559ebe048ae93302270e6059253a4ccf6149f22eb8117f428241"
SITES = ("first", "second")
TASK = "routed_convergence_rotated_teacher_v1"
ARMS = ("ordinary_native_game", "particle_native_game", "neutral_particle_native_game")


def canonical_complement(basis):
    """Lexicographic Gram-Schmidt of canonical axes, with reorthogonalization."""
    width, rank = basis.shape
    columns = []
    for coordinate in range(width):
        vector = torch.zeros(width, dtype=basis.dtype, device=basis.device)
        vector[coordinate] = 1.
        for _ in range(2):
            vector = vector - basis @ (basis.T @ vector)
            for previous in columns:
                vector = vector - previous * (previous @ vector)
        norm = vector.norm()
        if norm > 1e-10:
            columns.append(vector / norm)
        if len(columns) == rank:
            return torch.stack(columns, dim=1)
    raise ValueError("input width cannot provide the required orthogonal complement")


def rotated_down(down, rho=RHO):
    if not 0. <= rho <= 1.:
        raise ValueError("rho must lie in [0, 1]")
    if down.ndim != 2 or down.shape[1] < 2 * down.shape[0] or down.shape[0] == 0:
        raise ValueError("teacher rotation requires width >= 2*rank")
    singular = torch.linalg.svdvals(down.detach().double())
    if singular[-1] <= singular[0] * 1e-10:
        raise ValueError("teacher down must have full row rank")
    if rho == 1.:
        return down.clone()
    q, r = torch.linalg.qr(down.detach().double().T, mode="reduced")
    orthogonal = canonical_complement(q)
    result = r.T @ (rho**.5 * q.T + (1. - rho)**.5 * orthogonal.T)
    return result.to(down.dtype)


@torch.no_grad()
def make_rotated_data(original=None, rho=RHO):
    original = baseline.make_data() if original is None else original
    data = deepcopy(original)
    data.pop("digest")
    with torch.random.fork_rng(devices=[]):
        teacher = baseline.Host()
    teacher.load_state_dict(data["teacher"])
    teacher.eval().requires_grad_(False)
    for site in SITES:
        initial_down = data["initial_ordinary"][site + ".down.weight"]
        getattr(teacher, site).down.weight.copy_(rotated_down(initial_down, rho))
    data["teacher"] = teacher.state_dict()
    for pool in baseline.SPLITS:
        data[pool]["targets"] = baseline.batched_host(teacher, data[pool]["context"])
    data["raw_scale"] = (data["fit"]["targets"] - data["fit"]["base"]).std(dim=(0, 1))
    data["scale"] = data["raw_scale"].clamp_min(.04)
    data["teacher_span_rotation"] = {
        "id": "routed_convergence_rotated_teacher_v1", "rho": rho,
        "rho_source": "mean linear-delta Frobenius energy captured by the fresh Supra down span",
        "audit_sha256": AUDIT_SHA256, "parent_data_digest": original["digest"],
        "law": "Drot=R.T@(sqrt(rho)*Q.T+sqrt(1-rho)*O.T); D_initial.T=Q@R",
        "canonical_complement": "first rank independent canonical axes, projected and twice reorthogonalized",
        "changed": "teacher down row space; target outputs and fixed fit residual scales follow",
        "fixed": "all fresh student tensors and critic trainable initialization weights; teacher up, inputs, sources, times, latent IDs and native recipe; critic scale follows the derived target scale",
    }
    data["digest"] = baseline.digest(data)
    return data


def make_rotated_loop(arm, data, *, bindings=None):
    """Reuse the held native game; distinguish the new task in checkpoint law."""
    if arm not in ARMS:
        raise ValueError("unknown rotated-teacher arm")
    rotation = data.get("teacher_span_rotation", {})
    if rotation.get("id") != TASK or rotation.get("rho") != RHO or rotation.get("audit_sha256") != AUDIT_SHA256:
        raise ValueError("expected the declared rotated-teacher data law")
    if arm == ARMS[2]:
        loop = neutral.make_neutral_loop(neutral.make_neutral_data(data), bindings=bindings)
    else:
        loop = baseline.make_loop(arm, data, bindings=bindings)
    # Loop.arm keeps the held forward's particle dispatch; the external arm is
    # recorded in the authoritative checkpoint law rather than changing math.
    loop.law.update(task=TASK, arm=arm, teacher_span_rotation=deepcopy(rotation),
                    common_task_data_digest=data["digest"])
    return loop


@torch.no_grad()
def feasibility_witness(original=None):
    """Zero-update geometric, initialization, immutable-input and reachability checks."""
    original = baseline.make_data() if original is None else original
    before = baseline.digest(original)
    data = make_rotated_data(original)
    aligned = make_rotated_data(original, rho=1.)
    if baseline.digest(original) != before:
        raise AssertionError("teacher law mutated the source data")
    if not torch.equal(data["sources"], original["sources"]):
        raise AssertionError("source condition changed")
    geometry = {}
    for site in SITES:
        key = site + ".down.weight"
        initial = original["initial_ordinary"][key].double()
        rotated = data["teacher"][key].double()
        q_initial = torch.linalg.qr(initial.T, mode="reduced").Q
        q_rotated = torch.linalg.qr(rotated.T, mode="reduced").Q
        squared_cosines = torch.linalg.svdvals(q_initial.T @ q_rotated).square()
        gram_error = float((rotated @ rotated.T - initial @ initial.T).abs().max())
        if not torch.allclose(squared_cosines, torch.full_like(squared_cosines, RHO), rtol=1e-6, atol=1e-8):
            raise AssertionError("actual teacher rotation differs from the declared rho")
        if gram_error > 1e-7:
            raise AssertionError("teacher rotation changed its within-rank Gram")
        if not torch.equal(data["teacher"][site + ".up.weight"], original["teacher"][site + ".up.weight"]):
            raise AssertionError("teacher up changed")
        geometry[site] = {"squared_principal_cosines": squared_cosines.tolist(),
                          "within_rank_gram_max_absolute_error": gram_error}
    for family in ("initial_ordinary", "initial_particle", "initialization", "source_projection"):
        if baseline.digest(data[family]) != baseline.digest(original[family]):
            raise AssertionError("fresh initialization or source projection changed")
    for pool in baseline.SPLITS:
        for key, value in original[pool].items():
            if key != "targets" and not torch.equal(data[pool][key], value):
                raise AssertionError("fixed input/base panel changed")
        if not torch.equal(aligned[pool]["targets"], original[pool]["targets"]):
            raise AssertionError("rho=1 does not reproduce the aligned teacher")
    return {"optimizer_updates": 0, "rho": RHO, "geometry": geometry,
            "source_data_unchanged": True, "student_initial_owners_unchanged": True,
            "fixed_inputs_unchanged": True, "rho_one_targets_exact": True,
            "teacher_up_unchanged": True, "raw_scale_min": float(data["raw_scale"].min()),
            "raw_scale_max": float(data["raw_scale"].max()),
            "scale_floor_coordinates": int((data["raw_scale"] < .04).count_nonzero()),
            "data_digest": data["digest"],
            "reachability": baseline.reachability_witness(data)}
