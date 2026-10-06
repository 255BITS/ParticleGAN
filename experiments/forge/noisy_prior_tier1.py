"""Fixed-input NoisyParticlePrior Tier 1 declarations; no scientific execution.

The six task variants change only identity and the declared prior type. Owner
compatibility, fresh initialization and numerical grades remain separate gates.
"""
from copy import deepcopy
from pathlib import Path

from .contracts import atomic_json, file_hash, read_json, stable_hash

COHORT = "noisy_particle_prior_tier1_686_v1"
SUFFIX = "_noisy_prior686_v1"
PRIOR_KIND = "noisy_particle_cloud"
VARIANT_DIRECTORY = Path("configs/forge/task-variants/noisy-prior686")
VIEW_PATH = Path("configs/forge/views/noisy_prior_tier1_686_v1.json")
PROTOCOL_PATH = Path("configs/forge/protocols/screening.json")
PROTOCOL_SHA256 = "3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803"
PARENTS = {
    "gaussian1d_acquisition": {
        "raw_sha256": "b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13",
        "bytes": 3668,
        "payload_sha256": "2ee559cdf9589c0051d463b20d03b00c28801442b9c903bcebd63bf38200ec0c",
        "prior_kind": "mog",
        "sigma": 0.025
    },
    "two_pole": {
        "raw_sha256": "55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5",
        "bytes": 3099,
        "payload_sha256": "2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b",
        "prior_kind": "particle_cloud",
        "sigma": 0.0
    },
    "unused_token_hold": {
        "raw_sha256": "ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8",
        "bytes": 3032,
        "payload_sha256": "925288dba6c301657ae595ab6389d355a4c71662edd6d2a1931ce63516a8a169",
        "prior_kind": "particle_cloud",
        "sigma": 0.0
    },
    "ae_gan_hold": {
        "raw_sha256": "53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79",
        "bytes": 2897,
        "payload_sha256": "d7ce55cf6c7679fdb2dc7a5292bc7dfdb964f602daec09a39610599d20e4e103",
        "prior_kind": "mog",
        "sigma": 0.025
    },
    "ring16_acquisition": {
        "raw_sha256": "e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1",
        "bytes": 7835,
        "payload_sha256": "6858cca00f8efa1313da89a0e0dcb18d726ea593a5f44ebfb2f03a155523ec3c",
        "prior_kind": "mog",
        "sigma": 0.025
    },
    "five_word_joint_acquisition": {
        "raw_sha256": "5987d782efdc36ba9bb44bb03dfcd9c6aa321d29dc83dc56d184a9677c081e34",
        "bytes": 4904,
        "payload_sha256": "a7f6b8df63d8145abda8a7aff0beab8c3c81fa1585d6c328291948e3cd12c3c8",
        "prior_kind": "particle_cloud",
        "sigma": 0.0
    }
}


def _parent_metadata(name):
    return {"schema": "pg_noisy_prior686_parent_v1", "task": name,
            "json_sha256": PARENTS[name]["raw_sha256"], "axis": "prior_type_only",
            "ordinary_parent_credit": False, "qualification_reuse": False}


def is_noisy_task(task):
    if not isinstance(task, dict):
        return False
    execution = task.get("execution")
    prior = execution.get("prior") if isinstance(execution, dict) else None
    return (task.get("task_cohort") == COHORT
            or isinstance(task.get("id"), str) and task["id"].endswith(SUFFIX)
            or isinstance(prior, dict) and prior.get("kind") == PRIOR_KIND
            or "prior_substitution_parent" in task)


def validate(task, root=None):
    """Validate fixed declarations only; this does not certify an owner or grade."""
    if not is_noisy_task(task) or task.get("task_cohort") != COHORT:
        raise ValueError("Noisy Tier 1 requires its explicit isolated cohort")
    name = task.get("id", "")
    parent = name[:-len(SUFFIX)] if isinstance(name, str) and name.endswith(SUFFIX) else None
    if parent not in PARENTS:
        raise ValueError("Noisy Tier 1 variant must name one of the six fixed parents")
    execution = task.get("execution")
    prior = execution.get("prior") if isinstance(execution, dict) else None
    if (not isinstance(prior, dict) or prior.get("kind") != PRIOR_KIND
            or stable_hash(task.get("prior_substitution_parent")) != stable_hash(_parent_metadata(parent))):
        raise ValueError("Noisy Tier 1 requires its exact prior-kind and parent metadata")
    restored = deepcopy(task)
    for key in ("task_cohort", "prior_substitution_parent", "preflight_blockers", "field_ownership"):
        restored.pop(key, None)
    restored["id"] = parent
    restored["execution"]["prior"]["kind"] = PARENTS[parent]["prior_kind"]
    if stable_hash(restored) != PARENTS[parent]["payload_sha256"]:
        raise ValueError("Noisy Tier 1 may change only prior type, never fixed task inputs")
    if root is not None:
        root = Path(root)
        path = root / "configs/forge/tasks" / (parent + ".json")
        if (not path.is_file() or path.stat().st_size != PARENTS[parent]["bytes"]
                or file_hash(path) != PARENTS[parent]["raw_sha256"]
                or file_hash(root / PROTOCOL_PATH) != PROTOCOL_SHA256):
            raise ValueError("Noisy Tier 1 parent or seed/stream protocol Source drift")
    return {"cohort": COHORT, "parent_task_id": parent, "variant_id": task["id"],
            "prior_kind": PRIOR_KIND, "sigma": PARENTS[parent]["sigma"],
            "parent_raw_sha256": PARENTS[parent]["raw_sha256"],
            "ordinary_parent_credit": False, "owner_compatibility_proven": False}


def make_variant(parent):
    name = parent.get("id")
    if name not in PARENTS or stable_hash(parent) != PARENTS[name]["payload_sha256"]:
        raise ValueError("Noisy Tier 1 requires its unchanged full parent declaration")
    task = deepcopy(parent)
    task.update(id=name + SUFFIX, task_cohort=COHORT, prior_substitution_parent=_parent_metadata(name))
    task["execution"]["prior"]["kind"] = PRIOR_KIND
    validate(task)
    return task


def load_variants(root, parents):
    directory = Path(root) / VARIANT_DIRECTORY
    if not directory.is_dir():
        return {}
    paths = sorted(directory.glob("*.json"))
    expected = {name + SUFFIX for name in PARENTS}
    if {path.stem for path in paths} != expected:
        raise ValueError("Noisy Tier 1 must retain all six fixed variant declarations")
    result = {}
    for path in paths:
        task = read_json(path)
        # Physical Source drift is checked by validate(root=...) at preflight;
        # unrelated ordinary views may still load their own unchanged tasks.
        validate(task)
        parent = task["prior_substitution_parent"]["task"]
        if (task["id"] != path.stem or task["id"] in parents
                or parent not in parents):
            raise ValueError("Noisy Tier 1 filename/parent identity or task collision")
        result[task["id"]] = task
    return result


def branch_view():
    return {"schema_version": 1, "id": VIEW_PATH.stem, "revision": 1,
            "goal": "discriminator_stability", "evidence_scope": "prior_substitution_variant",
            "reporting": {"family_totals": False}, "eligibility": {},
            "calibration": {"status": "provisional",
                "adoption_blocker": "Isolated prior-type branch; no main-parent or default qualification."},
            "ranking": {"compare_compatible_cohorts": True, "cost_separate": True,
                        "policy": "qualified_tier_only_with_raw_metrics"},
            "assignments": [{"task": name + SUFFIX, "qualification_tier": 1,
                             "importance": "required", "order": order}
                            for order, name in enumerate(PARENTS)]}


def write_declarations(root):
    """ROOT-only paid metadata producer; it never trains or selects a family."""
    root = Path(root)
    tasks = []
    for name in PARENTS:
        task = make_variant(read_json(root / "configs/forge/tasks" / (name + ".json")))
        validate(task, root=root)
        tasks.append(task)
    # Validate the complete roster before any output mutation.
    for task in tasks:
        atomic_json(root / VARIANT_DIRECTORY / (task["id"] + ".json"), task)
    atomic_json(root / VIEW_PATH, branch_view())
    return {"variants": [task["id"] for task in tasks], "view": VIEW_PATH.as_posix(),
            "ordinary_parent_credit": False, "source_only_declarations": True}
