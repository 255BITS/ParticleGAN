"""Atlas717 sampled Noisy(.025) declarations and unchanged direct controls.

Only sampled prior kind/absolute kernel width and variant identity may differ.
The two direct-coordinate TaskSpecs are reused byte-for-byte. Numerical gates,
owners, source admission and freshness remain independent requirements.
"""
from copy import deepcopy
from pathlib import Path

from .contracts import atomic_json, file_hash, read_json, stable_hash

COHORT = "atlas_noisy_particle025_tier1_717_v1"
CANDIDATE_ID = "atlas-noisy025-tier1-717-v1"
TRACK_MARKER = "atlas717_noisy025"
CLAIM_CONTRACT = {"sampling_law": "task_declared", "schedule": "schedule_free",
                  "scoring_weights": "state_selected", "experimental_track": TRACK_MARKER}
SUFFIX = "_noisy025717_v1"
PRIOR_KIND = "noisy_particle_cloud"
SIGMA = .025
VARIANT_DIRECTORY = Path("configs/forge/task-variants/atlas-noisy025717")
VIEW_PATH = Path("configs/forge/views/atlas_noisy025_tier1_717_v1.json")
PROTOCOL_PATH = Path("configs/forge/protocols/screening.json")
PROTOCOL_SHA256 = "3fefb4d47fd2cd8aa6ed110c0a9f5bffefaae700431d1f57ca7b162c8efbb803"
REFERENCE_IDEA_PATH = Path("configs/forge/ideas/atlas.json")
REFERENCE_IDEA_SHA256 = "b8a7d65760d0c21d2b3ccd65564085a6e23cde20775791ecb0f8f932609cec65"
STUDY_PATH = Path("configs/forge/studies/atlas-noisy025-tier1-717-study-v1.json")
PRIOR_EVIDENCE_PATH = Path("reports/forge/prior-evidence/atlas-type-only686.json")
PRIOR_EVIDENCE_SHA256 = "b137f18b19adea865ce8e03babd29058493ef45b6690e693246846725148ee69"
PARENTS = {
    "gaussian1d_acquisition": {"raw_sha256": "b31df784dbe09357810a191247bbb0b17d3bb67918595334d16ff728fd5c2d13", "bytes": 3668, "payload_sha256": "2ee559cdf9589c0051d463b20d03b00c28801442b9c903bcebd63bf38200ec0c", "prior_kind": "mog", "sigma": .025},
    "two_pole": {"raw_sha256": "55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5", "bytes": 3099, "payload_sha256": "2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b", "prior_kind": "particle_cloud", "sigma": 0.},
    "unused_token_hold": {"raw_sha256": "ef8ccde8d1fa54af8bfce01c044e3671de8131c980eb4e8022d12ffc8caf51d8", "bytes": 3032, "payload_sha256": "925288dba6c301657ae595ab6389d355a4c71662edd6d2a1931ce63516a8a169", "prior_kind": "particle_cloud", "sigma": 0.},
    "ae_gan_hold": {"raw_sha256": "53a400c3f2b27ef347076f3cc603345e1442d2d8f97f8052f0b9496ba35bae79", "bytes": 2897, "payload_sha256": "d7ce55cf6c7679fdb2dc7a5292bc7dfdb964f602daec09a39610599d20e4e103", "prior_kind": "mog", "sigma": .025},
    "ring16_acquisition": {"raw_sha256": "e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1", "bytes": 7835, "payload_sha256": "6858cca00f8efa1313da89a0e0dcb18d726ea593a5f44ebfb2f03a155523ec3c", "prior_kind": "mog", "sigma": .025},
    "five_word_joint_acquisition": {"raw_sha256": "5987d782efdc36ba9bb44bb03dfcd9c6aa321d29dc83dc56d184a9677c081e34", "bytes": 4904, "payload_sha256": "a7f6b8df63d8145abda8a7aff0beab8c3c81fa1585d6c328291948e3cd12c3c8", "prior_kind": "particle_cloud", "sigma": 0.},
}
DIRECT_CONTROLS = frozenset({"two_pole", "unused_token_hold"})
SAMPLED_PARENTS = tuple(name for name in PARENTS if name not in DIRECT_CONTROLS)


def _parent_metadata(name):
    return {"schema": "pg_atlas717_noisy025_parent_v1", "task": name,
            "json_sha256": PARENTS[name]["raw_sha256"],
            "axis": "sampled_prior_kind_and_absolute_kernel",
            "parent_sigma": PARENTS[name]["sigma"], "sampled_sigma": SIGMA,
            "ordinary_parent_credit": False, "qualification_reuse": False}


def is_noisy_task(task):
    if not isinstance(task, dict):
        return False
    metadata = task.get("prior_substitution_parent")
    return (task.get("task_cohort") == COHORT
            or isinstance(task.get("id"), str) and task["id"].endswith(SUFFIX)
            or isinstance(metadata, dict) and metadata.get("schema") == "pg_atlas717_noisy025_parent_v1")


def _scientific(task):
    if not isinstance(task, dict):
        raise ValueError("Atlas717 task must be an object")
    result = deepcopy(task)
    cached = result.pop("preflight_blockers", [])
    if not isinstance(cached, list) or any(not isinstance(reason, str) for reason in cached):
        raise ValueError("malformed administrative preflight blockers")
    if "field_ownership" in result and not isinstance(result.pop("field_ownership"), dict):
        raise ValueError("malformed administrative ownership annotation")
    return result


def _verify_parent(root, parent):
    if root is None:
        return
    path = Path(root) / "configs/forge/tasks" / (parent + ".json")
    pin = PARENTS[parent]
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != pin["bytes"]
            or file_hash(path) != pin["raw_sha256"]
            or file_hash(Path(root) / PROTOCOL_PATH) != PROTOCOL_SHA256):
        raise ValueError("Atlas717 original task or seed/stream protocol Source drift")


def validate(task, root=None):
    """Require the whole fixed task after restoring only the declared prior axis."""
    if not is_noisy_task(task) or task.get("task_cohort") != COHORT:
        raise ValueError("Atlas717 sampled prior requires its separate explicit cohort")
    name = task.get("id", "")
    parent = name[:-len(SUFFIX)] if isinstance(name, str) and name.endswith(SUFFIX) else None
    if parent not in SAMPLED_PARENTS:
        raise ValueError("Atlas717 sampled variants cannot replace direct-coordinate controls")
    prior = task.get("execution", {}).get("prior", {})
    if (prior.get("kind") != PRIOR_KIND or type(prior.get("sigma")) is not float
            or prior["sigma"] != SIGMA or prior.get("standardize") is not False
            or prior.get("learnable") is not True
            or stable_hash(task.get("prior_substitution_parent")) != stable_hash(_parent_metadata(parent))):
        raise ValueError("Atlas717 requires sampled Noisy(.025) and exact parent-axis metadata")
    restored = _scientific(task)
    restored.pop("task_cohort")
    restored.pop("prior_substitution_parent")
    restored["id"] = parent
    restored["execution"]["prior"]["kind"] = PARENTS[parent]["prior_kind"]
    restored["execution"]["prior"]["sigma"] = PARENTS[parent]["sigma"]
    if stable_hash(restored) != PARENTS[parent]["payload_sha256"]:
        raise ValueError("Atlas717 may change only sampled prior kind/kernel and variant identity")
    _verify_parent(root, parent)
    return {"cohort": COHORT, "parent_task_id": parent, "variant_id": task["id"],
            "prior_kind": PRIOR_KIND, "sigma": SIGMA, "parent_sigma": PARENTS[parent]["sigma"],
            "parent_raw_sha256": PARENTS[parent]["raw_sha256"],
            "latent_prior_sampled": True, "ordinary_parent_credit": False,
            "qualification_reuse": False, "owner_compatibility_proven": False}


def validate_control(task, root=None):
    name = task.get("id") if isinstance(task, dict) else None
    if name not in DIRECT_CONTROLS or stable_hash(_scientific(task)) != PARENTS[name]["payload_sha256"]:
        raise ValueError("Atlas717 controls require the exact original direct-coordinate TaskSpecs")
    _verify_parent(root, name)
    return {"parent_task_id": name, "task_id": name, "latent_prior_sampled": False,
            "prior_kind": "direct_parameter", "sigma": 0., "ordinary_parent_credit": False,
            "qualification_reuse": False, "unchanged_control": True}


def make_variant(parent):
    name = parent.get("id")
    if name not in SAMPLED_PARENTS or stable_hash(parent) != PARENTS[name]["payload_sha256"]:
        raise ValueError("Atlas717 requires an unchanged sampled parent declaration")
    task = deepcopy(parent)
    task.update(id=name + SUFFIX, task_cohort=COHORT, prior_substitution_parent=_parent_metadata(name))
    task["execution"]["prior"].update(kind=PRIOR_KIND, sigma=SIGMA)
    validate(task)
    return task


def load_variants(root, parents):
    directory = Path(root) / VARIANT_DIRECTORY
    if not directory.is_dir():
        return {}
    paths = sorted(directory.glob("*.json"))
    if {path.stem for path in paths} != {name + SUFFIX for name in SAMPLED_PARENTS}:
        raise ValueError("Atlas717 must retain all four sampled variants and both original controls")
    result = {}
    for path in paths:
        task = read_json(path)
        binding = validate(task)
        if task["id"] != path.stem or task["id"] in parents or binding["parent_task_id"] not in parents:
            raise ValueError("Atlas717 filename/parent identity or task collision")
        result[task["id"]] = task
    for name in DIRECT_CONTROLS:
        validate_control(parents[name])
    return result


def task_id(parent):
    return parent if parent in DIRECT_CONTROLS else parent + SUFFIX


def reference_prior(idea, view, default_prior, *, root):
    """Resolve only the unchanged Atlas global reference, never a task prior.

Schema3 ideas cannot own a latent prior. The global capability/preset context
still needs the exact public Atlas reference cloud; actual owners always read
their separate TaskSpec through task_prior(), including sampled Noisy(.025).
"""
    if idea.get("id") != CANDIDATE_ID:
        return deepcopy(default_prior)
    if (idea.get("schema_version") != 3 or view != branch_view()
            or idea.get("recipe_preset") != "atlas" or idea.get("recipe_overrides", {}) != {}
            or idea.get("extensions", {}) != {} or idea.get("host_adaptation") is not None
            or idea.get("claim_contract") != CLAIM_CONTRACT
            or idea.get("initializer", "deterministic_orthogonal") != "deterministic_orthogonal"):
        raise ValueError("Atlas717 global metadata requires its current empty Atlas declaration and exact view")
    path = Path(root) / REFERENCE_IDEA_PATH
    if not path.is_file() or path.is_symlink() or file_hash(path) != REFERENCE_IDEA_SHA256:
        raise ValueError("original Atlas reference declaration Source drift")
    return deepcopy(read_json(path)["prior"])


def branch_view():
    return {"schema_version": 1, "id": VIEW_PATH.stem, "revision": 1,
            "goal": "discriminator_stability", "evidence_scope": "prior_substitution_variant",
            "reporting": {"family_totals": False},
            "eligibility": {"claim_contract": {"experimental_track": TRACK_MARKER}},
            "calibration": {"status": "provisional", "adoption_blocker": "Isolated717 sampled-kernel branch with unchanged direct controls; no main/default qualification."},
            "ranking": {"compare_compatible_cohorts": True, "cost_separate": True,
                        "policy": "qualified_tier_only_with_raw_metrics"},
            "assignments": [{"task": task_id(name), "qualification_tier": 1,
                             "importance": "required", "order": order}
                            for order, name in enumerate(PARENTS)]}


def validate_request_scope(request):
    if (request.get("candidate", {}).get("id") != CANDIDATE_ID
            or request.get("candidate", {}).get("claim_contract") != CLAIM_CONTRACT
            or request.get("view") != branch_view() or request.get("through_tier") != 1
            or set(request.get("tasks", {})) != {task_id(name) for name in PARENTS}):
        raise ValueError("Atlas717 requires its one candidate, six-row view and original Tier1 scope")
    for name, task in request["tasks"].items():
        if not isinstance(task, dict) or task.get("id") != name:
            raise ValueError("Atlas717 task map keys must equal their exact task identities")
        (validate if is_noisy_task(task) else validate_control)(task)


def supporting_source_paths(task):
    if is_noisy_task(task):
        parent = validate(task)["parent_task_id"]
        return (str(VARIANT_DIRECTORY / (task["id"] + ".json")),
                "configs/forge/tasks/" + parent + ".json", str(PROTOCOL_PATH), str(VIEW_PATH), str(REFERENCE_IDEA_PATH), str(STUDY_PATH), str(PRIOR_EVIDENCE_PATH))
    if isinstance(task, dict) and task.get("id") in DIRECT_CONTROLS:
        validate_control(task)
        return ("configs/forge/tasks/" + task["id"] + ".json", str(PROTOCOL_PATH), str(VIEW_PATH), str(REFERENCE_IDEA_PATH), str(STUDY_PATH), str(PRIOR_EVIDENCE_PATH))
    return ()


def request_source_paths(root, idea):
    """Exact Track A metadata closure for copied Study re-resolution.

The maintained source walker excludes configs/forge unless explicitly listed.
Keep the complete task/variant catalog because resolve_idea recomputes its
catalog support set; defaults and current declarations must travel too.
"""
    if idea.get("id") != CANDIDATE_ID:
        return ()
    if idea.get("claim_contract") != CLAIM_CONTRACT:
        raise ValueError("Track A source closure requires its hashed track claim")
    root = Path(root).resolve()
    paths = {"configs/forge/defaults.json", "configs/forge/legacy-ideas-v1.json",
             "configs/forge/ideas/" + CANDIDATE_ID + ".json",
             "configs/forge/ideas/ka2.json", str(REFERENCE_IDEA_PATH),
             str(STUDY_PATH), str(PRIOR_EVIDENCE_PATH), str(PROTOCOL_PATH), str(VIEW_PATH)}
    paths.update(str(path.relative_to(root))
                 for path in (root / "configs/forge/tasks").glob("*.json"))
    paths.update(str(path.relative_to(root))
                 for path in (root / "configs/forge/task-variants").rglob("*.json"))
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Track A copied metadata closure missing or aliased: " + relative)
    return tuple(sorted(paths))


def write_declarations(root):
    """ROOT paid metadata only; the original six TaskSpecs are never written."""
    root = Path(root)
    sampled = [make_variant(read_json(root / "configs/forge/tasks" / (name + ".json")))
               for name in SAMPLED_PARENTS]
    for task in sampled:
        validate(task, root=root)
    for name in DIRECT_CONTROLS:
        validate_control(read_json(root / "configs/forge/tasks" / (name + ".json")), root=root)
    for task in sampled:
        atomic_json(root / VARIANT_DIRECTORY / (task["id"] + ".json"), task)
    atomic_json(root / VIEW_PATH, branch_view())
    return {"sampled_variants": [task["id"] for task in sampled],
            "unchanged_controls": sorted(DIRECT_CONTROLS), "view": str(VIEW_PATH),
            "ordinary_parent_credit": False, "source_only_declarations": True}
