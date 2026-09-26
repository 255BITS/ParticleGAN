#!/usr/bin/env python3
"""Prepare six frozen vector declarations and sources; never execute training.

This stdlib-only entry point imports neither torch nor the candidate package.
The supervisor stopped C6-specific runtime integration after stationary
collapse. Backend/initialization semantics need review before a run adapter is
added. PREPARED is not a quality attempt or a qualification result.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import zipfile


HERE = Path(__file__).resolve().parent
PLAN = "benchmarks/transfer_suite/plans/default_comparison.json"
PROFILE = "reports/transfer_suite/unadjusted/leading_profile.json"
NAMES = (
    "vector_two_broad", "vector_unequal_mass", "vector_unequal_width",
    "vector_anisotropic", "vector_overlap", "vector_spiral",
)
C6_ZIP_SHA256 = "4c9776c8547db6e4029899bf2e8bab98b8c5b2e7c8674c8ea90e4718e65cb690"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def local_file(root, module):
    path = root.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(root).as_posix()
    return None


def dependency_closure(root, entries):
    """Conservatively snapshot every statically reachable local Python import.

    Imports inside unused host training functions are included as provenance,
    but are never executed here. External distributions are recorded separately.
    """
    pending, sources, external = list(entries), {}, set()
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing dependency: {name}")
        sources[name] = path.read_bytes()
        if path.suffix != ".py":
            continue
        tree = ast.parse(sources[name], filename=name)
        package = Path(name).parts[:-1]
        for depth in range(1, len(package) + 1):
            init = root.joinpath(*package[:depth], "__init__.py")
            if init.is_file():
                pending.append(init.relative_to(root).as_posix())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package) - node.level + 1]) if node.level else []
                base = ".".join(prefix + ([node.module] if node.module else []))
                modules.append(base)
                modules.extend(".".join(filter(None, (base, a.name))) for a in node.names if a.name != "*")
        for module in modules:
            if not module:
                continue
            source = local_file(root, module)
            if source is None:
                sibling = path.parent.joinpath(*module.split(".")).with_suffix(".py")
                if sibling.is_file():
                    source = sibling.relative_to(root).as_posix()
            if source:
                pending.append(source)
            elif not (root / module.split(".")[0]).exists():
                external.add(module.split(".")[0])
    return sources, sorted(external)


def verify_candidate(root, declaration_path, archive_path):
    declaration = read(declaration_path)
    raw_archive = archive_path.read_bytes()
    if sha(raw_archive) != C6_ZIP_SHA256:
        raise ValueError("candidate archive is not the supervisor-audited immutable C6 archive")
    expected = declaration["source_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        archived = {}
        for name, digest in expected.items():
            data = archive.read(name)
            if sha(data) != digest:
                raise ValueError(f"candidate ZIP declaration mismatch: {name}")
            archived[name] = data
            # Worker files may evolve after the run. Every package module must
            # still be exactly C6 for this target repo to be a valid source.
            if name.startswith("particlegan/") and sha((root / name).read_bytes()) != digest:
                raise ValueError(f"target package changed since C6: {name}")
    declared_package = {name for name in expected if name.startswith("particlegan/")}
    actual_package = {p.relative_to(root).as_posix() for p in (root / "particlegan").rglob("*.py")}
    if actual_package != declared_package:
        raise ValueError("target package module set differs from frozen C6")
    recipe = declaration["recipe"]
    required = dict(continuous=True, critic_memory="moving", game_update="secant_resolvent",
                    bounded_updates=False, optimistic_updates=False, lr=.00425,
                    d_lr_mult=1., prior_lr_mult=2., lr_floor=1., network_lr_floor=1.)
    if any(recipe.get(k) != v for k, v in required.items()):
        raise ValueError("declared learner does not match the audited C6 policy")
    if declaration.get("serial_backward") is not True:
        raise ValueError("C6 requires the recorded serial backward execution mode")
    return declaration, archived, raw_archive


def task_declaration(job, profile, candidate):
    original = deepcopy(job["spec"])
    spec = deepcopy(original)
    card = deepcopy(profile["discriminators"].get(spec["name"]))
    if card:
        spec.update(d_hidden=card.get("width", card.get("hidden")), d_layers=card["layers"],
                    fourier=card["fourier"], research_discriminator=card)
    if (spec["d_every"], spec["g_every"]) != (1, 1):
        raise ValueError("GANTrainer cannot preserve a non-1:1 host update ratio")
    if any(k.startswith("scale_") for k in spec):
        raise ValueError("only the six stationary frozen targets are in scope")
    recipe = deepcopy(candidate["recipe"])
    resources = dict(num_particles=spec["particles"], z_dim=spec["z_dim"], batch_size=spec["batch"])
    recipe.update(resources)
    assert all(recipe[k] == v for k, v in candidate["recipe"].items() if k not in resources)
    steps = [(i * spec["steps"] + 23) // 24 for i in range(1, 25)]
    if len(set(steps)) != 24:
        raise ValueError("frozen fixture does not provide 24 distinct observations")
    return dict(
        schema=1, candidate="API-C6", task=spec["name"], status="NOT_RUN",
        original_job=job, frozen_host_spec=spec, discriminator_card=card,
        recipe=recipe, changed_recipe_fields=resources,
        trainer=dict(api="particlegan.GANTrainer.step", seed=0, serial_backward=True,
                     optimizer_options=deepcopy(candidate["optimizer_options"])),
        evaluator=dict(budget=spec["steps"], observation_steps=steps, samples=4096,
                       thresholds=deepcopy(spec["thresholds"]), scorer="vector_tasks.score_samples",
                       final_verdict="protocol.test_verdict", minimum_stable_checks=5,
                       live_is_primary=True, ema_is_diagnostic=True),
        initialization=dict(order=["explicit prior", "G", "D", "GANTrainer"],
                            global_seed=0, prior_seed=0, prior_init_std=.5,
                            backend="UNRESOLVED: original CPU vs published CUDA-host initialization"),
        streams=dict(data_seed=0, latent_seed=1, penalty_seed=2,
                     generator_real="fresh independent target batch from the same data stream, once per accepted update",
                     evaluation_latent_seed=990, evaluation_target_seed=991, projection_seed=992,
                     evaluation_global_seed=402, paired_output_noise_seed=402 + 1901,
                     evaluation_output_seed_varies_with_step=False,
                     note="Vector paired-output seed is fixed; image seed402+step+1901 is not the vector contract."),
        learner_isolation=dict(budget_is_not_recipe_override=True, targets_and_thresholds_are_not_learner_inputs=True,
                               noise="candidate-owned absolute startup360/720 only", schedules="no external schedules"),
        unresolved=["Choose and pin host backend/initialization/RNG contract before implementing or running the adapter.",
                    "This bounded preparation artifact contains no executable training loop and no quality result."],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="explicit unchanged candidate source checkout")
    parser.add_argument("--output", type=Path, required=True, help="new preparation directory")
    parser.add_argument("--candidate-declaration", default="experiments/constant_game/C6-single/declaration.json")
    parser.add_argument("--candidate-zip", default="experiments/constant_game/C6-single/source.zip")
    args = parser.parse_args()
    root = args.repo.resolve()
    lock = read(HERE / "host-lock.json")
    for name, digest in lock["source_sha256"].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f"frozen host source changed: {name}")
    candidate, archived, archive_bytes = verify_candidate(
        root, root / args.candidate_declaration, root / args.candidate_zip)
    jobs = [job for job in read(root / PLAN) if job["spec"]["runner"] == "vector"]
    if tuple(job["spec"]["name"] for job in jobs) != NAMES:
        raise ValueError("six-task frozen vector declaration changed")
    profile = read(root / PROFILE)
    if set(profile["discriminators"]) != set(NAMES[1:5]):
        raise ValueError("four promoted discriminator cards changed")
    declarations = [task_declaration(job, profile, candidate) for job in jobs]
    sources, external = dependency_closure(root, lock["dependency_entrypoints"])
    # Include every exact package file, even a currently unused public module.
    sources.update({name: data for name, data in archived.items() if name.startswith("particlegan/")})
    payload = {"repo/" + name: data for name, data in sources.items()}
    payload.update({"candidate-source/" + name: data for name, data in archived.items()})
    payload["candidate-source.zip"] = archive_bytes
    payload["candidate-declaration.json"] = (root / args.candidate_declaration).read_bytes()
    for path in sorted(HERE.iterdir()):
        if path.is_file():
            payload["harness/" + path.name] = path.read_bytes()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for declaration in declarations:
        write(output / (declaration["task"] + "-declaration.json"), declaration)
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
    packages = {}
    for package in ("torch", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    receipt = dict(
        schema=1, status="PREPARED_NOT_RUN", candidate="API-C6", target_repo=str(root),
        tasks=list(NAMES), model_construction_calls=0, training_updates=0, gpu_calls=0,
        candidate_zip_sha256=sha(archive_bytes), fixture_lock_sha256=sha((HERE / "host-lock.json").read_bytes()),
        source_sha256={name: sha(data) for name, data in sorted(payload.items())},
        source_zip_sha256=sha((output / "source.zip").read_bytes()),
        local_dependency_files=len(sources), external_import_roots=external,
        preparation_runtime=dict(python=platform.python_version(), executable=sys.executable,
                                 installed_distributions=packages, candidate_package_imported=False),
        training_runtime=None, quality_results=None,
        semantics_status="BLOCKED_PENDING_HOST_BACKEND_REVIEW; C6 runtime integration stopped by supervisor",
    )
    write(output / "preparation.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "tasks", "local_dependency_files", "training_updates", "gpu_calls")}), flush=True)


if __name__ == "__main__":
    main()
