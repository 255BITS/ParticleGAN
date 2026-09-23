"""Run or independently regrade the 100-mode + 19-toy common-recipe gate.

``run`` creates fresh evidence for all three 100-mode problems and all 19
canonical transfer cases. ``regrade`` checks their saved evidence without
training. A separate installed-wheel public-default replay is an optional
control. A common 22/22 PASS requires one identical global recipe *including*
noise on every host, as well as each host's frozen live gate. Candidate runs
without complete training-noise receipts remain INCOMPLETE for that claim.

python -u -m benchmarks.toy_suite run \
    --config configs/toy100/recommended.json --output /tmp/toy-suite-22
python -m benchmarks.toy_suite regrade --output /tmp/toy-suite-22
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
from benchmarks.toy100.gate import evaluate_suite as coverage_suite
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy100.models import linear_input_noise
from benchmarks.toy100.train import resolve_config
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
)
from benchmarks.transfer_suite.toy100_compatibility import (
    VECTOR_NAMES, declared_recipe, output_noise_at,
)
from particlegan import Recipe, learning_rate_scale


ROOT = Path(__file__).resolve().parents[1]
NOISE_FIELDS = ("output_noise_std", "input_noise_std",
                "input_noise_anneal_end", "output_noise_warmup")


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _status(passed: int, required: int, *, complete: bool) -> str:
    return "INCOMPLETE" if not complete else "PASS" if passed == required else "FAIL"


def _json_value(value):
    """Compare resolved tuples and archived JSON with the same representation."""
    return json.loads(json.dumps(value, allow_nan=False))


def _close(actual, expected):
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14))


def _check_optimizer_receipts(record: dict, base: Recipe):
    receipts = record.get("applied")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError(f"optimizer receipts are absent: {record['name']}")
    roles = [item.get("role") for item in receipts]
    if len(set(roles)) != len(roles) or "d" not in roles or not {"g", "prior"}.intersection(roles):
        raise ValueError(f"optimizer roles differ: {record['name']}")
    if record["spec"]["runner"] != "legacy" and roles != ["g", "prior", "d"]:
        raise ValueError(f"trainer optimizer roles differ: {record['name']}")
    rates = dict(g=base.lr, prior=base.lr * base.prior_lr_mult,
                 d=base.lr * base.d_lr_mult)
    for item in receipts:
        role = item["role"]
        if role not in rates or not _close(item.get("lr"), rates[role]):
            raise ValueError(f"optimizer rate differs from common recipe: {record['name']}.{role}")
        betas = base.prior_betas or base.betas if role == "prior" else base.betas
        if item.get("betas") != list(betas):
            raise ValueError(f"optimizer betas differ from common recipe: {record['name']}.{role}")
        if type(item.get("parameters")) is not int or item["parameters"] <= 0:
            raise ValueError(f"optimizer parameter receipt is invalid: {record['name']}.{role}")
        if record["spec"]["runner"] != "legacy" and item.get("optimizer") != "Adam":
            raise ValueError(f"trainer optimizer type differs: {record['name']}.{role}")


def _check_actions(record: dict, base: Recipe, noise: dict | None):
    spec, result = record["spec"], record["result"]
    name, steps = spec["name"], spec["steps"]
    actions = result.get("actions", [])
    if spec["runner"] == "legacy":
        expected = [(step, role) for step in range(0, steps, 20)
                    for role in ("d", "g")]
        if [(item.get("step"), item.get("role")) for item in actions] != expected:
            raise ValueError(f"custom-host optimizer trace is incomplete: {name}")
        for item in actions:
            scale = learning_rate_scale(item["step"], steps,
                                        base.lr_anneal_start, base.lr_floor)
            if not _close(item.get("multiplier"), scale):
                raise ValueError(f"custom-host LR schedule differs from common recipe: {name}")
        return
    if len(actions) != steps or result.get("update_counts") != {"g": steps, "d": steps}:
        raise ValueError(f"trainer action or update trace is incomplete: {name}")
    for completed, action in enumerate(actions, start=1):
        scale = learning_rate_scale(completed - 1, steps,
                                    base.lr_anneal_start, base.lr_floor)
        expected_rates = dict(step=completed, multiplier=scale,
                              lr_g=base.lr * scale,
                              lr_prior=base.lr * base.prior_lr_mult * scale,
                              lr_d=base.lr * base.d_lr_mult * scale)
        if any(not _close(action.get(key), value) for key, value in expected_rates.items()):
            raise ValueError(f"trainer LR action differs from common recipe: {name}.{completed}")
        if noise is not None:
            input_sigma = linear_input_noise(
                noise["input_noise_std"], completed - 1, steps,
                noise["input_noise_anneal_end"],
            )
            output_sigma = output_noise_at(
                noise["output_noise_std"], completed - 1, steps,
                noise.get("output_noise_warmup", 0.0),
            )
            if not _close(action.get("input_sigma"), input_sigma):
                raise ValueError(f"trainer input noise schedule differs: {name}.{completed}")
            if ("output_noise_warmup" in noise
                    and not _close(action.get("output_sigma"), output_sigma)):
                raise ValueError(f"trainer output warmup differs: {name}.{completed}")


def _verify_saved_provenance(directory: Path, protocol: dict, *, candidate: bool):
    """Bind the saved executable source and declared config to their hashes."""
    source_hashes = dict(protocol["source_sha256"])
    noise_name = "benchmarks/toy100/models.py"
    if candidate:
        expected_noise_hash = source_hashes.pop(noise_name)
        if (expected_noise_hash != protocol["noise_source_sha256"]
                or hashlib.sha256((directory / "noise_source.py").read_bytes()).hexdigest()
                != expected_noise_hash):
            raise ValueError("saved noise source differs from protocol hash")
        config_name = Path(protocol["config_file"])
        if config_name.is_absolute() or len(config_name.parts) != 1:
            raise ValueError("candidate config path escapes evidence directory")
        config_bytes = (directory / config_name).read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != protocol["config_sha256"]:
            raise ValueError("saved candidate config differs from protocol hash")
        base, noise, overrides = declared_recipe(json.loads(config_bytes))
        declared_noise = {**protocol["noise"],
                          "output_noise_warmup": protocol["noise"].get(
                              "output_noise_warmup", 0.0)}
        if (_json_value(base.to_dict()) != protocol["global_recipe"]
                or noise != declared_noise
                or overrides != protocol["ignored_toy100_resource_overrides"]):
            raise ValueError("saved candidate config does not resolve to declared recipe")
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        all_members = archive.getmembers()
        members = {member.name: member for member in all_members}
        if len(members) != len(all_members) or set(members) != set(source_hashes):
            raise ValueError("source archive members differ from protocol manifest")
        for name, expected_hash in source_hashes.items():
            member = members[name]
            stream = archive.extractfile(member) if member.isfile() else None
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected_hash:
                raise ValueError(f"saved source archive differs: {name}")


def _episode_rows(directory: Path, expected_names: tuple[str, ...], *, candidate: bool):
    """Recompute every live verdict from the compressed episode, not the stamp."""
    if not (directory / "protocol.json").is_file():
        return dict(status="MISSING", passed=0, required=len(expected_names),
                    cases={}, reason="protocol.json is absent")
    try:
        protocol = _read(directory / "protocol.json")
        _verify_saved_provenance(directory, protocol, candidate=candidate)
        index = _read(directory / "index.json")
        rows = index["records"]
        names = [row["name"] for row in rows]
        if len(names) != len(set(names)) or set(names) - set(expected_names):
            raise ValueError("indexed episode names are duplicated or unknown")
        jobs, profile = load_declaration()
        frozen_jobs = {job["spec"]["name"]: job for job in jobs}
        if protocol["jobs"] != [frozen_jobs[job["spec"]["name"]]
                                for job in protocol["jobs"]]:
            raise ValueError("archived jobs differ from frozen declarations")
        if candidate:
            if protocol["frozen_discriminators"] != profile["discriminators"]:
                raise ValueError("candidate discriminator profile differs from frozen card")
            recipe_fields = protocol["global_recipe"]
        else:
            if protocol["frozen_profile"] != profile:
                raise ValueError("public control profile differs from frozen card")
            recipe_fields = protocol["base_get_recipe"]
        base = Recipe(**recipe_fields)
        if _json_value(base.to_dict()) != recipe_fields:
            raise ValueError("archived recipe does not resolve to declared fields")
        cases = {}
        for row in rows:
            artifact = (directory / row["artifact"]).resolve()
            if not artifact.is_relative_to(directory.resolve()):
                raise ValueError("episode path escapes evidence directory")
            raw = gzip.decompress(artifact.read_bytes())
            if hashlib.sha256(raw).hexdigest() != row["uncompressed_sha256"]:
                raise ValueError(f"episode hash differs: {row['name']}")
            record = json.loads(raw)
            name = row["name"]
            if record["name"] != name or record["original_spec"] != frozen_jobs[name]["spec"]:
                raise ValueError(f"frozen task declaration differs: {name}")
            expected_spec, _, variant = declared_spec(frozen_jobs[name], profile, base)
            if (record["spec"] != _json_value(expected_spec)
                    or record.get("discriminator_variant") != _json_value(variant)):
                raise ValueError(f"executed spec, budget, threshold, or discriminator differs: {name}")
            expected_host_recipe = (base if expected_spec["runner"] == "legacy"
                                    else host_recipe(base, expected_spec))
            if record.get("host_recipe") != _json_value(expected_host_recipe.to_dict()):
                raise ValueError(f"host resource recipe differs from declaration: {name}")
            if record["source_sha256"] != protocol["source_sha256"]:
                raise ValueError(f"episode source differs: {name}")
            if candidate:
                if record["recipe"] != protocol["global_recipe"] or record["noise"] != protocol["noise"]:
                    raise ValueError(f"candidate global fields differ between cases: {name}")
                receipt = record.get("noise_receipt")
                if not isinstance(receipt, dict):
                    raise ValueError(f"candidate noise receipt is absent: {name}")
                output_std = protocol["noise"]["output_noise_std"]
                input_std = protocol["noise"]["input_noise_std"]
                warmup = protocol["noise"].get("output_noise_warmup", 0.0)
                steps = record["spec"]["steps"]
                if receipt.get("step_calls") != steps:
                    raise ValueError(f"candidate noise schedule is incomplete: {name}")
                expected_first = output_noise_at(output_std, 0, steps, warmup)
                expected_last = output_noise_at(
                    output_std, steps - 1, steps, warmup,
                )
                expected_input_nonzero = sum(
                    linear_input_noise(input_std, step, steps,
                                       protocol["noise"]["input_noise_anneal_end"]) > 0
                    for step in range(steps)
                )
                if receipt.get("input_nonzero_steps") != expected_input_nonzero:
                    raise ValueError(f"candidate input noise duration differs: {name}")
                if "output_noise_warmup" in protocol["noise"] and warmup:
                    if (not _close(receipt.get("output_sigma_first"), expected_first)
                            or not _close(receipt.get("output_sigma_last"), expected_last)):
                        raise ValueError(f"candidate output warmup differs: {name}")
                    expected_output_nonzero = sum(
                        output_noise_at(output_std, step, steps, warmup) > 0
                        for step in range(steps)
                    )
                    if receipt.get("output_nonzero_steps") != expected_output_nonzero:
                        raise ValueError(f"candidate output warmup duration differs: {name}")
                if record["spec"]["runner"] == "legacy":
                    if (not _close(receipt.get("output_std"), output_std)
                            or not _close(receipt.get("input_std"), input_std)
                            or not _close(receipt.get("input_anneal_end"),
                                          protocol["noise"]["input_noise_anneal_end"])
                            or not _close(receipt.get("input_sigma_first"),
                                          linear_input_noise(input_std, 0, steps,
                                                             protocol["noise"]["input_noise_anneal_end"]))
                            or not _close(receipt.get("input_sigma_last"),
                                          linear_input_noise(input_std, steps - 1, steps,
                                                             protocol["noise"]["input_noise_anneal_end"]))):
                        raise ValueError(f"custom-host noise receipt differs from common policy: {name}")
                    actual_noise = ((output_std == 0 or receipt.get("train_output_applied"))
                                    and (input_std == 0 or receipt.get("train_input_applied")
                                         and receipt.get("input_nonzero_steps", 0) > 0))
                    if not receipt.get("eval_scope"):
                        raise ValueError(f"custom-host evaluation scope is absent: {name}")
                else:
                    actual_noise = ((output_std == 0 or receipt.get("output_module") == "OutputNoise")
                                    and (input_std == 0 or receipt.get("input_module") == "InputNoise"
                                         and receipt.get("input_nonzero_steps", 0) > 0))
                if bool(actual_noise) != record.get("noise_applied"):
                    raise ValueError(f"candidate noise claim differs from receipt: {name}")
            else:
                if record["recipe"] != protocol["base_get_recipe"]:
                    raise ValueError(f"public default recipe differs between cases: {name}")
            result = record["result"]
            _check_optimizer_receipts(record, expected_host_recipe)
            _check_actions(record, expected_host_recipe,
                           protocol["noise"] if candidate else None)
            verdict = test_verdict(record["spec"], result)
            if (verdict["status"] != record["verdict"]["status"]
                    or verdict["passed"] != record["verdict"]["passed"]
                    or verdict["status"] != row["verdict"]["status"]):
                raise ValueError(f"stored verdict differs from independent regrade: {name}")
            observations = result.get("observations", result.get("curve", []))
            expected_checkpoints = sorted({math.ceil(i * record["spec"]["steps"] / 24)
                                           for i in range(1, 25)})
            if [item.get("step") for item in observations] != expected_checkpoints:
                raise ValueError(f"frozen 24-checkpoint schedule differs: {name}")
            cases[name] = dict(status=verdict["status"], passed=verdict["passed"],
                               observations=len(observations), artifact=str(artifact),
                               noise_applied=record.get("noise_applied", not candidate),
                               eval_scope=(record.get("noise_receipt") or {}).get("eval_scope"),
                               final=result.get("live", {}),
                               passing_suffix=verdict.get("convergence", {}).get("passing_suffix"))
        passed = sum(row["passed"] for row in cases.values())
        return dict(status=_status(passed, len(expected_names),
                                   complete=set(cases) == set(expected_names)),
                    passed=passed, required=len(expected_names), cases=cases,
                    protocol=protocol, reason=None)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            tarfile.TarError,
            gzip.BadGzipFile, EOFError) as error:
        return dict(status="INVALID", passed=0, required=len(expected_names),
                    cases={}, reason=str(error))


def _toy100_rows(directory: Path):
    if not (directory / "run_manifest.json").is_file():
        return dict(status="MISSING", passed=0, required=len(PROBLEM_NAMES),
                    cases={}, reason="run_manifest.json is absent")
    try:
        manifest = _read(directory / "run_manifest.json")
        coverage = coverage_suite(directory, write=False)
        accuracy = accuracy_suite(directory, write=False)
        if (coverage["protocol"] != "toy100-v1"
                or accuracy["protocol"] != "toy100-accuracy-v1"
                or coverage["scope"] != "all declared problems"
                or accuracy["scope"] != "all declared problems"
                or set(coverage["problems"]) != set(PROBLEM_NAMES)
                or set(accuracy["problems"]) != set(PROBLEM_NAMES)):
            raise ValueError("100-mode gates do not cover all three declared problems")
        declared = manifest["declared_manifest"]
        if hashlib.sha256(manifest["config_contents"].encode()).hexdigest() != manifest["config_sha256"]:
            raise ValueError("100-mode manifest config hash differs")
        recipe, noise, _ = declared_recipe(declared)
        config_fields = {name: getattr(recipe, name) for name in GLOBAL_RECIPE_FIELDS}
        for name in PROBLEM_NAMES:
            executed = _read(directory / name / "config.json")
            expected_config, _ = resolve_config(manifest["resolved_problem_configs"][name])
            if executed != json.loads(json.dumps(expected_config)):
                raise ValueError(f"executed 100-mode configuration differs: {name}")
            for key in GLOBAL_RECIPE_FIELDS:
                actual = executed[key]
                expected = config_fields[key]
                if isinstance(expected, tuple):
                    expected = list(expected)
                if actual != expected:
                    raise ValueError(f"100-mode global field varies by problem: {name}.{key}")
            if any(executed.get(key, 0.0) != noise[key] for key in NOISE_FIELDS):
                raise ValueError(f"100-mode noise varies by problem: {name}")
        cases = {name: dict(
            status="PASS" if coverage["problems"][name]["passed"]
                             and accuracy["problems"][name]["passed"] else "FAIL",
            passed=bool(coverage["problems"][name]["passed"]
                        and accuracy["problems"][name]["passed"]),
            coverage=coverage["problems"][name]["status"],
            accuracy=accuracy["problems"][name]["status"],
            holdout=accuracy["problems"][name].get("holdout_metrics"),
        ) for name in PROBLEM_NAMES}
        passed = sum(row["passed"] for row in cases.values())
        return dict(status=_status(passed, len(PROBLEM_NAMES), complete=True),
                    passed=passed, required=len(PROBLEM_NAMES), cases=cases,
                    recipe=recipe.to_dict(), noise=noise,
                    config_sha256=manifest["config_sha256"],
                    coverage_status=coverage["status"],
                    accuracy_status=accuracy["status"], reason=None)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return dict(status="INVALID", passed=0, required=len(PROBLEM_NAMES),
                    cases={}, reason=str(error))


def regrade(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    toy = _toy100_rows(output / "toy100")
    expected19 = tuple(job["spec"]["name"] for job in plan())
    candidate = _episode_rows(output / "candidate19", expected19, candidate=True)
    vector = _episode_rows(output / "vector6", VECTOR_NAMES, candidate=True)
    if vector["status"] == "MISSING" and candidate["cases"]:
        subset = {name: candidate["cases"][name] for name in VECTOR_NAMES
                  if name in candidate["cases"]}
        passed = sum(row["passed"] for row in subset.values())
        vector = dict(status=_status(passed, len(VECTOR_NAMES),
                                     complete=len(subset) == len(VECTOR_NAMES)),
                      passed=passed, required=len(VECTOR_NAMES), cases=subset,
                      reason="derived from candidate-19 episodes")
    control = _episode_rows(output / "public19", expected19, candidate=False)
    identity = False
    reason = None
    if "recipe" in toy and "protocol" in candidate:
        candidate_recipe = candidate["protocol"]["global_recipe"]
        candidate_noise = candidate["protocol"]["noise"]
        candidate_noise = {**candidate_noise,
                           "output_noise_warmup": candidate_noise.get("output_noise_warmup", 0.0)}
        identity = (all(toy["recipe"][name] == candidate_recipe[name]
                        for name in GLOBAL_RECIPE_FIELDS)
                    and toy["noise"] == candidate_noise)
        if not identity:
            reason = "100-mode and candidate-19 global recipe or noise fields differ"
    else:
        reason = "complete 100-mode and candidate-19 evidence is required"
    noise_covered = (len(candidate["cases"]) == len(expected19)
                     and all(row["noise_applied"] for row in candidate["cases"].values()))
    if identity and not noise_covered:
        reason = "common noise mechanism lacks a complete host receipt"
    complete = toy["status"] in ("PASS", "FAIL") and candidate["status"] in ("PASS", "FAIL")
    if not complete or not identity or not noise_covered:
        status = "INCOMPLETE"
    elif toy["status"] == candidate["status"] == "PASS":
        status = "PASS"
    else:
        status = "FAIL"
    observed_passes = toy["passed"] + candidate["passed"]
    report = dict(protocol="toy-suite-common22-v1", status=status,
                  observed_passes=observed_passes, required=22,
                  global_recipe_identical=identity, noise_applied_on_all_19=noise_covered,
                  reason=reason, toy100=toy, candidate19=candidate,
                  vector6=vector, public_default19_control=control)
    _write(output / "compatibility.json", report)
    lines = [f"# One-recipe 22-toy gate: {status}", "",
             "A PASS requires the same global optimizer, loss, schedule, and noise settings "
             "on all 22 hosts. The three 100-mode problems must pass both coverage and "
             "accuracy gates; the 19 canonical cases must sustain their frozen live gates. "
             "Architecture and host resource sizes follow the frozen task declarations.", "",
             "| Evidence | Live passes | Status |",
             "| --- | ---: | --- |",
             f"| Three 100-mode problems, coverage + accuracy | {toy['passed']}/3 | {toy['status']} |",
             f"| Same candidate on 19 canonical hosts | {candidate['passed']}/19 | {candidate['status']} |",
             f"| Candidate six-vector full-noise screen | {vector['passed']}/6 | {vector['status']} |",
             f"| Public v3 installed-wheel control | {control['passed']}/19 | {control['status']} |",
             "", f"Global fields identical: **{identity}**. Noise applied on all 19: "
             f"**{noise_covered}**. {reason or ''}", "",
             "## Per-case result", "",
             "| Group | Case | Live | Detail |",
             "| --- | --- | --- | --- |"]
    for name in PROBLEM_NAMES:
        row = toy["cases"].get(name, {})
        lines.append(f"| 100-mode | `{name}` | {row.get('status', 'MISSING')} | "
                     f"coverage {row.get('coverage', 'MISSING')}; accuracy {row.get('accuracy', 'MISSING')} |")
    for name in expected19:
        row = candidate["cases"].get(name, {})
        lines.append(f"| canonical | `{name}` | {row.get('status', 'MISSING')} | "
                     f"noise applied {row.get('noise_applied', False)}; "
                     f"eval {row.get('eval_scope') or 'generated samples'}; "
                     f"final suffix {row.get('passing_suffix', '—')} |")
    lines += ["", "The public v3 control uses its own recipe and cannot supply missing "
              "candidate passes. A single-case or six-case screen is incomplete for 22/22. "
              "EMA is recorded separately and never determines the live gate.", ""]
    (output / "compatibility.md").write_text("\n".join(lines))
    return report


def _run_command(command: list[str], *, cwd: Path, log: Path, env: dict[str, str]):
    with log.open("w") as stream:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    print(json.dumps(dict(event="command_complete", command=command,
                          returncode=process.returncode, log=str(log))), flush=True)
    return process.returncode


def run(config: Path, output: Path, *, with_default_control: bool = False):
    config = config.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("use a new suite output directory")
    output.mkdir(parents=True)
    python = sys.executable
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT))
    commands = [
        ([python, "-u", "-m", "benchmarks.toy100", "run", "--config", str(config),
          "--output", str(output / "toy100"), "--no-render"], ROOT, output / "toy100.log"),
        ([python, "-u", "-m", "benchmarks.toy100.accuracy_gate", "--output",
          str(output / "toy100")], ROOT, output / "accuracy.log"),
        ([python, "-u", "-m", "benchmarks.transfer_suite.toy100_compatibility",
          "--config", str(config), "--all", "--output", str(output / "candidate19")],
         ROOT, output / "candidate19.log"),
    ]
    returns = {}
    for index, (command, cwd, log) in enumerate(commands):
        print(json.dumps(dict(event="command_start", command=command,
                              log=str(log))), flush=True)
        returns[str(index)] = _run_command(command, cwd=cwd, log=log, env=env)
    if with_default_control:
        wheel_dir, site = output / "wheel", output / "site"
        wheel_dir.mkdir()
        returns["wheel"] = _run_command(
            [python, "-m", "pip", "wheel", "--no-deps", ".", "--wheel-dir",
             str(wheel_dir)], cwd=ROOT, log=output / "wheel.log", env=env)
        wheels = sorted(wheel_dir.glob("particlegan-*.whl"))
        if len(wheels) == 1 and returns["wheel"] == 0:
            returns["install"] = _run_command(
                [python, "-m", "pip", "install", "--no-deps", "--target", str(site),
                 str(wheels[0])], cwd=ROOT, log=output / "install.log", env=env)
            installed_env = env | {"PYTHONPATH": str(site) + os.pathsep + str(ROOT)}
            if returns["install"] == 0:
                returns["public19"] = _run_command(
                    [python, "-u", "-m", "benchmarks.transfer_suite.public_default_verification",
                     "--require-installed-root", str(site), "--output", str(output / "public19")],
                    cwd=Path("/tmp"), log=output / "public19.log", env=installed_env)
    _write(output / "command_returns.json", returns)
    return regrade(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="train and grade fresh evidence")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--with-default-control", action="store_true")
    regrade_parser = commands.add_parser("regrade", help="independently grade saved evidence")
    regrade_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.config, args.output,
                     with_default_control=args.with_default_control)
    else:
        result = regrade(args.output)
    print(json.dumps(dict(status=result["status"], observed_passes=result["observed_passes"],
                          required=result["required"], report=str(args.output / "compatibility.md"))),
          flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
