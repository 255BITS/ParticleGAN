"""Guard the common-recipe gate against mixed settings and forged evidence."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import tarfile

import pytest

from benchmarks import toy_suite
from benchmarks.toy100.models import linear_input_noise
from benchmarks.toy100.schedule import policy_multipliers
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.toy100.train import resolve_config
from benchmarks.toy100.train import train as train_toy100
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
)
from benchmarks.transfer_suite.toy100_compatibility import (
    declared_model_policy, declared_recipe, output_noise_at,
    run_image, run_vector, setup_image, setup_vector,
)
from benchmarks.transfer_suite import vector_tasks
from lib.toy_models import SimpleMLPGenerator
from particlegan import get_recipe, learning_rate_scale


def _write_candidate_episode(directory, mutate=lambda record: None, *, learned=False,
                             cap=None):
    jobs, profile = load_declaration()
    job = next(job for job in jobs if job["spec"]["name"] == "vector_two_broad")
    base = get_recipe()
    spec, _, variant = declared_spec(job, profile, base)
    steps = spec["steps"]
    recipe = base.to_dict()
    noise = dict(output_noise_std=0.029, input_noise_std=0.5,
                 input_noise_anneal_end=0.1, output_noise_warmup=0.2)
    if learned:
        noise["output_noise_learnable"] = True
    source_bytes = b"frozen benchmark source"
    noise_source_bytes = b"frozen noise source"
    source_hashes = dict(frozen=hashlib.sha256(source_bytes).hexdigest(),
                         **{"benchmarks/toy100/models.py": hashlib.sha256(
                             noise_source_bytes).hexdigest()})
    policy_source_bytes = {
        "benchmarks/toy100/schedule.py": b"frozen shared policy schedule",
        "benchmarks/toy100/train.py": b"frozen native trainer",
        "benchmarks/toy100/config.py": b"frozen manifest parser",
        "benchmarks/toy100/__main__.py": b"frozen native CLI",
    }
    if cap is not None:
        source_hashes.update({name: hashlib.sha256(contents).hexdigest()
                              for name, contents in policy_source_bytes.items()})
    result = dict(
        observations=[dict(step=math.ceil(i * steps / 24)) for i in range(1, 25)],
        actions=[dict(
            step=step + 1,
            multiplier=learning_rate_scale(step, steps, base.lr_anneal_start, base.lr_floor),
            lr_g=base.lr * learning_rate_scale(step, steps, base.lr_anneal_start, base.lr_floor),
            lr_prior=base.lr * base.prior_lr_mult * learning_rate_scale(
                step, steps, base.lr_anneal_start, base.lr_floor),
            lr_d=base.lr * base.d_lr_mult * learning_rate_scale(
                step, steps, base.lr_anneal_start, base.lr_floor),
            input_sigma=linear_input_noise(0.5, step, steps, 0.1),
            output_sigma=output_noise_at(0.029, step, steps, 0.2),
        ) for step in range(steps)],
        update_counts=dict(g=steps, d=steps),
    )
    if cap is not None:
        for completed, action in enumerate(result["actions"], start=1):
            network, prior = policy_multipliers(
                completed - 1, steps, base.lr_anneal_start, base.lr_floor, cap,
            )
            action.update(network_multiplier=network, prior_multiplier=prior,
                          network_lr_horizon_cap=cap,
                          lr_g=base.lr * network,
                          lr_prior=base.lr * base.prior_lr_mult * prior,
                          lr_d=base.lr * base.d_lr_mult * network)
        model_policy = {"network_lr_horizon_cap": cap}
    verdict = dict(status="PASS", passed=True, convergence=dict(passing_suffix=5))
    record = dict(
        name=spec["name"], original_spec=deepcopy(job["spec"]), spec=spec,
        discriminator_variant=variant,
        host_recipe=host_recipe(base, spec).to_dict(),
        source_sha256=source_hashes, recipe=deepcopy(recipe), noise=deepcopy(noise),
        applied=[dict(role=role, lr=base.lr * multiplier, betas=list(base.betas),
                      parameters=1, optimizer="Adam")
                 for role, multiplier in (("g", 1), ("prior", base.prior_lr_mult),
                                          ("d", base.d_lr_mult))],
        noise_receipt=dict(step_calls=steps, output_module="OutputNoise",
                           input_module="InputNoise",
                           input_nonzero_steps=sum(linear_input_noise(
                               0.5, step, steps, 0.1) > 0 for step in range(steps)),
                           output_nonzero_steps=steps - 1,
                           output_sigma_first=0.0, output_sigma_last=0.029),
        noise_applied=True, result=result, verdict=verdict,
    )
    if cap is not None:
        record["model_policy"] = deepcopy(model_policy)
    if learned:
        cfg = vector_tasks.resolve(spec)
        bare = SimpleMLPGenerator(cfg["z_dim"], cfg["hidden"], cfg["layers"], 2)
        base_parameters = sum(parameter.numel() for parameter in bare.parameters())
        record["shapes"] = {"generator_parameters": base_parameters + 1}
        record["applied"][0]["parameters"] = base_parameters + 1
        effective_trace = [
            output_noise_at(.029, step, steps, .2) / .029 * (.029 - .002 * step / steps)
            for step in range(steps)
        ]
        for action, effective in zip(result["actions"], effective_trace):
            action["output_sigma_effective"] = effective
        result["observations"][-1].update(output_sigma_live=.027,
                                           output_sigma_ema=.028)
        record["noise_receipt"].update(
            output_noise_learnable=True,
            output_scale_parameter_count=1,
            output_scale_optimizer_owned=True,
            generator_base_parameters=base_parameters,
            generator_total_parameters=base_parameters + 1,
            output_scale_initial=.029,
            output_scale_final=.027,
            output_scale_ema_final=.028,
            output_sigma_final_evaluation=.029,
            output_sigma_effective_first=effective_trace[0],
            output_sigma_effective_last=effective_trace[-1],
            output_sigma_effective_final_evaluation=.027,
            output_sigma_effective_ema_final_evaluation=.028,
            output_sigma_effective_step_trace=effective_trace,
        )
    mutate(record)
    directory.mkdir()
    (directory / "episodes").mkdir()
    artifact = "episodes/vector_two_broad.json.gz"
    raw = (json.dumps(record, sort_keys=True) + "\n").encode()
    (directory / artifact).write_bytes(gzip.compress(raw, mtime=0))
    config_bytes = (json.dumps(dict(
        name="gan_v3", output_noise_std=0.029, input_noise_std=0.5,
        input_noise_anneal_end=0.1, output_noise_warmup=0.2,
        **({"output_noise_learnable": True} if learned else {}),
        **({"network_lr_horizon_cap": cap} if cap is not None else {}),
    )) + "\n").encode()
    (directory / "candidate.json").write_bytes(config_bytes)
    (directory / "noise_source.py").write_bytes(noise_source_bytes)
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("frozen")
        member.size = len(source_bytes)
        archive.addfile(member, io.BytesIO(source_bytes))
        if cap is not None:
            for name, contents in policy_source_bytes.items():
                member = tarfile.TarInfo(name)
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
    protocol = dict(
        source_sha256=source_hashes, global_recipe=recipe, noise=noise,
        noise_source_sha256=source_hashes["benchmarks/toy100/models.py"],
        config_file="candidate.json",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        ignored_toy100_resource_overrides={},
        jobs=[job], frozen_discriminators=profile["discriminators"],
    )
    if cap is not None:
        protocol["model_policy"] = model_policy
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "index.json").write_text(json.dumps(dict(records=[dict(
        name=spec["name"], artifact=artifact,
        uncompressed_sha256=hashlib.sha256(raw).hexdigest(), verdict=verdict,
    )])))
    return spec["name"]


@pytest.mark.parametrize("tamper,expected", [
    (lambda record: record["recipe"].update(lr=0.001), "global fields differ"),
    (lambda record: record["spec"]["thresholds"][0].__setitem__(2, 1.0),
     "executed spec, budget, threshold"),
    (lambda record: record["spec"].update(steps=2400),
     "executed spec, budget, threshold"),
    (lambda record: record["result"]["observations"][0].update(step=1),
     "frozen 24-checkpoint schedule differs"),
    (lambda record: record["result"]["actions"][1].update(input_sigma=0.0),
     "input noise schedule differs"),
    (lambda record: record["noise_receipt"].update(input_nonzero_steps=1),
     "input noise duration differs"),
    (lambda record: record["result"]["actions"][1].update(output_sigma=0.0),
     "output warmup differs"),
    (lambda record: record["applied"][0].update(lr=0.0001),
     "optimizer rate differs"),
    (lambda record: record.update(noise_applied=False), "noise claim differs"),
])
def test_candidate_episode_rejects_mixed_recipe_or_forged_noise(
    tmp_path, monkeypatch, tamper, expected,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    name = _write_candidate_episode(tmp_path / "candidate", tamper)
    grade = toy_suite._episode_rows(tmp_path / "candidate", (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert expected in grade["reason"]


@pytest.mark.parametrize("file_name,expected", [
    ("candidate.json", "saved candidate config differs"),
    ("noise_source.py", "saved noise source differs"),
    ("source.tar.gz", "saved source archive differs"),
])
def test_candidate_episode_binds_saved_config_and_sources(
    tmp_path, monkeypatch, file_name, expected,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory)
    assert toy_suite._episode_rows(directory, (name,), candidate=True)["status"] == "PASS"
    path = directory / file_name
    if file_name == "source.tar.gz":
        with tarfile.open(path, "w:gz") as archive:
            payload = b"altered benchmark source"
            member = tarfile.TarInfo("frozen")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    else:
        path.write_bytes(b"forged source or config")
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert expected in grade["reason"]


def test_candidate_episode_rejects_omitted_nonzero_output_warmup(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol["noise"].pop("output_noise_warmup")
    (directory / "protocol.json").write_text(json.dumps(protocol))
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "saved candidate config does not resolve to declared recipe" in grade["reason"]


def test_candidate_episode_rejects_duplicate_source_archive_member(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory)
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for _ in range(2):
            payload = b"frozen benchmark source"
            member = tarfile.TarInfo("frozen")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "source archive members differ" in grade["reason"]


@pytest.mark.parametrize("tamper,expected", [
    (lambda record: record["model_policy"].update(network_lr_horizon_cap=41),
     "global fields differ"),
    (lambda record: record["result"]["actions"][80].update(lr_g=.99),
     "trainer LR action differs"),
    (lambda record: record["result"]["actions"][80].update(lr_prior=.99),
     "trainer LR action differs"),
    (lambda record: record["result"]["actions"][80].update(network_multiplier=.99),
     "trainer LR action differs"),
])
def test_candidate_horizon_policy_binds_each_network_and_prior_rate(
    tmp_path, monkeypatch, tamper, expected,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory, cap=40)
    assert toy_suite._episode_rows(directory, (name,), candidate=True)["status"] == "PASS"
    directory = tmp_path / "tampered"
    _write_candidate_episode(directory, tamper, cap=40)
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert expected in grade["reason"]


def test_candidate_horizon_policy_binds_config_and_archived_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory, cap=40)
    assert toy_suite._episode_rows(directory, (name,), candidate=True)["status"] == "PASS"
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol["model_policy"]["network_lr_horizon_cap"] = 41
    (directory / "protocol.json").write_text(json.dumps(protocol))
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "saved candidate config does not resolve" in grade["reason"]
    protocol["model_policy"]["network_lr_horizon_cap"] = 40
    (directory / "protocol.json").write_text(json.dumps(protocol))
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for member_name, contents in (("frozen", b"frozen benchmark source"),
                                      ("benchmarks/toy100/schedule.py", b"tampered"),
                                      ("benchmarks/toy100/train.py", b"frozen native trainer"),
                                      ("benchmarks/toy100/config.py", b"frozen manifest parser"),
                                      ("benchmarks/toy100/__main__.py", b"frozen native CLI")):
            member = tarfile.TarInfo(member_name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "saved source archive differs" in grade["reason"]


def test_native_affine_policy_regrade_rejects_scratch_and_tampered_receipts(tmp_path):
    config = dict(problem="grid100", seed=31, steps=6, device="cpu",
                  z_dim=2, num_particles=32, batch_size=8,
                  d_hidden=8, n_hidden=1, fourier=3,
                  eval_samples=128, snapshot_samples=16,
                  eval_interval=3, snapshot_interval=3, log_interval=3,
                  threads=1, output_noise_std=.029, input_noise_std=.5,
                  toy100_model="affine_square_v1", network_lr_horizon_cap=3)
    directory = tmp_path / "native"
    summary = train_toy100(config, directory)
    resolved = summary["config"]
    policy = declared_model_policy(config)
    sources = toy_suite._check_toy100_policy(directory, summary, resolved, policy)
    assert "benchmarks/toy100/schedule.py" in sources

    event_path = directory / "events.jsonl"
    original_events = event_path.read_text()
    events = [json.loads(line) for line in original_events.splitlines()]
    next(row for row in events if row.get("event") == "train")["lr_prior"] = .99
    event_path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    with pytest.raises(ValueError, match="policy action differs"):
        toy_suite._check_toy100_policy(directory, summary, resolved, policy)
    event_path.write_text(original_events)

    original_card = deepcopy(summary["model_policy"])
    summary["model_policy"]["generator_initial_weight"] = [[2.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="affine initialization receipt differs"):
        toy_suite._check_toy100_policy(directory, summary, resolved, policy)
    summary["model_policy"] = original_card

    provenance_path = directory / "provenance.json"
    original_provenance = provenance_path.read_text()
    scratch = json.loads(original_provenance)
    scratch["trainer_factory"] = "scratch monkeypatch"
    scratch["model_options"] = {"shared_gate_eligible": False}
    provenance_path.write_text(json.dumps(scratch))
    with pytest.raises(ValueError, match="scratch trainer override"):
        toy_suite._check_toy100_policy(directory, summary, resolved, policy)
    provenance_path.write_text(original_provenance)

    archive_path = directory / "source.tar.gz"
    archive = archive_path.read_bytes()
    archive_path.write_bytes(archive + b"tampered")
    with pytest.raises(ValueError, match="archive hash differs"):
        toy_suite._check_toy100_policy(directory, summary, resolved, policy)


def test_native_policy_with_learned_noise_regrades_from_archive_after_relocation(
    tmp_path, monkeypatch,
):
    config = dict(problem="grid100", seed=31, steps=6, device="cpu",
                  z_dim=2, num_particles=32, batch_size=8,
                  d_hidden=8, n_hidden=1, fourier=3,
                  eval_samples=128, snapshot_samples=16,
                  eval_interval=3, snapshot_interval=3, log_interval=3,
                  threads=1, output_noise_std=.029,
                  output_noise_learnable=True, output_noise_warmup=.2,
                  input_noise_std=.5,
                  toy100_model="affine_square_v1", network_lr_horizon_cap=3)
    directory = tmp_path / "learned-native"
    summary = train_toy100(config, directory)
    resolved = summary["config"]
    toy_suite._check_toy100_policy(
        directory, summary, resolved, declared_model_policy(config),
    )
    monkeypatch.setattr(toy_suite, "ROOT", tmp_path / "no-live-source-tree")
    toy_suite._check_toy100_learned_noise(
        directory, summary, resolved, policy_archive_verified=True,
    )


@pytest.mark.parametrize("name", ["vector_two_broad", "img_stripes2"])
def test_transfer_trainer_hosts_record_actual_capped_network_rates(name, monkeypatch):
    recipe, noise, _ = declared_recipe({"network_lr_horizon_cap": 8})
    jobs, profile = load_declaration()
    job = next(row for row in jobs if row["spec"]["name"] == name)
    spec, card, _ = declared_spec(job, profile, recipe)
    spec["steps"] = 24
    monkeypatch.setattr(vector_tasks, "EVAL_SAMPLES", 256)
    policy = {"network_lr_horizon_cap": 8}
    result, context = (run_vector(spec, card, recipe, noise, model_policy=policy)
                       if name.startswith("vector") else
                       run_image(spec, recipe, noise, model_policy=policy))
    assert len(result["actions"]) == 24
    network, prior = policy_multipliers(23, 24, recipe.lr_anneal_start,
                                        recipe.lr_floor, 8)
    assert network < prior
    action = result["actions"][-1]
    assert action["step"] == 24
    assert action["network_multiplier"] == network
    assert action["prior_multiplier"] == prior
    assert action["lr_g"] == pytest.approx(recipe.lr * network)
    assert action["lr_prior"] == pytest.approx(recipe.lr * recipe.prior_lr_mult * prior)
    assert action["lr_d"] == pytest.approx(recipe.lr * recipe.d_lr_mult * network)
    assert context["trainer"].completed_steps == 24


@pytest.mark.parametrize("tamper,expected", [
    (lambda record: record["noise_receipt"].update(output_scale_optimizer_owned=False),
     "lacks one G-owned parameter"),
    (lambda record: record["noise_receipt"].update(generator_base_parameters=1),
     "wrapper parameter count differs"),
    (lambda record: record["shapes"].update(generator_parameters=1),
     "native generator optimizer or shape count differs"),
    (lambda record: record["result"]["actions"][1].update(output_sigma_effective=.99),
     "sigma action differs"),
    (lambda record: record["noise_receipt"].update(output_scale_final=0),
     "learned output scale is invalid"),
    (lambda record: record["noise_receipt"].update(output_sigma_final_evaluation=.01),
     "learned output base schedule differs"),
    (lambda record: record["noise_receipt"].update(
        output_sigma_effective_ema_final_evaluation=.99),
     "learned EMA output sigma differs"),
    (lambda record: record["result"]["observations"][-1].update(output_sigma_live=.99),
     "learned output evaluation differs"),
])
def test_learned_candidate_receipt_binds_scale_and_actual_noise(
    tmp_path, monkeypatch, tamper, expected,
):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory, tamper, learned=True)
    grade = toy_suite._episode_rows(directory, (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert expected in grade["reason"]


def test_learned_candidate_receipt_accepts_valid_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    directory = tmp_path / "candidate"
    name = _write_candidate_episode(directory, learned=True)
    assert toy_suite._episode_rows(directory, (name,), candidate=True)["status"] == "PASS"


@pytest.mark.parametrize("name", ["vector_two_broad", "img_stripes2"])
def test_native_transfer_hosts_register_exactly_one_g_noise_scalar(name):
    recipe, noise, _ = declared_recipe(dict(
        output_noise_std=.029, output_noise_learnable=True,
        output_noise_warmup=.2, input_noise_std=.5,
        input_noise_anneal_end=.1,
    ))
    jobs, profile = load_declaration()
    job = next(job for job in jobs if job["spec"]["name"] == name)
    spec, card, _ = declared_spec(job, profile, recipe)
    context = (setup_vector(spec, card, recipe, noise) if name.startswith("vector")
               else setup_image(spec, recipe, noise))
    trainer = context["trainer"]
    scalar = trainer.G.output_scale.raw_scale
    assert context["applied"][0]["parameters"] == context["generator_base_parameters"] + 1
    assert context["shapes"]["generator_parameters"] == context["applied"][0]["parameters"]
    assert sum(p is scalar for group in trainer.opt_g.param_groups
               for p in group["params"]) == 1
    assert all(p is not scalar for group in trainer.opt_d.param_groups
               for p in group["params"])
    assert trainer.ema_G.output_scale.raw_scale is not scalar


def _write_learned_toy100_evidence(directory):
    directory.mkdir()
    config = dict(problem="grid100", steps=4, output_noise_std=.029,
                  output_noise_warmup=.5, output_noise_learnable=True)
    hashes = {
        source: hashlib.sha256((toy_suite.ROOT / source).read_bytes()).hexdigest()
        for source in toy_suite.NATIVE_SOURCE_FILES
    }
    provenance = {"source_sha256": hashes}
    receipt = dict(initial_std=.029, added_trainable_parameters=1,
                   parameter="G.output_scale.raw_scale", optimizer="G",
                   optimizer_group=0, initial_output_sigma_live=0.,
                   initial_output_sigma_ema=0., final_output_sigma_live=.027,
                   final_output_sigma_ema=.028, final_base_std_live=.027,
                   final_base_std_ema=.028)
    summary = dict(eval_steps=[0, 4], learnable_output_noise=receipt,
                   provenance=provenance)
    events = [dict(event="eval", step=0, model=model, output_sigma=0.)
              for model in ("live", "ema")]
    events.extend(dict(event="train", step=step,
                       output_sigma_live=base / .029 * .027,
                       output_sigma_ema=base / .029 * .028)
                  for step in range(1, 5)
                  for base in [output_noise_at(.029, step, 4, .5)])
    events.extend(dict(event="eval", step=4, model=model,
                       output_sigma=.027 if model == "live" else .028)
                  for model in ("live", "ema"))
    (directory / "provenance.json").write_text(json.dumps(provenance))
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
    )
    return config, summary, events


@pytest.mark.parametrize("alter,expected", [
    (lambda summary, events: summary["learnable_output_noise"].update(optimizer="D"),
     "not bound to G optimizer"),
    (lambda summary, events: summary["learnable_output_noise"].update(
        final_base_std_live=.01), "output sigma endpoint differs"),
    (lambda summary, events: events[3].update(output_sigma_live=-1),
     "invalid learned training sigma"),
    (lambda summary, events: events[-2].update(output_sigma=.1),
     "learned evaluation sigma differs"),
    (lambda summary, events: summary["provenance"]["source_sha256"].update(
        {"benchmarks/toy100/train.py": "0" * 64}), "source provenance differs"),
])
def test_toy100_learned_evidence_rejects_tampering(tmp_path, alter, expected):
    directory = tmp_path / "grid100"
    config, summary, events = _write_learned_toy100_evidence(directory)
    alter(summary, events)
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
    )
    with pytest.raises(ValueError, match=expected):
        toy_suite._check_toy100_learned_noise(directory, summary, config)


def test_toy100_learned_evidence_accepts_complete_trace(tmp_path):
    directory = tmp_path / "grid100"
    config, summary, _ = _write_learned_toy100_evidence(directory)
    toy_suite._check_toy100_learned_noise(directory, summary, config)


def test_toy100_learned_evidence_binds_current_native_source(tmp_path):
    directory = tmp_path / "grid100"
    config, summary, _ = _write_learned_toy100_evidence(directory)
    summary["provenance"]["source_sha256"]["benchmarks/toy100/train.py"] = "0" * 64
    (directory / "provenance.json").write_text(json.dumps(summary["provenance"]))
    with pytest.raises(ValueError, match="source hash differs"):
        toy_suite._check_toy100_learned_noise(directory, summary, config)


def test_three_problem_gate_requires_each_learned_receipt(tmp_path, monkeypatch):
    declared = dict(steps=4, output_noise_std=.029,
                    output_noise_warmup=.5, output_noise_learnable=True)
    config_contents = json.dumps(declared)
    resolved_configs = {}
    for name in PROBLEM_NAMES:
        directory = tmp_path / name
        _write_learned_toy100_evidence(directory)
        resolved, _ = resolve_config({**declared, "problem": name})
        resolved_configs[name] = resolved
        (directory / "config.json").write_text(json.dumps(resolved))
    (tmp_path / "run_manifest.json").write_text(json.dumps(dict(
        declared_manifest=declared, resolved_problem_configs=resolved_configs,
        config_contents=config_contents,
        config_sha256=hashlib.sha256(config_contents.encode()).hexdigest(),
    )))
    passed_gate = dict(protocol="toy100-v1", scope="all declared problems",
                       problems={name: dict(passed=True, status="PASS")
                                 for name in PROBLEM_NAMES}, status="PASS")
    passed_accuracy = deepcopy(passed_gate)
    passed_accuracy["protocol"] = "toy100-accuracy-v1"
    monkeypatch.setattr(toy_suite, "coverage_suite", lambda *args, **kwargs: passed_gate)
    monkeypatch.setattr(toy_suite, "accuracy_suite", lambda *args, **kwargs: passed_accuracy)
    assert toy_suite._toy100_rows(tmp_path)["status"] == "PASS"
    summary_path = tmp_path / PROBLEM_NAMES[-1] / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.pop("learnable_output_noise")
    summary_path.write_text(json.dumps(summary))
    grade = toy_suite._toy100_rows(tmp_path)
    assert grade["status"] == "INVALID"
    assert "learned output-scale receipt is absent" in grade["reason"]


def test_common_22_gate_requires_exact_noise_and_recipe_identity(tmp_path, monkeypatch):
    names = tuple(job["spec"]["name"] for job in plan())
    recipe = {field: 1 for field in GLOBAL_RECIPE_FIELDS}
    noise = dict(output_noise_std=0.029, input_noise_std=0.5,
                 input_noise_anneal_end=0.1, output_noise_warmup=0.2)
    toy = dict(status="PASS", passed=3, required=3, recipe=recipe, noise=noise,
               cases={name: dict(status="PASS", coverage="PASS", accuracy="PASS")
                      for name in PROBLEM_NAMES})
    candidate = dict(status="PASS", passed=19, required=19,
                     protocol=dict(global_recipe=deepcopy(recipe), noise=deepcopy(noise)),
                     cases={name: dict(status="PASS", passed=True, noise_applied=True)
                            for name in names})
    monkeypatch.setattr(toy_suite, "_toy100_rows", lambda directory: toy)
    candidate_data = candidate
    def fake_episode_rows(directory, expected, *, candidate):
        if directory.name == "candidate19":
            return candidate_data
        return dict(status="MISSING", passed=0, cases={})

    monkeypatch.setattr(toy_suite, "_episode_rows", fake_episode_rows)
    assert toy_suite.regrade(tmp_path / "same")["status"] == "PASS"
    candidate_data = deepcopy(candidate)
    candidate_data["protocol"]["global_recipe"]["lr"] = 2
    assert toy_suite.regrade(tmp_path / "mixed-recipe")["status"] == "INCOMPLETE"
    candidate_data = deepcopy(candidate)
    candidate_data["protocol"]["noise"]["output_noise_warmup"] = 0.5
    assert toy_suite.regrade(tmp_path / "mixed-noise")["status"] == "INCOMPLETE"
    candidate_data = deepcopy(candidate)
    candidate_data["protocol"]["noise"]["output_noise_learnable"] = True
    assert toy_suite.regrade(tmp_path / "mixed-learned-noise")["status"] == "INCOMPLETE"
    candidate_data = deepcopy(candidate)
    candidate_data["protocol"]["noise"]["output_noise_learnable"] = False
    assert toy_suite.regrade(tmp_path / "explicit-fixed-noise")["status"] == "PASS"
    candidate_data = deepcopy(candidate)
    candidate_data["protocol"]["model_policy"] = {"network_lr_horizon_cap": 1600}
    assert toy_suite.regrade(tmp_path / "mixed-network-policy")["status"] == "INCOMPLETE"


def test_common_22_identity_normalizes_real_recipe_tuple_after_json(tmp_path, monkeypatch):
    # The saved CI protocol is JSON, while declared_recipe().to_dict() keeps
    # Adam betas as a tuple in memory. Their values still describe one recipe.
    candidate_config = json.loads((
        toy_suite.ROOT / "configs/toy100/shared_candidate.json"
    ).read_text())
    recipe, noise, _ = declared_recipe(candidate_config)
    live_recipe = recipe.to_dict()
    protocol_recipe = json.loads(json.dumps(live_recipe))
    assert isinstance(live_recipe["betas"], tuple)
    assert isinstance(protocol_recipe["betas"], list)
    names = tuple(job["spec"]["name"] for job in plan())
    toy = dict(status="PASS", passed=3, recipe=live_recipe, noise=noise,
               cases={name: dict(status="PASS") for name in PROBLEM_NAMES})
    candidate_row = dict(status="PASS", passed=19,
                         protocol=dict(global_recipe=protocol_recipe, noise=noise),
                         cases={name: dict(status="PASS", passed=True,
                                           noise_applied=True)
                                for name in names})
    monkeypatch.setattr(toy_suite, "_toy100_rows", lambda directory: toy)
    def fake_episode_rows(directory, expected, *, candidate):
        return (candidate_row if directory.name == "candidate19" else
                dict(status="MISSING", passed=0, cases={}))

    monkeypatch.setattr(toy_suite, "_episode_rows", fake_episode_rows)
    grade = toy_suite.regrade(tmp_path / "json-roundtrip")
    assert grade["status"] == "PASS"
    assert grade["global_recipe_identical"] is True
