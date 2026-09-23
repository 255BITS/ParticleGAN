"""Guard the common-recipe gate against mixed settings and forged evidence."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math

import pytest

from benchmarks import toy_suite
from benchmarks.toy100.models import linear_input_noise
from benchmarks.toy100.problems import PROBLEM_NAMES
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.public_default_verification import (
    GLOBAL_RECIPE_FIELDS, declared_spec, host_recipe, load_declaration,
)
from benchmarks.transfer_suite.toy100_compatibility import output_noise_at
from particlegan import get_recipe, learning_rate_scale


def _write_candidate_episode(directory, mutate=lambda record: None):
    jobs, profile = load_declaration()
    job = next(job for job in jobs if job["spec"]["name"] == "vector_two_broad")
    base = get_recipe()
    spec, _, variant = declared_spec(job, profile, base)
    steps = spec["steps"]
    recipe = base.to_dict()
    noise = dict(output_noise_std=0.029, input_noise_std=0.5,
                 input_noise_anneal_end=0.1, output_noise_warmup=0.2)
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
    verdict = dict(status="PASS", passed=True, convergence=dict(passing_suffix=5))
    record = dict(
        name=spec["name"], original_spec=deepcopy(job["spec"]), spec=spec,
        discriminator_variant=variant,
        host_recipe=host_recipe(base, spec).to_dict(),
        source_sha256={"frozen": "digest"}, recipe=deepcopy(recipe), noise=deepcopy(noise),
        applied=[dict(role=role, lr=base.lr * multiplier, betas=list(base.betas),
                      parameters=1, optimizer="Adam")
                 for role, multiplier in (("g", 1), ("prior", base.prior_lr_mult),
                                          ("d", base.d_lr_mult))],
        noise_receipt=dict(step_calls=steps, output_module="OutputNoise",
                           input_module="InputNoise", input_nonzero_steps=1,
                           output_sigma_first=0.0, output_sigma_last=0.029),
        noise_applied=True, result=result, verdict=verdict,
    )
    mutate(record)
    directory.mkdir()
    (directory / "episodes").mkdir()
    artifact = "episodes/vector_two_broad.json.gz"
    raw = (json.dumps(record, sort_keys=True) + "\n").encode()
    (directory / artifact).write_bytes(gzip.compress(raw, mtime=0))
    (directory / "protocol.json").write_text(json.dumps(dict(
        source_sha256={"frozen": "digest"}, global_recipe=recipe, noise=noise,
        jobs=[job], frozen_discriminators=profile["discriminators"],
    )))
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
