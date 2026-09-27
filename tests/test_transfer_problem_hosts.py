"""Migrated transfer hosts run on the shared runner and regrade from optimizer state."""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from benchmarks.gan_v3 import gan_v3_recipe
from benchmarks.toy_runner import Networks, ToyProblem
from benchmarks.transfer_suite import problem_hosts
from particlegan import get_recipe, init

NOISE = dict(output_noise_std=0.029, input_noise_std=0.5, input_noise_anneal_end=0.1,
             output_noise_warmup=0.2)
POLICY = dict(network_lr_horizon_cap=12, network_lr_floor=0.01)
SPEC = dict(name="tiny", runner="legacy", steps=24, thresholds=[["mean_err", "<=", 10.0]])


class Tiny(ToyProblem):
    name = "tiny"

    def recipe(self):
        return get_recipe(z_dim=2, num_particles=16, batch_size=32, total_steps=24)

    def networks(self, recipe, seed):
        generator = init.deterministic_orthogonal_(nn.Linear(2, 2), seed=seed)
        critic = init.deterministic_orthogonal_(nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 1)),
                                                seed=seed + 1)
        return Networks(generator=generator, critics=critic)

    def real(self, n, stream):
        return torch.randn(n, 2, generator=stream) + 1.0

    def metrics(self, model):
        return {"mean_err": float((model.sample(256).x.mean(0) - 1.0).norm())}

    def verdict(self, metrics):
        return "PASS" if metrics["mean_err"] <= 10.0 else "FAIL"


@pytest.fixture
def tiny_host(monkeypatch):
    monkeypatch.setattr(problem_hosts, "problem_class", lambda name: Tiny if name == "tiny" else None)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_mode_hold_is_discovered_as_a_problem_host():
    assert problem_hosts.problem_class("mode_hold").name == "mode_hold"
    assert problem_hosts.problem_class("vector_two_broad") is None


def test_problem_host_uses_common_recipe_and_regrades_from_schedule_state(tiny_host):
    base = gan_v3_recipe()
    result, context = problem_hosts.run_problem(SPEC, base, NOISE, model_policy=POLICY)
    assert [p["step"] for p in result["observations"]] == list(range(1, 25))
    executed = context["executed_recipe"]
    assert (executed["lr"], executed["total_steps"], executed["batch_size"]) == (base.lr, 24, 32)
    assert (executed["network_lr_horizon_cap"], executed["network_lr_floor"]) == (12, 0.01)
    assert executed["output_noise_std"] == 0.029 and context["noise_receipt"]["step_calls"] == 24
    record = json.loads(json.dumps(dict(spec=SPEC, executed_recipe=executed, result=result)))
    problem_hosts.check_receipts(record, base, NOISE, POLICY)
    tampers = [
        (lambda r: r["result"]["optimizers"][0]["lr_schedule"].update(completed_steps=23), "every update"),
        (lambda r: r["result"]["optimizers"][1]["groups"][0].update(base_lr=0.1), "base rate"),
        (lambda r: r["result"]["optimizers"][0].update(optimizer="Adam"), "not recipe-built"),
        (lambda r: r["executed_recipe"].update(lr=0.1), "declared recipe"),
    ]
    for tamper, message in tampers:
        bad = deepcopy(record)
        tamper(bad)
        with pytest.raises(ValueError, match=message):
            problem_hosts.check_receipts(bad, base, NOISE, POLICY)
    with pytest.raises(ValueError, match="declared recipe"):
        problem_hosts.check_receipts(record, base, dict(NOISE, input_noise_std=0.4), POLICY)


def test_problem_host_rejects_noise_options_the_runner_lacks(tiny_host):
    with pytest.raises(ValueError, match="output_noise_learnable"):
        problem_hosts.problem_recipe(Tiny(), gan_v3_recipe(), dict(NOISE, output_noise_learnable=True))
