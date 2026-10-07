"""Zero-update CUDA proof of the architecture-only task delta."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.api import task_formulation_context
from experiments.forge.contracts import atomic_json, file_hash
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models

CANDIDATE = "configs/forge/configurations/bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9.json"
BASELINE = Path("/home/martyn/dev/ParticleGAN-tier1-prior-smoke/runs/api/tier1-prior-smoke-v1/mog100-n256/gaussian1d_acquisition/initial-state.pt")


def compare_initial(current, saved):
    assert current["recipe"] == saved["recipe"]
    models = current["trainer"]["models"]
    baseline_models = saved["trainer"]["models"]
    for role in ("G", "prior", "ema_G", "ema_prior"):
        assert state_digest(models[role]) == state_digest(baseline_models[role]), role
    changed = []
    for name in models["D"].keys() | baseline_models["D"].keys():
        if name not in models["D"] or name not in baseline_models["D"] or state_digest(models["D"][name]) != state_digest(baseline_models["D"][name]):
            changed.append(name)
    assert sorted(changed) == ["freqs", "net.0.bias", "net.0.weight"], changed
    assert list(models["D"]["net.0.weight"].shape) == [32, 1]
    assert list(baseline_models["D"]["net.0.weight"].shape) == [32, 5]
    changed_streams = []
    for name in saved["streams"]["states"]:
        assert name in current["streams"]["states"]
        if not torch.equal(current["streams"]["states"][name], saved["streams"]["states"][name]):
            changed_streams.append(name)
    assert changed_streams == ['["init","discriminator","construction","cpu"]'], changed_streams
    assert state_digest(current["trainer"]["streams"]) == state_digest(saved["trainer"]["streams"])
    assert state_digest(current["trainer"]["optimizers"]) == state_digest(saved["trainer"]["optimizers"])
    assert current["trainer"]["initial_lrs"] == saved["trainer"]["initial_lrs"]
    return dict(passed=True, training_updates=0, baseline_sha256=file_hash(BASELINE),
                generator_sha256=state_digest(models["G"]), prior_sha256=state_digest(models["prior"]),
                changed_critic_tensors=sorted(changed), changed_constructor_streams=changed_streams,
                initial_bias_policy="public fan-in uniform; first-layer fan-in changes 5 to 1",
                train_streams_identical=True, exact_optimizer_and_rate_match=True)


@reproducible_execution
def run(*, device):
    assert torch.device(device).type == "cuda" and torch.cuda.is_available()
    task = json.loads((ROOT / "configs/forge/tasks/gaussian1d_acquisition.json").read_text())
    task = deepcopy(task)
    task["execution"]["prior"]["sigma"] = .1
    task["execution"]["host_definition"]["fourier"] = 0
    candidate = json.loads((ROOT / CANDIDATE).read_text())
    context = task_formulation_context(candidate, task, {"seed": 0}, device=device, root=ROOT)
    generator, critic = build_vector_models(context, task["execution"]["host_definition"])
    trainer = context.build_trainer(generator, critic, max_steps=1000)
    context.streams.generator("data", component="target", purpose="training", device="cpu")
    context.streams.generator("eval", component="live", purpose="samples")
    assert all(p.device.type == "cuda" for model in (generator, critic, trainer.prior) for p in model.parameters())
    result = compare_initial(context.state_dict(), torch.load(BASELINE, weights_only=True, map_location="cpu"))
    result.update(device=device, gpu=torch.cuda.get_device_name(device),
                  parameter_counts={"G": sum(p.numel() for p in generator.parameters()),
                                    "D": sum(p.numel() for p in critic.parameters())})
    assert result["parameter_counts"] == {"G": 1185, "D": 1153}
    return result


if __name__ == "__main__":
    result = run(device="cuda:0")
    atomic_json(ROOT / "reports/forge/gaussian-no-fourier/initial-proof.json", result)
    print(json.dumps(result, sort_keys=True))
