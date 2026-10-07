"""GPU controls verify initial identity, immobile prior and exact continuation."""
import pytest
import torch

from benchmarks.toy_audit import gaussian_frozen_prior as study
from benchmarks.toy_audit.reproducibility import reproducible_execution
from experiments.forge.state import state_digest, require_optimizer_steps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; no CPU fallback")


@reproducible_execution
def probe(arm, task, *, device):
    context, trainer, _ = study.build(arm, task, device)
    assert study.initial_proof(context, task, study.declaration())["matched"]
    assert not list(trainer.prior.parameters())
    assert trainer.prior.z.device.type == "cuda"
    initial = state_digest(trainer.prior.state_dict())
    target, _ = study.scorer(task)
    _, _, task_declaration = study.build(arm, task, device)
    spec = task_declaration["execution"]["host_definition"]
    stream = context.streams.generator("data", component="target", purpose="training", device="cpu")
    trainer.step(target(spec, 128, stream, 0).to(device))
    checkpoint = context.state_dict()
    assert state_digest(trainer.prior.state_dict()) == initial
    require_optimizer_steps(checkpoint, 1)
    if trainer._extrapolation is not None:
        assert not any(name.startswith("prior.") for name in trainer._extrapolation.previous)
    resumed, resumed_trainer, _ = study.build(arm, task, device)
    resumed.load_state_dict(checkpoint)
    assert state_digest(resumed.state_dict()) == state_digest(checkpoint)
    for current, current_trainer in ((context, trainer), (resumed, resumed_trainer)):
        rng = current.streams.generator("data", component="target", purpose="training", device="cpu")
        current_trainer.step(target(spec, 128, rng, 1).to(device))
    assert state_digest(context.state_dict()) == state_digest(resumed.state_dict())
    assert state_digest(trainer.prior.state_dict()) == initial


@pytest.mark.parametrize("arm", ["alternating", "simultaneous", "extrapolation_from_past"])
@pytest.mark.parametrize("task", ["gaussian1d_acquisition", "ring16_acquisition"])
def test_matched_frozen_control_and_exact_resume(arm, task):
    probe(arm, task, device="cuda:0")


def test_scoped_archived_host_restored_after_error():
    original = study.host.PROTOCOL, study.host.build, study.host.initial_proof
    with pytest.raises(RuntimeError):
        with study.archived_host_binding():
            assert study.host.PROTOCOL == study.PROTOCOL
            raise RuntimeError("fixture")
    assert original == (study.host.PROTOCOL, study.host.build, study.host.initial_proof)


def test_cpu_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="requires CUDA"):
        study.execute(tmp_path / "absent", device="cpu")
    assert not (tmp_path / "absent").exists()
