"""GPU software proof that a continuation changes only its external allowance."""
import pytest
import torch

from benchmarks.toy_audit import tier1_prior_duration as study
from experiments.forge.state import state_digest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="duration study requires CUDA")


def test_execution_extension_preserves_all_training_state():
    row = study.declaration()["runs"][0]
    _, trainer, _ = study.build(row, "cuda:0")
    study.extend_preserving_state(trainer, row["max_total_updates"])
    assert trainer.completed_steps == 0
    assert trainer.max_steps == 4000


def test_rejected_extension_keeps_state_intact():
    row = study.declaration()["runs"][1]
    _, trainer, _ = study.build(row, "cuda:0")
    before = state_digest(trainer.state_dict())
    with pytest.raises(ValueError, match="extension must exceed"):
        study.extend_preserving_state(trainer, row["parent_updates"])
    assert state_digest(trainer.state_dict()) == before
