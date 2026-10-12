"""Software-only checks of passive comparison identities, with no experiments."""
from types import SimpleNamespace

import pytest
import torch

from experiments.forge.comparison_protocol import ComparisonProtocol
from experiments.forge.rng import NamedStreams
from experiments.forge.state import state_digest


def test_target_identity_preserves_rng_tensor_and_graph_and_detects_order_shape_dtype():
    streams = NamedStreams(0)
    data = streams.generator("data", component="target", purpose="training")
    values = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    original = values.detach().clone()
    before_named, before_global = streams.audit(), torch.get_rng_state().clone()
    receipts = []
    for batches in ((values, values.flip(0)), (values.flip(0), values),
                    (values.reshape(1, 4),), (values.double(),)):
        proof = ComparisonProtocol(streams, {"prior": values})
        for batch in batches:
            proof.record_pair(batch, batch)
        receipts.append(proof.receipt(streams))
    assert streams.audit() == before_named
    assert torch.equal(torch.get_rng_state(), before_global)
    assert torch.equal(values.detach(), original)
    assert values.requires_grad and values.grad is None
    values.square().sum().backward()
    assert torch.equal(values.grad, original * 2)
    identities = [receipt["actual_target_batches"]["critic"]["sha256"] for receipt in receipts]
    assert len(set(identities)) == len(identities)
    first = receipts[0]
    assert first["actual_target_batches"]["critic"] == first["actual_target_batches"]["generator"]
    assert first["actual_target_batches"]["critic"]["batches"] == 2
    assert first["actual_target_batches"]["critic"]["samples"] == 4
    assert first["initial_tensor_metadata"]["prior"][""] == {
        "shape": [2, 2], "dtype": "torch.float32", "device": "cpu"}
    assert first["rng_checkpoints"]["all_named_streams_checkpointed"]
    assert not any(row["changed"] for row in first["rng_checkpoints"]["bindings"].values())


def test_stream_receipt_covers_late_bindings_and_separates_consumed_families():
    streams = NamedStreams(0)
    streams.generator("init", component="prior", purpose="locations")
    data = streams.generator("data", component="target", purpose="training")
    proof = ComparisonProtocol(streams, {"prior": torch.zeros(2, 1)})
    noise = streams.generator("noise", component="prior", purpose="gaussian")
    evaluation = streams.generator("eval", component="live", purpose="samples")
    torch.rand(3, generator=data)
    torch.rand(2, generator=noise)
    torch.rand(4, generator=evaluation)
    before = state_digest(streams.state_dict())
    receipt = proof.receipt(streams)
    assert state_digest(streams.state_dict()) == before
    rows = receipt["rng_checkpoints"]["bindings"]
    assert set(rows) == set(streams.state_dict()["states"])
    assert {row["family"] for row in rows.values() if row["changed"]} == {"data", "noise", "eval"}
    assert len({row["initial_state_sha256"] for row in rows.values()}) == len(rows)


def test_word_hook_records_both_original_inputs_and_removes_itself_on_success_or_failure():
    encoder = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(2.)
    fixture = SimpleNamespace(E=encoder)
    streams = NamedStreams(0)
    proof = ComparisonProtocol(streams, {"encoder": encoder})
    target_d, target_g = torch.ones(2, 2), torch.zeros(2, 2)
    expected_d, expected_g = encoder(target_d), encoder(target_g)
    before = torch.get_rng_state().clone()
    with proof.word_step(fixture):
        assert torch.equal(encoder(target_d), expected_d)
        assert torch.equal(encoder(target_g), expected_g)
        encoder(target_g)  # Existing protected-loss probes are not new data.
    assert torch.equal(torch.get_rng_state(), before)
    assert not encoder._forward_pre_hooks
    records = proof.receipt(streams)["actual_target_batches"]
    assert records["critic"]["batches"] == records["generator"]["batches"] == 1
    assert records["critic"]["sha256"] != records["generator"]["sha256"]
    with pytest.raises(RuntimeError, match="fixture failure"):
        with proof.word_step(fixture):
            raise RuntimeError("fixture failure")
    assert not encoder._forward_pre_hooks
    with pytest.raises(ValueError, match="both actual target batches"):
        with proof.word_step(fixture):
            encoder(target_d)
    assert not encoder._forward_pre_hooks
