"""SOURCE_ONLY RA10 observer-control primitives; deliberately not executable.

No Torch import or fixture interpretation occurs in this file. A runnable
revision must bind and guard the final owner source/helpers/fixtures first.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guard_files(root, mapping):
    for relative, expected in mapping.items():
        actual = sha256(Path(root) / relative)
        if actual != expected:
            raise AssertionError((relative, expected, actual))


def distinct(items):
    return list({id(item): item for item in items}.values())


def owned_snapshot(torch, roots, streams):
    modules = distinct(module for root in roots for module in root.modules())
    parameters = distinct(parameter for root in roots for parameter in root.parameters())
    return {
        "modules": [(module, module.training, dict(module._buffers),
                     {name: None if value is None else value.detach().clone()
                      for name, value in module._buffers.items()},
                     set(module._non_persistent_buffers_set)) for module in modules],
        "gradients": [(parameter, parameter.grad,
                       None if parameter.grad is None else parameter.grad.detach().clone())
                      for parameter in parameters],
        "global_cpu": torch.get_rng_state().clone(),
        "streams": [(stream, stream.get_state().clone()) for stream in distinct(streams)],
    }


def assert_owned_restored(torch, before):
    for module, mode, original, values, nonpersistent in before["modules"]:
        assert module.training is mode
        assert tuple(module._buffers) == tuple(original)
        assert module._non_persistent_buffers_set == nonpersistent
        for name, value in original.items():
            assert module._buffers[name] is value
            if value is not None:
                assert torch.equal(value, values[name])
    for parameter, original, value in before["gradients"]:
        assert parameter.grad is original
        if original is not None:
            assert torch.equal(original, value)
    assert torch.equal(torch.get_rng_state(), before["global_cpu"])
    for stream, value in before["streams"]:
        assert torch.equal(stream.get_state(), value)


def dirty_eval_module(torch, streams, *, raise_after=False):
    """Literal CPU module; future control makes no initialization RNG draw."""

    class DirtyEval(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.second = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            self.first.grad = torch.tensor([5.0, 6.0])
            self.second.grad = None
            self.register_buffer("keep", torch.tensor([7.0, 8.0]))
            self.register_buffer("replace", torch.tensor([9.0, 10.0]), persistent=False)
            self.register_buffer("optional", None)
            self.child = torch.nn.Identity()
            self.training = True
            self.child.training = False

        def forward(self, inputs):
            with torch.no_grad():
                self.keep.add_(1)
                self._buffers["replace"] = torch.tensor([11.0, 12.0])
                self._buffers["optional"] = torch.tensor([13.0, 14.0])
                self.register_buffer("extra", torch.tensor([15.0, 16.0]))
                self._non_persistent_buffers_set.clear()
                self._non_persistent_buffers_set.add("keep")
                self.child.register_buffer("nested_extra", torch.tensor([17.0]))
                self.first.grad.add_(1)
                self.first.grad = torch.tensor([18.0, 19.0])
                self.second.grad = torch.tensor([20.0, 21.0])
                self.training = False
                self.child.training = True
                torch.rand((), device="cpu")
                for stream in distinct(streams):
                    torch.rand((), device="cpu", generator=stream)
            if raise_after:
                raise RuntimeError("intentional dirty eval exception")
            return inputs

    return DirtyEval()


CONTROL_GROUPS = (
    "dirty_observer_normal_and_exception",
    "one_draw_exact_packet_no_draw_commit",
    "stale_consumed_malformed_prewrite_rejection",
    "complete_reservations_unique_rows_shared_budget",
    "actual_complete_moved_reset_rebase_cache_final_lease",
)


def main():
    raise SystemExit(
        "SOURCE_ONLY: bind final backend9 source/fixture guards in a sealed "
        "runnable revision before importing Torch or executing controls."
    )


if __name__ == "__main__":
    main()
