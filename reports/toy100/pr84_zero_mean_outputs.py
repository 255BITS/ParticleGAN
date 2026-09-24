"""Stall reach with hard zero-mean generator outputs on the train forward.

After each train-step ``fake = G(...)`` that feeds D or G (including the
particle cloud on the trajectory host), subtract the batch mean. Evaluation
forwards are untouched, so mode counts still read the raw cloud. The identity
fork leaves the recorder disabled and this centering is a no-op there.

Stall-reach width, curvature bounds, D, losses, Adam and the learning-rate
schedule are the #107 candidate. No coverage, anchor, quota or clip term.
"""

import ast
from contextlib import contextmanager

import torch

from reports.toy100 import pr84_smoothed_candidate as base
from reports.toy100.pr84_reach_candidate import REACH, ReachRecorder


METHOD = "stall_reach_zero_mean_g_outputs"


def _inject_zero_mean(tree):
    """Center every train-block assignment to ``fake``; leave other calls alone."""

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "fake"):
                center = ast.Assign(
                    targets=[ast.Name(id="fake", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="_extra_state", ctx=ast.Load()),
                                           attr="zero_mean", ctx=ast.Load()),
                        args=[ast.Name(id="fake", ctx=ast.Load())], keywords=[]))
                return [node, center]
            return node

    Inject().visit(tree)


class ZeroMeanReachRecorder(ReachRecorder):
    """#107 stall reach, plus hard centering of train-time fake batches."""

    def __init__(self, *, start_step=0):
        super().__init__(start_step=start_step)
        self.centered_batches = 0

    def zero_mean(self, fake):
        """Hard-center one train batch. Disabled and passthrough steps are exact no-ops."""
        if (self.passthrough or not self.enabled or not torch.is_tensor(fake)
                or fake.ndim < 2 or fake.shape[0] < 1):
            return fake
        self.centered_batches += 1
        return fake - fake.mean(dim=0, keepdim=True)

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, zero_mean_train_cloud=True,
                     centered_batches=self.centered_batches,
                     cpu_capability=str(torch.backends.cpu.get_cpu_capability()))
        return value


@contextmanager
def zero_mean_g_outputs(*, task="mode_hold", start_step=0):
    """Stall-reach host whose train fakes are hard-centered. Yields ``(recorder, source)``."""
    original = base.SmoothedBothBoundRecorder

    def init(self, *, start_step=0):
        ZeroMeanReachRecorder.__init__(self, start_step=start_step)
        self.curvature_bound = base.G_CURVATURE_BOUND

    base.SmoothedBothBoundRecorder = type(
        "ZeroMeanReachRecorder", (ZeroMeanReachRecorder,),
        dict(reach=REACH, ramp="stall", game_bound=False, game_steps=1, __init__=init))
    try:
        with base.pr84_smoothed_candidate(task=task, start_step=start_step,
                                         prepare_tree=_inject_zero_mean) as value:
            yield value
    finally:
        base.SmoothedBothBoundRecorder = original


__all__ = ["METHOD", "ZeroMeanReachRecorder", "zero_mean_g_outputs"]
