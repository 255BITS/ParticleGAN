"""Generator-side critic reads (research option; the default read is the critic).

``StallReachRead`` rebuilds closed PR #107's "stall reach": the generator reads
the critic through a symmetric axis stencil whose width adapts to the critic's
measured input slope. The critic's own loss and penalty are unchanged.

Per update, after the critic step, ``observe(fake)`` measures
``s = RMS_batch ||grad_x D(fake)||`` on the detached critic-phase fakes (one
autograd call, no parameter gradients, no RNG, no extra samples). With
``u = s / kappa``:

* a saturated counter increments while ``u >= SATURATED`` and resets otherwise;
* width ``w = REACH / kappa`` once the counter reaches ``WINDOW`` (stalled),
  else ``w = max(BASE, REACH * min(u, 1/u)) / kappa``.

``forward(x, *args)`` returns the mean critic logit over ``x`` and
``x +- w e_i`` for every flattened input coordinate (``2d + 1`` points, the
five-point stencil in 2D) in one batched call. Before the first ``observe`` the
read is the plain critic.
"""
import math

import torch
from torch import nn

BASE, REACH, SATURATED, WINDOW = .15, .5, .6, 50


def _repeat(value, copies):
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return value.unsqueeze(0).expand(copies, *value.shape).reshape(-1, *value.shape[1:])
    return value


class StallReachRead(nn.Module):
    """Mean critic over an adaptive axis stencil; see the module docstring."""

    def __init__(self, critic, *, kappa=1.0, probe=None):
        super().__init__()
        if type(kappa) not in (int, float) or not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("stall_reach requires a finite positive reg_kappa")
        # Not registered as submodules: the read owns no parameters or buffers.
        object.__setattr__(self, "critic", critic)
        object.__setattr__(self, "probe", critic if probe is None else probe)
        self.kappa = float(kappa)
        self.counter = 0
        self.width = None
        self.observes = self.widened = self.stalled = 0
        self.max_width = 0.0
        self.last_stats = {}

    def width_for(self, u, counter):
        if counter >= WINDOW:
            return REACH / self.kappa
        ratio = u if u <= 1 else 1 / u
        return max(BASE, REACH * ratio) / self.kappa

    def observe(self, fake, *args, **kwargs):
        """Update the width from the critic's slope on detached critic-phase fakes."""
        x = fake.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.probe(x, *args, **kwargs)
            (grad,) = torch.autograd.grad(logits.sum(), x, only_inputs=True)
        s = float(grad.flatten(1).square().sum(1).mean().sqrt())
        u = s / self.kappa
        self.counter = self.counter + 1 if u >= SATURATED else 0
        stalled = self.counter >= WINDOW
        self.width = self.width_for(u, self.counter)
        self.observes += 1
        self.widened += int(self.width > BASE / self.kappa * (1 + 1e-12))
        self.stalled += int(stalled)
        self.max_width = max(self.max_width, self.width)
        self.last_stats = dict(slope=s, u=u, width=self.width, stalled=stalled, counter=self.counter)
        return self.last_stats

    def forward(self, x, *args, **kwargs):
        if self.width is None:
            return self.critic(x, *args, **kwargs)
        n, shape = x.shape[0], x.shape[1:]
        d = math.prod(shape)
        eye = torch.eye(d, device=x.device, dtype=x.dtype).reshape(d, *shape) * self.width
        offsets = torch.cat((torch.zeros_like(eye[:1]), eye, -eye))  # (2d+1, *shape)
        copies = 2 * d + 1
        stacked = (x.unsqueeze(0) + offsets.unsqueeze(1)).reshape(copies * n, *shape)
        args = tuple(_repeat(a, copies) for a in args)
        kwargs = {key: _repeat(value, copies) for key, value in kwargs.items()}
        logits = self.critic(stacked, *args, **kwargs)
        return logits.reshape(copies, n, *logits.shape[1:]).mean(0)

    def summary(self):
        """Cumulative activation counts for receipts and diagnostics."""
        return dict(kind="stall_reach", observes=self.observes, widened=self.widened, stalled=self.stalled,
                    max_width=self.max_width, counter=self.counter, width=self.width)

    def state_dict(self, *args, **kwargs):
        return dict(schema=1, kappa=self.kappa, counter=self.counter, width=self.width,
                    observes=self.observes, widened=self.widened, stalled=self.stalled, max_width=self.max_width)

    def check_state(self, state):
        if (not isinstance(state, dict) or set(state) != set(self.state_dict())
                or state["schema"] != 1 or state["kappa"] != self.kappa
                or any(type(state[k]) is not int or state[k] < 0 for k in ("counter", "observes", "widened", "stalled"))
                or not (state["width"] is None or (type(state["width"]) is float and math.isfinite(state["width"])
                                                   and state["width"] > 0))):
            raise ValueError("invalid stall_reach critic-read state")

    def load_state_dict(self, state, strict=True):
        self.check_state(state)
        for key in ("counter", "width", "observes", "widened", "stalled", "max_width"):
            setattr(self, key, state[key])
        return None
