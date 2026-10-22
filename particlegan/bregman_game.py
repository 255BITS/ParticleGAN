"""Learned Bregman c-transform game on existing real/generated batches.

The critic learns target prices. Unlike a minibatch balanced assignment, each
fake selects its least cost-minus-price target; marginal balance is the dual
training signal, not an assertion about any finite critic iterate.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def bregman_cost_matrix(fake, real, *, kind="identity", prelink=None):
    """Return c[i,j]=D_h(real[j] || fake[i]), retaining only fake gradients.

    Identity uses mean squared distance / (2 real coordinate variance), with
    unit scale for exactly constant real batches. Sigmoid uses mean Bernoulli
    KL, evaluated from actual finite pre-sigmoid values rather than logit(x).
    The latter's prelink derivative is sigmoid(a)-y even at rounded x=0/1.
    No labels, task geometry, random draw, fitted width or solver enters.
    """
    if (fake.ndim < 2 or real.ndim != fake.ndim or fake.shape[1:] != real.shape[1:]
            or min(len(fake), len(real)) < 2 or fake[0].numel() == 0
            or not fake.is_floating_point() or fake.dtype != real.dtype
            or fake.device != real.device):
        raise ValueError("Bregman game requires compatible floating batches with at least two rows")
    x, y = fake.flatten(1), real.detach().flatten(1)
    d = x.shape[1]
    if kind == "identity":
        if prelink is not None:
            raise ValueError("identity output must not supply a sigmoid prelink")
        scale = (y - y.mean(0)).square().mean()
        scale = torch.where(scale > 0, scale, scale.new_ones(()))
        return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").square() / (2 * d * scale)
    if kind != "sigmoid" or prelink is None or prelink.shape != fake.shape:
        raise ValueError("sigmoid Bregman game requires a declared matching prelink tensor")
    if prelink.dtype != fake.dtype or prelink.device != fake.device:
        raise ValueError("sigmoid prelink must share output device and dtype")
    if bool(((y < 0) | (y > 1)).any()):
        raise ValueError("sigmoid Bregman targets must lie in [0,1]")
    a = prelink.flatten(1)
    negative_entropy = (torch.special.xlogy(y, y) + torch.special.xlogy(1 - y, 1 - y)).mean(1)
    return F.softplus(a).mean(1)[:, None] - a @ y.T / d + negative_entropy[None, :]


def critic_game_loss(cost, real_scores):
    """Minimize negative semi-dual value; fake/cost coordinates are detached.

    At independent target scores the negative-loss gradient is target mass
    minus selected mass. A missing mode can have zero cost at zero prices;
    its nonzero price gradient is essential. This is not a calibrated scalar
    discrepancy at a nonoptimal, regularized neural critic.
    """
    scores = real_scores.reshape(-1)
    if cost.ndim != 2 or cost.shape[1] != len(scores):
        raise ValueError("Bregman cost/critic score dimensions differ")
    return -(cost.detach() - scores[None, :]).min(1).values.mean() - scores.mean()


def generator_game_loss(cost, real_scores, *, thermal_scale=1.):
    """Semi-dual generator objective, with frozen routing prices and thermal law.

    Argmin indices are detached. The selected real-score subtraction affects
    the value but supplies no G gradient. At ties torch.min's deterministic
    first-index subgradient is used; no random tie-breaking is introduced.
    """
    scores = real_scores.detach().reshape(-1)
    if cost.ndim != 2 or cost.shape[1] != len(scores):
        raise ValueError("Bregman cost/critic score dimensions differ")
    if not 0 <= thermal_scale <= 1:
        raise ValueError("Bregman thermal scale must lie in [0,1]")
    routed = (cost - scores[None, :]).min(1).values.mean()
    return routed * thermal_scale + routed.detach() * (1 - thermal_scale)


class OutputLink:
    """Explicit capability contract; capture hooks are scoped to one forward.

    Public models declare ``particlegan_output_link`` as a dict with ``kind``
    and, for sigmoid, ``prelink_module`` naming the layer whose output is passed
    directly through sigmoid. No post-sigmoid averaging/reshaping is implied.
    Pure Linear and Sequential models have intrinsic exact terminal contracts.
    Unknown host transformations fail before training instead of guessing a link.
    """
    def __init__(self, generator):
        declaration = getattr(generator, "particlegan_output_link", None)
        if declaration is None and isinstance(generator, nn.Linear):
            declaration = {"kind": "identity"}
        if declaration is None and isinstance(generator, nn.Sequential) and len(generator):
            if isinstance(generator[-1], nn.Linear):
                declaration = {"kind": "identity"}
            elif isinstance(generator[-1], nn.Sigmoid):
                declaration = {"kind": "sigmoid", "prelink_module": str(len(generator)-1), "capture_input": True}
        if not isinstance(declaration, dict) or declaration.get("kind") not in ("identity", "sigmoid"):
            raise ValueError("Bregman game requires a declared identity/sigmoid output capability")
        self.contract = dict(declaration)
        self.kind = declaration["kind"]
        self.generator = generator
        self.module = None
        if self.kind == "sigmoid":
            path = declaration.get("prelink_module")
            if not isinstance(path, str) or not path:
                raise ValueError("sigmoid output capability requires a named prelink module")
            self.module = generator.get_submodule(path)

    @contextmanager
    def capture(self):
        observed = []
        handle = None
        if self.module is not None:
            def record(module, args, output):
                observed.append(args[0] if self.contract.get("capture_input", False) else output)
            handle = self.module.register_forward_hook(record)
        try:
            yield observed
        finally:
            if handle is not None:
                handle.remove()

    def attach(self, output, observed):
        if self.kind == "sigmoid":
            if len(observed) != 1 or observed[0].shape != output.shape:
                raise ValueError("declared sigmoid prelink was not consumed exactly once with matching shape")
            if not torch.equal(observed[0].detach().sigmoid(), output.detach()):
                raise ValueError("declared sigmoid capability does not match the actual generator output")
            output._particlegan_prelink = observed[0]
        return output

    def cost(self, output, real):
        return bregman_cost_matrix(output, real, kind=self.kind,
                                   prelink=getattr(output, "_particlegan_prelink", None))
