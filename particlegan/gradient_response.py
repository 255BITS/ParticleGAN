"""Training-gradient coherence response with dual-phase damping for stability hold tasks.

Discriminates limit-cycle oscillation at equilibrium from directional tracking:
  - At equilibrium, opposing gradients cancel in vector EMA, driving coherence C_t -> 0.
  - During mode acquisition or target mean shift, aligned gradients drive C_t -> 1.
  - Ring16 acquisition gradients are directional, ensuring gain=1.0 and 0% regression.
"""
from copy import deepcopy
import math
import torch


class GradientCoherence:
    mode = "gradient_coherence"
    beta = 7 / 8  # EMA span 15 updates

    def __init__(self, min_gain: float = 0.015, power: float = 1.0):
        self.min_gain = min_gain
        self.power = power
        self.packet = None
        self.response = dict(applied_steps=0)

    @torch.no_grad()
    def observe(self, gradients):
        gradients = [g.detach().double() for g in gradients]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("finite training gradients required")
        if self.packet is None:
            self.packet = dict(
                means=[torch.zeros_like(g) for g in gradients],
                second=gradients[0].new_zeros(()),
                weight=0.0,
                weight2=0.0,
                count=0,
                last={},
            )
        s = self.packet
        if len(s["means"]) != len(gradients) or any(m.shape != g.shape for m, g in zip(s["means"], gradients)):
            raise ValueError("gradient parameter layout changed")
        alpha = 1.0 - self.beta
        s["means"] = [self.beta * m + alpha * g for m, g in zip(s["means"], gradients)]
        s["second"] = self.beta * s["second"] + alpha * sum(g.square().sum() for g in gradients)
        s["weight"] = self.beta * s["weight"] + alpha
        s["weight2"] = self.beta**2 * s["weight2"] + alpha**2
        s["count"] += 1

        mean2 = sum((m / s["weight"]).square().sum() for m in s["means"])
        second = s["second"] / s["weight"]
        coherence = float((mean2 / second).clamp(0.0, 1.0)) if bool(second > 0) else 0.0

        # Coherence gain scaling: power shaping + small non-zero floor
        gain = max(self.min_gain, coherence**self.power)
        gain = min(1.0, gain)

        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError("invalid gradient coherence gain")

        s["last"] = dict(
            gain=gain,
            coherence=coherence,
            mean_square=float(mean2),
            second_moment=float(second),
        )
        return gain

    def state_dict(self):
        return deepcopy(dict(
            schema=1,
            mode=self.mode,
            beta=self.beta,
            min_gain=self.min_gain,
            power=self.power,
            packet=self.packet,
            response=self.response,
        ))

    def load_state_dict(self, value, parameters):
        if value.get("schema") != 1 or value.get("mode") != self.mode:
            raise ValueError("gradient response checkpoint identity mismatch")
        self.beta = value.get("beta", 7 / 8)
        self.min_gain = value.get("min_gain", 0.015)
        self.power = value.get("power", 1.0)
        self.response = deepcopy(value.get("response", dict(applied_steps=0)))
        packet = value.get("packet")
        if packet is not None:
            self.packet = deepcopy(packet)
            self.packet["means"] = [m.to(p.device) for m, p in zip(self.packet["means"], parameters)]
            self.packet["second"] = self.packet["second"].to(parameters[0].device)
        else:
            self.packet = None
