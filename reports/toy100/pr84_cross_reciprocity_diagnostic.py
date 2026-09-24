"""Read-only directional game-operator probe at archived PR84 states.

The common real/latent minibatch and output-noise draw come from the saved
pre-step state.  They are held fixed across every finite difference.  This is
an operator diagnostic, not the alternating training map or a new optimizer.
No target centers or quality grades enter a field evaluation.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan.gan_loss import GANLoss
from particlegan.grad_regularizers import GradRegularizer


STEPS = (1324, 1325, 1530, 1539)
# One-percent secants cross LeakyReLU activation boundaries on these saved
# states.  Preserve them as a failed derivative calibration, then check a
# finer pair with an exact-zero-sum control before interpreting local signs.
COARSE_FRACTIONS = (.01, .005)
FRACTIONS = (.0001, .00005)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def _model_state(saved: dict) -> dict:
    if any(not name.startswith("model.") for name in saved):
        raise RuntimeError("expected fixed-noise wrapper without extra parameters")
    return {name.removeprefix("model."): value for name, value in saved.items()}


def _metric(optimizer: dict, shapes: list[torch.Size]) -> list[torch.Tensor]:
    values = []
    for group in optimizer["param_groups"]:
        rate, beta2, eps = group["lr"], group["betas"][1], group["eps"]
        if group["betas"][0] != 0 or group.get("amsgrad", False):
            raise RuntimeError("this diagnostic assumes the saved beta1=0 Adam metric")
        for index in group["params"]:
            item = optimizer["state"][index]
            step = int(item["step"])
            denom = (item["exp_avg_sq"].double() / (1 - beta2 ** step)).sqrt() + eps
            values.append(rate / denom)
    if [x.shape for x in values] != shapes:
        raise RuntimeError("optimizer metric does not align with model parameters")
    if any(not torch.isfinite(x).all() or not (x > 0).all() for x in values):
        raise FloatingPointError("nonfinite or nonpositive saved Adam metric")
    return values


def _dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def _sub(a: list[torch.Tensor], b: list[torch.Tensor]) -> list[torch.Tensor]:
    return [x - y for x, y in zip(a, b)]


def _scaled_fd(plus: list[torch.Tensor], minus: list[torch.Tensor], h: float):
    return [(x - y) / (2 * h) for x, y in zip(plus, minus)]


def _smoothed(critic, points, width):
    values = [critic(points)]
    for axis in range(2):
        displacement = torch.zeros_like(points)
        displacement[:, axis] = width
        values.extend((critic(points + displacement), critic(points - displacement)))
    return torch.stack(values).mean(dim=0)


class PairedField:
    def __init__(self, phases: dict, row: dict, step: int):
        pre, base, accepted = (phases[name] for name in
                               ("pre_step", "post_accepted_d", "post_bounded_g"))
        self.step = step
        self.width = float(row["stages"]["record"]["critic_width"])
        if not 0 < self.width <= .15:
            raise RuntimeError("unexpected PR84 stencil width")
        self.generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                            mode_hold.N_HIDDEN, 2)
        self.critic = SimpleMLPDiscriminator(2, mode_hold.HIDDEN,
                                             mode_hold.N_HIDDEN, mode_hold.FOURIER)
        self.generator.load_state_dict(_model_state(base["generator"]))
        self.critic.load_state_dict(_model_state(base["critic"]))
        # Differentiate the frozen float32 state in float64.  The host still
        # trains in float32; this avoids dividing float32 rounding noise by a
        # one-percent-step finite-difference distance.
        self.generator.double()
        self.critic.double()
        self.z = torch.nn.Parameter(base["prior"]["z"].detach().clone().double())
        self.d_params = list(self.critic.parameters())
        self.g_params = list(self.generator.parameters()) + [self.z]
        self.d_base = [p.detach().clone() for p in self.d_params]
        self.g_base = [p.detach().clone() for p in self.g_params]
        pre_d = list(_model_state(pre["critic"]).items())
        accepted_g = list(_model_state(accepted["generator"]).items())
        base_g = list(_model_state(base["generator"]).items())
        if [name for name, _ in pre_d if name in dict(self.critic.named_parameters())] != [
                name for name, _ in self.critic.named_parameters()]:
            raise RuntimeError("critic state order changed")
        if [name for name, _ in accepted_g] != [name for name, _ in base_g]:
            raise RuntimeError("generator state keys changed")
        self.d_delta = [p.detach() - dict(pre_d)[name].double() for name, p in
                        self.critic.named_parameters()]
        self.g_delta = [dict(accepted_g)[name].double() - p.detach() for name, p in
                        self.generator.named_parameters()]
        self.g_delta.append(accepted["prior"]["z"].double() - self.z.detach())
        d_metric = _metric(accepted["optimizer_d"], [p.shape for p in self.d_params])
        g_metric = _metric(accepted["optimizer_g"], [p.shape for p in self.g_params])
        self.d_norm = _dot(self.d_delta, [d / p for d, p in zip(self.d_delta, d_metric)]) ** .5
        self.g_norm = _dot(self.g_delta, [d / p for d, p in zip(self.g_delta, g_metric)]) ** .5
        if self.d_norm <= 0 or self.g_norm <= 0:
            raise RuntimeError("accepted update has zero directional norm")
        samples = row["stages"]["batches"]
        real = next(x for x in samples if x["phase"] == 0 and x["kind"] == "real")
        latent = next(x for x in samples if x["phase"] == 0 and x["kind"] == "prior")
        self.real = torch.tensor(real["values"], dtype=torch.float32).double()
        self.indices = torch.tensor(latent["indices"], dtype=torch.long)
        if not torch.equal(self.z.detach()[self.indices],
                           torch.tensor(latent["latent"], dtype=torch.float32).double()):
            raise RuntimeError("saved prior sample does not match its particle indices")
        if pre["noise"]["input_sigma"] != 0 or pre["noise"]["output_sigma"] != .029:
            raise RuntimeError("unexpected post-horizon training noise")
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(pre["rng"]["torch"])
            self.output_noise = (torch.randn((len(self.indices), 2), dtype=torch.float32)
                                 * .029).double()
        self.noise_sha256 = hashlib.sha256(self.output_noise.numpy().tobytes()).hexdigest()
        self.gan = GANLoss("logistic", "rp")
        self.cap = GradRegularizer(arm="b_cap", coeff=1., kappa=1.,
                                   norm="l2", method="autograd")

    def _set_point(self, d_scale, g_scale):
        with torch.no_grad():
            for p, base, delta in zip(self.d_params, self.d_base, self.d_delta):
                p.copy_(base + d_scale * delta)
            for p, base, delta in zip(self.g_params, self.g_base, self.g_delta):
                p.copy_(base + g_scale * delta)

    def fields(self, d_scale: float, g_scale: float) -> dict:
        self._set_point(d_scale, g_scale)
        real = self.real
        fake = self.generator(self.z[self.indices]) + self.output_noise
        d_real, d_fake = self.critic(real), self.critic(fake)
        d_adv = self.gan.d_loss(d_real, d_fake)
        d_cap = self.cap(self.critic, real, fake.detach(), step=self.step)
        g_sharp = self.gan.g_loss(d_fake, d_real)
        g_smooth = self.gan.g_loss(_smoothed(self.critic, fake, self.width),
                                    _smoothed(self.critic, real, self.width))
        # Keep the shared fake graph until the zero-sum reference is evaluated.
        def gradient(loss, params):
            # Input-gradient penalties do not depend on the critic's final
            # additive bias; that component is exactly zero for this term.
            raw = torch.autograd.grad(loss, params, retain_graph=True,
                                      allow_unused=True)
            return [(torch.zeros_like(param) if value is None else value.detach().clone())
                    for param, value in zip(params, raw)]
        result = dict(d_adv=gradient(d_adv, self.d_params),
                      d_cap=gradient(d_cap, self.d_params),
                      g_sharp=gradient(g_sharp, self.g_params),
                      g_smooth=gradient(g_smooth, self.g_params),
                      g_zero_sum=[-x for x in gradient(d_adv, self.g_params)],
                      advantage=float((d_real - d_fake).detach().mean()),
                      d_penalty=float(d_cap.detach()))
        return result

    def at_fraction(self, h: float) -> dict:
        gp, gm = self.fields(0, h), self.fields(0, -h)
        dp, dm = self.fields(h, 0), self.fields(-h, 0)
        dg = self.d_norm * self.g_norm
        dd = self.d_norm ** 2
        gg = self.g_norm ** 2
        cd = {key: _dot(self.d_delta, _scaled_fd(gp[key], gm[key], h)) / dg
              for key in ("d_adv", "d_cap")}
        cg = {key: _dot(self.g_delta, _scaled_fd(dp[key], dm[key], h)) / dg
              for key in ("g_sharp", "g_smooth", "g_zero_sum")}
        od = {key: _dot(self.d_delta, _scaled_fd(dp[key], dm[key], h)) / dd
              for key in ("d_adv", "d_cap")}
        og = {key: _dot(self.g_delta, _scaled_fd(gp[key], gm[key], h)) / gg
              for key in ("g_sharp", "g_smooth")}
        return dict(h_fraction_of_accepted_update=h,
                    cross_d_adversarial=cd["d_adv"], cross_d_cap=cd["d_cap"],
                    cross_g_sharp=cg["g_sharp"],
                    cross_g_stencil_increment=cg["g_smooth"]-cg["g_sharp"],
                    cross_g_zero_sum_control=cg["g_zero_sum"],
                    zero_sum_cross_residual=cd["d_adv"]+cg["g_zero_sum"],
                    cross_total=cd["d_adv"]+cd["d_cap"]+cg["g_smooth"],
                    own_d_adversarial=od["d_adv"], own_d_cap=od["d_cap"],
                    own_g_sharp=og["g_sharp"],
                    own_g_stencil_increment=og["g_smooth"]-og["g_sharp"],
                    own_d_total=od["d_adv"]+od["d_cap"], own_g_total=og["g_smooth"],
                    full_directional_symmetric_rayleigh=(
                        od["d_adv"]+od["d_cap"]+og["g_smooth"]+
                        cd["d_adv"]+cd["d_cap"]+cg["g_smooth"])/2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    diagnosis = json.loads(_read(args.diagnosis))
    if diagnosis["status"] != "EXACT_REFERENCE_PARITY":
        raise RuntimeError("saved-state provenance failed")
    if args.states.suffix == ".gz":
        bundle = torch.load(BytesIO(_read(args.states)), weights_only=True)
        if bundle["parent_states_sha256"] != diagnosis["selected_states_sha256"]:
            raise RuntimeError("subset parent state hash differs from exact replay")
        all_states = bundle["states"]
    else:
        if sha(args.states) != diagnosis["selected_states_sha256"]:
            raise RuntimeError("full saved-state hash differs from exact replay")
        all_states = torch.load(args.states, weights_only=True)
    rows = {row["step"]: row for row in diagnosis["rows"]}
    results = []
    with torch.random.fork_rng(devices=[]):
        for step in STEPS:
            probe = PairedField(all_states[step], rows[step], step)
            baseline = probe.fields(0, 0)
            coarse = [probe.at_fraction(h) for h in COARSE_FRACTIONS]
            finite = [probe.at_fraction(h) for h in FRACTIONS]
            a, b = finite
            if any(abs(q["zero_sum_cross_residual"]) > 1e-5 for q in finite):
                raise RuntimeError("same-sample zero-sum derivative control failed")
            keys = ("cross_total", "own_d_total", "own_g_total",
                    "full_directional_symmetric_rayleigh")
            scale_consistency = {key: abs(a[key]-b[key]) /
                                 max(abs(a[key]),abs(b[key]),1e-12) for key in keys}
            grade = rows[step]["grades"]
            result = dict(step=step,
                          pre_grade=grade["pre_step"],
                          post_grade=grade["bounded_joint"],
                          width=probe.width, mean_advantage=baseline["advantage"],
                          d_penalty=baseline["d_penalty"],
                          d_metric_update_norm=probe.d_norm,
                          g_metric_update_norm=probe.g_norm,
                          output_noise_sha256=probe.noise_sha256,
                          coarse_calibration_secants=coarse,
                          finite_differences=finite,
                          relative_scale_disagreement=scale_consistency,
                          own_g_local_derivative_resolved=(
                              scale_consistency["own_g_total"] <= .05))
            results.append(result)
            print(json.dumps(dict(event="STATE_DONE", step=step,
                                  cross=a["cross_total"], own_g=a["own_g_total"],
                                  residual=a["zero_sum_cross_residual"])), flush=True)
    output = dict(scope="read_only_same_common_batch_directional_operator_secants",
                  linearization_dtype="float64_at_frozen_float32_weights_and_noise",
                  coarse_fractions=list(COARSE_FRACTIONS),
                  coarse_calibration="nonlocal LeakyReLU secants; failed zero-sum cancellation; retained but excluded from local-Jacobian interpretation",
                  caveat="Not the alternating Adam state-transition Jacobian; directional values do not prove stability or instability.",
                  steps=list(STEPS), fractions=list(FRACTIONS),
                  diagnosis_file_sha256=sha(args.diagnosis),
                  diagnosis_plain_sha256=hashlib.sha256(_read(args.diagnosis)).hexdigest(),
                  parent_full_states_sha256=diagnosis["selected_states_sha256"],
                  states_file_sha256=sha(args.states),
                  source_sha256=sha(Path(__file__)), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
