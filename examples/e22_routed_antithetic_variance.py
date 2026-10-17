"""Zero-update native routed-game gradient variance, using public APIs only.

Run the bounded public-API fixtures with
  PYTHONPATH=. python examples/e22_routed_antithetic_variance.py --bf16
  PYTHONPATH=. python examples/e22_routed_antithetic_variance.py --fixture routed-dv12 --bf16
Exit0 is variance PASS; exit1 is completed variance FAIL. Ownership checks
are reported separately. Neither result establishes Supra convergence.
"""
import time
STARTED = time.monotonic()

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from particlegan import E22Policy, RoutedRows, get_recipe, init

PAIRS, BATCHES, SIGMA, MAX_RATIO = 16, 2, .125, .75


def digest(value):
    """Byte identity includes native NaN diagnostic placeholders."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                result.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            result.update(repr(item).encode())
    visit(value)
    return result.hexdigest()


def assert_learned_finite(state):
    """Check learned owners; native monitor validity belongs to public load."""
    def visit(item):
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            if not bool(torch.isfinite(item).all()):
                raise ValueError("nonfinite learned parameter, gradient or optimizer moment")
        elif isinstance(item, dict):
            for child in item.values(): visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
    for key in ("models", "averages", "table", "averaged_table", "output_noise"):
        visit(state[key])
    for optimizer in state["optimizers"]:
        visit(optimizer["state"])
        if "regularizer" in optimizer: visit(optimizer["regularizer"].get("ema"))


def _roles(policy):
    result = {}
    for optimizer, roles in zip(policy.optimizers, policy.roles):
        for group, role in zip(optimizer.param_groups, roles):
            if role in ("generator", "router", "table"):
                result.setdefault(role, []).extend(p for p in group["params"] if p.requires_grad)
    if not result.get("generator"):
        raise ValueError("a live full generator role is required")
    flat = [p for values in result.values() for p in values]
    if len(flat) != len({id(p) for p in flat}):
        raise ValueError("ambiguous role ownership")
    return result


def _vectors(gradients, prediction, role_parameters):
    result = {"residual_upstream": gradients[0].detach().float().flatten().cpu()}
    cursor = 1
    for role, parameters in role_parameters.items():
        values = gradients[cursor:cursor+len(parameters)]
        result[role] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().flatten().cpu()
                                  for p, g in zip(parameters, values)])
        cursor += len(parameters)
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError("nonfinite diagnostic gradient")
    return result


def paired_gradients(policy, prediction, target, condition, gaussian, role_parameters):
    """Compare averaged loss with averaging already transported gradients.

    The F32 upstream identity does not imply an exact BF16 host transport
    identity. All three autograd.grad calls leave parameter .grad untouched.
    """
    loss = policy.recipe.make_loss()
    residual = (prediction-target) / policy.D.scale
    objectives = []
    for sign in (1., -1.):
        real = sign * SIGMA * gaussian
        with torch.no_grad():
            real_logits = policy.D(real, condition)
        objectives.append(loss.g_loss(policy.D(real+residual, condition), real_logits))
    inputs = [prediction, *(p for values in role_parameters.values() for p in values)]
    values = []
    for objective in (*objectives, (objectives[0]+objectives[1])/2):
        gradients = torch.autograd.grad(objective, inputs, retain_graph=True, allow_unused=True)
        values.append(_vectors(gradients, prediction, role_parameters))
    return values


def _statistics(single, paired, posttransport):
    # Center within each predeclared batch; context means are not noise.
    variance = lambda x: float((x-x.mean(0)).square().sum(1).mean())
    single_v, pair_v = variance(single), variance(paired)
    signal = float(single.mean(0).square().sum())
    discrepancy = paired-posttransport
    return {"single_variance": single_v, "antithetic_variance": pair_v,
            "ratio": pair_v/single_v if single_v > 0 else None,
            "mean_gradient_energy": signal,
            "single_signal_to_noise": signal/single_v if single_v > 0 else None,
            "antithetic_signal_to_noise": float(paired.mean(0).square().sum())/pair_v if pair_v > 0 else None,
            "transport_discrepancy_max_abs": float(discrepancy.abs().max()),
            "transport_discrepancy_relative_rms": float(discrepancy.square().mean().sqrt()
                / posttransport.square().mean().sqrt().clamp_min(1e-30))}


def variance_gate(single_variance, paired_variance):
    return bool(math.isfinite(single_variance) and math.isfinite(paired_variance)
                and single_variance > 0 and 0 <= paired_variance <= MAX_RATIO*single_variance)


def probe(policy, batches, paired_rng, *, deadline=lambda: None):
    """Two fixed B4 batches,16 private DV12/Gaussian pairs, no updates.

    A private DV12 stream advances once per perturbed whole-model forward.
    Each sign shares that prediction/draw; consecutive pairs have new draws.
    CPU Gaussian draws preserve the caption caller's CPU43 marginal law;
    one discarded critic draw precedes each generator draw. This is a frozen
    G-phase distribution, not a simulation of32 native training updates.
    """
    if len(batches) != BATCHES or policy.output_sigma() != SIGMA:
        raise ValueError("fixed two-batch, sigma=.125 protocol required")
    saved = policy.state_dict(); before = digest(saved)
    assert_learned_finite(saved)
    modules = [owner for owner in (policy.G, policy.D, policy.encoder, policy.router, policy.prior,
               policy.ema_G, policy.ema_encoder, policy.ema_router, policy.ema_prior,
               policy.opt_d.ema_critic) if owner is not None]
    modes = [(module, module.training) for owner in modules for module in owner.modules()]
    parameters = list(dict.fromkeys([*(p for owner in modules for p in owner.parameters()),
                                    policy.table, policy.log_output_sigma]))
    gradients = [(p, None if p.grad is None else p.grad.detach().clone()) for p in parameters if p is not None]
    if any(g is not None and not bool(torch.isfinite(g).all()) for _, g in gradients):
        raise ValueError("nonfinite saved learned gradient")
    global_cpu = torch.get_rng_state().clone()
    global_cuda = torch.cuda.get_rng_state(policy.device).clone() if policy.device.type == "cuda" else None
    dv12 = torch.Generator(device=policy.device); dv12.set_state(policy.noise_generator.get_state())
    gaussian_rng = torch.Generator(); gaussian_rng.set_state(paired_rng.get_state())
    role_parameters = _roles(policy)
    rows = {component: [] for component in ("total_dv12_gaussian", "clean_fixed_gaussian")}
    try:
        for context, target in batches:
            if len(context) != 4:
                raise ValueError("every batch must have four native contexts")
            condition = policy.encoder.condition(context)
            clean = policy.routed_generate(context, sigma=0, perturb=False)
            observations = {component: {role: [[], [], []] for role in (*role_parameters, "residual_upstream")}
                            for component in rows}
            for _ in range(PAIRS):
                deadline()
                torch.randn(target.shape, generator=gaussian_rng)  # native caller critic draw
                gaussian = torch.randn(target.shape, generator=gaussian_rng).to(target)
                perturbed = policy.routed_generate(context, sigma=0, perturb=True, stream=dv12)
                for component, prediction in (("total_dv12_gaussian", perturbed),
                                              ("clean_fixed_gaussian", clean)):
                    plus, minus, average = paired_gradients(policy, prediction, target, condition, gaussian, role_parameters)
                    for role in observations[component]:
                        single, paired, transported = observations[component][role]
                        single.extend((plus[role], minus[role])); paired.append(average[role])
                        transported.append((plus[role]+minus[role])/2)
                del perturbed
            for component in rows:
                rows[component].append({role: _statistics(*(torch.stack(values) for values in triples))
                                        for role, triples in observations[component].items()})
        deadline()
        if digest(policy.state_dict()) != before:
            raise AssertionError("zero-update probe changed native state or RNG")
        if any(not torch.equal(p.grad, g) if g is not None else p.grad is not None for p, g in gradients):
            raise AssertionError("autograd.grad changed stored gradients")
        if any(module.training != mode for module, mode in modes):
            raise AssertionError("probe changed module modes")
        summary = {}
        for component, batches_ in rows.items():
            summary[component] = {}
            for role in batches_[0]:
                single_v = sum(row[role]["single_variance"] for row in batches_)/BATCHES
                paired_v = sum(row[role]["antithetic_variance"] for row in batches_)/BATCHES
                summary[component][role] = {"single_variance": single_v, "antithetic_variance": paired_v,
                    "ratio": paired_v/single_v if single_v > 0 else None,
                    "mean_gradient_energy": sum(row[role]["mean_gradient_energy"] for row in batches_)/BATCHES,
                    "batch_statistics": [row[role] for row in batches_]}
        primary = summary["total_dv12_gaussian"]["generator"]
        return {"pass": variance_gate(primary["single_variance"], primary["antithetic_variance"]),
                "criterion": "Within-batch full generator role total(DV12+Gaussian) variance ratio <= .75",
                "pairs_per_batch": PAIRS, "batches": BATCHES, "sigma": SIGMA,
                "native_updates": 0, "state_unchanged_before_restore": True,
                "state_digest": before, "private_dv12_advanced": not torch.equal(dv12.get_state(), saved["streams"]["noise_generator"]),
                "private_gaussian_advanced": not torch.equal(gaussian_rng.get_state(), paired_rng.get_state()),
                "statistics": summary,
                "limits": "Variance gate only; no convergence/faster/Supra claim; antithetic cannot remove a biased mean critic force."}
    finally:
        policy.load_state_dict(saved)
        for module, mode in modes: module.training = mode
        for p, gradient in gradients: p.grad = None if gradient is None else gradient.clone()
        torch.set_rng_state(global_cpu)
        if global_cuda is not None: torch.cuda.set_rng_state(global_cuda, policy.device)
        if digest(policy.state_dict()) != before:
            raise AssertionError("final native state restoration failed")


class TinyHost(nn.Module):
    def __init__(self, bf16, code_gain):
        super().__init__(); self.first = nn.Linear(2, 2); self.second = nn.Linear(2, 2); self.bf16 = bf16
        self.register_buffer("code_gain", torch.tensor(code_gain))
    def layer(self, layer, hidden):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            return layer(hidden).float()


class TinyCondition(nn.Module):
    def forward(self, context): return context
    def condition(self, context): return context[:, 0, 0]


class TinyRouter(nn.Module):
    def __init__(self):
        super().__init__(); self.first = nn.Linear(2, 4); self.second = nn.Linear(2, 4)
        self.register_buffer("log_mass", torch.zeros(128))


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.hidden = nn.Linear(2, 4); self.score = nn.Linear(4, 1)
        self.register_buffer("scale", torch.ones(2))
    def features(self, error, condition):
        return (self.hidden(error)+condition[:, None, :1]+.7).tanh()
    def forward(self, error, condition): return self.score(self.features(error, condition)).mean(1)


def tiny_forward(models, context, candidate, routing):
    generator, router = models["generator"], models["router"]
    hidden = context
    for site in ("first", "second"):
        logits = getattr(router, site)(hidden) @ candidate.table.T / 2
        codes = routing.mix(site, logits)
        hidden = generator.layer(getattr(generator, site), hidden).tanh()+generator.code_gain*codes[..., :2]
    return .001*(3*hidden[:, 0]-2*hidden[:, 1])


def software_fixture(*, bf16=False, fixture="software"):
    """Two fixed public routed fixtures, not a coefficient/seed search.

    Software uses1e-5 code coupling to isolate Gaussian estimator transport.
    Routed-DV12 uses coefficient1 so private latent draws also change the
    next site's host inputs, queries and full-generator Jacobian. Both use
    identical named initialization/noise/context streams and native loss.
    """
    if fixture not in ("software", "routed-dv12"): raise ValueError("unknown fixed fixture")
    recipe = get_recipe("e22_routed", num_particles=128, z_dim=4, batch_size=4, output_noise_std=SIGMA,
                        reopen_guard="settled")
    with torch.random.fork_rng(devices=[]):
        generator, critic, router, prior = TinyHost(bf16, 1e-5 if fixture == "software" else 1.), TinyCritic(), TinyRouter(), recipe.make_prior()
    for index, owner in enumerate((generator, critic, router)):
        streams = {name: torch.Generator().manual_seed(1000*index+j)
                   for j, (name, _) in enumerate(owner.named_parameters())}
        init.initialize_(owner, method="sample_distributions_v1", parameter_generators=streams)
    init.deterministic_orthogonal_(prior)
    table = prior.z.requires_grad_(True)
    optimizer = recipe.make_generator_optimizer([
        {"params": list(generator.parameters()), "lr": 5e-5},
        {"params": list(router.parameters()), "lr": 5e-5},
        {"params": [table], "lr": recipe.lr*recipe.prior_lr_mult}], latent_table=table, foreach=False)
    critic_optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    rows = RoutedRows(model_forward=tiny_forward,
        features=lambda models, context, samples, targets: models["critic"].features(samples-targets, context[:, 0, 0]).flatten(1),
        sites=("first", "second"), max_context_harm=0., output_error_guard=False)
    policy = E22Policy(recipe, generator, critic, table=table, encoder=TinyCondition(), router=router,
        generator_optimizer=optimizer, critic_optimizer=critic_optimizer,
        roles=[["generator", "router", "table"], ["critic"]], routed_rows=rows, seed=21)
    policy.attach_penalty(recipe.make_critic_penalty(critic_optimizer, collect_stats=True))
    context = torch.randn(8, 2, 3, 2, generator=torch.Generator().manual_seed(72))
    batches = [(value, torch.zeros(4, 3, 2)) for value in context.split(4)]
    return policy, batches, torch.Generator().manual_seed(43)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("software", "routed-dv12"), default="software")
    parser.add_argument("--bf16", action="store_true"); parser.add_argument("--out", type=Path)
    args = parser.parse_args(); torch.set_num_threads(1)
    policy, batches, stream = software_fixture(bf16=args.bf16, fixture=args.fixture)
    result = probe(policy, batches, stream)
    result.update(fixture=args.fixture, estimator_fixture_only=True, ownership_checks="PASS",
                  variance_result="PASS" if result["pass"] else "FAIL", elapsed_seconds=time.monotonic()-STARTED)
    if args.out: args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"fixture": args.fixture, "ownership_checks": "PASS", "variance_result": result["variance_result"],
                      "full_G_ratio": result["statistics"]["total_dv12_gaussian"]["generator"]["ratio"],
                      "elapsed_seconds": result["elapsed_seconds"], "updates": 0}))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__": main()
