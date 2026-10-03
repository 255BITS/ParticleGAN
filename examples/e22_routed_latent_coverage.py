"""Two versus eight latent trajectories through the public routed ParticleGAN API.

This generated convergence diagnostic changes training support only. It is not
a native bug fix, a full-Supra result, or a default-recipe qualification.
For --run: exit 0 numerical PASS; 1 numerical FAIL; 2 incomplete.
--software-prerequisite runs CPU construction/scorer/API recovery only, with
qualification PASS/FAIL and no scientific status. Source/protocol review must
precede either mode; the generated cohort has no Supra BF16 parity claim.
"""
import time
STARTED = time.monotonic()
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from particlegan import E22Policy, RoutedRows, get_recipe, init
from examples import e22_routed_caption_accuracy as public_loop

TASK = "routed_latent_coverage_v2"
ARMS = ("two_seed", "eight_seed")
SITES = ("input", "edit")
TRAIN_SEEDS = tuple(range(7063, 7071))
TEST_SEEDS = (39001, 39002)
TIMES = tuple(i / 10 for i in range(10))
PATHS = ("neutral", "positive")
N, Z, B, RANK, STEPS, LIMIT = 32, 4, 4, 4, 1024, 300
SOFTWARE_LIMIT = 60
ACTIVE_LIMIT = LIMIT
SCORE_STEPS = (896, 928, 960, 992, 1024)
MEDIA_STEPS = (0, 128, 256, 512, 800, 1024)
MEDIA_INDICES = (0, 40, 80, 120, 160, 200)
GEOMETRY = public_loop.Geometry(width=32, text=16, rank=4, tokens=16,
                               length=1, heads=4, output=16, frequency=8)
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_latent_coverage_v2.json"
SOURCE_FILES = ("examples/e22_routed_latent_coverage.py",
                "examples/render_e22_routed_latent_coverage.py",
                "examples/e22_routed_caption_accuracy.py")
THRESHOLDS = dict(baseline_test_fit_mse_ratio=1.10, relative_rmse_gain=.001,
                  source_harm=1e-6, seed_harm=1e-6, gap_fraction=.9,
                  common_game_gain=1e-6, code_gain=1e-6, live_fraction=.9)
RECIPE_OPTIONS = dict(num_particles=N, z_dim=Z, batch_size=B,
                      output_noise_std=.125, birth_death_backend="auto",
                      reopen_guard="settled", standardize=False)
LAW = dict(task=TASK, arms=list(ARMS), geometry=asdict(GEOMETRY), sites=list(SITES),
           steps=STEPS, seconds=LIMIT, score_steps=list(SCORE_STEPS),
           media_steps=list(MEDIA_STEPS), media_indices=list(MEDIA_INDICES),
           train_seeds=list(TRAIN_SEEDS), baseline_seeds=list(TRAIN_SEEDS[:2]),
           test_seeds=list(TEST_SEEDS), times=list(TIMES), paths=list(PATHS),
           trajectory_steps=50, guidance=3., recipe_options=RECIPE_OPTIONS,
           branch_router_lr=5e-4, thresholds=THRESHOLDS,
           routing="shared-Up gated V3; H/b neutral; both sites sequential",
           initializer="public initialize_/sample_distributions_v1, named CPU generators",
           bank_initializer="public initialize_, Normal(0,1) override for z, CPU role bank",
           prior=dict(kind="particle_cloud", sigma=0., standardize=False,
                      learnable=True, exception="conditional dense routed bank; no MoG sampling"),
           support_counts=dict(two_seed=240, eight_seed=960, test=240, guard=64),
           support_count_units="ordered records; neutral/positive paths share each initial t0 context",
           unique_context_counts=dict(two_seed=228, eight_seed=912, test=228, guard=64),
           addressed_training_slots=960, sampled_contexts_per_arm=4096,
           paired_addressing="slot=160*source+20*j+10*path+time_index; baseline seed=floor(j/4), expanded seed=j",
           initial_noise="one CPU seed panel reused across all sources and both paths",
           coordinate_units="original two-seed FIT240 std(correction1), clamp_min .04; shared",
           guard="64 evenly selected adjacent-time midpoints of original FIT240; editing only",
           native_KA2="one penalty call/update; pure A calls1..799; blend starts800",
           arithmetic="generated carrier, teacher, adapters and critic all FP32; autocast disabled; not Supra BF16 parity",
           checkpoint_recovery="fresh owners for both trained checkpoint replay and unresolved initial checkpoint restore",
           software_prerequisite=dict(device="cpu", seconds=SOFTWARE_LIMIT,
                                      native_updates=6, optimizer_steps=12,
                                      scope="software/API compatibility only; no scientific gate/rank"),
           clean_serving="FAST sigma0/perturbFalse; accuracy offline only",
           common_game="both frozen terminal critics, 4 paired Gaussian panels at sigma .125",
           prohibited_feedback="no output objective/guard/selection/stopping or applied accuracy gradient")


def budget():
    if time.monotonic() - STARTED > ACTIVE_LIMIT:
        raise TimeoutError(f"fixed{ACTIVE_LIMIT}s startup-through-final-writes allowance exceeded")


def sources():
    root = CARD.parents[1]
    return {name: public_loop.sha(root / name) for name in SOURCE_FILES}


def native_sources(native):
    """Relative byte identities; installed location is provenance only."""
    root = Path(native["imported_package"])
    return {str(Path("particlegan") / path.relative_to(root)): public_loop.sha(path)
            for path in sorted(root.rglob("*.py"))}


def initialize(module, role):
    """Explicit public initialization; no constructor/global RNG is evidence."""
    declared = init.declarations(module)
    streams = {name: torch.Generator(device="cpu").manual_seed(
        int.from_bytes(hashlib.sha256(f"routed-latent-coverage-v1:{role}:{name}".encode()).digest()[:8], "little") % (2**63 - 1))
        for name, value in module.named_parameters()
        if value.requires_grad and value.numel() and declared[name] is not init.KEEP}
    init.initialize_(module, method="sample_distributions_v1", parameter_generators=streams)


class Carrier(nn.Module):
    """One frozen conditional velocity field, with no cancellation projection."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(34, 32)
        self.edit = nn.Linear(32, 16)

    def forward(self, features):
        with torch.autocast(features.device.type, enabled=False):
            return self.edit(self.input(features.float()).tanh())


def features_for_velocity(z, t, caption):
    t = torch.as_tensor(t, dtype=torch.float32, device=z.device).reshape(-1)
    if t.numel() == 1:
        t = t.expand(len(z))
    if t.shape != (len(z),) or caption.shape != (len(z), 16):
        raise ValueError("velocity needs matching caption/time/context rows")
    return torch.cat((z, caption[:, None].expand(-1, 16, -1),
                      t[:, None, None].expand(-1, 16, 1),
                      torch.sin(math.pi * t)[:, None, None].expand(-1, 16, 1)), -1)


def cfg_velocity(carrier, z, t, caption):
    z = torch.cat((z, z))
    caption = torch.cat((caption, torch.zeros_like(caption)))
    t = torch.as_tensor(t, device=z.device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        t = t.expand(len(z) // 2)
    conditional, unconditional = carrier(features_for_velocity(z, torch.cat((t, t)), caption)).chunk(2)
    return unconditional + 3 * (conditional - unconditional)


class SharedUpProjection(nn.Module):
    """Client-owned GATED_V3 formula; every mix delegates to public routing."""
    def __init__(self, frozen, site):
        super().__init__()
        self.base = frozen.requires_grad_(False)
        self.down = nn.Linear(frozen.in_features, RANK, bias=False)
        self.bridge = nn.Linear(RANK + Z, RANK)
        self.up = nn.Linear(RANK, frozen.out_features, bias=False)
        self.site, self.route = site, None

    def forward(self, x):
        if self.route is None or x.ndim != 3 or len(x) % 2:
            raise ValueError("one actual public routed CFG frame required")
        router, candidate, routing = self.route
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float()
            logits = router.queries[self.site](xf) @ candidate.table.float().T / math.sqrt(Z)
            grouped = torch.stack(logits.chunk(2), dim=1)
            codes = routing.mix(self.site, grouped)
            codes = torch.cat((codes[:, 0], codes[:, 1]))
            h = self.down(xf)
            m = h + F.linear(h, self.bridge.weight[:, :RANK], self.bridge.bias).tanh()
            m = m + h * F.linear(codes.float(), self.bridge.weight[:, RANK:]).tanh()
            return self.base(xf) + self.up(m)


class Host(nn.Module):
    def __init__(self, frozen, captions):
        super().__init__()
        self.backbone, self.teacher_backbone = Carrier(), Carrier()
        self.backbone.load_state_dict(frozen, strict=True)
        self.teacher_backbone.load_state_dict(frozen, strict=True)
        self.backbone.requires_grad_(False)
        self.teacher_backbone.requires_grad_(False)
        self.register_buffer("captions", captions.clone())
        for site in SITES:
            setattr(self.backbone, site, SharedUpProjection(getattr(self.backbone, site), site))

    def branches(self):
        return [getattr(self.backbone, site) for site in SITES]

    def forward(self, x):
        raise ValueError("particle host requires public routed_generate")

    def forward_routed(self, x, router, candidate, routing):
        if any(branch.route is not None for branch in self.branches()):
            raise RuntimeError("overlapping routed execution")
        z, t = x[..., :16], x[:, 0, 17]
        ids = x[:, 0, 16].long()
        if not torch.equal(ids.float(), x[:, 0, 16]) or bool((ids < 0).any()) or bool((ids >= 6).any()):
            raise ValueError("six integer source identities required")
        with torch.no_grad():
            target = cfg_velocity(self.teacher_backbone, z, t, self.captions[ids + 7, 0])
        try:
            for branch in self.branches():
                branch.route = (router, candidate, routing)
            prediction = cfg_velocity(self.backbone, z, t, self.captions[ids + 1, 0])
            return prediction - target
        finally:
            for branch in self.branches():
                branch.route = None


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.ModuleDict(dict(input=nn.Linear(34, Z), edit=nn.Linear(32, Z)))
        self.register_buffer("log_mass", torch.zeros(N))


@torch.no_grad()
def trajectory_pool(carrier, captions, seeds):
    """A seed defines one initial tensor shared across all six caption pairs."""
    initial = {seed: torch.randn(1, 16, 16,
                                generator=torch.Generator(device="cpu").manual_seed(seed))
               for seed in seeds}
    contexts, metadata = [], dict(source_ids=[], seeds=[], paths=[], times=[])
    initial_hashes = {str(seed): public_loop.digest(value) for seed, value in initial.items()}
    for source in range(6):
        for seed in seeds:
            for path_index, path in enumerate(PATHS):
                z = initial[seed].clone()
                caption = captions[source + (1 if path_index == 0 else 7), 0].unsqueeze(0)
                for step in range(50):
                    t = step / 50
                    if step % 5 == 0:
                        tail = torch.tensor([source, t]).expand(1, 16, -1)
                        contexts.append(torch.cat((z, tail), -1).squeeze(0).clone())
                        for key, value in (("source_ids", source), ("seeds", seed), ("paths", path), ("times", t)):
                            metadata[key].append(value)
                    z = z + cfg_velocity(carrier, z, t, caption) / 50
                budget()
    context = torch.stack(contexts)
    if not bool(torch.isfinite(context).all()):
        raise FloatingPointError("frozen trajectory context is nonfinite")
    return dict(context=context, targets=torch.zeros(len(context), 16, 16),
                metadata=metadata, initial_noise_sha256=initial_hashes)


@torch.no_grad()
def make_data(device):
    with torch.random.fork_rng(devices=[]):
        carrier = Carrier()
    initialize(carrier, "frozen_carrier")
    carrier.requires_grad_(False).eval()
    captions = torch.zeros(13, 1, 16)
    captions[1:, 0] = torch.eye(16)[:12]
    master = trajectory_pool(carrier, captions, TRAIN_SEEDS)
    original = trajectory_pool(carrier, captions, TRAIN_SEEDS[:2])
    test = trajectory_pool(carrier, captions, TEST_SEEDS)
    indices = [s * 160 + k * 20 + path * 10 + t
               for s in range(6) for k in range(2) for path in range(2) for t in range(10)]
    if not torch.equal(master["context"][indices], original["context"]):
        raise AssertionError("original two-seed subset differs byte-for-byte")
    if set(TRAIN_SEEDS) & set(TEST_SEEDS):
        raise AssertionError("TEST seed leaked into fitting")
    midpoints = []
    for start in range(0, 240, 10):
        rows = original["context"][start:start + 10]
        midpoints.extend((rows[:-1] + rows[1:]) * .5)
    midpoints = torch.stack(midpoints)
    _, inverse = torch.unique(midpoints, dim=0, return_inverse=True)
    seen, first = set(), []
    for index, group in enumerate(inverse.tolist()):
        if group not in seen:
            seen.add(group)
            first.append(index)
    candidates = midpoints[first]
    if len(candidates) < 64:
        raise ValueError("original editing trajectories cannot supply64 distinct guards")
    guard_indices = torch.linspace(0, len(candidates) - 1, 64).round().long()
    guard = dict(context=candidates[guard_indices], targets=torch.zeros(64, 16, 16))
    distinct = {name: len(torch.unique(pool["context"], dim=0)) for name, pool in
                (("two_seed", original), ("eight_seed", master), ("test", test), ("guard", guard))}
    if distinct != LAW["unique_context_counts"]:
        raise ValueError("actual distinct contexts differ from initial-path multiplicity law")
    whole = torch.cat((master["context"], guard["context"], test["context"]))
    if len(torch.unique(whole, dim=0)) != sum(distinct[name] for name in ("eight_seed", "guard", "test")):
        raise ValueError("FIT/guard/TEST identities overlap")
    baseline = []
    for begin in range(0, 240, B):
        x = original["context"][begin:begin + B]
        ids = x[:, 0, 16].long()
        baseline.append(cfg_velocity(carrier, x[..., :16], x[:, 0, 17], captions[ids + 1, 0])
                        - cfg_velocity(carrier, x[..., :16], x[:, 0, 17], captions[ids + 7, 0]))
    scale = torch.cat(baseline).flatten(0, 1).std(0, correction=1).clamp_min(.04)
    common = dict(geometry=GEOMETRY, frozen=deepcopy(carrier.state_dict()), captions=captions,
                  masks=torch.ones(13, 1, dtype=torch.bool), scale=scale,
                  original_fit=original, expanded_fit=master, test=test, guard=guard)
    data = {}
    for arm in ARMS:
        addressed = [s * (40 if arm == ARMS[0] else 160)
                     + (k // 4 if arm == ARMS[0] else k) * 20 + path * 10 + t
                     for s in range(6) for k in range(8) for path in range(2) for t in range(10)]
        pool = original if arm == ARMS[0] else master
        table = pool["context"][addressed]
        addresses = [[s, TRAIN_SEEDS[k // 4 if arm == ARMS[0] else k], PATHS[path], t / 10]
                     for s in range(6) for k in range(8) for path in range(2) for t in range(10)]
        decoded = [[s, k, path, t] for s in range(6) for k in range(8) for path in range(2) for t in range(10)]
        if len(pool["context"]) != LAW["support_counts"][arm] or len(torch.unique(table, dim=0)) != distinct[arm]:
            raise AssertionError("actual fitting support cardinality differs")
        value = {**common, "fit": dict(context=table, targets=torch.zeros(960, 16, 16),
                                      addresses=addresses, decoded_slots=decoded, pool_indices=addressed)}
        value["digest"] = public_loop.digest({k: asdict(v) if isinstance(v, public_loop.Geometry) else v
                                              for k, v in value.items()})
        data[arm] = value
    evidence = dict(original_subset_exact=True, globally_shared_seed_initial_tensors=True,
                    old_guard_and_scale_shared=True, support_counts=LAW["support_counts"],
                    unique_context_counts=distinct, cross_pool_disjoint=True,
                    initial_noise_sha256=master["initial_noise_sha256"],
                    test_initial_noise_sha256=test["initial_noise_sha256"],
                    guard_sha256=public_loop.digest(guard), scale_sha256=public_loop.digest(scale),
                    original_fit_sha256=public_loop.digest(original), expanded_fit_sha256=public_loop.digest(master),
                    test_sha256=public_loop.digest(test), data_digests={arm: value["digest"] for arm, value in data.items()})
    # Only actual training tensors and held pools move. Shared immutable capsules
    # remain CPU and are copied into newly constructed registered owners.
    for value in data.values():
        value["scale"] = scale.to(device)
        for name in ("fit", "original_fit", "expanded_fit", "test", "guard"):
            value[name] = {**value[name], **{key: value[name][key].to(device) for key in ("context", "targets")}}
    return data, evidence


def make_loop(arm, data, device):
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        g = Host(data["frozen"], data["captions"])
        d = public_loop.Critic(data["scale"].cpu(), GEOMETRY)
        e = public_loop.Encoder(data["captions"], data["masks"], GEOMETRY)
        r = Router()
        for owner, role in ((g, "generator"), (d, "critic"), (r, "router")):
            initialize(owner, role)
        with torch.no_grad():
            for branch in g.branches():
                branch.up.weight.zero_()
                branch.bridge.weight[:, :RANK].zero_()
                branch.bridge.bias.zero_()
        g, d, e, r = (owner.to(device) for owner in (g, d, e, r))
        recipe = get_recipe("e22_routed", **RECIPE_OPTIONS)
        prior = recipe.make_prior()
        init.initialize_(prior, method="sample_distributions_v1",
                         distributions={"z": init.Normal(0., 1.)},
                         parameter_generators={"z": torch.Generator(device="cpu").manual_seed(4)})
        table = prior.to(device).z
        og = recipe.make_generator_optimizer([
            dict(params=[p for p in g.parameters() if p.requires_grad], lr=LAW["branch_router_lr"]),
            dict(params=list(r.parameters()), lr=LAW["branch_router_lr"]),
            dict(params=[table], lr=recipe.lr * recipe.prior_lr_mult)], latent_table=table, foreach=False)
        od = recipe.make_critic_optimizer(d, ema_critic=deepcopy(d), foreach=False)
        rows = RoutedRows(model_forward=public_loop.model_forward, features=public_loop.features,
                          sites=SITES, probe_interval=100, max_context_harm=0., output_error_guard=False)
        p = E22Policy(recipe, g, d, table=table, encoder=e, router=r,
                      generator_optimizer=og, critic_optimizer=od,
                      roles=[["generator", "router", "table"], ["critic"]], seed=21, routed_rows=rows)
        p.attach_penalty(recipe.make_critic_penalty(od, collect_stats=True))
        torch.manual_seed(900)
        if device.type == "cuda":
            torch.cuda.manual_seed(900)
        owned = public_loop.global_rng(device)
    return public_loop.Loop(p, arm, data, torch.Generator().manual_seed(7),
                            torch.Generator().manual_seed(43), owned)


def metric(residual, metadata):
    if (residual.shape != (240, 16, 16) or not residual.is_floating_point()
            or not bool(torch.isfinite(residual).all())):
        raise ValueError("finite full240 physical residuals required")
    if (len(metadata["source_ids"]) != 240 or set(metadata["source_ids"]) != set(range(6))
            or len(metadata["seeds"]) != 240 or len(set(metadata["seeds"])) != 2):
        raise ValueError("six sources and two declared seed identities required")
    powers = residual.detach().double().square().flatten(1).mean(1).cpu().tolist()
    mean = math.fsum(powers) / 240
    return dict(count=240, mse=mean, rmse=math.sqrt(mean),
                by_source={str(s): math.sqrt(math.fsum(p for p, i in zip(powers, metadata["source_ids"]) if i == s) / 40) for s in range(6)},
                by_seed={str(seed): math.sqrt(math.fsum(p for p, i in zip(powers, metadata["seeds"]) if i == seed) / 120) for seed in sorted(set(metadata["seeds"]))})


def game_mean(array):
    if array.shape != (240, 4) or not bool(torch.isfinite(array).all()):
        raise ValueError("finite240x4 common native game scores required")
    return math.fsum(float(row.double().mean()) for row in array) / 240


def scientific_gate(quality, fit_quality, games, zero, retention):
    """Saved-score classifier only. This function supplies no training signal."""
    if set(quality) != set(ARMS) or set(fit_quality) != set(ARMS):
        raise ValueError("both full mandatory arms required")
    for table in (quality, fit_quality):
        for arm in ARMS:
            if set(table[arm]) != {str(step) for step in SCORE_STEPS}:
                raise ValueError("all five fixed terminal endpoints required")
            for value in table[arm].values():
                nums = (value["mse"], value["rmse"], *value["by_source"].values(), *value["by_seed"].values())
                if (value["count"] != 240 or set(value["by_source"]) != {str(s) for s in range(6)}
                        or len(value["by_seed"]) != 2 or not all(math.isfinite(x) and x >= 0 for x in nums)):
                    raise ValueError("finite six-source/two-seed240-row metrics required")
    baseline, expanded = (quality[arm][str(STEPS)] for arm in ARMS)
    fit_base, fit_expanded = (fit_quality[arm][str(STEPS)] for arm in ARMS)
    gap_base, gap_expanded = baseline["mse"] - fit_base["mse"], expanded["mse"] - fit_expanded["mse"]
    checks = {
        "baseline_fit_test_phenotype": fit_base["mse"] > 0 and baseline["mse"] >= THRESHOLDS["baseline_test_fit_mse_ratio"] * fit_base["mse"],
        "held_RMSE_improved_all_five": all(quality[ARMS[1]][str(t)]["rmse"] <= (1 - THRESHOLDS["relative_rmse_gain"]) * quality[ARMS[0]][str(t)]["rmse"] for t in SCORE_STEPS),
        "no_source_harmed_all_five": all(quality[ARMS[1]][str(t)]["by_source"][str(s)] <= quality[ARMS[0]][str(t)]["by_source"][str(s)] + THRESHOLDS["source_harm"] for t in SCORE_STEPS for s in range(6)),
        "no_TEST_seed_harmed_all_five": all(quality[ARMS[1]][str(t)]["by_seed"][str(s)] <= quality[ARMS[0]][str(t)]["by_seed"][str(s)] + THRESHOLDS["seed_harm"] for t in SCORE_STEPS for s in TEST_SEEDS),
        "common_originalFIT_gap_reduced": gap_base > 0 and gap_expanded <= THRESHOLDS["gap_fraction"] * gap_base,
        "both_common_games_improved_all_five": set(games) == {"two_seed_final", "eight_seed_final"} and all(games[j][ARMS[0]][str(t)] - games[j][ARMS[1]][str(t)] > THRESHOLDS["common_game_gain"] for j in games for t in SCORE_STEPS),
        "code_RMSE_beneficial": zero["rmse"] - expanded["rmse"] > THRESHOLDS["code_gain"],
        "code_RMSE_beneficial_each_source": all(zero["by_source"][str(s)] > expanded["by_source"][str(s)] for s in range(6)),
        "code_game_beneficial_both_judges": all(games[j]["eight_seed_zero_code"][str(STEPS)] - games[j][ARMS[1]][str(STEPS)] > THRESHOLDS["code_gain"] for j in games),
        "particles_retained_both_arms": all(retention[a]["bank_live"] / (STEPS - 1) >= THRESHOLDS["live_fraction"] and retention[a]["dense_rows_live"] / (STEPS - 1) >= THRESHOLDS["live_fraction"] and retention[a]["query_live"] / (STEPS - 1) >= THRESHOLDS["live_fraction"] and retention[a]["bank_changed"] and retention[a]["router_changed"] and retention[a]["finite"] and set(retention[a]["C_norms"]) == set(SITES) and set(retention[a]["Up_norms"]) == set(SITES) and all(math.isfinite(v) and v > 0 for v in (*retention[a]["C_norms"].values(), *retention[a]["Up_norms"].values())) for a in ARMS),
        "native_KA2_phase_qualified": all(retention[a]["KA2_phase"]["799"] == "a" and retention[a]["KA2_phase"]["800"] == "blend" and retention[a]["KA2_phase"]["1024"] == "blend" for a in ARMS),
    }
    return dict(pass_=all(checks.values()), checks=checks, failed=[name for name, ok in checks.items() if not ok],
                baseline_gap=gap_base, expanded_gap=gap_expanded, thresholds=THRESHOLDS,
                scope="Generated support-size diagnostic; not a native bug/full-Supra/default qualification")


def scorer_controls():
    def m(rmse, fit=False):
        return dict(count=240, mse=rmse * rmse, rmse=rmse,
                    by_source={str(s): rmse for s in range(6)},
                    by_seed={str(s): rmse for s in (TRAIN_SEEDS[:2] if fit else TEST_SEEDS)})
    q = {a: {str(t): m(1. if a == ARMS[0] else .95) for t in SCORE_STEPS} for a in ARMS}
    f = {a: {str(t): m(.8, True) for t in SCORE_STEPS} for a in ARMS}
    g = {j: {**{a: {str(t): (1. if a == ARMS[0] else .95) for t in SCORE_STEPS} for a in ARMS}, "eight_seed_zero_code": {str(STEPS): 1.}} for j in ("two_seed_final", "eight_seed_final")}
    retained = {a: dict(bank_live=1023, dense_rows_live=1023, query_live=1023, bank_changed=True,
                       router_changed=True, finite=True, C_norms={s: 1. for s in SITES}, Up_norms={s: 1. for s in SITES},
                       KA2_phase={"799": "a", "800": "blend", "1024": "blend"}) for a in ARMS}
    if not scientific_gate(q, f, g, m(1.), retained)["pass_"]:
        raise AssertionError("positive numerical gate oracle rejected")
    controls = {}
    for name in ("held_only_fit_improvement", "source_harm", "seed_harm", "no_phenotype", "harmful_code", "dead_bank", "losing_common_critic", "wrong_KA2_phase"):
        qq, ff, gg, rr, zz = deepcopy(q), deepcopy(f), deepcopy(g), deepcopy(retained), m(1.)
        if name == "held_only_fit_improvement": qq[ARMS[1]]["1024"] = m(1.)
        elif name == "source_harm": qq[ARMS[1]]["1024"]["by_source"]["5"] = 1.000002
        elif name == "seed_harm": qq[ARMS[1]]["1024"]["by_seed"]["39002"] = 1.000002
        elif name == "no_phenotype": ff[ARMS[0]]["1024"] = m(1., True)
        elif name == "harmful_code": zz = m(.94)
        elif name == "dead_bank": rr[ARMS[1]]["bank_live"] = 0
        elif name == "losing_common_critic": gg["two_seed_final"][ARMS[1]]["1024"] = 1.01
        elif name == "wrong_KA2_phase": rr[ARMS[1]]["KA2_phase"]["800"] = "a"
        if scientific_gate(qq, ff, gg, zz, rr)["pass_"]:
            raise AssertionError("destructive gate control passed: " + name)
        controls[name] = True
    broken = deepcopy(q); broken[ARMS[1]]["1024"]["rmse"] = float("nan")
    try: scientific_gate(broken, f, g, m(1.), retained)
    except ValueError: controls["nonfinite"] = True
    else: raise AssertionError("nonfinite gate control passed")
    return dict(positive_oracle=True, destructive_controls=controls)


def preflight(data, device):
    loops = {a: make_loop(a, data[a], device) for a in ARMS}
    a, b = (loops[arm] for arm in ARMS)
    if public_loop.digest(a.policy.state_dict()) != public_loop.digest(b.policy.state_dict()):
        raise AssertionError("initial native tensors/EMA/optimizer/control states differ")
    if public_loop.digest(a.globals) != public_loop.digest(b.globals):
        raise AssertionError("initial owned native penalty streams differ")
    camera = data[ARMS[0]]["test"]["context"][list(MEDIA_INDICES)]
    if not torch.equal(public_loop.observe(a, camera), public_loop.observe(b, camera)):
        raise AssertionError("common initial clean predictions differ")
    recovery = {}
    for arm in ARMS:
        loop = loops[arm]; entry = public_loop.checkpoint(loop)
        public_loop.update(loop); middle = public_loop.checkpoint(loop)
        second = public_loop.update(loop); direct = public_loop.checkpoint(loop)
        rebuilt = make_loop(arm, data[arm], device)
        public_loop.restore(rebuilt, middle)
        replay_second = public_loop.update(rebuilt)
        if (public_loop.digest(public_loop.checkpoint(rebuilt)) != public_loop.digest(direct)
                or public_loop.digest(second) != public_loop.digest(replay_second)):
            raise AssertionError("public native two-update1+1 recovery differs")
        # An initial auto-backend checkpoint has not yet resolved its output
        # shape. Restore it into a fresh owner, whose scope is also unresolved.
        initial_owner = make_loop(arm, data[arm], device)
        public_loop.restore(initial_owner, entry)
        if public_loop.digest(public_loop.checkpoint(initial_owner)) != public_loop.digest(entry):
            raise AssertionError("preflight did not restore initial state")
        loops[arm] = initial_owner
        recovery[arm] = dict(fresh_owner_public_resume_exact=True, restore_initial_exact=True,
                             initial_restore_owner="fresh unresolved policy")
        del rebuilt, loop
        budget()
    return loops, dict(common_initial_state_and_predictions_exact=True, recovery=recovery,
                       native_recovery_updates=6, optimizer_recovery_steps=12)


@torch.no_grad()
def common_game_scores(judges, residuals, test_context, device):
    panels = torch.randn(4, 240, 16, 16, generator=torch.Generator(device="cpu").manual_seed(72)).to(device) * .125
    arrays, scores = {}, {}
    for judge_name, loop in judges.items():
        p = loop.policy; before = public_loop.digest(public_loop.checkpoint(loop))
        arrays[judge_name], scores[judge_name] = {}, {}
        for arm, endpoints in residuals.items():
            arrays[judge_name][arm], scores[judge_name][arm] = {}, {}
            for step, residual in endpoints.items():
                values = torch.empty(240, 4)
                for begin in range(0, 240, B):
                    c = p.encoder.condition(test_context[begin:begin + B])
                    q = residual[begin:begin + B].to(device) / p.D.scale
                    for panel in range(4):
                        real = panels[panel, begin:begin + B]
                        real_scores, fake_scores = p.D(real, c), p.D(real + q, c)
                        for row in range(B):
                            values[begin + row, panel] = p.recipe.make_loss().g_loss(fake_scores[row], real_scores[row]).cpu()
                arrays[judge_name][arm][step] = values
                scores[judge_name][arm][step] = game_mean(values)
                budget()
        if before != public_loop.digest(public_loop.checkpoint(loop)):
            raise AssertionError("common judge rescoring changed native state/RNG")
    return arrays, scores, public_loop.digest(panels)


def run(data, evidence, out, device, source_identity, protocol_sha256):
    controls = scorer_controls()
    loops, prerequisite = preflight(data, device)
    quality, fit_quality, residuals, fit_residuals, media, media_metrics, retention = {}, {}, {}, {}, {}, {}, {}
    matched_stream = None
    checkpoints = {}
    for arm in ARMS:
        loop = loops[arm]; p = loop.policy
        original_table = p.table.detach().clone()
        original_router = public_loop.digest(p.router.state_dict())
        learned = {name for name, value in p.G.named_parameters() if value.requires_grad}
        def frozen_digest():
            return public_loop.digest({name: value for name, value in p.G.state_dict().items() if name not in learned})
        frozen_before = frozen_digest()
        quality[arm], fit_quality[arm], residuals[arm], fit_residuals[arm], media[arm], media_metrics[arm] = {}, {}, {}, {}, {}, {}
        camera = loop.data["test"]["context"][list(MEDIA_INDICES)]
        media[arm]["0"] = public_loop.observe(loop, camera).cpu()
        live = dict(bank_live=0, dense_rows_live=0, query_live=0, proposal_events=0,
                    accepted_proposals=0, accepted_moves=0, KA2_phase={})
        stream = hashlib.sha256()
        loss_ema = None
        for step in range(1, STEPS + 1):
            row = public_loop.update(loop)  # Existing public API orchestration is unchanged.
            row["actual_addresses"] = [loop.data["fit"]["addresses"][i] for i in row["batch_indices"]]
            row["decoded_slots"] = [loop.data["fit"]["decoded_slots"][i] for i in row["batch_indices"]]
            row["actual_pool_indices"] = [loop.data["fit"]["pool_indices"][i] for i in row["batch_indices"]]
            row["KA2_phase"] = p.penalty.last_stats["phase"]
            if step > 1:
                live["bank_live"] += int(row["bank_live"])
                live["query_live"] += int(row["query_live"])
                live["dense_rows_live"] += int(p.table.grad is not None and bool(p.table.grad.ne(0).any(1).all()))
            if step in (799, 800, 801, 1024):
                live["KA2_phase"][str(step)] = row["KA2_phase"]
                expected = "a" if step == 799 else "blend"
                if row["KA2_phase"] != expected:
                    raise AssertionError("actual native KA2 clock/phase differs")
            if row["move"] is not None:
                live["proposal_events"] += 1
                live["accepted_proposals"] += int(row["move"].get("accepted", False))
                live["accepted_moves"] += int(row["move"].get("moves", 0))
            stream.update(json.dumps({key: row[key] for key in ("step", "batch_indices", "paired_bases", "data_rng", "paired_rng", "penalty_globals")}, sort_keys=True).encode())
            with (out / f"{arm}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            loss_ema = row["loss_g"] if loss_ema is None else .98 * loss_ema + .02 * row["loss_g"]
            if step % 64 == 0:
                print(json.dumps(dict(arm=arm, step=step, steps=STEPS, native_G_game=row["loss_g"], native_G_EMA=loss_ema,
                                      native_D_game=row["loss_d_game"], KA2_phase=row["KA2_phase"], seconds=time.monotonic() - STARTED)), flush=True)
            if step in MEDIA_STEPS and step != STEPS:
                media[arm][str(step)] = public_loop.observe(loop, camera).cpu()
            if step in SCORE_STEPS:
                residuals[arm][str(step)] = public_loop.observe(loop, loop.data["test"]["context"]).cpu()
                fit_residuals[arm][str(step)] = public_loop.observe(loop, loop.data["original_fit"]["context"]).cpu()
                quality[arm][str(step)] = metric(residuals[arm][str(step)], loop.data["test"]["metadata"])
                fit_quality[arm][str(step)] = metric(fit_residuals[arm][str(step)], loop.data["original_fit"]["metadata"])
                if step == STEPS:
                    media[arm][str(step)] = residuals[arm][str(step)][list(MEDIA_INDICES)].clone()
            budget()
        public_loop.learned_finite(p)
        if frozen_before != frozen_digest():
            raise AssertionError("frozen teacher/carrier/caption owner changed")
        if matched_stream is None:
            matched_stream = stream.hexdigest()
        elif matched_stream != stream.hexdigest():
            raise AssertionError("common address/Gaussian/native penalty streams differ")
        live.update(bank_changed=not torch.equal(original_table, p.table.detach()),
                    router_changed=original_router != public_loop.digest(p.router.state_dict()), finite=True,
                    eligible_updates=STEPS - 1,
                    C_norms={s: float(b.bridge.weight[:, RANK:].detach().norm()) for s, b in zip(SITES, p.G.branches())},
                    Up_norms={s: float(b.up.weight.detach().norm()) for s, b in zip(SITES, p.G.branches())})
        retention[arm] = live
        for key, value in media[arm].items():
            power = float(value.double().square().mean())
            media_metrics[arm][key] = dict(count=6, mse=power, rmse=math.sqrt(power))
        checkpoints[arm] = public_loop.checkpoint(loop)
        torch.save(checkpoints[arm], out / f"{arm}-checkpoint-1024.pt")
        budget()
    zero_residual = public_loop.observe(loops[ARMS[1]], data[ARMS[1]]["test"]["context"], zero_code=True).cpu()
    zero_quality = metric(zero_residual, data[ARMS[1]]["test"]["metadata"])
    judge_residuals = {**residuals, "eight_seed_zero_code": {str(STEPS): zero_residual}}
    arrays, games, panel_hash = common_game_scores({a + "_final": loops[a] for a in ARMS}, judge_residuals,
                                                 data[ARMS[0]]["test"]["context"], device)
    gate = scientific_gate(quality, fit_quality, games, zero_quality, retention)
    gate["pass"] = gate.pop("pass_")
    raw = dict(task=TASK, arms=ARMS, media_steps=MEDIA_STEPS, media_indices=MEDIA_INDICES,
               media=media, media_metrics=media_metrics, quality=quality, fit_quality=fit_quality,
               endpoint_residuals=residuals, fit_endpoint_residuals=fit_residuals,
               zero_code_residual=zero_residual, zero_code_quality=zero_quality,
               common_games=arrays, retention=retention,
               source_identity=source_identity, protocol_sha256=protocol_sha256,
               test_source_ids=data[ARMS[0]]["test"]["metadata"]["source_ids"],
               test_seeds=data[ARMS[0]]["test"]["metadata"]["seeds"],
               test_paths=data[ARMS[0]]["test"]["metadata"]["paths"],
               test_times=data[ARMS[0]]["test"]["metadata"]["times"],
               fit_source_ids=data[ARMS[0]]["original_fit"]["metadata"]["source_ids"],
               fit_seeds=data[ARMS[0]]["original_fit"]["metadata"]["seeds"],
               fit_paths=data[ARMS[0]]["original_fit"]["metadata"]["paths"],
               fit_times=data[ARMS[0]]["original_fit"]["metadata"]["times"],
               evidence=evidence, capture_policy_caller_RNG_diagnostics_unchanged=True)
    observed = out / "observed-training.pt"
    torch.save(raw, observed)
    for arm, loop in loops.items():
        if public_loop.digest(public_loop.checkpoint(loop)) != public_loop.digest(checkpoints[arm]):
            raise AssertionError("final observations/judges changed native terminal state")
    return dict(task=TASK, complete=True, scientific_status="PASS" if gate["pass"] else "FAIL", gate=gate,
                arms=list(ARMS), quality=quality, fit_quality=fit_quality, zero_code_quality=zero_quality,
                common_games=games, retention=retention, prerequisite=prerequisite, scorer_controls=controls,
                data_evidence=evidence, native_quality_updates=2 * STEPS, native_recovery_updates=6,
                optimizer_quality_steps=4 * STEPS, optimizer_recovery_steps=12,
                recipe={a: loops[a].policy.recipe.to_dict() for a in ARMS},
                media_steps=list(MEDIA_STEPS), media_indices=list(MEDIA_INDICES), common_panel_sha256=panel_hash,
                observed_training_sha256=public_loop.sha(observed), matched_external_stream_sha256=matched_stream,
                checkpoint_sha256={a: public_loop.sha(out / f"{a}-checkpoint-1024.pt") for a in ARMS},
                trace_sha256={a: public_loop.sha(out / f"{a}.jsonl") for a in ARMS},
                scope="Data-support convergence diagnostic with particles/native game retained; no native bug/default/full-Supra claim")


def validate_card(card, source, native, native_files):
    if card.get("status") != "FROZEN_PRE_EXECUTION" or card.get("approved_for_execution") is not True:
        raise ValueError("root source/cost review and frozen pre-execution protocol required")
    if card.get("law") != LAW or card.get("sources") != source:
        raise ValueError("frozen task/source/metric/sampling laws differ")
    if (card.get("native_package") != {"python_sha256": native["python_sha256"]}
            or card.get("native_python_files") != native_files):
        raise ValueError("actual imported public package differs from frozen cohort")


def main():
    global ACTIVE_LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--software-prerequisite", action="store_true",
                      help="CPU60 data/scorer/public preflight only; no scientific verdict")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=CARD)
    args = parser.parse_args()
    ACTIVE_LIMIT = SOFTWARE_LIMIT if args.software_prerequisite else LIMIT
    cohort = "software_prerequisite" if args.software_prerequisite else "scientific_gpu"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    source, native = sources(), public_loop.package_identity()
    native_files = native_sources(native)
    out = entry = device = report = card_hash = None
    error, code = None, 2
    try:
        card = json.loads(args.protocol.read_text())
        card_hash = public_loop.sha(args.protocol)
        validate_card(card, source, native, native_files)
        if args.out.exists():
            raise ValueError("fresh exclusive output required; preserve failed attempts")
        if args.software_prerequisite:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("software prerequisite requires CUDA_VISIBLE_DEVICES='' and CPU only")
            device = torch.device("cpu")
        else:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
                raise ValueError("scientific cohort requires physical GPU0 with CUDA_VISIBLE_DEVICES=0")
            device = torch.device("cuda:0")
        if public_loop.backend_flags() != card["backend_flags"]:
            raise ValueError("FP32 backend law differs")
        out = args.out
        out.mkdir(parents=True)
        entry = public_loop.global_rng(device)
        data, evidence = make_data(device)
        budget()
        if args.software_prerequisite:
            controls = scorer_controls()
            loops, prerequisite = preflight(data, device)
            for loop in loops.values():
                public_loop.learned_finite(loop.policy)
            report = dict(task=TASK, complete=True, scientific_status=None,
                          software_status="PASS", cohort=cohort,
                          qualification="PASS",
                          data_evidence=evidence, scorer_controls=controls,
                          prerequisite=prerequisite, native_quality_updates=0,
                          native_recovery_updates=6, optimizer_quality_steps=0,
                          optimizer_recovery_steps=12,
                          scope="Software/API prerequisite only; no numerical scientific gate or ranking")
            code = 0
        else:
            report = run(data, evidence, out, device, source, card_hash)
            report["cohort"] = cohort
            code = 0 if report["gate"]["pass"] else 1
        report.update(source_identity=source, native_package=native, protocol_sha256=card_hash,
                      native_python_files=native_files, backend_flags=public_loop.backend_flags())
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        import traceback
        traceback.print_exc()
    finally:
        if entry is not None:
            public_loop.set_global_rng(entry, device)
            if public_loop.digest(public_loop.global_rng(device)) != public_loop.digest(entry):
                error = dict(type="AssertionError", message="caller RNG rollback differs")
        torch.set_num_threads(old_threads)
        elapsed = time.monotonic() - STARTED
        if (source != sources() or native != public_loop.package_identity()
                or native_files != native_sources(native)
                or (card_hash is not None and card_hash != public_loop.sha(args.protocol))):
            error = dict(type="AssertionError", message="source/card/native changed during measured cohort")
        if elapsed > ACTIVE_LIMIT:
            error = dict(type="TimeoutError", message=f"whole{ACTIVE_LIMIT}s allowance exceeded")
        if error is not None:
            code = 2
        completion = dict(task=TASK, complete=error is None and report is not None,
                          scientific_status=None if error or report is None else report["scientific_status"],
                          software_status=None if error or report is None else report.get("software_status"),
                          cohort=cohort, error=error, seconds=elapsed, limit_seconds=ACTIVE_LIMIT, exit_code=code,
                          source_identity=source, native_package=native, protocol_sha256=card_hash,
                          native_python_files=native_files,
                          caller_CPU_CUDA_RNG_restored=entry is not None,
                          budget_scope="startup through cleanup/report/completion writes; external watchdog also required")
        if args.software_prerequisite:
            completion["qualification"] = "PASS" if error is None and report is not None else "FAIL"
        if out is not None:
            if report is not None:
                report.update(seconds=elapsed, limit_seconds=ACTIVE_LIMIT)
                (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                completion["report_sha256"] = public_loop.sha(out / "report.json")
            path = out / "completion.json"
            path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
            if time.monotonic() - STARTED > ACTIVE_LIMIT:
                completion.update(complete=False, scientific_status=None, exit_code=2,
                                  error=dict(type="TimeoutError", message=f"final writes{ACTIVE_LIMIT}s exceeded"),
                                  seconds=time.monotonic() - STARTED)
                path.write_text(json.dumps(completion, indent=2, allow_nan=False) + "\n")
                code = 2
        print(json.dumps(dict(completion=completion, gate=None if report is None else report.get("gate")), allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
