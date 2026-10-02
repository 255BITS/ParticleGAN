"""Portable native-game convergence test for G-only clean routed training.

No external models/data or private trainer helpers. See the fixed JSON protocol.
Run the fixed test with --run --out runs/routed-g-clean-v1 (a fresh directory).
Preflight makes no optimizer updates.
"""
import time
STARTED = time.monotonic()

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from particlegan import E22Policy, RoutedBatch, RoutedRows, get_recipe, init


TASK = "routed_G_only_clean_convergence_v1"
ARMS = ("native", "G_clean")
WIDTH, RANK, TOKENS, PARTICLES, Z_DIM, BATCH = 16, 2, 16, 128, 4, 4
STEPS, LIMIT = 512, 300
CURVES = tuple(range(0, STEPS + 1, 64))
JUDGE_STEPS = (128, 512)
NOISY_DRAWS, PANEL_DRAWS, PANEL_SIGMA = 16, 4, .125
WIN_MARGIN, CODE_MARGIN = 1e-4, 1e-6
ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "docs/e22_routed_g_clean_v1.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_sha():
    result = hashlib.sha256()
    for path in sorted((ROOT / "particlegan").rglob("*.py")):
        result.update(str(path.relative_to(ROOT)).encode()); result.update(path.read_bytes())
    return result.hexdigest()


def digest(value):
    result = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            result.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item): result.update(str(key).encode()); add(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item: add(element)
        else:
            result.update(json.dumps(item, sort_keys=True).encode())
    add(value)
    return result.hexdigest()


def budget():
    if time.monotonic() - STARTED > LIMIT:
        raise TimeoutError("fixed300s startup-through-final-write CPU budget exceeded")


def initialize(module, role):
    streams = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and parameter.numel():
            seed = int.from_bytes(hashlib.sha256(
                f"routed-G-clean-v1:{role}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)
            streams[name] = torch.Generator().manual_seed(seed)
    init.initialize_(module, method="sample_distributions_v1", parameter_generators=streams)


class Projection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.base = nn.Linear(WIDTH, WIDTH, bias=False, dtype=torch.bfloat16).requires_grad_(False)
        with torch.no_grad(): self.base.weight.copy_(weight)
        self.down, self.up = nn.Linear(WIDTH, RANK, bias=False), nn.Linear(RANK, WIDTH, bias=False)
        self.bridge = nn.Linear(RANK + Z_DIM, RANK)

    def forward(self, value, code):
        h = self.down(value.float())
        h = h + F.linear(h, self.bridge.weight[:, :RANK], self.bridge.bias).tanh() \
            + h * F.linear(code, self.bridge.weight[:, RANK:]).tanh()
        return (self.base(value.bfloat16()).float() + self.up(h)).bfloat16().float()


class Host(nn.Module):
    def __init__(self, source, target, weights):
        super().__init__()
        self.register_buffer("sources", source.clone())
        self.register_buffer("target_caption", target.clone())
        self.first, self.second = (Projection(weight) for weight in weights)

    def halves(self, context, *, teacher=False):
        latent = context[..., :WIDTH]
        captions = self.sources[context[:, 0, WIDTH].long()]
        if teacher: captions = self.target_caption.expand_as(captions)
        time = .15 * (2 * context[:, 0, WIDTH + 1] - 1)
        wave = torch.sin(torch.arange(1, WIDTH + 1).float() * .19)
        conditional = latent + captions[:, None] + time[:, None, None] * wave
        unconditional = latent + self.sources.mean(0)[None, None] + time[:, None, None] * wave
        return torch.cat((conditional, unconditional))

    @staticmethod
    def guided(halves):
        conditional, unconditional = halves.chunk(2)
        return unconditional + 3 * (conditional - unconditional)

    @torch.no_grad()
    def teacher(self, context):
        value = self.halves(context, teacher=True)
        for branch in (self.first, self.second):
            value = branch.base(value.bfloat16()).tanh().float()
        return self.guided(value)

    def forward_routed(self, context, router, candidate, routing):
        value = self.halves(context)
        for site, branch in (("first", self.first), ("second", self.second)):
            logits = getattr(router, site)(value) @ candidate.table.T / math.sqrt(Z_DIM)
            logits = torch.stack(logits.chunk(2), dim=1)
            code = routing.mix(site, logits)
            value = branch(value, torch.cat((code[:, 0], code[:, 1]))).bfloat16().tanh().float()
        return self.guided(value) - self.teacher(context)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = nn.Linear(WIDTH, Z_DIM), nn.Linear(WIDTH, Z_DIM)
        self.register_buffer("log_mass", torch.zeros(PARTICLES))


class Encoder(nn.Module):
    def __init__(self, sources):
        super().__init__(); self.register_buffer("sources", sources.clone())
    def condition(self, context):
        return torch.cat((self.sources[context[:, 0, WIDTH].long()], context[:, 0, WIDTH + 1:]), dim=1)


class Critic(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.error_input, self.condition_input = nn.Linear(WIDTH, 48), nn.Linear(WIDTH + 1, 48, bias=False)
        self.feature_output, self.score = nn.Linear(48, 16), nn.Linear(16, 1)
        self.register_buffer("scale", scale.clone())
    def features(self, error, condition):
        return self.feature_output((self.error_input(error) + self.condition_input(condition)[:, None]).tanh()).tanh()
    def forward(self, error, condition):
        return self.score(self.features(error, condition).mean(1))


class PenaltyView(nn.Module):
    def __init__(self, critic):
        super().__init__(); self.critic = critic
    def forward(self, error, condition):
        return self.critic(error.reshape(-1, TOKENS, WIDTH), condition).repeat_interleave(TOKENS, 0)


def model_forward(models, context, candidate, routing):
    return models["generator"].forward_routed(context, models["router"], candidate, routing)


def features(models, context, samples, targets):
    critic = models["critic"]
    return critic.features((samples - targets) / critic.scale, models["encoder"].condition(context)).flatten(1)


def make_data():
    axis = torch.arange(1, WIDTH + 1).float()
    sources = .25 * torch.sin(torch.arange(1, 7).float()[:, None] * axis[None] * .17)
    target = .25 * torch.cos(axis * .31)
    weights = []
    for phase in (.13, .23):
        matrix = torch.cos(axis[:, None] * axis[None] * phase)
        weights.append(.8 * torch.eye(WIDTH) + .2 * matrix / matrix.norm(dim=1, keepdim=True))
    data = {"sources": sources, "target": target, "weights": weights}
    split = {"fit": ((.1, .35, .6, .85), (0, 1)),
             "guard": ((.22, .72), (2,)), "test": ((.18, .43, .68, .93), (3, 4))}
    token = torch.arange(1, TOKENS + 1).float()[:, None]
    for pool, (times, draws) in split.items():
        contexts, ids = [], []
        for source in range(6):
            for time in times:
                for draw in draws:
                    latent = .5 * (torch.sin((draw + 1) * token * axis * .071)
                                     + torch.cos((draw + 2) * token * axis * .039))
                    conditioning = torch.tensor([source, time]).expand(TOKENS, -1)
                    contexts.append(torch.cat((latent, conditioning), dim=1)); ids.append(source)
        data[pool] = {"context": torch.stack(contexts), "targets": torch.zeros(len(contexts), TOKENS, WIDTH), "sources": ids}
    with torch.random.fork_rng(devices=[]): host = Host(sources, target, weights)
    with torch.no_grad():
        for branch in (host.first, host.second): branch.up.weight.zero_()
    # Baseline residuals define fixed critic-coordinate units, never an objective.
    context = data["fit"]["context"]
    value = host.halves(context)
    with torch.no_grad():
        for branch in (host.first, host.second): value = branch.base(value.bfloat16()).tanh().float()
        data["fit_baseline"] = host.guided(value) - host.teacher(context)
        data["scale"] = data["fit_baseline"].flatten(0, 1).std(0, unbiased=False).clamp_min(.04)
        context = data["test"]["context"]; value = host.halves(context)
        for branch in (host.first, host.second): value = branch.base(value.bfloat16()).tanh().float()
        data["test_baseline"] = host.guided(value) - host.teacher(context)
    data["digest"] = digest(data)
    return data


@dataclass
class Loop:
    policy: E22Policy
    arm: str
    data: dict
    data_rng: torch.Generator
    paired_rng: torch.Generator


def make_loop(arm, data):
    if arm not in ARMS: raise ValueError("fixed native/G_clean arms required")
    with torch.random.fork_rng(devices=[]):
        g, d, r = Host(data["sources"], data["target"], data["weights"]), Critic(data["scale"]), Router()
        e = Encoder(data["sources"])
        recipe = get_recipe("e22_routed", num_particles=PARTICLES, z_dim=Z_DIM, batch_size=BATCH,
            output_noise_std=PANEL_SIGMA, birth_death_backend="auto", reopen_guard="settled")
        prior = recipe.make_prior()
    for module, role in ((g, "generator"), (d, "critic"), (r, "router")): initialize(module, role)
    with torch.no_grad():
        for branch in (g.first, g.second):
            branch.up.weight.zero_(); branch.bridge.weight[:, :RANK].zero_(); branch.bridge.bias.zero_()
    init.deterministic_orthogonal_(prior)
    table = prior.z.requires_grad_(True)
    groups = [{"params": [p for p in g.parameters() if p.requires_grad], "lr": 5e-5},
              {"params": list(r.parameters()), "lr": 5e-5},
              {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}]
    opt_g = recipe.make_generator_optimizer(groups, latent_table=table, foreach=False)
    opt_d = recipe.make_critic_optimizer(d, ema_critic=deepcopy(d), foreach=False)
    rows = RoutedRows(model_forward=model_forward, features=features, sites=("first", "second"),
        probe_interval=100, max_context_harm=0., output_error_guard=False)
    p = E22Policy(recipe, g, d, table=table, encoder=e, router=r, generator_optimizer=opt_g,
        critic_optimizer=opt_d, roles=[["generator", "router", "table"], ["critic"]], routed_rows=rows, seed=21)
    p.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
    return Loop(p, arm, data, torch.Generator().manual_seed(7), torch.Generator().manual_seed(43))


def update(loop):
    p, data = loop.policy, loop.data
    ids = torch.randint(len(data["fit"]["context"]), (BATCH,), generator=loop.data_rng)
    context, targets = (data["fit"][key][ids] for key in ("context", "targets"))
    p.G.train()
    noise = p.begin_step(targets, routed=RoutedBatch(context, targets, data["guard"]["context"], data["guard"]["targets"]))
    condition = p.encoder.condition(context)
    d_base = torch.randn(targets.shape, generator=loop.paired_rng)
    g_base = torch.randn(targets.shape, generator=loop.paired_rng)
    loss = p.recipe.make_loss(); p.D.train()
    with torch.no_grad():
        prediction = p.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * d_base
        fake = real + (prediction - targets) / p.D.scale
    p.observe_critic_pair(real, fake)
    penalty = p.penalty(PenaltyView(p.D), real.flatten(0, 1), fake.flatten(0, 1), condition)
    d_game = loss.d_loss(p.D(real, condition), p.D(fake, condition)); d_total = d_game + penalty
    p.opt_d.zero_grad(set_to_none=True); p.before_critic_backward(); d_total.backward(); p.opt_d.step(); p.after_critic_step()
    p.D.eval(); flags = [parameter.requires_grad for parameter in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        if loop.arm == "G_clean":
            # Consume exactly the native G site's draws/diagnostics, then take
            # a clean gradient. Extra complete forward cost is reported.
            with torch.no_grad(): p.routed_generate(context, sigma=0, perturb=True)
        prediction = p.routed_generate(context, sigma=0, perturb=loop.arm == "native")
        real = noise.output_sigma * g_base
        with torch.no_grad(): real_score = p.D(real.detach(), condition)
        g_game = loss.g_loss(p.D(real + (prediction - targets) / p.D.scale, condition), real_score)
        if not bool(torch.isfinite(d_total)) or not bool(torch.isfinite(g_game)):
            raise FloatingPointError("nonfinite native game")
        p.opt_g.zero_grad(set_to_none=True); p.before_generator_backward(); g_game.backward()
        def norm(parameters):
            return math.sqrt(sum(float(v.grad.detach().double().square().sum()) for v in parameters if v.grad is not None))
        bank, query = norm([p.table]), norm(p.router.parameters())
        p.after_generator_backward(loss_gan=g_game.detach(), loss_critic=d_game.detach())
        p.opt_g.step(); p.after_generator_step()
    finally:
        for parameter, flag in zip(p.D.parameters(), flags): parameter.requires_grad_(flag)
    event = p.finish_step()
    return {"step": p.completed_steps, "loss_g": float(g_game.detach()), "loss_d_game": float(d_game.detach()),
        "penalty": float(penalty.detach()), "batch_indices": ids.tolist(), "paired_base_digest": digest((d_base, g_base)),
        "data_rng": digest(loop.data_rng.get_state()), "paired_rng": digest(loop.paired_rng.get_state()),
        "dv12_rng": digest(p.noise_generator.get_state()), "bank_gradient_norm": bank, "query_gradient_norm": query, "move": event}


@torch.no_grad()
def capture(policy, context, *, noisy=False, zero_code=False):
    before = digest(policy.state_dict()); modes = [(m, m.training) for m in policy.G.modules()]
    private = torch.Generator().manual_seed(91)
    outputs = []
    try:
        policy.G.eval()
        for draw in range(NOISY_DRAWS if noisy else 1):
            values = []
            for start in range(0, len(context), BATCH):
                batch = context[start:start+BATCH]
                if zero_code:
                    value = policy.routed_control.generate(batch, candidate=policy.routed_control.candidate(),
                        perturb_fn=lambda codes: torch.zeros_like(codes))
                else: value = policy.routed_generate(batch, sigma=0, perturb=noisy, stream=private)
                values.append(value.detach().clone())
            outputs.append(torch.cat(values))
    finally:
        for module, mode in modes: module.training = mode
    if before != digest(policy.state_dict()): raise AssertionError("private evaluation mutated native state or streams")
    return torch.stack(outputs)


@torch.no_grad()
def score(judge, predictions, panels, condition):
    # Average losses of each actual noisy prediction, never loss of its mean.
    draws, contexts = predictions.shape[:2]
    error = predictions[None].expand(len(panels), -1, -1, -1, -1)
    real = panels[:, None].expand(-1, draws, -1, -1, -1)
    shape = (-1, TOKENS, WIDTH)
    conditions = condition.expand(len(panels), draws, -1, -1).reshape(-1, WIDTH+1)
    fake_score = judge((real + error / judge.scale).reshape(shape), conditions)
    real_score = judge(real.reshape(shape), conditions)
    return float(get_recipe("e22").make_loss().g_loss(fake_score, real_score))


def scientific_gate(clean, native, zero_code, *, calibrated, bank_live, query_live, C_live):
    if len(clean) != 4 or set(clean) != set(native) or set(clean) != set(zero_code):
        raise ValueError("all four common judges are mandatory")
    if any(not math.isfinite(x) for values in (clean, native, zero_code) for x in values.values()):
        raise ValueError("nonfinite terminal game")
    delta = max(clean[key] - native[key] for key in clean)
    code_gain = min(zero_code[key] - clean[key] for key in clean)
    return {"pass": delta < -WIN_MARGIN and code_gain > CODE_MARGIN and calibrated and bank_live and query_live and C_live,
        "metric_max_clean_minus_native_game": delta, "threshold": -WIN_MARGIN,
        "minimum_code_gain": code_gain, "code_threshold": CODE_MARGIN,
        "calibrated": calibrated, "bank_live": bank_live, "query_live": query_live, "C_live": C_live}


def learned_finite(policy):
    """Check learned tensors/moments, leaving monitor validation to the API."""
    def check(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite learned owner, gradient or optimizer moment")
        elif isinstance(value, dict):
            for item in value.values(): check(item)
        elif isinstance(value, (tuple, list)):
            for item in value: check(item)
    state = policy.state_dict()
    for key in ("models", "averages", "table", "averaged_table", "output_noise"):
        check(state[key])
    for optimizer in state["optimizers"]:
        check(optimizer["state"])
        if "regularizer" in optimizer: check(optimizer["regularizer"].get("ema"))
    check([p.grad for owner in (policy.G, policy.D, policy.router) for p in owner.parameters()])
    check((policy.table.grad, policy.log_output_sigma.grad))
    return True


def preflight(data):
    with torch.random.fork_rng(devices=[]): loops = [make_loop(arm, data) for arm in ARMS]
    initial = [digest(loop.policy.state_dict()) for loop in loops]
    if initial[0] != initial[1]: raise AssertionError("arms have different initial native owners")
    context = data["test"]["context"][:BATCH]
    values = [capture(loop.policy, context) for loop in loops]
    if not torch.equal(*values): raise AssertionError("matched initial clean predictions differ")
    if values[0].shape != (1, BATCH, TOKENS, WIDTH) or not bool(torch.isfinite(values[0]).all()) or not bool(values[0].abs().max() > 0):
        raise AssertionError("caption switch has no finite signal or wrong physical shape")
    for loop in loops:
        p = loop.policy
        learned_finite(p)
        assert digest(p.G.state_dict()) == digest(p.ema_G.state_dict())
        assert p.table.requires_grad and all(v.requires_grad for v in p.router.parameters())
        for branch in (p.G.first, p.G.second):
            assert not branch.base.weight.requires_grad and branch.up.weight.count_nonzero() == 0
            assert branch.bridge.weight[:, :RANK].count_nonzero() == 0 and branch.bridge.bias.count_nonzero() == 0
            assert branch.bridge.weight[:, RANK:].count_nonzero() > 0 and branch.bridge.weight.requires_grad
    noisy = [capture(loop.policy, context, noisy=True) for loop in loops]
    if not torch.equal(*noisy): raise AssertionError("matched initial private DV12 predictions differ")
    panels = torch.randn(PANEL_DRAWS, BATCH, TOKENS, WIDTH, generator=torch.Generator().manual_seed(72)) * PANEL_SIGMA
    anchor = score(loops[0].policy.D, torch.zeros_like(values[0]), panels, loops[0].policy.encoder.condition(context))
    if abs(anchor-math.log(2)) > 2e-6: raise AssertionError("native zero residual anchor differs")
    return {"pass": True, "initial_state_digest": initial[0], "initial_owners_exact": True,
        "source_target_shape": list(values[0].shape), "live_caption_signal": True,
        "frozen_BF16_and_trainable_particle_roles": True, "private_evaluation_state_immutable": True,
        "zero_residual_native_game": anchor, "quality_updates": 0, "native_updates": 0}


def run(data, out, card_sha):
    panels = torch.randn(PANEL_DRAWS, len(data["test"]["context"]), TOKENS, WIDTH,
        generator=torch.Generator().manual_seed(72)) * PANEL_SIGMA
    condition = Encoder(data["sources"]).condition(data["test"]["context"])
    raw, judges, traces, coverage, states = {}, {}, {}, {}, {}
    for arm in ARMS:
        with torch.random.fork_rng(devices=[]): loop = make_loop(arm, data)
        p = loop.policy; trace = []; updates = {"bank": 0, "query": 0}; frozen = digest({"first":p.G.first.base.state_dict(),"second":p.G.second.base.state_dict(),"sources":p.G.sources,"target":p.G.target_caption})
        raw[arm] = {}
        for step in range(STEPS+1):
            if step:
                row = update(loop); trace.append(row)
                updates["bank"] += int(row["bank_gradient_norm"] > 0); updates["query"] += int(row["query_gradient_norm"] > 0)
                with (out/f"{arm}.jsonl").open("a") as stream: stream.write(json.dumps(row, allow_nan=False)+"\n")
            if step in JUDGE_STEPS: judges[f"{arm}@{step}"] = deepcopy(p.D).eval().requires_grad_(False)
            if step in CURVES:
                raw[arm][str(step)] = {"clean":capture(p,data["test"]["context"]),"DV12":capture(p,data["test"]["context"],noisy=True)}
                print(json.dumps({"arm":arm,"step":step,"loss_g":None if not step else row["loss_g"],"seconds":time.monotonic()-STARTED}),flush=True)
            budget()
        learned_finite(p)
        raw[arm][str(STEPS)]["zero_code"] = capture(p,data["test"]["context"],zero_code=True)
        states[arm] = {"native":p.state_dict(),"data_rng":loop.data_rng.get_state(),"paired_rng":loop.paired_rng.get_state()}
        coverage[arm] = {**updates,"C_norms":[float(branch.bridge.weight[:,RANK:].norm()) for branch in (p.G.first,p.G.second)]}
        traces[arm] = [{k:row[k] for k in ("step","batch_indices","paired_base_digest","data_rng","paired_rng","dv12_rng")} for row in trace]
        if frozen != digest({"first":p.G.first.base.state_dict(),"second":p.G.second.base.state_dict(),"sources":p.G.sources,"target":p.G.target_caption}):
            raise AssertionError("training changed frozen teacher/backbone")
    if traces["native"] != traces["G_clean"]: raise AssertionError("matched data/paired/DV12 draw schedules differ")
    results = {arm:{step:{cohort:{name:score(judge,pred,panels,condition) for name,judge in judges.items()}
                         for cohort,pred in cohorts.items()} for step,cohorts in snapshots.items()} for arm,snapshots in raw.items()}
    references = {name:[score(judge,(alpha*data["test_baseline"])[None],panels,condition) for alpha in (0.,.25,.5,1.)] for name,judge in judges.items()}
    calibrated = all(all(b>=a-1e-5 for a,b in zip(path,path[1:])) and path[-1]>path[0]+1e-4 for path in references.values())
    flags = coverage["G_clean"]
    gate = scientific_gate(results["G_clean"][str(STEPS)]["clean"],results["native"][str(STEPS)]["clean"],
        results["G_clean"][str(STEPS)]["zero_code"],calibrated=calibrated,bank_live=flags["bank"]>0,
        query_live=flags["query"]>0,C_live=all(v>0 for v in flags["C_norms"]))
    torch.save({"predictions":raw,"panels":panels,"states":states,"judges":{name:judge.state_dict() for name,judge in judges.items()}},out/"retained.pt")
    return {"complete":True,"scientific_status":"PASS" if gate["pass"] else "FAIL","task":TASK,
        "gate":gate,"curves":results,"references":references,"coverage":coverage,"data_digest":data["digest"],
        "matched_data_paired_and_DV12_schedules":True,"fixed_updates_each":STEPS,"quality_updates_total":STEPS*2,
        "learned_owners_gradients_and_optimizer_moments_finite":True,
        "G_clean_extra_no_grad_forwards":STEPS,"retained_sha256":sha(out/"retained.pt"),"protocol_sha256":card_sha,
        "source_sha256":sha(__file__),"native_python_sha256":native_sha(),"seconds":time.monotonic()-STARTED,
        "initialization":"Both arms: public sampled Down/Up/H/b/C, fresh Up=0,H=0,b=0,C remains sampled; prior public R2",
        "ordinary_comparison":"UNAVAILABLE: only two particle arms are trained",
        "scope":"Synthetic conditional two-site acquisition, not full-Supra convergence or a general DV12 rejection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,default=CARD); parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--run",action="store_true",help="Execute the one fixed512-update-per-arm scientific PASS/FAIL test")
    parser.add_argument("--out",type=Path); args=parser.parse_args(); torch.set_num_threads(1)
    card=json.loads(args.protocol.read_text()); card_sha=sha(args.protocol)
    if (card["task"]!=TASK or card["steps"]!=STEPS or card["seconds"]!=LIMIT or tuple(card["arms"])!=ARMS
        or card["win_margin"]!=WIN_MARGIN or card["code_margin"]!=CODE_MARGIN
        or card["source_sha256"]!=sha(__file__) or card["native_python_sha256"]!=native_sha()):
        raise ValueError("frozen protocol/source/metric differs")
    data=make_data()
    if args.preflight:
        result=preflight(data); budget()
        if torch.cuda.is_initialized(): raise AssertionError("CPU-only prerequisite initialized CUDA")
        print(json.dumps({**result,"protocol_sha256":card_sha,"data_digest":data["digest"],"seconds":time.monotonic()-STARTED}),flush=True)
        return 0
    if not args.run or args.out is None or os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("explicit --run, fresh --out and CUDA_VISIBLE_DEVICES='' required")
    out=args.out.resolve()
    if out.exists(): raise ValueError("existing outputs are immutable; choose declared fresh directory")
    out.mkdir(parents=True)
    try:
        report=run(data,out,card_sha)
        if sha(args.protocol)!=card_sha or report["source_sha256"]!=card["source_sha256"] or native_sha()!=card["native_python_sha256"]:
            raise ValueError("held execution sources changed")
        budget(); (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); budget()
        receipt={"complete":True,"scientific_status":report["scientific_status"],"report_sha256":sha(out/"report.json"),"seconds":time.monotonic()-STARTED}
        (out/"completion.json").write_text(json.dumps(receipt,indent=2)+"\n")
        if time.monotonic()-STARTED>LIMIT:
            receipt["complete"]=False; (out/"completion.json").write_text(json.dumps(receipt,indent=2)+"\n"); budget()
        print(json.dumps({"scientific_status":report["scientific_status"],"gate":report["gate"]}),flush=True)
        return 0 if report["gate"]["pass"] else 1
    except Exception as exc:
        (out/"failure.json").write_text(json.dumps({"complete":False,"error":type(exc).__name__+": "+str(exc),"protocol_sha256":card_sha,"seconds":time.monotonic()-STARTED},indent=2)+"\n")
        raise


if __name__ == "__main__": raise SystemExit(main())
