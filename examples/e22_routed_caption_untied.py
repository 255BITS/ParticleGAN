"""Public-API three-arm caption regression with untied output heads.

python -m examples.e22_routed_caption_untied --run --out runs/caption-untied-up-v1
Exit0=scientific PASS,1=completed FAIL,2=incomplete. No pretrained assets.
The bundled PR239 fixture supplies unchanged public-API game/data helpers.
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
from examples import e22_routed_caption_accuracy as base

TASK = "routed_caption_untied_up_v1"
UNTIED = "particle_untied_BF16"
ARMS = (*base.ARMS, UNTIED)
STEPS, SECONDS = 512, 300
CARD = Path(__file__).resolve().parents[1] / "docs/e22_routed_caption_untied_v1.json"


def budget():
    if time.monotonic() - STARTED > SECONDS: raise TimeoutError("startup-through-final-write300s budget exceeded")


class UntiedAdapter(base.Adapter):
    """Two registered heads, one Down; no captured FAST/EMA module aliases."""
    def __init__(self, frozen, g):
        super().__init__(frozen, g, base.ARMS[1])
        self.particle_up = nn.Linear(g.rank, frozen.out_features, bias=False)

    def forward(self, x):
        frozen = self.base(x)
        router, candidate, routing = self.route
        with torch.autocast(x.device.type, enabled=False):
            query = router.queries[self.site.replace(".", "__")](x.float())
            logits = query @ candidate.table.float().T / math.sqrt(base.Z)
            b, t = logits.shape[0] // 2, logits.shape[1]
            grouped = logits.reshape(2, b, t, logits.shape[-1]).permute(1, 0, 2, 3)
            mixed = routing.mix(self.site, grouped)
            code = mixed.permute(1, 0, 2, 3).reshape(2 * b, t, base.Z)
            with torch.autocast(x.device.type, dtype=torch.bfloat16): h = self.down(x.float())
            h = h.float()
            main_features = h + F.linear(h, self.bridge.weight[:, :self.g.rank], self.bridge.bias).tanh()
            code_features = h * F.linear(code.float(), self.bridge.weight[:, self.g.rank:]).tanh()
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                main = self.up(main_features)
                particle = self.particle_up(code_features)
            return (frozen.float() + main.float() + particle.float()).to(frozen.dtype)


class UntiedHost(base.Host):
    def __init__(self, data):
        super().__init__(data, base.ARMS[1]); self.arm = UNTIED
        for site in base.SITES:
            branch = UntiedAdapter(self.backbone.get_submodule(site).base, self.g)
            branch.site = site
            base.replace(self.backbone, site, branch)


def make_loop(arm, data, *, software_C_zero=False):
    if arm not in ARMS: raise ValueError("one declared arm is required")
    if arm != UNTIED: return base.make_loop(arm, data)
    if software_C_zero and data["geometry"] == base.FULL:
        raise ValueError("C-zero is a destructive software control, not a science arm")
    device = data["fit"]["context"].device
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(900)
        if device.type == "cuda": torch.cuda.manual_seed(900)
        g = UntiedHost(data)
        d, e, r = base.Critic(data["scale"].cpu(), data["geometry"]), base.Encoder(data["captions"], data["masks"], data["geometry"]), base.Router(data["geometry"])
        # Common named streams retain Down/up/H/b/C/query/D values. The extra
        # head is registered and publicly initialized BEFORE optimizers/EMA.
        base.initialize(g, "generator"); base.initialize(d, "critic"); base.initialize(r, "router")
        with torch.no_grad():
            for branch in g.branches():
                branch.up.weight.zero_(); branch.particle_up.weight.zero_()
                branch.bridge.weight[:, :data["geometry"].rank].zero_(); branch.bridge.bias.zero_()
                if software_C_zero: branch.bridge.weight[:, data["geometry"].rank:].zero_()
        g.to(device); d.to(device); e.to(device); r.to(device)
        recipe = get_recipe("e22_routed", num_particles=base.N, z_dim=base.Z, batch_size=base.B,
                            output_noise_std=base.SIGMA, birth_death_backend="auto", reopen_guard="settled")
        prior = recipe.make_prior(); init.deterministic_orthogonal_(prior); table = prior.to(device).z
        groups = [{"params": [v for v in g.parameters() if v.requires_grad], "lr": 5e-5},
                  {"params": list(r.parameters()), "lr": 5e-5}, {"params": [table], "lr": recipe.lr * recipe.prior_lr_mult}]
        opt_g = recipe.make_generator_optimizer(groups, latent_table=table, foreach=False)
        opt_d = recipe.make_critic_optimizer(d, ema_critic=deepcopy(d), foreach=False)
        p = E22Policy(recipe, g, d, table=table, encoder=e, router=r, generator_optimizer=opt_g, critic_optimizer=opt_d,
            roles=[["generator", "router", "table"], ["critic"]], seed=21,
            routed_rows=RoutedRows(model_forward=base.model_forward, features=base.features, sites=base.SITES,
                                  probe_interval=100, max_context_harm=0., output_error_guard=False))
        p.attach_penalty(recipe.make_critic_penalty(opt_d, collect_stats=True))
        torch.manual_seed(900)
        if device.type == "cuda": torch.cuda.manual_seed(900)
        owned = base.global_rng(device)
    return base.Loop(p, arm, data, torch.Generator().manual_seed(7), torch.Generator().manual_seed(43), owned)


def scientific_gate(ordinary, shared, untied, zero, *, bank_updates, query_updates, C_norms, particle_Up_norms):
    vs_ordinary = base.scientific_gate(ordinary, untied, zero, bank_updates=bank_updates, query_updates=query_updates, C_norms=C_norms)
    vs_shared = base.scientific_gate(shared, untied, zero, bank_updates=bank_updates, query_updates=query_updates, C_norms=C_norms)
    head_live = set(particle_Up_norms) == set(base.SITES) and all(math.isfinite(v) and v > 0 for v in particle_Up_norms.values())
    return {"pass": vs_ordinary["pass"] and vs_shared["pass"] and head_live,
            "versus_ordinary": vs_ordinary, "versus_shared": vs_shared, "particle_Up_live_all_six_sites": head_live}


def controls():
    def metric(v): return {"rmse": v, "by_source": {str(i): v for i in range(6)}}
    kwargs = {"bank_updates": 511, "query_updates": 511, "C_norms": {s: .1 for s in base.SITES}, "particle_Up_norms": {s: .1 for s in base.SITES}}
    if not scientific_gate(metric(1), metric(1), metric(.99), metric(1), **kwargs)["pass"]: raise AssertionError("positive oracle rejected")
    if scientific_gate(metric(1), metric(.98), metric(.99), metric(1), **kwargs)["pass"]: raise AssertionError("losing one control passed")
    harmed = metric(.99); harmed["by_source"]["5"] = 1 + 2e-6
    if scientific_gate(metric(1), metric(1), harmed, metric(1.01), **kwargs)["pass"]: raise AssertionError("source harm passed")
    dead = dict(kwargs, particle_Up_norms={s: 0. for s in base.SITES})
    if scientific_gate(metric(1), metric(1), metric(.99), metric(1), **dead)["pass"]: raise AssertionError("dead head passed")
    return {**base.scorer_controls(), "both_control_source_and_live_head_destructive_controls": True}


def counts(g):
    old = base.trainable_counts(g); extra = 9 * g.width * g.rank
    return {**old, "untied_extra_Up": extra, "untied_particle_total": old["particle_total"] + extra,
            "untied_vs_ordinary_parameter_ratio": (old["particle_total"] + extra) / old["ordinary_DownUp"]}


def initial_tangent(ordinary, untied):
    """Zero-update public loss proof, with every owned state restored finally."""
    loops=(ordinary,untied);saved=[base.checkpoint(loop) for loop in loops]
    g=ordinary.data["geometry"];indices=[0,8,16,24]
    x=ordinary.data["fit"]["context"][indices]
    panel=base.SIGMA*torch.randn(len(x),g.tokens,g.output,generator=torch.Generator().manual_seed(72)).to(x.device)
    gradients=[]
    try:
        for loop in loops:
            p=loop.policy;flags=[v.requires_grad for v in p.D.parameters()]
            try:
                p.D.eval();p.D.requires_grad_(False);condition=p.encoder.condition(x)
                residual=p.G(x) if loop.arm==ARMS[0] else p.routed_generate(x,sigma=0,perturb=True)
                with torch.no_grad():reference=p.D(panel,condition)
                loss=p.recipe.make_loss().g_loss(p.D(panel+residual/p.D.scale,condition),reference)
                heads=[b.up.weight for b in p.G.branches()]
                if loop.arm==UNTIED:heads += [b.particle_up.weight for b in p.G.branches()]
                gradients.append(torch.autograd.grad(loss,heads))
            finally:
                for value,flag in zip(p.D.parameters(),flags):value.requires_grad_(flag)
        main_exact=all(torch.equal(a,b) for a,b in zip(gradients[0],gradients[1][:6]))
        norms={s:float(v.detach().double().norm()) for s,v in zip(base.SITES,gradients[1][6:])}
        if not main_exact or not all(math.isfinite(v) and v>0 for v in norms.values()):
            raise AssertionError("initial ordinary-head tangent or live perturbed particle-head gradient failed")
        return {"main_Up_gradient_matches_ordinary_exact":True,"particle_Up_gradients_finite_positive_all_six_sites":True,
                "particle_Up_gradient_norms":norms,"fit_indices":indices,"private_Gaussian_seed":72,
                "native_DV12_perturbation":True,"native_updates":0,"scope":"Initial point only; no total-variance or adaptive-update theorem."}
    finally:
        for loop,state in zip(loops,saved):
            base.restore(loop,state)
            if base.digest(base.checkpoint(loop))!=base.digest(state):raise AssertionError("tangent proof changed native/caller/mode/gradient state")


def preflight(data):
    """Fresh three-arm zero-update proof; never a convergence PASS."""
    device = data["fit"]["context"].device; entry = base.global_rng(device)
    loops = [make_loop(arm, data) for arm in ARMS]
    reference, shared, candidate = loops
    for loop in loops:
        p = loop.policy
        for a, b in zip(reference.policy.G.branches(), p.G.branches()):
            if not torch.equal(a.down.weight, b.down.weight) or not torch.equal(a.up.weight, b.up.weight): raise AssertionError("common named Down/main-Up differ")
        if base.digest(p.D.state_dict()) != base.digest(reference.policy.D.state_dict()): raise AssertionError("common critic differs")
        if base.digest(loop.globals) != base.digest(reference.globals): raise AssertionError("native penalty initial streams differ")
        base.learned_finite(p)
    for a, b in zip(shared.policy.G.branches(), candidate.policy.G.branches()):
        if not torch.equal(a.bridge.weight, b.bridge.weight) or not torch.equal(a.bridge.bias, b.bridge.bias): raise AssertionError("H/b/C named initialization differs")
        if b.particle_up.weight.count_nonzero() or b.up.weight.count_nonzero(): raise AssertionError("both fresh heads must be zero")
        if not b.bridge.weight[:, data["geometry"].rank:].count_nonzero(): raise AssertionError("C-zero dead conditional branch refused")
    if base.digest(shared.policy.router.state_dict()) != base.digest(candidate.policy.router.state_dict()) or not torch.equal(shared.policy.table, candidate.policy.table): raise AssertionError("native route/bank start differs")
    if shared.policy.recipe.to_dict() != candidate.policy.recipe.to_dict(): raise AssertionError("particle recipes differ")
    x = data["test"]["context"][:base.B]
    predictions = [base.observe(loop, x) for loop in loops]
    if not all(torch.equal(predictions[0], y) for y in predictions[1:]): raise AssertionError("initial frozen-output equality differs")
    state = base.checkpoint(candidate); recovered = make_loop(UNTIED, data); base.restore(recovered, state)
    if base.digest(base.checkpoint(recovered)) != base.digest(state) or not torch.equal(base.observe(recovered, x), predictions[-1]): raise AssertionError("fresh-owner public restore differs")
    tangent=initial_tangent(reference,candidate)
    if base.digest(entry) != base.digest(base.global_rng(device)): raise AssertionError("initial prerequisite changed caller RNG")
    return {"pass": True, "native_updates": 0, "common_initial_owners_predictions_exact": True, "fresh_public_restore_exact": True,
            "scope": "software only" if data["geometry"] != base.FULL else "zero-update full-geometry prerequisite", "counts": counts(data["geometry"]),"initial_tangent":tangent}


def run(data, out):
    quality, physical, witnesses, media, traces = {}, {}, {}, {}, {}
    common_stream = None; live = {"bank": 0, "query": 0}; cnorm, unorm = {}, {}
    oracles = controls(); camera = data["test"]["context"][list(base.MEDIA_INDICES)]
    for arm in ARMS:
        loop = make_loop(arm, data); p = loop.policy; stream = hashlib.sha256()
        media[arm] = {"0": base.observe(loop, camera).cpu()}
        learned_names = {name for name, value in p.G.named_parameters() if value.requires_grad}
        def frozen():
            return base.digest({n: v for n,v in p.G.state_dict().items() if n not in learned_names})
        frozen_sha = frozen(); events = accepted_rows = accepted_proposals = 0; loss_ema = None
        for step in range(1, STEPS + 1):
            row = base.update(loop)
            stream.update(json.dumps({k:row[k] for k in ("step","batch_indices","paired_bases","data_rng","paired_rng","penalty_globals")}, sort_keys=True).encode())
            if row["move"] is not None:
                events += 1; accepted_rows += int(row["move"].get("moves", 0)); accepted_proposals += int(row["move"].get("accepted",False))
            if arm == UNTIED and step > 1:
                live["bank"] += int(row["bank_live"]); live["query"] += int(row["query_live"])
            with (out/f"{arm}.jsonl").open("a") as handle: handle.write(json.dumps(row, allow_nan=False)+"\n")
            loss_ema = row["loss_g"] if loss_ema is None else .98*loss_ema+.02*row["loss_g"]
            if step % 64 == 0: print(json.dumps({"arm":arm,"step":step,"steps":STEPS,"native_G_loss":row["loss_g"],"native_G_loss_EMA":loss_ema,"native_D_loss":row["loss_d_game"],"seconds":time.monotonic()-STARTED}),flush=True)
            if step in base.MEDIA_STEPS: media[arm][str(step)] = base.observe(loop,camera).cpu()
            budget()
        base.learned_finite(p)
        if frozen_sha != frozen(): raise AssertionError("frozen host/teacher/caption owners changed")
        if common_stream is None: common_stream = stream.hexdigest()
        elif stream.hexdigest() != common_stream: raise AssertionError("data/Gaussian/native-penalty streams differ")
        state = base.checkpoint(loop); physical[arm] = base.observe(loop,data["test"]["context"])
        base.restore(loop,state)
        if base.digest(base.checkpoint(loop)) != base.digest(state) or not torch.equal(physical[arm],base.observe(loop,data["test"]["context"])): raise AssertionError("terminal public restore/output replay differs")
        quality[arm] = base.accuracy(physical[arm],data["test"]["source_ids"])
        witnesses[arm] = {"steps":p.completed_steps,"recipe":p.recipe.to_dict(),"native_state_digest":base.digest(state["native"]),"public_terminal_restore_output_exact":True,"learned_weights_gradients_moments_finite":True}
        traces[arm] = {"proposal_events_including_skips":events,"accepted_row_moves":accepted_rows,"accepted_proposals":accepted_proposals}
        if arm == UNTIED:
            physical["zero_code"] = base.observe(loop,data["test"]["context"],zero_code=True)
            quality["zero_code"] = base.accuracy(physical["zero_code"],data["test"]["source_ids"])
            cnorm = {b.site:float(b.bridge.weight[:,data["geometry"].rank:].detach().norm()) for b in p.G.branches()}
            unorm = {b.site:float(b.particle_up.weight.detach().norm()) for b in p.G.branches()}
        del loop,p,state
        budget()
    gate = scientific_gate(*(quality[arm] for arm in ARMS),quality["zero_code"],bank_updates=live["bank"],query_updates=live["query"],C_norms=cnorm,particle_Up_norms=unorm)
    raw = out/"endpoint-residuals.pt"
    torch.save({"physical_residuals":physical,"source_ids":data["test"]["source_ids"],"test_context_digest":base.digest(data["test"]["context"]),"target_digest":base.digest(data["test"]["targets"])},raw)
    frames = out/"observed-media.pt"
    torch.save({"actual_residuals":media,"target_residual":torch.zeros_like(media[ARMS[0]]["0"]),"steps":base.MEDIA_STEPS,"indices":base.MEDIA_INDICES,"source_ids":[data["test"]["source_ids"][i] for i in base.MEDIA_INDICES],"context_digest":base.digest(camera),"capture_native_state_rng_diagnostics_unchanged":True},frames)
    budget()
    return {"task":TASK,"complete":True,"scientific_status":"PASS" if gate["pass"] else "FAIL","gate":gate,"accuracy":quality,"arms":witnesses,
        "quality_updates":3*STEPS,"replay_updates":0,"live":{**live,"denominator":511},"C_norms":cnorm,"particle_Up_norms":unorm,"population_events":traces,
        "matched_external_data_Gaussian_native_penalty_streams":True,"data_digest":data["digest"],"scorer_oracles_and_destructive_controls":oracles,
        "media_steps":list(base.MEDIA_STEPS),"observed_media_sha256":base.sha(frames),"endpoint_residual_sha256":base.sha(raw),
        "counts":counts(data["geometry"]),"geometry":asdict(data["geometry"]),"scope":"Additional capacity and separate BF16 head rounding; neither a native bug fix nor a variance theorem. No actual-Supra win established."}


def source_identity(): return {"variant":base.sha(__file__),"bundled_fixture":base.sha(base.__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true",required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--protocol",type=Path,default=CARD);args=parser.parse_args()
    threads=torch.get_num_threads();torch.set_num_threads(1)
    source=source_identity();native=base.package_identity();card_sha=None;out=None;entry=None;report=None;error=None;code=2;device=None
    try:
        card=json.loads(args.protocol.read_text());card_sha=base.sha(args.protocol)
        expected={"task":TASK,"arms":list(ARMS),"geometry":asdict(base.FULL),"steps":STEPS,"seconds":SECONDS,"media_steps":list(base.MEDIA_STEPS),"media_indices":list(base.MEDIA_INDICES)}
        if any(card[k]!=v for k,v in expected.items()):raise ValueError("fixed task/horizon/geometry/media law differs")
        if card["source_provenance"]["sha256"]!=source["bundled_fixture"]:raise ValueError("bundled fixed fixture source differs")
        if card["accuracy_thresholds"]!={"relative_aggregate_improvement_gte":.001,"source_harm_lte":1e-6,"relative_code_benefit_gte":.001,"strict_code_benefit_each_source":True,"bank_router_live_fraction_gte":.9,"live_denominator":511,"all_six_particle_Up_and_C_nonzero":True}:raise ValueError("fixed numerical gate differs")
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available():raise ValueError("physicalGPU0 requires CUDA_VISIBLE_DEVICES=0")
        if base.backend_flags()!=card["precision_backend"]:raise ValueError("declared precision backend differs")
        if args.out.exists():raise ValueError("fresh output directory required")
        args.out.mkdir(parents=True);out=args.out;device=torch.device("cuda:0");entry=base.global_rng(device)
        controls();data=base.make_data(base.FULL,device);initial=preflight(data);budget();report=run(data,out)
        if source_identity()!=source or base.package_identity()!=native or base.sha(args.protocol)!=card_sha or base.backend_flags()!=card["precision_backend"]:raise ValueError("source/card/API/backend changed within run")
        report.update(imported_package=native,source_identity=source,protocol_sha256=card_sha,precision_backend=base.backend_flags(),initial_prerequisite=initial,imported_package_unchanged=True)
        code=0 if report["gate"]["pass"] else 1
    except BaseException as caught:
        error={"type":type(caught).__name__,"message":str(caught)}
        import traceback;traceback.print_exc()
    finally:
        if entry is not None:
            base.set_global_rng(entry,device)
            if base.digest(base.global_rng(device))!=base.digest(entry):error={"type":"AssertionError","message":"caller RNG restore differs"}
        torch.set_num_threads(threads);elapsed=time.monotonic()-STARTED
        if elapsed>SECONDS:error={"type":"TimeoutError","message":"startup/cleanup300s overrun"}
        if error is not None:code=2
        completion={"complete":error is None and report is not None,"scientific_status":None if error or report is None else report["scientific_status"],"error":error,"seconds":elapsed,"limit_seconds":SECONDS,
                    "source_identity":source,"protocol_sha256":card_sha,"imported_package":native,"caller_CPU_CUDA_RNG_restored":entry is not None and base.digest(base.global_rng(device))==base.digest(entry)}
        if out is not None:
            if report is not None:
                report["seconds"]=elapsed;(out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");completion["report_sha256"]=base.sha(out/"report.json")
            path=out/"completion.json";path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n")
            if time.monotonic()-STARTED>SECONDS:
                completion.update(complete=False,error={"type":"TimeoutError","message":"final serialization300s overrun"},seconds=time.monotonic()-STARTED)
                path.write_text(json.dumps(completion,indent=2,allow_nan=False)+"\n");code=2
        print(json.dumps({"completion":completion,"scientific_gate":None if report is None else report.get("gate")},allow_nan=False),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
