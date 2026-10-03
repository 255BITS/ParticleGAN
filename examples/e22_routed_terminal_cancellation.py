"""Constructed terminal cancellation stress case; public APIs, no external assets.

python -m examples.e22_routed_terminal_cancellation --run --out runs/terminal-cancellation-v1
Exit0=terminal convergence PASS,1=complete FAIL,2=incomplete. Precision witness
is separate and never requires a future API to reproduce historical harm.
"""
import time
STARTED = time.monotonic()
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from particlegan import E22Policy, RoutedRows, get_recipe, init
from examples import e22_routed_caption_accuracy as base
from examples import e22_routed_caption_untied as untied
from examples import e22_routed_caption_flow as flow

TASK = "routed_terminal_cancellation_v1"
ARMS = ("ordinary_BF16", "ordinary_FP32", "particle_BF16", "particle_FP32")
SITES = ("input", "edit")
N, Z, B, STEPS, LIMIT, EDIT_SCALE = 32, 4, 4, 512, 300, 1 / 64
GEOMETRY = base.Geometry(width=32,text=16,rank=4,tokens=16,length=1,heads=4,output=16,frequency=8)
MEDIA_STEPS = (0,64,128,256,384,512)
MEDIA_INDICES = (0,8,16,24,32,40,1,9)
CARD = Path(__file__).resolve().parents[1]/"docs/e22_routed_terminal_cancellation_v1.json"
HELPERS = (base,untied,flow)
DEPENDENCY_PATHS = ("examples/e22_routed_caption_normalization.py",
                    "examples/render_e22_routed_caption_untied.py")


def budget():
    if time.monotonic()-STARTED > LIMIT: raise TimeoutError("300s startup-through-final-writes exceeded")


def sources():
    return {str(Path(m.__file__).resolve().relative_to(CARD.parents[1])):base.sha(m.__file__)
            for m in HELPERS} | {path:base.sha(CARD.parents[1]/path) for path in DEPENDENCY_PATHS} | {
                "examples/e22_routed_terminal_cancellation.py":base.sha(__file__)}


class PairedTerminal(nn.Module):
    """Correlated FROZEN projection is a declared parameterization, not a rewrite.

    W receives public initialization. Effective [W,-W] is computed from that one
    owner; neither an extra learned projection nor a post-init weight copy exists.
    FP32 changes input/weight/matmul/output arithmetic, not just output casting.
    """
    def __init__(self, width, output, arithmetic="BF16"):
        super().__init__(); self.matrix=nn.Linear(width,output,bias=False); self.arithmetic=arithmetic
    def forward(self,a,u):
        x=torch.cat((a.float()+u.float(),a.float()-u.float()),-1)
        weight=torch.cat((self.matrix.weight,-self.matrix.weight),-1)
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=self.arithmetic=="BF16"):
            return F.linear(x.float(),weight.float()).float()


class Core(nn.Module):
    def __init__(self,g,arithmetic="BF16"):
        super().__init__(); features=g.output+g.text+2
        self.carrier=nn.Linear(features,g.output)
        self.input=nn.Linear(features,g.width); self.edit=nn.Linear(g.width,g.output)
        self.final=PairedTerminal(g.output,g.output,arithmetic)
    def forward(self,x):
        with torch.autocast(x.device.type,enabled=False): a=self.carrier(x.float())
        with torch.autocast(x.device.type,dtype=torch.bfloat16):
            u=self.edit(self.input(x).tanh()).float()*EDIT_SCALE
        return self.final(a,u)


class Host(nn.Module):
    def __init__(self,data,arm):
        super().__init__(); self.arm=arm; self.g=data["geometry"]; self.particle=arm.startswith("particle")
        self.backbone=Core(self.g,arm.rsplit("_",1)[1]);self.teacher_backbone=Core(self.g)
        self.backbone.load_state_dict(data["frozen"]);self.teacher_backbone.load_state_dict(data["frozen"])
        self.backbone.requires_grad_(False);self.teacher_backbone.requires_grad_(False)
        self.register_buffer("captions",data["captions"].clone())
        for site in SITES:
            frozen=self.backbone.get_submodule(site)
            branch=untied.UntiedAdapter(frozen,self.g) if self.particle else base.Adapter(frozen,self.g,base.ARMS[0])
            branch.site=site;setattr(self.backbone,site,branch)
    def branches(self):return [self.backbone.get_submodule(s) for s in SITES]
    def inputs(self,x,teacher=False):
        ids=x[:,0,self.g.output].long()+(7 if teacher else 1)
        ids=torch.cat((ids,torch.zeros_like(ids)))
        latent=torch.cat((x[...,:self.g.output],)*2)
        t=torch.cat((x[:,0,self.g.output+1],)*2)
        return torch.cat((latent,self.captions[ids].expand(-1,self.g.tokens,-1),
                          t[:,None,None].expand(-1,self.g.tokens,1),
                          torch.sin(math.pi*t)[:,None,None].expand(-1,self.g.tokens,1)),-1)
    @staticmethod
    def guided(y):
        cond,uncond=y.float().chunk(2);return uncond+3*(cond-uncond)
    @torch.no_grad()
    def teacher(self,x):return self.guided(self.teacher_backbone(self.inputs(x,True)))
    def forward(self,x):
        if self.particle:raise ValueError("use public routed execution")
        return self.guided(self.backbone(self.inputs(x)))-self.teacher(x)
    def forward_routed(self,x,router,candidate,routing):
        try:
            if any(b.route is not None for b in self.branches()):raise RuntimeError("nested route")
            for b in self.branches():b.route=(router,candidate,routing)
            return self.guided(self.backbone(self.inputs(x)))-self.teacher(x)
        finally:
            for b in self.branches():b.route=None


class Router(nn.Module):
    def __init__(self,g):
        super().__init__();self.queries=nn.ModuleDict({"input":nn.Linear(g.output+g.text+2,Z),"edit":nn.Linear(g.width,Z)})
        self.register_buffer("log_mass",torch.zeros(N))


def make_data(g=GEOMETRY,device=torch.device("cpu")):
    with torch.random.fork_rng(devices=[]):frozen=Core(g)
    base.initialize(frozen,"terminal_cancellation_frozen");frozen.requires_grad_(False)
    captions=torch.zeros(13,1,g.text);captions[1:]=torch.eye(g.text)[:12,None]
    masks=torch.ones(13,1,dtype=torch.bool)
    pools,anchors=flow.context_data(g,device)
    data={"geometry":g,"frozen":deepcopy(frozen.state_dict()),"captions":captions,"masks":masks,**pools}
    with torch.random.fork_rng(devices=[]):h=Host(data,ARMS[0])
    base.initialize(h,"generator")
    with torch.no_grad():
        for b in h.branches():b.up.weight.zero_()
        h.to(device);baseline=torch.cat([h(data["fit"]["context"][i:i+B]) for i in range(0,48,B)])
        raw_std=baseline.flatten(0,1).std(0)
        if not bool(torch.isfinite(raw_std).all()) or bool((raw_std<=1e-8).any()):raise ValueError("nondegenerate untrained FITstd >1e-8 required")
    # One common scale from the declared BF16 initial reference, reused by all D/EMA arms.
    data.update(scale=raw_std,fit_baseline=baseline,flow=anchors)
    data["digest"]=base.digest({k:asdict(v) if isinstance(v,base.Geometry) else v for k,v in data.items()})
    return data


def make_loop(arm,data):
    if arm not in ARMS:raise ValueError("fixed four arms")
    device=data["fit"]["context"].device;particle=arm.startswith("particle")
    devices=[device.index or 0] if device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(900)
        if devices:torch.cuda.manual_seed(900)
        g=Host(data,arm);d=base.Critic(data["scale"].cpu(),data["geometry"])
        e=base.Encoder(data["captions"],data["masks"],data["geometry"]);r=Router(data["geometry"]) if particle else None
        base.initialize(g,"generator");base.initialize(d,"critic")
        if r is not None:base.initialize(r,"router")
        with torch.no_grad():
            for b in g.branches():
                b.up.weight.zero_()
                if particle:b.particle_up.weight.zero_();b.bridge.weight[:,:data["geometry"].rank].zero_();b.bridge.bias.zero_()
        g.to(device);d.to(device);e.to(device)
        if r is not None:r.to(device)
        options=dict(num_particles=N,z_dim=Z,batch_size=B,output_noise_std=.125,
                     birth_death_backend="auto" if particle else "knn",reopen_guard="settled")
        if not particle:options.update(particle_birth_death=False,row_evidence_gate=False,birth_death_feature_scale="none",birth_death_isolation=False)
        recipe=get_recipe("e22_routed" if particle else "e22",**options)
        prior=recipe.make_prior();init.deterministic_orthogonal_(prior);table=prior.to(device).z.requires_grad_(particle)
        groups=[{"params":[p for p in g.parameters() if p.requires_grad],"lr":5e-5}]
        if particle:groups.extend(({"params":list(r.parameters()),"lr":5e-5},{"params":[table],"lr":recipe.lr*recipe.prior_lr_mult}))
        og=recipe.make_generator_optimizer(groups,latent_table=table if particle else None,foreach=False)
        od=recipe.make_critic_optimizer(d,ema_critic=deepcopy(d),foreach=False)
        kwargs={"routed_rows":RoutedRows(model_forward=base.model_forward,features=base.features,sites=SITES,
                                       probe_interval=100,max_context_harm=0.,output_error_guard=False)} if particle else {"row_semantics":"conditional"}
        p=E22Policy(recipe,g,d,table=table,encoder=e,router=r,generator_optimizer=og,critic_optimizer=od,
                    roles=[["generator","router","table"] if particle else ["generator"],["critic"]],seed=21,**kwargs)
        p.attach_penalty(recipe.make_critic_penalty(od,collect_stats=True))
        torch.manual_seed(900)
        if devices:torch.cuda.manual_seed(900)
        owned=base.global_rng(device)
    return base.Loop(p,arm,data,torch.Generator().manual_seed(7),torch.Generator().manual_seed(43),owned)


def owners(p):
    return {k:v for k,v in {"G":p.G,"D":p.D,"encoder":p.encoder,"router":p.router,"ema_G":p.ema_G,
        "ema_D":p.opt_d.ema_critic,"ema_encoder":p.ema_encoder,"ema_router":p.ema_router}.items() if v is not None}


def snapshot(loop):
    saved=base.checkpoint(loop)
    saved["all_modes"]={k:{n:m.training for n,m in o.named_modules()} for k,o in owners(loop.policy).items()}
    saved["all_flags"]={k:{n:p.requires_grad for n,p in o.named_parameters()} for k,o in owners(loop.policy).items()}
    saved["all_grads"]={k:{n:None if p.grad is None else p.grad.detach().clone() for n,p in o.named_parameters()} for k,o in owners(loop.policy).items()}
    saved["arithmetic"]={k:o.backbone.final.arithmetic for k,o in owners(loop.policy).items() if hasattr(o,"backbone")}
    return deepcopy(saved)


def restore(loop,saved):
    loop.policy.abort_step()
    for k,o in owners(loop.policy).items():
        for n,p in o.named_parameters():p.requires_grad_(saved["all_flags"][k][n])
    base.restore(loop,saved)
    for k,o in owners(loop.policy).items():
        for n,m in o.named_modules():m.training=saved["all_modes"][k][n]
        for n,p in o.named_parameters():
            grad=saved["all_grads"][k][n];p.grad=None if grad is None else grad.to(p.device).clone()
        if k in saved["arithmetic"]:o.backbone.final.arithmetic=saved["arithmetic"][k]


def observe(loop,x,zero_code=False):
    before=snapshot(loop);value=base.capture(loop.policy,x,zero_code=zero_code)
    if base.digest(snapshot(loop))!=base.digest(before):raise AssertionError("capture changed owner/caller modes/grads/RNG")
    return value.detach().cpu()


def coordinates(p):
    names={id(v):"G."+n for n,v in p.G.named_parameters()}
    if p.router is not None:names.update({id(v):"router."+n for n,v in p.router.named_parameters()})
    names[id(p.table)]="table";names[id(p.log_output_sigma)]="log_output_sigma"
    result=[];seen=set()
    for group in p.opt_g.param_groups:
        for v in group["params"]:
            if id(v) not in seen:result.append((names[id(v)],v));seen.add(id(v))
    return result


@contextmanager
def arithmetic(loop,mode):
    head=loop.policy.G.backbone.final;before=head.arithmetic
    try:head.arithmetic=mode;yield
    finally:head.arithmetic=before


def clean_vjp(loop):
    before=snapshot(loop);params=coordinates(loop.policy);x=loop.data["test"]["context"]
    gradient={n:torch.zeros_like(p,device="cpu",dtype=torch.float64) for n,p in params};pred=[]
    try:
        loop.policy.G.eval()
        for i in range(0,len(x),B):
            batch=x[i:i+B];p=loop.policy
            value=p.G(batch) if p.routed_control is None else p.routed_generate(batch,sigma=0,perturb=False)
            grads=torch.autograd.grad(value.double().square().mean()/(len(x)/B),[p for _,p in params],allow_unused=True)
            for (name,_),grad in zip(params,grads):
                if grad is not None:gradient[name]+=grad.detach().cpu().double()
            pred.append(value.detach().cpu());budget()
    finally:restore(loop,before)
    if base.digest(snapshot(loop))!=base.digest(before):raise AssertionError("offline VJP changed owners/streams")
    return torch.cat(pred),gradient


def secant(y0,y1):
    if y0.shape!=y1.shape or not all(bool(torch.isfinite(v).all()) for v in (y0,y1)):raise ValueError("finite same-shaped secant arrays")
    e=y0.double();dy=y1.double()-e
    result={"MSE0":float(e.square().mean()),"MSE1":float(y1.double().square().mean()),
            "delta_MSE":float((y1.double().square()-e.square()).mean()),
            "output_linear":float((2*e*dy).mean()),"Q":float(dy.square().mean())}
    if abs(result["delta_MSE"]-result["output_linear"]-result["Q"])>1e-14+1e-12*max(abs(v) for v in result.values()):raise AssertionError("F64 secant identity")
    return result


def precision_witness(raw):
    a=secant(raw["BF16_before"],raw["BF16_after"]);b=secant(raw["FP32_before"],raw["FP32_after"])
    slopes={m:math.fsum(float((g*raw["delta"][n]).sum()) for n,g in raw[m+"_gradient"].items()) for m in ("BF16","FP32")}
    discrepancies={m:abs(v["delta_MSE"]-slopes[m]) for m,v in (("BF16",a),("FP32",b))}
    defined=a["Q"]>0 and discrepancies["BF16"]>0
    ratios={"Q":b["Q"]/a["Q"],"discrepancy":discrepancies["FP32"]/discrepancies["BF16"]} if defined else None
    return {"scope":"local final native update, same Parameters/delta; not a convergence criterion","BF16":a,"FP32":b,
            "parameter_slopes":slopes,"absolute_discrepancies":discrepancies,"ratios":ratios,
            "both_precision_ratios_lte":.75,
            "by_source":{str(s):{m:secant(raw[m+"_before"][s*8:(s+1)*8],raw[m+"_after"][s*8:(s+1)*8]) for m in ("BF16","FP32")} for s in range(6)},
            "helpful_parameter_slope_and_finite_harm":bool(slopes["BF16"]<0<a["delta_MSE"]),
            "precision_status":"NA" if not defined else "PASS" if all(r<=.75 for r in ratios.values()) else "FAIL"}


def scientific_gate(accuracy,zero,live,norms):
    ordinary,candidate=accuracy["ordinary_FP32"],accuracy["particle_FP32"]
    for item in (ordinary,candidate,zero):
        if set(item["by_source"])!={str(s) for s in range(6)} or not all(math.isfinite(v) and v>=0 for v in (item["rmse"],*item["by_source"].values())):raise ValueError("finite full six-source metric required")
    checks={"aggregate":candidate["rmse"]<=ordinary["rmse"]*.999,
            "no_source_harm":all(candidate["by_source"][str(s)]<=ordinary["by_source"][str(s)]+1e-6 for s in range(6)),
            "code_aggregate":zero["rmse"]>=candidate["rmse"]*1.001,
            "code_each_source":all(zero["by_source"][str(s)]>candidate["by_source"][str(s)] for s in range(6)),
            "bank_live":live["bank"]>=.9*(STEPS-1),"query_live":live["query"]>=.9*(STEPS-1),
            "heads_live":set(norms)==set(SITES) and all(math.isfinite(v[k]) and v[k]>0 for v in norms.values() for k in ("C","particle_up"))}
    return {"pass":all(checks.values()),"checks":checks,"failed_bounds":[k for k,v in checks.items() if not v],
            "candidate":"particle_FP32","control":"ordinary_FP32","offline_terminal_steps":STEPS}


def finite(loop):
    p=loop.policy;values=[v for o in owners(p).values() for v in o.parameters()]+[p.table,p.log_output_sigma]
    for v in values:
        if not bool(torch.isfinite(v).all()) or (v.grad is not None and not bool(torch.isfinite(v.grad).all())):raise FloatingPointError("learned value/gradient nonfinite")
    for opt in (p.opt_g,p.opt_d):
        for state in opt.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):raise FloatingPointError("optimizer moment nonfinite")


def preflight(data):
    loops={a:make_loop(a,data) for a in ARMS};x=data["test"]["context"][list(MEDIA_INDICES)]
    values={a:observe(l,x) for a,l in loops.items()}
    for mode in ("BF16","FP32"):
        ordinary,particle=loops["ordinary_"+mode].policy,loops["particle_"+mode].policy
        if not torch.equal(values["ordinary_"+mode],values["particle_"+mode]):raise AssertionError("same-arithmetic initial outputs differ")
        for a,b in zip(ordinary.G.branches(),particle.G.branches()):
            if not torch.equal(a.down.weight,b.down.weight) or not torch.equal(a.up.weight,b.up.weight):raise AssertionError("common named initial weights differ")
    for loop in loops.values():
        p=loop.policy;copy=p.ema_G
        if any(v is q or v.data_ptr()==q.data_ptr() for v,q in zip(p.G.parameters(),copy.parameters())):raise AssertionError("FAST/EMA Parameter alias")
        if copy.backbone.final.arithmetic!=p.G.backbone.final.arithmetic or p.G.teacher_backbone.final.arithmetic!="BF16":raise AssertionError("student/teacher/EMA arithmetic")
        finite(loop)
    return {"same_precision_initial_outputs_exact":True,"common_named_DownUp_exact":True,
            "EMA_disjoint_and_precision_preserved":True,"teacher_independent_BF16":True,
            "captures_native_caller_RNG_modes_gradients_immutable":True,"native_updates":0,"forward_count":8}


def run(data,out):
    media={"task":TASK,"steps":list(MEDIA_STEPS),"indices":list(MEDIA_INDICES),"source_ids":list(range(6))+[0,1],
           "target_residual":data["test"]["targets"][list(MEDIA_INDICES)].cpu(),"actual_residuals":{},"captures_immutable":True}
    endpoint={};norms={};live={};traces={};matched={};precision_raw=None;recipes={};initial={}
    for arm in ARMS:
        loop=make_loop(arm,data);p=loop.policy;recipes[arm]=p.recipe.to_dict()
        initial[arm]=base.digest(p.state_dict());media["actual_residuals"][arm]={"0":observe(loop,data["test"]["context"][list(MEDIA_INDICES)])}
        live[arm]={"bank":0,"query":0};camera=data["test"]["context"][list(MEDIA_INDICES)];rows=[]
        frozen={n:v.detach().clone() for n,v in p.G.named_parameters() if not v.requires_grad}
        with (out/(arm+".jsonl")).open("x") as log:
            for step in range(1,STEPS+1):
                if arm=="particle_BF16" and step==STEPS:
                    y,grad=clean_vjp(loop)
                    with arithmetic(loop,"FP32"):yf,gf=clean_vjp(loop)
                    values={n:v.detach().cpu().double().clone() for n,v in coordinates(p)}
                    precision_raw={"BF16_before":y,"FP32_before":yf,"BF16_gradient":grad,"FP32_gradient":gf}
                row=base.update(loop);rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
                for key,label in (("bank_live","bank"),("query_live","query")):
                    if step>1:live[arm][label]+=int(row[key])
                if step in MEDIA_STEPS:media["actual_residuals"][arm][str(step)]=observe(loop,camera)
                if step%64==0:print(json.dumps({"arm":arm,**row}),flush=True)
                budget()
        endpoint[arm]=observe(loop,data["test"]["context"])
        if arm=="particle_BF16":
            precision_raw["BF16_after"]=endpoint[arm]
            with arithmetic(loop,"FP32"):precision_raw["FP32_after"]=observe(loop,data["test"]["context"])
            precision_raw["delta"]={n:v.detach().cpu().double()-values[n] for n,v in coordinates(p)}
        if arm=="particle_FP32":endpoint["zero_code"]=observe(loop,data["test"]["context"],True)
        if arm.startswith("particle"):
            norms[arm]={s:{"C":float(b.bridge.weight[:,data["geometry"].rank:].norm()),"particle_up":float(b.particle_up.weight.norm())} for s,b in zip(SITES,p.G.branches())}
        if any(not torch.equal(v,frozen[n]) for n,v in p.G.named_parameters() if n in frozen):raise AssertionError("frozen task owner changed")
        finite(loop);traces[arm]=base.sha(out/(arm+".jsonl"))
        matched[arm]=base.digest([{k:r[k] for k in ("step","batch_indices","paired_bases","data_rng","paired_rng","penalty_globals")} for r in rows])
        torch.save(snapshot(loop),out/(arm+"-512.pt"));budget()
    if len(set(matched.values()))!=1:raise AssertionError("data/paired/penalty draw cadence differs")
    quality={a:base.accuracy(v,data["test"]["source_ids"]) for a,v in endpoint.items()}
    gate=scientific_gate(quality,quality["zero_code"],live["particle_FP32"],norms["particle_FP32"])
    for path,value in (("endpoint-residuals.pt",endpoint),("observed-media.pt",media),("precision-witness.pt",precision_raw)):
        torch.save(value,out/path);budget()
    return {"complete":True,"task":TASK,"scientific_status":"PASS" if gate["pass"] else "FAIL","gate":gate,
            "accuracy":quality,"precision_witness":precision_witness(precision_raw),"recipes":recipes,"live":live,"norms":norms,
            "trace_sha256":traces,"matched_stream_sha256":matched,"initial_native_sha256":initial,
            "data_digest":data["digest"],"observed_media_sha256":base.sha(out/"observed-media.pt"),
            "endpoint_residual_sha256":base.sha(out/"endpoint-residuals.pt"),"precision_raw_sha256":base.sha(out/"precision-witness.pt"),
            "native_updates":4*STEPS,"scope":"Analytic paired-column stress case; not measured actual geometry or sustained Supra improvement."}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--protocol",type=Path,default=CARD);args=parser.parse_args()
    original=base.backend_flags();threads=torch.get_num_threads();rng=torch.get_rng_state().clone()
    cuda_rng=None;cuda_device=None
    report=None;out=None;error=None;code=2;source=sources();native=base.package_identity();card_sha=base.sha(args.protocol)
    try:
        card=json.loads(args.protocol.read_text())
        expected={"sources":source,"task":TASK,"steps":STEPS,"arms":list(ARMS),"limit_seconds":LIMIT,
                  "geometry":asdict(GEOMETRY),"edit_amplitude":EDIT_SCALE,"sites":list(SITES),"bank_N":N,"bank_Z":Z,
                  "batch_size":B,"media_steps":list(MEDIA_STEPS),"media_indices":list(MEDIA_INDICES)}
        if any(card[k]!=v for k,v in expected.items()):raise ValueError("fixed protocol/source differs")
        if not args.run:raise ValueError("--run required; no model-only preflight command")
        if args.out.exists():raise ValueError("exclusive new output directory required")
        args.out.mkdir(parents=True);out=args.out
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        device=torch.device(args.device)
        if device.type=="cuda":
            cuda_device=torch.cuda.current_device();cuda_rng=torch.cuda.get_rng_state_all();torch.cuda.set_device(device)
        data=make_data(device=device);initial=preflight(data);torch.save(data,out/"inputs.pt");budget()
        report=run(data,out)
        report.update(source_identity=source,imported_native_observed=native,protocol_sha256=card_sha,input_sha256=base.sha(out/"inputs.pt"),initial_prerequisite=initial)
        if sources()!=source or base.package_identity()!=native or base.sha(args.protocol)!=card_sha:raise AssertionError("source/API/card changed withinrun")
        budget();code=0 if report["gate"]["pass"] else 1
    except BaseException as exc:error={"type":type(exc).__name__,"message":str(exc)}
    finally:
        torch.backends.cuda.matmul.allow_tf32=original["allow_tf32"];torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original["allow_bf16_reduced_precision_reduction"]
        torch.set_num_threads(threads);torch.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng);torch.cuda.set_device(cuda_device)
        if time.monotonic()-STARTED>LIMIT:error={"type":"TimeoutError","message":"cleanup exceeds300s"}
        if out is not None:
            if error is not None:report={"complete":False,"task":TASK,"error":error};code=2
            rp=out/"report.json";rp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
            completion={"complete":bool(report["complete"]),"task":TASK,"report_sha256":base.sha(rp),
                        "scientific_status":report.get("scientific_status") if report["complete"] else None,
                        "exit_code":code,"seconds":time.monotonic()-STARTED,"limit_seconds":LIMIT}
            cp=out/"completion.json";cp.write_text(json.dumps(completion,indent=2)+"\n")
            if time.monotonic()-STARTED>LIMIT:
                completion.update(complete=False,scientific_status=None,exit_code=2,error="post-write300s overrun",seconds=time.monotonic()-STARTED);cp.write_text(json.dumps(completion,indent=2)+"\n");code=2
            print(json.dumps(completion),flush=True)
        elif error is not None:print(json.dumps({"complete":False,"error":error}),flush=True)
    return code


if __name__=="__main__":raise SystemExit(main())
