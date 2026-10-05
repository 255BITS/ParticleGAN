"""Executable public-API controls, reconstruction and routed diagnostics.

Every fixture advances actual optimizer/policy state. Gates below define new
diagnostic questions and never reclassify archived cards or train receipts.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED, construction_rng

from copy import deepcopy
import importlib
import math
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import torch
from torch import nn

from particlegan import get_recipe, init, scale_learning_rates
from lib import yue2_particle_toy as sign
from lib import safe_fast_landing as landing
from lib.vendor.concept_slider_core.reference import noise_std
from benchmarks.toy_audit import definition_quality
from benchmarks.toy_audit.api_conditionals import _rng, _bound, observation, view

VERSION="diagnostic-api-v2"
ROOT=Path(__file__).resolve().parents[2]


def _case(name,legacy,title,goal,steps,recipe,thresholds,scope,sampling,batch=16):
    return dict(id="api-"+name,legacy_ids=[legacy],title=title,goal=goal,kind="controlled_diagnostic",
        default_steps=steps,batch_size=batch,eval_samples=1024,default_recipe=recipe,
        terminal_observations=1 if name.startswith("stiff-") else 5,
        thresholds=thresholds,scope=scope,sampling=sampling,
        seed_policy="Protocol seed0; frozen original named native streams remain explicit mathematical controls.")


_CASES=[
    _case("sign-lander-controls","source-family-11","Paired sign restoration versus joint reflection",
        "Restore the sign through paired RpGAN while a reflected joint-marginal controller fails playback",200,"ka2",
        {"paired_landings_min":.95,"paired_relative_mse_max":.05,"collapsed_landings_max":.05,"collapsed_relative_mse_min":2.,"applied_penalty_min":1},
        "Bounded scalar-sign kinematic control; public GANLoss replaces vendor loss calls. No YuE2 or gym claim.",
        "Original odd expert and symmetric states, alpha=-1; joint arm also starts beta=-1;80 heldout40-step episodes",64),
    _case("safe-fast-controls","source-family-12","Live paired GAN plus safe-fast descent",
        "Land quickly and softly while retaining a live adversary; compare slow GAN-only and supervised-only controls",250,"ka2",
        {"combined_landings_min":.95,"combined_crash_rate_max":.02,"combined_timeout_rate_max":.03,"combined_restricted_mean_steps_max":28.,
         "baseline_landings_max":.25,"baseline_mean_steps_min":40.,"landing_gap_min":.5,"step_gap_min":12.,"gan_gradient_sum_min":1.,"baseline_sink_error_max":.02,"late_combined_gan_gradient_min":.2},
        "Original single-sink 48-step kinematic problem, full denominators. RpGAN weight1, combined differentiable plant cost; supervised is an explicitly rejected control.",
        "Original initial states/plant and beta=-4,250 updates;200 independent heldout starts",64),
    _case("ae-anchor-hold","develop-ae_gan_hold","Particle AE reconstruction and prior support",
        "Reconstruct the noisy two-anchor inputs while the independently sampled prior covers both anchors",250,"ka2",
        {"reconstruction_mse_max":.05,"anchor_hold_distance_max":.35,"quality_fraction_min":.90,"quality_mass_tv_max":.10},
        "New public Recipe.encode particle AE with reconstruction weight1 and applied GAN/KA2; independently sampled prior quality is separately scored.",
        "Two equal anchors (+/-1.5,0), sigma.05;12-component MoG z2; clean decoded prior and matched noisy reconstruction",64),
    *[_case("stiff-"+arm,"pr224","Stiff game "+arm,"Remain in the settled paired-GAN neighborhood through the native LR release",96,"e22",
        {"peak_game_excess_max":1e-6},"Constructed native SettleTest unit, not a learned dataset. Native is the expected failing counterexample; cancellation and safe geometry are meaningful positive controls.",
        "Exact published float64 stationary feature game; specified reachable Adam memory; one controlled release at update48",1)
        for arm in ("native","cancel","safe")],
    *[_case("critic-lag-"+arm,"pr226","Paired residual "+arm,"Recover clean heldout paired residuals and report the odd critic's force at correct fit",800,"e22_routed",
        {"normalized_rmse_max":.10},"New absolute clean heldout gate on the original current/even/D-antithetic hosts. This does not convert historical relative improvement into a full-quality PASS.",
        "Published frozen BF16 affine host,64 residual coordinates; original fit/guard/report grid and private paired noise")
        for arm in ("current","even_critic","d_antithetic")],
    *[_case("routed-acquisition-"+arm,"pr227","Routed H/b acquisition "+arm,"Recover the reachable two-site teacher edit and demonstrate a beneficial signed code ablation",6400,"e22_routed",
        {"relative_edit_mse_max":.10,"zero_code_minus_live_game_min":1e-6},
        "Fresh current-source protocol, no archived source qualification or four-cross-trained-critic campaign reuse. Code ablation uses this arm's trained critic and identical heldout noise.",
        "Published six subjects,disjoint fit/guard/test times and latent IDs,128x4 routed bank; original or H/b-neutral initialization",4)
        for arm in ("original","neutral")],
    *[_case("film-"+arm,"pr231","Whole additive FiLM "+arm,"Fit the paired source/time edit under the actual native generator-rate policy",1200,"e22_routed",
        {"relative_recipient_mse_max":.10},
        "New absolute learned gate relative to the zero-recipient witness (no outer host identity). Original PR231 remains NO_FROZEN_GATE. Shift-zero is whole additive FiLM erasure, distinct from PR227 H/b neutralization. A bypass PASS would not prove causal damping repair.",
        "Published nonlinear frozen BF16 host,width16/two blocks,third heldout grid; original native or declared shift-zero/bypass/antithetic profile")
        for arm in ("original_native","shift_zero_native","shift_zero_g_bypass","shift_zero_antithetic")],
    _case("routed-paired","source-family-14","Source routed paired edit","Recover a paired edit while applying native routed controls",160,"e22_routed",
        {"heldout_rmse_max":.10},"Original one-site E22 paired source with a new absolute heldout error gate, no archived qualification.","Published fit/guard/test grid,16x2 bank,batch32",32),
    _case("routed-support","source-family-14","Two-site support acquisition","Acquire the two moving spatial transitions on heldout contexts with both routed sites active",1200,"e22_routed",
        {"heldout_rmse_max":.10},"Original public-initialized two-site support source,tokens128,128x4 bank. Default zero-context-harm row acceptance is preserved.","Original source grids and spatial target; independent heldout times/positions",8),
    _case("routed-moving","source-family-14","Moving edit acquisition and hold","Meet an absolute heldout error at each of three target orientations without losing pair identity",1500,"e22_routed",
        {"completed_orientation_checks_min":3,"max_terminal_rmse_max":.10},"Original current-native moving paired host with two30-degree target shifts at500/1000. Three absolute500-update endpoints required; no early-prefix convergence credit.","Same source paired contexts; edit rotates while source BF16 host is fixed",32),
    _case("routed-replay","source-family-14","Complete routed activation replay","Preserve exact update owners/RNG under activation checkpointing and fit the heldout paired task",160,"e22_routed",
        {"checkpoint_owner_mismatch_max":0.,"heldout_rmse_max":.10},"Actual paired baseline versus full two-site checkpointed native updates; exact owner parity is software evidence, clean error is a separate learned bound.","Original two-site source,tokens8,16x2 bank; same private DV12 and paired-noise streams",8),
]
for _row in _CASES:
    _name=_row["id"].removeprefix("api-")
    if _name.startswith("stiff-"): _row["eval_samples"]=1
    elif _name.startswith("critic-lag-"): _row["eval_samples"]=144
    elif _name.startswith("routed-acquisition-"): _row["eval_samples"]=96
    elif _name.startswith("film-"): _row["eval_samples"]=256
    elif _name.startswith("routed-"): _row["eval_samples"]=180
    elif _name=="safe-fast-controls": _row["eval_samples"]=200
    elif _name=="sign-lander-controls": _row["eval_samples"]=256
    if _name!="ae-anchor-hold":
        _row["evaluation_count_policy"]="Fixed original source panels/streams, not redrawn when caller supplies n; actual cohort counts are reported"
    if _name=="routed-replay":
        _row["sampling"]+="; explicit public API initialization and zero-context-harm row guard"
    if _name=="routed-moving":
        # The three frozen orientation checks complete only at update 1500.
        # Requiring several final observations would make this finite protocol
        # impossible to qualify even when all three endpoint bounds pass.
        _row["terminal_observations"]=1
CASES={c["id"]:c for c in _CASES}


def list_cases(): return deepcopy(_CASES)


def _example(name):
    """Load shipped caller sources, rejecting cached imports from another checkout."""
    path=ROOT/"examples"
    for imported,module in list(sys.modules.items()):
        if imported.startswith("e22_routed_") and getattr(module,"__file__",None):
            if Path(module.__file__).resolve().parent!=path: raise ValueError("foreign-checkout example import: "+imported)
    old=list(sys.path)
    try:
        sys.path.insert(0,str(path))
        return SimpleNamespace(**runpy.run_path(str(path/(name+".py"))))
    finally: sys.path[:]=old


def _tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        return (isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
                and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a) is type(b) and len(a)==len(b) and all(_tree_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b): return True
    return type(a) is type(b) and a==b


def _owned(models):
    return {name:dict(state=module.state_dict(),modes={k:m.training for k,m in module.named_modules()},
        gradients={k:None if p.grad is None else p.grad.clone() for k,p in module.named_parameters()},requires_grad={k:p.requires_grad for k,p in module.named_parameters()})
        for name,module in models.items() if isinstance(module,nn.Module)}


class NativeFixture:
    api_components=("Recipe","E22Policy","RoutedRows","GANLoss","native policy lifecycle","public initialization")
    def __init__(self,case,max_steps):
        self.case=case;self.limit=case["default_steps"] if max_steps is None else max_steps
        self.completed_steps=0;self.last={};self.period_checks=[];self.replay_equal=True
        name=case["id"].removeprefix("api-")
        self.name=name
        if name.startswith("stiff"):
            self.api=importlib.import_module("examples.e22_stiff_game_reopen")
            self.loop=self.api.make_fixture(contracted_stiff_factor=.8 if name.endswith("safe") else 1.6)
            self.recipe=self.loop.recipe if hasattr(self.loop,"recipe") else get_recipe("e22",lr=self.api.BASE_LR)
            self.initial_game=float(self.loop.game().detach());self.peak_game=self.initial_game
            self.api_components=("Recipe.make_generator_optimizer","GANLoss","SettleTest")
        elif name.startswith("critic-lag"):
            self.api=importlib.import_module("examples.e22_paired_residual_toy")
            self.loop=self.api.make_loop(name.removeprefix("critic-lag-"));self.recipe=self.loop.policy.recipe
        elif name.startswith("routed-acquisition"):
            self.api=importlib.import_module("examples.e22_routed_convergence")
            data=self.api.make_data()
            if name.endswith("neutral"):
                neutral=importlib.import_module("examples.e22_routed_convergence_neutral")
                self.loop=neutral.make_neutral_loop(neutral.make_neutral_data(data))
            else: self.loop=self.api.make_loop("particle_native_game",data)
            self.panels=self.api.evaluation_panels(self.loop.data);self.recipe=self.loop.policy.recipe
        elif name.startswith("film"):
            self.api=importlib.import_module("benchmarks.routed_conditioning.film_damping")
            self.loop=self.api.make_loop(name.removeprefix("film-"));self.recipe=self.loop.policy.recipe
        else:
            example="e22_routed_support" if name=="routed-support" else "e22_routed_sites" if name=="routed-replay" else "e22_routed_paired"
            self.api=_example(example)
            self.loop=self.api.make_loop(initialization="api") if name=="routed-replay" else self.api.make_loop()
            self.recipe=self.loop.policy.recipe
            if name=="routed-moving":
                self.moving=_example("e22_routed_moving").MovingTarget(self.loop)
            if name=="routed-replay":
                self.replay_api=_example("e22_routed_replay")
                self.baseline=self.api.make_loop(initialization="api")
        if type(self.limit) is not int or not 0<self.limit<=case["default_steps"]: raise ValueError("execution cap exceeds frozen protocol")
    def step(self):
        if self.completed_steps>=self.limit: raise RuntimeError("execution cap reached")
        if self.name.startswith("stiff"):
            self.last=self.api.advance(self.loop,self.completed_steps+1,cancel_first_release=self.name.endswith("cancel"))
            self.peak_game=max(self.peak_game,self.last["game_after"])
        else:
            if self.name=="routed-moving" and self.completed_steps in (500,1000):
                self.moving.turn_to(math.radians(30)*(self.completed_steps//500))
            if self.name=="routed-replay":
                self.api.update(self.baseline)
                self.last=self.api.update(self.loop,generator_forward=self.replay_api.checkpointed_generate)
                self.replay_equal &= _tree_equal(self.api.checkpoint(self.loop),self.api.checkpoint(self.baseline))
                self.replay_equal &= _tree_equal(_owned(self.loop.policy._training_modules()),_owned(self.baseline.policy._training_modules()))
            else: self.last=self.api.update(self.loop)
        self.completed_steps+=1
        if self.name=="routed-moving" and self.completed_steps%500==0:
            self.period_checks.append(self.api.evaluate(self.loop)["heldout_rmse"])
        return deepcopy(self.last)
    @torch.no_grad()
    def observe(self,n=1024,seed=99123):
        if self.name.startswith("stiff"):
            residual=self.loop.generator.bias.detach()*self.loop.generator.bias.new_tensor(self.api.FEATURE_SCALES)
            metrics=dict(game=float(self.loop.game().detach()),peak_game_excess=self.peak_game-self.initial_game,
                completed_updates=self.completed_steps,settled_scale=float(self.loop.tester.s),evaluation_states=1)
            return observation(metrics,_bound(metrics,"peak_game_excess",hi=1e-6),[view("Settled paired-game feature residual",torch.zeros_like(residual),residual,"bar",caption="Native release is allowed if safe; the stiff native arm is the deliberate counterexample.")])
        p=self.loop.policy
        if self.name.startswith("routed-acquisition"):
            values=self.loop.data["test"]
            predictions=torch.cat([self.api.forward(self.loop,values["context"][s:s+self.api.BATCH_SIZE]) for s in range(0,len(values["context"]),self.api.BATCH_SIZE)])
            target=values["targets"]
            baseline=(values["base"]-target).square().mean()
            live=self.api.evaluate(self.loop,p.D,"test",self.panels)
            zero=self.api.evaluate(self.loop,p.D,"test",self.panels,code_ablation=True)
            metrics=dict(relative_edit_mse=float((predictions-target).square().mean()/baseline),zero_code_minus_live_game=zero["paired_game"]-live["paired_game"],evaluation_contexts=len(target),ablation_noise_draws=self.api.PANEL_DRAWS)
            failures=_bound(metrics,"relative_edit_mse",hi=.10)+_bound(metrics,"zero_code_minus_live_game",lo=1e-6)
            views=[view("Two-site teacher edit at heldout subjects/times",(target-values["base"])[:8],(predictions-values["base"])[:8],"image",vmin=-.4,vmax=.4,caption="Actual live clean edit. Signed common-noise ablation must worsen this arm's trained judge, not merely change it.")]
        else:
            if self.name.startswith("critic-lag"):
                context,target=self.loop.report_context,self.loop.report_targets
            else: context,target=self.loop.test_context,self.loop.test_targets
            # Independent frozen serving, no training RNG or gradient ownership changes.
            prediction=p.served_model().routed_forward(context)
            mse=float((prediction-target).square().mean())
            if self.name.startswith("critic-lag"):
                metrics=dict(normalized_rmse=math.sqrt(mse),zero_residual_odd_force=self.api.evaluate(self.loop)["diagnostics"]["zero_residual_G_odd_force_norm"],evaluation_contexts=len(target))
                failures=_bound(metrics,"normalized_rmse",hi=.10)
            elif self.name.startswith("film"):
                # This host predicts recipient features through small heads;
                # it has no outer source identity. The untouched source host
                # would be an artificially distant denominator and let zero
                # recipient output pass. Use its correct zero-output witness.
                baseline=float(target.square().mean())
                metrics=dict(relative_recipient_mse=mse/max(baseline,1e-12),heldout_mse=mse,evaluation_contexts=len(target))
                failures=_bound(metrics,"relative_recipient_mse",hi=.10)
            else:
                metrics=dict(heldout_rmse=math.sqrt(mse),evaluation_contexts=len(target))
                failures=_bound(metrics,"heldout_rmse",hi=.10)
                if self.name=="routed-moving":
                    metrics.update(completed_orientation_checks=len(self.period_checks),max_terminal_rmse=max(self.period_checks,default=1e6))
                    failures+=_bound(metrics,"completed_orientation_checks",lo=3)+_bound(metrics,"max_terminal_rmse",hi=.10)
                elif self.name=="routed-replay":
                    metrics["checkpoint_owner_mismatch"]=int(not self.replay_equal)
                    failures+=_bound(metrics,"checkpoint_owner_mismatch",hi=0)
            if prediction.ndim==2:
                views=[view("Matched heldout conditional targets and served outputs",target,prediction,caption="Each output uses its own source/time context; target marginal agreement alone cannot pass.")]
            elif self.name.startswith("critic-lag"):
                # These are two 32-coordinate feature blocks, not two color
                # channels. Move the singleton axis without changing values.
                views=[view("Matched paired residual feature blocks",
                    target[:8].reshape(-1,1,2,32),prediction[:8].reshape(-1,1,2,32),
                    "image",vmin=-2.5,vmax=2.5,
                    caption="Two 32-coordinate residual feature blocks (64 total), shown as grayscale rows. Target and served output use the same heldout context.")]
            else:
                views=[view("Matched heldout paired outputs",target[:8],prediction[:8],"image",vmin=-2.5,vmax=2.5,caption="Clean actual served output, with the same context's target. Original host/bank/control sizes are retained.")]
        return observation(metrics,failures,views)
    def state_dict(self):
        if self.name.startswith("stiff"):
            state=self.loop.state_dict()
        else:
            state=self.api.checkpoint(self.loop)
            state["owned_training_modules"]=_owned(self.loop.policy._training_modules())
        result=dict(version=VERSION,case_id=self.case["id"],limit=self.limit,completed_steps=self.completed_steps,state=state,last=deepcopy(self.last),period_checks=list(self.period_checks),replay_equal=self.replay_equal,global_rng=torch.random.get_rng_state())
        if self.name.startswith("stiff"): result.update(initial_game=self.initial_game,peak_game=self.peak_game)
        if self.name=="routed-replay":
            result["baseline"]=self.api.checkpoint(self.baseline)
            result["baseline_owned_modules"]=_owned(self.baseline.policy._training_modules())
        return deepcopy(result)


class ScalarCritic(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.LeakyReLU(.2),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class GainGame:
    """Incremental scalar game using the current public loss/optimizer/penalty."""
    def __init__(self,mode,*,safe_fast=False):
        self.mode,self.safe_fast=mode,safe_fast;self.steps=0
        self.value=nn.Parameter(torch.tensor(-4. if safe_fast else -1.))
        self.beta=nn.Parameter(torch.tensor(-1.)) if mode=="collapsed" else None
        self.recipe=get_recipe("ka2",total_steps=250 if safe_fast else 200,batch_size=64)
        self.norm=landing._edit_scale() if safe_fast else sign._edit_scale()
        with torch.random.fork_rng(devices=[]): self.D=ScalarCritic(6 if self.beta is not None else 2)
        init.deterministic_orthogonal_(self.D,seed=1)
        params=[self.value]+([] if self.beta is None else [self.beta])
        self.opt_g=self.recipe.make_generator_optimizer(params,lr=.15 if safe_fast else .05,foreach=False)
        self.opt_d=self.recipe.make_critic_optimizer(self.D,ema_critic=deepcopy(self.D),foreach=False)
        self.penalty=self.recipe.make_critic_penalty(self.opt_d,collect_stats=True)
        self.loss=self.recipe.make_loss();self.rng=_rng(2)
        self.rates=[[group["lr"] for group in opt.param_groups] for opt in (self.opt_g,self.opt_d)]
        self.gan_gradient_sum=0.;self.late_gradient_sum=0.;self.late_count=0;self.applications=0
    def policy(self,state):
        return landing.pd_action(state,landing.sink_of(self.value)) if self.safe_fast else (self.value*sign.expert_action(state)).clamp(-1,1)
    def step(self):
        scale_learning_rates(self.steps,self.recipe,(self.opt_g,self.opt_d),self.rates)
        if self.safe_fast: state=landing.initial_states(64,10001+self.steps);target=landing.expert_action(state)
        else: state=torch.rand(64,4,generator=self.rng)*2-1;target=sign.expert_action(state)
        if self.mode=="supervised":
            objective=landing._safe_fast(self.value,landing.initial_states(32,3+self.steps)) if self.safe_fast else (self.policy(state)-target).square().mean()/target.square().mean()
            self.opt_g.zero_grad();objective.backward();self.opt_g.step();self.steps+=1
            return dict(step=self.steps,supervised_control_loss=float(objective.detach()),gan_applied=0)
        sigma=noise_std(self.steps,start=self.norm.noise_start,decay_steps=self.recipe.total_steps,hold=1.3*float(self.norm.edit_rms))
        def pair(pred,epsilon):
            if self.beta is not None:
                return torch.cat((state,target),1),torch.cat((self.beta*state,pred),1)
            return epsilon,epsilon+(pred-target)/self.norm.target_std
        with torch.no_grad():
            epsilon=torch.randn(target.shape,generator=self.rng)*sigma
            real,fake=pair(self.policy(state),epsilon)
        dl=self.loss.d_loss(self.D(real),self.D(fake));penalty=self.penalty(self.D,real,fake)
        self.opt_d.zero_grad();(dl+penalty).backward();self.opt_d.step();self.applications+=int(penalty.requires_grad)
        flags=[p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            real,fake=pair(self.policy(state),torch.randn(target.shape,generator=self.rng)*sigma)
            with torch.no_grad(): reference=self.D(real)
            gl=self.loss.g_loss(self.D(fake),reference)
            self.opt_g.zero_grad();gl.backward()
            gradient=abs(float(self.value.grad));self.gan_gradient_sum+=gradient
            if self.steps+1>self.recipe.total_steps*.6: self.late_gradient_sum+=gradient;self.late_count+=1
            if self.mode=="combined": landing._safe_fast(self.value,landing.initial_states(24,20001+self.steps)).backward()
            self.opt_g.step()
        finally:
            for p,f in zip(self.D.parameters(),flags): p.requires_grad_(f)
        self.steps+=1
        return dict(step=self.steps,g_gan=float(gl.detach()),d_gan=float(dl.detach()),critic_penalty=float(penalty.detach()),gan_applied=1)
    def state_dict(self):
        return deepcopy(dict(value=self.value.detach(),beta=None if self.beta is None else self.beta.detach(),critic=_owned({"D":self.D}),opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),rng=self.rng.get_state(),steps=self.steps,gan_gradient_sum=self.gan_gradient_sum,late_gradient_sum=self.late_gradient_sum,late_count=self.late_count,applications=self.applications,
            value_gradient=None if self.value.grad is None else self.value.grad.detach().clone(),beta_gradient=None if self.beta is None or self.beta.grad is None else self.beta.grad.detach().clone()))


class LandingFixture:
    api_components=("Recipe.make_generator_optimizer","Recipe.make_critic_optimizer","Recipe.make_loss","Recipe.make_critic_penalty","scale_learning_rates")
    def __init__(self,case,max_steps):
        self.case,self.limit=case,case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0<self.limit<=case["default_steps"]: raise ValueError("execution cap exceeds frozen budget")
        self.safe_fast=case["id"]=="api-safe-fast-controls"
        modes=("baseline","combined","supervised") if self.safe_fast else ("paired","collapsed","supervised")
        self.games={name:GainGame(name,safe_fast=self.safe_fast) for name in modes};self.recipe=next(iter(self.games.values())).recipe
        self.completed_steps=0;self.last={}
    def step(self):
        if self.completed_steps>=self.limit: raise RuntimeError("execution cap reached")
        self.last={name:game.step() for name,game in self.games.items()};self.completed_steps+=1
        return deepcopy(self.last)
    @torch.no_grad()
    def observe(self,n=1024,seed=99123):
        metrics,views,failures={},[],[]
        if self.safe_fast:
            metrics["rollout_episodes_per_arm"]=200
            for name,game in self.games.items():
                starts=landing.initial_states(200,1000)
                landed,crashed,steps=landing._hard_rollout(game.policy,starts)
                result=definition_quality.landing_metrics(landed.numpy(),crashed.numpy(),steps.numpy())
                metrics.update({name+"_"+k:float(result[k]) for k in ("landings","crash_rate","timeout_rate","restricted_mean_steps")})
                metrics[name+"_mean_steps"]=float(steps[landed].mean()) if bool(landed.any()) else float(landing.HORIZON)
                state=starts[:8].clone();path=[state[:,:2].clone()]
                target=starts[:8].clone();truth=[target[:,:2].clone()]
                for _ in range(48):
                    state=landing.kinematic_step(state,game.policy(state));target=landing.kinematic_step(target,landing.pd_action(target,landing.QUICK_SINK))
                    path.append(state[:,:2].clone());truth.append(target[:,:2].clone())
                views.append(view(name+": actual48-step descent",torch.stack(truth,1),torch.stack(path,1),"line",caption="Quick-soft reference and actual scalar policy. Every start, including crash and timeout, stays in the metric denominator."))
            baseline,combined=self.games["baseline"],self.games["combined"]
            metrics.update(landing_gap=metrics["combined_landings"]-metrics["baseline_landings"],step_gap=metrics["baseline_mean_steps"]-metrics["combined_mean_steps"],gan_gradient_sum=combined.gan_gradient_sum,
                baseline_sink_error=abs(float(landing.sink_of(baseline.value))-landing.SLOW),late_combined_gan_gradient=combined.late_gradient_sum/max(1,combined.late_count),applied_penalties=combined.applications,supervised_gan_applications=self.games["supervised"].applications)
            limits=[("combined_landings",.95,None),("combined_crash_rate",None,.02),("combined_timeout_rate",None,.03),("combined_restricted_mean_steps",None,28),("baseline_landings",None,.25),("baseline_mean_steps",40,None),("landing_gap",.5,None),("step_gap",12,None),("gan_gradient_sum",1,None),("baseline_sink_error",None,.02),("late_combined_gan_gradient",.2,None),("applied_penalties",1,None),("supervised_gan_applications",None,0)]
        else:
            metrics.update(rollout_episodes_per_arm=80,action_rows_per_arm=256)
            for name,game in self.games.items():
                landings,relative=sign._rollout(game.policy)
                metrics[name+"_landings"],metrics[name+"_relative_mse"]=landings,relative
                rng=_rng(123);pos=torch.rand(8,2,generator=rng)*1.2-.6;velocity=torch.rand(8,2,generator=rng)*.4-.2
                exact_pos,exact_vel=pos.clone(),velocity.clone();path=[pos.clone()];truth=[pos.clone()]
                for _ in range(40):
                    velocity+=sign.GAIN*game.policy(torch.cat((pos,velocity),1));pos+=sign.GAIN*velocity
                    exact_vel+=sign.GAIN*sign.expert_action(torch.cat((exact_pos,exact_vel),1));exact_pos+=sign.GAIN*exact_vel
                    path.append(pos.clone());truth.append(exact_pos.clone())
                views.append(view(name+": sign-dependent40-step playback",torch.stack(truth,1),torch.stack(path,1),"line",caption="Real state is fed back; joint-reflected state coordinates are never substituted into the plant."))
            metrics["applied_penalties"]=self.games["paired"].applications
            limits=[("paired_landings",.95,None),("paired_relative_mse",None,.05),("collapsed_landings",None,.05),("collapsed_relative_mse",2,None),("applied_penalties",1,None)]
        failures=[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)]
        return observation(metrics,failures,views)
    def state_dict(self):
        return deepcopy(dict(version=VERSION,case_id=self.case["id"],limit=self.limit,completed_steps=self.completed_steps,
            games={name:game.state_dict() for name,game in self.games.items()},global_rng=torch.random.get_rng_state(),last=self.last))


class AEFixture:
    api_components=("Recipe.encode","Recipe.make_prior","Recipe.make_optimizers","GANLoss","Recipe.make_critic_penalty")
    def __init__(self,case,seed,max_steps):
        self.case,self.limit=case,case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0<self.limit<=case["default_steps"]: raise ValueError("invalid execution cap")
        self.recipe=get_recipe("ae_gan",total_steps=250,num_particles=12,z_dim=2,batch_size=64,lr=.002,input_noise_std=0.,output_noise_std=0.,sigma_rel=.025)
        with construction_rng(seed, "cpu"):
            self.E=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(.2),nn.Linear(32,4))
            self.G=nn.Sequential(nn.Linear(2,32),nn.LeakyReLU(.2),nn.Linear(32,2))
            self.D=ScalarCritic(2);self.prior=self.recipe.make_prior()
        for i,module in enumerate((self.E,self.G,self.D,self.prior)): init.deterministic_orthogonal_(module,seed=seed+i)
        self.opt_g,self.opt_d=self.recipe.make_optimizers(self.G,self.D,self.prior,encoder=self.E,ema_critic=deepcopy(self.D),foreach=False)
        self.loss=self.recipe.make_loss();self.penalty=self.recipe.make_critic_penalty(self.opt_d,collect_stats=True)
        self.rates=[[group["lr"] for group in opt.param_groups] for opt in (self.opt_g,self.opt_d)]
        self.data_rng,self.latent_rng=_rng(seed+10),_rng(seed+11);self.completed_steps=0;self.last={}
    def draw(self,n,rng):
        anchors=torch.tensor([[-1.5,0.],[1.5,0.]])
        return anchors[torch.randint(2,(n,),generator=rng)]+.05*torch.randn(n,2,generator=rng)
    def reconstruct(self,x):
        query,offset=self.E(x).chunk(2,dim=1)
        return self.G(self.recipe.encode(query,self.prior,offset=offset).codes[:,0])
    def step(self):
        if self.completed_steps>=self.limit: raise RuntimeError("execution cap reached")
        scale_learning_rates(self.completed_steps,self.recipe,(self.opt_g,self.opt_d),self.rates,self.prior)
        real=self.draw(64,self.data_rng)
        with torch.no_grad(): fake=self.G(self.prior.sample(64,generator=self.latent_rng)[0])
        dl=self.loss.d_loss(self.D(real),self.D(fake));reg=self.penalty(self.D,real,fake)
        self.opt_d.zero_grad();(dl+reg).backward();self.opt_d.step()
        flags=[p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            fake=self.G(self.prior.sample(64,generator=self.latent_rng)[0])
            with torch.no_grad(): logits=self.D(real)
            gl=self.loss.g_loss(self.D(fake),logits);recon=(self.reconstruct(real)-real).square().mean()
            self.opt_g.zero_grad();(gl+self.recipe.reconstruction_weight*recon).backward();self.opt_g.step()
        finally:
            for p,f in zip(self.D.parameters(),flags):p.requires_grad_(f)
        self.completed_steps+=1;self.last=dict(step=self.completed_steps,g_gan=float(gl.detach()),reconstruction_mse=float(recon.detach()),d_gan=float(dl.detach()),critic_penalty=float(reg.detach()))
        return deepcopy(self.last)
    @torch.no_grad()
    def observe(self,n=1024,seed=99123):
        data=self.draw(n,_rng(seed));recon=self.reconstruct(data);fake=self.G(self.prior.sample(n,generator=_rng(seed+1))[0])
        anchors=torch.tensor([[-1.5,0.],[1.5,0.]])
        distance,ids=torch.cdist(fake,anchors).min(1);quality=distance<=.15
        mass=torch.bincount(ids[quality],minlength=2).float()/n
        metrics=dict(reconstruction_mse=float((recon-data).square().mean()),anchor_hold_distance=float(torch.cdist(anchors,fake).min(1).values.mean()),quality_fraction=float(quality.float().mean()),quality_mass_tv=float(((mass-.5).abs().sum()+1-mass.sum())/2))
        limits=[("reconstruction_mse",None,.05),("anchor_hold_distance",None,.35),("quality_fraction",.90,None),("quality_mass_tv",None,.10)]
        return observation(metrics,[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)],
            [view("Noisy inputs and matched AE reconstructions",data,recon,caption="Matched rows score reconstruction separately from the learned prior."),view("Independent clean decoded prior: both anchors",self.draw(n,_rng(seed+2)),fake,caption="Two good reconstruction rows cannot substitute for prior coverage.")])
    def state_dict(self):
        return deepcopy(dict(version=VERSION,case_id=self.case["id"],limit=self.limit,completed_steps=self.completed_steps,models=_owned({"E":self.E,"G":self.G,"D":self.D,"prior":self.prior}),
            recipe=self.recipe.to_dict(),opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),data_rng=self.data_rng.get_state(),latent_rng=self.latent_rng.get_state(),global_rng=torch.random.get_rng_state(),last=self.last))


def build_case(case_id,*,device="cpu",seed=DEFAULT_SEED,recipe_name="atlas",max_steps=None):
    if case_id not in CASES: raise ValueError("unknown diagnostic API case: "+str(case_id))
    if torch.device(device).type!="cpu": raise ValueError("Frozen diagnostic sources are CPU-only")
    case=deepcopy(CASES[case_id])
    if recipe_name!=case["default_recipe"]: raise ValueError("This source protocol requires its declared recipe: "+case["default_recipe"])
    if type(seed) is not int or seed != DEFAULT_SEED:
        raise ValueError("Fixed native protocol uses named source streams; use protocol seed0")
    if case_id=="api-ae-anchor-hold": return AEFixture(case,seed,max_steps)
    if case_id in ("api-sign-lander-controls","api-safe-fast-controls"): return LandingFixture(case,max_steps)
    return NativeFixture(case,max_steps)
