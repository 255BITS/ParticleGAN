"""Current public-API conditional toys with frozen, question-specific gates.

These are new hosts/protocols, not revisions of archived qualifications. The
stochastic hosts fit a clean conditional kernel directly (not a DDGAN chain).
The paired hosts fit error against noise through the public RpGAN/KA2 factories.
No oracle prediction enters a generator and no pass is assumed from its recipe.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED, construction_rng

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import get_recipe, init, scale_learning_rates
from lib.sparse_toy import SparseMixedToy
from lib.denoising_toy import GaussianGrid
from lib.trajectory import Routes
from lib.transition import Transitions
from benchmarks.toy_audit import source_conditional_scoring as density
from benchmarks.toy_audit.api_sources import circle, sprite, misgan

VERSION = "conditional-api-v1"
EVAL_SEED = 99123


def _definition(name, legacy, title, goal, steps, *, kind="paired", thresholds=None,
                scope=None, sampling=None):
    return dict(id="api-" + name, legacy_ids=[legacy], title=title, goal=goal,
        kind=kind, default_steps=steps, batch_size=64, eval_samples=1024,
        default_recipe="ka2", thresholds=thresholds or {"relative_mse_max": .10},
        sampling=sampling or "private named training stream; independent fixed heldout rows",
        scope=scope or "New caller-owned clean paired-error RpGAN/KA2 host; historical results remain unchanged.")


_CASES = [
    _definition("sparse-identity", "source-family-00", "Sparse identity symbols", "Recover the joint class, sparse mode and matching symbol law", 8000, kind="conditional_distribution",
        thresholds=density.GATES, sampling="8 uniform classes, 8 uniform modes/class, 3 active coordinates; Gaussian .05 only on active support; identity symbol",
        scope="Direct clean joint conditional kernel; no diffusion-chain or historical source convergence claim."),
    _definition("sparse-split", "source-family-01", "Sparse split symbols", "Recover both valid symbols per class and their pairing with each sparse mode", 8000, kind="conditional_distribution",
        thresholds=density.GATES, sampling="Same fixed 64-mode sparse bank; split symbol 2*class+(mode//8)%2",
        scope="Direct clean joint conditional kernel with 16 symbols; no diffusion-chain claim."),
    *[_definition(f"posterior-{classes}class", f"source-family-{2 if classes == 1 else 3:02}", f"Analytic posterior, {classes} class(es)", "Sample the full clean conditional posterior, including its mode masses and local variance", 7000, kind="conditional_distribution",
        thresholds={"quality_mass_tv_max": .10, "hq_min": .95, "shape": density.GATES},
        sampling="Exact GaussianGrid posterior at class/observation/alpha_bar; heldout zero and displaced observations at alpha_bar .5 and .9",
        scope="Direct posterior sampler over four fixed observation/time panels per class; not iterative denoising or all possible observations.") for classes in (1, 4)],
    *[_definition(f"routes-{mode}", f"source-family-{4 if mode == 'discrete' else 5:02}", f"Two-route sequence ({mode} exposure)", "Cover both class-weighted obstacle routes at unseen geometries with genuine within-route variation", 28000, kind="conditional_distribution",
        thresholds={"valid_fraction_min": .95, "quality_route_mass_tv_max": .10, "coefficient_support_min": .95,
                    "coefficient_mean_abs_max": .10, "coefficient_variance": [.22, .45], "coefficient_ks_max": .10},
        sampling=f"Original Routes conditional kernel; {mode} training geometries, four unseen test geometries, p(upper|class)=(.8,.3), 3 independent U[-1,1] coefficients",
        scope="Direct complete 64-point trajectory generator; class and geometry only, never route/coefficients as input.") for mode in ("discrete", "continuous")],
    *[_definition(f"transitions-{mode}", f"source-family-{6 if mode == 'discrete' else 7:02}", f"Local route transitions ({mode} exposure)", "Fit a coherent joint state/action/successor law conditional on class, geometry and tick", 28000, kind="conditional_distribution",
        thresholds={"joint_consistency_rms_max": .002, "max_projection_cdf_error_max": .12},
        sampling=f"Original Transitions 6D kernel, {mode} training contexts and uniform ticks 0..62; heldout geometry with ticks 0,8,31,54,62",
        scope="Direct joint conditional transition kernel; five frozen time panels include both boundaries. This is not a policy or full-trajectory rollout.") for mode in ("discrete", "continuous")],
    *[_definition(f"transport-{task}", f"source-family-{8 if task == 'affine2' else 9:02}", f"Paired {task} transport", "Recover the correct source-to-target map rather than just the target marginal", 6000,
        thresholds={"normalized_mse_max": .01, "p95_l2_max": .20}, sampling="Independent uniform square [-sqrt(3),sqrt(3)]^2; affine rotation/translation or radius-dependent 1.7-radian swirl",
        scope="New direct conditional MLP and public paired-error game; target map and heldout pairing retained, archived dense adapter recipe not replayed.") for task in ("affine2", "swirl2")],
    _definition("trajectory-edit", "develop-trajectory", "Paired slow-to-fast trajectory edit", "Change angular speed while preserving each trajectory's radius and starting phase", 400,
        thresholds={"paired_mse_max": .02}, sampling="Original 12 phases, radii .45/.7/.95, 8 frames; slow .45 to fast 2.2 angular speed"),
    _definition("trajectory-residual", "develop-residual_student", "Residual trajectory edit with identity hold", "Recover the fast trajectory with a residual head and reject a correct marginal with wrong identities", 400,
        thresholds={"paired_mse_max": .02, "wrong_pair_margin_min": .02}, sampling="Same 12 exact pairs as trajectory-edit; residual output is added to its own slow input"),
    _definition("unipolar-hold", "develop-unipolar", "Positive edit with a learned neutral hold", "Make the positive 4D edit while holding the free scale-zero origin", 400,
        thresholds={"cover_min": .85, "off_axis_max": .05, "neutral_hold_min": .85}, sampling="Scale 0/1 with equal mass; plus=[1,0,0,0], free trainable neutral output"),
    _definition("unused-token-hold", "develop-unused_token_hold", "Unused slot hold during concept motion", "Move the concept slot on its target axis while keeping the unused slot fixed", 200,
        thresholds={"concept_move_min": .85, "unused_hold_min": .85}, sampling="Two slots; unused [1,0], concept [0,0]; concept target [0,1]; shared residual plus per-slot correction"),
    _definition("guarded-leftover", "develop-cover_leftover", "Guarded bipolar edit and content", "Cover both signed poles while preserving content and removing the guarded leak", 800,
        thresholds={"pole_relative_l2_max": .10, "leak_ratio_max": .10}, sampling="Original LeftoverField .55 content, .45 leak; faithful_guard_e target at -1,0,+1"),
    _definition("midscale-identity", "develop-mid_scale_identity", "Intermediate identity and concept control", "Retain identity at half strength in addition to correct neutral and signed poles", 800,
        thresholds={"relative_mse_max": .10, "max_context_l2_max": .15}, sampling="Original smile_teacher guarded concept and .55 content identity at -1,0,.5,+1; the new nonlinear conditional MLP does not impose linear interpolation"),
    _definition("circle-controller", "pr22", "Contextual circle action and 1024-step playback", "Recover radius and signed speed at unseen geometry/speed cells under true closed-loop playback", 2000,
        thresholds={"radial_rmse_max": .10, "signed_speed_error_max": .03, "direction_agreement_min": .95,
                    "recovery_radial_rmse_max": .10, "turns_min": 1.}, sampling="Pinned PR22 independent rows, disjoint heldout geometry/speed buckets, both directions; 128 episodes, main1024/recovery64",
        scope="New direct paired action network, exact source split/plant/rollout gates; no archived four-module pretraining or passed old PR22 qualification."),
    _definition("sprite-dynamics", "pr153", "Fully observed sprite state dreams and exact render", "Given all six state coordinates, predict the next state and free-run 5/20/50-step ID and ceiling-bounce OOD dreams", 8000,
        thresholds={"one_step_mse_max": .0025, "dream_position_rmse_max": .05, "dream_state_rmse_max": .10,
                    "dream_render_rmse_max": .20, "out_of_box_fraction_max": .01}, sampling="Pinned PR153 dynamics; independent ID starts and disjoint upward/fast ceiling-bounce OOD starts; 80-step episodes",
        scope="Fully observed six-coordinate learned world model plus exact source renderer. Narrower than an image-only latent world model; no oracle next-state input to the predictor."),
    _definition("previous-command-action", "source-family-13", "Action depends on previous command", "Recover the paired tanh(-2.2*previous) action, rather than a state/action marginal", 400,
        thresholds={"relative_mse_max": .18}, sampling="Independent state and previous command U[-1,1]; exact action=tanh(-2.2*previous)",
        scope="New direct two-input action network and current public paired-error GAN. This isolates command dependence; it does not replay the historical 250-step autoencoder pretrain or its 400-step latent-joint game."),
    *[_definition("mask-" + mechanism.replace("_", "-"), "pr196-" + mechanism, "Oracle-complete-label imputer: " + mechanism,
        "Using oracle complete training labels, preserve observed coordinates and recover ambiguous conditional posteriors; this does not test learning from incomplete data alone", 7000,
        kind="conditional_distribution", thresholds={"samples_per_row_min":4096,"observed_max_error_max": 1e-6, "posterior_mode_tv_max": .12,
            "missing_mean_rms_sigma_max": .15, "missing_variance_ratio": [.70, 1.30], "orthogonal_rms_ratio": [.70, 1.30]},
        sampling=f"Pinned PR196 lift/observed-standardization with 20000 training/10000 test rows; {mechanism}; exact conditional Bayes samples; 16 fixed ambiguous single-sensor queries (a positive-probability mask, explicitly oversampled at evaluation)",
        scope="Oracle-complete-data supervised conditional GAN reform, explicitly different information access from original incomplete-data MisGAN. Joint recovery is not inferred from imputation, and no archived MisGAN arm is qualified.")
        for mechanism in ("block", "mcar_p20", "mcar_p50", "mcar_p80")],
]
for _row in _CASES:
    if _row["id"].startswith(("api-mask-","api-posterior-")): _row["eval_samples"]=4096
    if _row["id"].startswith("api-sparse-"):
        _row["thresholds"]={k:v for k,v in _row["thresholds"].items() if k not in ("full_original_update_budget","terminal_observations")}
    _fixed_rows={"api-trajectory-edit":12,"api-trajectory-residual":12,"api-unipolar-hold":2,
        "api-unused-token-hold":2,"api-guarded-leftover":4,"api-midscale-identity":4,
        "api-circle-controller":128,"api-sprite-dynamics":64}
    if _row["id"] in _fixed_rows:
        _row["eval_samples"]=_fixed_rows[_row["id"]]
        _row["evaluation_count_policy"]="Fixed exact source panel, not redrawn when a caller supplies n"
CASES = {row["id"]: row for row in _CASES}


def list_cases():
    return deepcopy(_CASES)


def _rng(seed):
    return torch.Generator().manual_seed(int(seed))


def _bound(metrics, name, lo=None, hi=None):
    value = float(metrics[name])
    return [] if math.isfinite(value) and (lo is None or value >= lo) and (hi is None or value <= hi) else [name]


def observation(metrics, failures, views):
    return dict(metrics=metrics, passed=not failures, failed_bounds=sorted(set(failures)), views=views)


def view(title, target, samples, kind="scatter", **extra):
    return dict(kind=kind, title=title, target=target.detach().cpu() if isinstance(target, torch.Tensor) else target,
                samples=samples.detach().cpu() if isinstance(samples, torch.Tensor) else samples, **extra)


def cdf_error(points, reference):
    """Fixed-projection two-sample CDF distance, not a coordinate-mean proxy."""
    dimension = points.shape[1]
    mean, scale = reference.mean(0), reference.std(0).clamp_min(1e-5)
    x, y = (points-mean)/scale, (reference-mean)/scale
    directions = torch.cat((torch.eye(dimension), torch.randn(16, dimension, generator=_rng(5501))))
    directions = directions / directions.norm(dim=1, keepdim=True)
    worst = 0.
    for direction in directions:
        a, b = (x@direction).sort().values, (y@direction).sort().values
        # Check both sides of jumps; ties are allowed and empirical target is explicit.
        knots = torch.cat((a, b)).sort().values
        for right in (False, True):
            distance = (torch.searchsorted(a, knots, right=right)/len(a)
                        - torch.searchsorted(b, knots, right=right)/len(b)).abs().max()
            worst = max(worst, float(distance))
    return worst


class SparseTask:
    paired = False
    def __init__(self, split):
        self.toy = SparseMixedToy(symbol_map="split" if split else "identity", n_symbols=16 if split else 8)
        self.context_dim, self.output_dim = 8, 24+self.toy.n_symbols
    def batch(self, n, rng):
        c = torch.randint(8, (n,), generator=rng)
        x, s, _ = self.toy.sample_given_class(c, rng)
        return F.one_hot(c, 8).float(), torch.cat((x, F.one_hot(s, self.toy.n_symbols).float()), 1)
    def decode(self, raw):
        # A declared hard serving adapter gives exact sparsity; noise can only
        # occur on the three largest learned coordinates. Symbol remains learned.
        x, symbol = raw[:, :24], raw[:, 24:].argmax(1)
        active = x.abs().topk(3, dim=1).indices
        return torch.zeros_like(x).scatter(1, active, x.gather(1, active)), symbol
    def evaluate(self, predict, n, seed):
        c = torch.arange(8).repeat_interleave(n)
        raw = predict(F.one_hot(c, 8).float(), _rng(seed))
        x, symbols = self.decode(raw)
        result = density.sparse_metrics(self.toy, x, symbols, c)
        local = result["local_shape"]
        metrics = {key: result[key] for key in ("sample_count", "modes", "hq", "cond_acc", "sym_acc_mode", "exact_zero_frac", "max_conditional_quality_mass_tv", "max_symbol_tv")}
        metrics.update(mean_error_sigma=local["mean_error_sigma"], covariance_min=min(local["covariance_eigenvalues"]), covariance_max=max(local["covariance_eigenvalues"]), radial_ks=local["radial_ks"], projection_ks=local["max_projection_ks"])
        bounds = [("modes", 64, None), ("hq", .95, None), ("exact_zero_frac", 1, None), ("max_conditional_quality_mass_tv", None, .10), ("max_symbol_tv", None, .10), ("mean_error_sigma", None, .10), ("covariance_min", .85, None), ("covariance_max", None, 1.15), ("radial_ks", None, .075), ("projection_ks", None, .06), ("sample_count", 1024, None)]
        failures = [item for name, lo, hi in bounds for item in _bound(metrics, name, lo, hi)]
        target, target_s, _ = self.toy.sample_given_class(c, _rng(seed+1))
        views = [view("Matched requested-class sparse coordinates", target[:32], x[:32], "image", vmin=-2.2,vmax=2.2,caption="Rows are conditional draws, not uniquely paired targets; exact inactive zeros are part of the served adapter."),
                 view("Symbol distribution conditioned on each class", self.toy.p_symbol_given_class, torch.tensor(result["confusion"]), "image", caption="Both valid split symbols and mode-symbol coherence are scored.")]
        return observation(metrics, failures, views)


class PosteriorTask:
    paired = False
    def __init__(self, classes):
        self.toy = GaussianGrid(classes=classes)
        self.context_dim, self.output_dim = classes+3, 2
        self.panels = [(c, x, a) for c in range(classes) for x, a in [((0., 0.), .5), ((.2, -.3), .5), ((0., 0.), .9), ((.4, -.2), .9)]]
    def condition(self, c, x, a):
        return torch.cat((F.one_hot(c, self.toy.classes).float(), x, torch.full((len(c), 1), float(a))), 1)
    def batch(self, n, rng):
        c = torch.randint(self.toy.classes, (n,), generator=rng)
        clean = self.toy.sample(c, rng)
        a = (.5, .9)[int(torch.randint(2, (), generator=rng))]
        x = math.sqrt(a)*clean+math.sqrt(1-a)*torch.randn(n, 2, generator=rng)
        return self.condition(c, x, a), self.toy.oracle_clean(x, c, a, rng)
    def evaluate(self, predict, n, seed):
        results, views = [], []
        for i, (cls, xy, abar) in enumerate(self.panels):
            c, x = torch.full((n,), cls), torch.tensor(xy).expand(n, 2)
            points = predict(self.condition(c, x, abar), _rng(seed+i))
            result = density.posterior_metrics(self.toy, points, x[0], cls, abar)
            results.append(result)
            if i in (0, len(self.panels)-1):
                target = self.toy.oracle_clean(x, c, abar, _rng(seed+100+i))
                views.append(view(f"Posterior at class={cls}, observation={xy}, alpha_bar={abar}", target, points,
                    caption="Exact clean conditional draws and learned samples; noise input is not the target."))
        metrics = dict(panel_count=len(results), sample_count_per_panel=n,
            worst_quality_mass_tv=max(r["conditional_quality_mass_tv"] for r in results),
            min_hq=min(r["hq"] for r in results), worst_mean_sigma=max(r["local_shape"]["mean_error_sigma"] for r in results),
            min_covariance=min(min(r["local_shape"]["covariance_eigenvalues"]) for r in results),
            max_covariance=max(max(r["local_shape"]["covariance_eigenvalues"]) for r in results),
            worst_radial_ks=max(r["local_shape"]["radial_ks"] for r in results),
            worst_projection_ks=max(r["local_shape"]["max_projection_ks"] for r in results))
        failures = [item for name, lo, hi in [("sample_count_per_panel",1024,None),("worst_quality_mass_tv",None,.10),("min_hq",.95,None),("worst_mean_sigma",None,.10),("min_covariance",.85,None),("max_covariance",None,1.15),("worst_radial_ks",None,.075),("worst_projection_ks",None,.06)] for item in _bound(metrics,name,lo,hi)]
        return observation(metrics, failures, views)


class RouteTask:
    paired = False
    def __init__(self, mode, transition=False):
        self.transition = transition
        self.toy = Transitions(geometry_mode=mode) if transition else Routes(geometry_mode=mode)
        self.context_dim, self.output_dim = (6, 6) if transition else (5, 128)
    def condition(self, c, geom, tick=None):
        parts = [F.one_hot(c,2).float(),geom]
        if self.transition: parts.append((tick.float()/63)[:,None])
        return torch.cat(parts,1)
    def batch(self, n, rng):
        if self.transition:
            c,g,t,x=self.toy.batch(n,rng)
            return self.condition(c,g,t),x
        c,g,x=self.toy.batch(n,rng)
        return self.condition(c,g),x.flatten(1)
    def evaluate(self,predict,n,seed):
        cc,gg=self.toy.contexts("test")
        records,views=[],[]
        for i,(c0,g0) in enumerate(zip(cc,gg)):
            c,g=c0.expand(n),g0.expand(n,3)
            if self.transition:
                for tick0 in (0,8,31,54,62):
                    tick=torch.full((n,),tick0)
                    points=predict(self.condition(c,g,tick),_rng(seed+i*64+tick0))
                    target=self.toy.sample(c,g,tick,_rng(seed+1000+i*64+tick0))
                    records.append(dict(consistency=float((points[:,:2]+points[:,2:4]-points[:,4:]).square().mean().sqrt()),cdf=cdf_error(points,target)))
                    if i==0 and tick0 in (8,31,54):
                        views.append(view(f"Heldout joint transition at tick {tick0}", target[:,:2],points[:,:2],caption="State cloud; adjacent view shows displacement and successor, all sampled from the same 6D output."))
                        views.append(view(f"Action and successor, tick {tick0}",target[:,2:],points[:,2:],"image",vmin=-1.2,vmax=1.2,caption="Rows align state/action/successor; algebra and joint conditional CDF are both scored."))
            else:
                points=predict(self.condition(c,g),_rng(seed+i)).reshape(n,2,64)
                target,_=self.toy.sample(c,g,_rng(seed+1000+i))
                d=self.toy.diagnose(points,g)
                coeff=d["coeff"]
                masses=torch.bincount(d["route"][d["valid"]],minlength=2).float()/n
                p=self.toy.upper_probability[int(c0)]
                tv=float(((masses-torch.tensor([1-p,p])).abs().sum()+1-masses.sum())/2)
                ks=max(density.ks(coeff[:,k].numpy(),lambda a:((a+1)/2).clip(0,1)) for k in range(3))
                records.append(dict(valid=float(d["valid"].float().mean()),tv=tv,
                    support=float((coeff.abs()<=1.0001).all(1).float().mean()),mean=float(coeff.mean(0).abs().max()),
                    var_min=float(coeff.var(0,unbiased=False).min()),var_max=float(coeff.var(0,unbiased=False).max()),ks=ks))
                if i in (0,1,6,7):
                    views.append(view(f"Unseen geometry {i//2}, class {int(c0)}: 64-point routes", target[:12].transpose(1,2),points[:12].transpose(1,2),"line",
                        caption="Both routes, class weights, continuous segment clearance and uniform coefficient variation are scored; obstacle center/radius="+str(g0.tolist()),xlim=[-1.1,1.1],ylim=[-.9,1.0]))
        if self.transition:
            metrics=dict(panel_count=len(records),sample_count_per_panel=n,joint_consistency_rms=max(r["consistency"] for r in records),max_projection_cdf_error=max(r["cdf"] for r in records))
            limits=[("sample_count_per_panel",256,None),("joint_consistency_rms",None,.002),("max_projection_cdf_error",None,.12)]
        else:
            metrics=dict(panel_count=len(records),sample_count_per_panel=n,min_valid_fraction=min(r["valid"] for r in records),max_quality_route_mass_tv=max(r["tv"] for r in records),
                min_coefficient_support=min(r["support"] for r in records),max_coefficient_mean=max(r["mean"] for r in records),min_coefficient_variance=min(r["var_min"] for r in records),max_coefficient_variance=max(r["var_max"] for r in records),max_coefficient_ks=max(r["ks"] for r in records))
            limits=[("sample_count_per_panel",256,None),("min_valid_fraction",.95,None),("max_quality_route_mass_tv",None,.10),("min_coefficient_support",.95,None),("max_coefficient_mean",None,.10),("min_coefficient_variance",.22,None),("max_coefficient_variance",None,.45),("max_coefficient_ks",None,.10)]
        return observation(metrics,[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)],views)


class MaskTask:
    paired = False
    def __init__(self,mechanism):
        self.problem=misgan.Problem(mechanism,20000,10000,0,"cpu")
        self.context_dim,self.output_dim=16,8
        x,m=self.problem.x_test*self.problem.m_test,self.problem.m_test
        probs,_=misgan.bayes_posterior(self.problem,x,m)
        ids=((m.sum(1)<=1)&(probs.max(1).values<.95)).nonzero().flatten()[:16]
        if len(ids)<16:
            # Under p20 these masks are very rare. Keep the original training
            # mechanism, but explicitly condition the evaluation query on the
            # supported one-sensor pattern; never silently change training masks.
            m=torch.zeros_like(self.problem.m_test);m[:,7]=1
            x=self.problem.x_test*m
            probs,_=misgan.bayes_posterior(self.problem,x,m)
            ids=(probs.max(1).values<.95).nonzero().flatten()[:16]
        if len(ids)<16: raise ValueError("required 16 ambiguous posterior queries absent")
        self.ids=ids;self.eval_mask=m[ids].clone()
    def batch(self,n,rng):
        p=self.problem
        x,m,full=p.train_batch(n,rng)
        # Explicit oracle-full access: this is a supervised imputer reform, not MisGAN.
        return torch.cat((x,m),1),full
    def adapt(self,raw,context):
        x,m=context[:,:8],context[:,8:]
        return x*m+(1-m)*raw
    def evaluate(self,predict,n,seed):
        p,ids=self.problem,self.ids
        m=self.eval_mask;x=p.x_test[ids]*m
        probs,target=misgan.bayes_posterior(p,x,m,draws=n,generator=_rng(seed+1))
        # Same test rows and same fixed missingness; never resample the mask between draws.
        context=torch.cat((x,m),1).repeat(n,1)
        points=predict(context,_rng(seed)).reshape(n,len(ids),8)
        raw=p.raw(points); reference=p.raw(target)
        mu=misgan.grid_centers(dtype=torch.float64)
        projected=raw@p.A
        assigned=torch.cdist(projected.flatten(0,1),mu).argmin(1).reshape(n,len(ids))
        tv=[]
        for j in range(len(ids)):
            mass=torch.bincount(assigned[:,j],minlength=100).float()/n
            tv.append(float((mass-probs[j]).abs().sum()/2))
        missing=(1-m).bool()
        target_var=reference.var(0,unbiased=False)
        # Use exact finite Bayes reference variability for each missing coordinate;
        # observed coordinates are checked separately and never enter the ratio.
        variance_ratio=(raw.var(0,unbiased=False)/target_var.clamp_min(1e-12))[missing]
        mean_error=((raw.mean(0)-reference.mean(0))/target_var.clamp_min(1e-12).sqrt())[missing]
        ortho=raw-raw@p.A@p.A.T
        target_ortho=reference-reference@p.A@p.A.T
        ortho_ratio=float(ortho.square().mean().sqrt()/target_ortho.square().mean().sqrt())
        metrics=dict(ambiguous_rows=len(ids),samples_per_row=n,observed_max_error=float(((points-x)*m).abs().max()),
            max_posterior_mode_tv=max(tv),missing_mean_rms_sigma=float(mean_error.square().mean().sqrt()),
            min_missing_variance_ratio=float(variance_ratio.min()),max_missing_variance_ratio=float(variance_ratio.max()),orthogonal_rms_ratio=ortho_ratio)
        limits=[("ambiguous_rows",16,None),("samples_per_row",4096,None),("observed_max_error",None,1e-6),("max_posterior_mode_tv",None,.12),("missing_mean_rms_sigma",None,.15),("min_missing_variance_ratio",.70,None),("max_missing_variance_ratio",None,1.30),("orthogonal_rms_ratio",.70,1.30)]
        failures=[item for name,lo,hi in limits for item in _bound(metrics,name,lo,hi)]
        views=[view("Ambiguous row 0: conditional 2D posterior",p.to2d(target[:,0]),p.to2d(points[:,0]),caption=f"Oracle complete-label training baseline; heldout row {int(ids[0])}, mask {m[0].tolist()}. This does not test original incomplete-data-only MisGAN learning. All eight coordinates are scored."),
               view("Missing-coordinate draws across fixed rows",target[:16].reshape(-1,8),points[:16].reshape(-1,8),"image",vmin=-2.2,vmax=2.2,caption="Oracle complete-label baseline; observed entries must stay fixed, and posterior variance plus orthogonal lift noise must remain. Original incomplete-data-only learning remains unverified.")]
        return observation(metrics,failures,views)


class PairedTask:
    paired=True
    def __init__(self,name):
        self.name=name
        if name.startswith("transport"):
            self.context_dim=self.output_dim=2
        elif name.startswith("trajectory"):
            from benchmarks.locked_shared.trajectory import trajectories
            self.slow,self.fast=trajectories()
            self.context_dim=self.output_dim=16
        elif name=="unipolar-hold": self.context_dim,self.output_dim=1,4
        elif name=="unused-token-hold": self.context_dim,self.output_dim=1,4
        elif name in ("guarded-leftover","midscale-identity"):
            self.context_dim,self.output_dim=1,4
            from benchmarks.locked_shared.hosts.cover_leftover import LeftoverField,teacher_poles
            if name=="guarded-leftover":
                plus,minus,neutral=teacher_poles(LeftoverField(),"faithful_guard_e")
                self.neutral,self.direction=neutral,(plus-minus)/2
            else:
                from benchmarks.locked_shared.hosts.mid_scale_identity import smile_teacher
                teacher=smile_teacher(); self.neutral,self.direction=teacher.identity,teacher.concept
        elif name=="circle-controller": self.context_dim,self.output_dim=6,2
        elif name=="sprite-dynamics": self.context_dim=self.output_dim=6
        elif name=="previous-command-action": self.context_dim,self.output_dim=2,1
        else: raise ValueError(name)
    def targets(self,c):
        name=self.name
        if name=="transport-affine2": return c@c.new_tensor([[.8,-.6],[.6,.8]])+c.new_tensor([.2,-.3])
        if name=="transport-swirl2":
            angle=1.7*(c/math.sqrt(3)).square().sum(1)
            x,y=c.unbind(1); return torch.stack((x*angle.cos()-y*angle.sin(),x*angle.sin()+y*angle.cos()),1)
        if name=="unipolar-hold": return c*c.new_tensor([[1.,0.,0.,0.]])
        if name=="unused-token-hold": return c.new_tensor([[1.,0.,0.,0.]])+c*c.new_tensor([[0.,0.,0.,1.]])
        if name in ("guarded-leftover","midscale-identity"): return self.neutral+c*self.direction
        if name=="circle-controller": return circle.expert_transition(c[:,:2],c[:,2:4],c[:,4],c[:,5])[0]
        if name=="sprite-dynamics": return sprite.step(c)
        if name=="previous-command-action": return (-2.2*c[:,1:]).tanh()
        raise ValueError("trajectory targets have explicit pair indices")
    def batch(self,n,rng):
        name=self.name
        if name.startswith("transport"): c=(torch.rand(n,2,generator=rng)*2-1)*math.sqrt(3)
        elif name=="previous-command-action": c=torch.rand(n,2,generator=rng)*2-1
        elif name.startswith("trajectory"):
            ids=torch.randint(12,(n,),generator=rng); return self.slow[ids],self.fast[ids]
        elif name in ("unipolar-hold","unused-token-hold"): c=torch.randint(2,(n,1),generator=rng).float()
        elif name in ("guarded-leftover","midscale-identity"):
            scales=torch.tensor([-1.,0.,1.] if name=="guarded-leftover" else [-1.,0.,.5,1.])
            c=scales[torch.randint(len(scales),(n,1),generator=rng)]
        elif name=="circle-controller":
            b=circle.sample_rows(n,rng); c=torch.cat((b.position,b.context()),1)
        elif name=="sprite-dynamics":
            # ID episodes only. Training draws independent episode/time rows.
            starts=sprite.sample_starts(n,(.2,.6),(0.,1.),(0.,2*math.pi),np.random.default_rng(int(torch.randint(2**31,(),generator=rng))))
            states=torch.tensor(starts,dtype=torch.float32)
            ticks=torch.randint(80,(n,),generator=rng)
            for t in range(int(ticks.max())):
                states=torch.where((ticks>t)[:,None],sprite.step(states),states)
            c=states
        else: raise ValueError(name)
        return c,self.targets(c)
    def evaluate(self,predict,n,seed):
        name=self.name
        if name=="circle-controller": return circle_observation(predict)
        if name=="sprite-dynamics": return sprite_observation(predict,seed)
        c,t=self.batch(max(n,12),_rng(seed))
        if name.startswith("trajectory"): c,t=self.slow,self.fast
        elif name in ("unipolar-hold","unused-token-hold"): c=torch.tensor([[0.],[1.]]); t=self.targets(c)
        elif name in ("guarded-leftover","midscale-identity"):
            c=torch.tensor([[-1.],[0.],[.5],[1.]]); t=self.targets(c)
        p=predict(c,_rng(seed+1))
        error=p-t; metrics=dict(evaluation_rows=len(c),paired_mse=float(error.square().mean()),max_context_l2=float(error.norm(dim=1).max()))
        failures=[]
        if name.startswith("transport"):
            metrics.update(normalized_mse=float(error.square().mean()/t.var(0,unbiased=False).mean()),p95_l2=float(torch.quantile(error.norm(dim=1),.95)))
            failures=_bound(metrics,"normalized_mse",hi=.01)+_bound(metrics,"p95_l2",hi=.20)
        elif name=="previous-command-action":
            metrics["relative_mse"]=float(error.square().mean()/t.square().mean())
            failures=_bound(metrics,"relative_mse",hi=.18)
        elif name.startswith("trajectory"):
            failures=_bound(metrics,"paired_mse",hi=.02)
            metrics["wrong_pair_margin"]=float((p-t.roll(6,0)).square().mean()-error.square().mean())
            if name=="trajectory-residual": failures+=_bound(metrics,"wrong_pair_margin",lo=.02)
        elif name=="unipolar-hold":
            norm=float(p[1].norm()); cosine=float(F.cosine_similarity(p[1:2],t[1:2]))
            metrics.update(cover=max(0.,cosine)*max(0.,1-abs(norm-1)),off_axis=float(p[1,1:].square().sum()/(p[1].square().sum()+1e-12)),neutral_hold=1-min(1.,float(p[0].norm())))
            failures=_bound(metrics,"cover",lo=.85)+_bound(metrics,"off_axis",hi=.05)+_bound(metrics,"neutral_hold",lo=.85)
        elif name=="unused-token-hold":
            delta=p[1]-t[0]; metrics.update(concept_move=1-min(1.,float((delta[2:]-torch.tensor([0.,1.])).norm())),unused_hold=1-min(1.,float(delta[:2].norm())))
            failures=_bound(metrics,"concept_move",lo=.85)+_bound(metrics,"unused_hold",lo=.85)
        else:
            baseline=(t-self.neutral).square().mean()
            metrics["relative_mse"]=float(error.square().mean()/baseline)
            if name=="guarded-leftover":
                metrics["pole_relative_l2"]=float((error[[0,3]].norm(dim=1)/t[[0,3]].norm(dim=1)).max())
                metrics["leak_ratio"]=float((p[3]-self.neutral)[2].abs()/(p[3]-self.neutral)[0].abs().clamp_min(1e-12))
                failures=_bound(metrics,"pole_relative_l2",hi=.10)+_bound(metrics,"leak_ratio",hi=.10)
            else: failures=_bound(metrics,"relative_mse",hi=.10)+_bound(metrics,"max_context_l2",hi=.15)
        if name.startswith("trajectory"):
            views=[view("Paired fast trajectories with starting identity retained",t.reshape(12,8,2),p.reshape(12,8,2),"line",caption="Each learned path is compared with its own phase/radius target, not nearest target.")]
        elif name=="previous-command-action":
            order=c[:,1].argsort()
            views=[view("Previous command to action, with independent state",torch.cat((c[order,1:2],t[order]),1),torch.cat((c[order,1:2],p[order]),1),"line",caption="Horizontal coordinate is the actual previous command; the independent state cannot replace this condition.")]
        elif name.startswith("transport"):
            views=[view("Heldout source-to-target displacement",torch.stack((c[:16],t[:16]),1),torch.stack((c[:16],p[:16]),1),"line",caption="Each path starts at the same heldout source. A correct marginal with permuted correspondence fails.")]
        else:
            views=[view("Matched heldout inputs: target and learned output",t,p,"scatter" if self.output_dim==2 else "image",vmin=-1.5,vmax=1.5,caption="Target/output rows share the same heldout input; neutral and intermediate cases are included.")]
        return observation(metrics,failures,views)


def circle_observation(predict):
    metrics,views={"episodes_per_panel":128},[]
    failures=[]
    for kind,horizon in (("main",1024),("recovery",64)):
        b=circle.evaluation_panel("test",kind)
        position=b.position.clone(); exact=position.clone()
        ids=torch.cat(((b.angular_step>0).nonzero().flatten()[:2],(b.angular_step<0).nonzero().flatten()[:2]))
        paths,targets=[position[ids].clone()],[exact[ids].clone()]
        radial,speed,direction,angular=[],[],[],[]
        completed=0;nonfinite=0
        for _ in range(horizon):
            context=torch.cat((position,b.context()),1)
            action=predict(context,_rng(EVAL_SEED))
            nxt=position+action
            nonfinite=int((~torch.isfinite(nxt).all(1)).sum())
            if nonfinite: break
            # Evaluate large but finite rollout errors in float64. A genuine
            # nonfinite learned successor stops the observed prefix; it never
            # gets clipped, reset to the oracle, or counted as completed.
            oldq, newq=(position-b.center).double(),(nxt-b.center).double()
            angle=torch.atan2(oldq[:,0]*newq[:,1]-oldq[:,1]*newq[:,0],(oldq*newq).sum(1))
            radial.append((newq.norm(dim=1)/b.radius-1).square())
            speed.append((angle-b.angular_step).abs()); direction.append((angle*b.angular_step>0).float()); angular.append(angle)
            position=nxt
            completed+=1
            exact=circle.expert_transition(exact,b.center,b.radius,b.angular_step)[1]
            paths.append(position[ids].clone());targets.append(exact[ids].clone())
        # Recovery is scored after its fixed64 window, not while correcting the initial offset.
        metrics[kind+"_completed_rollout_steps"]=completed
        metrics[kind+"_nonfinite_episodes"]=nonfinite
        metrics[kind+"_radial_rmse"]=float((radial[-1] if kind=="recovery" else torch.stack(radial).mean(0)).mean().sqrt()) if radial else 1e6
        failures+=_bound(metrics,kind+"_completed_rollout_steps",lo=horizon)+_bound(metrics,kind+"_nonfinite_episodes",hi=0)
        if kind=="main":
            metrics.update(signed_speed_error=float(torch.stack(speed).mean()) if speed else 1e6,direction_agreement=float(torch.stack(direction).mean()) if direction else 0.,min_turns=float(torch.stack(angular).sum(0).abs().min()/(2*math.pi)) if angular else 0.)
            failures+=_bound(metrics,"signed_speed_error",hi=.03)+_bound(metrics,"direction_agreement",lo=.95)+_bound(metrics,"min_turns",lo=1.)
        failures+=_bound(metrics,kind+"_radial_rmse",hi=.10)
        views.append(view(f"True closed-loop {kind}, requested {horizon} transitions",torch.stack(targets,1),torch.stack(paths,1),"line",caption=f"Observed {completed}/{horizon} finite transitions; nonfinite successors={nonfinite}. No teacher successor is fed back, reset or clipped. Both signed directions are present; an incomplete evaluation cannot pass."))
    return observation(metrics,failures,views)


def sprite_observation(predict,seed):
    metrics,views,failures={"episodes_per_split":64},[],[]
    for ood in (False,True):
        name="ood" if ood else "id"
        starts=sprite.sample_starts(64,(.75,.9) if ood else (.2,.6),(1.2,1.6) if ood else (0.,1.),(math.pi/6,5*math.pi/6) if ood else (0.,2*math.pi),np.random.default_rng(seed+(3000 if ood else 2000)))
        state=torch.tensor(starts,dtype=torch.float32); exact=state.clone()
        metrics[name+"_one_step_mse"]=float((predict(state,_rng(seed))-sprite.step(state)).square().mean())
        failures+=_bound(metrics,name+"_one_step_mse",hi=.0025)
        positions,truth=[state[:4,:2].clone()],[exact[:4,:2].clone()]
        for step in range(1,51):
            state=predict(state,_rng(seed));exact=sprite.step(exact)
            positions.append(state[:4,:2].clone());truth.append(exact[:4,:2].clone())
            if step in (5,20,50):
                prefix=f"{name}_{step}"
                metrics[prefix+"_position_rmse"]=float((state[:,:2]-exact[:,:2]).square().mean().sqrt())
                metrics[prefix+"_state_rmse"]=float((state-exact).square().mean().sqrt())
                metrics[prefix+"_render_rmse"]=float((sprite.render(state)-sprite.render(exact)).square().mean().sqrt())
                metrics[prefix+"_out_of_box"]=float(((state[:,:2]<.1)|(state[:,:2]>.9)).any(1).float().mean())
                for suffix,hi in (("position_rmse",.05),("state_rmse",.10),("render_rmse",.20),("out_of_box",.01)):
                    failures+=_bound(metrics,prefix+"_"+suffix,hi=hi)
                if step==20:
                    views.append(view(f"{name} 20-step dreamed frames",sprite.render(exact[:4]),sprite.render(state[:4]),"image",caption="Fully observed six-state learned model and exact renderer, narrower than the original image-only latent world model. Actual predictions; no interpolation or oracle next-state input."))
        views.append(view(f"{name} 50-step free-running positions",torch.stack(truth,1),torch.stack(positions,1),"line",caption="OOD starts rise into ceiling bounces absent from the training episode band."))
    return observation(metrics,failures,views)


def _task(name):
    if name.startswith("sparse"): return SparseTask(name.endswith("split"))
    if name.startswith("posterior"): return PosteriorTask(int(name.split("-")[1].split("class")[0]))
    if name.startswith("routes"): return RouteTask(name.split("-")[1])
    if name.startswith("transitions"): return RouteTask(name.split("-")[1],True)
    if name.startswith("mask"): return MaskTask(name.removeprefix("mask-").replace("-","_"))
    return PairedTask(name)


class ConditionalGenerator(nn.Module):
    def __init__(self,task,z_dim):
        super().__init__();self.task_name=task.name if isinstance(task,PairedTask) else None
        self.paired=task.paired
        self.net=nn.Sequential(nn.Linear(task.context_dim+(0 if task.paired else z_dim),64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,task.output_dim))
    def forward(self,context,z=None):
        raw=self.net(context if self.paired else torch.cat((context,z),1))
        return raw+context if self.task_name=="trajectory-residual" else raw


class ConditionalCritic(nn.Module):
    def __init__(self,dim,context_dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim+context_dim,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class TokenHoldGenerator(nn.Module):
    """Retain the source's shared-vector coupling and independent slot correction."""
    def __init__(self):
        super().__init__()
        self.shared=nn.Parameter(torch.zeros(2));self.slot=nn.Parameter(torch.zeros(2,2))
        self.register_buffer("neutral",torch.tensor([[1.,0.],[0.,0.]]))
    def forward(self,context,z=None):
        return (self.neutral+context[:,:,None]*(self.shared+self.slot)[None]).flatten(1)


init.register(TokenHoldGenerator,{"shared":init.KEEP,"slot":init.KEEP})


class ConditionalFixture:
    api_components=("Recipe.make_prior","Recipe.make_optimizers","Recipe.make_loss","Recipe.make_critic_penalty","scale_learning_rates")
    def __init__(self,case,task,seed,recipe_name,max_steps):
        if recipe_name not in ("ka2","k3p","mog"):
            raise ValueError("These caller-owned conditional samplers require the declared KA2/MoG adaptation; independent Atlas row controls are not bound. Use --recipe auto or ka2.")
        self.case,self.task,self.seed=case,task,int(seed);self.completed_steps=0
        self.limit=case["default_steps"] if max_steps is None else max_steps
        if type(self.limit) is not int or not 0<self.limit<=case["default_steps"]: raise ValueError("max_steps must be in the declared budget")
        self.recipe=get_recipe("k3p" if recipe_name=="k3p" else "ka2",
            batch_size=case["batch_size"],total_steps=case["default_steps"],num_particles=128,z_dim=8,
            prior_kind="mog",sigma_rel=.25,standardize=False,input_noise_std=0.,output_noise_std=0.,lr=.002,
            particle_birth_death=False,row_evidence_gate=False,serve_average=0.).replace(name="conditional_"+recipe_name)
        with construction_rng(seed, "cpu"):
            self.G=TokenHoldGenerator() if isinstance(task,PairedTask) and task.name=="unused-token-hold" else ConditionalGenerator(task,8)
            self.D=ConditionalCritic(task.output_dim,task.context_dim)
            self.prior=None if task.paired else self.recipe.make_prior()
        init.deterministic_orthogonal_(self.G,seed=seed+1);init.deterministic_orthogonal_(self.D,seed=seed+2)
        if self.prior is not None: init.deterministic_orthogonal_(self.prior,seed=seed+3)
        if self.prior is None: self.api_components=tuple(name for name in self.api_components if name!="Recipe.make_prior")
        self.opt_g,self.opt_d=self.recipe.make_optimizers(self.G,self.D,self.prior,ema_critic=deepcopy(self.D),foreach=False)
        self.base_rates=[[group["lr"] for group in opt.param_groups] for opt in (self.opt_g,self.opt_d)]
        self.loss=self.recipe.make_loss();self.penalty=self.recipe.make_critic_penalty(self.opt_d,collect_stats=True)
        self.data_rng,self.latent_rng,self.noise_rng=_rng(seed+10),_rng(seed+11),_rng(seed+12)
        self.last_losses={}
        self.initialization="public deterministic orthogonal; no target/oracle weights"
    def _generate(self,c,rng):
        z=None if self.prior is None else self.prior.sample(len(c),generator=rng)[0]
        raw=self.G(c,z)
        return self.task.adapt(raw,c) if isinstance(self.task,MaskTask) else raw
    def step(self):
        if self.completed_steps>=self.limit: raise RuntimeError("declared execution cap reached")
        scale_learning_rates(self.completed_steps,self.recipe,[self.opt_g,self.opt_d],self.base_rates,self.prior)
        c,t=self.task.batch(self.recipe.batch_size,self.data_rng)
        paired=self.task.paired
        def coordinates(pred,epsilon):
            real=epsilon if paired else t
            fake=epsilon+pred-t if paired else pred
            return torch.cat((real,c),1),torch.cat((fake,c),1)
        self.G.train();self.D.train()
        with torch.no_grad():
            pred=self._generate(c,self.latent_rng);eps=.10*torch.randn(t.shape,generator=self.noise_rng)
            real,fake=coordinates(pred,eps)
        self.opt_d.zero_grad();dl=self.loss.d_loss(self.D(real),self.D(fake));reg=self.penalty(self.D,real,fake)
        (dl+reg).backward();self.opt_d.step()
        flags=[p.requires_grad for p in self.D.parameters()]
        try:
            self.D.requires_grad_(False)
            pred=self._generate(c,self.latent_rng);eps=.10*torch.randn(t.shape,generator=self.noise_rng)
            real,fake=coordinates(pred,eps)
            with torch.no_grad(): real_logits=self.D(real)
            gl=self.loss.g_loss(self.D(fake),real_logits)
            self.opt_g.zero_grad();gl.backward();self.opt_g.step()
        finally:
            for p,f in zip(self.D.parameters(),flags): p.requires_grad_(f)
        self.completed_steps+=1
        self.last_losses=dict(g_gan=float(gl.detach()),d_gan=float(dl.detach()),critic_penalty=float(reg.detach()))
        if not all(math.isfinite(v) for v in self.last_losses.values()): raise FloatingPointError("nonfinite actual GAN update")
        return dict(step=self.completed_steps,**self.last_losses)
    @torch.no_grad()
    def observe(self,n=1024,seed=EVAL_SEED):
        if type(n) is not int or n<2: raise ValueError("evaluation count must be >=2")
        flags={m:m.training for model in (self.G,self.D) for m in model.modules()}
        try:
            self.G.eval();self.D.eval()
            return self.task.evaluate(self._generate,n,int(seed))
        finally:
            for m,flag in flags.items(): m.training=flag
    def state_dict(self):
        models=(("G",self.G),("D",self.D))+(() if self.prior is None else (("prior",self.prior),))
        task_state={key:value for key,value in vars(self.task).items() if key not in ("toy","problem")}
        if hasattr(self.task,"toy"): task_state["sampler"]=vars(self.task.toy)
        if hasattr(self.task,"problem"): task_state["problem"]=vars(self.task.problem)
        return deepcopy(dict(version=VERSION,case_id=self.case["id"],seed=self.seed,limit=self.limit,
            completed_steps=self.completed_steps,recipe=self.recipe.to_dict(),generator=self.G.state_dict(),critic=self.D.state_dict(),
            prior=None if self.prior is None else self.prior.state_dict(),opt_g=self.opt_g.state_dict(),opt_d=self.opt_d.state_dict(),
            data_rng=self.data_rng.get_state(),latent_rng=self.latent_rng.get_state(),noise_rng=self.noise_rng.get_state(),
            modes={role:{name:m.training for name,m in model.named_modules()} for role,model in models},
            requires_grad={role:{name:p.requires_grad for name,p in model.named_parameters()} for role,model in models},
            gradients={role:{name:None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()} for role,model in models},
            fixed_task=task_state,global_rng=torch.random.get_rng_state(),last_losses=self.last_losses))


def build_case(case_id,*,device="cpu",seed=DEFAULT_SEED,recipe_name="atlas",max_steps=None):
    if case_id not in CASES: raise ValueError(f"unknown conditional API case {case_id!r}")
    if torch.device(device).type!="cpu": raise ValueError("This frozen caller-owned conditional protocol is CPU-only")
    return ConditionalFixture(deepcopy(CASES[case_id]),_task(case_id.removeprefix("api-")),seed,recipe_name,max_steps)
