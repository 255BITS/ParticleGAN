"""BISPF MisGAN with genuinely incomplete training and a neural amortized imputer.

Explicit four-mode architecture/target variant; never qualifies the historical
100-mode MLP recovery or imputer tasks. Complete examples are evaluation-only.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from particlegan import Recipe, UpdatePolicy, init
from particlegan.masked_mixture import NeuralComponentImputer, masked_mixture_nll
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from .reproducibility import construction_rng
from .api_sources.misgan import BLOCK_PATTERNS, sample_masks, pattern_index, pattern_probs

ROOT = Path(__file__).resolve().parents[2]
VERSION = "misgan-four-mode-v1"
DEFAULT_RECIPE = "bispf-optimized-quintic-v1"
CASE_ID = "api-model-misgan-four-mode"
TASK_ID = "model_misgan_four_mode"
RESOURCES = dict(num_particles=64, z_dim=2, batch_size=256)
CENTERS = ((-1., -1.), (-1., 1.), (1., -1.), (1., 1.))
SIGMA = .08
BOUNDS = (("modes", ">=", 4), ("hq", ">=", .90), ("mass_tv", "<=", .10),
          ("off_plane", "<=", .04), ("mask_tv", "<=", .05),
          ("imputer_observed_error", "<=", 1e-6), ("imputer_accuracy", ">=", .94),
          ("ambiguous_hq", ">=", .90), ("ambiguous_mass_tv", "<=", .12),
          ("ambiguous_query_coverage", ">=", .90), ("ambiguous_std_ratio", ">=", .70),
          ("ambiguous_std_ratio", "<=", 1.30))


def _digest(value):
    from experiments.forge.state import state_digest
    return state_digest(value)


def list_cases():
    task = json.loads((ROOT / f"configs/forge/tasks/{TASK_ID}.json").read_text())
    return [dict(id=CASE_ID, legacy_ids=["pr196-block-recovery", "pr196-block-neural-imputer"],
                 title="MisGAN: incomplete four-mode recovery and neural multimodal imputation",
                 goal=task["description"], kind="vector", default_steps=1200,
                 batch_size=256, eval_samples=4096, default_recipe=DEFAULT_RECIPE,
                 evaluation_observations=24, terminal_observations=5, protocol_seed=0,
                 thresholds=[list(x) for x in BOUNDS], forge_thresholds=[list(x) for x in BOUNDS],
                 sampling="Clean live learned MoG64 R2 sigma=.025 -> learned affine 8D G_x; learned four-pattern categorical G_m. G_i neural logits on x*m,m sample a component and independent MoG noise through G_x; no component distances, ppost or oracle at inference.",
                 scope="Explicit four-corner sigma=.08 target with a fixed Hadamard 8x2 lift. Same four block masks and masses as original MisGAN; sensor7 sees x+y, leaving two equally probable branches at zero. Original 100-mode MLP questions remain unchanged.",
                 adaptation="New identifiable target, affine generator, categorical mask support learned from observed masks, and neural categorical amortizer; no complete real data reaches training.",
                 host_definition=deepcopy(task["execution"]["host_definition"]))]


class Problem:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.A = torch.tensor([[1., 1.], [1., -1.]] * 3 + [[1., -1.], [1., 1.]], dtype=torch.float32, device=device) / math.sqrt(8)
        rng = torch.Generator().manual_seed(0)
        centers = torch.tensor(CENTERS)
        def draw(n):
            y = torch.randint(4, (n,), generator=rng)
            z = centers[y] + SIGMA * torch.randn(n, 2, generator=rng)
            return (z @ self.A.cpu().T).to(device), y.to(device)
        full, _ = draw(20000)
        self.m_train = sample_masks("block", 20000, rng, device=device)
        count = self.m_train.sum(0)
        self.mean = (full * self.m_train).sum(0) / count
        self.std = (((full-self.mean).square() * self.m_train).sum(0) / count).sqrt()
        # The complete training pool is discarded here, before training exists.
        self.observed = self.standardize(full) * self.m_train
        raw, self.y_test = draw(4096)
        self.x_test = self.standardize(raw)
        self.m_test = sample_masks("block", 4096, rng, device=device)
        self.centers = centers.to(device)
        self.patterns = torch.zeros(4, 8, device=device)
        for row, (sensors, _) in enumerate(BLOCK_PATTERNS):
            self.patterns[row, list(sensors)] = 1

    def standardize(self, x):
        return (x-self.mean)/self.std

    def to2d(self, x):
        return (x*self.std+self.mean) @ self.A

    def batch(self, count, generator):
        ids = torch.randint(len(self.observed), (count,), generator=generator, device=self.device)
        return self.observed[ids], self.m_train[ids]

    def ambiguous_queries(self):
        # Sixteen held-out noisy sums, all overwhelmingly identify the two
        # mixed-sign modes; each conditional retains equiprobable branches.
        raw = torch.zeros(16, 8, device=self.device)
        raw[:, 7] = torch.linspace(-.03, .03, 16, device=self.device)
        masks = torch.zeros_like(raw); masks[:, 7] = 1
        return self.standardize(raw) * masks, masks


class Generators(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = nn.Linear(2, 8)
        self.mask = nn.Linear(1, 4)
        self.imputer = NeuralComponentImputer(8, 64, 64)

    def forward(self, z):
        return self.data(z)


def critic(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(.2), nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 1))


def task_recipe(case, recipe_name=None, recipe=None):
    injected = recipe is not None
    if recipe is None:
        name = DEFAULT_RECIPE if recipe_name in (None, "auto") else recipe_name
        recipe = Recipe(**json.loads((ROOT / f"configs/forge/ideas/{name}.json").read_text())["recipe_overrides"])
    fixed = dict(**RESOURCES, prior_kind="mog", standardize=False, sigma_rel=0.,
                 model="gan", conditioning="scalar", encoder_mode="none", observation_sigma=.08,
                 reconstruction_weight=1.)
    if injected and any(getattr(recipe, k) != v for k, v in fixed.items()):
        raise ValueError("injected recipe conflicts with frozen MisGAN task resources/objectives")
    if not injected:
        recipe = recipe.replace(**fixed, total_steps=1200)
    if (recipe.effective_critic_formulation != "bispf" or recipe.continuous_policy is not None
            or recipe.output_noise_std or recipe.input_noise_std or recipe.serve_average
            or recipe.kinetic_transport_weight or recipe.kinetic_transport_local_weight):
        raise ValueError("this scoped MisGAN host requires clean scheduled BISPF without other sample objectives")
    return recipe


def score(problem, generated, masks, predictions, ambiguous):
    x2 = problem.to2d(generated)
    distances = torch.cdist(x2, problem.centers)
    nearest, ids = distances.min(1)
    counts = torch.bincount(ids, minlength=4).float()/len(ids)
    quality_counts = torch.bincount(ids[nearest <= 3*SIGMA], minlength=4)
    raw = generated*problem.std+problem.mean
    residual = raw - x2 @ problem.A.T
    empiric = torch.bincount(pattern_index(masks), minlength=256).float()/len(masks)
    imp2 = problem.to2d(predictions)
    pred_ids = torch.cdist(imp2, problem.centers).argmin(-1)
    ax2 = problem.to2d(ambiguous)
    adist, aid = torch.cdist(ax2, problem.centers).min(-1)
    mass1 = (aid == 1).float().mean(0)
    mass2 = (aid == 2).float().mean(0)
    observed_error = ((predictions - problem.x_test) * problem.m_test).abs().max()
    amask = torch.zeros(16, 8, device=generated.device); amask[:, 7] = 1
    ax, _ = problem.ambiguous_queries()
    observed_error = torch.maximum(observed_error, ((ambiguous - ax) * amask).abs().max())
    coordinate = (ax2[..., 0] - ax2[..., 1])/math.sqrt(2)
    metrics = dict(modes=int((quality_counts >= 10).sum()), hq=float((nearest <= 3*SIGMA).float().mean()),
                   mass_tv=float((counts-.25).abs().sum()/2), off_plane=float(residual.square().mean().sqrt()),
                   mask_tv=float((empiric-pattern_probs("block").to(empiric)).abs().sum()/2),
                   imputer_observed_error=float(observed_error),
                   imputer_accuracy=float((pred_ids == problem.y_test).float().mean()),
                   ambiguous_hq=float((adist <= 3*SIGMA).float().mean()),
                   ambiguous_mass_tv=float(((mass1-.5).abs()+(mass2-.5).abs()+1-mass1-mass2).mean()/2),
                   ambiguous_query_coverage=float(((mass1 >= .15) & (mass2 >= .15)).float().mean()),
                   ambiguous_std_ratio=float(coordinate.std(0).mean()/math.sqrt(2+SIGMA**2)))
    failures = [f"{name}={metrics[name]:.6g} {op} {bound} failed" for name, op, bound in BOUNDS
                if not (math.isfinite(metrics[name]) and (metrics[name] >= bound if op == ">=" else metrics[name] <= bound))]
    return dict(metrics=metrics, passed=not failures, failed_bounds=failures,
                views=[dict(kind="scatter", title="Complete generator: target and actual", target=problem.centers.cpu(), samples=x2[:1024].cpu(), xlim=[-1.7,1.7], ylim=[-1.7,1.7]),
                       dict(kind="scatter", title="Held-out single-sensor posterior: two branches", target=problem.centers[[1,2]].cpu(), samples=ax2[:,0].cpu(), xlim=[-1.7,1.7], ylim=[-1.7,1.7], caption="Actual neural conditional logits are sampled; no component distances or ppost at inference.")])


class MisGANFixture:
    api_components = ("particlegan.Recipe", "Recipe.make_prior", "Recipe.make_optimizers", "Recipe.make_critic_penalty",
                      "Recipe.prior_regularization", "particlegan.UpdatePolicy", "particlegan.init",
                      "particlegan.masked_mixture.masked_mixture_nll", "particlegan.masked_mixture.NeuralComponentImputer")

    def __init__(self, case, *, device="cpu", seed=0, recipe_name=None, max_steps=None, recipe=None):
        if seed != 0:
            raise ValueError("this comparison is bound to protocol seed 0")
        self.metadata = deepcopy(case); self.device = torch.device(device)
        self.execution_steps = case["default_steps"] if max_steps is None else max_steps
        self.recipe = task_recipe(case, recipe_name, recipe)
        self.problem = Problem(device)
        with construction_rng(seed, self.device):
            self.G = Generators().to(device)
            self.D = nn.ModuleDict(dict(data=critic(16), mask=critic(8), imputer=critic(16))).to(device)
            self.prior_constructor_rng = torch.Generator(device=self.device).manual_seed(0)
            self.prior = self.recipe.replace(prior_update="learned").make_prior(sigma=.025, init_std=.5, device=device, generator=self.prior_constructor_rng)
            self.constructor_rng = torch.random.get_rng_state().clone()
            self.constructor_cuda_rng = torch.cuda.get_rng_state(device).clone() if self.device.type == "cuda" else None
            for index, model in enumerate((self.G, self.D, self.prior)):
                init.deterministic_orthogonal_(model, seed=index)
        self.opt_g, self.opt_d = self.recipe.make_optimizers(self.G, self.D, self.prior, foreach=False, fused=False)
        self.loss = self.recipe.make_loss()
        self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=True)
        self.regularizer = self.recipe.make_prior_regularizer(prior=self.prior)
        self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior, generator_optimizer=self.opt_g,
                                   critic_optimizer=self.opt_d, penalty=self.penalty, seed=seed)
        for index, name in enumerate(("data_rng", "generator_data_rng", "prior_noise_rng", "mask_rng", "imputer_rng")):
            setattr(self, name, torch.Generator(device=self.device).manual_seed(101+index))
        self.batch_sequence_sha256 = hashlib.sha256(b"misgan-four-mode-batches-v1").hexdigest()

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def _draw(self):
        return self.prior.sample(self.recipe.batch_size, generator=self.policy.latent_generator, noise_generator=self.prior_noise_rng)

    def _mask(self, n, rng):
        logits = self.G.mask(torch.ones(n, 1, device=self.device))
        u = torch.rand(logits.shape, device=self.device, generator=rng).clamp(1e-6,1-1e-6)
        soft = ((logits - torch.log(-torch.log(u)))/.5).softmax(-1)
        hard = F.one_hot(soft.argmax(-1),4).to(soft)
        return ((hard-soft).detach()+soft) @ self.problem.patterns, logits

    def _impute(self, x, m, rng, draws=1):
        ids = self.G.imputer.draw_indices(x, m, draws=draws, generator=rng)
        eps = torch.randn((*ids.shape, 2), generator=rng, device=self.device)
        values = self.G.data(self.prior.z[ids] + self.prior.sigma*eps)
        return x*m + values*(1-m)

    def step(self):
        if self.completed_steps >= self.execution_steps:
            raise RuntimeError("declared update budget reached")
        with torch.autograd.set_multithreading_enabled(False):
            return self._step()

    def _step(self):
        x_d,m_d = self.problem.batch(self.recipe.batch_size,self.data_rng)
        x_g,m_g = self.problem.batch(self.recipe.batch_size,self.generator_data_rng)
        digest = _digest(dict(x_d=x_d,m_d=m_d,x_g=x_g,m_g=m_g))
        self.batch_sequence_sha256 = hashlib.sha256(bytes.fromhex(self.batch_sequence_sha256)+bytes.fromhex(digest)).hexdigest()
        self.policy.begin_step(x_d,game_record=self.penalty.regularizer.record,execution_limit=self.execution_steps)
        self.G.eval(); self.D.train()
        with torch.no_grad():
            z,rows = self._draw(); fake = self.policy.generate(z,rows=rows)
            mf,_ = self._mask(len(x_d),self.mask_rng)
            complete = self._impute(x_d,m_d,self.imputer_rng)[0]
        self.policy.observe_support(z)
        real = (torch.cat((x_d,m_d),1), m_d, torch.cat((fake,m_d),1))
        fakes = (torch.cat((fake*mf,mf),1), mf, torch.cat((complete,m_d),1))
        dl = x_d.new_zeros(())
        for model,r,f in zip(self.D.values(),real,fakes):
            dl = dl + self.loss.d_loss(model(r),model(f)) + self.penalty(model,r,f)
        self.opt_d.zero_grad(set_to_none=True); dl.backward(); self.opt_d.step()
        self.policy.after_critic_step()
        self.D.eval(); self.G.train(); self.D.requires_grad_(False)
        try:
            z,rows = self._draw(); fake = self.policy.generate(z,rows=rows)
            mf,logits = self._mask(len(x_g),self.mask_rng)
            # The categorical imputer's GAN gradient uses a straight-through
            # Gumbel draw, while serving uses its actual categorical law.
            ilogits = self.G.imputer(x_g,m_g)
            u = torch.rand(ilogits.shape,generator=self.imputer_rng,device=self.device).clamp(1e-6,1-1e-6)
            soft = ((ilogits-torch.log(-torch.log(u)))/.5).softmax(-1)
            hard = F.one_hot(soft.argmax(-1),64).to(soft)
            table = self.G.data(self.prior.z)
            imputed = x_g + (1-m_g) * (((hard-soft).detach()+soft) @ table.detach())
            gx = self.loss.g_loss(self.D["data"](torch.cat((fake*mf.detach(),mf.detach()),1)),self.D["data"](torch.cat((x_g,m_g),1)))
            gm = self.loss.g_loss(self.D["mask"](mf),self.D["mask"](m_g))
            gi = self.loss.g_loss(self.D["imputer"](torch.cat((imputed,m_g),1)),self.D["imputer"](torch.cat((fake.detach(),m_g),1)))
            adversarial = gx+gm+gi
            # Draw synthetic labels in every arm, so an auxiliary cannot shift
            # the later prior/noise/data sequence relative to the baseline.
            teacher_z,teacher_ids = self._draw()
            synthetic = self.G.data(teacher_z).detach()
            categorical = self.G.imputer.synthetic_loss(synthetic,m_g,teacher_ids)
            mixture = masked_mixture_nll(x_g,m_g,table,sigma=self.recipe.observation_sigma)
            mask_ids = torch.cdist(m_g,self.problem.patterns).argmin(1)
            mask_nll = F.cross_entropy(logits,mask_ids)
            auxiliary = self.recipe.masked_mixture_weight*(mixture+mask_nll) + self.recipe.imputer_component_weight*categorical
            regularization = self.recipe.prior_regularization(self.prior.z,regularizer=self.regularizer)
            self.policy.observe_sampled_rows(torch.arange(64,device=self.device))
            self.opt_g.zero_grad(set_to_none=True)
            protected = (adversarial,auxiliary) if auxiliary.requires_grad and (self.recipe.masked_mixture_weight or self.recipe.imputer_component_weight) else (adversarial,)
            constraint_geometry_backward(adversarial+auxiliary+regularization,self.opt_g,protected)
            self.policy.after_generator_backward(loss_gan=adversarial.detach(),loss_critic=dl.detach())
            self.opt_g.step(); self.policy.after_generator_step()
        finally:
            self.D.requires_grad_(True)
        self.policy.finish_step()
        return dict(step=self.completed_steps,loss_d=float(dl.detach()),loss_g=float(adversarial.detach()),masked_nll=float(mixture.detach()),synthetic_ce=float(categorical.detach()))

    @torch.no_grad()
    def observe(self,n=4096,seed=10000):
        rng = torch.Generator(device=self.device).manual_seed(seed)
        z,_ = self.prior.sample(n,generator=rng)
        fake = self.G.data(z)
        masks,_ = self._mask(n,rng)
        predictions = self._impute(self.problem.x_test*self.problem.m_test,self.problem.m_test,rng,draws=1)[0]
        x,m = self.problem.ambiguous_queries()
        ambiguous = self._impute(x,m,rng,draws=256)
        return score(self.problem,fake,masks,predictions,ambiguous)

    def state_dict(self):
        streams = {name:getattr(self,name).get_state() for name in ("data_rng","generator_data_rng","prior_noise_rng","mask_rng","imputer_rng","prior_constructor_rng")}
        return deepcopy(dict(version=VERSION,metadata_sha256=_digest(self.metadata),recipe=self.recipe.to_dict(),execution_steps=self.execution_steps,
                             policy=self.policy.state_dict(),prior=self.prior.state_dict(),streams=streams,constructor_rng=self.constructor_rng,
                             constructor_cuda_rng=self.constructor_cuda_rng,batch_sequence_sha256=self.batch_sequence_sha256,
                             **self.recipe.prior_regularizer_checkpoint(self.regularizer)))

    def load_state_dict(self,state):
        for name in ("version","metadata_sha256","recipe","execution_steps"):
            if state[name] != self.state_dict()[name]:
                raise ValueError(f"incompatible MisGAN checkpoint {name}")
        self.policy.load_state_dict(state["policy"]); self.prior.load_state_dict(state["prior"])
        self.recipe.restore_prior_regularizer(self.regularizer,state.get("prior_regularizer"))
        for name,value in state["streams"].items():
            getattr(self,name).set_state(value)
        self.constructor_rng = state["constructor_rng"].clone()
        self.constructor_cuda_rng = state["constructor_cuda_rng"]
        self.batch_sequence_sha256 = state["batch_sequence_sha256"]


def build_case(case_id, **kwargs):
    if case_id != CASE_ID:
        raise ValueError("unknown MisGAN case")
    return MisGANFixture(list_cases()[0],**kwargs)
