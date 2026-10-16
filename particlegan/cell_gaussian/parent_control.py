"""Unlabelled local moments with log mass deviance replacing squared mass ratio.

This research adapter needs no target law, evaluator, component labels, new
sampling, extra optimizer steps or persistent controller state. Coefficient 1,
fourth-other-neighbor bandwidth times 2, and all three moment orders are fixed.
"""
from copy import deepcopy

def log_mass_deviance_loss(fake, real):
    import torch
    if fake.shape != real.shape or fake.ndim != 2 or len(fake) < 5:
        raise ValueError('Matching flat floating minibatches with at least five rows required')
    x, y = fake, real.detach()
    assert x.device == y.device and x.dtype == y.dtype
    n, d = y.shape
    with torch.no_grad():
        delta_y = y[:, None, :] - y[None, :, :]
        distances = delta_y.square().sum(-1)
        neighbors = distances.clone()
        neighbors.fill_diagonal_(float('inf'))
        scale = (y-y.mean(0)).square().mean().clamp_min(torch.finfo(y.dtype).eps)
        radius2 = 4 * neighbors.kthvalue(4, dim=0).values.clamp_min(torch.finfo(y.dtype).eps * scale)
        radius = radius2.sqrt()
        zy = delta_y / radius[None, :, None]
        ky = torch.exp(-zy.square().sum(-1) / 2)
        py = ky.mean(0)
        my = (ky[:, :, None] * zy).mean(0)
        sy = torch.einsum('na,nai,naj->aij', ky, zy, zy) / n
    zx = (x[:, None, :] - y[None, :, :]) / radius[None, :, None]
    kx = torch.exp(-zx.square().sum(-1) / 2)
    px = kx.mean(0)
    mx = (kx[:, :, None] * zx).mean(0)
    sx = torch.einsum('na,nai,naj->aij', kx, zx, zx) / n
    # Every real anchor includes itself, so py >= 1/n without a tuned floor.
    # Same quadratic curvature at ratio=1; logsumexp avoids empty-kernel underflow.
    log_px = torch.logsumexp(-zx.square().sum(-1) / 2, dim=0) - __import__('math').log(n)
    log_ratio = log_px - py.log()
    mass = 2 * (torch.expm1(log_ratio) - log_ratio).mean()
    first = ((mx-my)/py[:, None]).square().mean()
    second = ((sx-sy)/py[:, None, None]).square().mean()
    return mass + first + second, dict(mass=mass.detach(), first=first.detach(), second=second.detach())

class LogMassDevianceAdapter:
    """Add one loss to native G/prior backward; observe native displacements."""
    def __init__(self, trainer, context):
        import torch
        from experiments.forge.state import state_digest
        assert trainer.device.type == 'cuda'
        self.trainer, self.context, self.digest = trainer, context, state_digest
        self.roles = {'G': list(trainer.G.parameters()), 'prior': list(trainer.prior.parameters())}
        owned = {id(p) for g in trainer.opt_g.param_groups for p in g['params']}
        assert owned == {id(p) for role in self.roles.values() for p in role}
        assert owned.isdisjoint({id(p) for p in trainer.D.parameters()})
        self.records, self.before = [], {}
        self.fake, self.real, self.loss_record = None, None, None
        self.generate = trainer._generate
        self.g_loss = trainer.loss.g_loss
        def generate(*args, **kwargs):
            result = self.generate(*args, **kwargs)
            if torch.is_grad_enabled() and trainer.G.training and result.requires_grad:
                self.fake = result
            return result
        def g_loss(*args, **kwargs):
            native = self.g_loss(*args, **kwargs)
            assert self.fake is not None and self.real is not None
            loss, parts = log_mass_deviance_loss(self.fake, self.real)
            self.loss_record = dict(native=float(native.detach()), local=float(loss.detach()),
                parts={k:float(v) for k,v in parts.items()},
                generator_real_sha256=state_digest(self.real), fake_shape=list(self.fake.shape))
            return native + loss
        trainer._generate, trainer.loss.g_loss = generate, g_loss
        self.handles = [trainer.opt_g.register_step_pre_hook(self.pre), trainer.opt_g.register_step_post_hook(self.post)]
    def bind_generator_real(self, real, generator_real):
        self.fake, self.real, self.loss_record = None, None, None
        if callable(generator_real):
            def capture():
                self.real = generator_real()
                return self.real
            return capture
        self.real = real if generator_real is None else generator_real
        return generator_real
    def pre(self, opt, args, kwargs):
        self.before = {name:[p.detach().clone() for p in params] for name,params in self.roles.items()}
    def post(self, opt, args, kwargs):
        import torch
        state = self.digest(opt.state_dict())
        grads = self.digest([[p.grad for p in role] for role in self.roles.values()])
        rng = self.context.streams.audit()
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state(self.trainer.device).clone()
        measures = {}
        with torch.no_grad():
            for name, params in self.roles.items():
                delta = torch.cat([(p.detach()-old).reshape(-1) for p,old in zip(params,self.before[name])])
                measures[name] = dict(native_l2=float(delta.norm()), applied_l2=float(delta.norm()),
                    parameters_with_grad=sum(p.grad is not None for p in params))
        assert state == self.digest(opt.state_dict())
        assert grads == self.digest([[p.grad for p in role] for role in self.roles.values()])
        assert rng == self.context.streams.audit() and torch.equal(cpu, torch.get_rng_state())
        assert torch.equal(cuda, torch.cuda.get_rng_state(self.trainer.device))
        self.records.append(dict(step=self.trainer.completed_steps+1, roles=measures,
            objective=deepcopy(self.loss_record), coefficient=1., native_optimizer_state_unchanged=True,
            gradients_unchanged=True, RNG_unchanged=True, post_step_parameter_writes=False))
    def close(self):
        for h in self.handles: h.remove()
        self.trainer._generate, self.trainer.loss.g_loss = self.generate, self.g_loss
