"""Transactional joint predictor/corrector using the installed Adam optimizers.

Both fields in a stage see the same joint parameter state. A preview never
advances accepted optimizer/controller/EMA/RNG clocks. No evaluator inputs.
"""
from copy import deepcopy
import torch


@torch.no_grad()
def restore(pairs):
    for parameter, value in pairs:
        parameter.copy_(value)


def copies(parameters):
    return [(p, p.detach().clone()) for p in parameters]


class JointStage:
    def __init__(self, trainer, base, correction=False):
        self.trainer, self.base, self.correction = trainer, base, correction
        self.pending = None
        self.predicted_d = None
        self.reference_before = None
        self.predictor_state = None
        self.predicted = None
        self.secant_stats = None

    def before(self, role):
        if role == "d" and self.trainer.recipe.game_update == "secant_resolvent":
            self.reference_before = deepcopy(self.trainer.ema_D.state_dict())
            self.record_before = self.trainer.opt_d.record.state_dict()
        if self.correction:
            if role == 'd':
                self.predicted_d = copies(self.trainer.D.parameters())
            restore(self.base[role])

    def after(self, role):
        if role == 'd':
            self.pending = copies(self.trainer.D.parameters())
            restore(self.predicted_d if self.correction else self.base['d'])
        else:
            restore(self.pending)
            self.pending = None

    def finalize(self):
        # Called only after the trainer restores the critic's original flags.
        if self.correction and self.trainer.recipe.game_update == "secant_resolvent":
            self.resolve()

    def resolve(self):
        pairs = self.base['d'] + self.base['g']
        predicted = self.predicted['d'] + self.predicted['g']
        u = [q - b for (_, q), (_, b) in zip(predicted, pairs)]
        v = [p.detach() - b for p, b in pairs]
        delta, self.secant_stats = secant_resolvent(u, v)
        restore([(p, b + d) for (p, b), d in zip(pairs, delta)])
        # Persist ordinary gradients at the actual state, so extreme preview
        # gradients cannot blind the next base-state Adam normalization.
        for opt, state in zip((self.trainer.opt_g, self.trainer.opt_d), self.predictor_state):
            opt.load_state_dict(deepcopy(state))
        self.trainer.ema_D.load_state_dict(self.predictor_reference)
        record = self.trainer.opt_d.record
        if record.ema_reseeds > self.predictor_record_before["ema_reseeds"]:
            self.trainer.opt_d.anchor.start_()
        if record.ema_updates > self.predictor_record_before["ema_updates"]:
            self.trainer.opt_d.anchor.update_()


@torch.no_grad()
def secant_resolvent(u, v):
    """Backward-Euler response in a fitted dissipative/rotational plane.

    Fit J u = v-u = a*u+r, r orthogonal to u. Project a to <=0;
    on this plane use J=a*I+b*R (R a quarter-turn). Then
    (I-J)^-1 u = ((1-a)*u+r)/((1-a)^2+b^2).
    No norm threshold or tuned clipping coefficient. Exact for that local
    linear game model; it is only a secant approximation for a nonlinear GAN.
    """
    norm2 = sum(x.square().sum() for x in u).clamp_min(torch.finfo(u[0].dtype).tiny)
    w = [y-x for x,y in zip(u,v)]
    a = sum((x*y).sum() for x,y in zip(u,w)) / norm2
    perpendicular = [y-a*x for x,y in zip(u,w)]
    b2 = sum(x.square().sum() for x in perpendicular) / norm2
    radial = 1 + (-a).clamp_min(0)
    denominator = radial.square() + b2
    delta = [(radial*x+y)/denominator for x,y in zip(u,perpendicular)]
    stats = {'radial_secant': float(a), 'rotation_squared': float(b2),
             'denominator': float(denominator),
             'norm_ratio': float(denominator.rsqrt()), 'moment_source': 'base_gradient'}
    return delta, stats


def joint_step(trainer, real, generator_real, collect_stats):
    if callable(generator_real):
        generator_real = generator_real()
    checkpoint = trainer.state_dict()
    base = {'d': copies(trainer.D.parameters()),
            'g': copies(list(trainer.G.parameters()) + list(trainer.prior.parameters()))}
    trainer._game_stage = JointStage(trainer, base)
    try:
        trainer._ordinary_step(real, generator_real=generator_real, collect_stats=False)
        predictor_state = None
        predictor_reference = None
        if trainer.recipe.game_update == "secant_resolvent":
            predictor_state = [deepcopy(o.state_dict()) for o in (trainer.opt_g, trainer.opt_d)]
            predictor_reference = trainer._game_stage.reference_before
            predictor_record_before = trainer._game_stage.record_before
        predicted = {role: copies([p for p, _ in pairs]) for role, pairs in base.items()}
        # Keep preview parameters, restore all other accepted state. Adam's
        # ordinary CPU scalar step metadata remains on CPU.
        for opt, state in zip((trainer.opt_g, trainer.opt_d), checkpoint['optimizers']):
            opt.load_state_dict(deepcopy(state))
        for name in ('ema_G', 'ema_prior'):
            getattr(trainer, name).load_state_dict(checkpoint['models'][name])
        for name in ('G', 'D', 'prior'):
            module = getattr(trainer, name)
            restore([(b, checkpoint['models'][name][key]) for key, b in module.named_buffers()])
        trainer.completed_steps = checkpoint['completed_steps']
        for name, state in checkpoint['streams'].items():
            getattr(trainer, name).set_state(state.cpu())
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        if trainer.device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), trainer.device)
        trainer._game_stage = JointStage(trainer, base, correction=True)
        trainer._game_stage.predictor_state = predictor_state
        trainer._game_stage.predictor_reference = predictor_reference
        trainer._game_stage.predictor_record_before = predictor_record_before
        trainer._game_stage.predicted = predicted
        result = trainer._ordinary_step(real, generator_real=generator_real, collect_stats=collect_stats)
        telemetry = {'policy': trainer.recipe.game_update, 'field_evaluations': 2,
                     'accepted_updates': trainer.completed_steps, 'preview_moments_discarded': trainer.recipe.game_update == 'extragradient'}
        if trainer._game_stage.secant_stats is not None:
            telemetry["secant"] = trainer._game_stage.secant_stats
        for role, pairs in base.items():
            preview = [q - b for (_, q), (_, b) in zip(predicted[role], pairs)]
            actual = [p.detach() - b for p, b in pairs]
            telemetry[role] = {'preview_l2': float(sum(x.square().sum() for x in preview).sqrt()),
                               'applied_l2': float(sum(x.square().sum() for x in actual).sqrt()),
                               'correction_l2': float(sum((x-y).square().sum() for x,y in zip(actual,preview)).sqrt())}
        if trainer.precision is not None:
            before = {id(p): value for p, value in base["g"]}
            activity = trainer.precision.update_activity(trainer.opt_g, before)
            trainer.precision.observe(trainer.D, real, activity)
        trainer.game_stats = telemetry
        return result
    except Exception:
        trainer.load_state_dict(checkpoint)
        raise
    finally:
        trainer._game_stage = None
