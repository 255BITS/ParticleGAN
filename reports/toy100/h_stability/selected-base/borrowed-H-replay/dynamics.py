"""Small update-order and paired-minibatch variants of the pinned H game.
No labels, centers, metric values, new objectives, or changed Adam settings.
"""
import torch
from stability_runner import Updates as HUpdates

POLICIES = {
    'network_half_rate': dict(family='fixed network response, full prior mobility', paired=False, simultaneous=False, g_rate_mult=.5, prior_rate_mult=1.),
    'g_threequarter_rate': dict(family='fixed acquisition/response compromise', paired=False, simultaneous=False, g_rate_mult=.75),
    'g_temporal025': dict(family='bounded G temporal damping', paired=False, simultaneous=False, temporal_cap=.25),
    'g_half_rate': dict(family='fixed G response rate', paired=False, simultaneous=False, g_rate_mult=.5),
    'paired_alt': dict(family='minibatch coupling', paired=True, simultaneous=False),
    'paired_sim': dict(family='paired simultaneous game', paired=True, simultaneous=True),
    'independent_sim': dict(family='simultaneous game', paired=False, simultaneous=True),
}
for policy in POLICIES.values():
    policy.update(d_draws=1, g_draws=1, d_commits=1, g_commits=1,
                  provisional_pairs=0, noise_coupling='independent H observation noise')

class Updates(HUpdates):
    def gradients(self, player, completed, draws=1):
        if not POLICIES[self.variant]['paired']:
            return super().gradients(player, completed, draws)
        assert draws == 1
        if player == 'd':
            self.paired_real = self.sample_real()
            self.paired_latent, _ = self.prior.sample(128, generator=self.stream)
            with self.policy.discriminator():
                fake = self.g(self.paired_latent).detach()
            loss = self.gan.d_loss(self.d(self.paired_real), self.d(fake))
            loss = loss + self.regularizer(self.d, self.paired_real, fake, step=completed+1)
        else:
            fake = self.g(self.paired_latent)
            loss = self.gan.g_loss(self.d(fake), self.d(self.paired_real))
        self.opts['opt_'+player].zero_grad()
        loss.backward()
        self.work[player+'_backwards'] += 1
        return float(loss.detach())

    def state_dict(self):
        return dict(previous_g=getattr(self, 'previous_g', None))

    def correct_generator(self):
        cap = POLICIES[self.variant].get('temporal_cap', 0.)
        if not cap:
            return
        params = self.params('g')
        current = [p.grad.detach().clone() for p in params]
        if getattr(self, 'previous_g', None) is not None:
            delta = [old - new for old, new in zip(self.previous_g, current)]
            norm = torch.stack([x.square().sum() for x in current]).sum().sqrt()
            delta_norm = torch.stack([x.square().sum() for x in delta]).sum().sqrt()
            scale = min(1., float(cap * norm / delta_norm.clamp_min(1e-30)))
            for param, correction in zip(params, delta):
                param.grad.add_(correction, alpha=scale)
            self.work['bounded_gradient_corrections'] = self.work.get('bounded_gradient_corrections', 0) + 1
        self.previous_g = current

    def round(self, variant, completed):
        self.variant = variant
        loss_d = self.gradients('d', completed)
        if not POLICIES[variant]['simultaneous']:
            self.commit('d', completed+1)
            loss_g = self.gradients('g', completed)
        else:
            # G backward also fills D.grad; preserve only D's own objective.
            grad_d = [p.grad.detach().clone() for p in self.params('d')]
            loss_g = self.gradients('g', completed)
            for param, gradient in zip(self.params('d'), grad_d):
                param.grad = gradient
            self.commit('d', completed+1)
        self.correct_generator()
        self.commit('g', completed+1)
        return dict(loss_d=loss_d, loss_g=loss_g)
