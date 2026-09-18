"""Unconditional DDGAN adapters for the matched fixed-sigma particle study."""
import torch
from torch import nn
from lib.image_ddgan import ImageGenerator, sample_images
from lib.image_moonshots import PretrainedFeatureDiscriminator
from particlegan import DDGAN


def model_config(width, z_dim, alpha_bar):
    return {'g_width': width, 'd_width': width, 'z_dim': z_dim, 'classes': 1,
            'alpha_bar': alpha_bar, 'd_mode': 'ucd', 'ucd_target': 'time_class',
            'd_norm': 'group', 'd_backbone': 'pretrained_resnet18'}


class ParticleDDGenerator(nn.Module):
    def __init__(self, z_dim, width, alpha_bar):
        super().__init__()
        self.net = ImageGenerator(model_config(width, z_dim, alpha_bar))
        self.schedule = DDGAN(alpha_bar, validate_args=False)

    def forward(self, z, xt, t):
        return self.net(z, torch.zeros_like(t), xt, t)

    @torch.no_grad()
    def sample(self, prior, n, generator):
        labels = torch.zeros(n, device=prior.z.device, dtype=torch.long)
        return sample_images(self.net, prior, self.schedule, labels, generator)


class ParticleDDDiscriminator(nn.Module):
    def __init__(self, width, z_dim, alpha_bar):
        super().__init__()
        self.critic = PretrainedFeatureDiscriminator(model_config(width, z_dim, alpha_bar))

    def requires_grad_(self, requires_grad=True):
        self.critic.requires_grad_(requires_grad)
        return self

    def conditioned(self, xt, t):
        features = self.critic.condition_features(xt)
        c = torch.zeros_like(t)
        return lambda x: self.critic(x, c, xt, t, condition_features=features)[0]
