"""Installed-package smoke example; one caller-owned step for AE and VAE."""
import torch
from torch import nn
from particlegan import get_recipe


def main():
    for name in ('ae_gan', 'vae_gan'):
        recipe = get_recipe(name, num_particles=16, z_dim=2)
        prior = recipe.make_prior()
        encoder, decoder, critic = nn.Linear(2, 4), nn.Linear(2, 2), nn.Linear(2, 1)
        opt_g, opt_d = recipe.make_optimizers(decoder, critic, prior, encoder=encoder)
        x = torch.randn(8, 2)
        query, offset = encoder(x).chunk(2, dim=1)
        encoded = recipe.encode(query, prior, offset=offset if recipe.encoder_mode == 'ae' else None)
        # This tests the encoding and optimizer, not a diffusion or GAN loop.
        loss = encoded.reconstruction_loss(decoder(encoded.codes), x)
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
        assert torch.isfinite(loss)
        print(f'{name}: reconstruction step passed', flush=True)


if __name__ == '__main__':
    main()
