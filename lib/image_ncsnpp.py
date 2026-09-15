"""Official DDGAN NCSN++ architecture adapted to the particle/class interface."""
from types import SimpleNamespace
from torch import nn
from lib.ddgan_ncsnpp.ncsnpp_generator_adagn import NCSNpp


class NCSNppParticleGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Architecture defaults are the official CIFAR command and parser.
        options = dict(
            num_channels_dae=cfg['g_width'], ch_mult=cfg['ncsnpp_ch_mult'],
            num_res_blocks=cfg['ncsnpp_res_blocks'],
            attn_resolutions=cfg['ncsnpp_attn_resolutions'],
            z_emb_dim=cfg['ncsnpp_z_emb_dim'], n_mlp=cfg['ncsnpp_n_mlp'],
            nz=cfg['z_dim'], image_size=32, num_channels=3,
            dropout=0., resamp_with_conv=True, conditional=True,
            fir=True, fir_kernel=[1, 3, 3, 1], skip_rescale=True,
            resblock_type='biggan', progressive='none',
            progressive_input='residual', progressive_combine='sum',
            embedding_type='positional', fourier_scale=16.,
            centered=True, not_use_tanh=False,
        )
        self.net = NCSNpp(SimpleNamespace(**options))
        self.cls = nn.Embedding(cfg['classes'], cfg['g_width'] * 4)

    def forward(self, z, c, xt, t):
        return self.net(xt, t, z, class_emb=self.cls(c))
