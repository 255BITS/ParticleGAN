"""Flag-gated GAN dynamics. Nothing here runs unless a mechanism is installed.

The screen sets ``K3P_DYNAMICS`` and imports the named module from
``sitecustomize``. With the variable unset, training is unchanged.
``ema_g`` averages G and the particles (decay 0.999) and scores that average
beside the live model. ``ema_g_fake`` is the one declared alternative: D's
fakes are that average.
"""
