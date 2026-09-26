"""Flag-gated GAN dynamics. Nothing here runs unless a mechanism is installed.

The screen sets ``K3P_DYNAMICS`` and imports the named module from
``sitecustomize``. With the variable unset, training is unchanged.
``extragradient`` is simultaneous Extra-Adam (Gidel et al. 2019).
"""
