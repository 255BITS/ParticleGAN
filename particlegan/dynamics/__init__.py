"""Flag-gated GAN dynamics. Nothing here runs unless a mechanism is installed.

The screen sets ``K3P_DYNAMICS`` and imports the named module from
``sitecustomize``. With the variable unset, training is unchanged.
``sga`` is symplectic gradient adjustment on the D/G/particle game.
"""
