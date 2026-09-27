"""Historical formulations pinned for reproducing old benchmark reports.

ParticleGAN ships one formulation (see docs/k3p.md). The modules here are
frozen copies of components it no longer ships -- the multi-arm gradient
penalty, the multi-mode GAN loss and the locked_shared demo stamp -- kept
only so existing benchmarks and their reports stay reproducible. Do not use
them in new code.
"""
