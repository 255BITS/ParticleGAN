# lrfree custom22 VERBATIM COPY of benchmarks/legacy/__init__.py (PR #155 checkout, last commit e0cdb036b8bb), sha256 83de2dd27bb1fec5ba74b066b49274ba161e4e480037129e3c99e13b04767f04
# Everything below these 3 header lines is byte-identical to that file; harness/custom22.py checks it.
# ------------------------------------------------------------------------------------------------
"""Historical formulations pinned for reproducing old benchmark reports.

ParticleGAN ships one formulation (see docs/k3p.md). The modules here are
frozen copies of components it no longer ships -- the multi-arm gradient
penalty, the multi-mode GAN loss and the locked_shared demo stamp -- kept
only so existing benchmarks and their reports stay reproducible. Do not use
them in new code.
"""
