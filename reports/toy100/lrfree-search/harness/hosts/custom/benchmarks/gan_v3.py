"""lrfree custom22 IMPORT-ONLY SHIM for benchmarks/gan_v3.py.

``ae_gan_hold`` imports ``gan_v3_recipe`` at module level for its legacy ``make_recipe``;
the custom22 port never calls it (the learner is the candidate package). Calling it raises.
"""


def gan_v3_recipe(*args, **kwargs):
    raise RuntimeError("import-only shim: the legacy GAN v3 recipe is not part of the custom22 learner")
