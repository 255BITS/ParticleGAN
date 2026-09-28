"""lrfree custom22 IMPORT-ONLY SHIM for benchmarks/legacy/recipe.py.

``ae_gan_hold`` imports ``get_recipe`` at module level; the custom22 port never calls it.
Calling it raises.
"""


def get_recipe(*args, **kwargs):
    raise RuntimeError("import-only shim: the legacy recipe is not part of the custom22 learner")
