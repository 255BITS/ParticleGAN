"""Archived comparison inputs, not selectable public recipes.

These complete receipts keep old leaderboard rows reproducible when defaults
change. Application code uses particlegan.get_recipe() directly.
"""
import json
from pathlib import Path
from particlegan import Recipe
from benchmarks.gan_v3 import legacy_recipe

_RECORDS = json.loads((Path(__file__).parents[1] / "transfer_suite/plans/recipe_history.json").read_text())["recipes"]
GAN_V1 = legacy_recipe(_RECORDS["gan_v1"]).replace(name="gan_legacy")
GAN_V2 = legacy_recipe(_RECORDS["gan_v2"]).replace(name="gan")
