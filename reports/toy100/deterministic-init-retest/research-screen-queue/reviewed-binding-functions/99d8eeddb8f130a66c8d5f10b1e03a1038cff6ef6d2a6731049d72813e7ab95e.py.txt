"""Horizon-blind constant rate/noise primitives for this candidate.
Accept host adapter signatures, but do not inspect step counts or budgets.
The frozen drivers still own termination, observations and metadata.
"""
import sys
from particlegan import recipes
from benchmarks.toy100 import models, schedule

def constant_rate(*args, **kwargs):
    return 1.0

def constant_multipliers(*args, **kwargs):
    return 1.0, 1.0

def no_input_noise(*args, **kwargs):
    return 0.0

def constant_output_noise(peak, *args, **kwargs):
    return float(peak)

replacements = [(recipes.learning_rate_scale,constant_rate),
                (schedule.policy_multipliers,constant_multipliers),
                (models.linear_input_noise,no_input_noise),
                (models.linear_output_noise,constant_output_noise)]
# Include aliases loaded before this mechanism import; later imports see patched definitions.
for module in list(sys.modules.values()):
    if module is None or not getattr(module,'__name__','').startswith(('particlegan','benchmarks')): continue
    for name,value in list(vars(module).items()):
        for original,replacement in replacements:
            if value is original: setattr(module,name,replacement)
