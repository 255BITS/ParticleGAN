"""Declared card of the retired relative-step Adam research rule.

The rule capped each Adam proposal by its tensor's pre-update RMS. It changed
the optimizer update rule from outside the optimizer (it patched Adam's step),
so its runner (``shared_adapter_search``) and implementation were removed. The
experiment was negative: 12/19 against the base recipe's 15/19
(``relative_step_adapter.md``). The frozen reproduction lives in
``reports/transfer_suite/unadjusted/runs/shared-adapter-search/reproduction/``.

Only the card stays, so ``reports/transfer_suite/unadjusted/build.py`` can
still validate archived rows against the declared equation. A new attempt
belongs in the recipe-built optimizers (``particlegan.recipes``) as an option.
"""
import math


def mechanism(fraction):
    if fraction is not None and (not math.isfinite(fraction) or fraction <= 0):
        raise ValueError('fraction must be positive and finite, or None for identity')
    return dict(kind='adam_relative_step_cap', fraction=fraction,
                parameter_rms_floor=.1, epsilon=1e-12, trace_interval=20,
                granularity='parameter tensor', moment_update='ordinary Adam from raw gradients',
                formula='delta * min(1, fraction * max(rms(parameter_before), floor) / (rms(delta) + epsilon))',
                role_blind=True, clock_input=False, metric_input=False)
