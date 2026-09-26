"""Horizon-blind rate latch for p2_generator_collapse.

Noise is not replaced. Ring drivers keep the copied K3P input and output
schedules on their frozen noise horizon of 1200. That noise is a retained
scheduled component. The multiplier ignores step and horizon values except as
a memo key so the post-step rate check sees the applied value.
"""
from benchmarks.toy100 import schedule

DWELL = 0.10
ADAPT = 0.25
state = {
    'multiplier': 1.0,
    'phase': 'acquire',
    's': 1.0,
    'precision_locked': False,
    'g_peak': 0.0,
    'g_quiet': None,
    'g_floor': None,
    'g_rms': None,
    'confirm': 0.0,
    'cache': {},
    'multiplier_trace': [],
    'last_multiplier': (1.0, 1.0),
}


def multipliers(*args, **kwargs):
    key = args[0] if args and type(args[0]) is int else None
    if key is not None and key in state['cache']:
        return state['cache'][key]
    multiplier = float(state['multiplier'])
    result = (multiplier, multiplier)
    state['last_multiplier'] = result
    if key is not None:
        state['cache'][key] = result
        if key % 50 == 0:
            state['multiplier_trace'].append([
                key, multiplier, state['phase'], round(state['s'], 6),
                state['precision_locked'], state['g_rms'], state['g_peak'],
                round(state['confirm'], 6),
            ])
    return result


schedule.policy_multipliers = multipliers
