"""Declare discriminator architecture trials without changing a shared recipe."""
from copy import deepcopy

from .formulations import axes


def architecture_identity(spec):
    """Ignore labels and normalize implicit D dimensions when rejecting repeats."""
    def without_names(value):
        if isinstance(value, dict):
            return {k: without_names(v) for k, v in value.items() if k != 'name'}
        if isinstance(value, list):
            return [without_names(v) for v in value]
        return value
    if spec['runner'] == 'vector':
        return without_names(axes(spec, 'vector')['architecture']['discriminator'])
    return {'reference_host': spec['name']}


def architecture_spec(original, variant=None):
    """Only discriminator fields may differ from the frozen reference host."""
    spec = deepcopy(original)
    if variant is None:
        return spec
    if original['runner'] != 'vector':
        raise ValueError('explicit discriminator variants currently support vector hosts only')
    if set(variant) != {'name', 'overrides'} or not isinstance(variant['name'], str) or not variant['name']:
        raise ValueError('variant needs a name and discriminator-only overrides')
    overrides = variant['overrides']
    if not isinstance(overrides, dict) or not overrides or set(overrides) - {
        'd_hidden', 'd_layers', 'fourier', 'research_discriminator'
    }:
        raise ValueError('only discriminator architecture fields may change')
    for key in ('d_hidden', 'd_layers', 'fourier'):
        if key in overrides and (type(overrides[key]) is not int or overrides[key] < (0 if key == 'fourier' else 1)):
            raise ValueError(f'invalid discriminator dimension: {key}')
    spec.update(deepcopy(overrides))
    # A null card explicitly returns to the ordinary MLP implementation.
    if spec.get('research_discriminator', True) is None:
        del spec['research_discriminator']
    before, after = axes(original, 'vector'), axes(spec, 'vector')
    for key in ('formulation', 'training', 'resources', 'target'):
        if before[key] != after[key]:
            raise ValueError(f'discriminator variant changed {key}')
    if before['architecture']['generator'] != after['architecture']['generator']:
        raise ValueError('discriminator variant changed generator')
    return spec


def select_architecture(trials):
    """Choose supported D architecture deterministically; retain every attempt."""
    selected = min(trials, key=lambda r: (
        not r['verdict']['passed'], r['verdict']['shortfall'],
        r['verdict']['confirmation_fraction'], r['architecture'], r['artifact']))
    summaries = [dict(architecture=t['architecture'], artifact=t['artifact'],
                      discriminator_variant=t['discriminator_variant'],
                      status=t['verdict']['status'] if 'status' in t['verdict'] else ('PASS' if t['verdict']['passed'] else 'FAIL'),
                      shortfall=t['verdict']['shortfall']) for t in trials]
    return dict(selected, trials=summaries,
                reference_passed=any(t['verdict']['passed'] and t['discriminator_variant'] is None for t in trials))
