"""Explicit construction of the unchanged frozen hosts on the current API.

Initialization belongs to this fixture adapter, outside the candidate policy.
The trusted historical initializer is loaded under an isolated module name;
it never installs global PyTorch hooks or replaces the candidate package.
"""
import importlib
import sys
from pathlib import Path
from types import ModuleType


REFERENCE = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan')
_ALIAS = '_frozen_ra11_initializer'


def initializer():
    if _ALIAS not in sys.modules:
        namespace = ModuleType(_ALIAS)
        namespace.__path__ = [str(REFERENCE)]
        sys.modules[_ALIAS] = namespace
        # Preserve the special readout declaration for the actual current class.
        sys.modules[_ALIAS + '.discriminators'] = importlib.import_module('particlegan.discriminators')
    return importlib.import_module(_ALIAS + '.initialization')


def frozen_recipe(package, **options):
    options = dict(options)
    mode = options.pop('initialization', 'batch_feature_zero')
    assert mode == 'batch_feature_zero', 'only the original frozen initialization is supported'
    return package.get_recipe(**options)


def frozen_prior(recipe, **options):
    prior = recipe.make_prior(**options)
    return initializer()._initialize_prior(prior, options.get('init_std', 1.0))


def frozen_trainer(package, recipe, generator, critic, **options):
    options = dict(options)
    if options.get('prior') is None:
        first = next(p for p in generator.parameters() if p.requires_grad)
        options['prior'] = frozen_prior(recipe).to(device=first.device, dtype=first.dtype)
    init = initializer()
    init._initialize(generator, key=0, only_new=True)
    init._initialize(critic, key=1, only_new=True)
    return package.GANTrainer(recipe, generator, critic, **options)
