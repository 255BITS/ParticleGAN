"""Tiny research-host initializer binding. Importing this file imports no Torch.

Prepared source only. The old research learner and all losses stay in their
original module namespace. Exact public initializer code is loaded separately.
"""
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from unittest.mock import patch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform_function(source):
    prior='ParticlePrior(recipe.n_particles, Z_DIM, init_std=0.5, generator=stream)'
    model='    critic = SimpleMLPDiscriminator(2, HIDDEN, N_HIDDEN, FOURIER)'
    start='    recipe = ModeHoldRecipe() if recipe is None else recipe'
    assert source.count(prior)==source.count(model)==source.count(start)==1
    changed=source.replace(prior,'_newinit_make_prior(recipe.n_particles, Z_DIM, init_std=0.5, generator=stream)')
    changed=changed.replace(model,model+'\n    _newinit_models(generator, critic, prior, stream)')
    changed=changed.replace(start,'    if training_recipe is not None:\n        raise RuntimeError("research binding requires original custom host path")\n'+start)
    restored=changed.replace('_newinit_make_prior(recipe.n_particles, Z_DIM, init_std=0.5, generator=stream)',prior)
    restored=restored.replace('\n    _newinit_models(generator, critic, prior, stream)','')
    restored=restored.replace('    if training_recipe is not None:\n        raise RuntimeError("research binding requires original custom host path")\n','')
    assert restored==source
    compile(changed,'<prepared-newinit-research-mode-hold>','exec')
    return changed


def load_initializer(root, expected):
    """Load a complete verified package under a private, relative-import name."""
    root=Path(root)
    for rel,want in expected.items():
        if sha((root/rel).read_bytes())!=want:
            raise ValueError('initializer source changed: '+rel)
    name='_research_retest_public_initializer'
    if name in sys.modules:
        raise RuntimeError('initializer namespace already loaded; use a fresh process')
    package=root/'particlegan'
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@contextmanager
def bind_mode_hold(host, public, expected_function_sha256, capture):
    """Bind only fresh constructor initialization; restore host globals on exit.

    Factory class substitution is local to the private initializer package.
    The old prior class and its original research registration hooks still run.
    Only host constructor statements change; update/loss/Adam/schedule arithmetic stays original.
    """
    old_function=host.train_mode_hold
    original=inspect.getsource(old_function).rstrip('\n')
    if sha(original.encode())!=expected_function_sha256:
        raise ValueError('unreviewed full research host function')
    changed=transform_function(original)
    old_prior_class=host.ParticlePrior
    prior_module=importlib.import_module(public.__name__+'.particle_prior')
    init_module=importlib.import_module(public.__name__+'.initialization')
    if init_module._external_init is not None:
        raise RuntimeError('global/research initializer hooks are forbidden here')

    def make_prior(num_particles,z_dim,*,init_std,generator):
        # Actual new public factory: one original constructor draw, then R2.
        recipe=public.get_recipe('gan',num_particles=num_particles,z_dim=z_dim,
                                 prior_kind='particles',initialization='batch_feature_zero')
        with patch.object(prior_module,'ParticlePrior',old_prior_class):
            prior=recipe.make_prior(init_std=init_std,generator=generator)
        if type(prior) is not old_prior_class:
            raise RuntimeError('historical prior class identity changed')
        return prior

    def models(generator,critic,prior,stream):
        if type(generator).__name__!='SimpleMLPGenerator' or type(critic).__name__!='SimpleMLPDiscriminator':
            raise RuntimeError('only the frozen dense tiny host is reviewed')
        # Isolated BatchDistanceDiscriminator isinstance checks cannot identify
        # an old external class. Reject such extensions until explicitly bound.
        for model in (generator,critic):
            for module in model.modules():
                if type(module).__name__=='BatchDistanceDiscriminator' or (hasattr(module,'scales') and hasattr(module,'head')):
                    raise RuntimeError('batch-distance/custom head needs independent binding')
        init_module.initialize_(generator,key=0)
        init_module.initialize_(critic,key=1)
        capture(generator,critic,prior,stream)

    namespace=host.__dict__
    sentry=object();saved={k:namespace.get(k,sentry) for k in ['_newinit_make_prior','_newinit_models']}
    try:
        namespace.update(_newinit_make_prior=make_prior,_newinit_models=models)
        exec(compile(changed,inspect.getsourcefile(old_function),'exec'),namespace)
        yield dict(original_sha256=sha(original.encode()),transformed_sha256=sha(changed.encode()),source=changed)
    finally:
        host.train_mode_hold=old_function
        for key,value in saved.items():
            if value is sentry:namespace.pop(key,None)
            else:namespace[key]=value
