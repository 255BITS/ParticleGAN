"""Run one shared recipe with an explicit discriminator architecture profile.

python -m benchmarks.transfer_suite.shared_profile_search --plan PLAN --output OUTPUT
Recipes have only global overrides. Profile cards change vector discriminators;
the original host, generator, data, resources, metrics and budgets stay fixed.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from . import shared_default_search as reference
from . import shared_discriminator_search as architecture
from . import suite
from .compare_defaults import plan, write
from .shared_variants import architecture_spec


IMPLEMENTATIONS = {
    'shared_critic_v1': 'shared_critic_research',
    'shared_pointnorm_v1': 'shared_pointnorm_research',
    'shared_ensemble_v1': 'shared_ensemble_research',
    'shared_local_density_v1': 'shared_local_density_research',
    'shared_residual_curvature_v1': 'shared_residual_curvature_research',
    'shared_norm_structure_v1': 'shared_norm_structure_research',
    'shared_batch_feature_v1': 'shared_batch_feature_research',
}


def implementation(card):
    return importlib.import_module('.'+IMPLEMENTATIONS[card['implementation']], __package__)


def prepare(declaration):
    if set(declaration) - {'candidates', 'tasks', 'purpose', 'discriminators'}:
        raise ValueError('only shared recipes and a declared D profile are supported')
    jobs, recipes = reference.prepare({k: v for k, v in declaration.items() if k != 'discriminators'})
    profile = deepcopy(declaration.get('discriminators', {}))
    vector_jobs = {j['spec']['name']: j for j in plan() if j['spec']['runner'] == 'vector'}
    if not isinstance(profile, dict) or not set(profile) <= set(vector_jobs):
        raise ValueError('profile entries must name canonical vector hosts')
    for name, card in profile.items():
        module = implementation(card)
        architecture_spec(vector_jobs[name]['spec'], module.variant(card))
    return jobs, recipes, profile


def episode(job, recipe, card=None):
    if card is None:
        return reference.episode(job, recipe)
    module = implementation(card)
    # The existing audited D runner supplies the host, scoring and actual
    # optimizer receipts. Only its declared recipe and D constructor vary.
    with patch.object(architecture, 'recipe', lambda: recipe), \
         patch.object(architecture, 'constructor', module.constructor), \
         patch.object(architecture, 'variant', module.variant):
        return architecture.episode(job, deepcopy(card))


def run(declaration, output):
    jobs, recipes, profile = prepare(declaration)
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(version='shared-profile-v1', seed=0, jobs=jobs,
                    declaration=declaration,
                    selection='One unchanged recipe across all hosts. Explicit D-only architecture profile; '
                    'all 19 live behavioral tests sustained at the final five observations. EMA separate.')
    write(output/'protocol.json', protocol)
    write(output/'plan.json', declaration)
    records = []
    for _, recipe in recipes:
        for job in jobs:
            suite.verify_source(protocol)
            name = job['spec']['name']
            print(f'START {recipe.name} {name}', flush=True)
            payload = episode(job, recipe, profile.get(name))
            payload['source_sha256'] = protocol['source_sha256']
            raw = (json.dumps(payload, sort_keys=True, allow_nan=False)+'\n').encode()
            artifact = f'episodes/{recipe.name}__{name}.json.gz'
            (output/artifact).write_bytes(gzip.compress(raw, mtime=0))
            result = payload['result']
            records.append({k: v for k, v in payload.items() if k not in ('result', 'source_sha256')} |
                           dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
            write(output/'index.json', dict(records=records))
            reference.render(records, output)
            print(json.dumps(dict(event='DONE', candidate=recipe.name, task=name,
                                  architecture=payload['architecture'], status=payload['verdict']['status'],
                                  suffix=payload['verdict'].get('convergence', {}).get('passing_suffix'),
                                  live=result.get('live'), error=result.get('error'))), flush=True)
    suite.verify_source(protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
