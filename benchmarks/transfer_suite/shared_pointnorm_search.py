"""Run declared pointwise-normalized D variants with one unchanged shared cap6 recipe.

python -u -m benchmarks.transfer_suite.shared_pointnorm_search --plan PLAN --output OUTPUT
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import torch
from particlegan import get_recipe
from . import suite, vector_tasks
from .compare_defaults import candidate, effective_spec, ema_verdict, optimizer_defaults, plan, write
from .protocol import test_verdict
from .shared_pointnorm_research import ARCHITECTURES, constructor, variant
from .shared_variants import architecture_spec


def recipe():
    return get_recipe(lr=.00425, d_lr_mult=1., prior_lr_mult=2., betas=(0., .99),
                      reg_coeff=6., reg_kappa=1.25, prior_reg=.05).replace(name='shared_c6')


def prepare(declaration):
    if set(declaration) != {'purpose', 'architectures', 'tasks'}:
        raise ValueError('declare purpose, architecture names and task names only')
    cards = {card['name']: card for card in ARCHITECTURES}
    jobs = {job['spec']['name']: job for job in plan() if job['spec']['runner'] == 'vector'}
    for key, valid in [('architectures', cards), ('tasks', jobs)]:
        names = declaration[key]
        if not names or len(names) != len(set(names)) or not set(names) <= set(valid):
            raise ValueError(f'invalid {key}')
    return [jobs[name] for name in declaration['tasks']], [cards[name] for name in declaration['architectures']]


def episode(job, card):
    selected = variant(card)
    settings = recipe()
    spec = effective_spec(architecture_spec(job['spec'], selected), settings)
    applied = []
    started = time.perf_counter()
    try:
        with optimizer_defaults(settings, applied), patch.object(vector_tasks, 'SimpleMLPDiscriminator', constructor(card)):
            result = suite.run_episode(spec, vector_tasks.fixed_policy('cosine'), fixed=True)
        json.dumps(result, allow_nan=False)
    except Exception:
        result = dict(error=traceback.format_exc(), seconds=time.perf_counter()-started)
    return dict(recipe=settings.to_dict(), candidate=asdict(candidate(settings)), original_spec=deepcopy(job['spec']),
                spec=spec, discriminator_variant=selected, architecture=card['name'],
                reference=job['reference'], reference_sha256=job['reference_sha256'], applied=applied,
                verdict=test_verdict(spec, result), ema_verdict=ema_verdict(spec, result), result=result)


def render(records, output):
    names = sorted({row['architecture'] for row in records})
    tasks = list(dict.fromkeys(row['spec']['name'] for row in records))
    rows = ['# Shared cap6 pointwise normalized discriminator research', '',
            'Every architecture uses the same shared_c6 recipe; original G, resources, targets and gates. '
            'Live PASS requires all 24 observations and a final suffix of at least five. EMA is separate. '
            'Architecture may vary per case; incomplete screens are not complete six-data profiles or 19/19 claims.', '',
            '| Discriminator | Parameters | '+ ' | '.join(tasks)+' |',
            '| --- | ---: | '+' | '.join('---' for _ in tasks)+' |']
    for name in names:
        subset = [r for r in records if r['architecture'] == name]
        params = next((a['parameters'] for a in subset[0]['applied'] if a['role'] == 'd'), '?')
        cells = []
        for task in tasks:
            row = next((r for r in subset if r['spec']['name'] == task), None)
            cells.append('—' if row is None else f"{row['verdict']['status']} ({row['verdict'].get('convergence', {}).get('passing_suffix', 0)}/24)")
        rows.append(f'| {name} | {params} | '+' | '.join(cells)+' |')
    rows += ['', '| D | Task | Live | EMA | Seconds | Artifact |', '| --- | --- | --- | --- | ---: | --- |']
    for row in records:
        rows.append(f"| {row['architecture']} | {row['spec']['name']} | {row['verdict']['status']} | "
                    f"{row['ema_verdict']['status']} | {row['seconds']:.2f} | [JSON]({row['artifact']}) |")
    (output/'README.md').write_text('\n'.join(rows)+'\n')


def run(declaration, output):
    jobs, cards = prepare(declaration)
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(version='shared-pointnorm-v1', declaration=declaration, seed=0, jobs=jobs,
                    architectures=cards, recipe=recipe().to_dict(),
                    selection='D-only architecture trials. All 24 checks, final 5 live PASS; EMA separate.')
    write(output/'protocol.json', protocol)
    write(output/'plan.json', declaration)
    records = []
    for card in cards:
        for job in jobs:
            suite.verify_source(protocol)
            name = job['spec']['name']
            print(f'START {card["name"]} {name}', flush=True)
            payload = episode(job, card)
            payload['source_sha256'] = protocol['source_sha256']
            raw = (json.dumps(payload, sort_keys=True, allow_nan=False)+'\n').encode()
            artifact = f'episodes/shared_c6__{card["name"]}__{name}.json.gz'
            (output/artifact).write_bytes(gzip.compress(raw, mtime=0))
            result = payload['result']
            records.append({k: v for k, v in payload.items() if k not in ('result', 'source_sha256')} |
                           dict(artifact=artifact, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                live=result.get('live'), ema=result.get('ema'), seconds=result['seconds']))
            write(output/'index.json', dict(records=records))
            render(records, output)
            print(json.dumps(dict(event='DONE', architecture=card['name'], task=name,
                                  status=payload['verdict']['status'],
                                  suffix=payload['verdict'].get('convergence', {}).get('passing_suffix'),
                                  live=result.get('live'), seconds=result['seconds'], error=result.get('error'))), flush=True)
    suite.verify_source(protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()), args.output)
