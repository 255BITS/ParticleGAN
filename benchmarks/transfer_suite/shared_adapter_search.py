"""Bounded, frozen relative-step-adapter search on the unchanged shared recipe."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch

from particlegan import get_recipe
from . import shared_default_search as shared, suite
from .compare_defaults import plan, read, write
from .relative_step_adapter import adapted_steps, mechanism
from benchmarks.gan_v3 import gan_v3_recipe

RECIPE = dict(lr=.00425, d_lr_mult=1., prior_lr_mult=2., betas=[0., .99],
              reg_coeff=3., reg_kappa=1.25, prior_reg=.05)
CARDS = [dict(name=f'relative_cap_{label}', mechanism=mechanism(fraction))
         for label, fraction in [('01', .01), ('025', .025), ('05', .05)]]
SCREEN = ['ae_gan_hold', 'img_bars4', 'mode_hold', 'vector_unequal_mass',
          'vector_unequal_width', 'vector_overlap']
CONTROLS = ['mode_hold', 'vector_unequal_mass', 'img_bars4']
BASELINE = Path('reports/transfer_suite/unadjusted/runs/round0-2')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in
                ('seconds', 'controller_seconds', 'confirmed_seconds', 'stable_from_seconds', 'created_at')}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def canonical(value):
    """Normalize tuple/list differences between in-memory specs and JSON evidence."""
    return json.loads(json.dumps(value, allow_nan=False))


def render(records, baseline, output):
    candidates = {}
    for record in records:
        if record['stage'] != 'control':
            candidates.setdefault(record['adapter_candidate'], []).append(record)
    lines = ['# Shared gradient-update adaptation', '',
             'All candidates retain one common recipe and fixed architectures. The same role-blind equation '
             'caps every tensor update on every task. EMA is separate; only19/19 live sustained qualifies.', '',
             '| Candidate | Sustained live | Attempted /19 | Overall |',
             '| --- | ---: | ---: | --- |',
             f"| lr00425_prior2 (archived) | {sum(r['verdict']['passed'] for r in baseline)} |19/19 | FAIL |"]
    for name, rows in candidates.items():
        passes = sum(r['verdict']['passed'] for r in rows)
        status = 'PASS' if passes == len(rows) == 19 else 'FAIL' if len(rows) == 19 else 'INCOMPLETE'
        lines.append(f'| {name} | {passes} | {len(rows)}/19 | {status} |')
    lines += ['', '| Candidate | Task | Live | Final passing checks | EMA |',
              '| --- | --- | --- | ---: | --- |']
    for row in records:
        if row['stage'] == 'control':
            continue
        lines.append(f"| {row['adapter_candidate']} | {row['spec']['name']} | {row['verdict']['status']} | "
                     f"{row['verdict'].get('convergence', {}).get('passing_suffix', 0)} | {row['ema_verdict']['status']} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    torch.set_num_threads(1)
    jobs = plan()
    by_name = {j['spec']['name']: j for j in jobs}
    declaration = dict(base_recipe=RECIPE, candidates=CARDS, screen=SCREEN, controls=CONTROLS,
        selection='Most sustained passes on six-screen, then lowest mean final normalized shortfall, then candidate name. '
                  'Complete the other13tasks for exactly one selected candidate; no retuning.',
        architecture='Frozen common reference profile from compare_defaults.plan; unchanged for all candidates.',
        budget='Three identity replay controls,18screen episodes,13completion episodes. Seed0 only; original task budgets.',
        base_revision='78c872236b70f6e5527169db7143559536e052a1')
    write(output/'plan.json', declaration)
    protocol = suite.snapshot(output)
    protocol.update(jobs=jobs, declaration=declaration, seed=0,
                    version='shared-relative-adam-adapter-v1', runtime='CPU, one Torch thread')
    write(output/'protocol.json', protocol)
    baseline_index = read(BASELINE/'index.json')
    base_rows = [r for r in baseline_index['records'] if r['recipe']['name'] == 'lr00425_prior2']
    assert len(base_rows) == 19
    reference_paths = {r['spec']['name']: BASELINE/r['artifact'] for r in base_rows}
    write(output/'references.json', [dict(task=name, path=str(path),
          sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in reference_paths.items()])
    records = []

    def episode(card, name, stage):
        suite.verify_source(protocol)
        print(f"START {stage} {card['name']} {name}", flush=True)
        recipe_name = 'lr00425_prior2' if stage == 'control' else card['name']
        recipe = gan_v3_recipe(**RECIPE).replace(name=recipe_name)
        evidence = {}
        with adapted_steps(card['mechanism'], evidence):
            payload = shared.episode(by_name[name], recipe)
        payload.update(adapter_candidate=card['name'], mechanism=card['mechanism'],
                       adapter=evidence, stage=stage, source_sha256=protocol['source_sha256'])
        if stage == 'control':
            reference = read(reference_paths[name])
            parity = dict(result_except_timing=clean(payload['result']) == clean(reference['result']),
                          recipe=canonical(payload['recipe']) == reference['recipe'],
                          applied_groups=canonical(payload['applied']) == reference['applied'],
                          spec=canonical(payload['spec']) == reference['spec'])
            payload['parity'] = parity
        raw = (json.dumps(payload, sort_keys=True, allow_nan=False)+'\n').encode()
        filename = f"episodes/{card['name']}__{name}.json.gz"
        (output/filename).write_bytes(gzip.compress(raw, mtime=0))
        row = {k: v for k, v in payload.items() if k not in ('result', 'source_sha256', 'adapter')}
        row.update(artifact=filename, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                   live=payload['result'].get('live'), ema=payload['result'].get('ema'),
                   seconds=payload['result']['seconds'], adapter_seconds=evidence['seconds'],
                   attenuation=evidence['summaries'])
        records.append(row)
        write(output/'index.json', dict(records=records))
        render(records, base_rows, output)
        verdict = payload['verdict']
        print(json.dumps(dict(event='DONE', candidate=card['name'], task=name,
              status=verdict['status'], suffix=verdict.get('convergence', {}).get('passing_suffix'),
              shortfall=verdict['shortfall'], live=payload['result'].get('live'),
              adapter_seconds=evidence['seconds'], error=payload['result'].get('error'),
              parity=payload.get('parity'))), flush=True)
        if stage == 'control':
            assert all(parity.values()), parity

    for name in CONTROLS:
        episode(dict(name='identity', mechanism=mechanism(None)), name, 'control')
    for card in CARDS:
        for name in SCREEN:
            episode(card, name, 'screen')
    def rank(card):
        rows = [r for r in records if r['adapter_candidate'] == card['name']]
        return (-sum(r['verdict']['passed'] for r in rows),
                sum(r['verdict']['shortfall'] for r in rows)/len(rows), card['name'])
    ranked = sorted(CARDS, key=rank)
    selected = ranked[0]
    write(output/'selection.json', dict(rule=declaration['selection'], selected=selected,
          ranking=[dict(candidate=card['name'], key=rank(card)) for card in ranked]))
    print('SELECTED', selected['name'], flush=True)
    for job in jobs:
        name = job['spec']['name']
        if name not in SCREEN:
            episode(selected, name, 'completion')
    suite.verify_source(protocol)
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args = parser.parse_args()
    apply_device_policy(args.device, log=True)
    run(args.output)
