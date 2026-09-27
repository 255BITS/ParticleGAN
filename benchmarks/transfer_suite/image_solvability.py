"""Shared image-GAN configuration search with immutable behavioral gates.

Research only. Supervised expressivity witnesses are a separate control and
never enter the GAN leaderboard. All previously reserved image data is now
seen development data; this module does not introduce or inspect a new holdout.

Each card is a task change or a recipe change; episodes train on the shared
toy runner through ``image_tasks.run_episode`` (no patched host globals).
"""
from copy import deepcopy
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import traceback

import torch

from . import image_tasks as host

HEALTHY = [s for s in host.TASKS if s['tier'] == 'ranking']
CARDS = [
    dict(name='baseline', scope='same architecture/budget', changes={}),
    dict(name='r1r2_01', scope='same architecture/budget', changes=dict(gradient_penalty='a_r1r2', penalty_coeff=.1)),
    dict(name='r1r2_1', scope='same architecture/budget', changes=dict(gradient_penalty='a_r1r2', penalty_coeff=1.)),
    dict(name='cap_coeff03', scope='same architecture/budget', changes=dict(penalty_coeff=.3)),
    dict(name='vanilla_r1r2_01', scope='same architecture/budget', changes=dict(gan_mode='vanilla', gradient_penalty='a_r1r2', penalty_coeff=.1)),
    dict(name='lsgan_vanilla', scope='same architecture/budget', changes=dict(loss_type='lsgan', gan_mode='vanilla')),
    dict(name='prior_lr10', scope='same architecture/budget', changes=dict(prior_lr_multiplier=10.)),
    dict(name='fixed_prior', scope='same architecture/budget', changes=dict(prior_learnable=False)),
    dict(name='residual16', scope='architecture changed', changes=dict(architecture='residual_upsample', width=16)),
    dict(name='residual16_r1r2_01', scope='architecture changed', changes=dict(architecture='residual_upsample', width=16, gradient_penalty='a_r1r2', penalty_coeff=.1)),
    dict(name='transpose24', scope='architecture changed', changes=dict(width=24)),
    dict(name='baseline_1200', scope='budget doubled', changes=dict(steps=1200)),
]


def policy():
    """Declaration record only: the recipe-built optimizers own the LR schedule."""
    return dict(weights=torch.zeros(2, 2, 5).tolist(), interval=5, schedule='recipe')


def resolve(original, card):
    """A card changes the task (architecture, width, budget, fixed prior) or the
    recipe fields ``spec_recipe`` reads (loss, penalty, prior LR multiplier)."""
    spec = deepcopy(original)
    spec.update(loss_type='logistic', gan_mode='rp', prior_lr_multiplier=1., prior_learnable=True)
    spec.update(card['changes'])
    if spec['thresholds'] != original['thresholds']:
        raise ValueError('changing gates is forbidden')
    return spec


def episode(original, card):
    spec = resolve(original, card)
    result = host.run_episode(spec, policy())
    result['research_card'] = card
    result['effective_spec'] = spec
    result['research_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def supervised(original):
    """MSE with balanced latent-to-template labels; representability only.

    Same networks, prior, recipe generator optimizer (schedule included) and
    EMA as the GAN baseline; no critic and no prior spread term.
    """
    spec = deepcopy(original)
    torch.set_num_threads(1)
    started = time.perf_counter()
    result = dict(kind='supervised expressivity control — not a GAN', spec=spec,
                  live={}, ema={}, observations=[], losses=[], convergence={})
    try:
        recipe = host.spec_recipe(spec).replace(prior_reg=0.)
        result.update(host.train(spec, recipe, problem=host.SupervisedWitness(spec)))
    except Exception:
        result['error'] = traceback.format_exc()
    result['seconds'] = time.perf_counter()-started
    return result


def quality(result):
    live = result.get('live', {})
    return result.get('convergence', {}).get('confirmed_step') is not None, live.get('modes', 0), live.get('hq', 0.)


def ranking(row):
    results = list(row['results'].values())
    complete = len(results) == len(HEALTHY)
    passes = sum(quality(r)[0] for r in results)
    shortfall = sum(1-quality(r)[1]/r['effective_spec']['modes'] + max(0.,.9-quality(r)[2])/.9 for r in results)
    return (not complete, -passes, shortfall)


def render(report, output):
    lines = ['# Image GAN solvability: shared configurations', '',
             'Seed0, unchanged live quality/coverage gates, 24 observations and five final passing checks. '
             'Architecture and budget changes are explicit. Supervised controls do not enter GAN ranking. '
             'Previously reserved residual bars are now seen development data; no fresh holdout is evaluated.', '',
             '| Shared card | Scope | Sustained / 4 | Stripes | Bars | Blobs | Intensity | CPU seconds |',
             '| --- | --- | ---: | --- | --- | --- | --- | ---: |']
    def cell(result):
        if result is None: return '—'
        if result.get('error'): return 'ERROR'
        stable, modes, hq = quality(result)
        return f'{"PASS" if stable else "FAIL"} {modes}/{result["spec"]["modes"]} · {hq:.1%}'
    for row in sorted(report['rows'], key=ranking):
        values = row['results']
        lines.append(f'| {row["card"]["name"]} | {row["card"]["scope"]} | {sum(quality(r)[0] for r in values.values())}/4 | ' +
                     ' | '.join(cell(values.get(s['name'])) for s in HEALTHY) +
                     f' | {sum(r["seconds"] for r in values.values()):.1f} |')
    lines += ['', 'Each cell reports sustained verdict, final quality-qualified modes and HQ. '
              'A final pass with an insufficient passing suffix remains FAIL. Missing tasks cannot form a shared winner.', '',
              '| Supervised expressivity control (not a GAN) | Result | Confirmed step |', '| --- | --- | ---: |']
    for name, result in report.get('supervised', {}).items():
        lines.append(f'| {name} | {cell(result)} | {result.get("convergence", {}).get("confirmed_step") or "—"} |')
    lines += ['', 'Every exact spec, failure, curve, action trace and source/runtime hash is retained in the episode JSON.gz artifacts.', '']
    (output/'README.md').write_text('\n'.join(lines))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def snapshot(output):
    root = Path(__file__).resolve().parents[2]
    protocol = host.fingerprint()
    protocol['source_sha256'][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name, digest in sorted(protocol['source_sha256'].items()):
            raw = (root/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == digest
            info = tarfile.TarInfo(name); info.size=len(raw); info.mtime=0; info.mode=0o644
            tar.addfile(info, io.BytesIO(raw))
    (output/'source.tar.gz').write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    return protocol


def run(output, cards, *, controls=False):
    if output.exists():
        raise FileExistsError('use a fresh output directory')
    output.mkdir(parents=True)
    (output/'episodes').mkdir()
    protocol = snapshot(output)
    declaration = dict(cards=cards, tasks=HEALTHY, supervised_controls=controls, protocol=protocol,
                       selection='Shared sustained count across all4, then summed final mode/HQ deficits; diagnostic comparisons separate.',
                       parent_revision='afe615264221eda47c5d1b7fd2cf4082552e30e9', fresh_holdout_evaluated=False)
    write_json(output/'declaration.json', declaration)
    report = dict(declaration=declaration, rows=[], supervised={})
    def save_result(name, result):
        raw=(json.dumps(result, sort_keys=True, allow_nan=False)+'\n').encode()
        filename=f'episodes/{name}.json.gz'
        (output/filename).write_bytes(gzip.compress(raw,mtime=0))
        # Keep readable results compact; the archive retains full action traces.
        result = {k:v for k,v in result.items() if k not in ('actions','protocol')}
        result.update(artifact=filename, original_sha256=hashlib.sha256(raw).hexdigest())
        return result
    def save():
        write_json(output/'results.json',report)
        render(report,output)
    save()
    if controls:
        for task in HEALTHY:
            print('START supervised',task['name'],flush=True)
            result=supervised(task)
            report['supervised'][task['name']]=save_result('supervised__'+task['name'],result)
            save()
            print(json.dumps(dict(control='supervised',task=task['name'],quality=quality(result),error=result.get('error'))),flush=True)
    for card in cards:
        row=dict(card=card,results={}); report['rows'].append(row)
        for task in HEALTHY:
            print('START',card['name'],task['name'],flush=True)
            result=episode(task,card)
            row['results'][task['name']]=save_result(card['name']+'__'+task['name'],result)
            save()
            print(json.dumps(dict(card=card['name'],task=task['name'],quality=quality(result),seconds=result['seconds'],error=result.get('error'))),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cards',type=Path)
    parser.add_argument('--controls',action='store_true')
    from benchmarks.toy100.device import add_device_argument, apply_device_policy
    add_device_argument(parser)
    args=parser.parse_args()
    apply_device_policy(args.device, log=True)
    cards=json.loads(args.cards.read_text()) if args.cards else CARDS
    run(args.output,cards,controls=args.controls)

if __name__=='__main__':
    main()
