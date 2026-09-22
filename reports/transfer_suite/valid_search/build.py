"""Rebuild the valid-toy comparison from complete, archived numerical evidence."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.transfer_suite.formulations import architecture_cell
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def record(path):
    value = read(path)
    result = value['result']
    spec = value.get('effective_spec', value['spec'])
    verdict = test_verdict(spec, result)
    assert verdict == value['verdict'], path
    return dict(name=spec['name'], spec=spec, result=result, verdict=verdict,
                artifact=str(path.relative_to(SUITE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def make_cells(paths, *, check_architecture=False):
    grouped = defaultdict(list)
    for path in paths:
        r = record(path)
        grouped[r['name']].append(r)
    cells = {}
    for name, trials in grouped.items():
        if check_architecture and name.startswith('img_'):
            architecture_cell([dict(label=t['artifact'], spec=t['spec'], result=t['result']) for t in trials], 'image')
        passed = [t for t in trials if t['verdict']['passed']]
        best = min(passed, key=lambda t: t['verdict']['convergence']['confirmed_step']) if passed else None
        cells[name] = dict(passed=bool(passed), steps=trials[0]['spec']['steps'],
                           confirmed_step=best['verdict']['convergence']['confirmed_step'] if best else None,
                           trials=[{k: t[k] for k in ('artifact', 'sha256', 'verdict')} for t in trials])
    return cells


def summary(name, required, data, images, description):
    cells = required | data | images
    passed = [c for c in cells.values() if c['passed']]
    return dict(name=name, description=description, required=required, data=data, images=images,
                required_passes=sum(c['passed'] for c in required.values()),
                data_passes=sum(c['passed'] for c in data.values()), image_passes=sum(c['passed'] for c in images.values()),
                practical_passes=sum(c['passed'] for c in (data | images).values()),
                eligible=all(c['passed'] for c in required.values()) and len(required) == 9,
                complete=len(required) == 9 and len(data) == 6 and len(images) == 4,
                mean_confirmation_fraction=sum(c['confirmed_step']/c['steps'] for c in passed)/len(passed))


def build():
    manifest = {s['name']: s for s in read(SUITE/'study/manifest.json')['tasks']}
    baseline = read(SUITE/'formulations/leaderboard.json')['rows'][0]
    required = {}
    for name, item in baseline['required'].items():
        required[name] = dict(passed=item['verdict']['passed'], steps=manifest[name]['steps'],
                              confirmed_step=item['verdict']['convergence']['confirmed_step'], trials=[item])
    practical = {}
    for name, cell in baseline['cases'].items():
        passes = [t['verdict'] for t in cell['trials'] if t['verdict']['passed']]
        practical[name] = dict(passed=bool(passes), steps=cell['resources']['steps'],
                               confirmed_step=min(v['convergence']['confirmed_step'] for v in passes) if passes else None,
                               trials=cell['trials'])
    rows = [summary('Original recipe + supported D architectures', required,
                    {k:v for k,v in practical.items() if k.startswith('vector_')},
                    {k:v for k,v in practical.items() if k.startswith('img_')},
                    'Original optimizer settings. D128x3/Fourier3 or4 adds overlap; D64x2/Fourier2 with Softplus(beta5 or10) adds unequal width. Suitable D can differ by toy; residual16 covers all images. Same b_cap3 formulation.')]
    for name, folder, data_paths, image_paths, description in [
        ('Adam beta2=.999', 'adam999_hosts',
         list((SUITE/'solvability/vectors/screen/episodes').glob('adam999__*.gz')) + list((SUITE/'solvability/vectors/regressions/episodes').glob('adam999__*.gz')),
         list((ROOT/'adam999_images/episodes').glob('*.gz')),
         'Adam(0,.999) across hosts; other optimizer settings unchanged. Images require architecture choices: residual16/24 for stripes,bars,intensity; transpose12 for blobs.'),
        ('Coordinated LR recipe', 'coordinated_hosts',
         list((ROOT/'recipes').glob('**/episodes/b999_lr075_d2_p30__*.gz')), [],
         'Adam(0,.999); relative to each host base rates G x.75, D x1, particles x2.25. Same mapping everywhere; exact parity verified on the rare vector and neutral image control.')]:
        hosts = list((ROOT/folder/'episodes').glob('*.gz'))
        correction = 'adam999_group_fix' if folder == 'adam999_hosts' else 'coordinated_group_fix'
        req = [p for p in hosts if '__img_' not in p.name and '__ae_gan_hold' not in p.name]
        req += list((ROOT/correction/'episodes').glob('*.gz'))
        imgs = [p for p in hosts if '__img_' in p.name] + image_paths
        rows.append(summary(name, make_cells(req), make_cells(data_paths),
                            make_cells(imgs, check_architecture=folder=='adam999_hosts'), description))
    assert [(r['required_passes'],r['data_passes'],r['image_passes']) for r in rows] == [
        (9,baseline['domains']['vector']['passed'],4),(9,4,4),(7,4,2)]
    assert all(r['complete'] for r in rows)
    eligible = [r for r in rows if r['eligible']]
    common = set.intersection(*[{k for group in ('required','data','images') for k,c in row[group].items() if c['passed']} for row in eligible])
    for row in eligible:
        cells = row['required'] | row['data'] | row['images']
        row['common_confirmation_fraction'] = sum(cells[k]['confirmed_step']/cells[k]['steps'] for k in common)/len(common)
    report = dict(formulation='Rp logistic, b_cap3,kappa1.25,prior regularization .05,no particle L2', rows=rows,
                  scope='Nine required, six data, four healthy image toys; imposed dynamics are diagnostic, longer training is separate.',
                  grouping='Recipe trials belong to the same core formulation. Never combine successes from different optimizer recipes or resource budgets into one row. Architecture may vary within a recipe.',
                  timing=f'Mean confirmation fraction compares the same {len(common)} passing cases using earliest supported architecture per case. This is development-selected update evidence, not wall-time speedup.',
                  correction='Original cross-host AE runs preserved host per-group beta1=.5. Current comparison replaces those two AE results with explicit per-group Adam(0,.999) reruns; both pass. Originals retained as superseded evidence.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'leaderboard.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Valid behavioral toys: search for b_cap3 defaults','',
           '**The original b_cap3 recipe now passes 9/9 required and 9/10 practical toys using supported discriminator architectures.** '
           'The rare 2% mode is the remaining failure. Different toys may use different D architectures; a single D does not pass them all. Adam beta2=.999 reaches 8/10 with different supported image architectures.',
           '', 'All rows use Rp logistic, b_cap3/κ1.25, prior regularization .05 and no particle L2. '
           'Optimizer choices are separate trials under that formulation. Architecture variants stay within a recipe; failures and every untested case remain visible.',
           '', '| Training recipe | Required live | Data | Images with supported architecture | Practical | Qualified on required |',
           '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['name']} | {row['required_passes']}/9 | {row['data_passes']}/6 | {row['image_passes']}/4 | {row['practical_passes']}/10 | {'Yes' if row['eligible'] else 'No'} |")
    lines += ['', 'The main comparison stays at the original update budgets: Gaussian data 1,200, spiral 1,600, images 600 and each required host’s existing budget. '
              '[Longer training](../formulations/LONG_TRAINING.md) is a separate toy. Live thresholds and the final five of 24 rule are unchanged. EMA is retained separately.',
              '', '## What improved', '',
              '- **Discriminator architecture alone fixes overlap.** D128×3 with Fourier 3 or 4 passes broad, anisotropic, overlap and spiral. '
              'G remains 4,610 parameters; D grows from 4,929 to 35,073/35,585. The required and image results reuse unchanged settings and exact archived evidence. '
              'This is an architecture improvement under the same formulation, with no extra updates.',
              '- **Smooth discriminator activation fixes unequal width.** Replacing LeakyReLU with Softplus(beta5) in the original D64×2/Fourier2 holds its 4,929 parameters and every training setting fixed. '
              'It passes the final seven checks; beta10 also passes. Their full six-data profiles pass 3/6; the formulation uses appropriate D architectures for the other toys. '
              '[Architecture results](smooth_discriminator/README.md) · [Reusable critic and reproduction](../../../benchmarks/transfer_suite/smooth_critic_research.md).',
              '- **Adam beta2=.999 is another recipe under the same formulation.** It passes all nine required hosts and fixes overlap with the smaller original vector D. '
              'Residual16 fails blobs (HQ 84.4%); transpose12 passes that toy under the same recipe. All four image architecture profiles are retained.',
              '- **The coordinated recipe trades away other passes.** The coordinated recipe solves rare mass and overlap at 256 particles and the original budget, '
              'but loses anisotropic data, required trajectory identity and the required eight-mode ring. Its residual16 images pass 2/4.',
              '', f'Across the same {len(common)} passing behavioral cases, mean confirmed-step / budget, choosing the earliest supported architecture per case: '
              + ', '.join(f"{r['name']}={r['common_confirmation_fraction']:.3f}" for r in rows if r['eligible']) + '. '
              'These are inspected development results. More D capacity and concurrent CPU load prevent a wall-time speed claim.',
              '', '## Evidence and reproduction', '',
              '[All D architectures](discriminator/README.md) · [18 coordinated recipes](recipes/README.md) · '
              '[D/recipe combinations](discriminator_combinations/README.md) · [Adam999 required/image checks](adam999_hosts/README.md) · '
              '[Adam999 image architectures](adam999_images/README.md) · [Coordinated recipe across hosts](coordinated_hosts/README.md) · '
              '[Resource searches](resources/README.md) · [Softplus refinement](softplus_refinement/README.md) · [512-particle smooth-D checks](smooth512/README.md).',
              '', 'The research host adapter applies the same LR factors to G, D and ParticlePrior parameter groups across hosts, including direct particle-only optimizers. '
              'It reproduces the native rare-vector result and the neutral image control exactly before cross-host evaluation; '
              '[parity evidence](coordinated_hosts/parity.json.gz) and exact driver source are archived. No production API or defaults changed.',
              '', 'Independent review caught an explicit AE prior-group beta1=.5 overriding the declared Adam pair. '
              'Both affected AE cases were rerun with Adam(0,.999) enforced on every parameter group; both pass. '
              'The comparison uses only the corrected AE results. Original runs remain as superseded evidence. '
              '[Adam999 correction](adam999_group_fix/README.md) · [Coordinated correction](coordinated_group_fix/README.md).',
              '', 'Seed 0 only; no seed sweeps. Every attempted result, source archive, configuration and failure is retained. '
              '[Machine-readable comparison](leaderboard.json) · [Artifact validation](validation.json) · [29 focused tests](tests.log).', '', '```bash',
              'python -m reports.transfer_suite.formulations.build', 'python -m reports.transfer_suite.valid_search.build', '```']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    print([(r['name'],r['required_passes'],r['practical_passes']) for r in rows])


if __name__=='__main__': build()
