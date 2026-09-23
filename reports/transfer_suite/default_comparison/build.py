"""Verify matched default/core runs and regenerate their readable comparison."""
from dataclasses import asdict, replace
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from particlegan import get_recipe
from benchmarks.smart_descent import study
from benchmarks.transfer_suite.compare_defaults import candidate, ema_verdict, effective_spec
from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent
ARMS = ('current', 'proposed', 'current_core')


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def clean_curve(result):
    return [{k: v for k, v in point.items() if k != 'seconds'} for point in result['observations']]


def counts(rows):
    return {kind: dict(passed=sum(r['verdict']['passed'] for r in rows if r['spec']['runner'] == kind),
                       total=sum(r['spec']['runner'] == kind for r in rows))
            for kind in ('legacy', 'vector', 'image')}


def score_text(rows):
    parts = counts(rows)
    return ' | '.join([*(f"{parts[k]['passed']}/{parts[k]['total']}" for k in ('legacy', 'vector', 'image')),
                       f"**{sum(r['verdict']['passed'] for r in rows)}/19**"])


def compact(record):
    return {k: v for k, v in record.items() if k not in ('result', 'source_sha256')}


def build():
    manifest = read(ROOT/'archive_manifest.json')
    for item in manifest['files']:
        assert sha((ROOT/item['path']).read_bytes()) == item['sha256'], item['path']
    source = (ROOT/'current-master-recipes.py.txt').read_bytes()
    audit = read(ROOT/'current-master-audit.json')
    assert sha(source) == audit['sha256']
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Recipe')
    defaults = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}
    assert defaults == get_recipe('gan_legacy').replace(name='gan').to_dict()
    assert json.loads(json.dumps(defaults)) == audit['recipe']
    assert '5 passed' in (ROOT/'tests.log').read_text()
    rows = {}
    references = {}
    sources = 0
    vector_parity = []
    for arm in ARMS:
        folder = ROOT/arm
        protocol = read(folder/'protocol.json')
        with tarfile.open(folder/'source.tar.gz') as archive:
            for name, digest in protocol['source_sha256'].items():
                assert sha(archive.extractfile(name).read()) == digest, name
                sources += 1
        index = read(folder/'index.json')['records']
        assert len(index) == 19 and len({r['spec']['name'] for r in index}) == 19
        rows[arm] = []
        for entry in index:
            raw = gzip.decompress((folder/entry['artifact']).read_bytes())
            assert sha(raw) == entry['uncompressed_sha256']
            payload = json.loads(raw)
            result, spec = payload['result'], payload['spec']
            assert not result.get('error') and len(result['observations']) == 24
            assert payload['source_sha256'] == protocol['source_sha256']
            verdict = test_verdict(spec, result)
            assert verdict == entry['verdict'] == payload['verdict']
            assert verdict['convergence']['complete']
            assert ema_verdict(spec, result) == entry['ema_verdict'] == payload['ema_verdict']
            reference_path = SUITE/payload['reference']
            assert sha(reference_path.read_bytes()) == payload['reference_sha256']
            reference = read(reference_path)
            expected = reference.get('result', reference)
            assert test_verdict(spec, expected)['passed'], 'Published winning reference must pass original thresholds'
            references[spec['name']] = dict(spec=payload['original_spec'], verdict=test_verdict(spec, expected),
                                           ema_verdict=ema_verdict(spec, expected), result=expected,
                                           artifact='../'+payload['reference'])
            assert payload['applied'], 'Missing actual optimizer group audit'
            if arm in ('current', 'proposed'):
                recipe = get_recipe('gan_legacy' if arm == 'current' else 'gan')
                assert payload['recipe'] == json.loads(json.dumps(recipe.to_dict()))
                assert payload['candidate'] == json.loads(json.dumps(asdict(candidate(recipe))))
                assert spec == effective_spec(payload['original_spec'], recipe)
                for group in payload['applied']:
                    assert group['lr'] == recipe.lr * {'g': 1., 'd': recipe.d_lr_mult, 'prior': recipe.prior_lr_mult}[group['role']]
                    assert group['betas'] == list(recipe.betas)
                if arm == 'proposed' and spec['runner'] == 'vector':
                    assert clean_curve(result) == clean_curve(expected)
                    assert result['actions'] == expected['actions']
                    vector_parity.append(spec['name'])
            else:
                expected_config = replace(study.BASE, name='current_core_host_recipe',
                                          reg_coeff=1., reg_kappa=1., vicreg_weight=1., particle_l2=0.)
                assert payload['candidate'] == asdict(expected_config)
                allowed = {'reg_coeff', 'reg_kappa', 'prior_reg'} if spec['runner'] == 'vector' else (
                    {'penalty_coeff', 'kappa', 'prior_weight'} if spec['runner'] == 'image' else set())
                assert {k: v for k, v in spec.items() if k not in allowed} == {
                    k: v for k, v in payload['original_spec'].items() if k not in allowed}
                assert result['actions'] == expected['actions']
            payload['artifact'] = f"{arm}/{entry['artifact']}"
            rows[arm].append(payload)
    for old, new, core in zip(*(rows[a] for a in ARMS)):
        assert old['original_spec'] == new['original_spec'] == core['original_spec']
        assert old['architecture'] == new['architecture'] == core['architecture']
        assert old['result']['actions'] == new['result']['actions']
        if old['spec']['runner'] != 'legacy':
            a, b = axes(old['spec'], old['spec']['runner']), axes(new['spec'], new['spec']['runner'])
            for key in ('target', 'architecture', 'resources'):
                assert a[key] == b[key]
    assert len(vector_parity) == 6 and len(references) == 19
    reference_rows = list(references.values())
    leaderboard = dict(version='matched-default-comparison-v1', counts={a: counts(v) for a, v in rows.items()},
                       references=counts(reference_rows), rows={a: [compact(r) | dict(live=r['result']['live'],
                       ema=r['result'].get('ema')) for r in v] for a, v in rows.items()},
                       note='Public preset and per-host formulation comparisons are separate rankings.')
    (ROOT/'leaderboard.json').write_text(json.dumps(leaderboard, indent=2)+'\n')
    lines = ['# Current default versus proposed default: measured tests', '',
             '**Both presets were trained on all 19 behavioral tests.** Data, architecture, particle support, batch, '
             'initialization and update budget match within every pair. The [current master recipe](current-master-audit.json) '
             'matches every `gan_legacy` field exactly. Each arm uses its public recipe\'s loss, '
             'regularization, absolute G/D/particle LRs and Adam betas. Seed 0; no seed search or changed thresholds.', '',
             '## Direct public-default comparison', '',
             '| Preset | Required | Data | Images | Total live |', '| --- | ---: | ---: | ---: | ---: |',
             '| Current master / `gan_legacy` | '+score_text(rows['current'])+' |',
             '| Proposed / `gan` | '+score_text(rows['proposed'])+' |', '',
             '**The public preset does not pass all 19 tests unchanged.** The earlier 19/19 result belongs to the '
             'new core formulation with established per-host training recipes. It must not be presented as an '
             'all-pass result for one universal numerical optimizer preset.', '',
             'Current: cap 1/κ1, spread 1, Adam(0,.999), G/D/prior LRs .0006/.0009/.006. '
             'Proposed: cap 3/κ1.25, spread .05, Adam(0,.99), LRs .001/.0015/.01. '
             'Both use Rp logistic, no particle L2, and the same delayed cosine schedule. '
             'Toy-specific reconstruction/identity/cover losses and initialization remain fixed. '
             'The toy\'s support and budget replace the generic 20,000 particles/7,000 updates; this is a matched '
             'test-host transfer, not a rerun of the separate 100-Gaussian benchmark.', '',
             '## Isolating the core formulation', '',
             'This second comparison retains exactly the published host LRs, Adam settings, EMA, schedules, '
             'resources and architectures. It changes only cap coefficient, cap threshold and spread weight. '
             'The new formulation column reuses the archived passing run for each architecture; the old core '
             'was freshly trained under those same settings. These are not scores for the public optimizer presets.', '',
             '| Formulation with matched host training | Required | Data | Images | Total live |',
             '| --- | ---: | ---: | ---: | ---: |',
             '| Current core: cap 1/κ1, spread 1 | '+score_text(rows['current_core'])+' |',
             '| Proposed core: cap 3/κ1.25, spread .05 | '+score_text(reference_rows)+' |', '',
             'The current core meets every final metric, but two-pole has only **4** final passing observations '
             'and mode-hold only **1**, below the required 5. The new core sustains both. '
             'This is a stability improvement; it does not mean every individual final metric improves.', '',
             '## Every test', '',
             '| Test | Current preset | Proposed preset | Current core + host recipe | Proposed core + host recipe |',
             '| --- | --- | --- | --- | --- |']
    for old, new, core in zip(*(rows[a] for a in ARMS)):
        name = old['spec']['name']
        cells = [f"[{r['verdict']['status']}]({r['artifact']})" for r in (old, new, core)]
        ref = references[name]
        cells.append(f"[{ref['verdict']['status']}]({ref['artifact']})")
        lines.append(f"| {name} | "+' | '.join(cells)+' |')
    lines += ['', '## Failing metrics and convergence', '',
              'Each row needs all bounds for at least five final observations of the complete 24-point curve. '
              'A final passing point alone cannot earn PASS. The artifact links above contain every live/EMA '
              'measurement and schedule action; [machine-readable leaderboard](leaderboard.json) includes '
              'all thresholds, measured values, margins and convergence results.', '',
              '| Test / preset | Final failed bounds | Final passing streak |', '| --- | --- | ---: |']
    for arm in ('current', 'proposed'):
        for r in rows[arm]:
            if r['verdict']['passed']:
                continue
            failures = '; '.join(f"{m['metric']}={m['value']:.5g} (needs {m['op']}{m['threshold']})"
                                 for m in r['verdict']['metrics'] if m['status'] != 'PASS') or 'Final bounds pass; insufficient sustained checks'
            lines.append(f"| {r['spec']['name']} / {arm} | {failures} | {r['verdict']['convergence']['passing_suffix']} |")
    lines += ['', '## EMA, reported separately', '',
              'EMA is scored using the same complete-curve rule where the host records all EMA metrics. '
              'N/A means no complete EMA curve is available; it earns no pass. Live selection is unchanged.', '',
              '| Test | Current preset EMA | Proposed preset EMA | Current core EMA | Proposed core EMA |',
              '| --- | --- | --- | --- | --- |']
    for old, new, core in zip(*(rows[a] for a in ARMS)):
        name = old['spec']['name']
        lines.append(f"| {name} | "+' | '.join(r['ema_verdict']['status'] for r in (old, new, core, references[name]))+' |')
    lines += ['', '## Verification and reproduction', '',
              '57 new complete training episodes, 19 archived reference episodes, and all failed attempts are retained. '
              'All six proposed vector curves and action traces exactly match their published winners, including '
              'the rare-mode linear-skip fix. The verifier checks all 1,368 new observations, source archives, '
              'artifact hashes, actual optimizer-group LRs/betas, identical paired test conditions and unchanged '
              'core-comparison schedules. Five focused adapter tests catch mixed/AE prior groups and direct particle optimizers.', '',
              'Architectures were selected before these runs from the already published winning row and shared '
              'by both arms. No architecture search was performed for stock, so the comparison does not establish '
              'its best achievable score after architecture tuning. Wall times are single CPU observations and '
              'do not establish a speedup. Legacy EMA behavior remains host-specific; vector/image preset runs '
              'use recipe decay .995.', '',
              '```bash',
              'python -u -m benchmarks.transfer_suite.compare_defaults --arm current --output /tmp/compare-current > /tmp/compare-current.log 2>&1',
              'python -u -m benchmarks.transfer_suite.compare_defaults --arm proposed --output /tmp/compare-proposed > /tmp/compare-proposed.log 2>&1',
              'python -u -m benchmarks.transfer_suite.compare_formulations --output /tmp/compare-core > /tmp/compare-core.log 2>&1',
              'tail -f /tmp/compare-proposed.log',
              'python -m reports.transfer_suite.default_comparison.build',
              '```', '', '[Adapter tests](tests.log) · [Artifact validation](validation.json) · [Archive manifest](archive_manifest.json).', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    validation = dict(new_episodes=57, reference_episodes=19, new_observations=57*24,
                      source_file_instances=sources, archive_files=len(manifest['files']),
                      exact_proposed_vector_replays=vector_parity, live_counts={a:sum(r['verdict']['passed'] for r in v) for a,v in rows.items()},
                      focused_adapter_tests=5, errors=[])
    (ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    build()
