"""Build the primary leaderboard: one unchanged recipe, all 19 live tests."""
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile

from particlegan import Recipe
from benchmarks.transfer_suite.compare_defaults import candidate, effective_spec, ema_verdict, plan, read
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def link(path):
    return os.path.relpath(path, ROOT)


def build():
    declared = {j['spec']['name']: json.loads(json.dumps(j)) for j in plan()}
    rows = []
    checked_sources = set()
    for entry in read(ROOT/'entries.json'):
        records, recipe_dict = {}, None
        for index_name in entry['indexes']:
            index_path = REPO/index_name
            protocol = read(index_path.parent/'protocol.json')
            if index_name not in checked_sources:
                with tarfile.open(index_path.parent/'source.tar.gz') as archive:
                    for name, sha in protocol['source_sha256'].items():
                        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == sha, name
                checked_sources.add(index_name)
            for indexed in read(index_path)['records']:
                if indexed.get('recipe', {}).get('name') != entry['name']:
                    continue
                path = index_path.parent/indexed['artifact']
                raw = gzip.decompress(path.read_bytes())
                assert hashlib.sha256(raw).hexdigest() == indexed['uncompressed_sha256']
                payload = json.loads(raw)
                assert payload['source_sha256'] == protocol['source_sha256']
                if recipe_dict is None:
                    recipe_dict = payload['recipe']
                assert recipe_dict == payload['recipe'], 'Per-test recipe adjustment is forbidden'
                recipe = Recipe(**recipe_dict)
                spec, result = payload['spec'], payload['result']
                name = spec['name']
                assert name in declared and name not in records, 'Duplicate or unknown test'
                assert payload['original_spec'] == declared[name]['spec'], 'Changed test setup'
                assert payload['architecture'] == declared[name]['architecture'], 'Changed fixed architecture'
                assert spec == effective_spec(declared[name]['spec'], recipe), 'Hidden task override'
                assert payload['candidate'] == json.loads(json.dumps(asdict(candidate(recipe))))
                for group in payload['applied']:
                    role = group['role']
                    assert group['lr'] == recipe.lr * {'g': 1., 'd': recipe.d_lr_mult, 'prior': recipe.prior_lr_mult}[role]
                    betas = (recipe.prior_betas or recipe.betas) if role == 'prior' else recipe.betas
                    assert group['betas'] == list(betas)
                verdict = test_verdict(spec, result)
                assert verdict == indexed['verdict'] == payload['verdict']
                ema = ema_verdict(spec, result)
                assert ema == indexed['ema_verdict'] == payload['ema_verdict']
                records[name] = dict(verdict=verdict, ema_verdict=ema, live=result.get('live'),
                                     ema=result.get('ema'), artifact=link(path), architecture=payload['architecture'])
        assert recipe_dict is not None and records, 'Entry has no matching runs'
        complete = len(records) == 19
        passed = sum(r['verdict']['passed'] for r in records.values())
        counts = {}
        for kind in ('legacy', 'vector', 'image'):
            names = [name for name, job in declared.items() if job['spec']['runner'] == kind]
            counts[kind] = dict(passed=sum(records.get(n, {}).get('verdict', {}).get('passed', False) for n in names), total=len(names))
        rows.append(dict(name=entry['name'], label=entry['label'], recipe=recipe_dict, records=records,
                         attempted=len(records), passed=passed, counts=counts,
                         overall='PASS' if complete and passed == 19 else 'FAIL' if complete else 'INCOMPLETE',
                         shortfall=sum(r['verdict']['shortfall'] for r in records.values())+(19-len(records))*2))
    rows.sort(key=lambda r: (r['attempted'] != 19, -r['passed'], r['shortfall'], r['name']))
    (ROOT/'leaderboard.json').write_text(json.dumps(dict(version='unadjusted-defaults-v1', rows=rows), indent=2)+'\n')
    lines = ['# Unadjusted ParticleGAN default leaderboard', '',
             '**This is the primary comparison for selecting a shared default.** Each candidate uses one unchanged '
             'loss/regularization/optimizer recipe on every test. No per-example LR, Adam, prior-rate or loss-weight '
             'adjustments. The earlier adjusted 19/19 result does not compete on this leaderboard.', '',
             '**Overall PASS requires 19/19 live behavioral tests**, each passing every metric for at least five '
             'final observations of a complete 24-point curve. EMA is separate. Missing cases stay in the denominator; '
             'partial rows cannot beat a completed candidate or qualify as winners.', '',
             '| Candidate | Required | Data | Images | Live total | Attempted | Overall |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        c = r['counts']
        lines.append(f"| {r['label']} (`{r['name']}`) | {c['legacy']['passed']}/9 | {c['vector']['passed']}/6 | "
                     f"{c['image']['passed']}/4 | **{r['passed']}/19** | {r['attempted']}/19 | **{r['overall']}** |")
    lines += ['', 'Rank complete candidates by live passes, then lower normalized final metric shortfall. '
              'An all-pass candidate is the target; a partial improvement is not an all-pass stamp.', '',
              '## One recipe per row', '',
              '| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight |',
              '| --- | --- | --- | --- | ---: |']
    for r in rows:
        p = r['recipe']
        betas = str(tuple(p['betas']))+(f"; prior {tuple(p['prior_betas'])}" if p['prior_betas'] is not None else '')
        lines.append(f"| {r['name']} | {p['lr']:.6g} / {p['lr']*p['d_lr_mult']:.6g} / {p['lr']*p['prior_lr_mult']:.6g} | "
                     f"{betas} | {p['reg_coeff']:g} / {p['reg_kappa']:g} | {p['prior_reg']:g} |")
    lines += ['', 'All current entries use Rp logistic, no particle L2, and the same schedule: hold for 60% of '
              'the budget, then cosine toward 5%. Rates above are absolute and are applied to every optimizer '
              'group, including directly optimized particles and AE prior groups.', '',
              '## Every test', '', '| Test | '+' | '.join(r['name'] for r in rows)+' |',
              '| --- | '+' | '.join('---' for _ in rows)+' |']
    for name in declared:
        cells = []
        for row in rows:
            record = row['records'].get(name)
            cells.append(f"[{record['verdict']['status']}]({record['artifact']})" if record else 'NOT RUN')
        lines.append(f"| {name} | "+' | '.join(cells)+' |')
    lines += ['', '## What stays fixed in the tests', '',
              'Data, target metrics, thresholds, seed 0, architectures, initializations, particle counts, batch sizes '
              'and update budgets are the frozen test setup. They match for every candidate. Each task keeps its '
              'existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a '
              'candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.', '',
              'The selected architecture for each test is shared by all candidates. A different architecture study '
              'must be reported separately; no candidate may silently cherry-pick a different network per result. '
              'Legacy EMA measurement remains host-specific and never affects ranking.', '',
              '## Join the search', '',
              '[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). '
              'The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. '
              'Screening is allowed, but only a complete row can qualify.', '',
              '[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · '
              '[Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    validation = dict(entries=len(rows), episodes=sum(r['attempted'] for r in rows),
                      live_counts={r['name']: r['passed'] for r in rows},
                      overall_passes=[r['name'] for r in rows if r['overall']=='PASS'], errors=[])
    (ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    build()
