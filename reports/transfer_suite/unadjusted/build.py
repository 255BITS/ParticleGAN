"""Build the primary leaderboard: one unchanged recipe, all 19 live tests."""
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile

from particlegan import Recipe
from benchmarks.transfer_suite.compare_defaults import candidate, effective_spec, ema_verdict, plan, read
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.shared_variants import architecture_identity, architecture_spec, select_architecture

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def link(path):
    return os.path.relpath(path, ROOT)


def validate_update_rule(payload, indexed):
    rule = payload.get('mechanism', {'kind': 'adam'})
    assert indexed.get('mechanism', {'kind': 'adam'}) == rule
    if rule['kind'] == 'adam':
        assert rule == {'kind': 'adam'} and 'adapter' not in payload, 'Undeclared Adam transformation'
        return rule
    if rule['kind'] != 'adam_relative_step_cap':
        raise ValueError('Unreviewed update mechanism')
    from benchmarks.transfer_suite.relative_step_adapter import mechanism
    assert rule == mechanism(rule['fraction']) and rule['fraction'] is not None
    assert payload['adapter']['mechanism'] == rule
    trace = payload['adapter']['trace']
    assert trace or payload['result'].get('error'), 'Missing adapted-step evidence'
    for step in trace:
        assert all(math.isfinite(step[k]) and step[k] >= 0 for k in ('parameter_rms', 'proposal_rms', 'factor'))
        expected = min(1., rule['fraction'] * max(step['parameter_rms'], rule['parameter_rms_floor']) /
                       (step['proposal_rms'] + rule['epsilon']))
        assert 0 < step['factor'] <= 1 and math.isclose(step['factor'], expected, rel_tol=1e-10, abs_tol=1e-12), 'Adaptation receipt contradicts equation'
    return rule


def build():
    declared = {j['spec']['name']: json.loads(json.dumps(j)) for j in plan()}
    rows = []
    checked_sources = set()
    for entry in read(ROOT/'entries.json'):
        trials, identities, recipe_dict, update_rule = {}, {}, None, None
        for index_name in entry['indexes']:
            index_path = REPO/index_name
            protocol_path = index_path.parent/'protocol.json'
            protocol = read(protocol_path if protocol_path.exists() else protocol_path.with_suffix('.json.gz'))
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
                assert payload['recipe'] == indexed['recipe'] and payload['recipe']['name'] == entry['name']
                assert payload['source_sha256'] == protocol['source_sha256']
                if recipe_dict is None:
                    recipe_dict = payload['recipe']
                    update_rule = payload.get('mechanism', {'kind': 'adam'})
                assert recipe_dict == payload['recipe'], 'Per-test recipe adjustment is forbidden'
                assert update_rule == validate_update_rule(payload, indexed), 'Per-test update rule adjustment is forbidden'
                recipe = Recipe(**recipe_dict)
                spec, result = payload['spec'], payload['result']
                name = spec['name']
                assert name in declared, 'Unknown test'
                assert payload['original_spec'] == declared[name]['spec'], 'Changed test setup'
                variant = payload.get('discriminator_variant')
                original = architecture_spec(declared[name]['spec'], variant)
                expected_architecture = variant['name'] if variant else declared[name]['architecture']
                assert payload['architecture'] == expected_architecture, 'Undeclared architecture'
                assert spec == effective_spec(original, recipe), 'Hidden task override'
                previous = trials.setdefault(name, [])
                identity = json.dumps(architecture_identity(original), sort_keys=True)
                seen = identities.setdefault(name, set())
                assert identity not in seen, 'Duplicate architecture trial (including renamed or no-op variants)'
                seen.add(identity)
                assert payload['candidate'] == json.loads(json.dumps(asdict(candidate(recipe))))
                assert payload['applied'] or payload['result'].get('error'), 'Missing actual optimizer receipts'
                for group in payload['applied']:
                    role = group['role']
                    assert group['lr'] == recipe.lr * {'g': 1., 'd': recipe.d_lr_mult, 'prior': recipe.prior_lr_mult}[role]
                    betas = (recipe.prior_betas or recipe.betas) if role == 'prior' else recipe.betas
                    assert group['betas'] == list(betas)
                verdict = test_verdict(spec, result)
                assert verdict == indexed['verdict'] == payload['verdict']
                ema = ema_verdict(spec, result)
                assert ema == indexed['ema_verdict'] == payload['ema_verdict']
                previous.append(dict(verdict=verdict, ema_verdict=ema, live=result.get('live'),
                                     ema=result.get('ema'), artifact=link(path), architecture=payload['architecture'],
                                     discriminator_variant=variant))
        records = {name: select_architecture(attempts) for name, attempts in trials.items()}
        assert recipe_dict is not None and records, 'Entry has no matching runs'
        complete = len(records) == 19
        passed = sum(r['verdict']['passed'] for r in records.values())
        counts = {}
        for kind in ('legacy', 'vector', 'image'):
            names = [name for name, job in declared.items() if job['spec']['runner'] == kind]
            counts[kind] = dict(passed=sum(records.get(n, {}).get('verdict', {}).get('passed', False) for n in names), total=len(names))
        rows.append(dict(name=entry['name'], label=entry['label'], recipe=recipe_dict, mechanism=update_rule, records=records,
                         attempted=len(records), passed=passed, counts=counts,
                         episodes=sum(len(t) for t in trials.values()),
                         reference_passed=sum(r['reference_passed'] for r in records.values()),
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
             '| Candidate | Required | Data | Images | Live total | Reference D profile | Attempted | Overall |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        if r['attempted'] != 19:
            continue
        c = r['counts']
        lines.append(f"| {r['label']} (`{r['name']}`) | {c['legacy']['passed']}/9 | {c['vector']['passed']}/6 | "
                     f"{c['image']['passed']}/4 | **{r['passed']}/19** | {r['reference_passed']}/19 | {r['attempted']}/19 | **{r['overall']}** |")
    lines += ['', 'Rank complete candidates by live passes, then lower normalized final metric shortfall. '
              'An all-pass candidate is the target; a partial improvement is not an all-pass stamp.', '',
              'Live total counts a test once when the unchanged recipe supports a declared discriminator '
              'architecture. The reference column shows passes with the original frozen D profile. '
              'Architecture trials never change the optimizer recipe; every failed trial remains archived.', '',
              '## Screening results — not ranked', '',
              '| Candidate | Passing measured tests | Attempted | Status |',
              '| --- | ---: | ---: | --- |']
    for r in rows:
        if r['attempted'] != 19:
            lines.append(f"| {r['name']} | {r['passed']} | {r['attempted']}/19 | INCOMPLETE |")
    if all(r['attempted'] == 19 for r in rows):
        lines.append('| No partial candidates | — | — | — |')
    lines += ['',
              '## One recipe per row', '',
              '| Candidate | G / D / particle LR | Adam betas | b_cap coefficient / κ | Spread weight | Update rule |',
              '| --- | --- | --- | --- | ---: | --- |']
    for r in rows:
        p = r['recipe']
        betas = str(tuple(p['betas']))+(f"; prior {tuple(p['prior_betas'])}" if p['prior_betas'] is not None else '')
        rule = r['mechanism']
        update_label = 'Adam' if rule['kind'] == 'adam' else f"Adam + relative step cap {rule['fraction']:g}"
        lines.append(f"| {r['name']} | {p['lr']:.6g} / {p['lr']*p['d_lr_mult']:.6g} / {p['lr']*p['prior_lr_mult']:.6g} | "
                     f"{betas} | {p['reg_coeff']:g} / {p['reg_kappa']:g} | {p['prior_reg']:g} | {update_label} |")
    display = [r for r in rows if r['attempted'] == 19][:3]
    display += [r for r in rows if r['name'] in ('gan', 'gan_legacy') and r not in display]
    lines += ['', 'All current entries use Rp logistic, no particle L2, and the same schedule: hold for 60% of '
              'the budget, then cosine toward 5%. Rates above are absolute and are applied to every optimizer '
              'group, including directly optimized particles and AE prior groups.', '',
              'The update-rule column declares any additional transformation of the Adam proposal. '
              'Its complete equation and identical global parameters are retained in leaderboard.json '
              'and each episode; reported LRs are the base schedule before that transformation.', '',
              '## Every test: leading complete candidates and public baselines', '',
              'All candidate metrics and EMA profiles remain in [leaderboard.json](leaderboard.json).', '',
              '| Test | '+' | '.join(r['name'] for r in display)+' |',
              '| --- | '+' | '.join('---' for _ in display)+' |']
    for name in declared:
        cells = []
        for row in display:
            record = row['records'].get(name)
            cells.append(f"[{record['verdict']['status']}]({record['artifact']})" if record else 'NOT RUN')
        lines.append(f"| {name} | "+' | '.join(cells)+' |')
    lines += ['', '## Remaining failures in the leading complete recipes', '',
              'The final passing streak must reach five observations. A good last checkpoint alone does not pass.', '',
              '| Recipe | Test | Final failing metrics (value; required bound) | Final passing streak |',
              '| --- | --- | --- | ---: |']
    for row in [r for r in rows if r['attempted'] == 19][:3]:
        for name, record in row['records'].items():
            verdict = record['verdict']
            if verdict['passed']:
                continue
            details = []
            for metric in verdict.get('metrics', []):
                if metric['status'] != 'PASS':
                    value = 'missing' if metric['value'] is None else f"{metric['value']:.5g}"
                    details.append(f"{metric['metric']}: {value}; needs {metric['op']} {metric['threshold']:g}")
            reason = '; '.join(details) or ('Final metrics pass' if verdict['status'] == 'FAIL' else verdict['status'])
            suffix = verdict.get('convergence', {}).get('passing_suffix', 0)
            lines.append(f"| {row['name']} | [{name}]({record['artifact']}) | {reason} | {suffix}/5 |")
    variants = [(r['name'], name, t) for r in rows for name, record in r['records'].items()
                for t in record['trials'] if t['discriminator_variant'] is not None]
    if variants:
        lines += ['', '## Discriminator architecture trials', '',
                  'Architecture support is within a single unchanged recipe. All trials are shown, including failures; '
                  'it does not mean one universal discriminator works everywhere.', '',
                  '| Recipe | Test | Discriminator | Live |', '| --- | --- | --- | --- |']
        for recipe_name, name, trial in variants:
            lines.append(f"| {recipe_name} | {name} | {trial['architecture']} | "
                         f"[{trial['status']}]({trial['artifact']}) |")
    lines += ['', '## What stays fixed in the tests', '',
              'Data, target metrics, thresholds, seed 0, generators, initialization rules, particle counts, batch sizes '
              'and update budgets are the frozen test setup. They match for every candidate. Each task keeps its '
              'existing reconstruction/identity/cover objective. Resource sizes differ between tests, but a '
              'candidate cannot change them to obtain a pass. These are development cases, not unseen holdouts.', '',
              'Architecture remains separate from formulation: explicit discriminator variants are allowed under '
              'the same unchanged recipe, with all trials/failures recorded. The importer checks D-only changes '
              'against the frozen reference test and reports reference-profile performance separately. '
              'Legacy EMA measurement remains host-specific and never affects ranking.', '',
              '## Join the search', '',
              '[Contribution instructions and one-command run](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md). '
              'The runner accepts one global recipe card and runs all 19 tests by default. Preserve failed runs. '
              'Screening is allowed, but only a complete row can qualify.', '',
              '[Current search findings and remaining failures](FINDINGS.md) · [Reproduce the leading recipes](leading_candidates.json).', '',
              '[All metrics, convergence and separate EMA](leaderboard.json) · [Registered entries](entries.json) · '
              '[Validation](validation.json) · [Historical adjusted comparison](../default_comparison/README.md).', '']
    (ROOT/'README.md').write_text('\n'.join(lines))
    validation = dict(entries=len(rows), episodes=sum(r['episodes'] for r in rows),
                      discriminator_variants=len(variants),
                      live_counts={r['name']: r['passed'] for r in rows},
                      attempted_counts={r['name']: r['attempted'] for r in rows},
                      overall_passes=[r['name'] for r in rows if r['overall']=='PASS'], errors=[])
    (ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    build()
