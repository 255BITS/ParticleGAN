"""Build the focused rare-mode architecture comparison from every full episode."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from benchmarks.transfer_suite.formulations import architecture_cell
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.vector_tasks import fixed_policy

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def build():
    by_case = defaultdict(list)
    records = []
    for path in sorted(ROOT.glob('**/episodes/*.json.gz')):
        value = read(path)
        spec = value.get('effective_spec', value['spec'])
        result = value['result']
        verdict = test_verdict(spec, result)
        ema_verdict = test_verdict(spec, dict(live=result['ema'], observations=[
            dict(o['ema'], step=o['step'], seconds=o['seconds']) for o in result['observations']]))
        assert verdict == value['verdict'], path
        assert verdict['convergence']['complete'], path
        assert value['policy'] == fixed_policy('cosine'), path
        assert len(result['observations']) == 24 and not result.get('error'), path
        family = path.relative_to(ROOT).parts[0]
        label = path.name.split('__')[0]
        artifact = str(path.relative_to(SUITE))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        item = dict(family=family, label=label, spec=spec, verdict=verdict, ema_verdict=ema_verdict,
                    live=result['live'], ema=result['ema'],
                    observations=result['observations'], artifact=artifact, sha256=sha)
        records.append(item)
        by_case[spec['name']].append(dict(label=label, spec=spec, result=result, artifact=artifact))
    # Compare every architecture against the original target, recipe and budget.
    cells = {}
    manifest = {s['name']: s for s in read(SUITE/'study/manifest.json')['tasks']}
    for name, variants in by_case.items():
        baseline = read(SUITE/'study/episodes'/f'cosine__{name}.json.gz')
        cells[name] = architecture_cell([dict(label='original architecture', spec=manifest[name],
                                              result=baseline['result'])] + variants, 'vector')
    rare = [r for r in records if r['spec']['name'] == 'vector_unequal_mass']
    rare.sort(key=lambda r: (not r['verdict']['passed'],
                            -r['verdict']['convergence']['passing_suffix'],
                            r['verdict']['shortfall'], r['label']))
    winners = [r for r in rare if r['verdict']['passed']]
    report = dict(episodes=len(records), rare_candidates=len(rare), sustained_rare_winners=len(winners),
                  cross_checks=len(records)-len(rare),
                  records=[{k: v for k, v in r.items() if k != 'observations'} for r in records], cells=cells,
                  selection='Development search, seed0. All metrics at every one of the final five of 24 live observations; EMA separate. No target or threshold changes.',
                  grouping='Only D architecture changes within the original b_cap3 recipe and resources. Different architectures may support different toys. Diagnostic replays/label-aware geometry controls do not enter this leaderboard.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'leaderboard.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Focused rare-mode architecture search', '',
             f"**{len(winners)} sustained rare-mode {'winner' if len(winners) == 1 else 'winners'} among {len(rare)} new discriminator architectures.** "
             f'{len(records)-len(rare)} additional data-toy cross-checks. '
             'All runs retain Rp logistic, b_cap3 / κ1.25, prior regularization .05, no particle L2, '
             'Adam(0,.99), G LR .001, D LR .0015, prior LR .01, cosine, the original G, '
             '256 particles, batch128 and 1,200 rare-toy updates. Architecture may vary by toy.', '',
             'A final PASS is insufficient: every metric must pass at all five late checks. '
             'The spread ratio is the smallest covariance eigenvalue relative to the target, across all four components. '
             'It detects a component flattened into a line even when occupancy and sample quality look good. '
             'EMA is measured separately and never determines the live result.', '']
    if winners:
        winner = winners[0]
        c = winner['verdict']['convergence']
        lines += [f"**The winner is {winner['label']}: D96×2, Softplus(beta5), Fourier2, plus a learned raw-coordinate linear skip initialized to zero.** "
                  f"Its two-parameter skip brings D to 10,467 parameters. Live bounds hold from update {c['stable_from_step']:,} through 1,200, "
                  f"confirm at {c['confirmed_step']:,}, and pass {c['passing_suffix']} final checks. "
                  'This supplies the missing rare-toy witness: the same formulation now supports **9/9 required + 10/10 practical = 19/19 behavioral toys**. '
                  'Different toys use different architectures; the winning D itself passes 3/6 data toys.', '',
                  f"The winner's EMA has only {winner['ema_verdict']['convergence']['passing_suffix']} final passing checks and remains **FAIL** under the same sustained rule. Live weights supply this PASS.", '',
                  '[Reproduction command and reusable discriminator](../../../benchmarks/transfer_suite/linear_skip_research.md) · '
                  '[Independent exact replay](winner_replay/parity.json.gz) · [CLI exact numerical replay](cli_replay/parity.json.gz).', '']
    lines += [
             '| Architecture family | Candidates | Sustained live passes | Closest live candidate | Final passing streak / required 5 | Sustained EMA passes (separate) |',
             '| --- | ---: | ---: | --- | ---: | ---: |']
    for family in dict.fromkeys(r['family'] for r in rare):
        family_rows = [r for r in rare if r['family'] == family]
        best = family_rows[0]
        lines.append(f"| {family} | {len(family_rows)} | {sum(r['verdict']['passed'] for r in family_rows)} | "
                     f"{best['label']} | {best['verdict']['convergence']['passing_suffix']} | "
                     f"{sum(r['ema_verdict']['passed'] for r in family_rows)} |")
    lines += ['', '<details><summary>Every architecture, including all failures</summary>', '',
             '| D architecture | Family | Sustained live | Final passing streak / required 5 | Final spread / minimum .15 | Final failing metrics |',
             '| --- | --- | --- | ---: | ---: | --- |']
    for r in rare:
        v = r['verdict']
        failures = ', '.join(c['metric'] for c in v['metrics'] if c['status'] != 'PASS') or 'All final bounds pass'
        lines.append(f"| [{r['label']}]({r['artifact'].removeprefix('rare_focus/')}) | {r['family']} | {v['status']} | "
                     f"{v['convergence']['passing_suffix']} | {r['live']['component_min_eigen_ratio']:.4f} | {failures} |")
    lines += ['', '</details>', '', '## Five late measurements', '',
              'Closest candidates by sustained result, final passing streak, then final normalized bound shortfall. '
              'All 24 observations and every failure are retained in the machine-readable report.', '',
              '| Architecture | Metric | 1,000 | 1,050 | 1,100 | 1,150 | 1,200 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rare[:3]:
        for metric in ('component_min_eigen_ratio', 'component_covariance_error', 'hq', 'mass_tv', 'min_mass_ratio'):
            values = ' | '.join(f"{o[metric]:.4f}" for o in r['observations'][-5:])
            lines.append(f"| {r['label']} | {metric} | {values} |")
    if winners:
        lines += ['', '## Full six-data profiles for rare-mode winners', '',
                  '| Architecture | Toy | Sustained live | Final passing streak | Confirmed step |',
                  '| --- | --- | --- | ---: | ---: |']
        for winner in winners:
            profile = [r for r in records if r['label'] == winner['label'] and r['family'] == winner['family']]
            assert len(profile) == 6, winner['label']
            for r in profile:
                v = r['verdict']; c = v['convergence']
                lines.append(f"| {r['label']} | {r['spec']['name']} | {v['status']} | {c['passing_suffix']} | {c['confirmed_step'] or '—'} |")
    lines += ['', '[Failure diagnosis and exact replay](DIAGNOSIS.md) · '
              '[Current formulation leaderboard](../formulations/README.md) · '
              '[Final live/EMA metrics, specs and full-curve artifact links](leaderboard.json) · [Artifact audit](validation.json).', '',
              'The search adapts to inspected development results. Fixed cards are recorded before each batch. '
              'No seed sweeps, extra training, metric relaxation or diagnostic controls are counted as fixes. '
              'Source archives and drivers retain every attempted architecture; runtime measurements include concurrent CPU work.', '',
              '```bash', 'python -m reports.transfer_suite.rare_focus.build',
              'python -m reports.transfer_suite.rare_focus.verify', '```', '',
              'The current verifier understands the compressed repository layout. Commands inside frozen agent handoffs refer to their original layouts. '
              'Use the reusable command above for the winning architecture.']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    print({k: report[k] for k in ('episodes', 'rare_candidates', 'sustained_rare_winners', 'cross_checks')})


if __name__ == '__main__':
    build()
