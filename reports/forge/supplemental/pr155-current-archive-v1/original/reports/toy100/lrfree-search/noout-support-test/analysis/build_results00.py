"""Build section 00 of RESULTS.md from analysis/results00.template.md (prose) and the run directories / pool ledger (tables) and insert it above the old section 0 of RESULTS.md
(between the markers <!-- sec00:begin --> and <!-- sec00:end -->; idempotent). usage: python build_results00.py [--write]"""
import json, re, subprocess, os, sys, shutil
W = '/ml2/hypergan/gan-attempts/noout-20260928'; os.chdir(W)
T = open('analysis/results00.template.md').read()
def sh(*a):
    return subprocess.run(list(a), capture_output=True, text=True).stdout.strip()
def rows(labels): return sh('python3', 'analysis/make_tables.py', *labels).split('\n')
HEAD = '| run | grid100 | rotated100 | staggered100 | live |\n|---|---|---|---|---:|\n'
def table(pairs):
    out = []
    for r, (l, n) in zip(rows([l for l, _ in pairs]), pairs): out.append(r.replace(f'| {l} |', f'| {n} |', 1))
    return HEAD + '\n'.join(out)
NATIVE = ('### 00.0 Native gates, 7k (single deterministic seed per cell; cell = verdict, last-5 worst live precision, last-5 worst centre RMS in sigma, (holdout precision / holdout centre), worst mode covariance eigenvalue ratio of the last 5 checks (limit 1.70); frozen limits: live precision >= .97, mode masses in [.005, .02], eigenvalue ratios in [.4, 1.7], radial median ratios in [.65, 1.4], mass TV <= .10; accuracy on the final five 20k clouds and the 100k holdout: mass TV <= .06, centre RMS <= .20, |cov trace bias| <= .10, radial KS <= .04)\n'
          + table([('E22a', '**E22** (confirmation of E19a)'), ('E19a', 'E19a (rank cap; the evidence base of E22)'), ('E20a', 'E20a (p-weighted parents, persistence, duplicate guard)'), ('E21a', 'E21a (E20 + distance limit on parents)'), ('E18a', 'E18a (+ feature scale)'),
                   ('E17a', 'E17a (ball parents, raw features)'), ('E17b', 'E17b = E17a without the row-evidence gate'), ('E16a', 'E16a (k-nearest parents)'), ('E16b', 'E16b = E16a without the gate'), ('E15a', 'E15a (uniform parents)'), ('E14s', 'E14s (before this pass)')]))
S4 = ('### 00.2 Robustness to the base learning rate (S4), same seed, only `overrides["lr"]` changes (x.75 = .0031875, x1.33 = .0056525; the E14s rows are the E13s runs, byte-identical)\n'
      + table([('E13slr075', 'E14s lr x.75'), ('E17lr075', 'E17a lr x.75'), ('E19lr075', '**E19a lr x.75**'), ('E20lr075', 'E20a lr x.75'), ('E17blr075', 'E17b (no gate) lr x.75'),
               ('E13slr133', 'E14s lr x1.33'), ('E17lr133', 'E17a lr x1.33'), ('E19lr133', '**E19a lr x1.33**'), ('E20lr133', 'E20a lr x1.33')])
      + '\nE14s 5/6 (x1.33 grid fails the shape limit, 1.83); E17a, E19a, E20a 6/6; E17b (no gate) 5/6. The thinnest stress cells: E19a lr x1.33 grid (centre .198 against .20), E20a lr x1.33 grid (eigenvalue ratio 1.66), E17a lr x.75 staggered (1.67).\n')
HOR = ('### 00.3 Sustainability (the recipe carries no horizon: the 14k run is a prefix of the 28k run)\n'
       + table([('E14s14k', 'E14s 14k'), ('E14s28k', 'E14s 28k'), ('E19a14k', '**E19a 14k**'), ('E19a28k', '**E19a 28k**'), ('E20a14k', 'E20a 14k')])
       + '\nE14s rotated precision drifted .9728 (7k) -> .9789 (14k) -> .9747 (28k) and its grid/staggered centres grew (.138 -> .168, .155 -> .175); E19a holds precision .983-.989 with centres .113-.149 at 14k and 28k (all six runs pass; rotated 28k .9888 / .113, holdout .070); the E20a 14k run passes too (staggered centre .181, holdout .150: closest to the limit).\n')
def suite():
    t = sh('python3', 'suite_table.py').split('\n')
    head = [c.strip() for c in t[0].strip('|').split('|')]; keep = [0] + [head.index(n) for n in ('E14s', 'E17b', 'E17a', 'E19a', 'E20a') if n in head]
    out = []
    for i, line in enumerate(t):
        cs = [c.strip() for c in line.strip('|').split('|')]
        out.append('| ' + ' | '.join(cs[j] for j in keep) + ' |')
    return ('### 00.4 Portability suite (13 gates through the shared pool, same config unchanged; `suite_table.py`)\n' + '\n'.join(out) + '\n'
            'E19a passes all 13: img_bars4 (0/24 for every earlier candidate, including the base), img_intensity2 and vector_unequal_mass pass for the first time in this lineage, with the feature scale as the only relevant change (the support test never acted on the N <= 256 tables: `iso_acted` 0, e.g. vector_unequal_mass 0 of 600 evaluations); ring_shift and stationary (N = 20,000) are the two tasks on which the support test acts (E19a: 11,484 and 919 re-draws). The gates are advisory (the user\'s rule); the earlier toy failures were real failures, not bad tests (an ideal 256-row table passes vector_unequal_mass 99.8% of the time, `scratchpad/ideal_table_unequal_mass.py`), and are fixed here.\n')
VARIANTS = ('#### E20/E21 on ring_shift (the reason they are not the candidate)\n'
            'E20a passes the natives (table above) and 12 of the 13 gates but **fails ring_shift 239/460** (three of its suite tasks first died of a GPU out-of-memory at start, another job had filled the card, and were rerun: PASS) (final streak 3: the noisy hq hovers at .89-.92 against the .90 limit; E19a .96) with 58% more ordinary birth-death moves (46,981 vs 29,706) and a lower clean hq (.936 vs .983). E21a (E20 + the distance limit on the parents, built on the hypothesis that a rank cap alone draws parents from other modes after the shift) passes the natives with the best centre errors of the pass (.142 / .142 / .156; holdout .106 / .108 / .134) and stationary (706/750) but also fails ring_shift (292/460, streak 0), differently: an abrupt collapse (hq .95 -> .2 within 50 steps at step ~4140, partial recovery at the end); E19a and E14s have the same kind of collapse earlier (~3440 and ~3600) and recover in time. So ring_shift is flaky in this lineage (the timing of the episode decides the streak-of-5 verdict) and I cannot say whether persistence, the p-weights or chance moved the outcome; with one seed this is where a multi-seed run is needed.\n')
PROBES = open('analysis/probes00.txt').read().strip() if os.path.exists('analysis/probes00.txt') else '(pending)'
FORECAST = ('### 00.7 Forecast scoreboard for this pass (pass/fail forecasts written to `forecast-E15..E22*.json` before the runs they concern; Brier = mean squared error of the stated probability, .25 = coin flip)\n' + sh('python3', 'analysis/forecast_score.py') + '\n'
            'Value forecasts: E15a rotated precision .982 [.976, .986] -> .9799 inside; E15a rotated centre .13 [.10, .17] -> .245/.217 outside (the failure the forecast missed); E16a mass TV .038 [.030, .052] -> live .026-.043 inside; E17 rows re-drawn on rotated 6,000 [1,500, 13,000] -> 15,859 outside (the churn was underestimated); E20 rotated re-draws 4,500 [1,500, 11,000] -> 9,106 inside. The forecasts were over-confident about E15 and E17b, under-confident about the S4 cells.\n')
T = (T.replace('@@NATIVE@@', NATIVE).replace('@@S4@@', S4).replace('@@HORIZON@@', HOR).replace('@@SUITE@@', suite()).replace('@@VARIANTS@@', VARIANTS).replace('@@PROBES@@', PROBES).replace('@@FORECAST@@', FORECAST))
if '--write' in sys.argv:
    R = open('RESULTS.md').read()
    if not os.path.exists('RESULTS.E14s-only.bak'): shutil.copy('RESULTS.md', 'RESULTS.E14s-only.bak')
    block = '<!-- sec00:begin -->\n' + T.rstrip('\n') + '\n<!-- sec00:end -->\n\n'
    if '<!-- sec00:begin -->' in R:
        R = re.sub(r'<!-- sec00:begin -->.*?<!-- sec00:end -->\n\n', lambda m: block, R, flags=re.S)
    else:
        i = R.index('## 0. Current candidate: E14s')
        R = R[:i] + block + R[i:].replace('## 0. Current candidate: E14s', '## 0. Previous candidate: E14s', 1)
    open('RESULTS.md', 'w').write(R); print('written', len(T.split('\n')), 'lines')
else:
    print(T)
