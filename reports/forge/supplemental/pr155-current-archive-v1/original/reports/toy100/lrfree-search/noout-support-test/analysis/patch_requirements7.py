"""Patch REQUIREMENTS.md from draft 6 to draft 7 (idempotent by markers in the text). usage: python patch_requirements7.py"""
p = '/ml2/hypergan/lrfree-20260926/REQUIREMENTS.md'; s = open(p).read()
if 'draft 7' in s.split('\n')[0]:
    print('already draft 7'); raise SystemExit
s = s.replace('(draft 6, 2026-09-29)', '(draft 7, 2026-09-29)', 1)
old = 'and the current candidate is E14s (RESULTS 0). Read section 7'
new = ('and the current candidate is E14s (RESULTS 0). Draft 7 (2026-09-29, after the user\'s answers to the E14s report, two more independent reviews and codex\'s feature-gauge audit) records what solved the thin rotated100 margin on one seed: a support test in the critic\'s feature space that re-draws table rows with no real support (E15-E19, E22; rotated precision margin +.0028 -> +.0151), a parametrisation-invariant feature scale (E18: every critic feature divided by its std on the reference half of the reservoir; the 13-gate suite goes from 10/13 to 13/13), and the current candidate E22 (= E19a + a duplicate guard; RESULTS 00). Read section 7')
assert old in s; s = s.replace(old, new, 1)
# section 3: parametrisation invariance
old = '## 4. Acceptance tests ("solved")'
new = ('- **Parametrisation invariance of feature-space statistics (E18; codex\'s audit `reports/toy100/lrfree-search/e17-feature-gauge-review`, commit 517c5e6f).** The critic function does not fix the scale (or, for a head fed by a linear map, the basis) of the features its head reads, so a Euclidean distance in the raw feature space depends on an arbitrary parametrisation: the same critic in two parametrisations gave different birth-death flags and moves. A statistic computed in the critic\'s feature space must be invariant to the exact symmetry group of the critic (for ReLU-type units: a positive rescaling of any hidden unit) or say which symmetry it is not invariant to. E18 divides every feature by its standard deviation on the reference half of the real reservoir (a function of the reference half only, so split-conformal validity is untouched; units without spread are dropped). Whitening is invariant to more but amplifies low-variance directions and makes the pseudo-inverse floor gauge dependent; on the native critic the raw metric loses recall (.91 -> .61) only at a 16x random rescaling, but on the toy critics the raw metric is dominated by a few units (the 8 largest of 128 features carry 42% of the squared distance on img_bars4) and standardisation alone takes the suite from 10/13 to 13/13.\n\n## 4. Acceptance tests ("solved")')
assert old in s; s = s.replace(old, new, 1)
# S3
old = '**E14s (`E14s-gates`, `E14s-va`) passes 10 of 13** (img_bars4, img_intensity2 5/24 and vector_unequal_mass fail; the last two are chaotic single-seed flips on tiny tables, RESULTS 0.4).'
new = ('**E14s (`E14s-gates`, `E14s-va`) passes 10 of 13** (img_bars4, img_intensity2 5/24 and vector_unequal_mass fail; the last two are chaotic single-seed flips on tiny tables, RESULTS 0.4); **E19a (`E19a-gates`) passes 13 of 13** (img_bars4 17/24, img_intensity2 16/24 and vector_unequal_mass 20/24 pass for the first time in this lineage; the support test is inert on those tables, the feature scale is the only relevant change; those three were real failures, not bad tests: an ideal 256-row table passes vector_unequal_mass 99.8% of the time); '
       'E20a (a variant with p-weighted parents) fails ring_shift (239/460, hq 0.89-0.92 against 0.90) and E21a fails it too (a collapse episode at step ~4140): the shift gate is flaky in this lineage (RESULTS 00.4).')
assert old in s; s = s.replace(old, new, 1)
# S4
old = '**E14s x.75 3/3, x1.33 2/3** (grid fails one mode\'s covariance eigenvalue ratio, 1.72-1.83 against the live limit 1.70; E11\'s 6/6 came from the defective code).'
new = '**E14s x.75 3/3, x1.33 2/3** (grid fails one mode\'s covariance eigenvalue ratio, 1.72-1.83 against the live limit 1.70; E11\'s 6/6 came from the defective code); **E17a 6/6, E19a 6/6** (thinnest cell: x1.33 grid centre .198 against .20), E20a 6/6.'
assert old in s; s = s.replace(old, new, 1)
open(p, 'w').write(s); print('patched sections: title/history, section 3, S3, S4')

# ---- part 2 (run after part 1; separate guard): section 7 table rows and section 9 open directions
s = open(p).read()
if 'E19a = E14s + support test' not in s:
    old = '| **E14s = E13 + feature space = all learned scalar heads** (`pkg-E14` + `overrides-E13-scaled.json`)'
    i = s.index(old); j = s.index('\n', i)
    row14 = s[i:j].replace('**current candidate**; ', 'superseded by E22; ', 1)
    rows = (row14 + '\n'
            '| **E19a = E14s + support test (M5) + feature scale (M6); E22 = E19a + duplicate guard** (`pkg-E22` + `overrides-E22.json`; bit-identical to E19a while the guard does not fire) | **P .9834 / .157 (.129); P .9851 / .152 (.129); P .9822 / .150 (.117)** (E22a confirmations: RESULTS 00.0) | **14k 3/3** (grid .9843 / .134, rot .9872 / .114, stag .9867 / .149), 28k grid P (.138), stag P (.127), rot see RESULTS 00.3 | clean in the executed path (the reservoir enters only through the critic\'s features; z-space geometry and p-values for the parents); declared inputs unchanged; flagged A2 (Q reused three more times, ball factor 2 chosen on the natives, older constants), A5 (six mechanisms; ablations RESULTS 00.1), A7, A1 | **current candidate**; S2 FULL (x_hat .121 / .121 / .109, rotated precision +.0151); S4 6/6 (x1.33 grid centre .198); **suite 13/13**; independent statistical and code reviews of E17: no blocking finding, majors handled or listed (RESULTS 00.6); open: iid reservoir, churn (14.9k re-draws on rotated), inert below N = 800 |\n'
            '| E20a/E21a = E19 with p-weighted parents (no ball factor), persistence of two evaluations, duplicate guard (E21a adds a distance limit) | E20a P .9829 / .145; P .9850 / .146; P .9809 / .169; E21a natives: RESULTS 00.0 | E20a 14k in progress | as E19a, ball factor removed in E20 | variants, not adopted: churn -40% but **ring_shift fails** (E20a 239/460, E21a collapse at step ~4140); E20a S4 6/6 |\n')
    s = s[:i] + rows.rstrip('\n') + s[j:]
    old8 = '8. **Multi-seed confirmation.**'
    assert old8 in s
    i8 = s.index(old8); j8 = s.index('\n', i8)
    new8 = ('8. **Multi-seed confirmation (the user\'s condition, "solve the thin margin first", is met on one seed: rotated +.0151, S2 FULL).** Every number here is one deterministic seed; three seeds of the three native gates for E22 with E14s as the control (18 runs, about one GPU-hour on a quiet machine) would separate luck from mechanism; the working rule against seed sweeps has to be lifted by the user for this one confirmation.\n'
            '9. **Evidence accumulation for the support test.** The test acts on one evaluation\'s p-value; a component absent from the reservoir for a few evaluations loses its rows (probe: the 2% component withheld for 100 steps drops to a mass ratio of .10 within 50 steps, in E14s as well as E19a/E20a) and finite pools with near-duplicates are not guarded. Replace the single-evaluation flag by a sequential test over the last evaluations (an e-value or the ordinary evidence\'s meander), or a reservoir that is longer than one turnover.\n'
            '10. **Critic-side negatives instead of teleports.** The strays hold up a funnel in the critic (69-85% of random points feel an inward force out to 14 sigma in E14s, 54-72% in E19a, ~50% in E15a); removing them leaks the bulk into the shell (~2 rows/step, 14.9k re-draws per run on rotated). Feed the culled rows\' positions to the critic as extra fake samples for a while (a replay of recently removed fakes) and see whether the churn falls without losing the precision.\n'
            '11. **A calibration reservoir larger than the table** so that the support test has power below N = 800 (BH needs at least 40 tied rows at once; the guard allows 5% of N): the smallest p-value must be able to reach Q / N.\n'
            '12. **ring_shift flakiness** (abrupt collapse episodes in every variant): find what triggers them (a tester release of the network group at ~3577 in E19a/E21a, or the critic) before trusting the shift gate as evidence for or against any parent rule.\n'
            '13. **The mode-shape metric** (worst mode\'s covariance eigenvalue ratio, limit 1.70, is 1.41-1.66 in the natives and stress cells): what elongates a mode late in a run; an amplitude-aware annealing (item 2) is the natural fix.')
    s = s[:i8] + new8 + s[j8:]
    open(p, 'w').write(s); print('patched section 7 rows and section 9 items 8-13')
