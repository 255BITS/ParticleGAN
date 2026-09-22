from pathlib import Path
import json
from benchmarks.transfer_suite import image_solvability as study
rows=[]; controls={}; attempts=[]
for stage in ('stage1','stage2','cross'):
 report=json.loads(Path(f'/tmp/pr36-image-solvability-{stage}/results.json').read_text())
 rows+=report['rows']; controls.update(report['supervised'])
 attempts += [r for row in report['rows'] for r in row['results'].values()]
 attempts += list(report['supervised'].values())
diagnostics=json.loads(Path('/tmp/pr36-image-solvability-diagnostics/results.json').read_text())
attempts += [r['result'] for r in diagnostics['rows']]
assert len(attempts)==66 and all(not r.get('error') and r['convergence']['complete'] for r in attempts)
lines=['# Image GAN solvability', '',
'**One shared configuration sustains all four healthy image tasks at the original 600-step budget:** '
'residual nearest-neighbor upsampling, width16, with the original b_cap coefficient3, kappa1.25, '
'LRs0.0017, Adam(0,0.99), and32 learned particles. This is an architecture repair; '
'it does not establish that the original transpose network or a new LR controller solved the tasks. '
'The best loss-only card on the original width12 architecture sustains2/4.', '',
'All gates remain fixed: 24 observations, at least five final passing observations, live HQ≥90%, '
'and sufficient quality-qualified mass in every mode. No seed sweeps: every GAN uses seed0. '
'EMA is retained separately and never determines selection. All60 healthy GAN episodes, '
'four supervised controls, and two diagnostic episodes are preserved, including every failure.', '',
'Cells below show **sustained verdict · final modes/total · HQ**. A passing final snapshot alone '
'does not qualify. Each row uses one shared card across all four tasks; no per-task oracle union '
'is reported as a default.', '']
def cell(r):
 p,m,h=study.quality(r)
 return f'{"PASS" if p else "FAIL"} · {m}/{r["spec"]["modes"]} · {h:.1%}'
def table(selected):
 result=['| Shared card | Sustained /4 | Stripes | Bars | Blobs | Intensity |',
         '| --- | ---: | --- | --- | --- | --- |']
 for row in sorted(selected,key=study.ranking):
  result.append('| '+row['card']['name']+' | '+str(sum(study.quality(r)[0] for r in row['results'].values()))+'/4 | '+
                ' | '.join(cell(row['results'][s['name']]) for s in study.HEALTHY)+' |')
 return result
lines+=['## Original architecture and budget', '']+table([r for r in rows if r['card']['scope']=='same architecture/budget'])
lines+=['', 'R1+R2 coefficient1 fixes stripe coverage and preserves blobs, but bars HQ84.375% and intensity '
'quality still fail. Frozen particles, 10× prior LR, weaker cap and least-squares variants do not produce '
'a shared solution. All exact cards are in the machine declarations; particle-LR changes use a separate '
'prior parameter group, preserving the generator rate.', '', '## Architecture controls at 600 steps', '']
lines+=table([r for r in rows if r['card']['scope'] not in ('same architecture/budget','budget doubled')])
lines+=['', 'Residual16 with cap3 confirms stripes/bars/blobs/intensity at steps225/550/550/575. '
'Final HQ is100%/93.75%/100%/100%. R1+R2 is unnecessary: the residual16 R1(.1) card also passes4/4, '
'but intensity HQ is lower. The cross-domain cap10 card falls to3/4 because bars HQ falls to81.25%. '
'It is not a uniformly better setting.', '',
'Width controls separate architecture from capacity. Residual12 uses the original D width and fewer G '
'parameters, yet fixes stripes and intensity. Transpose16 has the same D as residual16 and more G '
'parameters, but passes0/4. Residual architecture plus adequate width is the supported recipe here; '
'greater width alone is not enough.', '',
'| Architecture | G parameters | D parameters |', '| --- | ---: | ---: |',
'| transpose12 | 5173 | 2833 |', '| residual12 | 3157 | 2833 |', '| transpose16 | 8945 | 4929 |',
'| residual16 | 5361 | 4929 |', '| transpose24 | 19561 | 10849 |', '',
'## Extra-budget control', '']+table([r for r in rows if r['card']['scope']=='budget doubled'])
lines+=['', 'Doubling the original transpose baseline to1200 steps does not repair the shared failure. '
'Blobs ends with4/4 and HQ100%, but lacks the required final passing suffix and correctly remainsFAIL.', '',
'## Expressivity and diagnostic controls', '',
'The supervised witnesses use balanced latent-index/template labels and MSE on the exact original '
'G/prior architecture for600 steps. They are **not GAN successes**, do not enter selection, and '
'only establish representability when they converge.', '',
'| Supervised control | Result | Confirmed step |','| --- | --- | ---: |']
for name,r in controls.items(): lines.append(f'| {name} | {cell(r)} | {r["convergence"]["confirmed_step"] or "—"} |')
lines+=['', 'All three originally failing healthy targets are representable with the original architecture '
'and budget under direct supervision. The supervised blobs attempt fails and is retained; ordinary '
'GAN training already demonstrates blobs solvability.', '',
'| Diagnostic (zero selection weight) | Shared card | Result |','| --- | --- | --- |']
for row in diagnostics['rows']: lines.append(f'| img_bars8 | {row["card"]["name"]} | {cell(row["result"])} |')
lines+=['', 'The denser eight-mode diagnostic remains unsolved and does not veto the healthy-task result. '
'The information-poor and spatially uniform architecture diagnostics keep their declared nonblocking '
'importance; their constraints were not removed to manufacture a pass.', '',
'## Provenance and limits', '',
f'Total recorded episode wall time: {sum(r["seconds"] for r in attempts):.1f}s across66 complete episodes; no numerical errors. '
'These are single CPU observations under shared system load, not a speed estimate. Three analytical '
'and parity tests passed: scoped loss/prior/Adam restoration, prior-only LR scaling and frozen-prior '
'behavior, forbidden gate changes, and exact baseline numerical parity.', '',
'The four baseline reruns exactly reproduce every one of24 live/EMA checkpoints and every loss '
'checkpoint in the parent transfer study. Residual16 bars likewise exactly reproduces the formerly '
'reserved bars setup; it is the same numerical case, **not an independent transfer success**. '
'All former reserved cases are now seen development data. No fresh holdout, natural-image dataset '
'or production default was evaluated or changed.', '',
'Run the declared initial search:', '',
'```bash', 'python -u -m benchmarks.transfer_suite.image_solvability --controls \\',
'  --output /tmp/pr36-image-solvability-stage1', '```', '',
'Additional stages use `--cards <JSON>` with exact declared changes. The handoff bundle '
'`/tmp/pr36-image-solvability-artifacts/` contains all stages, logs, deterministic JSON.gz '
'archives, original-byte SHA256s, exact source bundles and the calibration scripts. '
'The parent PR report retains that bundle. `source.tar.gz` includes the exact scoped shim '
'used to expose loss/prior options without changing the original task runner.', '']
text='\n'.join(lines)
for a,b in [('width16','width 16'),('width12','width 12'),('coefficient3','coefficient 3'),('coefficient1','coefficient 1'),('kappa1.25','kappa 1.25'),('LRs0.0017','LRs 0.0017'),('Adam(0,0.99)','Adam (0,0.99)'),('and32','and 32'),('sustains2/4','sustains 2/4'),('seed0','seed 0'),('All60','All 60'),('at steps225','at steps 225'),('is100%','is 100%'),('passes4/4','passes 4/4'),('to3/4','to 3/4'),('to81.25%','to 81.25%'),('passes0/4','passes 0/4'),('to1200','to 1200'),('with4/4','with 4/4'),('HQ100%','HQ 100%'),('remainsFAIL','remains FAIL'),('for600','for 600'),('across66','across 66'),('of24','of 24'),('HQ84.375%','HQ 84.375%')]: text=text.replace(a,b)
Path('benchmarks/transfer_suite/image_solvability.md').write_text(text)
Path('/tmp/pr36-image-solvability-combined/README.md').write_text(text)
print('Wrote final report for66 episodes')
