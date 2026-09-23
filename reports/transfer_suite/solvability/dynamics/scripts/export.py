"""Export immutable episode records and recompute verdicts from recorded curves."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-stress-solvability')
from benchmarks.locked_shared.observation import sustained
from benchmarks.locked_shared.baseline import write_json
from benchmarks.transfer_suite import vector_tasks

root=Path('/tmp/pr36-stress-solvability')
raw=(root/'results.json').read_bytes()
report=json.loads(raw)
originals={spec['name']:spec for spec in report['original_tasks']}
rows=[]
for index,row in enumerate(report['rows']):
 result=row['result'];spec=row['spec'];bounds=spec['thresholds']
 original=originals[row['task']]
 assert bounds==original['thresholds']
 expected={math.ceil(i*spec['steps']/24) for i in range(1,25)}
 convergence=sustained(result.get('observations',[]),bounds,expected_steps=expected,minimum=5)
 final_pass=vector_tasks.passes(result.get('live',{}),bounds)
 rows.append({'episode_index':index,'candidate':deepcopy(row['card']),
              'original_spec':deepcopy(original),'spec':deepcopy(spec),'resources':row['resources'],
              'evaluation_context':'seen-development; cadence previously inspected before this search',
              'result':deepcopy(result),
              'verdict':{'final_pass':final_pass,'sustained_pass':final_pass and convergence['confirmed_step'] is not None,
                         'convergence':convergence,'ema_pass':vector_tasks.passes(result.get('ema',{}),bounds)},
              'normalized_final_shortfall':row['shortfall']})
export={'source_commit':report['source_commit'],'protocol':report['protocol'],
        'raw_results_sha256':hashlib.sha256(raw).hexdigest(),
        'target_positive_controls':report['target_positive_controls'],
        'episodes':rows,'attempts':len(rows),
        'limitations':'Adaptive development search, one fixed seed; target-sample positives are scoring sanity, not trained witnesses.'}
write_json(root/'episodes.json',export)
task_order=[spec['name'] for spec in report['original_tasks']]
bycard={}
for row in rows:
 bycard.setdefault(row['candidate']['name'],[]).append(row)
ordered=sorted(bycard.items(),key=lambda item:(-sum(r['verdict']['sustained_pass'] for r in item[1]),
                                              -sum(r['verdict']['final_pass'] for r in item[1]),item[0]))
lines=['# Stress/dynamics solvability search','','Actual GAN training under unchanged numerical gates. Live weights determine final and sustained passes; EMA is separate. Every episode has24 fixed observations and needs a passing final suffix of at least5. Cadence has already been seen and is development data.','',
       'The short card screen used the fast-critic case. Prior30 and five resource/tuning finalists were then evaluated across all six ranking stresses and the seen cadence. Untested cells are shown explicitly.','',
       '| Card | Resource class | '+ ' | '.join(task_order)+' | Sustained / evaluated | Final / evaluated |',
       '| --- | --- | '+' | '.join(['---']*len(task_order))+' | ---: | ---: |']
for name,records in ordered:
 bytask={r['original_spec']['name']:r for r in records};cells=[]
 for task in task_order:
  row=bytask.get(task)
  if row is None:cells.append('Not run');continue
  verdict=row['verdict'];suffix=verdict['convergence']['passing_suffix']
  cells.append(('PASS' if verdict['sustained_pass'] else 'Final only' if verdict['final_pass'] else 'FAIL')+f' ({suffix}/24)')
 stable=sum(r['verdict']['sustained_pass'] for r in records);final=sum(r['verdict']['final_pass'] for r in records)
 lines.append(f"| {name} | {records[0]['resources']} | "+' | '.join(cells)+f' | {stable}/{len(records)} | {final}/{len(records)} |')
lines+=['','`PASS` means final numerical pass AND sustained final suffix. `Final only` cannot count as converged. A one-task screen result is not an all-task win.','',
        'Full episode cards, original/effective specs, rawlive/EMA curves, actions, runtime and independently recomputed verdicts: [episodes.json](episodes.json). Original serial results: [results.json](results.json); per-attempt numerical table: [README.md](README.md).','',
        'R1+R2 remains its declared formulation when cap-only knobs change. Task D-LR ratios, minibatch sizes and capacity/horizon perturbations are preserved. All metric thresholds are identical to the original task cards.','',
        f"Recorded episode wall time: {sum(r['result'].get('seconds',0) for r in rows):.2f} seconds on a shared CPU host. Source files and exact runners are archived alongside the records."]
(root/'MATRIX.md').write_text('\n'.join(lines)+'\n')
print('Exported',len(rows),'episodes; full cards',[(n,len(rs)) for n,rs in ordered if len(rs)==len(task_order)])
