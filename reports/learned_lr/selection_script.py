from pathlib import Path
import hashlib,json
from benchmarks.learned_lr.study import write_json
root=Path('/tmp/pr36-learned-lr-study')
report=json.loads((root/'training.json').read_text())
if len(report['attempts']) != 32:
    raise RuntimeError('fit not complete')
overall=min(report['attempts'],key=lambda r:r['objective'])
adaptive=min((r for r in report['attempts'] if any(w != 0 for row in r['policy']['weights'] for w in row[2:])),key=lambda r:r['objective'])
meta={'overall_selected_attempt':overall['name'],'overall_training_objective':overall['objective'],
      'adaptive_selected_attempt':adaptive['name'],'adaptive_training_objective':adaptive['objective'],
      'adaptive_selection_role':'overall_winner' if adaptive['name']==overall['name'] else 'adaptive_challenger_rejected_by_overall_selection',
      'selection_rule':'Both choices use training objective only; best policy with at least one nonzero feedback coefficient is exported even if constant wins. No held-out metrics used.',
      'selection_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
policy={**adaptive['policy'],**meta,'selected_attempt':adaptive['name'],'training_objective':adaptive['objective'],
        'training_tasks':report['protocol']['training_tasks'],'source_sha256':report['protocol']['source_sha256']}
write_json(root/'adaptive_policy.json',policy)
selected=json.loads((root/'policy.json').read_text())
selected.update(meta)
write_json(root/'policy.json',selected)
meta['policy_sha256']=hashlib.sha256((root/'policy.json').read_bytes()).hexdigest()
meta['adaptive_policy_sha256']=hashlib.sha256((root/'adaptive_policy.json').read_bytes()).hexdigest()
write_json(root/'selection.json',meta)
report['frozen_selection']={**meta,'policy_file':'policy.json','adaptive_policy_file':'adaptive_policy.json'}
write_json(root/'training.json',report)
print(json.dumps(meta,indent=2),flush=True)
