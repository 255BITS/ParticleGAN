"""Apply the predeclared screen rule; no numeric candidate changes."""
from pathlib import Path
import json
from benchmarks.transfer_suite.compare_defaults import plan
ROOT=Path('/tmp/pr38-shared-ratio-search')
read=lambda name:json.loads((ROOT/name).read_text())
def write(name,value):(ROOT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
study=read('study_plan.json');screen=study['screen'];rows=read('screen/index.json')['records']
assert len(rows)==42
ranks=[]
for card in screen['candidates']:
 found=[r for r in rows if r['recipe']['name']==card['name']]
 assert len(found)==7 and {r['spec']['name'] for r in found}==set(screen['tasks'])
 key=(-sum(r['verdict']['passed'] for r in found),sum(r['verdict']['shortfall'] for r in found)/7,sum(r['verdict']['confirmation_fraction'] for r in found)/7,card['name'])
 ranks.append(dict(card=card,key=key,passed=-key[0],final_shortfall=key[1]))
ranks.sort(key=lambda row:row['key']);selected=[row['card'] for row in ranks[:2]]
remaining=[j['spec']['name'] for j in plan() if j['spec']['name'] not in screen['tasks']]
assert len(remaining)==12
write('selection.json',dict(rule=study['selection'],ranking=ranks,selected=[c['name'] for c in selected]))
write('completion_plan.json',dict(candidates=selected,tasks=remaining,purpose='Complete the two predeclared screen winners without changing any recipe setting; twelve remaining tests, no duplicates.'))
print(json.dumps(dict(selected=[c['name'] for c in selected],ranking=[dict(name=r['card']['name'],passed=r['passed'],shortfall=r['final_shortfall']) for r in ranks]),indent=2))
