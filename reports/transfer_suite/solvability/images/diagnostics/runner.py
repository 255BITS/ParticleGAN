import gzip
import hashlib
import json
from pathlib import Path
from benchmarks.transfer_suite import image_solvability as study

output=Path('/tmp/pr36-image-solvability-diagnostics')
output.mkdir(exist_ok=False)
source=Path(__file__).read_bytes()
(output/'runner.py').write_bytes(source)
cards=[dict(name='residual16',scope='architecture changed',changes=dict(architecture='residual_upsample',width=16)),
       dict(name='residual16_cap10',scope='architecture changed; cross-domain cap10',changes=dict(architecture='residual_upsample',width=16,penalty_coeff=10.))]
spec=next(s for s in study.host.TASKS if s['name']=='img_bars8')
protocol=study.snapshot(output)
declaration=dict(task=spec,cards=cards,protocol=protocol,script_sha256=hashlib.sha256(source).hexdigest(),
                 selection_weight=0,importance='Diagnostic only; eight-mode stress cannot veto selection.')
study.write_json(output/'declaration.json',declaration)
rows=[]
for card in cards:
 print('START diagnostic',card['name'],spec['name'],flush=True)
 result=study.episode(spec,card)
 raw=(json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode()
 filename=card['name']+'__'+spec['name']+'.json.gz'
 (output/filename).write_bytes(gzip.compress(raw,mtime=0))
 rows.append(dict(card=card,result={k:v for k,v in result.items() if k not in ['actions','protocol']},
                  artifact=filename,original_sha256=hashlib.sha256(raw).hexdigest()))
 study.write_json(output/'results.json',dict(declaration=declaration,rows=rows))
 print(json.dumps(dict(card=card['name'],quality=study.quality(result),seconds=result['seconds'],error=result.get('error'))),flush=True)
