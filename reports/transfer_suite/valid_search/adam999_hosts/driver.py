"""Cross-host check of the already screened b_cap3/Adam(0,.999) recipe."""
from dataclasses import asdict, replace
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import torch

from benchmarks.smart_descent import evaluate, study
from benchmarks.transfer_suite import image_tasks, suite, vector_tasks
from benchmarks.transfer_suite.protocol import required_tasks, test_verdict

OUTPUT = Path('/tmp/pr36-valid-adam999')


@contextmanager
def adam_betas(betas, records):
    original = torch.optim.Adam.__init__
    def initialize(self, params, *args, **kwargs):
        args = list(args)
        if len(args) >= 2:
            args[1] = tuple(betas)
        else:
            kwargs['betas'] = tuple(betas)
        original(self, params, *args, **kwargs)
        records.append([dict(lr=g['lr'], betas=list(g['betas']), tensors=len(g['params'])) for g in self.param_groups])
    with patch.object(torch.optim.Adam, '__init__', initialize):
        yield


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/'episodes').mkdir()
    torch.set_num_threads(1)
    protocol = suite.snapshot(OUTPUT)
    source = Path(__file__).read_bytes()
    (OUTPUT/'driver.py').write_bytes(source)
    protocol['driver_sha256'] = hashlib.sha256(source).hexdigest()
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    candidate = replace(study.BASE, name='bcap3_adam999')
    policy = vector_tasks.fixed_policy()
    rows=[]
    jobs=[('required',s) for s in required_tasks()]
    jobs += [('image',dict(s, architecture='residual_upsample',width=16,adam_betas=[0.,.999])) for s in image_tasks.TASKS if s['tier']=='ranking']
    for runner,spec in jobs:
        suite.verify_source(protocol)
        print('START',runner,spec['name'],flush=True)
        optimizers=[]
        if runner=='required':
            with patch.object(study,'BASE',candidate),adam_betas((0.,.999),optimizers):
                result=evaluate.fixed_toy(spec['name'],policy)
        else:
            result=image_tasks.run_episode(spec,policy,fixed=True)
        verdict=test_verdict(spec,result)
        row=dict(candidate=asdict(candidate),adam_betas=[0.,.999],runner=runner,spec=spec,policy=policy,
                 optimizer_specs=optimizers,verdict=verdict,result=result)
        raw=(json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode()
        filename=f'episodes/bcap3_adam999__{spec["name"]}.json.gz'
        (OUTPUT/filename).write_bytes(gzip.compress(raw,mtime=0))
        rows.append({k:v for k,v in row.items() if k!='result'}|dict(artifact=filename,uncompressed_sha256=hashlib.sha256(raw).hexdigest(),live=result.get('live'),ema=result.get('ema'),seconds=result['seconds']))
        (OUTPUT/'index.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
        lines=['# b_cap3 / Adam beta2=.999 cross-host verification','',
               'Only Adam betas change from the declared host recipe; image architecture is the established residual16. Seed0, complete24 live checks, final5 required; EMA separate. Host base LRs, supports and budgets are retained.','',
               '| Host | Sustained live | Confirmed step | Final failing metrics |','| --- | --- | ---: | --- |']
        for r in rows:
            fails=', '.join(m['metric'] for m in r['verdict'].get('metrics',[]) if m['status']!='PASS') or '—'
            lines.append(f"| {r['spec']['name']} | {r['verdict']['status']} | {r['verdict'].get('convergence',{}).get('confirmed_step')} | {fails} |")
        (OUTPUT/'README.md').write_text('\n'.join(lines)+'\n')
        print('DONE',spec['name'],verdict['status'],result.get('live'),result.get('error'),flush=True)
    suite.verify_source(protocol)


if __name__=='__main__':
    main()
