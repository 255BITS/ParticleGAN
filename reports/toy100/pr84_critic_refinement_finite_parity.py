"""Exact finite-path comparison on all44 previously passing refinement steps."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_prediction_state_filter as replay
from reports.toy100.pr84_critic_refinement_finite import METHOD, pr84_critic_refinement_finite
from reports.toy100.pr84_critic_refinement_probe import untimed

CAPTURE_SHA = '37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47'
BRANCHES = ((1324,1335),(1380,1395),(1530,1545))


def project(actual, template):
    if isinstance(template, dict):
        return {key: project(actual[key], item) for key, item in template.items()}
    if isinstance(template, list):
        if len(actual) != len(template):
            raise RuntimeError('changed finite-path list length')
        return [project(a,b) for a,b in zip(actual,template)]
    return actual


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--states',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(args.states.read_bytes()).hexdigest()!=CAPTURE_SHA:
        raise RuntimeError('wrong original saved-state bank')
    reference=ROOT/'reports/toy100/continuous-evidence/pr84-critic-refinement-filter'
    old=json.loads(gzip.decompress((reference/'declaration.json.gz').read_bytes()))
    names=set(old['source']) | {'reports/toy100/pr84_critic_refinement_finite.py',
        'reports/toy100/pr84_critic_refinement_cold.py',
        'reports/toy100/pr84_critic_refinement_finite_parity.py',
        'reports/toy100/pr84_critic_refinement_probe.py'}
    for name,digest in old['source'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'original filter source changed: {name}')
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}
    args.output.mkdir(parents=True,exist_ok=False)
    source_dir=args.output/'source';source_dir.mkdir()
    for name in source:
        p=source_dir/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    declaration=dict(method=METHOD,source=source,capture_sha256=CAPTURE_SHA,branches=BRANCHES,
        scope='finite-path full-state equivalence, not a new cold or long-hold result',shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    states=torch.load(args.states,weights_only=True);torch.set_num_threads(1)
    config=json.loads((ROOT/'configs/toy100/constraints_simple_regularization.json').read_text())
    @contextmanager
    def selected(*,task,prediction):
        with pr84_critic_refinement_finite(task=task,refinement=prediction) as value:
            yield value
    rows=[]
    with patch.object(replay.prediction_module,'pr84_opponent_prediction',selected):
        for start,end in BRANCHES:
            expected=json.loads(gzip.decompress((reference/f'refinement-{start}-{end}.json.gz').read_bytes()))
            actual,_=replay.run_local(config,states[start]['pre_step'],start=start,end=end,
                                      opponent='predicted',source_dir=source_dir)
            for key in ('points','accepted_states','final_state_sha256','moment_steps','rates','noise','rng_final_sha256'):
                if actual[key]!=expected[key]:raise RuntimeError(f'finite-path {key} differs at {start}')
            a,b=untimed(actual['dynamics']['records']),untimed(expected['dynamics']['records'])
            if project(a,b)!=b:raise RuntimeError(f'finite-path original fit/update records differ at {start}')
            if actual['dynamics']['later_nonfinite_rejections']!=0:
                raise RuntimeError('unexpected invalid trial in an originally finite branch')
            (args.output/f'guarded-{start}-{end}.json').write_text(json.dumps(actual,allow_nan=False)+'\n')
            row=dict(start=start,end=end,steps=end-start+1,all_state_hashes_exact=True,
                all_supports_and_metrics_exact=True,original_records_exact=True,nonfinite_rejections=0)
            rows.append(row);print(json.dumps(dict(event='BRANCH_PARITY',**row)),flush=True)
    result=dict(status='PASS_FINITE_PATH_PARITY',method=METHOD,source_hashes=source,
        steps=sum(r['steps']for r in rows),branches=rows,shared_gate_eligible=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='DONE',**result)),flush=True)

if __name__=='__main__':main()
