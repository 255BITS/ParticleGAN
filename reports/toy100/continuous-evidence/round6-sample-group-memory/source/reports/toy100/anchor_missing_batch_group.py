"""One conditional minibatch event on the SAME ring target, not a target shift.

Condition the next existing data stream draw on containing no component0.
That event has probability (7/8)^128 under the fixed eight-component sampler.
Compare the current-bank anchor map against its ordinary next-bank control.
True labels define this diagnostic event and grade only, never the map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_free1200 import initial_support
from reports.toy100.sample_group_anchor import mst_groups,output_mm_step
from reports.toy100.coverage_fixed_eval import fixed_draw,score_support
from reports.toy100.pr84_critic_refinement_capture import _sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    saved=torch.load(args.state,weights_only=True,map_location='cpu')
    points=initial_support(saved);means=mode_hold.ring_means()
    index,noise=fixed_draw(2401,points)
    rows={}
    for name,source_means in [('ordinary',means),('component0_absent',means[1:])]:
        stream=torch.Generator().set_state(saved['rng']['data'])
        real=mode_hold.sample_ring(source_means,128,mode_hold.SIGMA,stream)
        centers,grouping=mst_groups(real)
        update=output_mm_step(points.double(),centers)
        target=torch.tensor(update['target'],dtype=points.dtype)
        rows[name]=dict(grouping=grouping,centers=centers.tolist(),update=update,
            real128=real.tolist(),grade=score_support(target,index,noise,means))
    report=dict(scope='free-output conditional-minibatch diagnostic; fixed target unchanged; not a native neural run',
        input_state_sha256=_sha(saved),input_file_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        event='all128 latent component labels differ from0',event_probability=(7/8)**128,
        initial_grade=score_support(points,index,noise,means),points=points.tolist(),branches=rows)
    args.output.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ['reports/toy100/anchor_missing_batch_group.py','reports/toy100/sample_anchor_free1200.py',
        'reports/toy100/sample_group_anchor.py','reports/toy100/coverage_fixed_eval.py',
        'reports/toy100/pr84_early_geometry.py','benchmarks/locked_shared/mode_hold.py',
        'benchmarks/locked_shared/mlp.py']:
        raw=(ROOT/name).read_bytes();dest=args.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        sources[name]=hashlib.sha256(raw).hexdigest()
    report['source']=sources
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event_probability=report['event_probability'],initial=report['initial_grade'],
        branches={k:dict(groups=len(v['centers']),grade=v['grade'],cost_before=v['update']['before'],cost_after=v['update']['after']) for k,v in rows.items()}),indent=2))


if __name__=='__main__':main()
