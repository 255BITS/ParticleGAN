"""Public Forge submission/drain without full-qualification publication callbacks."""
from pathlib import Path
import argparse, json, os, sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.planning import resolve_idea, plan_summary
from experiments.forge.queue import Queue, drain
from experiments.forge.contracts import atomic_json

STUDIES={
 'hydraulic-secant-deformation-v2':'hydraulic-deformation-candidate-round2-v1',
 'bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36':'hydraulic-deformation-control-round2-v1'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['plan','enqueue','drain'])
    parser.add_argument('--queue-root',type=Path,required=True)
    parser.add_argument('--progress',type=Path)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    os.environ['PARTICLEGAN_FORGE_QUEUE']=str(args.queue_root)
    queue=Queue(args.queue_root,report_root=ROOT/'reports/forge',on_completion=None)
    if args.phase=='drain':
        drain(queue,[args.gpu],workers_per_gpu=1,allow_sharing=True,watch=False)
        print(json.dumps({'phase':'drained','campaign':queue.inspect()['campaigns']['hydraulic-deformation-round2-v1']}),flush=True)
        return
    requests=[resolve_idea(ROOT,c,queue_root=args.queue_root,study=s,freeze_source=args.phase=='enqueue')
              for c,s in STUDIES.items()]
    assert len({r['source']['digest'] for r in requests})==1
    output=[]
    for r in requests:
        summary=plan_summary(r,queue.inspect())
        if summary['preflight_blockers']:raise ValueError(summary['preflight_blockers'])
        if args.phase=='enqueue':
            e=queue.submit(r,r['study']['campaign'])
            output.append({'candidate':r['candidate']['id'],'study':r['study']['id'],
                           'request_id':e['request']['request_id'],'source':r['source']})
        else:output.append(summary)
    if args.progress and args.phase=='enqueue':
        previous=json.loads(args.progress.read_text());previous.update(phase='enqueued',requests=output)
        atomic_json(args.progress,previous)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
