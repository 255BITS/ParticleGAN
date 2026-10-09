"""Public matched Queue/drain; no full-qualification completion callback."""
from pathlib import Path
import argparse
import json
import os
import sys
import threading

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments.forge.planning import resolve_idea, plan_summary
from experiments.forge.queue import Queue, drain
from experiments.forge.contracts import atomic_json

CAMPAIGN = 'component_tails_round5'
QUEUE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails/queue')
STUDIES = {'kinetic_transport_local_v2':'component_tails_control_round5',
           'component_prior_transport_r5':'component_tails_candidate_round5'}


def mirror_events(queue, stopped):
    path = queue.root/'events.jsonl'
    path.touch(exist_ok=True)
    with path.open() as source:
        while True:
            for line in source:
                if json.loads(line).get('event') in ('worker_observation','worker_completed','worker_started'):
                    print(line.rstrip(),flush=True)
            if stopped.wait(1):
                for line in source:
                    print(line.rstrip(),flush=True)
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['plan','enqueue','drain'])
    parser.add_argument('--gpu',default='1')
    args = parser.parse_args()
    os.environ['PARTICLEGAN_FORGE_QUEUE'] = str(QUEUE)
    queue = Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None)
    if args.phase == 'drain':
        stopped = threading.Event()
        monitor = threading.Thread(target=mirror_events,args=(queue,stopped));monitor.start()
        try:
            drain(queue,[args.gpu],workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
        finally:
            stopped.set();monitor.join()
        print(json.dumps({'event':'bounded_drain_complete','campaign':queue.inspect()['campaigns'][CAMPAIGN]}),flush=True)
        return
    requests = [resolve_idea(ROOT,c,queue_root=QUEUE,study=s,freeze_source=args.phase=='enqueue')
                for c,s in STUDIES.items()]
    assert len({r['source']['digest'] for r in requests}) == 1
    output = []
    for request in requests:
        summary = plan_summary(request,queue.inspect())
        if summary['preflight_blockers']:
            raise ValueError(summary['preflight_blockers'])
        if args.phase == 'enqueue':
            result = queue.submit(request,request['study']['campaign'])
            output.append({'candidate':request['candidate']['id'],'study':request['study']['id'],
                'request_id':result['request']['request_id'],'source_commit':request['source']['origin_commit'],
                'source_digest':request['source']['digest']})
        else:
            output.append(summary)
    if args.phase == 'enqueue':
        progress = Path('/tmp/bcap-physics-round5-20261009/component_tails/progress.json')
        previous = json.loads(progress.read_text())
        previous.update(phase='enqueued',requests=output,source_commit=output[0]['source_commit'],
                        source_digest=output[0]['source_digest'])
        atomic_json(progress,previous)
        atomic_json(Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/handoff/component_tails/progress.json'),previous)
    print(json.dumps(output,indent=2),flush=True)


if __name__ == '__main__':
    main()
