"""Freeze and drain the matched ordinary full-suite requests.

Invoke from a clean, committed integration tree. Raw queue artifacts and logs
remain on the artifact drive. Publication/compilation never runs as a callback.
"""
from pathlib import Path
import json

from experiments.forge.contracts import atomic_json, utc_now
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next')
PREFIX = 'bcap-develop-integration'


def main():
    queue = Queue(ARCHIVE/'queue', report_root=ROOT/'reports/forge', on_completion=None)
    progress = dict(phase='submitting', updated_at=utc_now(), requests={}, source=None,
                    log=str(ARCHIVE/'logs/driver.log'))
    for role in ('winner', 'combined'):
        request = resolve_idea(ROOT, f'{PREFIX}-{role}-v1', study=f'{PREFIX}-{role}-study-v1',
                               queue_root=ARCHIVE/'queue', freeze_source=True)
        assert not request['preflight_blockers'], request['preflight_blockers']
        assert request['study_review']['status'] == 'READY', request['study_review']
        bad = {name: task['preflight_blockers'] for name, task in request['tasks'].items()
               if task['preflight_blockers'] and any(a['task']==name and a['qualification_tier']<=2
                   and a['importance']=='required' for a in request['view']['assignments'])}
        assert not bad, bad
        assert progress['source'] in (None, request['source']['digest'])
        progress['source'] = request['source']['digest']
        receipt = queue.submit(request, request['study']['campaign'])
        progress['requests'][role] = receipt['request']['request_id']
        atomic_json(ARCHIVE/'progress.json', progress)
        print(json.dumps(dict(event='submitted', time=utc_now(), role=role,
            request=receipt['request']['request_id'], source_commit=request['source']['origin_commit'])), flush=True)
    progress.update(phase='running', updated_at=utc_now())
    atomic_json(ARCHIVE/'progress.json', progress)
    drain(queue, ['0', '1'], workers_per_gpu=1, allow_sharing=True, watch=False, campaign=f'{PREFIX}-v1')
    progress.update(phase='ordinary_complete', updated_at=utc_now())
    atomic_json(ARCHIVE/'progress.json', progress)
    print(json.dumps(dict(event='ordinary_complete', time=utc_now(), requests=progress['requests'])), flush=True)


if __name__ == '__main__':
    main()
