"""Verify the completed measurement source, accounting and saved media; zero training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--scientific-root',type=Path,default=ROOT)
    args=parser.parse_args();report=Path(__file__).parent
    state=json.loads((args.queue/'queue/state.json').read_text())
    entries={a:next(e for e in state['submissions'].values() if e['request']['study']['id']==f'blacksmith-tempering-r2-{a}-v1') for a in ['candidate','control']}
    source=entries['candidate']['request']['source']
    assert source==entries['control']['request']['source']
    for name,digest in source['files'].items():
        path=args.scientific_root/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    checks=json.loads((report/'verification.json').read_text())
    assert checks['trained_source_digest']==source['digest']
    assert len(source['files'])==checks['frozen_files_verified']
    assert all(e['status'] not in {'queued','running','paused'} for e in entries.values())
    assert state['campaigns']['blacksmith-tempering-r2-v1']['reserved_seconds']==0
    attempts=[]
    for entry in entries.values():
        for job in entry['request']['jobs']:
            result=state['jobs'][job['compatibility_key']].get('result')
            if result:attempts.append(result)
    assert len(attempts)==checks['attempt_certificates_verified']
    paid=sum(r['task_results'][0]['cost']['execution_seconds'] for r in attempts)
    assert paid==checks['paid_worker_seconds']
    media=json.loads((report/'media-index.json').read_text())['receipts']
    assert len(media)==checks['actual_training_gifs_verified']
    for receipt in media:
        path=report/'media'/receipt['arm']/(receipt['task_id']+'.gif')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['gif_sha256'],str(path)
    print(json.dumps({'scientific_files_verified':len(source['files']),'attempts':len(attempts),'gifs':len(media),'paid_worker_seconds':paid}))

if __name__=='__main__':main()
