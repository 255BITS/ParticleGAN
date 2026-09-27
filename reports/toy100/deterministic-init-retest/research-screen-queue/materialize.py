#!/usr/bin/env python3
"""Prepare exact compatible research bundles; never imports Torch or launches."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
TEMPLATE=HERE.parent/'research-mode-hold-preparation'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def materialize(row):
    assert row['status']=='COMPATIBLE_SOURCE_GROUP_REQUIRES_CANDIDATE_CONSTRUCTOR_REVIEW'
    name=re.sub('[^a-z0-9]+','-',row['candidate'].lower()).strip('-')
    # The row digest prevents collisions among reused old candidate labels.
    suffix=hashlib.sha256(row['id'].encode()).hexdigest()[:8]
    target=HERE/'prepared'/(name+'-'+suffix)
    if target.exists():raise FileExistsError('Never overwrite a sealed preparation: '+str(target))
    source_archive=Path(row['candidate_archive']['path'])
    assert sha(source_archive)==row['candidate_archive']['sha256']
    seal=read(TEMPLATE/'manifest.json')
    for relative,want in seal['files'].items():assert sha(TEMPLATE/relative)==want
    target.mkdir(parents=True)
    excluded={'README.md','source-plan.json','source-preflight.json','run_research_mode_hold.py'}
    for relative in seal['files']:
        if relative in excluded or relative.startswith('ka2-source/'):continue
        dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(TEMPLATE/relative,dest)
    with zipfile.ZipFile(source_archive) as archive:
        for relative in archive.namelist():
            assert Path(relative).name==relative and relative in row['candidate_files']
            content=archive.read(relative)
            assert hashlib.sha256(content).hexdigest()==row['candidate_files'][relative]
            dest=target/'candidate-source'/relative;dest.parent.mkdir(exist_ok=True);dest.write_bytes(content)
    plan=read(TEMPLATE/'source-plan.json')
    plan.update(candidate='RESEARCH-'+row['candidate']+'-new-init',source_candidate=row['source_directory'],candidate_files=row['candidate_files'],source_queue_row=row['id'],source_queue_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_definition=row,probe_source=str(target/'candidate-source/probe.py'),probe_binding='Byte-identical original shared probe; each candidate retains its own exact mechanism/configuration and all local hooks.',historical_runtime=row['benchmark_runtime'],runtime_binding=row['runtime_binding'],historical_runtime_receipts=row['historical_runtime_receipts'],learner_preservation='Exact candidate-local original research hooks/configuration and the shared frozen probe/runtime; no public learner substitution.',group_template_manifest_sha256=sha(TEMPLATE/'manifest.json'))
    # These are declarative config values; internal candidate hooks may implement
    # a different autonomous policy. The original config is never normalized.
    plan['policy']={'evaluation_steps':1200,'num_particles':12,'z_dim':4,'batch_size':128,'seed':0,'prior_init_std':.5,'evaluation_latent_seed':9,'evaluation_global_noise':'402+completed_step','original_candidate_config':row['configuration'],'note':'Frozen tiny host resources and evaluation budget; exact candidate config/hook rate and noise semantics remain unchanged.'}
    dump(target/'source-plan.json',plan)
    worker=(TEMPLATE/'run_research_mode_hold.py').read_text()
    assert worker.count("source=HERE/'ka2-source'")==1
    assert worker.count("candidate='RESEARCH-KA2-new-init'")==1
    worker=worker.replace("source=HERE/'ka2-source'","source=HERE/'candidate-source'")
    worker=worker.replace("candidate='RESEARCH-KA2-new-init'","candidate=plan['candidate']")
    worker=worker.replace('Prepared exact research-KA2 screen','Prepared exact grouped research screen')
    ast.parse(worker);(target/'run_research_mode_hold.py').write_text(worker)
    dump(target/'source-preflight.json',dict(status='PASS_SOURCE_ONLY_CPU_REVIEW_NOT_RUN',template_manifest_sha256=sha(TEMPLATE/'manifest.json'),queue_row=row['id'],candidate_archive=row['candidate_archive'],bridge_sha256=sha(target/'initialization_bridge.py'),worker_sha256=sha(target/'run_research_mode_hold.py'),source_plan_sha256=sha(target/'source-plan.json'),constructor_checks='REQUIRED_SEPARATE_MATCHING_RECEIPT',no_training=True))
    (target/'README.md').write_text(f"# {plan['candidate']} source preparation\n\nOriginal candidate `{row['id']}` on the declared common frozen tiny host. Exact source/config preserved; new standalone public initialization only. This is research-host evidence and retains its original schedule eligibility. No old pass or new quality is inherited.\n\nThe wrapper differs from the reviewed KA2 wrapper only in its source-directory and candidate label. It requires a matching independent CPU constructor/source proof before execution and checks every actual CUDA initial tensor against that proof before optimizer construction. This bundle is prepared, not authorized or executed. Runtime provenance: `{row['runtime_binding']}`.\n")
    files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    dump(target/'manifest.json',dict(schema=1,status='PREPARED_REQUIRES_INDEPENDENT_CPU_REVIEW',files=files))
    return dict(queue_row=row['id'],candidate=plan['candidate'],directory=str(target),manifest_sha256=sha(target/'manifest.json'),source_plan_sha256=sha(target/'source-plan.json'),bridge_sha256=sha(target/'initialization_bridge.py'),worker_sha256=sha(target/'run_research_mode_hold.py'),status='PREPARED_CPU_REVIEW_PENDING',required_review=str(HERE/'reviews'/target.name/'cpu-constructor-proof.json'),command=['/tmp/pr38-default-env/bin/python',str(target/'run_research_mode_hold.py'),'--reviewed-cpu-proof',str(HERE/'reviews'/target.name/'cpu-constructor-proof.json'),'--output','NEW_OUTPUT_DIRECTORY'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row',action='append',required=True)
    args=parser.parse_args();queue=read(HERE/'queue.json');by_id={r['id']:r for r in queue['rows']}
    index=HERE/'prepared-index.json';existing=read(index) if index.exists() else dict(schema=1,status='SOURCE_PREPARED_NOT_AUTHORIZED',rows=[])
    assert not set(args.row).intersection(x['queue_row'] for x in existing['rows'])
    additions=[]
    for row_id in args.row:additions.append(materialize(by_id[row_id]))
    existing['rows'].extend(additions);existing['queue_sha256']=sha(HERE/'queue.json');dump(index,existing)
    print(json.dumps(additions,indent=2))

if __name__=='__main__':main()
