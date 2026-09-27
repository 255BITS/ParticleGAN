#!/usr/bin/env python3
"""Create a fixed launch spec only from matching independent constructor proofs."""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent
CHECKS=('all_initial_tensors_match_public_host','repeat_without_rng_reset',
        'constructor_rng_cursor_preserved','initializer_rng_neutral',
        'historical_prior_registration_preserved','all_bindings_restore_on_exception',
        'batch_distance_scope_explicit')

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row',action='append',required=True)
    parser.add_argument('--lane',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite a reviewed batch specification')
    assert args.lane.replace('-','').isalnum()
    index={r['queue_row']:r for r in read(HERE/'prepared-index.json')['rows']}
    assert len(set(args.row))==len(args.row)
    sources={};reviews=[];cases=[]
    for i,row_id in enumerate(args.row):
        row=index[row_id];root=HERE/row['directory_relative'] if row.get('directory_relative') else Path(row['directory']);proof_path=HERE/row['required_review_relative'] if row.get('required_review_relative') else Path(row['required_review'])
        proof=read(proof_path);manifest=read(root/'manifest.json')
        assert sha(root/'manifest.json')==row['manifest_sha256']
        assert proof['status']=='PASS'
        assert proof['source_plan_sha256']==sha(root/'source-plan.json')
        assert proof['bridge_sha256']==sha(root/'initialization_bridge.py')
        assert proof['runner_sha256']==sha(root/'run_research_mode_hold.py')
        assert proof['manifest_sha256']==sha(root/'manifest.json')
        assert proof['cuda_initialized'] is False and proof['learner_steps']==0
        assert all(proof.get('checks',{}).get(k) is True for k in CHECKS)
        assert set(proof['all_initial_material'])=={'generator','critic','prior'}
        files=manifest['files']|{'manifest.json':sha(root/'manifest.json')}
        for rel,want in files.items():assert sha(root/rel)==want
        source_name='case'+str(i);review_name='review'+str(i)
        sources[source_name]=dict(root=str(root),files=files)
        sources[review_name]=dict(root=str(proof_path.parent),files={p.name:sha(p) for p in sorted(proof_path.parent.iterdir()) if p.is_file() and p.suffix in ('.py','.json','.md','.log')})
        reviews.append(dict(path=str(proof_path),sha256=sha(proof_path),required_status='PASS'))
        case_id=root.name
        cases.append(dict(id=case_id,queue_row=row_id,candidate=row['candidate'],argv=['/tmp/pr38-default-env/bin/python','{inputs}/'+source_name+'/run_research_mode_hold.py','--reviewed-cpu-proof','{inputs}/'+review_name+'/'+proof_path.name,'--output','{output}/'+case_id]))
    result=dict(status='REVIEWED_READY_FOR_EXTERNAL_RUN',lane=args.lane,sources=sources,independent_reviews=reviews,cases=cases,instructions='Fixed historical research tiny-host retest batch. One fresh process for each listed exact original candidate; public standalone initialization only, no old weights. Retain original losses/hooks/schedules and original ordinary Adam metadata (no eager counter injection),1200 accepted steps/all24 observations/final-five gate. These are research-host results, not public API scores; historical quality and eligibility do not transfer. Preserve exceptions and every receipt; no repairs, retries, seeds or additional tests. Candidate source-bound constructor proofs and actual initial-tensor assertions are mandatory.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(spec=str(args.output),sha256=sha(args.output),cases=[x['id'] for x in cases])))

if __name__=='__main__':main()
