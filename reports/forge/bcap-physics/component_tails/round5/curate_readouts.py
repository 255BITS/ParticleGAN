"""Link concluded compact records without repeating rendering or training."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
OUT=Path(__file__).resolve().parent
QUEUE=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/component_tails/queue')


def main():
    results=read_json(OUT/'results.json');state=read_json(QUEUE/'queue/state.json')
    records={}
    for arm,entry in results['arms'].items():
        submission=state['submissions'][entry['request_id']]
        assert submission['status']=='concluded'
        request=submission['request']
        identity={'candidate':entry['candidate_id'],'revision':entry['revision'],
                  'study_id':request['study']['id'],'study_sha256':stable_hash(request['study'])}
        path=ROOT/'reports/forge/records'/('readout-'+stable_hash(identity)[:24]+'.json')
        record=read_json(path)
        assert record['qualification_input'] is record['qualification_reuse'] is False
        assert {r['task_id']:r['gate_status'] for r in record['task_results']}=={
            r['task_id']:r['status'] for r in entry['tasks']}
        record.setdefault('original_source',record['source'])
        record['source']={'path':'reports/forge/bcap-physics/component_tails/round5/results.json'}
        record['summary_curation']='Compact report navigation; original grades, attempts and numerical receipts unchanged.'
        atomic_json(path,record)
        records[arm]={'record_id':record['record_id'],'path':str(path.relative_to(ROOT)),'sha256':file_hash(path)}
        entry['submission_status']=submission['status']
    results['readout_records']=records
    atomic_json(OUT/'results.json',results)
    print({'event':'curated_readouts','records':records})


if __name__=='__main__':main()
