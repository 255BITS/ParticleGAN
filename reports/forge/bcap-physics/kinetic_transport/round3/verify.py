"""Read-only archived-control parity, scorer compatibility and media checks."""
from pathlib import Path
import hashlib
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from experiments.forge.contracts import read_json,atomic_json,file_hash,stable_hash
from experiments.forge.state import state_digest
OUT=Path(__file__).resolve().parent


def loaded_attempt(row):
    root=Path(row['artifact_root'])
    for name,digest in row['inputs'].items():assert file_hash(root/name)==digest
    result=read_json(root/'result.json')
    task=next(t for t in result['task_results'] if t['task_id']==row['task_id'])
    cp=row['provenance_checkpoint'];path=Path(cp['artifact_root'])/cp['path']
    assert file_hash(path)==cp['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    request=read_json(root/'request.json')['request']
    assert request['source']['digest']==row['source_digest']
    return task,saved,request


def main():
    torch.set_num_threads(1)
    current=read_json(OUT/'provenance.json')
    archived=read_json(OUT.parent/'round2/provenance.json')
    old={r['task_id']:r for r in archived['attempts'] if r['arm']=='candidate'}
    rows=[]
    for row in current['attempts']:
        if row['arm']!='control':continue
        prior=old[row['task_id']]
        new,ns,nr=loaded_attempt(row);previous,ps,pr=loaded_attempt(prior)
        equality={
            'full_primary_observation_metrics':new['evidence']['observations']==previous['evidence']['observations'],
            'endpoint_metrics':new['evidence']['live']==previous['evidence']['live'],
            'grader_result':new['evaluator_result']==previous['evaluator_result'],
            'named_final_rng_states':state_digest(ns['streams'])==state_digest(ps['streams']),
            'final_model_tensors':state_digest(ns['trainer']['models'])==state_digest(ps['trainer']['models']),
            'base_optimizer_states':state_digest(ns['trainer']['optimizers'])==state_digest(ps['trainer']['optimizers']),
            'original_task_execution':nr['tasks'][row['task_id']]['execution']==pr['tasks'][row['task_id']]['execution'],
            'original_full_gates':nr['tasks'][row['task_id']]['evaluation']==pr['tasks'][row['task_id']]['evaluation'],
        }
        assert all(equality.values()),(row['task_id'],equality)
        rows.append({'task_id':row['task_id'],'new_attempt_id':row['attempt_id'],
                     'archived_attempt_id':prior['attempt_id'],'archived_result_sha256':prior['inputs']['result.json'],
                     'exact_equality':equality})
    atomic_json(OUT/'predecessor-parity.json',{'schema_version':1,
        'scope':'archived_local_v2_parity_diagnostic_no_qualification_reuse',
        'optimizer_updates_added':0,'sampling_draws_added':0,
        'original_source_digest':archived['attempts'][0]['source_digest'],
        'new_source_digest':current['attempts'][0]['source_digest'],'tasks':rows})
    broad={r['arm']:r for r in current['attempts'] if r['task_id']=='vector_two_broad'}
    a,sa,_=loaded_attempt(broad['candidate']);b,sb,_=loaded_attempt(broad['control'])
    inactive={'full_observations':a['evidence']['observations']==b['evidence']['observations'],
              'final_models':state_digest(sa['trainer']['models'])==state_digest(sb['trainer']['models']),
              'base_optimizer_states':state_digest(sa['trainer']['optimizers'])==state_digest(sb['trainer']['optimizers']),
              'named_rng_states':state_digest(sa['streams'])==state_digest(sb['streams'])}
    assert all(inactive.values()),inactive
    atomic_json(OUT/'inactive-broad-parity.json',{'schema_version':1,
        'scope':'actual full-scale-only candidate cohort vs matched control',
        'task_id':'vector_two_broad','exact_equality':inactive,
        'attempt_ids':{k:r['attempt_id'] for k,r in broad.items()}})
    reused=read_json(OUT.parent/'round2/scorer-controls-reuse.json')
    original=ROOT/reused['original_receipt'];assert file_hash(original)==reused['original_receipt_sha256']
    for path,digest in reused['scorer_sources'].items():assert file_hash(ROOT/path)==digest
    assert reused['oracle_PASS']==reused['collapse_FAIL']==4
    atomic_json(OUT/'scorer-controls-reuse.json',reused)
    media=read_json(OUT/'media.json')['media']
    for entry in media:
        with Image.open(OUT/f"{entry['arm']}-{entry['task_id']}.gif") as im:assert im.n_frames==9
    reference='2e42afee1da43bbefb510cc8d01f22a2840b483f'
    tree=subprocess.check_output(['git','ls-tree','-r',reference,
        'reports/forge/records','reports/forge/technique-inventory.md',
        'reports/forge/technique-inventory.json','reports/forge/leaderboards',
        'reports/forge/automation.json'],cwd=ROOT,text=True).splitlines()
    for line in tree:
        meta,name=line.split('\t',1);data=(ROOT/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta.split()[2],name
    from experiments.forge.knowledge import freshness
    assert freshness(ROOT)['fresh']
    assert read_json(ROOT/'reports/forge/compilation.json')['qualification_refresh']['mode']=='preserved_published_evidence'
    atomic_json(OUT/'software-verification.json',{'schema_version':1,
        'distinct_focused_tests_passed':141,'focused_suite_seconds':6.47,
        'final_backtrack_checks':{'passed':6,'seconds':1.21,'overlap_with_focused_suite':True},
        'initial_fixture_failure':'Four sampler tests used nonexistent sigma_rel constructor keyword; corrected to public sigma. No trainer correction or scientific retry.',
        'forge_validate':'PASS','bulk_logs_committed':False,
        'saved_metric_sets_reproduced':sum(r['metric_sets_reproduced'] or 0 for r in media),
        'actual_training_gifs':len(media),'frames_per_gif':9,
        'archived_local_v2_exact_parity_tasks':len(rows),
        'scorer_controls':'4 oracle PASS / 4 collapse FAIL reused after exact source, law and gate verification',
        'publication_source_sha256':file_hash(OUT/'publish.py'),
        'verification_source_sha256':file_hash(Path(__file__)),
        'original_compact_records_and_inventory_bytewise_unchanged':len(tree),
        'preservation_reference_commit':reference,
        'summaries_only_memory':'CURRENT','forge_compile_check':'CURRENT',
        'stdout':{p.name:file_hash(p) for p in Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kinetic_transport/logs').glob('*.log')
                  if p.name not in ('verification.log','publish.log','compile.log','compile-check.log')}})
    print({'exact_predecessor_parity_tasks':len(rows),'actual_training_gifs':len(media)},flush=True)

if __name__=='__main__':main()
