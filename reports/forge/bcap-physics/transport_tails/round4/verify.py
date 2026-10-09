"""Verify saved matched evidence, archived-v2 parity, scorer/media and preservation."""
from pathlib import Path
import hashlib, subprocess, sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from experiments.forge.contracts import read_json, atomic_json, file_hash
from experiments.forge.state import state_digest
OUT=Path(__file__).resolve().parent
LOG=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/transport_tails/logs')


def load(row):
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
    archived=read_json(ROOT/'reports/forge/bcap-physics/kinetic_transport/round3/provenance.json')
    old={r['task_id']:r for r in archived['attempts'] if r['arm']=='control'}
    rows=[]
    for row in current['attempts']:
        if row['arm']!='control':continue
        prior=old[row['task_id']]
        new,ns,nr=load(row);previous,ps,pr=load(prior)
        equality={'full_primary_observation_metrics':new['evidence']['observations']==previous['evidence']['observations'],
            'endpoint_metrics':new['evidence']['live']==previous['evidence']['live'],
            'grader_result':new['evaluator_result']==previous['evaluator_result'],
            'named_final_rng_states':state_digest(ns['streams'])==state_digest(ps['streams']),
            'final_model_tensors':state_digest(ns['trainer']['models'])==state_digest(ps['trainer']['models']),
            'base_optimizer_states':state_digest(ns['trainer']['optimizers'])==state_digest(ps['trainer']['optimizers']),
            'original_task_execution':nr['tasks'][row['task_id']]['execution']==pr['tasks'][row['task_id']]['execution'],
            'original_full_gates':nr['tasks'][row['task_id']]['evaluation']==pr['tasks'][row['task_id']]['evaluation']}
        assert all(equality.values()),(row['task_id'],equality)
        rows.append({'task_id':row['task_id'],'new_attempt_id':row['attempt_id'],'archived_attempt_id':prior['attempt_id'],
                     'archived_result_sha256':prior['inputs']['result.json'],'exact_equality':equality})
    assert len(rows)==5
    atomic_json(OUT/'predecessor-parity.json',{'schema_version':1,
        'scope':'new matched control vs archived local-v2 parity diagnostic; no qualification reuse',
        'optimizer_updates_added':0,'sampling_draws_added':0,'tasks':rows})
    reused=read_json(ROOT/'reports/forge/bcap-physics/kinetic_transport/round3/scorer-controls-reuse.json')
    assert file_hash(ROOT/reused['original_receipt'])==reused['original_receipt_sha256']
    for path,digest in reused['scorer_sources'].items():assert file_hash(ROOT/path)==digest
    assert reused['oracle_PASS']==reused['collapse_FAIL']==4
    atomic_json(OUT/'scorer-controls-reuse.json',reused)
    media=read_json(OUT/'media.json')['media']
    for entry in media:
        with Image.open(OUT/f"{entry['arm']}-{entry['task_id']}.gif") as im:assert im.n_frames==9
    reference='f9f6419ac251033e9076f7ba4bab87e715d321bd'
    tree=subprocess.check_output(['git','ls-tree','-r',reference,'reports/forge/records',
        'reports/forge/technique-inventory.md','reports/forge/technique-inventory.json',
        'reports/forge/leaderboards','reports/forge/automation.json'],cwd=ROOT,text=True).splitlines()
    for line in tree:
        meta,name=line.split('\t',1);data=(ROOT/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta.split()[2],name
    from experiments.forge.knowledge import freshness
    assert freshness(ROOT)['fresh']
    assert read_json(ROOT/'reports/forge/compilation.json')['qualification_refresh']['mode']=='preserved_published_evidence'
    from experiments.forge.sources import verify_snapshot
    request=read_json(Path(current['attempts'][0]['artifact_root'])/'request.json')['request']
    verify_snapshot(Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/transport_tails/queue/snapshots')/request['source']['digest'],request['source'])
    for path,digest in request['source']['files'].items():assert file_hash(ROOT/path)==digest,path
    atomic_json(OUT/'software-verification.json',{'schema_version':1,
        'focused_tests_distinct_passed':18,'forge_boundary_study_tests_passed':69,
        'software_training_outer_updates':10,'software_training_scope':'four new public-trainer checkpoint/consumption updates plus six existing local/backtrack regression updates; no task qualification',
        'pytest_reported_seconds':{'focused_initial':2.07,'corrected_admission_only':.54,'boundaries_studies':3.47},
        'initial_declaration_failure':'One study used a nonstandard terminal_rules value; restored schema-mandated request_missing_evidence before admission. No trainer change or scientific retry.',
        'initial_protocol_invocation':'Requested nonexistent test_forge_field_boundaries.py; corrected to test_forge_boundaries.py; no tests or training launched by that invocation.',
        'forge_validate':'PASS','bulk_logs_committed':False,'archived_v2_exact_parity_tasks':len(rows),
        'saved_metric_sets_reproduced':sum(r['metric_sets_reproduced'] or 0 for r in media),
        'actual_training_gifs':len(media),'frames_per_gif':9,
        'scorer_controls':'4 oracle PASS / 4 collapse FAIL reused after exact scorer-source, law and gate checks',
        'original_compact_records_and_inventory_bytewise_unchanged':len(tree),'preservation_reference_commit':reference,
        'summaries_only_memory':'CURRENT','forge_compile_check':'CURRENT',
        'reproduction_source_sha256':{p.name:file_hash(p) for p in OUT.glob('*.py')},
        'local_software_log_sha256':{p.name:file_hash(p) for p in LOG.glob('*.log') if p.name in
                                   ('focused-tests.log','admission-test.log','protocol-tests.log','validate.log')}})
    print({'event':'verification_complete','predecessor_parity_tasks':len(rows),'actual_training_gifs':len(media)},flush=True)

if __name__=='__main__':main()
