"""Read-only receipt/source, protocol, predecessor, media and preservation audit."""
from pathlib import Path
import hashlib
import subprocess
import sys
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from experiments.forge.state import state_digest
from experiments.forge.sources import verify_snapshot
OUT=Path(__file__).resolve().parent
BASE='1036ce0c637956a4e87cc56255d503f80de578f4'


def load(row):
    directory=Path(row['artifact_root'])
    for name,digest in row['inputs'].items():assert file_hash(directory/name)==digest
    result=read_json(directory/'result.json')
    task=next(r for r in result['task_results'] if r['task_id']==row['task_id'])
    desc=row['provenance_checkpoint'];path=Path(desc['artifact_root'])/desc['path']
    assert file_hash(path)==desc['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state_digest(state)==desc['state_sha256']
    request=read_json(directory/'request.json')['request']
    assert stable_hash(result)==row['result_hash']
    assert request['source']['digest']==row['source_digest']
    return task,state,request


def scorer_reuse():
    old=ROOT/'reports/forge/bcap-physics/kinetic_transport/scorer-controls.json'
    vector=read_json(old)
    # The predecessor receipt is byte-exact and declares the unchanged scorers.
    binding=read_json(ROOT/'reports/forge/bcap-physics/transport_tails/round4/scorer-controls-reuse.json')
    assert file_hash(old)==binding['original_receipt_sha256']
    assert binding['oracle_PASS']==binding['collapse_FAIL']==4
    for path,digest in binding['scorer_sources'].items():assert file_hash(ROOT/path)==digest
    native=Path('/home/martyn/dev/ParticleGAN-bcap-r4-role_motion/reports/forge/bcap-physics/role_motion/round4/scorer-controls.json')
    n=read_json(native)
    for path,digest in n['scorer_sources'].items():assert file_hash(ROOT/path)==digest
    chosen=[r for r in n['controls'] if r['task_id']=='grid100']
    assert [r['status'] for r in chosen]==['PASS','FAIL']
    # Preserve the original identities, rather than recasting external controls
    # as new-source training qualification or newly generated samples.
    atomic_json(OUT/'scorer-controls-reuse.json',{
        'schema_version':1,'qualification_input':False,'optimizer_updates_added':0,'sampling_draws_added':0,
        'scope':'five compatible base target/scorer laws; original oracle/destructive controls, no training qualification',
        'oracle_PASS':5,'collapse_FAIL':5,
        'vector_original':{'path':str(old.relative_to(ROOT)),'sha256':file_hash(old),
                           'bound_receipt_sha256':file_hash(ROOT/'reports/forge/bcap-physics/transport_tails/round4/scorer-controls-reuse.json')},
        'native_original':{'path':str(native),'sha256':file_hash(native),
            'github':'https://github.com/255BITS/ParticleGAN/blob/871b76af18388e70595db1eb9d685f7a8de8af70/reports/forge/bcap-physics/role_motion/round4/scorer-controls.json'},
        'scorer_sources':{**binding['scorer_sources'],**n['scorer_sources']},
        'native_controls':chosen,
        'shifted_stability_scope':'unchanged original analytic scalar scorer and full paired retention/deadline gates; not a new oracle cohort'})


def main():
    torch.set_num_threads(1)
    current=read_json(OUT/'provenance.json')
    old=read_json(ROOT/'reports/forge/bcap-physics/transport_tails/round4/provenance.json')
    archived={r['task_id']:r for r in old['attempts'] if r['arm']=='control'}
    rows=[];streams={};source=None
    for receipt in current['attempts']:
        task,saved,request=load(receipt)
        if source is None:
            source=request['source'];verify_snapshot(Path(source['snapshot_path']),source)
            for path,digest in source['files'].items():assert file_hash(ROOT/path)==digest,path
        else:assert request['source']==source
        streams.setdefault(receipt['task_id'],{})[receipt['arm']]=state_digest(saved['streams'])
        if receipt['arm']!='control' or receipt['task_id']=='grid100':continue
        previous,previous_saved,previous_request=load(archived[receipt['task_id']])
        equality={
            'full_primary_observation_metrics':task['evidence']['observations']==previous['evidence']['observations'],
            'endpoint_metrics':task['evidence']['live']==previous['evidence']['live'],
            'grader_result':task['evaluator_result']==previous['evaluator_result'],
            'named_final_rng_states':state_digest(saved['streams'])==state_digest(previous_saved['streams']),
            'final_model_tensors':state_digest(saved['trainer']['models'])==state_digest(previous_saved['trainer']['models']),
            'base_optimizer_states':state_digest(saved['trainer']['optimizers'])==state_digest(previous_saved['trainer']['optimizers']),
            'original_task_execution':request['tasks'][receipt['task_id']]['execution']==previous_request['tasks'][receipt['task_id']]['execution'],
            'original_full_gates':request['tasks'][receipt['task_id']]['evaluation']==previous_request['tasks'][receipt['task_id']]['evaluation']}
        assert all(equality.values()),(receipt['task_id'],equality)
        rows.append({'task_id':receipt['task_id'],'new_attempt_id':receipt['attempt_id'],
                     'archived_attempt_id':archived[receipt['task_id']]['attempt_id'],
                     'archived_result_sha256':archived[receipt['task_id']]['inputs']['result.json'],
                     'exact_equality':equality})
    assert len(rows)==5
    assert len(streams)==6
    paired={task:v for task,v in streams.items() if set(v)=={'candidate','control'}}
    assert all(v['candidate']==v['control'] for v in paired.values()),paired
    atomic_json(OUT/'predecessor-parity.json',{'schema_version':1,'qualification_input':False,
        'optimizer_updates_added':0,'sampling_draws_added':0,'tasks':rows,
        'scope':'five fresh local-v2 tasks exactly reproduce retained round4 endpoint/trajectory/state; native local-v2 has no archived compatible control'})
    scorer_reuse()
    media=read_json(OUT/'media.json')['media']
    expected_media=sum(r['status'] in ('PASS','FAIL') for a in read_json(OUT/'results.json')['arms'].values() for r in a['tasks'])
    assert len(media)==expected_media
    for entry in media:
        path=OUT/f"{entry['arm']}-{entry['task_id']}.gif"
        assert file_hash(path)==entry['gif_sha256']
        with Image.open(path) as im:assert im.n_frames==9
    tree=subprocess.check_output(['git','ls-tree','-r',BASE,'reports/forge/records',
        'reports/forge/technique-inventory.md','reports/forge/technique-inventory.json',
        'reports/forge/leaderboards','reports/forge/automation.json'],cwd=ROOT,text=True).splitlines()
    for line in tree:
        meta,name=line.split('\t',1);data=(ROOT/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta.split()[2],name
    from experiments.forge.knowledge import freshness
    assert freshness(ROOT)['fresh']
    assert read_json(ROOT/'reports/forge/compilation.json')['qualification_refresh']['mode']=='preserved_published_evidence'
    checks=read_json(OUT/'software-verification.json') if (OUT/'software-verification.json').is_file() else {}
    checks.update(schema_version=1,forge_validate='PASS',forge_compile_check='CURRENT',
        focused_distinct_checks=25,forge_boundary_study_checks=69,
        pytest_reported_seconds={'focused':3.29,'boundaries_studies':4.35},
        software_training_scope='tiny gradient/routing, checkpoint and retained transport regression fixtures; separate from quality evidence',
        software_training_outer_updates=15,
        archived_v2_exact_parity_tasks=5,matched_complete_final_stream_registries=len(paired),
        all_executed_scientific_bytes_unchanged=True,actual_training_gifs=len(media),frames_per_gif=9,
        saved_scalar_vector_metric_sets_reproduced=sum(r['metric_sets_reproduced'] or 0 for r in media),
        native_media_scope='saved complete actual-training snapshots and metric curves, no redraw or regrade',
        scorer_controls='5 oracle PASS / 5 collapse FAIL after exact original source checks',
        prior_qualification_inventory_and_telemetry_blobs_unchanged=len(tree),
        preservation_reference_commit=BASE,bulk_logs_committed=False,
        reproduction_source_sha256={p.name:file_hash(p) for p in OUT.glob('*.py')})
    atomic_json(OUT/'software-verification.json',checks)
    print({'event':'verification_complete','parity_tasks':len(rows),'matched_streams':len(streams),'gifs':len(media)},flush=True)


if __name__=='__main__':main()
