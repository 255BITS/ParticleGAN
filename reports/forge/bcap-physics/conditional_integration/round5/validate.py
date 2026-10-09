"""Validate the published artifacts and stopped, fully accounted campaign."""
from pathlib import Path
import json
from PIL import Image
from experiments.forge.contracts import atomic_json,read_json,file_hash
from experiments.forge.queue import process_identity
from publish import ROOT,QUEUE,OUT,REQUESTS

def main():
    state=read_json(QUEUE/'queue/state.json');results=read_json(OUT/'results.json')
    assert len(results['task_results'])==28
    assert all(r['gate_status'] in ('PASS','FAIL','BLOCKED') for r in results['task_results'])
    assert all(j['status'] in ('terminal','blocked') for j in state['jobs'].values())
    assert all(not j.get('worker') or process_identity(j['worker']['pid'])!=j['worker']['process_identity'] for j in state['jobs'].values())
    campaign=state['campaigns']['conditional-integration-round5-v1'];assert campaign['reserved_seconds']==0
    assert results['executed_full_reservations']+1200<=43200
    audit=read_json(OUT/'audit.json');assert all(x['bitwise_equal'] for x in audit['checks'])
    assert all(file_hash(ROOT/path)==digest for path,digest in audit['protected_original_files_unchanged'].items())
    request=state['submissions'][next(iter(REQUESTS.values()))]['request']
    prefixes=('particlegan/','benchmarks/','experiments/','lib/','configs/forge/')
    assert all(file_hash(ROOT/path)==digest for path,digest in request['source']['files'].items() if path.startswith(prefixes))
    assert read_json(QUEUE.parent/'logs/compile-check.log')['fresh']
    frames={}
    for item in read_json(OUT/'media/index.json')['media']:
        path=OUT/item['gif'];assert file_hash(path)==item['gif_sha256']
        with Image.open(path) as gif:
            assert gif.n_frames>=2;frames[item['gif']]=gif.n_frames
    assert len(frames)==sum(r['gate_status'] in ('PASS','FAIL') for r in results['task_results'])
    software=QUEUE.parent/'logs/software.log';assert '28 passed' in software.read_text()
    atomic_json(OUT/'validation.json',dict(schema_version=1,qualification_input=False,
        completed_cells=len(results['task_results']),attempts=results['scientific_attempts'],retries=results['scientific_retries'],
        final_grades=results['outcomes'],source_commit=results['source_commit'],source_digest=results['source_digest'],
        mechanism_protocol_tests=28,software_stdout=dict(path=str(software),sha256=file_hash(software)),
        forge_validation=read_json(QUEUE.parent/'logs/validate-final.log'),
        compile_summaries_check=read_json(QUEUE.parent/'logs/compile-check.log'),
        frozen_scientific_files_unchanged=audit['scientific_files_unchanged'],
        original_task_qualification_telemetry_files_unchanged=len(audit['protected_original_files_unchanged']),
        saved_bitwise_parity_checks=len(audit['checks']),matched_named_stream_cohorts=len(audit['matched_conditions']),
        actual_training_gifs=frames,paid_worker_seconds=results['paid_seconds'],
        planned_full_reservations=38880,executed_full_reservations=results['executed_full_reservations'],
        ancillary_allowance_seconds=1200,track_ceiling=43200,active_workers=0,reservation_remaining=0,
        main_execution_and_log_follower_stopped=True,capacity_probes=0,optimizer_updates_added=0,training_draws_added=0))
    print(json.dumps(dict(valid=True,completed_cells=28,media=len(frames),paid_seconds=results['paid_seconds'])))
if __name__=='__main__':main()
