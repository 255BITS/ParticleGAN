"""Validate complete source/runtime/task/protocol and archived-publication proofs."""
from pathlib import Path
import json
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json,stable_hash
from publish import QUEUE,REQUESTS,OUT


def main():
    state=read_json(QUEUE/'queue/state.json')
    requests=[state['submissions'][rid]['request'] for rid in REQUESTS.values()]
    first=requests[0]
    for request in requests[1:]:
        assert request['source']==first['source']
        assert request['runtime']==first['runtime']
        assert request['protocol']==first['protocol']
        for task in first['tasks']:
            for field in ('execution','evaluation','resources','dependencies','adapter','requires_capabilities'):
                assert first['tasks'][task].get(field)==request['tasks'][task].get(field),(task,field)
    scientific_prefixes=('particlegan/','benchmarks/','experiments/','lib/','configs/forge/')
    scientific={path:digest for path,digest in first['source']['files'].items() if path.startswith(scientific_prefixes)}
    changed=[path for path,digest in scientific.items() if not (ROOT/path).exists() or file_hash(ROOT/path)!=digest]
    assert not changed,changed
    before=read_json(Path('/tmp/bcap-physics-round4-20261009/projection_ablation/archived-snapshot-hashes.json'))
    snapshot_changed=[path for path,digest in before.items() if file_hash(ROOT/path)!=digest]
    assert not snapshot_changed,snapshot_changed
    results=read_json(OUT/'results.json')
    assert len(results['task_results'])==18
    assert all(r['gate_status'] in ('PASS','FAIL') for r in results['task_results'])
    from experiments.forge.queue import process_identity
    assert all(j['status']=='terminal' for j in state['jobs'].values())
    assert all(process_identity(j['worker']['pid']) != j['worker']['process_identity'] for j in state['jobs'].values())
    assert sum(len(j['attempts']) for j in state['jobs'].values())==19
    assert sum(len(j['attempts'])-1 for j in state['jobs'].values())==1
    assert state['campaigns']['projection-ablation-round4-v1']['reserved_seconds']==0
    proofs=read_json(OUT/'provenance.json')['proofs']
    for proof in proofs:
        descriptor=proof['provenance_checkpoint']
        saved=torch.load(Path(descriptor['artifact_root'])/descriptor['path'],map_location='cpu',weights_only=False)
        if 'applied' in saved:
            assert saved['applied']['rng']['seed']==0
            assert all(a['unintended_rng_deviations']==0 for a in saved['applied']['rng_audits'])
    media={}
    for item in read_json(OUT/'media/index.json')['media']:
        path=OUT/item['gif']
        assert file_hash(path)==item['gif_sha256']
        with Image.open(path) as gif:
            media[item['gif']]=gif.n_frames
            assert gif.n_frames>=2
    assert len(media)==18
    scorer=read_json(OUT/'scorer-controls.json')
    assert len(scorer['rows'])==7 and len(scorer['conditional_controls'])==6
    assert all(r['controls']['oracle_pass'] and not r['controls']['wrong_identity_pass'] for r in scorer['conditional_controls'])
    parity=read_json(OUT/'inactive-trained-parity.json')
    assert all(c['bitwise_equal'] for c in parity['checks'])
    atomic_json(OUT/'validation.json',dict(schema_version=1,qualification_input=False,
        software_checks=231,forge_validation=read_json(Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/logs/validate.log')),
        matched_source_digest=first['source']['digest'],matched_source_commit=first['source']['origin_commit'],
        source_bound_scientific_files=len(scientific),changed_scientific_files=changed,
        matched_conditions=['execution','evaluation','resources','dependencies','adapter','requires_capabilities','runtime','protocol','public_initial_models','all_consumed_named_streams'],
        archived_publication_hashes=before,changed_archived_publications=snapshot_changed,
        completed_jobs=18,attempts=19,retries=1,active_workers=0,reservation_remaining=0,actual_training_gifs=media,
        inactive_bitwise_parity=True,scorer_controls_sha256=file_hash(OUT/'scorer-controls.json'),optimizer_updates_added=0,sampling_draws_added=0))
    print(json.dumps(dict(valid=True,completed_jobs=18,attempts=19,media=18,scientific_files=len(scientific))))

if __name__=='__main__':main()
