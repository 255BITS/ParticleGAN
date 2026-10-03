"""Close immutable actual paired capture evidence; CPU reads only."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
STUDY=ROOT.parent
sys.path.insert(0,str(ROOT))
from instrumentation import compare
from run_capture import verify

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

assert not torch.cuda.is_initialized()
before=verify()
full=ROOT/'full-attempt-1'
parity=ROOT/'parity-attempt-1/PARITY-COMPLETION.json'
completed=read(full/'COMPLETION.json')
assert completed['status']=='COMPLETE' and completed['evidence_validity']=='VALID'
assert read(parity)['status']=='PASS'
assert completed['source_integrity_after']==before
inputs=[ROOT/'SOURCE-FREEZE.json',ROOT/'BASELINE-SOURCE-FREEZE.json',
        full/'COMPLETION.json',parity,Path(__file__).resolve()]
scores,diagnostics,arrays={}, {}, []
for variant in ('E22','Atlas'):
    directory=full/variant.lower()/'capture'
    npz=directory/'dense-frames.npz'
    with np.load(npz,allow_pickle=False) as data:
        values={key:data[key].copy() for key in data.files}
    arrays.append(values)
    assert values['frames'].shape==(153,4096,2) and values['frames'].dtype==np.float32
    assert np.isfinite(values['frames']).all()
    events=np.flatnonzero(values['event_kind']=='target_shift')
    assert list(values['steps'][events])==[500,1000]
    assert all(np.array_equal(values['frames'][i],values['frames'][i-1]) for i in events)
    ordinary=values['event_kind']!='target_shift'
    assert np.array_equal(values['steps'][ordinary],np.arange(0,1501,10))
    assert np.all(np.diff(values['steps'])>=0)
    phases=[]
    for start,end in ((0,500),(500,1000),(1000,1500)):
        chosen=(values['event_kind']=='update')&(values['steps']>start)&(values['steps']<=end)
        good=np.flatnonzero(chosen&(values['capture_hq']>=.9))
        phases.append(dict(updates=[start,end],uniform_per10frame_HQ_mean=float(values['capture_hq'][chosen].mean()),
            first_recorded_updates_to_visual_HQ_90_percent=None if not len(good) else int(values['steps'][good[0]])-start))
    verdict=read(directory/'original-frames.npz.verdict.json')
    assert verdict['status']=='PASS' and verdict['turns']==2
    scores[variant]=verdict
    diagnostics[variant]=dict(phase_metrics=phases,
        sample_count=4096,mode_min_HQ_count=10,are_original_acceptance_metrics=False,
        observed_clouds=151,target_shift_events=2,frames=153,
        point_extents=[float(values['frames'].min()),float(values['frames'].max())],
        target_event_clouds_exact=True,point_dtype='float32')
    capture=read(directory/'CAPTURE-COMPLETION.json')
    assert capture['preservation_checks']==151 and capture['all_observations_owned_state_exact']
    inputs.extend(directory/name for name in ('dense-frames.npz','original-frames.npz.verdict.json',
        'CAPTURE-COMPLETION.json','LAUNCH.json','SOURCE-CLOSE.json','training-trace.jsonl','visualization-metrics.jsonl'))
for key in ('steps','angles','event_kind','centers'):
    assert np.array_equal(arrays[0][key],arrays[1][key]),key
assert np.array_equal(arrays[0]['frames'][0],arrays[1]['frames'][0])
checkpoints=[]
for step in (500,1000,1500):
    old=STUDY/f'validation-ra15/moving/rotated100/checkpoint-{step:06d}.pt'
    new=full/f'atlas/capture/checkpoint-{step:06d}.pt'
    a=torch.load(old,map_location='cpu',weights_only=False)
    b=torch.load(new,map_location='cpu',weights_only=False)
    difference=compare(a,b,skip=('birth_death.last.eval_seconds',))
    assert not difference,difference
    checkpoints.append(dict(step=step,exact_semantic_state=True,
        historical_checkpoint_sha256=sha(old),actual_current_checkpoint_sha256=sha(new),
        exclusions=['birth_death.last.eval_seconds']))
    inputs.extend((old,new))
original_verdict=STUDY/'validation-ra15/moving/rotated100/frames.npz.verdict.json'
assert scores['Atlas']==read(original_verdict)
inputs.append(original_verdict)
assert verify()==before and not torch.cuda.is_initialized()
result=dict(status='CLOSED_VALID_BOTH_ORIGINAL_QUALITY_GATES_PASS',
    actual_source_labels=['current PR155 E22 at cabe2084','ParticleGAN Atlas RA17'],
    source_integrity=before,original_seed=1234,original_complete_updates_per_variant=1500,
    original_gate_metrics=scores,separate_visualization_diagnostics=diagnostics,
    all_observation_state_guards_pass=302,initial_observed_cloud_exact=True,
    common_steps_angles_events_centers_exact=True,
    current_Atlas_original_RA15_checkpoint_semantic_parity=checkpoints,
    Atlas_original_gate_metrics_exact=True,
    comparison_scope='Actual fresh native rotated100 initial learning and two original30-degree turns only.',
    interpretation='Atlas reaches90%visual HQ earlier in initial learning and first-turn recovery. Both pass both original gates; E22 has higher final original HQ and100versus98modes.',
    final_Atlas_minus_E22_HQ_percentage_points=round(100*(scores['Atlas']['periods'][-1]['hq']-scores['E22']['periods'][-1]['hq']),4),
    no_interpolation=True,no_seed_experiments=True,no_training_or_GPU_operations_in_closure=True,
    cuda_initialized=False,inputs_sha256={str(p):sha(p) for p in sorted(set(inputs))})
target=HERE/'DATA-RECEIPT.json'
assert not target.exists();target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status=result['status'],data_receipt_sha256=sha(target),source_freeze_sha256=before['source_freeze_sha256'],
    Atlas_checkpoint_steps_exact=[500,1000,1500],original_quality=['PASS','PASS'],cuda_initialized=False),sort_keys=True),flush=True)
