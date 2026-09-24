"""Read-only independent source, reference, and accounting audit of round 7."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.pr84_critic_refinement_capture import _sha
from reports.toy100.pr84_prediction_state_filter import state_hash


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path):return json.loads(path.read_text())


def audit_saved44(directory,capture):
    data=read(directory/'summary.json')
    decl=data['declaration']
    assert data['status']=='PASS' and data['warm_eligible'] is True
    for name,digest in decl['sources'].items():
        assert sha((directory/'source'/name).read_bytes())==digest
    state_file=capture/'selected-states.pt'
    assert sha(state_file.read_bytes())==decl['states_sha256']
    states=torch.load(state_file,weights_only=True,map_location='cpu')
    diagnosis=read(capture/'diagnosis.json')
    assert diagnosis['selected_states_sha256']==decl['states_sha256']
    reference={row['step']:row for row in diagnosis['rows']}
    total=0
    windows=[]
    for window in data['branches']:
        start,end=window['start'],window['end']
        expected=list(range(start,end+1));n=len(expected)
        ordinary=window['variants']['original']
        guarded=window['variants']['reallocation']
        assert [r['step'] for r in ordinary['points']]==expected
        assert [r['step'] for r in guarded['points']]==expected
        assert guarded['local_gate']['checks']==n and guarded['local_gate']['passing_checks']==n
        assert guarded['local_gate']['pass_all'] and guarded['local_gate']['min_modes']==8
        assert guarded['local_gate']['min_hq']>=.9
        assert guarded['rng_final_sha256']==ordinary['rng_final_sha256']
        assert guarded['noise']==ordinary['noise']
        assert guarded['moment_steps']==ordinary['moment_steps']=={'d':[end],'g':[end]}
        assert guarded['rates']==ordinary['rates']
        assert [r['step'] for r in guarded['rates'] if r['role']=='d']==expected
        assert [r['step'] for r in guarded['rates'] if r['role']=='g_prior']==expected
        assert all(r['rates']==[.00425] if r['role']=='d' else r['rates']==[.00425,.0085]
                   for r in guarded['rates'])
        dynamics=guarded['dynamics'];corrections=dynamics['corrections']
        assert len(corrections)==len(dynamics['records'])==n
        assert dynamics['native_batch_checks']==3*n
        assert dynamics['correction_rng_checks']==dynamics['correction_owner_checks']==n
        assert dynamics['records'][0]['d']==ordinary['dynamics']['records'][0]['d']
        assert [r['grouping']['observation']['status'] for r in corrections[:2]]==[
            'PENDING_FIRST_BANK','CONFIRMED']
        assert all(r['grouping']['observation']['status']=='UPDATED_FIXED_PARTITION'
                   for r in corrections[2:])
        refs=corrections[1]['grouping']['reference_centers']
        assert len(refs)==8
        assert all(r['grouping']['reference_centers']==refs for r in corrections[2:])
        assert all(r['learner_state_sha256'] for r in corrections)
        for row,record,accepted in zip(ordinary['points'],ordinary['dynamics']['records'],ordinary['accepted_states']):
            step=row['step'];ref=reference[step]
            assert row['support']==ref['stages']['bounded_joint']
            assert record==dict(ref['stages']['record'],outer_step=step-start+1)
            if step in states:
                assert accepted['accepted_state_sha256']==state_hash(states[step]['post_bounded_g'])
        windows.append(dict(start=start,end=end,checks=n,min_hq=guarded['local_gate']['min_hq'],
                            min_modes=guarded['local_gate']['min_modes'],
                            original_passing=ordinary['local_gate']['passing_checks'],
                            candidate_passing=guarded['local_gate']['passing_checks'],
                            confirmation_groups=len(refs),
                            correction_rng_checks=dynamics['correction_rng_checks'],
                            native_phase_checks=dynamics['native_batch_checks']))
        total+=n
    assert total==44
    return dict(source_sha256=sha((directory/'summary.json').read_bytes()),
                capture_sha256=decl['states_sha256'],windows=windows,total_checks=total)


def audit_native3(directory):
    manifest=read(directory/'manifest.json')
    for name,receipt in manifest['files'].items():
        compressed=(directory/name).read_bytes()
        assert sha(compressed)==receipt['sha256']
        assert sha(gzip.decompress(compressed))==receipt['raw_sha256']
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress((directory/'source.tar.gz').read_bytes()))) as bundle:
        assert set(bundle.getnames())==set(manifest['source_sha256'])
        for item in bundle:
            assert sha(bundle.extractfile(item).read())==manifest['source_sha256'][item.name]
    data=json.loads(gzip.decompress((directory/'summary.json.gz').read_bytes()))
    assert data['status']=='COMPLETE'
    assert data['first_two_updates_exact_across_branches']
    assert data['rng_final_exact_across_branches']
    assert data['declaration']['source']==manifest['source_sha256']
    ordinary=data['branches']['ordinary']
    mode0=None
    counts={}
    for name,branch in data['branches'].items():
        receipt=branch['receipt'];corrections=branch['dynamics']['corrections']
        assert [r['step'] for r in receipt['checkpoints']]==[2401,2402,2403]
        assert all(r['modes']==8 and r['hq']==1. for r in receipt['checkpoints'])
        assert receipt['actual_adam_updates']=={'d':3,'g':3}
        assert receipt['optimizer_callbacks']=={'d':9,'g':9}
        assert [r['grouping']['observation']['status'] for r in corrections]==[
            'PENDING_FIRST_BANK','CONFIRMED','UPDATED_FIXED_PARTITION']
        assert corrections[0]['selected']=='rest'
        assert all(r['fit']['status']=='CONVERGED' for r in corrections[1:])
        assert branch['calls'][:12]==ordinary['calls'][:12]
        assert branch['dynamics']['corrections'][:2]==ordinary['dynamics']['corrections'][:2]
        assert branch['dynamics']['records'][:2]==ordinary['dynamics']['records'][:2]
        assert branch['rng_final_sha256']==ordinary['rng_final_sha256']
        for step in range(2401,2404):
            d=[r['bank_sha256'] for r in branch['calls'] if r['update']==step and r['is_d']]
            assert len(d)==3 and len(set(d))==1
        refs=torch.tensor(corrections[1]['grouping']['reference_centers'],dtype=torch.float64)
        this_mode0=int(torch.cdist(refs,mode_hold.ring_means()[:1].double()).argmin())
        if mode0 is None:mode0=this_mode0
        assert this_mode0==mode0
        assert corrections[2]['grouping']['reference_centers']==corrections[1]['grouping']['reference_centers']
        counts[name]=(corrections[1]['grouping']['accumulated_counts'][mode0],
                      corrections[2]['grouping']['accumulated_counts'][mode0])
        saved=torch.load(io.BytesIO(gzip.decompress((directory/(name+'-final-state.pt.gz')).read_bytes())),
                         weights_only=True,map_location='cpu')
        assert _sha(saved)==branch['final_state_sha256']
        assert _sha(saved['rng'])==branch['rng_final_sha256']
    assert counts['ordinary']==(31,53)
    assert counts['omitted_third_D']==(31,31)
    assert counts['singleton_third_D']==(31,32)
    return dict(manifest_sha256=sha((directory/'manifest.json').read_bytes()),
                branches=list(data['branches']),mode0_identity=mode0,mode0_counts=counts,
                grade='all nine native checkpoints have eight modes/HQ 1')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--saved44',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--native3',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    result=dict(saved44=audit_saved44(a.saved44,a.capture),native3=audit_native3(a.native3),
                audit_source_sha256=sha(Path(__file__).read_bytes()),status='PASS')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],saved44_checks=result['saved44']['total_checks'],
                          native3_mode0_counts=result['native3']['mode0_counts'])))


if __name__=='__main__':main()
