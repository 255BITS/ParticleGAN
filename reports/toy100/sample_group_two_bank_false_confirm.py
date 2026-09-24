"""Three-update falsifier: two omitted D banks confirm seven, then a full bank.

Conditioning changes only the first two D real banks. G real banks and the
fixed eight-mode target remain native. Mode labels grade this diagnostic only;
the candidate receives raw samples and its own inferred support memory.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import sample_anchor_own_state_probe as own
from reports.toy100.sample_anchor_memory_candidate import METHOD,sample_anchor_memory_candidate
from reports.toy100.sample_group_two_bank_native3 import SOURCE_NAMES
from reports.toy100.pr84_critic_refinement_capture import _sha


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--ordinary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False)
    saved_bytes=a.state.read_bytes();saved=torch.load(a.state,weights_only=True,map_location='cpu')
    ordinary=json.loads(a.ordinary.read_text())['branches']['ordinary']
    assert saved['noise']['step_calls']==2400 and saved['noise_policy']['total_steps']==1200
    config_bytes=a.config.read_bytes();config=json.loads(config_bytes)
    assert config['lr_floor']==1. and config['lr_anneal_start']==0.
    recipe,noise,_=declared_recipe(config)
    sources={}
    for name in sorted(SOURCE_NAMES|{'reports/toy100/sample_group_two_bank_false_confirm.py'}):
        raw=(ROOT/name).read_bytes();sources[name]=sha(raw)
        path=a.output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    declaration=dict(method=METHOD,scope='three native neural updates from exact passing own-state2400',
        condition='first two D real128 banks omit mode0; third D and all G banks use full fixed target',
        input_state_sha256=_sha(saved),input_file_sha256=sha(saved_bytes),
        config_sha256=sha(config_bytes),ordinary_summary_sha256=sha(a.ordinary.read_bytes()),
        source=sources,learner_first_bank_id=2401,nominal_rates=dict(d=.00425,g=.00425,prior=.0085),
        noise_horizon=1200,shared_gate_eligible=False)
    (a.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='DECLARED',declaration=declaration)),flush=True)
    calls=[]
    @contextmanager
    def selected(**kwargs):
        kwargs['start_step']=2400
        with sample_anchor_memory_candidate(**kwargs) as (recorder,generated):
            ordinary_sample=mode_hold.sample_ring
            def sample(means,n,sigma,stream):
                index=len(calls);update=index//6;phase=index%6//2;is_d=index%2==0
                if update>2 or phase!=recorder.phase or n!=128:
                    raise RuntimeError('native D/G sampler order changed')
                conditioned=update<2 and is_d
                real=ordinary_sample(means[1:] if conditioned else means,n,sigma,stream)
                labels=torch.cdist(real,mode_hold.ring_means()).argmin(1)
                rejected_mode0=rejected_other=None
                if update==2 and is_d:
                    refs=torch.stack(recorder.memory.reference_centers)
                    rejected=torch.cdist(real.double(),refs).min(1).values>=recorder.memory.fixed_half_separation
                    rejected_mode0=int((rejected&(labels==0)).sum())
                    rejected_other=int((rejected&(labels!=0)).sum())
                calls.append(dict(step=2401+update,phase=phase,is_d=is_d,
                    conditioned=conditioned,mode0_sample_count=int((labels==0).sum()),
                    rejected_mode0=rejected_mode0,rejected_other=rejected_other,
                    sha256=sha(real.detach().cpu().contiguous().numpy().tobytes())))
                return real
            with patch.object(mode_hold,'sample_ring',sample):yield recorder,generated
    branch,generated=own.run_bound(saved,recipe,noise,selected,completed=2400,target=2403)
    corrections=branch['dynamics']['corrections']
    assert len(calls)==18 and len(corrections)==3
    assert branch['receipt']['actual_adam_updates']=={'d':3,'g':3}
    assert branch['receipt']['optimizer_callbacks']=={'d':9,'g':9}
    assert [r['grouping']['observation']['status'] for r in corrections]==[
        'PENDING_FIRST_BANK','CONFIRMED','UPDATED_FIXED_PARTITION']
    assert len(corrections[1]['centers'])==len(corrections[2]['centers'])==7
    assert corrections[2]['grouping']['reference_centers']==corrections[1]['grouping']['reference_centers']
    assert calls[12]['mode0_sample_count']>0
    assert calls[12]['rejected_mode0']==calls[12]['mode0_sample_count']
    assert calls[12]['rejected_other']==0
    for update in range(3):
        d=[r for r in calls if r['step']==2401+update and r['is_d']]
        assert len(d)==3 and len({r['sha256'] for r in d})==1
        assert all(r['conditioned']==(update<2) for r in d)
    assert branch['receipt']['checkpoints'][0]['modes']==8
    assert _sha(branch['state']['rng'])==ordinary['rng_final_sha256']
    saved_path=a.output/'final-state.pt';torch.save(branch['state'],saved_path)
    result=dict(status='COMPLETE',declaration=declaration,calls=calls,
        checkpoints=branch['receipt']['checkpoints'],receipt=branch['receipt'],
        corrections=corrections,records=branch['dynamics']['records'],
        third_bank_mode0_samples=calls[12]['mode0_sample_count'],
        third_bank_rejected=corrections[2]['grouping']['observation']['rejected'],
        third_bank_accepted=corrections[2]['grouping']['observation']['accepted'],
        third_bank_rejected_mode0=calls[12]['rejected_mode0'],
        third_bank_rejected_other=calls[12]['rejected_other'],
        confirmed_groups=7,final_state_sha256=_sha(branch['state']),
        final_file_sha256=sha(saved_path.read_bytes()),
        final_rng_sha256=_sha(branch['state']['rng']),
        generated_source_sha256=sha(generated.encode()),
        ordinary_final_rng_exact=True,shared_gate_eligible=False)
    (a.output/'result.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(event='DONE',status=result['status'],
        grades=[(r['step'],r['modes'],r['hq']) for r in result['checkpoints']],
        third_mode0=result['third_bank_mode0_samples'],
        third_rejected=result['third_bank_rejected'])),flush=True)


if __name__=='__main__':main()
