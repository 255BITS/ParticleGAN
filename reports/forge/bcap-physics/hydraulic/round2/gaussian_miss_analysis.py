"""Describe exact saved failed outputs; no training, RNG draws or regrading."""
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import atomic_json,file_hash,read_json


def ks(values,mean,sigma):
    ordered=np.sort(np.asarray(values,dtype=np.float64).reshape(-1))
    cdf=ndtr((ordered-mean)/sigma)
    ranks=np.arange(len(ordered),dtype=np.float64)/len(ordered)
    return max(float(np.max(cdf-ranks)),float(np.max(ranks+1/len(ordered)-cdf)))


def main():
    directory=Path(__file__).parent
    provenance=read_json(directory/'provenance.json')
    receipt=next(r for r in provenance['receipts'] if r['task_id']=='gaussian1d_stability' and r['candidate_id']=='hydraulic-secant-deformation-v2')
    path=Path(receipt['artifact_root'])/'observed-samples.pt'
    certificate=read_json(ROOT/'reports/forge/attempts'/receipt['attempt_id']/'result.json')
    evidence=next(r for r in certificate['task_results'] if r['task_id']=='gaussian1d_stability')['evidence']
    assert file_hash(path)==evidence['saved_observer_outputs']['sha256']
    records=torch.load(path,map_location='cpu',weights_only=True)
    rows=[]
    for record in records:
        step=record['step']
        if not (1000 < step <= 4000 or step > 5000) or record['metrics']['cdf_ks'] <= .05:
            continue
        values=record['samples'].numpy().astype(np.float64).reshape(-1)
        mean,std=float(values.mean()),float(values.std())
        target_mean=2. if step<=4000 else 3.
        target_ks=ks(values,target_mean,.5)
        assert abs(target_ks-record['metrics']['cdf_ks'])<1e-12
        rows.append(dict(step=step,n=len(values),sample_mean=mean,sample_std=std,
            target_mean=target_mean,target_sigma=.5,target_ks=target_ks,
            target_ks_failed_margin=target_ks-.05,
            fitted_normal_ks=ks(values,mean,std),original_metrics=record['metrics']))
    result=dict(schema_version=1,kind='saved_failed_state_descriptive_shape_analysis',
        optimizer_updates_added=0,random_sampling_draws_added=0,qualification_input=False,
        original_attempt_id=receipt['attempt_id'],source_digest=receipt['source_digest'],
        input_file=str(path),input_sha256=file_hash(path),states=rows,
        interpretation='Moment bounds passing does not identify shape as the KS cause. Fitted-normal KS is descriptive only; target-law gates remain FAIL.')
    atomic_json(directory/'gaussian-misses.json',result)
    print(rows)

if __name__=='__main__':main()
