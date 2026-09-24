"""Offline population and held-out check of the warm one-bank mode loss.

True ring centers are used only to diagnose a frozen, already rejected
free-output proposal. They never enter a candidate update or acceptance rule.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_free1200 import initial_support,load_states,sha
from reports.toy100.sample_anchor_local_mmd_continuation import j_emitted,native_bank
from reports.toy100.sample_anchor_local_mmd_filter import local_width
from reports.toy100.sample_anchor_mmd_filter import squared_distances


def population_j(points,width,means,real_sigma,emitted_sigma):
    y=points.double();mu=means.double();h2=width**2
    cross_variance=h2+real_sigma**2+emitted_sigma**2
    fake_variance=h2+2*emitted_sigma**2
    d=y.shape[1]
    pq=(h2/cross_variance)**(d/2)*torch.exp(
        -squared_distances(mu,y)/(2*cross_variance)).mean()
    qq=(h2/fake_variance)**(d/2)*torch.exp(
        -squared_distances(y,y)/(2*fake_variance)).mean()
    return float(qq-2*pq)


def paired_delta_samples(real,before,after,width,sigma):
    x=real.double();a=before.double();b=after.double();h2=width**2;s2=sigma**2
    pref_pq=(h2/(h2+s2))**(x.shape[1]/2)
    pref_qq=(h2/(h2+2*s2))**(x.shape[1]/2)
    cross_a=pref_pq*torch.exp(-squared_distances(x,a)/(2*(h2+s2))).mean(1)
    cross_b=pref_pq*torch.exp(-squared_distances(x,b)/(2*(h2+s2))).mean(1)
    qq_a=pref_qq*torch.exp(-squared_distances(a,a)/(2*(h2+2*s2))).mean()
    qq_b=pref_qq*torch.exp(-squared_distances(b,b)/(2*(h2+2*s2))).mean()
    return (qq_b-qq_a)-2*(cross_b-cross_a)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cumulative',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    raw=a.cumulative.read_bytes();previous=json.loads(raw)
    assert previous['status']=='COMPLETE' and previous['gates']['warm_all16'] is False
    for name,digest in previous['declaration']['source_sha256'].items():
        assert sha((ROOT/name).read_bytes())==digest
    cold,warm,inputs=load_states();means=mode_hold.ring_means()
    cold_stream=torch.Generator().set_state(cold['rng']['data'])
    width,_=local_width(mode_hold.sample_ring(means,128,mode_hold.SIGMA,cold_stream))
    assert width==previous['declaration']['frozen_width']
    stream=torch.Generator().set_state(warm['rng']['data'])
    proposal_bank=native_bank(stream,means)
    first=previous['warm_full']['rows'][0]
    assert sha(proposal_bank.contiguous().numpy().tobytes())==first['real_bank_sha256']
    initial=initial_support(warm).double()
    cloud=initial.clone();rows=[]
    population_initial=population_j(initial,width,means,mode_hold.SIGMA,.029)
    move8_before=move8_after=move9_before=move9_after=None
    for item in first['donor_moves']:
        old=cloud.clone()
        cloud[item['donor_index']]=proposal_bank[item['real_index']].double()
        ordinal=item['move']
        rows.append(dict(move=ordinal,population_j_before=population_j(old,width,means,mode_hold.SIGMA,.029),
                         population_j_after=population_j(cloud,width,means,mode_hold.SIGMA,.029),
                         empirical_j_after=float(j_emitted(proposal_bank.double(),cloud,width,.029)),
                         observed_modes=item['grade']['modes']))
        if ordinal==8:move8_before,move8_after=old,cloud.clone()
        if ordinal==9:move9_before,move9_after=old,cloud.clone()
    assert move9_before is not None and first['donor_moves'][8]['grade']['modes']==7
    final=torch.tensor(first['points'],dtype=torch.float64)
    assert abs(float(j_emitted(proposal_bank.double(),final,width,.029))-first['after_j'])<1e-12
    heldout=[native_bank(stream,means) for _ in range(16)]
    bank2=heldout[0].double();all16=torch.cat(heldout).double()
    whole_second=float(j_emitted(bank2,final,width,.029)-j_emitted(bank2,initial,width,.029))
    move9_second=paired_delta_samples(bank2,move9_before,move9_after,width,.029)
    move9_all=paired_delta_samples(all16,move9_before,move9_after,width,.029)
    estimate=float(move9_all.mean());se=float(move9_all.std(unbiased=True)/math.sqrt(len(move9_all)))
    result=dict(scope='posthoc diagnostic only; known target ring centers never enter candidate update',
        cumulative_sha256=sha(raw),source_sha256=sha(Path(__file__).read_bytes()),
        inputs=inputs,width=width,proposal_bank_sha256=first['real_bank_sha256'],
        proposal_bank_mode_counts=torch.bincount(torch.cdist(proposal_bank,means).argmin(1),minlength=8).tolist(),
        initial_clean_mode_counts=torch.bincount(torch.cdist(initial.float(),means).argmin(1),minlength=8).tolist(),
        final_clean_mode_counts=torch.bincount(torch.cdist(final.float(),means).argmin(1),minlength=8).tolist(),
        population_initial_j=population_initial,
        population_final_j=population_j(final,width,means,mode_hold.SIGMA,.029),
        whole_proposal_second_bank_delta_j=whole_second,
        moves=rows,
        move9=dict(population_delta_j=rows[8]['population_j_after']-rows[8]['population_j_before'],
            second_bank_delta_j=float(move9_second.mean()),
            later16_bank_samples=len(all16),later16_bank_mean_delta_j=estimate,
            later16_bank_standard_error=se,
            later16_bank_nominal_95_interval=[estimate-1.96*se,estimate+1.96*se],
            inference='fixed posthoc paired-sample normal interval; not a sequential safety certificate'),
        conclusion='whole proper-population MMD improves despite losing a covered mode; move9 alone harms population MMD but one heldout128 bank mis-signs it',
        no_training=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(population_initial_j=population_initial,
        population_final_j=result['population_final_j'],
        whole_second_delta_j=whole_second,move9=result['move9'])))


if __name__=='__main__':main()
