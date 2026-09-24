"""Free-output bootstrap checks on two distinct native-sized real banks.

The data generator is restored from a passing host state. Between D-bank
draws, this script consumes exactly the native prior-index and G-real draws
shown in mode_hold.run_toy, without executing any model/optimizer updates.
Conditioned omission and singleton branches are diagnostic target subsets.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.coverage_fixed_eval import fixed_draw,score_support
from reports.toy100.sample_anchor_free1200 import initial_support
from reports.toy100.sample_group_anchor import mst_groups,output_mm_step
from reports.toy100.sample_group_two_bank_memory import TwoBankFixedSupportMemory
from reports.toy100.pr84_critic_refinement_capture import _sha


SOURCES=("reports/toy100/sample_group_two_bank_filter.py",
         "reports/toy100/sample_group_two_bank_memory.py",
         "reports/toy100/sample_group_anchor.py",
         "reports/toy100/sample_anchor_free1200.py",
         "reports/toy100/coverage_fixed_eval.py",
         "benchmarks/locked_shared/mode_hold.py",
         "particlegan/particle_prior.py")


def step_data(stream,means,source=None):
    real=mode_hold.sample_ring(means if source is None else source,128,mode_hold.SIGMA,stream)
    # ParticlePrior.sample twice consumes indices only (sigma_rel=0 here).
    torch.randint(0,12,(128,),generator=stream)
    torch.randint(0,12,(128,),generator=stream)
    mode_hold.sample_ring(means,128,mode_hold.SIGMA,stream)
    return real


def singleton_bank(stream,means,reference_bank):
    centers,groups=mst_groups(reference_bank)
    index=int(torch.cdist(centers,means[:1].double()).argmin())
    members=reference_bank[groups["member_indices"][index]].double()
    radius=float((members-members.mean(0)).square().sum(1).mean().sqrt())
    position=int(torch.randint(0,128,(1,),generator=stream))
    labels=torch.randint(1,8,(127,),generator=stream)
    labels=torch.cat((labels[:position],torch.zeros(1,dtype=torch.long),labels[position:]))
    for attempt in range(1,101):
        noise=torch.randn(128,2,generator=stream)
        real=means[labels]+mode_hold.SIGMA*noise
        distance=float(torch.linalg.vector_norm(real[position].double()-centers[index]))
        if distance>radius:
            return real,dict(position=position,accepted_noise_bank_attempt=attempt,
                singleton_distance=distance,first_bank_group_rms=radius,
                label_event_probability=128*(1/8)*(7/8)**127)
    raise RuntimeError("singleton tail did not appear within fixed diagnostic budget")


def one_mm(points,centers):
    return torch.tensor(output_mm_step(points.double(),centers)["target"],dtype=torch.float64)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    raw=args.state.read_bytes();saved=torch.load(args.state,weights_only=True,map_location='cpu')
    previous=json.loads(args.previous.read_text())
    assert hashlib.sha256(raw).hexdigest()==previous['input_file_sha256']
    assert _sha(saved)==previous['input_state_sha256']
    points=initial_support(saved).double();means=mode_hold.ring_means()
    index,noise=fixed_draw(2401,points.float())
    grade=lambda p:score_support(p.float(),index,noise,means)
    rng_before=torch.random.get_rng_state().clone()
    fresh=lambda:torch.Generator().set_state(saved['rng']['data'])
    cases={}

    stream=fresh();first=step_data(stream,means)
    assert torch.equal(first,torch.tensor(previous['branches']['ordinary']['real128']))
    second=step_data(stream,means)
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    pending=memory.observe(first,bank_id=2401)
    pending_state=memory.state_dict()
    assert pending['status']=='PENDING_FIRST_BANK'
    resumed=TwoBankFixedSupportMemory(expected_first_bank_id=1)
    resumed.load_state_dict(pending_state)
    assert resumed.state_dict()['pending']['bank_sha256']==pending_state['pending']['bank_sha256']
    confirmed=resumed.observe(second,bank_id=2402)
    assert confirmed['status']=='CONFIRMED' and confirmed['confirmed_groups']==8
    assert confirmed['comparison']['margin']>0
    confirmed_state=resumed.learner_state_dict()
    restored=TwoBankFixedSupportMemory()
    restored.load_learner_state_dict(confirmed_state)
    assert torch.equal(restored.centers(),resumed.centers())
    assert all(torch.equal(a,b) for a,b in zip(restored.reference_centers,resumed.reference_centers))
    bad_certificate=deepcopy(confirmed_state)
    bad_certificate['confirmation']['margin']+=.01
    try:
        TwoBankFixedSupportMemory().load_learner_state_dict(bad_certificate)
    except ValueError:
        malformed_certificate_rejected=True
    else:
        raise AssertionError('altered confirmation certificate was accepted')
    try:
        restored.observe(second,bank_id=2404)
    except RuntimeError:
        skipped_native_bank_rejected=True
    else:
        raise AssertionError('skipped native real bank was accepted')
    target=one_mm(points,resumed.centers())
    assert grade(target)['modes']==8 and grade(target)['hq']>=.9
    cases['two_native_full_banks']=dict(first=pending,second=confirmed,
        first_bank_sha256=pending['bank_sha256'],second_bank_sha256=confirmed['bank_sha256'],
        distinct_bank_hashes=pending['bank_sha256']!=confirmed['bank_sha256'],
        target_grade=grade(target),pending_resume_exact=True,
        confirmed_resume_exact=True,malformed_certificate_rejected=malformed_certificate_rejected,
        skipped_native_bank_rejected=skipped_native_bank_rejected)

    # Once confirmed, an absent group keeps its accumulated sufficient stats.
    stream=fresh();first=step_data(stream,means);second=step_data(stream,means)
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    memory.observe(first,bank_id=2401);memory.observe(second,bank_id=2402)
    frozen=torch.stack(memory.reference_centers).clone()
    mode0=int(torch.cdist(frozen,means[:1].double()).argmin())
    old_count=memory.confirmed_counts[mode0]
    omitted=step_data(stream,means,source=means[1:])
    update=memory.observe(omitted,bank_id=2403)
    assert update['status']=='UPDATED_FIXED_PARTITION' and memory.confirmed_counts[mode0]==old_count
    assert torch.equal(torch.stack(memory.reference_centers),frozen)
    cases['confirmed_then_omitted']=dict(update=update,absent_count_preserved=True,
        reference_centers_bitwise_preserved=True,target_grade=grade(one_mm(points,memory.centers())))

    # A full first bank followed by an independent missing bank cannot confirm.
    stream=fresh();first=step_data(stream,means);omitted=step_data(stream,means,source=means[1:])
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    a=memory.observe(first,bank_id=2401);b=memory.observe(omitted,bank_id=2402)
    assert a['bank']['groups']==8 and b['bank']['groups']==7
    assert b['status']=='UNRESOLVED_REPLACED' and not memory.confirmed
    cases['full_then_omitted']=dict(first=a,second=b,held_support_grade=grade(points))

    # Missing-first bootstrap waits for two later compatible complete banks.
    stream=fresh();omitted=step_data(stream,means,source=means[1:])
    full1=step_data(stream,means);full2=step_data(stream,means)
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    a=memory.observe(omitted,bank_id=2401)
    b=memory.observe(full1,bank_id=2402)
    c=memory.observe(full2,bank_id=2403)
    assert [row['status'] for row in (a,b,c)]==['PENDING_FIRST_BANK','UNRESOLVED_REPLACED','CONFIRMED']
    assert c['confirmed_groups']==8
    seven_centers,_=mst_groups(omitted)
    seven_support=one_mm(points,seven_centers)
    first_repair=one_mm(seven_support,memory.centers())
    second_repair=one_mm(first_repair,memory.centers())
    assert grade(seven_support)['modes']==7 and grade(second_repair)['modes']==8
    cases['incomplete_then_discovered']=dict(first=a,second=b,third=c,
        diagnostic_seven_support_grade=grade(seven_support),
        first_mm_grade=grade(first_repair),second_mm_grade=grade(second_repair))

    # The previously troublesome singleton is paired with its true identity,
    # instead of becoming a ninth remembered group.
    stream=fresh();first=step_data(stream,means)
    singleton,condition=singleton_bank(stream,means,first)
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    a=memory.observe(first,bank_id=2401);b=memory.observe(singleton,bank_id=2402)
    assert b['status']=='CONFIRMED' and b['confirmed_groups']==8
    assert b['comparison']['margin']>0
    cases['full_then_singleton_tail']=dict(first=a,second=b,conditioning=condition,
        target_grade=grade(one_mm(points,memory.centers())))

    # Two agreeing incomplete banks may still confirm the wrong support set;
    # this is the declared finite-sample limitation, not a hidden PASS.
    stream=fresh();first=step_data(stream,means,source=means[1:])
    second=step_data(stream,means,source=means[1:])
    memory=TwoBankFixedSupportMemory(expected_first_bank_id=2401)
    a=memory.observe(first,bank_id=2401);b=memory.observe(second,bank_id=2402)
    cases['two_agreeing_incomplete_banks']=dict(first=a,second=b,
        finite_sample_limitation=(b['status']=='CONFIRMED' and b['confirmed_groups']==7))
    assert cases['two_agreeing_incomplete_banks']['finite_sample_limitation']

    try:
        TwoBankFixedSupportMemory(expected_first_bank_id=1).observe(first,bank_id=2401)
    except RuntimeError:
        fresh_late_start_rejected=True
    else:
        raise AssertionError('empty memory silently restarted at step2401')
    assert torch.equal(torch.random.get_rng_state(),rng_before)
    result=dict(scope='two or three distinct native-sized D-bank draws per controlled branch; no model/optimizer update',
        state_file_sha256=hashlib.sha256(raw).hexdigest(),input_state_sha256=_sha(saved),
        previous_result_sha256=hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},
        method='two-bank reciprocal support confirmation, then fixed-reference per-real-sample partition',
        cases=cases,fresh_late_start_rejected=fresh_late_start_rejected,
        torch_rng_unchanged=True,
        native_stream_scope='one D real, two prior-index draws, one G real per outer update; no model training')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:(row.get('third') or row.get('second') or row.get('update'))['status']
                      for name,row in cases.items()}))


if __name__=='__main__':main()
