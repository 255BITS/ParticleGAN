"""Normal-path bit parity and exact-rest remembered-donor rescue."""

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import forward_kl_gh9_stress as frozen
from reports.toy100 import forward_kl_gh9_remembered as wrapper
from reports.toy100.forward_kl_free_filter import cross_entropy
from reports.toy100.sample_anchor_free1200 import initial_support, load_states


WIDTH=.031286240422040236


def test_normal_saved_state_paths_have_exact_frozen_output_receipts_and_rng():
    cold,warm,_=load_states()
    for state,sigma in ((cold,0.),(warm,.029)):
        stream=torch.Generator().set_state(state['rng']['data'])
        bank=mode_hold.sample_ring(mode_hold.ring_means(),128,mode_hold.SIGMA,stream)
        points=initial_support(state).double()
        pristine=points.clone()
        rng=torch.random.get_rng_state().clone()
        original,old_row=frozen.optimize(bank,bank,points,WIDTH,sigma,
                                          means=mode_hold.ring_means())
        selected,row=wrapper.optimize(bank,bank,points,WIDTH,sigma,
                                       means=mode_hold.ring_means())
        assert old_row['selected']!='EXACT_REST'
        torch.testing.assert_close(selected,original,atol=0,rtol=0)
        assert row==old_row
        torch.testing.assert_close(points,pristine,atol=0,rtol=0)
        assert torch.equal(torch.random.get_rng_state(),rng)
        no_oracle,receipt=wrapper.optimize(bank,bank,points,WIDTH,sigma,means=None)
        torch.testing.assert_close(no_oracle,original,atol=0,rtol=0)
        assert 'initial_quality' not in receipt and 'final_quality' not in receipt
        for key,value in receipt.items():
            assert value==old_row[key]
        api_target,api_receipt,rule=wrapper.propose_target(bank,bank,points,WIDTH,
                                                            sigma)
        torch.testing.assert_close(api_target,original,atol=0,rtol=0)
        assert api_receipt==receipt
        variance=WIDTH**2+sigma**2
        assert abs(float(cross_entropy(*rule,points,variance))-
                   receipt['initial_audit9'])<1e-12
        assert abs(float(cross_entropy(*rule,api_target,variance))-
                   receipt['final_audit9'])<1e-12


def test_fixed_target_trap_rescued_only_after_exact_rest():
    left=torch.tensor([[-2.,0.]],dtype=torch.float32).repeat(128,1)
    middle=torch.zeros((128,2),dtype=torch.float32)
    right=torch.tensor([[2.,0.]],dtype=torch.float32).repeat(128,1)
    history=torch.cat((left,right,middle),0)
    collapsed=torch.zeros((12,2),dtype=torch.float64)
    original,old_row=frozen.optimize(history,middle,collapsed,WIDTH,.029,
                                     means=mode_hold.ring_means())
    assert old_row['selected']=='EXACT_REST'
    assert torch.equal(original,collapsed)
    rng=torch.random.get_rng_state().clone()
    selected,row=wrapper.optimize(history,middle,collapsed,WIDTH,.029,
                                   means=None)
    assert row['selected']=='REMEMBERED_GH9_DONOR'
    assert row['frozen_selection']=='EXACT_REST'
    assert row['remembered_search']['status']=='STRICT_FINITE_GH9_DONOR'
    assert row['final_audit9']<row['initial_audit9']-row['gh9_strict_tolerance']
    assert not torch.equal(selected,collapsed)
    assert torch.equal(torch.random.get_rng_state(),rng)
    torch.testing.assert_close(collapsed,torch.zeros_like(collapsed),atol=0,rtol=0)
    assert 'initial_quality' not in row and 'final_quality' not in row


def test_no_remembered_signal_keeps_exact_rest_and_records_full_scan():
    bank=torch.zeros((128,2),dtype=torch.float32)
    points=torch.zeros((12,2),dtype=torch.float64)
    selected,row=wrapper.optimize(bank,bank,points,WIDTH,.029,means=None)
    assert row['selected']=='EXACT_REST'
    assert row['remembered_search']['status']=='NO_SINGLE_REMEMBERED_DONOR'
    assert row['remembered_search']['inspected_candidates']==128
    assert torch.equal(selected,points)
    assert row['final_audit9']==row['initial_audit9']
