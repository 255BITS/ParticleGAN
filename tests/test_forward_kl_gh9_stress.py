"""Finite-GH9 whole-map acceptance and exact-rest tests."""

import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100 import forward_kl_gh9_stress as method
from reports.toy100.forward_kl_free_filter import (
    em_centroids as real_em, log_kernels, quadrature,
)
from reports.toy100.sample_anchor_free1200 import initial_support, load_states


ROOT=Path(__file__).resolve().parents[1]
FIRST=ROOT/'reports/toy100/continuous-evidence/round8-forward-kl-first-bank/v2/result.json'


def fixture():
    cold,_,_=load_states()
    record=json.loads(FIRST.read_text())
    stream=torch.Generator().set_state(cold['rng']['data'])
    bank=mode_hold.sample_ring(mode_hold.ring_means(),128,mode_hold.SIGMA,stream)
    points=initial_support(cold).double()
    width=record['declaration']['frozen_width']
    return bank,points,width


def test_native_first_bank_whole_selected_lowers_finite_gh9():
    bank,points,width=fixture()
    selected,row=method.optimize(bank,bank,points,width,0.,
                                 means=mode_hold.ring_means())
    assert row['selected']!='EXACT_REST'
    assert row['final_audit9']<row['initial_audit9']-row['gh9_strict_tolerance']
    assert row['donor_candidate_points']==128
    assert bool(torch.isfinite(selected).all())


def test_gh5_noop_still_uses_gh9_signal(monkeypatch):
    bank,points,width=fixture()
    monkeypatch.setattr(method,'global_donor',lambda candidates,current,*args,**kwargs:
        (current.clone(),[]))
    calls=[]
    def controlled_em(current,locations,weights,variance,*,limit,audit):
        calls.append(limit)
        if limit==20:
            return current.clone(),[]
        return real_em(current,locations,weights,variance,limit=limit,audit=audit)
    monkeypatch.setattr(method,'em_centroids',controlled_em)
    selected,row=method.optimize(bank,bank,points,width,0.,
                                 means=mode_hold.ring_means())
    assert calls==[20,1]
    assert row['selected']=='GH9_ONE_EM_FALLBACK'
    assert row['final_audit9']<row['initial_audit9']-row['gh9_strict_tolerance']
    assert not torch.equal(selected,points)
    target,weights=quadrature(bank,width,9)
    variance=width**2
    responsibilities=torch.softmax(log_kernels(target,points,variance),dim=1)
    mass=(responsibilities*weights[:,None]).sum(0)
    mm_decrease=float((mass[:,None]*(selected-points).square()).sum()/(2*variance))
    assert row['initial_audit9']-row['final_audit9']>=mm_decrease-1e-10


def test_two_noop_proposals_restore_exact_output(monkeypatch):
    bank,points,width=fixture()
    monkeypatch.setattr(method,'global_donor',lambda candidates,current,*args,**kwargs:
        (current.clone(),[]))
    monkeypatch.setattr(method,'em_centroids',lambda current,*args,**kwargs:
        (current.clone(),[]))
    selected,row=method.optimize(bank,bank,points,width,0.,
                                 means=mode_hold.ring_means())
    assert row['selected']=='EXACT_REST'
    torch.testing.assert_close(selected,points,atol=0,rtol=0)
    assert row['initial_audit9']==row['final_audit9']
