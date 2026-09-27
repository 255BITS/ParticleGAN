import pytest
import torch

from benchmarks.locked_shared.observation import checkpoint, recording, sustained


def test_transient_pass_does_not_count_and_suffix_has_confirmation():
    curve = [{"step": (i+1)*50, "seconds": float(i+1), "modes": 8, "hq": hq}
             for i,hq in enumerate([1,1,1,1,1,.8,1,1,1,1,1])]
    expected=[p['step'] for p in curve]
    score=sustained(curve, [("modes", ">=",8),("hq",">=",.9)],expected_steps=expected)
    assert score['first_pass_step']==50
    assert score['stable_from_step']==350
    assert score['confirmed_step']==550
    assert score['confirmed_seconds']==11
    assert sustained(curve[:-1],[("hq",">=",.9)],expected_steps=expected)['stable_from_step'] is None
    curve[-1]['hq']=float('nan')
    assert sustained(curve,[("hq",">=",.9)],expected_steps=expected)['stable_from_step'] is None
    with pytest.raises(ValueError):
        sustained([curve[0],curve[0]],[("hq",">=",.9)],expected_steps=expected)


def test_observation_preserves_rng_and_nested_context_restores():
    torch.manual_seed(0)
    expected=torch.rand(3)
    torch.manual_seed(0)
    with recording(24) as outer:
        checkpoint(1,lambda: {'value':float(torch.rand(()))})
        with recording(24) as inner:
            checkpoint(2,lambda:{'value':float(torch.rand(()))})
        checkpoint(3,lambda:{'value':float(torch.rand(()))})
    assert torch.equal(torch.rand(3),expected)
    assert [x['step'] for x in outer.curve]==[1,3]
    assert [x['step'] for x in inner.curve]==[2]
    checkpoint(4,lambda:1/0)


def test_incomplete_curve_cannot_certify_convergence():
    curve=[{"step":50*i,"hq":1.} for i in range(1,25)]
    expected=[p['step'] for p in curve]
    for partial in (curve[:5],curve[:-1],curve[:8]+curve[9:]):
        scored=sustained(partial,[("hq",">=",.9)],expected_steps=expected)
        assert not scored['complete']
        assert scored['stable_from_step'] is None
    assert sustained(curve,[("hq",">=",.9)],expected_steps=expected)['confirmed_step']==250
