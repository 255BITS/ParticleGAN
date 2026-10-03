"""The original row-paired loss invalidates a claimed coincidence invariant."""
from benchmarks.toy_audit.check_paired_gradient import counterexample


def test_different_real_pairs_break_coincident_fake_gradient_symmetry():
    result = counterexample()
    assert result["particles"] == 12
    assert result["fake_logits_all_equal"]
    assert result["real_logits_max"] > result["real_logits_min"]
    assert not result["paired_gradients_all_equal"]
    assert result["paired_gradient_range"] > 5e-4
    # Replacing the row-specific real logits by one shared logit restores
    # equal gradients; reversing the real rows reverses the fake gradients.
    assert result["constant_real_gradient_range"] == 0
    assert result["reversed_pair_gradient_max_error"] < 1e-8
    assert result["caller_global_rng_preserved"]
