import torch

from lib.misgan import (BLOCK_PATTERNS, DIM, Problem, bayes_posterior, imputation_metrics, mask_metrics,
                        mean_impute, pattern_probs, sample_masks)


def test_mask_mechanisms_shapes_and_probabilities():
    gen = torch.Generator().manual_seed(0)
    for mech in ("mcar_p20", "mcar_p50", "mcar_p80", "block"):
        probs = pattern_probs(mech)
        assert probs.shape == (2 ** DIM,) and abs(float(probs.sum()) - 1) < 1e-6
        masks = sample_masks(mech, 5000, gen)
        assert masks.shape == (5000, DIM) and set(masks.unique().tolist()) <= {0.0, 1.0}
    rate = 1 - sample_masks("mcar_p80", 20000, gen).mean()
    assert abs(float(rate) - 0.8) < 0.02
    # Block patterns jointly observe every coordinate pair (identifiable joint).
    pairs = {(i, j) for obs, _ in BLOCK_PATTERNS for i in obs for j in obs if i < j}
    assert len(pairs) == DIM * (DIM - 1) // 2


def test_bayes_imputer_beats_mean_and_exact_mask_scores_zero():
    problem = Problem("mcar_p80", n_train=2000, n_test=1500, seed=3)
    gen = torch.Generator().manual_seed(1)
    post, draws = bayes_posterior(problem, problem.x_test, problem.m_test, draws=4, generator=gen)
    assert post.shape == (1500, 100) and torch.allclose(post.sum(1), torch.ones(1500), atol=1e-4)
    assert draws.shape == (4, 1500, DIM)
    observed = problem.m_test.bool()
    assert torch.equal(draws[0][observed], problem.x_test[observed])
    bayes = imputation_metrics(problem, draws, post)
    mean = imputation_metrics(problem, mean_impute(problem, problem.x_test, problem.m_test), post)
    assert bayes["acc"] > mean["acc"] + 0.4 and bayes["rmse"] < mean["rmse"]
    assert bayes["itv"] < mean["itv"] and bayes["istd"] > 0 == mean["istd"]
    exact = mask_metrics(problem, problem.m_test)
    assert exact["m_mae"] < 0.02 and exact["m_soft"] == 0
