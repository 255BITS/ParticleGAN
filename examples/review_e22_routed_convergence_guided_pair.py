"""Independently review the fixed guided-pair native campaign.

PYTHONPATH=. python examples/review_e22_routed_convergence_guided_pair.py --source-only
PYTHONPATH=. python examples/review_e22_routed_convergence_guided_pair.py \
    --run runs/routed-convergence-guided-pair-v1 --out runs/guided-pair-review.json
"""
if __package__:
    from .e22_routed_convergence_guided_campaign import review_main
else:
    from e22_routed_convergence_guided_campaign import review_main

if __name__ == "__main__":
    review_main()
