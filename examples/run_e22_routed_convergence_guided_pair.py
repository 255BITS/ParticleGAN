"""Run the one fixed guided-pair native campaign; the card must authorize it.

PYTHONPATH=. python -u examples/run_e22_routed_convergence_guided_pair.py \
    --out runs/routed-convergence-guided-pair-v1
"""
if __package__:
    from .e22_routed_convergence_guided_campaign import run_main
else:
    from e22_routed_convergence_guided_campaign import run_main

if __name__ == "__main__":
    run_main()
