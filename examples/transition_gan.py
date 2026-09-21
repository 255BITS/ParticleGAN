"""Run the winning shared-state encoder transition example. See docs/transition-gan.md for the setup.

    python -u examples/transition_gan.py
    python -u examples/transition_gan.py --device cpu --steps 20 --out-dir results/transition/smoke
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_transition import main


if __name__ == "__main__":
    main()
