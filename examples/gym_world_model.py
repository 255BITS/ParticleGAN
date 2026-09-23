"""Train the three-generator Lunar Lander example on a collected finite dataset.

    python -u examples/gym_world_model.py
    tail -F results/gym/lunar_lander/live.log

Use configs/gym/lunar_lander/direct.yaml or reconstruction.yaml for comparisons.
See docs/gym-world-model-plan.md for collection and evaluation requirements.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gym_transition import main

if __name__ == "__main__":
    main()
