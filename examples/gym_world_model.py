"""Train the three-generator Lunar Lander example on a collected finite dataset.

    python -u examples/gym_world_model.py
    tail -F results/gym/lunar_lander/live.log

Use configs/gym/lunar_lander/direct.yaml or reconstruction.yaml for comparisons.
Needs a pre-collected dataset (--data-dir, default results/gym/lunar_lander/data;
see experiments/collect_gym_transition.py and docs/gym-world-model-plan.md).
The problem (GymWorldModel) trains on the shared benchmarks.toy_runner.ToyRun.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_gym_transition import main

if __name__ == "__main__":
    main()
