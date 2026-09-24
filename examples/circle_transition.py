#!/usr/bin/env python
"""CPU circle gate: paired-error controller versus zero and reversed motion."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_circle_transition import main


if __name__ == "__main__":
    main()
