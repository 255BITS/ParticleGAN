"""CPU slow→fast pairing gate. One JSON object per line, so logs tail cleanly.

Locked shared-identity run (the arm that should pass):

    python -u experiments/train_shared_trajectory.py

Drift arms (same recipe, stranger pairing, should fail the identity gate):

    python -u experiments/train_shared_trajectory.py --drift --pairing stranger
    python -u experiments/train_shared_trajectory.py --drift --pairing nearest_stranger

A pass is a CPU toy result. It is not a Music or Anima GPU transfer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.shared_trajectory import PAIRINGS, train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", default="shared", choices=PAIRINGS)
    parser.add_argument(
        "--drift", action="store_true",
        help="run stranger or nearest-stranger; locked_shared refuses those pairings",
    )
    args = parser.parse_args()
    if args.drift:
        result = train(pairing=args.pairing, locked=False, echo=True)
        return 0
    if args.pairing != "shared":
        parser.error(
            "locked_shared refuses stranger pairing; pass --drift to run the failing arm"
        )
    result = train(pairing="shared", locked=True, echo=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
