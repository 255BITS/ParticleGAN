"""CPU-only focused lineage tests; preserves prior failed evidence."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True
import argparse
from pathlib import Path
import torch
import lineage_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parent/'pkg-LINEAGE')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return lineage_checks.run(args.package_root, args.output, 'cpu')


if __name__ == '__main__': raise SystemExit(main())
