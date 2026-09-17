#!/usr/bin/env python
"""Generate the authorized Stage 0 gate. Later stages require owner go/no-go."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from experiments.train_100gaussians import DEFAULTS


def main():
    folder = Path('configs/mog/stage0')
    folder.mkdir(parents=True, exist_ok=True)
    for arm, prior in [('C0', 'mog'), ('C1', 'fresh_gaussian')]:
        for seed in (1, 2, 3):
            name = f'{arm}_s{seed}'
            cfg = {**DEFAULTS, 'prior_kind': prior, 'seed': seed, 'sigma_rel': 0.,
                   'standardize': False, 'mog_metrics': True, 'log_interval': 100,
                   'final_samples': 200000, 'out_dir': f'results/mog/stage0/{name}'}
            (folder/f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(name)


if __name__ == '__main__':
    main()
