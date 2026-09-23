"""Fingerprint one numerical path: run the rare-mode winner to its first live check (step 50).

The CPU path is chosen only by environment variables or an emulated CPU, e.g.
  MKL_CBWR=AVX2 python3 -m reports.transfer_suite.host_replication.portability.probe LABEL
  qemu-x86_64 -cpu Haswell-v4 /usr/bin/python3 -m reports.transfer_suite.host_replication.portability.probe LABEL
Prints one JSON line. Training code, spec, seed and thresholds are unchanged.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor

REFERENCE = Path(__file__).resolve().parents[2] / 'rare_focus/cli_replay/reference.json.gz'
ENVIRONMENT = ('ATEN_CPU_CAPABILITY', 'MKL_CBWR', 'MKL_ENABLE_INSTRUCTIONS', 'ONEDNN_MAX_CPU_ISA')


class Stop(Exception):
    pass


def main():
    reference = json.loads(gzip.decompress(REFERENCE.read_bytes()))
    card = next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5')
    seen, score = [], vector_tasks.score_samples

    def first_only(fake, cfg, completed):
        seen.append(score(fake, cfg, completed))
        if len(seen) == 2:
            raise Stop
        return seen[-1]
    with patch.object(vector_tasks, 'SimpleMLPDiscriminator', constructor(card)), \
            patch.object(vector_tasks, 'score_samples', first_only):
        vector_tasks.run_episode(reference['spec'], vector_tasks.fixed_policy('cosine'), fixed=True)
    live, target = seen[0], reference['result']['observations'][0]
    keys = sorted(k for k, v in target.items() if isinstance(v, float) and k != 'seconds')
    print(json.dumps(dict(label=sys.argv[1], capability=torch.backends.cpu.get_cpu_capability(),
                          environment={k: os.environ.get(k) for k in ENVIRONMENT},
                          fingerprint=hashlib.sha256(json.dumps({k: live[k] for k in keys}).encode()).hexdigest()[:16],
                          archive_fingerprint=hashlib.sha256(json.dumps({k: target[k] for k in keys}).encode()).hexdigest()[:16],
                          equals_archive=all(live[k] == target[k] for k in keys),
                          max_abs_vs_archive=max(abs(live[k] - target[k]) for k in keys))), flush=True)


if __name__ == '__main__':
    main()
