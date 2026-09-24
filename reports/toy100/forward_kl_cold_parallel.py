"""Cold cumulative companion to frozen warm-first forward-KL screen.

Runs the exact same `run_case` as the warm-first source. This concurrent
diagnostic does not grant gate credit until that source's warm arm passes.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.forward_kl_cumulative_filter import (
    ROOT, SOURCE_NAMES, run_case, sha,
)
from reports.toy100.sample_anchor_free1200 import load_states
from reports.toy100.sample_anchor_local_mmd_filter import local_width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--first-bank', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    first_raw = args.first_bank.read_bytes()
    first = json.loads(first_raw)
    cold, _, input_hashes = load_states()
    means = mode_hold.ring_means()
    first_stream = torch.Generator().set_state(cold['rng']['data'])
    bank = mode_hold.sample_ring(means, 128, mode_hold.SIGMA, first_stream)
    width, _ = local_width(bank)
    if width != first['declaration']['frozen_width'] or not first['all_cheap_gates_pass']:
        raise RuntimeError('first-bank source/gate/width mismatch')
    names = ('reports/toy100/forward_kl_cold_parallel.py',) + SOURCE_NAMES
    paths = {n: (ROOT/n if (ROOT/n).exists() else args.code_root/n) for n in names}
    hashes = {n: sha(p.read_bytes()) for n,p in paths.items()}
    if hashes['reports/toy100/forward_kl_free_filter.py'] != first['declaration']['source'][
            'reports/toy100/forward_kl_free_filter.py']:
        raise RuntimeError('one-bank optimizer source differs')
    args.output.mkdir(parents=True)
    for name,path in paths.items():
        target = args.output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    declaration = dict(scope='concurrent pure-output cold16 diagnostic', source=hashes,
        input_hashes=input_hashes, first_bank_sha256=sha(first_raw), frozen_width=width,
        exact_run_case_source='reports/toy100/forward_kl_cumulative_filter.py',
        warm_gate_credit='none until frozen warm-first run passes',
        no_neural_update=True, no_seed_or_bandwidth_sweep=True)
    (args.output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    print(json.dumps(dict(event='FORWARD_KL_COLD_DECLARED',source=hashes[
        'reports/toy100/forward_kl_cold_parallel.py'])),flush=True)
    before_rng = torch.random.get_rng_state().clone()
    row = run_case(cold,'cold1',width,means,first,args.output)
    result = dict(status='COMPLETE',declaration=declaration,case=row,
        global_torch_rng_unchanged=torch.equal(torch.random.get_rng_state(),before_rng))
    if not result['global_torch_rng_unchanged']:
        raise RuntimeError('pure-output cold diagnostic changed global RNG')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='FORWARD_KL_COLD_DONE',attempted=row['attempted'],
        cold_terminal5=row['cold_terminal5'])),flush=True)


if __name__ == '__main__':
    main()
