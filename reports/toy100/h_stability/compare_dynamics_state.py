"""Compare complete replay state, including optimizer tensors and every RNG."""
import argparse
import json
from pathlib import Path
import time
import torch


def compare(original, replay):
    counts = dict(tensor_leaves=0, generator_rng_objects=0)

    def equal(a, b, path='state'):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), path
            counts['tensor_leaves'] += 1
        elif isinstance(a, torch.Generator):
            assert torch.equal(a.get_state(), b.get_state()), path
            counts['generator_rng_objects'] += 1
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for key in a:
                equal(a[key], b[key], path+'.'+str(key))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                equal(x, y, path+'.'+str(i))
        else:
            assert a == b, path

    equal(torch.load(original/'final-state.pt', weights_only=False, map_location='cpu'),
          torch.load(replay/'final-state.pt', weights_only=False, map_location='cpu'))
    for name in ('metrics.json', 'losses.json'):
        assert json.loads((original/name).read_text()) == json.loads((replay/name).read_text()), name
    return dict(bitwise_models_optimizers_samples_rng_and_update_state=True,
                identical_all_metrics_and_losses=True, **counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        metrics, status = compare(args.original, args.replay), 'PASS'
    except Exception as error:
        metrics, status = dict(error=repr(error)), 'ERROR'
    result = dict(candidate='regression', gate='archived_warm_full_state_replay_fixed',
                  status=status, seconds=time.perf_counter()-started, metrics=metrics,
                  artifact=str((args.replay/'full-state-comparison.json').resolve()))
    Path(result['artifact']).write_text(json.dumps(result, indent=2)+'\n')
    with args.ledger.open('a') as ledger:
        ledger.write(json.dumps(result)+'\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if status == 'PASS' else 1)
