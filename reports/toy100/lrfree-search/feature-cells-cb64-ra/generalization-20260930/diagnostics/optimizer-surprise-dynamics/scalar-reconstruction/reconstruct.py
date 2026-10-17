"""Exact detector classes, closed scalar observations; no models or PT files."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classes(path, names, torch):
    source = ast.parse(path.read_text())
    nodes = [node for node in source.body
             if isinstance(node, ast.ClassDef) and node.name in names]
    assert [node.name for node in nodes] == list(names)
    namespace = {'torch': torch, 'math': math, 'deepcopy': deepcopy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return [namespace[name] for name in names]


def seeded(cls, first):
    detector = cls()
    before, after = first['before_decide'], first['after_decide']
    for key in ('fast', 'slow', 'armed', 'streak', 'since_calm', 'since_fire',
                'last_ratio', 'last_ratios'):
        setattr(detector, key, deepcopy(before[key]))
    assert not after['fire']
    for key in ('fires', 'log', 'anchor_event', 'anchor_events'):
        setattr(detector, key, deepcopy(after[key]))
    return detector


def network(context, guard_cls):
    roles = context['roles']
    result = {}
    for key, tester in context['testers'].items():
        i, j = map(int, key.split('.'))
        role, scale = roles[i][j], tester['s']
        if role in guard_cls.NETWORK_ROLES and 0 < scale < 1:
            result[key] = {'role': role, 'scale': scale}
    return result


def epoch(context):
    record = context['ka2']
    return record['anchor_started'] if record['formulation'] == 'ka2' else None


def main():
    prep = json.loads((HERE / 'INPUTS-FROZEN.json').read_text())
    for path, digest in prep['sha256'].items():
        assert sha(Path(path)) == digest, path
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    rng_before = torch.get_rng_state().clone()
    old_path = ROOT / 'pkg-RA12-auto/particlegan/continuous.py'
    new_path = ROOT / 'pkg-RA13-settled/particlegan/continuous.py'
    old_cls, = classes(old_path, ('OptimizerSurprise',), torch)
    new_cls, guard_cls = classes(new_path, ('OptimizerSurprise', 'SettledReopenGuard'), torch)
    cases = {}
    for name, source in prep['traces'].items():
        rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
        assert rows and all(b['update'] == a['update'] + 1 for a, b in zip(rows, rows[1:]))
        legacy, default = seeded(old_cls, rows[0]), seeded(new_cls, rows[0])
        legacy_fires = []
        for row in rows:
            pending = {key: torch.tensor(value, dtype=torch.float32)
                       for key, value in row['before_decide']['pending_q'].items()}
            step = row['before_decide']['step_arg']
            legacy.pending, default.pending = deepcopy(pending), deepcopy(pending)
            a, b = legacy.decide(step), default.decide(step)
            assert a == b == row['after_decide']['fire'], (name, step)
            for key in ('fast', 'slow', 'armed', 'streak', 'since_calm', 'since_fire',
                        'fires', 'log', 'last_ratio', 'last_ratios'):
                assert getattr(legacy, key) == row['after_decide'][key], (name, step, key)
                assert getattr(default, key) == getattr(legacy, key), (name, step, key, 'default')
            if a:
                legacy_fires.append(step)
        assert len(legacy_fires) == 1
        guarded, guard = seeded(new_cls, rows[0]), guard_cls()
        initial = rows[0]['before_decide']
        guard.observe_epoch(epoch(initial['context']), guarded)
        # A scalar-prefix reconstruction, not a valid new-law checkpoint.
        # The left boundary uses the visible network witness; the toy epoch
        # later clears it. Both moving windows begin with a calm observation.
        if initial['last_ratio'] is not None:
            guard.observe(initial['last_ratio'], guarded.CALM, initial['step_arg'],
                          network(initial['context'], guard_cls))
        prefix = []
        for row in rows:
            before = row['before_decide']
            step = before['step_arg']
            guarded.pending = {key: torch.tensor(value, dtype=torch.float32)
                               for key, value in before['pending_q'].items()}
            witness = network(before['context'], guard_cls)
            fire = guarded.decide(step, guard=guard, network=witness)
            prefix.append(dict(step=step, update=row['update'], fire=fire,
                               ratio=guarded.last_ratio, streak=guarded.streak,
                               network=witness, guard=guard.state_dict()))
            if fire or row['after_decide']['fire']:
                break
            guard.observe_epoch(epoch(row['after_update']), guarded)
        cases[name] = dict(
            source=source, source_sha256=sha(Path(source)), total_rows=len(rows),
            exact_legacy_default_rows=len(rows), legacy_fires=legacy_fires,
            guarded_fires=[r['step'] for r in prefix if r['fire']],
            tested_prefix=prefix,
            stop='first original or proposed action; no post-divergence quality inference')
    assert torch.equal(rng_before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    for path, digest in prep['sha256'].items():
        assert sha(Path(path)) == digest, path
    result = dict(status='COMPLETE_DETECTOR_ONLY_RECONSTRUCTION',
                  candidate_source_sha256=sha(new_path), cases=cases,
                  global_cpu_rng_unchanged=True, cuda_initialized=False,
                  models=0, pt_reads=0, forwards=0, draws=0, updates=0, scorers=0,
                  limitations=[
                      'Observed scalar windows are not fresh guarded training trajectories.',
                      'Legacy scalar history initializes the analysis prefix; this is not resumable guarded state.',
                      'No quality or complete-run retention claim follows from suppressed or retained events.'])
    with (HERE / 'result.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'result_sha256': sha(HERE / 'result.json'),
                      'fires': {name: {'old': c['legacy_fires'], 'guarded': c['guarded_fires']}
                                for name, c in cases.items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
