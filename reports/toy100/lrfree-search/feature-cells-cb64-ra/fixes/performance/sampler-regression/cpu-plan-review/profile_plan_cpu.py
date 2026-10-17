"""Count scalar/index operations on fixed saved mass fixtures, CPU only."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import traceback
from types import ModuleType
import torch
from torch.utils._python_dispatch import TorchDispatchMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_PACKAGE = ROOT/'pkg-CB64-RA3'
FIXED_FILES = [ROOT/'stability/mass-gpu-inputs.pt',
               ROOT/'integration/review/training-regression/snapshot-1000.pt',
               ROOT/'integration/review/training-regression/snapshot-2000.pt']
WATCH = {'aten::_local_scalar_dense', 'aten::nonzero'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(name, path):
    holder = ModuleType(name); holder.__path__ = [str(path/'particlegan')]
    sys.modules[name] = holder
    return importlib.import_module(name+'.feature_cells')


def source_map(path):
    return {str(p): sha(p) for p in sorted((path/'particlegan').rglob('*.py'))}


def fixtures():
    original = torch.load(FIXED_FILES[0], map_location='cpu', weights_only=False)
    values = [(name, deepcopy(value)) for name, value in original['scenarios'].items()]
    for name, value in values: value['fixture_seed'] = original['seed']
    for step, path in zip((1000, 2000), FIXED_FILES[1:]):
        values.append((f'learned_toy_{step}', torch.load(path, map_location='cpu', weights_only=False)))
    return values


def snapshot(module, value):
    result = module.FeatureCellSnapshot.__new__(module.FeatureCellSnapshot)
    for name, item in deepcopy(value['snapshot']).items(): setattr(result, name, item)
    return result


def stream(value):
    result = torch.Generator(device='cpu')
    if 'planning_rng' in value: result.set_state(value['planning_rng'].cpu())
    else: result.manual_seed(value['fixture_seed'])
    return result


class SourceOps(TorchDispatchMode):
    def __init__(self):
        self.operations, self.sites = Counter(), Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        self.operations[name] += 1
        if name in WATCH:
            frames = [frame for frame in traceback.extract_stack()
                      if frame.filename.endswith('/particlegan/feature_cells.py')]
            if frames:
                frame = frames[-1]
                self.sites[(name, frame.filename, frame.name, frame.lineno, frame.line)] += 1
        return func(*args, **(kwargs or {}))


def profile_call(call, source_replay):
    source_ops = SourceOps()
    # CPU events only. Counts indicate potential CUDA barriers; these CPU
    # timings are not a measured CUDA speedup or a trajectory diagnosis.
    # Trace source sites on an independent identical snapshot/stream. Nesting
    # TorchDispatchMode inside the profiler duplicates native scalar events.
    with source_ops:
        source_replay()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        result = call()
    averages = {event.key: event.count for event in profiler.key_averages()}
    sites = [dict(operator=key[0], file=key[1], function=key[2], line=key[3], code=key[4], calls=count)
             for key, count in source_ops.sites.most_common()]
    return result, dict(profiler_counts={name: averages.get(name, 0) for name in
        ('aten::item', 'aten::_local_scalar_dense', 'aten::nonzero', 'aten::is_nonzero', 'aten::equal')},
        dispatch_counts={name: source_ops.operations[name] for name in sorted(WATCH)},
        source_sites=sites)


def plan_profiles(module, name, value):
    snap = snapshot(module, value); generator = stream(value)
    topology, cold = profile_call(snap._mass_topology, deepcopy(snap)._mass_topology)
    replay_snap, replay_stream = deepcopy(snap), torch.Generator().set_state(generator.get_state())
    ordinary, ordinary_profile = profile_call(lambda: snap.ordinary_transport(
        value['q'], value['flags'], deepcopy(value['comparison']),
        generator=generator, pvalues=value['pvalues']), lambda: replay_snap.ordinary_transport(
        value['q'], value['flags'], deepcopy(value['comparison']),
        generator=replay_stream, pvalues=value['pvalues']))
    child, parent, detail = ordinary
    replay_snap, replay_stream = deepcopy(snap), torch.Generator().set_state(generator.get_state())
    isolated, isolation_profile = profile_call(lambda: snap.select_parents(
        value['q'], value['flags'], ordinary_children=child, ordinary_parents=parent,
        generator=generator, pvalues=value['pvalues']), lambda: replay_snap.select_parents(
        value['q'], value['flags'], ordinary_children=child, ordinary_parents=parent,
        generator=replay_stream, pvalues=value['pvalues']))
    iso_child, iso_parent, isolation = isolated
    return dict(name=name, population=len(value['q']), cells=snap.cells, groups=snap.mass_groups,
                flags=int(value['flags'].sum()), ordinary_moves=len(child), isolation_moves=len(iso_child),
                topology_cold=cold, ordinary_warm_topology=ordinary_profile,
                isolation_warm_topology=isolation_profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError('Output evidence already exists')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    before = source_map(args.package_root)
    before.update({str(path): sha(path) for path in FIXED_FILES})
    module = package('cpu_plan_profile', args.package_root)
    rows = []
    for name, value in fixtures():
        result = plan_profiles(module, name, value)
        rows.append(result)
        print(json.dumps(dict(name=name, groups=result['groups'], ordinary_moves=result['ordinary_moves'],
                             isolation_moves=result['isolation_moves'], counts={stage: result[stage]['profiler_counts']
                             for stage in ('topology_cold', 'ordinary_warm_topology', 'isolation_warm_topology')})), flush=True)
    assert all(sha(path) == expected for path, expected in before.items())
    assert not torch.cuda.is_initialized()
    receipt = dict(status='PASS', scope='CPU fixed-fixture operation counts and source hotspots; no CUDA timing attribution',
                   package_root=str(args.package_root.resolve()), source_sha256=before,
                   script_sha256=sha(Path(__file__)), cpu_threads=1, cuda_initialized=False,
                   new_seeds=0, optimizer_updates=0, cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__': main()
