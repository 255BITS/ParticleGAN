"""Stdlib-only immutable source and raw storage readers copied from the prior independent audit."""
import ast
import collections
import datetime
import gzip
import hashlib
import io
import json
import math
import pickle
import statistics
import struct
import zipfile
from pathlib import Path
sha = lambda data: hashlib.sha256(data).hexdigest()
read_json = lambda p: json.loads(p.read_text())
read_rows = lambda p: [json.loads(line) for line in p.read_text().splitlines()]

def rebuild(storage, offset, size, stride, *unused):
    return dict(storage=storage, offset=offset, size=size, stride=stride)

class StorageOnlyUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return rebuild
        if module == 'torch' and name in ('FloatStorage', 'ByteStorage'):
            return name
        raise ValueError(f'unapproved pickle global: {module}.{name}')

    def persistent_load(self, value):
        assert value[0] == 'storage'
        return dict(dtype=value[1], key=value[2], device=value[3], length=value[4])

def checkpoint_summary(path):
    with zipfile.ZipFile(path) as archive:
        name = next((n for n in archive.namelist() if n.endswith('/data.pkl')))
        prefix = name[:-len('data.pkl')]
        state = StorageOnlyUnpickler(io.BytesIO(archive.read(name))).load()
        state = state.get('trainer', state)

        def tensor_bytes(t):
            width = {'FloatStorage': 4, 'ByteStorage': 1}[t['storage']['dtype']]
            count = math.prod(t['size'])
            expected = 1
            for length, stride in reversed(list(zip(t['size'], t['stride']))):
                assert length <= 1 or stride == expected
                expected *= length
            raw = archive.read(prefix + 'data/' + t['storage']['key'])
            start = t['offset'] * width
            return raw[start:start + count * width]
        steps = []
        devices = []
        for optimizer in state['optimizers']:
            row = [s['step'] for s in optimizer['state'].values()]
            steps.append(sorted(set((struct.unpack('<f', tensor_bytes(t))[0] for t in row))))
            devices.append(sorted(set((t['storage']['device'] for t in row))))
        extra = state['optimizers'][1]['regularizer']
        reference = extra['ema']
        critic = state['models']['D']
        return dict(path=str(path), sha256=sha(path.read_bytes()), completed_steps=state['completed_steps'], adam_step_values=steps, adam_step_devices=devices, serial_backward=state['serial_backward'], critic_record={k: extra['record'][k] for k in ('calls', 'observed_steps', 'ema_updates', 'ema_skips', 'ema_reseeds', 'formulation')}, critic_reference_tensor_count=len(critic), critic_reference_equal=critic.keys() == reference.keys() and all((tensor_bytes(t) == tensor_bytes(reference[k]) for k, t in critic.items())))

def source_summary(run):
    path = RUNS / run / 'source.zip'
    manifest = read_json(RUNS / run / 'declaration.json')['source_sha256']
    with zipfile.ZipFile(path) as archive:
        hashes = {n: sha(archive.read(n)) for n in archive.namelist()}
    return dict(source_zip_sha256=sha(path.read_bytes()), source_count=len(hashes), declaration_hash_mismatches=[n for n, v in hashes.items() if manifest.get(n) != v], missing_declared_files=sorted(set(manifest) - set(hashes)), package_sha256={n: v for n, v in hashes.items() if n.startswith('particlegan/')})

def rate_summary(run):
    rows = read_rows(RUNS / run / 'learning-rates.jsonl')
    fields = [row['game']['field_evaluations'] for row in rows]
    summary = dict(updates=len(rows), rate_ranges={k: [min((x[k] for x in rows)), max((x[k] for x in rows))] for k in ('generator_0', 'critic_0', 'prior_1')}, controller_and_accepted_counts_exact=all((row['step'] == row['controller']['calls'] == row['controller']['observed_steps'] == row['game']['accepted_updates'] for row in rows)), reference_counts_exact=all((row['controller']['ema_updates'] == max(0, row['step'] - 799) and row['controller']['ema_skips'] == row['controller']['ema_reseeds'] == 0 for row in rows)), mean_field_evaluations=statistics.mean(fields), field_evaluations_range=[min(fields), max(fields)], last500_mean_accepted_l2={role: statistics.mean((row['updates'][role]['l2'] for row in rows[-500:])) for role in ('G', 'D', 'prior')})
    if 'trials' in rows[0]['game'] and 'probe_fraction' in rows[0]['game']['trials'][0]:
        eta = [row['game']['trials'][-1]['probe_fraction'] for row in rows]
        summary.update(probe_fraction_range=[min(eta), max(eta)], median_probe_fraction=statistics.median(eta), all_final_locality_ratios_at_most_half=all((row['game']['trials'][-1]['relative_field_change'] <= 0.5 for row in rows)), max_accepted_locality_ratio=max((row['game']['trials'][-1]['relative_field_change'] for row in rows)))
    elif 'trials' in rows[0]['game']:
        eta = [row['game']['trials'][-1]['fraction'] for row in rows]
        residuals = [value for row in rows for value in row['game']['trials'][-1]['relative_implicit_residual'].values()]
        summary.update(fraction_range=[min(eta), max(eta)], median_fraction=statistics.median(eta), all_accepted_role_residuals_at_most_half=max(residuals) <= .5, maximum_accepted_role_residual=max(residuals))
    return summary
