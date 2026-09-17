"""Audit completed scout provenance and common evaluation panels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(name):
    report = Path('reports/memory-handoff')/name
    queue = Path('runs/memory_path')/name
    names = json.loads((report/'scouts.json').read_text())
    assert (queue/'SEALED').exists()
    for state in ('failed', 'running', 'pending'):
        assert not list((queue/state).glob('*.json')), state
    jobs = [json.loads(p.read_text()) for p in (queue/'done').glob('*.json')]
    assert sorted(j['name'] for j in jobs) == sorted(names)
    ref = np.load('runs/memory_path/principles_round12/runs/match_shuffle25/trajectories.npz')
    keys = ['real_noisy', 'real_clean', 'continuation_reference', 'observed_prefix8', 'observed_prefix32']
    sources = []
    passes = {}
    for name in names:
        path = queue/'runs'/name
        with np.load(path/'trajectories.npz') as data:
            for key in keys:
                assert np.array_equal(data[key], ref[key]), (name, key)
        provenance = json.loads((path/'provenance.json').read_text())['sources']
        for key, value in provenance.items():
            assert hashlib.sha256((path/key).read_bytes()).hexdigest() == value
        sources.append(provenance)
        summary = json.loads((path/'summary.json').read_text())
        cfg = summary['config']
        assert cfg['max_sequential_generated_writes'] == 1
        assert cfg['gradient_clipping'] is None and cfg['ema'] is False and cfg['adversarial_only']
        passes[name] = {'cold': {k: v['circle_like_fraction'] for k, v in summary['metrics'].items() if k.startswith('generated')},
                        'warm': {k: {f: m['reference_orbit_fraction'] for f, m in v.items() if f.startswith('fidelity')}
                                 for k, v in summary['metrics'].items() if k.startswith('prefix')}}
    assert all(s == sources[0] for s in sources)
    result = dict(completed=len(names), reference_fields_bitwise_equal=keys,
                  archived_sources_match_hashes=True, identical_sources_within_round=True,
                  one_generated_write_per_branch=True, passes=passes)
    (report/'panel_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round')
    audit(parser.parse_args().round)
