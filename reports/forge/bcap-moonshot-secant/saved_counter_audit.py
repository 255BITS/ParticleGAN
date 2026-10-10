"""Read certified secant checkpoint counters; no training, forwards or sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from experiments.forge.state import state_digest


def walk(value, path='state'):
    if isinstance(value, dict):
        if isinstance(value.get('secant'), dict):
            yield path, value['secant'], value.get('state', {}), value.get('param_groups', [])
        for key, child in value.items():
            if key not in {'models', 'role_parameters', 'streams', 'initialization'}:
                yield from walk(child, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk(child, f'{path}[{index}]')


def summary(counters):
    result = dict(counters)
    n = counters['observations']
    result['mean_applied_fraction'] = counters['scale_sum'] / n if n else None
    result['floor_hit_fraction'] = counters['floor_hits'] / n if n else None
    result['damped_fraction'] = counters['damped'] / n if n else None
    length = counters['proposal_length_sum']
    result['sum_block_length_ratio'] = counters['applied_length_sum'] / length if length else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    results = json.loads((options.publication / 'phase3-results.json').read_text())
    rows = []
    for row in results['task_results']:
        proof = row.get('provenance_checkpoint')
        item = dict(role=row['role'], task_id=row['task_id'], gate_status=row['gate_status'],
                    attempt_id=row['attempt_id'], source_commit=row['source_commit'],
                    source_digest=row['source_digest'], checkpoint_available=bool(proof),
                    counters_available=False)
        if proof:
            path = Path(proof['artifact_root']) / proof['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256'], path
            assert path.stat().st_size == proof['bytes'], path
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state_digest(state) == proof['state_sha256'], path
            item.update(checkpoint_sha256=proof['sha256'], state_sha256=proof['state_sha256'],
                        completed_steps=proof['completed_steps'], optimizers=[])
            for path_name, meta, histories, groups in walk(state):
                assert meta['mode'] == 'bounded' and meta['floor'] == 1/16
                for name, value in meta['stats'].items():
                    if name in {'steps', 'minimum_scale'}:
                        continue
                    assert math.isclose(value, sum(role[name] for role in meta['roles'].values()),
                                        rel_tol=1e-7, abs_tol=1e-7), (path_name, name)
                clocks = []
                bindings = {identifier: group for group in groups for identifier in group["params"]}
                for identifier, history in histories.items():
                    if 'secant_clock' not in history:
                        continue
                    clock, valid = history['secant_clock'], history['secant_valid']
                    assert clock.dtype == torch.long and valid.dtype == torch.bool
                    assert torch.equal(valid, clock > 0)
                    group = bindings[identifier]
                    rate = group['lr']
                    fractions = history['secant_eta'][valid] / rate if rate > 0 else None
                    if fractions is not None and fractions.numel():
                        assert bool(((fractions >= 1/16 - 1e-6) & (fractions <= 1 + 1e-6)).all())
                    clocks.append(dict(parameter_id=identifier, role=group['role'], nominal_rate=rate,
                        parameter_shape=list(history['secant_x'].shape),
                        last_effective_fraction_min=float(fractions.min()) if fractions is not None and fractions.numel() else None,
                        last_effective_fraction_max=float(fractions.max()) if fractions is not None and fractions.numel() else None,
                        last_effective_fraction_mean=float(fractions.mean()) if fractions is not None and fractions.numel() else None,
                        rowwise=clock.ndim == 1,
                        owned_entries=int(valid.sum()), total_entries=valid.numel(),
                        minimum_visits=int(clock.min()) if clock.numel() else None,
                        maximum_visits=int(clock.max()) if clock.numel() else None))
                item['optimizers'].append(dict(path=path_name, counters=summary(meta['stats']),
                    by_role={role: summary(counters) for role, counters in meta['roles'].items()}, clocks=clocks))
            item['counters_available'] = bool(item['optimizers'])
        rows.append(item)
    payload = dict(schema_version=1, qualification_input=False,
        source_commit=results['source_commit'], source_digest=results['source_digest'],
        optimizer_updates_added=0, sampling_draws_added=0,
        interpretation='Fractions are averaged over observed parameter blocks or owned prior rows. Length ratios use sums of Euclidean block lengths, not a joint parameter norm. Absent certified checkpoints remain unknown. Visit clocks count actual owned observations; curvature is confounded by stochastic and opposing-player changes.',
        rows=rows)
    options.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
