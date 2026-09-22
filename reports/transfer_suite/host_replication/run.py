"""Replay every archived episode behind the b_cap3 19/19 claim on the current host.

Jobs are fixed before training: the nine required episodes and every passing
architecture trial in formulations/leaderboard.json. Each replay reuses the
archived spec, policy and discriminator constructor; nothing is retuned. Seed 0
only. Bitwise curve parity and the recomputed sustained verdict are recorded.

python -u -m reports.transfer_suite.host_replication.run --output DIR > DIR.log 2>&1; tail -f DIR.log
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import platform
from unittest.mock import patch

import torch

from benchmarks.transfer_suite import image_tasks, suite, vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict

SUITE = Path(__file__).resolve().parents[1]
TIMING = {'seconds', 'controller_seconds', 'stable_from_seconds', 'confirmed_seconds', 'created_at'}
ENVIRONMENT = ('ATEN_CPU_CAPABILITY', 'MKL_CBWR', 'MKL_ENABLE_INSTRUCTIONS', 'ONEDNN_MAX_CPU_ISA', 'OMP_NUM_THREADS')


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def untimed(value):
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def plan():
    row = read_json(SUITE / 'formulations/leaderboard.json')['rows'][0]
    assert row['name'] == 'rp_logistic_bcap3' and row['required_passes'] == 9 and row['practical_passes'] == 10
    jobs = [dict(case=name, label='original required host', artifact=item['artifact'], kind='required')
            for name, item in row['required'].items()]
    for name, cell in row['cases'].items():
        for trial in cell['trials']:
            if trial['verdict']['passed']:
                jobs.append(dict(case=name, label=trial['label'], artifact=trial['artifact'],
                                 kind='image' if cell['runner'] == 'image' else 'vector'))
    return jobs


def read_json(path):
    return json.loads(path.read_text())


def archived(job):
    payload = read(SUITE / job['artifact'])
    if job['kind'] == 'image':
        return payload['effective_spec'], payload['policy'], payload
    if 'task' in payload and 'spec' not in payload:
        return payload['task'], payload['policy'], payload['result']
    return payload['spec'], payload['policy'], payload['result']


def discriminator(spec):
    card = spec.get('research_discriminator')
    if card is None:
        return None
    if card.get('skip') == 'raw_linear':
        from benchmarks.transfer_suite.linear_skip_refinement_research import constructor
    else:
        from benchmarks.transfer_suite.smooth_critic_research import constructor
    return constructor(card)


def replay(job):
    torch.set_num_threads(1)
    spec, policy, reference = archived(job)
    spec, policy = deepcopy(spec), deepcopy(policy)
    if job['kind'] == 'required':
        result = suite.run_episode(spec, policy, fixed=True)
    elif job['kind'] == 'image':
        result = image_tasks.run_episode(spec, policy, fixed=True)
    else:
        create = discriminator(spec)
        if create is None:
            result = vector_tasks.run_episode(spec, policy, fixed=True)
        else:
            with patch.object(vector_tasks, 'SimpleMLPDiscriminator', create):
                result = vector_tasks.run_episode(spec, policy, fixed=True)
    old, new = test_verdict(spec, reference), test_verdict(spec, result)
    ro, no = reference.get('observations', []), result.get('observations', [])
    keys = sorted(k for k, v in (ro[0] if ro else {}).items() if isinstance(v, float) and k not in TIMING)
    first = next((i for i, (a, b) in enumerate(zip(ro, no)) if any(a.get(k) != b.get(k) for k in keys)), None)
    comparison = dict(
        result_equal_except_timing=untimed(result) == untimed(reference),
        live_curves_equal=first is None and len(ro) == len(no),
        first_divergent_step=None if first is None else ro[first]['step'],
        first_divergence_abs=None if first is None else max(abs(ro[first][k] - no[first][k]) for k in keys
                                                               if isinstance(no[first].get(k), float)),
        archived=dict(status=old['status'], passing_suffix=old.get('convergence', {}).get('passing_suffix'),
                      confirmed_step=old.get('convergence', {}).get('confirmed_step')),
        replay=dict(status=new['status'], passing_suffix=new.get('convergence', {}).get('passing_suffix'),
                    confirmed_step=new.get('convergence', {}).get('confirmed_step'),
                    final_failing=[m['metric'] for m in new.get('metrics', []) if m['status'] != 'PASS']))
    return job, spec, policy, result, new, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'episodes').mkdir()
    jobs = plan()
    protocol = suite.snapshot(args.output)
    protocol.update(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), platform=platform.platform(),
                    processor=platform.processor(), environment={k: os.environ.get(k) for k in ENVIRONMENT},
                    processes=args.processes)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (args.output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, rule=__doc__.split('\n\n')[1]), indent=2) + '\n')
    print(f'PLAN {len(jobs)} episodes', flush=True)
    records = []
    with Pool(args.processes) as pool:
        for job, spec, policy, result, verdict, comparison in pool.imap(replay, jobs):
            suite.verify_source(protocol)
            value = dict(job=job, source_artifact_sha256=hashlib.sha256((SUITE / job['artifact']).read_bytes()).hexdigest(),
                         spec=spec, policy=policy, result=result, verdict=verdict, comparison=comparison)
            raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
            name = f"episodes/{job['label'].replace(' ', '_')}__{job['case']}.json.gz"
            (args.output / name).write_bytes(gzip.compress(raw, mtime=0))
            records.append(dict(job=job, artifact=name, uncompressed_sha256=hashlib.sha256(raw).hexdigest(),
                                verdict=verdict, comparison=comparison, error=result.get('error')))
            (args.output / 'index.json').write_text(json.dumps(dict(records=records), indent=2) + '\n')
            print(f"DONE {job['case']:22} {job['label'][:24]:24} archived={comparison['archived']['status']:4} "
                  f"replay={comparison['replay']['status']:4} suffix={comparison['replay']['passing_suffix']} "
                  f"bitwise={comparison['live_curves_equal']} diverge@{comparison['first_divergent_step']} "
                  f"fails={','.join(comparison['replay']['final_failing']) or '-'}", flush=True)
    suite.verify_source(protocol)
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
