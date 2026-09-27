"""Offline scores; no information is returned to any learner."""
import hashlib
import json
from pathlib import Path
import sys


def good(p):
    return p['modes'] == 8 and p['hq'] >= .9


def segment(points, start, end):
    xs = [p for p in points if start < p['step'] <= end]
    first = next((p['step'] for p in xs if good(p)), None)
    after = [p for p in xs if first is not None and p['step'] >= first]
    suffix = []
    for p in reversed(xs):
        if not good(p):
            break
        suffix.append(p)
    failures = [p['step'] for p in after if not good(p)]
    departures = [p['step'] for i, p in enumerate(after) if i and good(after[i-1]) and not good(p)]
    return dict(start=start, end=end, first_arrival=first,
                delay=None if first is None else first-start, passing_since_arrival=sum(map(good, after)),
                checks_since_arrival=len(after), every_failing_observation_since_arrival=failures,
                departures=departures, stable_suffix_start=suffix[-1]['step'] if suffix else None,
                stable_suffix_checks=len(suffix),
                minimum_hq_since_arrival=min((p['hq'] for p in after), default=None),
                minimum_modes_since_arrival=min((p['modes'] for p in after), default=None),
                minimum_hq_entire_segment=min((p['hq'] for p in xs), default=None),
                final=xs[-1] if xs else None)


def summarize(path):
    rows=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
    points=[{k:r[k] for k in ('step','modes','hq')} for r in rows]
    rates=[json.loads(l) for l in (path/'learning-rates.jsonl').read_text().splitlines()]
    protocol=json.loads((path/'declaration.json').read_text()).get('evaluation_protocol',{})
    changes=protocol.get('target_changes',[{'after_update':2400}])
    boundaries=[0]+[c['after_update'] for c in changes]+[points[-1]['step']]
    boundaries=sorted(set(v for v in boundaries if v <= points[-1]['step']))
    initial=json.loads((path/'initial.json').read_text())
    baseline=json.loads((Path(__file__).parents[1]/'ka2-default-candidate/constant-lr-api/evidence/constant/initial.json').read_text())
    summary=dict(segments=[segment(points,a,b) for a,b in zip(boundaries,boundaries[1:])],
        events=[dict(step=r['step'], **r['precision']) for r in rates if r['precision']['event']],
        initial_open=json.loads((path/'declaration.json').read_text())['schedule'],
        initial_baseline_parity={k:initial[k]==baseline[k] for k in ('models_sha256','streams_sha256','cpu_rng_sha256','cuda_rng_sha256','real_stream_sha256','means_sha256')},
        final_policy=rates[-1]['precision'])
    (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    hashes={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'}
    (path/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    summarize(Path(sys.argv[1]))
