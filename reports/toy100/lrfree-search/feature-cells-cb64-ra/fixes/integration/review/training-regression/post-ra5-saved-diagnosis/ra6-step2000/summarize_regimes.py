"""Summarize archived saved regimes and measured timings without numerical execution."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[4]
output = here / 'regime-summary.json'
assert not output.exists()
inputs = [here / 'receipt.json', here / 'candidate-metrics-captured.jsonl',
    root / 'pkg-CB64-RA6/particlegan/feature_cells.py',
    root / 'pkg-CB64-RA6/particlegan/birth_death.py', root / 'pkg-CB64-RA6/particlegan/training.py']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p): sha(p) for p in inputs}
diagnosis = json.loads(inputs[0].read_text())
diag_by_step = {r['step']: r for r in diagnosis['records']}
metrics = [json.loads(line) for line in inputs[1].read_text().splitlines()]
records = []
previous = None
for m in metrics:
    d = m['diagnostics']; bd = d['birth_death']; last = bd['last']; counters = bd['counters']
    prior = d['lr_settle']['g1']; controller = d['controller']; current = diag_by_step[m['step']]
    entry = dict(step=m['step'], training_seconds=m['training_seconds'],
        saved_emitted_metrics={k: m['metrics'][k] for k in ('precision', 'coverage', 'mass_tv')},
        clean_fast={k: current['fast'][k] for k in ('precision', 'coverage', 'mass_tv')},
        clean_ema={k: current['ema'][k] for k in ('precision', 'coverage', 'mass_tv')},
        serving=current['reported_serving_table'],
        controller_closed=controller['closed'], controller_updates=controller['updates'],
        prior=dict(s=prior['s'], b=prior['b'], last_decisive=current['population']['last_decisive'],
            last_test=prior['last'], last_population=prior.get('last_population'),
            population_active=prior.get('population_active'),
            population_coverage_rejections=prior['counts'].get('population_coverage_rejections', 0),
            population_expiries=prior['counts'].get('population_expiries', 0)),
        network_testers={k: dict(s=v['s'], b=v['b'], last=v['last'])
            for k, v in d['lr_settle'].items() if k != 'g1'},
        reaction=dict(last_step=last.get('step'), cumulative_evaluations=counters['evals'],
            cumulative_births=counters.get('novel_birth_moves', 0),
            cumulative_attempts=counters.get('novel_birth_attempts', 0),
            cumulative_moves=counters['moves'], dimension_skips=counters['dim_skips'],
            rows_since_eval=bd['rows_since_eval'], fill=bd['fill'],
            last_eval_seconds=last.get('eval_seconds'), metric_rank=last.get('metric_rank'),
            copy_moves=last.get('ordinary_copy_moves', 0), new_birth_moves=last.get('ordinary_novel_birth_moves', 0),
            mass_moves=last.get('ordinary_mass_moves', 0), support_moves=last.get('ordinary_support_moves', 0),
            global_moves=last.get('ordinary_global_moves', 0), ordinary_moves=last.get('ordinary_moves', 0),
            iso_moves=last.get('iso_moves', 0), work=last.get('work'),
            accepted_births=last.get('novel_birth', {}).get('acceptance', [])))
    reaction = entry['reaction']
    assert reaction['copy_moves'] == sum(reaction[k] for k in ('mass_moves', 'support_moves', 'global_moves'))
    assert reaction['ordinary_moves'] == reaction['copy_moves'] + reaction['new_birth_moves']
    assert reaction['ordinary_moves'] <= 51
    if previous is not None:
        dt = m['training_seconds'] - previous['training_seconds']; steps = m['step'] - previous['step']
        old = previous['reaction']
        entry['interval'] = dict(updates=steps, training_seconds=dt, seconds_per_update=dt / steps,
            reactions=reaction['cumulative_evaluations'] - old['cumulative_evaluations'],
            births=reaction['cumulative_births'] - old['cumulative_births'],
            moves=reaction['cumulative_moves'] - old['cumulative_moves'])
    records.append(entry); previous = entry
assert before == {str(p): sha(p) for p in inputs}
output.write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_REGIME_SUMMARY', records=records,
    source_sha256=sha(Path(__file__)), input_sha256=before, all_inputs_unchanged=True, torch_imported=False,
    limits=['Last reaction timing is a sampled observation, not an interval average.',
        'Changed timing does not identify CPU/GPU load, clock, synchronization or kernel causes.',
        'Population participant count belongs to last_test.tested_b, not the b advanced after that verdict.',
        'No new quality law, controller setting, serving choice or noise setting is proposed.']), indent=2) + '\n')
print(json.dumps(dict(records=len(records), final_reaction=records[-1]['reaction']['last_step'],
    evaluations=records[-1]['reaction']['cumulative_evaluations'], births=records[-1]['reaction']['cumulative_births'])))
