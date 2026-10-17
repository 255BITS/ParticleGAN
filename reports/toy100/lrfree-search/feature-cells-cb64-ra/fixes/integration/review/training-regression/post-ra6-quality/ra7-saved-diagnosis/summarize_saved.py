"""Join saved RA7 diagnostics with frozen RA6/RA4/E22; no scientific execution."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DIAG = HERE.parents[1] / 'post-ra5-saved-diagnosis'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--diagnosis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit('Existing output; preserve it.')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
ready = ROOT / 'quality/ra7/READY.json'
assert sha(ready) == 'f300089ce4fece3a812b890d4060e0567e833dddd236e5a955993f65ef15e1dd'
paths = [args.diagnosis, args.diagnosis.parent / 'candidate-metrics-captured.jsonl',
    DIAG / 'ra6-step2000/receipt.json', DIAG / 'ra6-step2000/FROZEN.json', ready,
    ROOT / 'configs/overrides-CB64-RA7.json', ROOT / 'configs/overrides-CB64-RA6.json',
    Path(__file__), HERE / 'PROTOCOL.md']
before = {str(p): sha(p) for p in paths}
manifest = json.loads(paths[3].read_text())
assert sha(paths[2]) == manifest['files_sha256'][str(paths[2].resolve())]
candidate = json.loads(args.diagnosis.read_text())
assert candidate['all_sources_and_checkpoints_unchanged'] and candidate['global_rng_unchanged']
assert not candidate['cuda_initialized'] and candidate['new_training_steps'] == 0
ra6 = {r['step']: r for r in json.loads(paths[2].read_text())['records']}
metric_by_step = {r['step']: r for r in map(json.loads, paths[1].read_text().splitlines())}
base, changed = [json.loads(p.read_text()) for p in (paths[6], paths[5])]
differences = {k: [base.get(k), changed.get(k)] for k in base.keys() | changed.keys()
    if base.get(k) != changed.get(k)}
assert differences == {'lr': [.00425, .0010625], 'prior_lr_mult': [2., 8.], 'd_lr_mult': [1., 4.]}
assert base['lr'] * base['prior_lr_mult'] == changed['lr'] * changed['prior_lr_mult']
assert base['lr'] * base['d_lr_mult'] == changed['lr'] * changed['d_lr_mult']
records = []
for c in candidate['records']:
    step = c['step']; controls = dict(RA6=ra6[step], **c['references'])
    rows = {}
    for name, value in dict(RA7=c, **controls).items():
        metric = value.get('saved_metrics')
        rows[name] = dict(fast={k: value['fast'][k] for k in ('precision', 'coverage', 'mass_tv', 'missing_modes')},
            ema={k: value['ema'][k] for k in ('precision', 'coverage', 'mass_tv', 'missing_modes')},
            emitted=None if metric is None else {k: metric[k] for k in ('precision', 'coverage', 'mass_tv')})
    diagnostic = metric_by_step.get(step, {}).get('diagnostics', {})
    last = diagnostic.get('birth_death', {}).get('last', {})
    prior = diagnostic.get('lr_settle', {}).get('g1', {})
    supply = c['parent_supply']
    supply_summary = None if supply is None else dict(
        flags=supply['flags'], eligible_pQ=supply['eligible_pQ'], eligible_inside=supply['eligible_inside'],
        physical_unique_copy_capacity_upper_bound=supply['total_unique_copy_capacity_upper_bound'],
        groups=supply['groups'],
        no_inside_parent_annotated_modes=[m for m in supply['modes'] if m['eligible_inside_by_reference_cell_mode'] == 0],
        fast_missing_annotated_modes=[m for m in supply['modes'] if m['mode'] in c['fast']['missing_modes']],
        minimum_inside_pool_by_annotated_mode=min(m['eligible_inside_by_reference_cell_mode'] for m in supply['modes']))
    records.append(dict(step=step, comparisons=rows, serving=c['reported_serving_table'],
        served_clean_counts_match_saved=c['served_clean_counts_match_saved'], population=c['population'],
        prior_last_test=prior.get('last'),
        scope='Participant count belongs to this last tested_b; population-clock validity is independently audited.',
        birth_event=c['birth_event'], current_birth_rows=c['current_birth_rows'], parent_supply=supply_summary,
        reaction_phases={k: last.get(k) for k in ('step', 'ordinary_mass_moves', 'ordinary_support_moves',
            'ordinary_global_moves', 'ordinary_copy_moves', 'ordinary_novel_birth_moves', 'ordinary_moves',
            'iso_moves', 'eval_seconds')},
        since_previous=c.get('since_previous')))
assert before == {str(p): sha(p) for p in paths}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_SAVED_COMPARISON', records=records,
    input_and_source_sha256=before, sources_and_inputs_unchanged=True, torch_imported=False,
    rate_scope=dict(config_differences=differences, generator_and_sigma_base_factor=.25,
        prior_base_preserved=True, critic_base_preserved=True, noise_formula_floor_mode_unchanged=True),
    quality_verdict=None,
    limits=['Intermediate metrics are not final quality qualification.',
        'RA6 is reused from its frozen saved receipt; no extra numerical re-evaluation.',
        'CPU learned partitions are current refits, not historical GPU action geometry.',
        'Mode labels annotate accessibility; they do not enter production or serving.',
        'Birth-row indices do not establish survival of an earlier incarnation.']), indent=2) + '\n')
print(json.dumps(dict(milestone=records[-1]['step'], comparisons=records[-1]['comparisons'],
    serving=records[-1]['serving'], birth_event=records[-1]['birth_event'])), flush=True)
