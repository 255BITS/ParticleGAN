"""Seal this independent reserved diagnostic, never write to its inputs."""
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p,v: Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert not (HERE/'FROZEN.json').exists()
data = read(HERE/'observations.json')
assert data['status'] == 'VALID'
assert all(sha(p) == expected for p,expected in data['input_sha256'].items())
assert sha(HERE/'diagnose.py') == data['script_sha256']
assert not data['cuda_initialized'] and data['global_rng_unchanged']
assert all(data[k] == 0 for k in ('model_forward_calls','model_gradient_calls','optimizer_calls','new_seeds','new_quality_emissions'))
checks = dict(saved_sources_and_checkpoints_exact=True,
    current_row_evidence_missingness_not_assumed_random=True,
    all_saved_pair_identification_intervals_span_zero=True,
    deterministic_permutation_population_flux_reachable=True,
    no_ancestor_evidence_assigned_to_newborn=True,
    first_moment_cancellation_control=True,
    bounded_conditional_mean_null_proof_and_exact_fixed_enumeration=True,
    no_retrospective_RA7_population_certificate=True,
    design_reserved_not_production_implemented=True,
    read_only_schema_inspection_failure_preserved=True)
for row in data['saved']:
    for pairs in row['pair_completion_bounds'].values():
        assert all(p['full_population_identification_interval'][0] <= 0 <= p['full_population_identification_interval'][1] for p in pairs)
receipt = dict(status='VALID', design_status='RESERVED',
    scope='compact independent saved-state and tiny fixed population-flux proof',
    checks=checks, reviewed_hashes=data['input_sha256'],
    saved_summary=[{k:r[k] for k in ('step','required_rows','current_window_participation','table_s','table_b','generator_s','generator_stamp')} for r in data['saved']],
    synthetic=data['synthetic'], bounded_betting=data['bounded_betting_example'],
    original_quality='RA7 strict toy FAIL; both toy and original full Grid100 required.',
    next_candidate='Separate empirical paired-average anti-blur guard; existing population training law unchanged.',
    model_forward_calls=0, model_gradient_calls=0, optimizer_calls=0,
    new_seeds=0, new_quality_emissions=0, cuda_initialized=False)
write(HERE/'receipt.json',receipt)
files={str(p):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
write(HERE/'FROZEN.json',dict(status='VALID',design_status='RESERVED',files=files,inputs=data['input_sha256']))
print(json.dumps(dict(status='VALID',design_status='RESERVED',receipt_sha256=sha(HERE/'receipt.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))
