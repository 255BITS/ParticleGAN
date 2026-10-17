"""Join the two fixed-input reactions and verify the metadata-only correction."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((HERE/'ra5-reaction.json').read_text());new=json.loads((HERE/'ra6-reaction.json').read_text())
assert old['status']==new['status']=='PASS'
rows=[]
for a,b in zip(old['records'],new['records']):
    keys=[key for key in a if key not in ('checks','JSON_serialization')]
    checks=dict(all_fixed_action_and_numerical_fields_exact=all(a[k]==b[k] for k in keys),
        fixed_plan_exact=a['fixed_plan_sha256']==b['fixed_plan_sha256'],
        live_EMA_optimizer_history_lineage_RNG_and_counters_exact=a['numerical_state_sha256']==b['numerical_state_sha256'],
        old_metadata_failure_reproduced=not any(a['JSON_serialization'].values()),
        corrected_whole_diagnostics_checkpoint_last_and_posthook_JSON=all(b['JSON_serialization'].values()),
        all_new_reaction_accounting_checks_pass=all(b['checks'].values()))
    assert all(checks.values()),checks
    rows.append(dict(step=a['step'],ordinary=a['ordinary_moves'],copy=a['copy_moves'],births=a['novel_births'],
        checks=checks,numerical_state_sha256=b['numerical_state_sha256'],fixed_plan_sha256=b['fixed_plan_sha256']))
composition=ROOT/'quality/ra6/COMPOSITION.json';c=json.loads(composition.read_text())
base=ROOT/'pkg-CB64-RA5';package=ROOT/'pkg-CB64-RA6'
old_source=(base/'particlegan/feature_cells.py').read_text();new_source=(package/'particlegan/feature_cells.py').read_text()
restored=new_source
for change in reversed(c['changes']):
    assert restored.count(change['after'])==1;restored=restored.replace(change['after'],change['before'],1)
assert restored==old_source and len(c['changes'])==4
changed=[str(p.relative_to(package)) for p in package.rglob('*.py') if sha(p)!=sha(base/p.relative_to(package))]
assert changed==['particlegan/feature_cells.py']
local=[HERE/'check_reaction.py',Path(__file__),HERE/'ra5-reaction.json',HERE/'ra6-reaction.json',HERE/'ra5-cpu.log',HERE/'ra6-cpu.log',composition]
receipt=dict(status='PASS',records=rows,changed_package_files=changed,four_metadata_splices_only=True,
    inverse_full_source_bytes_exact=True,no_new_count_law_or_budget_or_seed=True,
    source_sha256=new['source_sha256'],input_sha256=new['input_sha256'],
    package_source_sha256={str(p.relative_to(package/'particlegan')):sha(p) for p in sorted(package.rglob('*.py'))},
    local_source_evidence_sha256={str(p):sha(p) for p in local},
    base_package_source_sha256=old['source_sha256'],base_input_sha256=old['input_sha256'],
    cuda_initialized=False,cpu_only=True,new_training_steps=0,new_optimizer_steps=0,new_seeds=0,
    quality_verdict=None,scope='Two saved-input actual reactions exactly equal numerically; four JSON metadata splices corrected',
    full_checkpoint_scope='feature-cell backend fresh6 roundtrip and old4 atomic rejection; fulltrainer population reviewed separately')
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',fixed_cases=len(rows),receipt_sha256=sha(HERE/'receipt.json'),changes=changed)),flush=True)
