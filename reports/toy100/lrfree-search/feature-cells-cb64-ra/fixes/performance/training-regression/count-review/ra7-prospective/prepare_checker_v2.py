"""Build a new RA7-only checker from the frozen valid RA6 CPU checker."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE=ROOT/'performance/training-regression/count-review/ra6-prospective/audit_checkpoints_v2.py'
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)=='c2f595430ed9b70ae9d3cdc0383b7de7809d7540d8088dd4565f5cf78b2b4895'
HELPER=OUT/'audit_checkpoints.py'
assert not HELPER.exists()
text=BASE.read_text().replace('RA6','RA7').replace('ra6','ra7')
text=text.replace('873c4394d6cce5967dba4298bc626c96953bde744ba65960e777e89200bdec0e',
    'f300089ce4fece3a812b890d4060e0567e833dddd236e5a955993f65ef15e1dd')
changes=[]

def replace_one(old,new,label):
    global text
    assert text.count(old)==1,(label,text.count(old))
    text=text.replace(old,new,1)
    changes.append(label)

anchor="    ready = read(READY)\n"
replace_one(anchor,"    assert checked[str(VALIDATION/'source-freeze.json')] == '1724fc603970338b0ae94d58d6055a734e3cd424fcd514a53bb1ca9297426b3d'\n"+anchor,'pin lane source freeze')
anchor="    assert ready['backend_schema']==6 and ready['trainer_schema']==5\n"
replace_one(anchor,anchor+"    base_config=read(ROOT/'configs/overrides-CB64-RA6.json')\n    current_config=read(ROOT/'configs/overrides-CB64-RA7.json')\n    assert base_config.keys()==current_config.keys()\n    assert {k for k in current_config if current_config[k]!=base_config[k]}=={'lr','prior_lr_mult','d_lr_mult'}\n    assert (current_config['lr'],current_config['prior_lr_mult'],current_config['d_lr_mult'])==(.0010625,8.,4.)\n    assert current_config['lr']==base_config['lr']/4\n    assert current_config['lr']*current_config['prior_lr_mult']==base_config['lr']*base_config['prior_lr_mult']==.0085\n    assert current_config['lr']*current_config['d_lr_mult']==base_config['lr']*base_config['d_lr_mult']==.00425\n",'exact three declared config deltas')
anchor="    assert cfg['recipe']['serve_average']==4 and cfg['recipe']['z_dim']==128 and cfg['recipe']['num_particles']==1024\n"
replace_one(anchor,anchor+"    assert (cfg['recipe']['lr'],cfg['recipe']['prior_lr_mult'],cfg['recipe']['d_lr_mult'])==(.0010625,8.,4.)\n    assert state['initial_lrs']==[[.0010625,.0085,.0010625],[.00425]]\n    applied_rates=[[g['lr'] for g in opt['param_groups']] for opt in state['optimizers']]\n    assert applied_rates==record['diagnostics']['lr']\n",'genuine saved group base rates and applied diagnostics')
anchor="            population_state_load_exact=True,old_law_atomic_rejections=rejected,\n"
replace_one(anchor,"            genuine_quarter_generator_and_sigma_base_rates=True,prior_and_critic_base_rates_exact=True,\n"+anchor,'rate checks in endpoint receipt')
anchor="        birth_counters=bd['counters'],last_phase=phase,\n"
replace_one(anchor,"        base_learning_rates=state['initial_lrs'],applied_learning_rates=applied_rates,\n"+anchor,'base/applied rates retained')
anchor="        metrics=record['metrics'],quality_verdict=None if step<2000 else\n            ('PASS' if record['metrics']['precision']>=.9 and record['metrics']['coverage']==25 and record['metrics']['mass_tv']<=.1 else 'FAIL'),\n"
replace_one(anchor,"        metrics=record['metrics'],quality_verdict=None,\n        quality_scope='Watcher records frozen metrics only; numerical queue owns strict quality adjudication.',\n",'saved metrics with no separate quality adjudication')
anchor="        population_trace=[dict(step=r['step'],population=r.get('population'),serving=r.get('serving'),\n"
replace_one(anchor,anchor+"            base_learning_rates=r.get('base_learning_rates'),applied_learning_rates=r.get('applied_learning_rates'),\n",'rate fields in summary')
ast.parse(text)
HELPER.write_text(text)
receipt=dict(status='DERIVED_FROM_FROZEN_CPU_CHECKER',base_checker=str(BASE),base_checker_sha256=sha(BASE),
    checker_sha256=sha(HELPER),changes=['RA7 paths/variant/READY identity/new private output']+changes,
    old_checker_untouched=True,production_source_edits=0,
    retained_private_failure='failed-attempt1/generation-error.json: indentation mismatch in source-construction anchor; no numerical run')
(OUT/'DERIVATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(helper=str(HELPER),sha256=sha(HELPER))))
