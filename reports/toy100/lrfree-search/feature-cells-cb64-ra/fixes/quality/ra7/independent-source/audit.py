"""Narrow independent RA7 source/config proof; no numerical or model execution."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OUT=Path(__file__).resolve().parent
AREA=ROOT/'quality/ra7'
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile'
BASE=ROOT/'pkg-CB64-RA6/particlegan'
CANDIDATE=ROOT/'pkg-CB64-RA7/particlegan'
PRIVATE=OWNER/'pkg-GROUP-COUNT/particlegan'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert not (OUT/'receipt.json').exists(), 'independent receipt already exists'
checked={}

def check(values,base=None):
    for name, expected in values.items():
        p=Path(name) if Path(name).is_absolute() else base/name
        assert sha(p)==expected,str(p)
        checked[str(p)]=expected

composition=read(AREA/'COMPOSITION.json')
assert sha(AREA/'COMPOSITION.json')=='27dc199147662c489ed33341b5d54a7e477f3e07b2c6d71d6cc00fffa24c8b8f'
checked[str(AREA/'COMPOSITION.json')]=sha(AREA/'COMPOSITION.json')
check(composition['composed_from'])
owner=read(OWNER/'READY.json');owner_freeze=read(OWNER/'FROZEN.json')
check(owner_freeze['files']);check(owner['numerical_source_sha256']);check(owner['helper_source_sha256'])
assert owner_freeze['ready_sha256']==sha(OWNER/'READY.json')
assert owner['base_package_sha256']=='6bb967c405c486d2a5cbd5dcc3bce7d1bd2cb93b3f9c7e916229f1327a8e5fc5'
base_ready=read(ROOT/'quality/ra6/READY.json')
check(base_ready['numerical_source_sha256'])
assert sha(ROOT/'quality/ra6/READY.json')==composition['base_ready_sha256']==owner['base_ready_sha256']
assert {str(p.relative_to(CANDIDATE)) for p in CANDIDATE.rglob('*.py')}==set(owner['package_source_sha256'])
assert len(owner['package_source_sha256'])==29
assert composition['source_sha256']==owner['package_source_sha256']
check(owner['package_source_sha256'],CANDIDATE);check(owner['package_source_sha256'],PRIVATE)
check(base_ready['package_source_sha256'],BASE)
digest=hashlib.sha256();changed=[]
for name in sorted(owner['package_source_sha256']):
    proposed=(CANDIDATE/name).read_bytes()
    assert proposed==(PRIVATE/name).read_bytes(),name
    digest.update(name.encode()+b'\0'+proposed+b'\0')
    if proposed!=(BASE/name).read_bytes():changed.append(name)
assert changed==['feature_cells.py']
assert digest.hexdigest()==owner['package_sha256']==composition['package_sha256']=='671404988209f615aefd1aec32ab1ef0a51807a9b07514c9bb35a662a6154c7c'

old=(BASE/'feature_cells.py').read_text();new=(CANDIDATE/'feature_cells.py').read_text()
old_ast,new_ast=ast.parse(old),ast.parse(new)
find_class=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellSnapshot')
find_method=lambda node:next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_group_counts')
old_class,new_class=find_class(old_ast),find_class(new_ast)
old_method,new_method=find_method(old_class),find_method(new_class)
get_text=lambda text,node:'\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
assert new.replace(get_text(new,new_method),get_text(old,old_method),1)==old
assert ast.dump(new_method.body[-1],include_attributes=False)==ast.dump(old_method.body[-1],include_attributes=False)
new_class.body[new_class.body.index(new_method)]=deepcopy(old_method)
assert ast.dump(new_ast,include_attributes=False)==ast.dump(old_ast,include_attributes=False)
prior=ROOT/'performance/training-regression/count-review/ra6-group-count-review'
for n in ('receipt.json','FROZEN.json','OWNER-FREEZE-RECEIPT.json','OWNER-FROZEN.json'):
    p=prior/n;checked[str(p)]=sha(p)
assert read(prior/'receipt.json')['status']=='PASS' and read(prior/'OWNER-FREEZE-RECEIPT.json')['status']=='PASS'
for n in ('FROZEN.json','OWNER-FROZEN.json'):
    record=read(prior/n)
    for key in ('files','reviewed_hashes'):
        if key in record:check(record[key])
gpu=read(ROOT/'integration/review/group-count-gpu/result.json')
phase=read(ROOT/'integration/review/group-count-gpu/PHASE-RESULT.json')
gpu_audit=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra4-quality/group-gpu-review'
assert gpu['status']==phase['status']==read(gpu_audit/'receipt.json')['status']=='PASS'
assert phase['source_integrity']=='VALID' and phase['returncode']==0 and phase['numerical_parallelism']==1
assert phase['result_sha256']==sha(ROOT/'integration/review/group-count-gpu/result.json')
assert gpu['optimizer_updates']==gpu['new_seeds']==0 and gpu['quality_verdict'] is None
check(gpu['source_sha256']);check(read(gpu_audit/'receipt.json')['verified_phase_inputs_sha256'])
for key in ('files','source_sha256'):
    value=read(gpu_audit/'FROZEN.json')
    if key in value:check(value[key])

old_config=read(ROOT/'configs/overrides-CB64-RA6.json')
new_config=read(ROOT/'configs/overrides-CB64-RA7.json')
assert old_config.keys()==new_config.keys()
delta={k:dict(before=old_config[k],after=new_config[k]) for k in old_config if old_config[k]!=new_config[k]}
assert delta==composition['config_changes']==dict(lr=dict(before=.00425,after=.0010625),
    prior_lr_mult=dict(before=2.,after=8.),d_lr_mult=dict(before=1.,after=4.))
assert sha(ROOT/'configs/overrides-CB64-RA7.json')==composition['config_sha256']=='08359ed4406148faf6915144ef4265d930829de53775eeafc4f17a11b5667b4c'
for n in ('overrides-CB64-RA6.json','overrides-CB64-RA7.json'):
    p=ROOT/'configs'/n;checked[str(p)]=sha(p)
assert new_config['lr']==old_config['lr']/4
rates={label:dict(before=old_config['lr']*(old_config[mult] if mult else 1),
    after=new_config['lr']*(new_config[mult] if mult else 1)) for label,mult in
    [('generator',None),('learned_sigma',None),('prior','prior_lr_mult'),('critic','d_lr_mult')]}
assert rates['prior']['before']==rates['prior']['after']==.0085
assert rates['critic']['before']==rates['critic']['after']==.00425

# Existing source declarations bind these arithmetic products to actual groups.
recipe=(CANDIDATE/'recipes.py').read_text();trainer=(CANDIDATE/'training.py').read_text()
assert '"lr": self.lr * self.d_lr_mult' in recipe
assert '"lr": self.lr * self.prior_lr_mult' in recipe
assert '"lr": recipe.lr' in trainer
assert 'group["lr"] / rate' in trainer and 'max(own_scale, 0.75 * prior_scale)' in trainer
assert composition['backend_schema']==owner['backend_schema']==6
assert composition['trainer_schema']==owner['trainer_schema']==5
assert new_config['output_noise_mode']=='learnable' and new_config['serve_average']==4.
assert composition['quality_verdict'] is None
assert all(sha(p)==v for p,v in checked.items())
report='''# Independent RA7 source and configuration proof

PASS for prospective source/configuration readiness. The full29-module RA7 package is byte-identical to the frozen GROUP-COUNT proposal and its existing CPU/CUDA mechanical proofs. Relative to RA6,28 modules are byte-identical; replacing only FeatureCellSnapshot._group_counts with its original method restores the whole feature_cells.py bytes and AST exactly. Owner sources, inputs, retained proofs and base numerical maps are unchanged.

Exactly three override values change: lr .00425→.0010625, prior_lr_mult2→8, d_lr_mult1→4. Generator and learned log-sigma base learning rates both become one quarter. Prior base LR stays .0085; critic base LR stays .00425. The unchanged per-group stationarity scales, applied/base intrinsic clocks and critic .75 prior-scale floor remain active. Learned sigma is coupled to this generator-base change; it is not a G-only ablation.

No source law, population survival threshold, count family, birth/copy budget, sampling/serving rule, noise formula/floor, checkpoint schema or hidden state changes. This audit executes no Torch/model code, numerical test, CUDA context, new seed or optimizer update. Existing mechanical parity is separate from trajectory quality. The candidate has no quality result; both the strict learned toy and original full Grid100 still must pass.
'''
(OUT/'REPORT.md').write_text(report)
receipt=dict(status='PASS',created_at=datetime.now(timezone.utc).isoformat(),scope='narrow independent RA7 source/config proof',
    package_sha256=digest.hexdigest(),config_sha256=sha(ROOT/'configs/overrides-CB64-RA7.json'),
    checks=dict(full29_modules_byte_exact_to_private_owner=True,unchanged_ra6_modules=28,
        only_changed_method='FeatureCellSnapshot._group_counts',inverse_whole_feature_module_bytes_and_ast=True,
        owner_ready_freeze_sources_inputs_and_cpu_gpu_parity_receipts_exact=True,
        config_exactly_three_declared_deltas=True,quarter_generator_and_learned_sigma_rates=True,
        prior_and_critic_base_rates_preserved=True,unchanged_intrinsic_time_law=True,
        no_sampling_serving_noise_or_count_population_law_change=True,backend_schema=6,trainer_schema=5),
    config_changes=delta,group_base_rates=rates,verified_hashes=checked,numerical_reruns=0,
    model_optimizer_updates=0,new_seeds=0,cuda_imports=0,quality_verdict=None,
    limitation='Prospective candidate only: strict toy and original full Grid100 remain unqualified.')
write(OUT/'receipt.json',receipt)
for n in ('audit.py','receipt.json','REPORT.md'):checked[str(OUT/n)]=sha(OUT/n)
write(OUT/'FROZEN.json',dict(status='PASS',files=checked))
print(json.dumps(dict(status='PASS',receipt=str(OUT/'receipt.json'),receipt_sha256=sha(OUT/'receipt.json'),
    freeze_sha256=sha(OUT/'FROZEN.json'),config_changes=delta,base_rates=rates)))
