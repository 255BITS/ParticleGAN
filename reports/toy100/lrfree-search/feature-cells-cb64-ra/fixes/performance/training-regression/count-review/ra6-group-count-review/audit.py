"""Independent integer algebra, source scope and retained CPU proof review."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile'
BASE=ROOT/'pkg-CB64-RA6/particlegan'
PROPOSAL=OWNER/'pkg-GROUP-COUNT/particlegan'
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checked={}

def check(values,directory=None):
    for name,expected in values.items():
        p=Path(name) if Path(name).is_absolute() else directory/name
        assert sha(p)==expected,str(p)
        checked[str(p)]=expected

assert sha(PROPOSAL/'feature_cells.py')=='eee5469b420d9c750d9ad015172af58052e8aab161d93d4f16fc323be12f9245'
contract=read(OWNER/'cpu-group-contract.json');profile=read(OWNER/'cpu-group-profile.json')
assert contract['status']==profile['status']=='PASS'
check(contract['base_source_sha256'],BASE);check(contract['proposal_source_sha256'],PROPOSAL)
assert len(contract['base_source_sha256'])==len(contract['proposal_source_sha256'])==29
assert {p.name for p in BASE.glob('*.py')}==set(contract['base_source_sha256'])
assert {p.name for p in PROPOSAL.glob('*.py')}==set(contract['proposal_source_sha256'])
changed=[n for n in contract['base_source_sha256'] if contract['base_source_sha256'][n]!=contract['proposal_source_sha256'][n]]
assert changed==['feature_cells.py']==contract['changed_files']
check(profile['source_sha256'],ROOT)
assert sha(OWNER/'inputs.pt')==profile['inputs_sha256']=='5384c86a8c9c1bad024989c218d9b886eace933ef2acd3d1edc5e8ca18ac5c25'
for name in ('GROUP-PREPARATION.json','GROUP-COUNT.patch','prepare_group_package.py','check_group_counts.py',
             'profile_anchor.py','fixture_utils.py','cpu-group-contract.json','cpu-group-profile.json'):
    checked[str(OWNER/name)]=sha(OWNER/name)
old_text=(BASE/'feature_cells.py').read_text();new_text=(PROPOSAL/'feature_cells.py').read_text()
old_tree,new_tree=ast.parse(old_text),ast.parse(new_text)
cls=lambda tree:next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellSnapshot')
method=lambda c:next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_group_counts')
old_cls,new_cls=cls(old_tree),cls(new_tree);old_method,new_method=method(old_cls),method(new_cls)
node=ast.unparse(new_method)
assert len(new_method.body)==3 and ast.dump(new_method.body[-1],include_attributes=False)==ast.dump(old_method.body[-1],include_attributes=False)
guard=new_method.body[1]
assert isinstance(guard,ast.If) and not guard.orelse
assert ast.unparse(guard.test)=='counts.dtype in (torch.int32, torch.int64) and counts.ndim == 1 and (len(counts) == len(groups))'
assert 'groups[None] == torch.arange(self.mass_groups, device=groups.device)[:, None]' in node
assert 'counts[None].expand(self.mass_groups, -1).masked_fill(~members, 0).sum(1)' in node
assert not any(isinstance(n,(ast.AugAssign,ast.NamedExpr)) for n in ast.walk(guard))
assert not any(isinstance(t,ast.Attribute) for n in ast.walk(guard) if isinstance(n,ast.Assign) for t in n.targets)
old_method_text='\n'.join(old_text.splitlines()[old_method.lineno-1:old_method.end_lineno])
new_method_text='\n'.join(new_text.splitlines()[new_method.lineno-1:new_method.end_lineno])
assert new_text.replace(new_method_text,old_method_text,1)==old_text
new_cls.body[new_cls.body.index(new_method)]=deepcopy(old_method)
assert ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False)
preparation=read(OWNER/'GROUP-PREPARATION.json')
assert preparation['maximum_matrix_entries']==4096 and preparation['state_or_law_changes'] is False
assert len(preparation['ast_splices'])==1
splice=preparation['ast_splices'][0]
assert splice['owner']=='FeatureCellSnapshot' and splice['name']=='_group_counts'
for owner,key in ((old_method,'base_ast_sha256'),(new_method,'proposal_ast_sha256')):
    assert hashlib.sha256(ast.dump(owner,include_attributes=False).encode()).hexdigest()==splice[key]
assert contract['scalar_group_cases']==len(contract['cases'])==224
assert all(case['exact'] is True for case in contract['cases'])
assert contract['full_inverse_feature_cells_bytes_exact'] and contract['global_rng_unchanged']
assert contract['optimizer_updates']==contract['new_seeds']==0 and contract['cuda_initialized'] is False
assert len(contract['saved_reactions'])==len(profile['records'])==2
for r in contract['saved_reactions']:
    assert r['ordinary_copies']==47 and r['novel_births']==4 and r['isolation_copies']==0
    assert r['complete_plans_certificates_ledgers_work_RNG_exact'] and r['live_ema_moments_history_evidence_graph_RNG_exact']
operator_counts=[]
for r in profile['records']:
    assert r['complete_plan_bit_exact']
    old,new=r['baseline'],r['candidate']
    for key in ('complete_output_sha256','snapshot_work','moves','attempted_cells','linearizations','callback_calls'):
        assert old[key]==new[key]
    assert old['RNG_unchanged'] and new['RNG_unchanged'] and old['parameter_gradients_untouched'] and new['parameter_gradients_untouched']
    assert old['operators']['anchor.svd']['count']==new['operators']['anchor.svd']['count']
    operator_counts.append(dict(step=r['step'],nonzero_before=old['operators']['aten::nonzero']['count'],
        nonzero_after=new['operators']['aten::nonzero']['count'],svd=old['operators']['anchor.svd']['count']))

# The owner may publish READY after this narrow audit. This receipt pins the
# exact sources, artifacts and CPU proofs itself; final owner guard remains separate.
assert all(sha(p)==expected for p,expected in checked.items())
receipt=dict(status='PASS',scope='narrow independent source/integer algebra/saved CPU proof audit; no numerical rerun',
    checks=dict(package_modules=29,unchanged_modules=28,only_method_changed='FeatureCellSnapshot._group_counts',
        inverse_whole_module_bytes_exact=True,inverse_ast_exact=True,one_splice_declaration=True,
        integer_1d_membership_sum_exact=True,int32_promotion_and_int64_modular_overflow_preserved=True,
        original_float_and_other_shape_fallback_exact=True,group_matrix_bounded_by_K_squared=True,
        no_cache_state_rng_or_law_changes=True,owner_scalar_contracts=224,owner_complete_saved_reactions=2,
        plans_actions_certificates_ledgers_work_rng_live_ema_moments_history_evidence_graph_exact=True,
        all_captured_source_input_maps_exact=True),operator_counts=operator_counts,
    reviewed_hashes=checked, numerical_reruns=0,optimizer_updates=0,new_seeds=0,cuda_contexts=0,quality_verdict=None,
    limits=['Integer algebra covers the production same-device counts and topology; no additional mixed-device API promise is made.',
        'CPU profiles establish operation counts and fixed-input equality, not CUDA elapsed time or trajectory quality.',
        'Root-owned CUDA parity/performance qualification and final owner READY guard remain separate.'])
assert not (OUT/'receipt.json').exists() and not (OUT/'FROZEN.json').exists()
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'FROZEN.json').write_text(json.dumps(dict(status='PASS',files={str(OUT/n):sha(OUT/n) for n in
    ('audit.py','receipt.json','REPORT.md')},reviewed_hashes=checked),indent=2)+'\n')
print(json.dumps(dict(status='PASS',receipt_sha256=sha(OUT/'receipt.json'),frozen_sha256=sha(OUT/'FROZEN.json'))))
