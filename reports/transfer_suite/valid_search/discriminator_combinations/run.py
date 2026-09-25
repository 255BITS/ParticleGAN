"""Adaptive width-focused D selection applied to one fixed recipe."""
import datetime,gzip,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-d')
from benchmarks.transfer_suite import solvability_search as search
root=Path('/tmp/pr36-valid-data-d-recipe-combinations')
previous=Path('/tmp/pr36-valid-data-d-architecture')
assert (previous/'completed.json').exists(),'finish the frozen D study first'
previous_plan=json.loads((previous/'plan.json').read_text())
previous_rows=json.loads((previous/'index.json').read_text())['records']
width_rows=[r for r in previous_rows if r['task']=='vector_unequal_width']
def width_key(row):
    verdict=row['verdict']
    return (not verdict['passed'],verdict['shortfall'],-verdict['convergence']['passing_suffix'],row['live']['component_covariance_error'],row['candidate']['name'])
fixed_name='d128_l3_f2'
selected=[next(c for c in previous_plan['cards'] if c['name']==fixed_name)]
selected += [r['candidate'] for r in sorted(width_rows,key=width_key) if r['candidate']['name']!=fixed_name][:2]
architecture_selection={'adaptive':True,'reason':'The newly found shared recipe already solves rare mass and overlap, so parent explicitly redirected this follow-up to width-focused architecture selection. First study and its frozen top3 rule are unchanged.','rule':'Include d128_l3_f2 near-miss, then best2 others by width sustained verdict, final normalized bound shortfall, longer passing suffix, lower covariance error.','keys':{r['candidate']['name']:width_key(r) for r in width_rows},'selected':selected}
hard=['vector_unequal_mass','vector_unequal_width','vector_overlap']
other=['vector_two_broad','vector_anisotropic','vector_spiral']
source=Path('/tmp/pr36-valid-recipe/screen')
original=json.loads(gzip.decompress((source/'episodes/b999_lr075_d2_p30__vector_unequal_mass.json.gz').read_bytes()))
recipe=original['candidate']
assert recipe['name']=='b999_lr075_d2_p30'
for key,value in {'betas':[0.,.999],'lr':.00075,'d_lr_mult':2.,'prior_lr_mult':30.,'reg_arm':'b_cap','reg_coeff':3.,'reg_kappa':1.25,'prior_reg':.05,'d_every':1,'g_every':1,'loss_type':'logistic','gan_mode':'rp','particles':256}.items():
    assert recipe['overrides'][key]==value
cards=[{'name':recipe['name']+'__'+c['name'],'overrides':recipe['overrides']|c['overrides']} for c in selected]
# Public runner names permit underscores, so labels remain exact and safe.
plan={'tasks':hard,'candidates':cards}
parent_plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'recipe':recipe,'architectures':selected,'architecture_selection':architecture_selection,'screen_plan':plan,'selection':'After all3 hard tasks on all3 architectures, validate the other3 original-budget data tasks for each card with sustained unequal-width success OR more than2 sustained hard-case passes. Original D recipe already passes2/3. Never count incomplete six-task rows as full-suite successes.','settings':'Only the three previously screened D architectures vary within this one fixed recipe. Original budgets/G/data/gates/particles unchanged. Seed0 CPU thread1; no longer training.'}
search.write(root/'frozen_plan.json',parent_plan)
(root/'references').mkdir(exist_ok=False)
for task in hard:
    file=source/f'episodes/b999_lr075_d2_p30__{task}.json.gz'
    shutil.copy2(file,root/'references'/file.name)
shutil.copy2(source/'source.tar.gz',root/'references/recipe_source.tar.gz')
shutil.copy2(previous/'selected.json',root/'references/original_study_selection.json')
search.write(root/'references/adaptive_architecture_selection.json',architecture_selection)
rows=search.run(plan,root/'screen')
for row in rows: row['artifact']='screen/'+row['artifact']
qualifying=[]
for card in cards:
    group=[r for r in rows if r['candidate']['name']==card['name']]
    assert len(group)==3
    passed=sum(r['verdict']['passed'] for r in group)
    width=next(r for r in group if r['spec']['name']=='vector_unequal_width')['verdict']['passed']
    if width or passed>2:qualifying.append(card)
search.write(root/'selected.json',{'criterion':parent_plan['selection'],'candidates':qualifying})
if qualifying:
    validation=search.run({'tasks':other,'candidates':qualifying},root/'validation')
    for row in validation: row['artifact']='validation/'+row['artifact']
    rows+=validation
search.write(root/'results.json',{'records':rows})
lines=['# Shared-recipe D architecture combinations','','Fixed recipe: logistic RP; cap3/kappa1.25; prior regularization.05, no particle L2; Adam(0,.999); G LR.00075, D multiplier2, prior multiplier30; 1:1 cosine. Only D architecture varies among rows. This is an explicitly adaptive width-focused follow-up using prior screen evidence; the previous frozen study remains unchanged. Original G/data/particle count/budgets and every metric bound remain unchanged. Six valid data tasks only; no intentionally poor diagnostics or forced D/batch/capacity stress cases enter selection.','','| D architecture | Unequal mass | Unequal width | Overlap | Broad | Anisotropic | Spiral | Sustained / measured |','|---|---|---|---|---|---|---|---:|']
for card in cards:
    group=[r for r in rows if r['candidate']['name']==card['name']];by={r['spec']['name']:r for r in group}
    def cell(name):
        if name not in by:return 'unmeasured'
        row=by[name];return f'{row["verdict"]["status"]} ({row["verdict"].get("convergence",{}).get("passing_suffix",0)}/24)'
    lines.append('| '+card['name'].split('__')[-1]+' | '+' | '.join(cell(n) for n in hard+other)+f' | {sum(r["verdict"]["passed"] for r in group)}/{len(group)} |')
lines+=['','Cells use sustained live verdicts, complete24-point curves and at least5 final passing observations. EMA is recorded separately. The original-D recipe already solves unequal_mass and overlap but fails unequal_width; its exact episodes are retained in references/. A full-six result is reported only after the other3 cases are run. No union of different D architectures is a shared winner.','','[Frozen plan](frozen_plan.json), [full results](results.json), [screen leaderboard](screen/README.md), [exact script](run.py), [log](progress.log).']
(root/'README.md').write_text('\n'.join(lines)+'\n')
search.write(root/'completed.json',{'new_episodes':len(rows),'screened_cards':len(cards),'fully_validated_cards':[c['name'] for c in qualifying],'errors':sum(r['verdict']['status']=='ERROR' for r in rows)})
print('COMPLETE',flush=True)
index={}
for file in sorted(root.rglob('*')):
    if file.is_file() and file.name!='archive_manifest.json':index[str(file.relative_to(root))]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
search.write(root/'archive_manifest.json',index)
