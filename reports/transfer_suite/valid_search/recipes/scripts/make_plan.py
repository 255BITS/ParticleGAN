from pathlib import Path
import gzip,json,hashlib
root=Path('/ml2/hypergan/ParticleGAN-pr36-valid-recipe')
out=Path('/tmp/pr36-valid-recipe')
fixed=dict(gan_mode='rp',loss_type='logistic',reg_arm='b_cap',reg_coeff=3.,reg_kappa=1.25,prior_reg=.05,particles=256)
raw=[
 ('b999_lr05_p20',.0005,1.5,20.,[0.,.999],1,1),
 ('b999_lr075_p15',.00075,1.5,15.,[0.,.999],1,1),
 ('b999_lr15_p5',.0015,1.5,5.,[0.,.999],1,1),
 ('b999_lr075_d3_p20',.00075,3.,20.,[0.,.999],1,1),
 ('b999_d3_p10',.001,3.,10.,[0.,.999],1,1),
 ('b999_d075_p20',.001,.75,20.,[0.,.999],1,1),
 ('b99_d3_p30',.001,3.,30.,[0.,.99],1,1),
 ('b995_lr075_d2_p20',.00075,2.,20.,[0.,.995],1,1),
 ('b999_lr075_d2_p30',.00075,2.,30.,[0.,.999],1,1),
 ('mom05_lr05_d2_p20',.0005,2.,20.,[.5,.999],1,1),
 ('mom05_d2_p10',.001,2.,10.,[.5,.999],1,1),
 ('mom09_lr03_d3_p20',.0003,3.,20.,[.9,.999],1,1),
 ('g2_b999_p20',.001,1.5,20.,[0.,.999],1,2),
 ('g2_b999_lr15_d1',.0015,1.,10.,[0.,.999],1,2),
 ('g2_b999_lr2_d075',.002,.75,10.,[0.,.999],1,2),
 ('g3_b999_lr15_p20',.0015,1.5,20.,[0.,.999],1,3),
 ('d2_b999_lr075_d3_p20',.00075,3.,20.,[0.,.999],2,1),
 ('b999_lr075_d3_p5',.00075,3.,5.,[0.,.999],1,1),
]
cards=[dict(name=name,overrides=fixed|dict(lr=lr,d_lr_mult=dm,prior_lr_mult=pm,betas=betas,d_every=de,g_every=ge)) for name,lr,dm,pm,betas,de,ge in raw]
plan={'candidates':cards,'tasks':['vector_unequal_mass','vector_unequal_width','vector_overlap']}
(out/'screen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
# Compare effective optimizer/config fields against already executed episodes.
keys=set(cards[0]['overrides'])
prior=[]
for phase in ('screen','combinations','geometry','regressions'):
 for path in (root/'reports/transfer_suite/solvability/vectors'/phase/'episodes').glob('*.json.gz'):
  row=json.loads(gzip.decompress(path.read_bytes()));spec=row['spec']
  for card in cards:
   equivalent=all(spec.get(k,{'gan_mode':'rp','loss_type':'logistic'}.get(k))==v for k,v in card['overrides'].items())
   assert not equivalent,(card['name'],str(path))
  prior.append({'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidate':row['candidate']})
(out/'duplicate_audit.json').write_text(json.dumps({'checked_episodes':len(prior),'prior_episodes':prior,'new_card_count':len(cards),'exact_duplicates':[]},indent=2)+'\n')
print('Frozen',len(cards),'cards; checked',len(prior),'previous episodes; no exact effective-config duplicates.')
