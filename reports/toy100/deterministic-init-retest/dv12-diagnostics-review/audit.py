"""Read-only DV12 scalar/controller receipt checks; standard library only."""
from pathlib import Path
import importlib.util,json,math,hashlib
E=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('raw',E/'audit_public3_runtime.py');raw=importlib.util.module_from_spec(spec);spec.loader.exec_module(raw)
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(E/'api-dv12-runtime-audit.json');p=Path(a['source']);rows=[json.loads(s) for s in (p/'learning-rates.jsonl').read_text().splitlines()];recipe=read(p/'runtime.json')['recipe']
assert a['status']=='PASS' and recipe['total_steps'] is None and recipe['continuous_policy']=='dv12'
state,clocks=raw.checkpoint(p/'final-state.pt');initial,_=raw.checkpoint(p/'initial-state.pt')
previous={'mobility':1.,'payoff_error':0.,'alignment':0.,'data_memory':0.,'data_drive':0.}
rate_ranges=[[],[],[]];clip=[]
for i,row in enumerate(rows,1):
 c=row['controller'];assert c['updates']==i and c['variant']=='dv12'
 target=max(c['data_drive'],min(1.,previous['payoff_error']**2))
 speed=.05 if target>previous['mobility'] else .005
 assert c['mobility']==previous['mobility']+speed*(target-previous['mobility'])
 speed=.05 if c['data_drive']>previous['data_memory'] else .005
 assert c['data_memory']==previous['data_memory']+speed*(c['data_drive']-previous['data_memory'])
 assert c['data_drive']==min(1.,max(0.,(c['data_score']-3.)/3.))
 expected_trust=1./(1.+(max(0.,c['game_ratio']-1.)*(1.-previous['data_drive']))**2)
 assert c['game_trust']==expected_trust
 network=(.01+.99*c['mobility'])*c['game_trust'];prior=(.05+.95*c['mobility'])*c['game_trust']
 expected=[recipe['lr']*network,recipe['lr']*recipe['prior_lr_mult']*prior,recipe['lr']*recipe['d_lr_mult']*network/(1.+previous['payoff_error']**2)]
 rates=[row['applied_group_rates'][0][0]['lr'],row['applied_group_rates'][0][1]['lr'],row['applied_group_rates'][1][0]['lr']]
 for j,(x,y) in enumerate(zip(rates,expected)):
  assert math.isclose(x,y,rel_tol=2e-15,abs_tol=1e-18),(i,j,x,y)
  rate_ranges[j].append(x)
 assert row['input_noise']==0 and row['output_noise']==.029
 loss=row['losses'];error=max(0.,(loss['loss_gan']-(loss['loss_d']-loss['penalty']))/math.log(2.))
 assert abs(c['payoff_error']-(previous['payoff_error']+.02*(error-previous['payoff_error'])))<1e-7
 assert len(c['latent_bandwidth'])==4 and all(math.isfinite(v) and v>0 for v in c['latent_bandwidth'])
 assert len(c['latent_applications'])==2
 for app in c['latent_applications']:
  assert 0<=app['radius_min']<=app['radius_mean']<=app['radius_max'] and 0<=app['clipped_fraction']<=1 and app['perturbation_rms']>=0
  clip.append(app['clipped_fraction'])
 previous=c
assert len(rows)==1200
controller=state['trainer']['controller']
for k in ['variant','mobility','payoff_error','game_ratio','game_trust','updates','reopens','closed','data_memory','data_drive','data_score','alignment','last_cosine','latent_applications']:
 assert controller[k]==rows[-1]['controller'][k],k
out={'status':'PASS','scope':'Independent source and all1200 scalar/rate/controller receipts; no Torch, training, GPU or numerical field replay','runtime_audit_sha256':sha(E/'api-dv12-runtime-audit.json'),'source':str(p),'source_zip_sha256':sha(p/'source.zip'),'learning_rates_sha256':sha(p/'learning-rates.jsonl'),'result':a['summary'],'recipe':recipe,'verified_rows':1200,'rate_ranges':{k:[min(v),max(v)] for k,v in zip(['G','prior','D'],rate_ranges)},'final_rates':rates,'final_controller':rows[-1]['controller'],'clipped_fraction_range':[min(clip),max(clip)],'observations':'first new-initialization passing screen, not a matched old fail-to-pass claim','limits':['Scalar consistency verifies applied-rate ordering and retained diagnostics; it does not independently recompute data features, gradients or nearest-neighbor distances.','DV12 learns latent bandwidth and clips each perturbation at half the distance to a distinct nearest prior point; this stochastic support perturbation applies to evaluation too, before .029 output noise.','Standalone factories do not own the full controller/sampler/update transaction; this result exercises public GANTrainer.','One tiny screen does not establish long stability or broad default qualification.']}
(E/'api-dv12-diagnostics-audit.json').write_text(json.dumps(out,indent=2)+'\n')
(E/'api-dv12-diagnostics-audit.md').write_text('''# DV12 fixed initialization screen: independent diagnostics review\n\nThe first passing new-initialization screen is valid: 12/24 observations pass, first arrival650, all12 observations from650 through1200 pass, final8 modes/HQ .9833984375; minimum HQ afterarrival .9033203125. This is not a matched old fail-to-pass claim.\n\nThe existing complete source/raw-checkpoint/runtime audit is supplemented by independent checks of all1,200 applied G/prior/D rate rows. DV12 is horizon-free with adaptive rates, not constant numeric rates. The public trainer first observes prior/game/real data, applies network/prior scales from current mobility and trust, then multiplies D by the critic factor from the preceding accepted payoff estimate. Current gradients update the payoff estimate only afterward. All rows match that ordering; no LR compounding occurs. Input noise remains0 and output noise.029. Raw final controller scalars match the final diagnostic. Native17 CPU clocks finish at1200 with CUDA moments.\n\nFrozen source preserves DV12 local support: noise uses learned latent bandwidth; each displacement is clipped at half the nearest distinct prior-point distance. Both training calls and evaluation use this support perturbation before output noise. It is not a deterministic-prior-only evaluator. The initializer port leaves controller, sampler and update arithmetic unchanged, and bandwidth derives after deterministic prior initialization.\n\nThe report checks scalar consistency without recomputing gradients, features or distances. Existing unequal-mass and broader API constraints remain separate; this tiny pass does not qualify a default. Full ranges, final diagnostics, source hashes and limitations are in api-dv12-diagnostics-audit.json. No Torch import, training or GPU execution was performed.\n''')
print(json.dumps({'status':'PASS','summary':a['summary'],'rate_ranges':out['rate_ranges']}))
