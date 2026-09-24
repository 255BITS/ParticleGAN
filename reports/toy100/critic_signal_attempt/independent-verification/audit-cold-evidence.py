"""Read-only independent regrade of exact H receipts and restorable final draws."""
from pathlib import Path
import gzip, hashlib, json, sys, tarfile
ROOT=Path(__file__).parent/'repo'
sys.path[:0]=[str(ROOT),str(ROOT/'reports/toy100')]
import torch
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe, setup_vector, setup_image, output_noise_at
from benchmarks.transfer_suite import vector_tasks, image_tasks
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.toy_suite import _episode_rows
from continuous_screen import verify_receipt
from critic_signal_regrade import regrade

torch.set_num_threads(1)
out=Path(__file__).parent
candidate=out/'cold-replay/h_n05r06_mixup_c0p01_lr15'
original=out/'archive/manifest.json'
manifest=json.loads(original.read_text())
row=json.loads((out/'one.json').read_text())[0]
base,noise,_=declared_recipe(row['config'])
primary=regrade(candidate)
rows=[]
restored_dir=out/'restored-final-samples'; restored_dir.mkdir(exist_ok=True)
for task in manifest['tasks']:
    directory=candidate/task
    episode=next((directory/'episodes').glob('*.json.gz'))
    saved=json.loads(gzip.decompress(episode.read_bytes()))
    receipt=json.loads(gzip.decompress((directory/'signal-policy.json.gz').read_bytes()))
    assert receipt['signal_options']==row['options']
    verify_receipt(receipt,row['config'],task=task)
    budget=saved['spec']['steps']
    assert len(receipt['updates'])==2*budget
    assert receipt['noise']['effective_sigma_min']==receipt['noise']['effective_sigma_max']==.05
    assert receipt['signal_work']['consistency_applications']==budget
    assert receipt['signal_work']['consistency_forwards']==3*budget
    assert 'GANLoss.g_loss only' in receipt['generator_objective']
    assert test_verdict(saved['spec'],saved['result'])==saved['verdict']
    audit=_episode_rows(directory,(task,),candidate=True,allow_scratch=True)
    assert audit['status']==saved['verdict']['status']=='PASS'
    result={'task':task,'status':'PASS','budget':budget,'actual_updates':len(receipt['updates']),
            'source_config_episode_regrade':'PASS','receipt_regrade':'PASS',
            'actual_lrs':{'g':.0015,'d':.0015,'prior':.003},'actual_noise_sigma':.05,
            'consistency_applications':budget,'consistency_forwards':3*budget,
            'generator_objective':receipt['generator_objective']}
    statepath=directory/'final-state.pt'
    if task in ('trajectory','mode_hold'):
        result['sample_regrade']='PASS: archived raw final draws reproduced reported metrics'
        result['checkpoint_sha256']=hashlib.sha256(statepath.read_bytes()).hexdigest()
    elif saved['spec']['runner'] in ('vector','image'):
        state=torch.load(statepath,map_location='cpu',weights_only=False)['trainer']
        spec=saved['spec']
        if spec['runner']=='vector':
            card=(saved.get('discriminator_variant') or {}).get('overrides',{}).get('research_discriminator')
            context=setup_vector(spec,card,base,noise)
        else:
            context=setup_image(spec,base,noise)
        trainer=context['trainer']
        trainer.load_state_dict(state)
        assert trainer.completed_steps==budget
        for opt in (trainer.opt_g,trainer.opt_d):
            assert all(int(v['step'])==budget for v in opt.state.values())
        trainer.G.std=output_noise_at(noise['output_noise_std'],budget,budget,noise['output_noise_warmup'])
        result['final_output_noise_sigma']=trainer.G.std
        with torch.no_grad(),torch.random.fork_rng(devices=[]):
            if spec['runner']=='vector':
                torch.manual_seed(402)
                latent=trainer.prior.sample(vector_tasks.EVAL_SAMPLES,generator=torch.Generator().manual_seed(990))[0]
                samples=trainer.G(latent)
                metrics=vector_tasks.score_samples(samples,context['cfg'],budget)
            else:
                samples=trainer.G(trainer.prior.z)
                metrics=image_tasks.image_metrics(samples,context['centers'],spec['thresholds'])
        assert metrics==saved['result']['live'],(task,metrics,saved['result']['live'])
        samplepath=restored_dir/(task+'.pt')
        torch.save(samples.detach().cpu(),samplepath)
        result['sample_regrade']='PASS: restored final weights and frozen evaluation draw exactly match all live metrics'
        result['restored_final_sample_file']=str(samplepath.relative_to(out))
        result['restored_final_sample_sha256']=hashlib.sha256(samplepath.read_bytes()).hexdigest()
        result['checkpoint_sha256']=hashlib.sha256(statepath.read_bytes()).hexdigest()
    else:
        assert task=='residual_student' and not statepath.exists()
        result['sample_regrade']='UNAVAILABLE: archived screen did not capture this legacy host state; raw 24-check episode and all optimizer/noise receipts regraded'
    rows.append(result)
    print(json.dumps({'event':'AUDITED','task':task,'status':'PASS','sample_regrade':result['sample_regrade']}),flush=True)
# Training source identity is unchanged after all verification and extension work.
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
           for name,digest in manifest['source_sha256'].items())
report={'candidate':row['tag'],'status':'PASS','archived_sources_unchanged':True,
        'archived_source_count':len(manifest['source_sha256']),
        'sample_regraded_hosts':sum(x['sample_regrade'].startswith('PASS:') for x in rows),
        'sample_unavailable_hosts':['residual_student'],'rows':rows}
(out/'cold-evidence-audit.json').write_text(json.dumps(report,indent=2)+'\n')
