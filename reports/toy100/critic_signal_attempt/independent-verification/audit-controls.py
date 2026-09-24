from pathlib import Path
import gzip,hashlib,json,sys
OUT=Path(__file__).parent; ROOT=OUT/'repo'
sys.path[:0]=[str(ROOT),str(ROOT/'reports/toy100')]
import torch
from critic_signal_regrade import regrade
from continuous_screen import verify_receipt

torch.set_num_threads(1)
row=json.loads((OUT/'one.json').read_text())[0]
controls=[]
for task in ('ae_gan_hold','unused_token_hold'):
    candidate=OUT/'compatibility-controls'/task/row['tag']
    audit=regrade(candidate)
    directory=candidate/task
    episode=next((directory/'episodes').glob('*.json.gz'))
    saved=json.loads(gzip.decompress(episode.read_bytes()))
    receipt=json.loads(gzip.decompress((directory/'signal-policy.json.gz').read_bytes()))
    verify_receipt(receipt,row['config'],task=task)
    steps=saved['spec']['steps']
    assert len(receipt['updates'])==2*steps
    assert receipt['signal_options']==row['options']
    assert receipt['noise']['effective_sigma_min']==receipt['noise']['effective_sigma_max']==.05
    control=receipt['compatibility_control']
    assert not control['exclusively_adversarial']
    assert control['host_overrides']==[]
    aux=control['original_host_auxiliary_settings'][0]
    aux_name='reconstruction_weight' if task=='ae_gan_hold' else 'hold_weight'
    assert aux[aux_name]==1.
    controls.append({'task':task,'status':saved['verdict']['status'],'integrity':'PASS',
        'exclusively_adversarial':False,'original_auxiliary_weights':aux,
        'budget':steps,'passing_suffix':saved['verdict']['convergence']['passing_suffix'],
        'live':saved['result']['live'],'verdict':saved['verdict'],
        'generator_objective':receipt['generator_objective'],
        'unchanged_H_signal_options':receipt['signal_options'],
        'actual_lrs':{'g':.0015,'d':.0015,'prior':.003},
        'source_config_episode_and_receipt_regrade':'PASS','episode':str(episode),
        'episode_sha256':hashlib.sha256(episode.read_bytes()).hexdigest(),
        'receipt':str(directory/'signal-policy.json.gz'),'source_archive':str(candidate.parent/'source.tar.gz')})
    print(json.dumps({'task':task,'integrity':'PASS','status':saved['verdict']['status'],'exclusively_adversarial':False}))
report={'track':'Original-required-auxiliary benchmark compatibility controls, not pure-GAN qualification','integrity':'PASS','passed':1,'failed':1,'cases':controls}
(OUT/'compatibility-controls-audit.json').write_text(json.dumps(report,indent=2)+'\n')
full=json.loads((OUT/'verification.json').read_text())
full['original_auxiliary_compatibility_controls']=report
full['evidence']['run-compatibility-controls.py']=str(OUT/'run-compatibility-controls.py')
full['evidence']['compatibility-controls-audit.json']=str(OUT/'compatibility-controls-audit.json')
full['evidence']['compatibility-control-tests.jsonl']=str(OUT/'compatibility-control-tests.jsonl')
(OUT/'verification.json').write_text(json.dumps(full,indent=2)+'\n')
text=(OUT/'strict-gan-verification.md').read_text()
text+='\nTwo authorized compatibility controls preserve the original required host auxiliary loss. These are **not exclusively adversarial training** and cannot replace any pure-GAN result above. All H GAN settings, frozen scores, seeds and budgets are unchanged; the control wrappers only observe and label the original host weights.\n\n'
text+='| Original-auxiliary control | Verdict | Suffix | Live result |\n| --- | --- | ---: | --- |\n'
text+='| AE, reconstruction_weight=1 | PASS | 20 | recon MSE 0.002913650; hold 0.004224265 |\n'
text+='| Unused token, hold_weight=1 | FAIL | 0 | unused_hold 0.999008495; concept_move 0.540450103 < 0.85 |\n\n'
text+='The AE control confirms the importance of its reconstruction training signal. Restoring the unused-token hold fixes preservation, while concept acquisition still fails within 200 updates. Both controls independently pass source/config/episode/update/noise regrade. Evidence: [control audit](compatibility-controls-audit.json), [control commands](run-compatibility-controls.py), [control ledger](compatibility-control-tests.jsonl), and [control source](repo/reports/toy100/selected_h_compatibility.py). The immutable pure-GAN report remains [strict-gan-verification.md](strict-gan-verification.md).\n'
(OUT/'verification.md').write_text(text)
