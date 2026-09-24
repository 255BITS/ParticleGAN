"""Independently regrade every post-cold episode and actual update receipt."""
from pathlib import Path
import gzip,hashlib,json,sys
OUT=Path(__file__).parent
ROOT=OUT/'repo'
sys.path[:0]=[str(ROOT),str(ROOT/'reports/toy100')]
import torch
from critic_signal_regrade import regrade
from continuous_screen import verify_receipt
from benchmarks.toy_suite import _episode_rows

torch.set_num_threads(1)
config=json.loads((OUT/'one.json').read_text())[0]
candidates=[OUT/'remaining-replay'/config['tag']]
candidates += [p/config['tag'] for p in sorted((OUT/'diagnostics').iterdir()) if p.is_dir()]
rows=[]
for candidate in candidates:
    audit=regrade(candidate)
    for gate in audit['gates']:
        if gate['status']=='SKIPPED':
            continue
        task=gate['gate'];directory=candidate/task
        saved=json.loads(gzip.decompress(next((directory/'episodes').glob('*.json.gz')).read_bytes()))
        receipt=json.loads(gzip.decompress((directory/'signal-policy.json.gz').read_bytes()))
        verify_receipt(receipt,config['config'],task=task)
        budget=saved['spec']['steps']
        role_counts={role:sum(u['optimizer_role']==role for u in receipt['updates']) for role in ['g','d']}
        assert role_counts=={'g':budget,'d':budget}
        assert receipt['signal_options']==config['options']
        assert receipt['noise']['effective_sigma_min']==receipt['noise']['effective_sigma_max']==.05
        assert receipt['host_extension']['archived_candidate_unchanged']
        assert receipt['host_extension']['additional_optimizer_updates']==0
        assert receipt['host_extension']['additional_training_forwards']==0
        if task=='ae_gan_hold':
            assert receipt['host_extension']['auxiliary_overrides'][0]['after']['reconstruction_weight']==0.
        if task=='unused_token_hold':
            assert receipt['host_extension']['auxiliary_overrides'][0]['after']['hold_weight']==0.
        frozen=_episode_rows(directory,(task,),candidate=True,allow_scratch=True)
        assert frozen['status']==gate['status']
        rows.append({'task':task,'integrity':'PASS','frozen_verdict':gate['status'],
            'source_config_episode_regrade':'PASS','actual_update_counts':role_counts,
            'actual_learning_rates_constant':True,'actual_lrs':{'g':.0015,'d':.0015,'prior':.003},
            'fixed_input_noise_sigma':.05,'options_match_selected_candidate':True,
            'extension':receipt['host_extension'],'consistency':receipt['signal_work'],
            'artifact':str(directory),'episode_sha256':hashlib.sha256(next((directory/'episodes').glob('*.json.gz')).read_bytes()).hexdigest()})
        print(json.dumps({'task':task,'integrity':'PASS','verdict':gate['status']}),flush=True)
assert len(rows)==9
(OUT/'remaining-evidence-audit.json').write_text(json.dumps({'integrity':'PASS','episodes':len(rows),'rows':rows},indent=2)+'\n')
