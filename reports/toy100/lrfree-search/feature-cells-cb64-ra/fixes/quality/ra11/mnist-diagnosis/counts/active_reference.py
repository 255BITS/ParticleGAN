#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA=ROOT/'quality/ra11/mnist-diagnosis/counts'
CASE=ROOT.parent/'feature-cells-cuda-retest-20260929/learned/training/mnist/E22'
CURRENT=ROOT/'validation-cb64-ra11/learned/training/mnist/CB64-RA11'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def seal():
    files=[Path(__file__),AREA/'ACTIVE-REFERENCE.md',AREA/'SOURCE-INPUTS-FROZEN.json',AREA/'result.json']
    files += [CASE/name for name in ['result.json','metrics.jsonl','config.json','evaluator.json']]
    files += [CURRENT/name for name in ['config.json','evaluator.json']]
    target=AREA/'ACTIVE-SOURCE-INPUTS-FROZEN.json';assert not target.exists()
    out={'status':'FROZEN_BEFORE_MATCHED_ACTIVE39_JSON_EXTRACTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'protected_sha256':{str(p):sha(p) for p in files},'Torch_imported':False,'PT_objects_loaded':0}
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'sha256':sha(target),'files':len(files)}))
def run():
    freeze=read(AREA/'ACTIVE-SOURCE-INPUTS-FROZEN.json')
    for p,d in freeze['protected_sha256'].items():assert sha(p)==d,p
    cfg=read(CASE/'config.json');cur=read(CURRENT/'config.json')
    ev=read(CASE/'evaluator.json');cev=read(CURRENT/'evaluator.json')
    identity_keys=['seed','steps','initial_generator_sha256','initial_critic_sha256','initial_prior_sha256']
    identity={k:cfg.get(k)==cur.get(k) for k in identity_keys};assert all(identity.values())
    frame_keys=['active_dimensions','active_mask_sha256','active_mean_sha256','active_std_sha256','raw_reference_sha256','active_reference_sha256','evaluator_model_sha256']
    frame={k:ev.get(k)==cev.get(k) for k in frame_keys};assert all(frame.values()) and ev['active_dimensions']==39
    rows=[json.loads(l) for l in (CASE/'metrics.jsonl').read_text().splitlines() if l]
    assert [r['step'] for r in rows]==[0,100,250,500,750,1000,1250,1500,1750,2000]
    timeline=[]
    for r in rows:
        d=r['diagnostics'];bd=d.get('birth_death',{});m=r['metrics'];s=d.get('lr_settle',d.get('stationarity',{}))
        timeline.append({'step':r['step'],'active_embedding':m['active_embedding'],'raw_embedding':m['raw_embedding'],
            'class_mass_tv':m['class_mass_tv'],'confident_class_coverage':m['confident_class_coverage'],
            'confident_fraction':m['confident_fraction'],'pixel_clipping_fraction':m['pixel_clipping_fraction'],
            'output_sigma':d['output_sigma'],'moves':bd.get('counters',{}).get('moves'),
            'iso_moves':bd.get('counters',{}).get('iso_moves'),'last_eligible':bd.get('last',{}).get('eligible'),
            'last_iso_flagged':bd.get('last',{}).get('iso_flagged'),'table_settler':s.get('g1')})
    assert read(CASE/'result.json')['final']['step']==2000
    out={'status':'PASS_MATCHED_ACTIVE39_JSON_REFERENCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'case_path':str(CASE),'source_input_freeze_sha256':sha(AREA/'ACTIVE-SOURCE-INPUTS-FROZEN.json'),
         'initial_fixture_identities_exact':identity,'active39_frame_hashes_exact':frame,'timeline':timeline,
         'Torch_imported':False,'PT_objects_loaded':0,'model_forwards':0,'new_draws':0,'new_training_updates':0,
         'limits':['Separate preserved older E22 extraction remains descriptive and differently normalized.','No new scoring or causal experiment.']}
    target=AREA/'active-reference-result.json';assert not target.exists();target.write_text(json.dumps(out,indent=2,allow_nan=True)+'\n')
    for row in timeline:print(json.dumps({k:v for k,v in row.items() if k not in ['raw_embedding','table_settler']}))
    print(json.dumps({'status':out['status'],'result_sha256':sha(target)}))
    for p,d in freeze['protected_sha256'].items():assert sha(p)==d,p
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','run']);a=p.parse_args()
    seal() if a.mode=='seal' else run()
