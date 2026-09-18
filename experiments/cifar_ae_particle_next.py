#!/usr/bin/env python
"""Run one next particle arm and its endpoint diagnostics on a fixed GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_grid import code_provenance,has_valid_summary,load_config,trainer_defaults
COMMON=ROOT/'runs/cifar_particle_ae/particle_next/PIPELINE.log'
ARMS={
 '16k_80k':{'track':'particle_16k_80k','factor':16,'start':40000,'stop':80000,'interval':10000,'gpu':'0',
             'trainer':'experiments/train_cifar_ae_scaling.py','probe':'experiments/probe_cifar_ae_information.py',
             'parent':'runs/cifar_particle_ae/particle_scaling_40k/split_16384/checkpoint_040000.pt',
             'sha':'507f0795cbb9d6ad10612c6f6cde1eb2a94c6c65a4f3ec25bde8f1b47efc4867',
             'config':'configs/cifar_particle_ae/particle_scaling_40k/split_16384.yaml'},
 '32k_40k':{'track':'particle_32k_40k','factor':32,'start':10000,'stop':40000,'interval':5000,'gpu':'1',
             'trainer':'experiments/train_cifar_ae_scaling32.py','probe':'experiments/probe_cifar_ae_information32.py',
             'parent':'runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt',
             'sha':'d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c',
             'config':'configs/cifar_particle_ae/particle_scaling_scout/split_16384.yaml'},
}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def grid(track,trainer,gpu,configs):
    folder=ROOT/f'configs/cifar_particle_ae/{track}';folder.mkdir(parents=True,exist_ok=True);paths=[]
    for name,cfg in configs.items():
        path=folder/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=folder/'manifest.json';write(manifest,paths)
    run=ROOT/f'runs/cifar_particle_ae/{track}';run.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-u','experiments/follow_grid.py','--root',str(run),'--log',str(COMMON),'--runner-log',str(run/'runner.log'),'--',
                    '--config_manifest',str(manifest),'--gpus',gpu,'--workers_per_gpu','1','--python',sys.executable,'--trainer',trainer],cwd=ROOT,check=True)
    defaults=trainer_defaults(str(ROOT/trainer));provenance=code_provenance(str(ROOT/trainer),sys.executable);rows=[]
    for path in paths:
        cfg=load_config(path,defaults);out=ROOT/cfg['out_dir'];assert has_valid_summary(str(out),cfg,provenance),out
        rows.append(json.loads((out/'summary.json').read_text()))
    return rows


def training_config(arm,smoke=False):
    spec=ARMS[arm];parent=ROOT/spec['parent'];assert hashlib.sha256(parent.read_bytes()).hexdigest()==spec['sha']
    cfg=yaml.safe_load((ROOT/spec['config']).read_text())
    track=spec['track']+('_smoke' if smoke else '')
    cfg.update(expansion_factor=spec['factor'],steps=spec['start']+8 if smoke else spec['stop'],
               eval_interval=spec['start']+8 if smoke else spec['interval'],eval_samples=128 if smoke else 50000,
               final_samples=128 if smoke else 50000,initial_eval_samples=(128 if smoke else 50000) if spec['factor']==32 else 0,
               recon_samples=64 if smoke else 10000,resume_checkpoint=spec['parent'],resume_sha256=spec['sha'],
               out_dir=f'runs/cifar_particle_ae/{track}/{arm}',max_train_seconds=7200.,keep_checkpoints=True)
    return track,cfg


def run(arm,smoke=False):
    spec=ARMS[arm];track,cfg=training_config(arm,smoke)
    rows=grid(track,spec['trainer'],spec['gpu'],{arm:cfg});r=rows[0];out=ROOT/cfg['out_dir']
    assert r['start_step']==spec['start'] and r['final']['step']==cfg['steps']
    assert r['frozen_features_unchanged'] and r['sigma_unchanged']
    curve=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines() if '"generation"' in x]
    assert [x['step'] for x in curve]==([cfg['steps']] if smoke else list(range(spec['start']+spec['interval'],spec['stop']+1,spec['interval'])))
    assert all(x['generation']['samples']==cfg['final_samples'] for x in curve)
    metrics=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert all(x['learning_rates']==dict(G=.0003,E=.0003,prior=.003,D=.00045) for x in metrics)
    if cfg['initial_eval_samples']:
        initial=json.loads((out/'initial_evaluation.json').read_text())
        if not smoke:assert abs(initial['fid']-19.4482)<.01
    report=ROOT/f'reports/cifar-particle-ae/{track}';write(report/'results.json',[{'name':arm,'curve':curve,**r}])
    if smoke:print('SMOKE certified',arm,flush=True);return
    checkpoint=out/f"checkpoint_{spec['stop']:06d}.pt";sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write(report/'CHECKPOINTS.json',{arm:{'path':str(checkpoint.relative_to(ROOT)),'sha256':sha,'fid50k':r['final']['fid']}})
    lines=[f'# {arm} training result','','Certified full-state continuation; all evaluation scores FID50k.','', '| Step | FID50k |', '|---|---:|']
    for x in curve:lines.append(f"| {x['step']} | {x['generation']['fid']:.4f} |")
    if arm=='32k_40k':
        old=json.loads((ROOT/'reports/cifar-particle-ae/particle_scaling_40k/results.json').read_text())
        baseline=next(x for x in old if x['name']=='split_16384')
        lines+=['',f"Existing16k at40k: {baseline['final']['fid']:.4f}. New32k endpoint minus16k: {r['final']['fid']-baseline['final']['fid']:+.4f}."]
    else:lines+=['',f"Starting16k checkpointFID17.0982 at40k; final improvement {17.098215435106624-r['final']['fid']:+.4f}."]
    best=min(curve,key=lambda x:x['generation']['fid'])
    lines+=['',f"Train minutes {r['train_seconds']/60:.2f}. Best sampled point {best['generation']['fid']:.4f} at{best['step']}; final {r['final']['fid']:.4f}. Target<13. No automatic next training stage."]
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    (report/'FINDINGS.md').write_text('# Interpretation\n\nReview the complete FID trajectory and endpoint feature metrics. Keep the best sampled point separate from the final checkpoint. Particle count and duration are tested in separate arms with different step endpoints; do not rank them as if trained for equal duration. No automatic next count increase or200k promotion.\n')
    probe_track=track+'_information'
    probe_cfg={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha,'out_dir':f'runs/cifar_particle_ae/{probe_track}/{arm}'}
    q=grid(probe_track,spec['probe'],spec['gpu'],{arm:probe_cfg})[0]
    assert q['parent_unchanged'] and q['frozen_state_unchanged'] and q['parent_step']==spec['stop']
    assert abs(q['information']['identical_clones']['decodable_bits'])<1e-8
    if arm=='32k_40k':
        old=json.loads((ROOT/'reports/cifar-particle-ae/particle_scaling_40k_information/results.json').read_text());base=next(x for x in old if x['name']=='split_16384')
        assert q['real_features_sha256']==base['real_features_sha256']
    p=ROOT/f'reports/cifar-particle-ae/{probe_track}';write(p/'results.json',[{'name':arm,'checkpoint_fid50k':r['final']['fid'],**q}])
    info,quality=q['information'],q['density_coverage']
    text=f"# {arm} endpoint diagnostics\n\nCertified read-only checkpoint. FID50k {r['final']['fid']:.4f}; siblingbits {info['observed']['decodable_bits']:.4f}/{info['available_sibling_bits']:.0f}; density {quality['density']:.4f}; coverage {quality['coverage']:.4f}.\n\nRestricted-decoder held-out information estimate; feature distinctions can include artifacts. Density/coverage use10000real/fake images andk5. See per-parent/control statistics in results.json.\n"
    (p/'LEADERBOARD.md').write_text(text);print(text,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.arm,a.smoke)
