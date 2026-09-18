#!/usr/bin/env python
"""Extend8192/16384 to40k, certify FID, then measure endpoint information/quality."""
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
TRAINER='experiments/train_cifar_ae_scaling.py'
TRACK='particle_scaling_40k'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def grid(manifest,track,trainer):
    run=f'runs/cifar_particle_ae/{track}'
    subprocess.run([sys.executable,'-u','experiments/follow_grid.py','--root',run,'--log',f'{run}/PIPELINE.log','--',
                    '--config_manifest',str(manifest),'--gpus','0,1','--workers_per_gpu','1','--python',sys.executable,'--trainer',trainer],cwd=ROOT,check=True)


def prepare():
    previous=json.loads((ROOT/'reports/cifar-particle-ae/particle_scaling_scout/results.json').read_text())
    assert {r['name'] for r in previous}=={'split_8192','split_16384'}
    provenance=code_provenance(str(ROOT/TRAINER),sys.executable)
    paths=[]
    for row in sorted(previous,key=lambda r:r['config']['expansion_factor']):
        assert has_valid_summary(str(ROOT/row['config']['out_dir']),row['config'],provenance)
        parent=ROOT/row['config']['out_dir']/'checkpoint_020000.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==row['checkpoint_sha256']
        cfg={**row['config'],'steps':40000,'eval_interval':5000,'eval_samples':50000,'final_samples':50000,
             'initial_eval_samples':0,'out_dir':f'runs/cifar_particle_ae/{TRACK}/{row["name"]}',
             'resume_checkpoint':str(parent.relative_to(ROOT)),'resume_sha256':row['checkpoint_sha256'],
             'max_train_seconds':3600.}
        path=ROOT/f'configs/cifar_particle_ae/{TRACK}/{row["name"]}.yaml'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=ROOT/f'configs/cifar_particle_ae/{TRACK}/manifest.json';write(manifest,paths);return manifest


def analyze():
    defaults=trainer_defaults(str(ROOT/TRAINER));provenance=code_provenance(str(ROOT/TRAINER),sys.executable)
    paths=json.loads((ROOT/f'configs/cifar_particle_ae/{TRACK}/manifest.json').read_text());rows=[];checkpoints={}
    for path in paths:
        cfg=load_config(path,defaults);run=ROOT/cfg['out_dir'];assert has_valid_summary(str(run),cfg,provenance),run
        summary=json.loads((run/'summary.json').read_text());assert summary['start_step']==20000 and summary['final']['step']==40000
        assert summary['frozen_features_unchanged'] and summary['sigma_unchanged']
        metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        curve=[m for m in metrics if 'generation' in m];assert [m['step'] for m in curve]==[25000,30000,35000,40000]
        assert all(m['generation']['samples']==50000 for m in curve)
        assert all(m['learning_rates']==dict(G=.0003,E=.0003,prior=.003,D=.00045) for m in metrics)
        assert json.loads((run/'resume.json').read_text())['interventions']=={}
        rows.append({'name':run.name,'curve':curve,**summary})
        p=run/'checkpoint_040000.pt';checkpoints[run.name]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fid50k':summary['final']['fid']}
    assert rows[0]['rng_sha256']==rows[1]['rng_sha256']
    report=ROOT/f'reports/cifar-particle-ae/{TRACK}'
    write(report/'results.json',rows);write(report/'CHECKPOINTS.json',checkpoints)
    baseline=next(r for r in json.loads((ROOT/'reports/cifar-particle-ae/particle_expansion_40k/results.json').read_text()) if r['name']=='split_4096')
    comparison=[baseline,*rows]
    lines=['# Larger particle-count continuation','','2/2 new runs certified. Existing4096 trajectory reused as benchmark.','',
           '| Particles | FID25k | FID30k | FID35k | FID40k | Train minutes |','|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        fids=' | '.join(f"{m['generation']['fid']:.4f}" for m in r['curve'])
        lines.append(f"| {r['config']['num_particles']*r['config']['expansion_factor']} | {fids} | {r['train_seconds']/60:.2f} |")
    lines+=['','All runs continue their respective20k checkpoints with unchanged rates, optimizer/EMA/RNG and oneD update. FID50k uses the established protocol; no seed repeats or automatic promotion past40k.']
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    winner=min(rows,key=lambda r:r['final']['fid']);best=min((m['generation']['fid'],r['name'],m['step']) for r in comparison for m in r['curve'])
    finding=f"Best new final checkpoint: {winner['name']} FID50k {winner['final']['fid']:.4f}. Lowest sampled FID across all compared trajectories: {best[0]:.4f} ({best[1]}, step{best[2]}). Keep selected minimum distinct from endpoint."
    recommendation=('Review endpoint diagnostics and the recent FID trajectory before extending the leading particle-count arm further.' if winner['final']['fid']<baseline['final']['fid'] else 'Larger counts do not beat the existing4096 endpoint; inspect adaptation, coverage and the full curve before choosing further scaling.')
    (report/'FINDINGS.md').write_text('# Continuation findings\n\n'+finding+'\n\n'+recommendation+'\n\nEndpoint feature-information and density/coverage probes run after training. No automatic further training.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for r in comparison:ax.plot([m['step']/1000 for m in r['curve']],[m['generation']['fid'] for m in r['curve']],'-o',label=r['name'])
    ax.axhline(13,color='black',linestyle=':',label='target13');ax.set(xlabel='Joint updates (thousands)',ylabel='FID50k');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(report/'curves.png',dpi=170);plt.close(fig)
    return rows


def diagnostics(rows):
    track='particle_scaling_40k_information';trainer='experiments/probe_cifar_ae_information.py';paths=[]
    for row in rows:
        parent=ROOT/row['config']['out_dir']/'checkpoint_040000.pt'
        cfg={'checkpoint':str(parent.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
             'out_dir':f'runs/cifar_particle_ae/{track}/{row["name"]}'}
        path=ROOT/f'configs/cifar_particle_ae/{track}/{row["name"]}.yaml';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=ROOT/f'configs/cifar_particle_ae/{track}/manifest.json';write(manifest,paths)
    grid(manifest,track,trainer)
    defaults=trainer_defaults(str(ROOT/trainer));provenance=code_provenance(str(ROOT/trainer),sys.executable);probes=[]
    by_name={r['name']:r for r in rows}
    for path in paths:
        cfg=load_config(path,defaults);out=ROOT/cfg['out_dir'];assert has_valid_summary(str(out),cfg,provenance)
        r=json.loads((out/'summary.json').read_text());assert r['parent_unchanged'] and r['frozen_state_unchanged']
        assert r['parent_step']==40000 and r['density_coverage']['real_samples']==r['density_coverage']['fake_samples']==10000
        assert abs(r['information']['identical_clones']['decodable_bits'])<1e-8
        probes.append({'name':out.name,'checkpoint_fid50k':by_name[out.name]['final']['fid'],**r})
    baseline=next(r for r in json.loads((ROOT/'reports/cifar-particle-ae/particle_information/results.json').read_text()) if r['name']=='p4096_40k')
    comparison=[baseline,*probes];assert len({r['real_features_sha256'] for r in comparison})==1
    report=ROOT/f'reports/cifar-particle-ae/{track}';write(report/'results.json',probes)
    lines=['# Endpoint information and quality','','Two new probes certified; existing4096 endpoint diagnostic reused. All checkpoints40k.','',
           '| Particles | FID50k | Bits / available | Density | Coverage | Between-sibling variance |','|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        i,q,v=r['information'],r['density_coverage'],r['variance']
        lines.append(f"| {r['num_particles']} | {r['checkpoint_fid50k']:.4f} | {i['observed']['decodable_bits']:.4f} / {i['available_sibling_bits']:.0f} | {q['density']:.4f} | {q['coverage']:.4f} | {100*v['between_siblings_fraction']:.2f}% |")
    lines+=['','Bits are a held-out restricted-decoder estimate, not exact entropy. Review shuffled-label controls/per-parent scores in results.json. Feature distinctions can reflect artifacts; interpret with density/coverage and FID. Same cached10000-image real reference and10000generated samples,k5.']
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    (report/'FINDINGS.md').write_text('# Endpoint interpretation\n\nCompare the information and quality table with the fullFID trajectory. More decodable bits alone do not establish broader real-data coverage. Use the combination to assess whether further duration or count increases are worthwhile; no automatic next training stage.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analyze-only',action='store_true');p.add_argument('--diagnostics-only',action='store_true');a=p.parse_args()
    if not a.analyze_only and not a.diagnostics_only:grid(prepare(),TRACK,TRAINER)
    rows=analyze()
    if not a.analyze_only:diagnostics(rows)
