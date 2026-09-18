#!/usr/bin/env python
"""Certify read-only feature information, density/coverage and variance probes."""
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
TRAINER=ROOT/'experiments/probe_cifar_ae_information.py'
POINTS={
 'p1024_20k':'particle_expansion_scout/control_1024/checkpoint_020000.pt',
 'p4096_20k':'particle_expansion_scout/split_4096/checkpoint_020000.pt',
 'p8192_20k':'particle_scaling_scout/split_8192/checkpoint_020000.pt',
 'p16384_20k':'particle_scaling_scout/split_16384/checkpoint_020000.pt',
 'p1024_40k':'particle_expansion_40k/control_1024/checkpoint_040000.pt',
 'p4096_35k':'particle_expansion_40k/split_4096/checkpoint_035000.pt',
 'p4096_40k':'particle_expansion_40k/split_4096/checkpoint_040000.pt',
}


def prepare(track,smoke):
    folder=ROOT/f'configs/cifar_particle_ae/{track}';folder.mkdir(parents=True,exist_ok=True);paths=[]
    points={'p4096_20k':POINTS['p4096_20k']} if smoke else POINTS
    for name,relative in points.items():
        checkpoint=ROOT/'runs/cifar_particle_ae'/relative
        assert checkpoint.exists(),checkpoint
        cfg={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'out_dir':f'runs/cifar_particle_ae/{track}/{name}'}
        if smoke:cfg.update(samples=128,batch_size=64,parents=2,train_draws=4,val_draws=4,test_draws=4,variance_draws=4,k=3,distance_chunk=64)
        path=folder/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg));paths.append(str(path.relative_to(ROOT)))
    manifest=folder/'manifest.json';manifest.write_text(json.dumps(paths,indent=2)+'\n');return manifest


def analyze(track,smoke):
    defaults=trainer_defaults(str(TRAINER));provenance=code_provenance(str(TRAINER),sys.executable)
    paths=json.loads((ROOT/f'configs/cifar_particle_ae/{track}/manifest.json').read_text());rows=[]
    for path in paths:
        cfg=load_config(path,defaults);out=ROOT/cfg['out_dir']
        assert has_valid_summary(str(out),cfg,provenance),out
        result=json.loads((out/'summary.json').read_text())
        assert result['parent_unchanged'] and result['frozen_state_unchanged'],out
        parent_folder=(ROOT/cfg['checkpoint']).parent
        parent_summary=json.loads((parent_folder/'summary.json').read_text())
        parent_step=int(Path(cfg['checkpoint']).stem.split('_')[-1])
        parent_metrics=[json.loads(x) for x in (parent_folder/'metrics.jsonl').read_text().splitlines()]
        point=next(m for m in parent_metrics if m['step']==parent_step and 'generation' in m)
        assert point['generation']['samples']==50000
        rows.append({'name':out.name,'checkpoint_fid50k':point['generation']['fid'],
                     'checkpoint_step':parent_step,'particle_count':parent_summary['config']['num_particles']*parent_summary['config']['expansion_factor'],**result})
    report=ROOT/f'reports/cifar-particle-ae/{track}';report.mkdir(parents=True,exist_ok=True)
    (report/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert len({json.dumps(r['real_reference'],sort_keys=True) for r in rows})==1, 'real reference differs'
    assert len({r['real_features_sha256'] for r in rows})==1, 'actual cached reference features differ'
    for r in rows:
        assert r['parent_step']==r['checkpoint_step'] and r['num_particles']==r['particle_count']
        q=r['density_coverage']; info=r['information']; variance=r['variance']
        assert q['real_samples']==q['fake_samples']==r['config']['samples']
        assert 0<=q['coverage']<=1 and q['density']>=0
        assert abs(info['identical_clones']['decodable_bits'])<1e-8
        assert info['observed']['decodable_bits'] <= info['available_sibling_bits']+1e-8
        fraction_sum=sum(variance[k] for k in ('within_child_fraction','between_siblings_fraction','between_parents_fraction'))
        assert abs(fraction_sum-(1 if variance['total_sum_squares']>0 else 0))<1e-8
    rows.sort(key=lambda r:(r['checkpoint_step'],r['particle_count']))
    lines=['# Particle information diagnostics','',f'{len(rows)}/{len(paths)} certified; frozen checkpoints unchanged.','',
           'Fixed Inception features, equal real/generated budgets, same cached real reference. Existing checkpoint FID50k is shown separately; this probe does not recompute FID.','',
           '| Checkpoint | FID50k | Density | Coverage | Feature variance / real | Probe minutes |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        q=r['density_coverage']
        lines.append(f"| {r['name']} | {r['checkpoint_fid50k']:.4f} | {q['density']:.4f} | {q['coverage']:.4f} | {q['feature_variance_trace_ratio_to_real']:.4f} | {r['total_seconds']/60:.2f} |")
    lines += ['', '| Checkpoint | Decodable bits | Available bits | Shuffled-label bits | Between-sibling variance | Within-child variance |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        i,v=r['information'],r['variance']
        lines.append(f"| {r['name']} | {i['observed']['decodable_bits']:.4f} | {i['available_sibling_bits']:.0f} | {i['shuffled_labels']['decodable_bits']:.4f} | {100*v['between_siblings_fraction']:.2f}% | {100*v['within_child_fraction']:.2f}% |")
    lines += ['', 'Bits are a held-out, restricted-decoder estimate of a conditional mutual-information lower bound, not exact entropy or semantic coverage. Negative estimates are retained. The synthetic identical-clone control gives zero additional bits. Shuffled labels measure the null behavior with independent train/validation/test assignments.', '',
              'Density need not be bounded by1 and higher density alone does not demonstrate quality. Coverage depends on extractor, neighborhood k and sample counts. ANOVA fractions are finite-sample descriptive measurements; estimated child means include sampling noise. More distinguishable features can reflect artifacts.', '',
              'The compression hypothesis predicts additional decodable information accompanied by broader real-feature coverage without deterioration in fidelity. These metrics diagnose that pattern; FID50k remains the benchmark target.']
    if smoke:lines.insert(2,'SMOKE ONLY: reduced sample counts; no benchmark interpretation.')
    (report/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    if not smoke:
        findings=['# Feature information and coverage findings','',f'{len(rows)} certified read-only checkpoints. See LEADERBOARD.md for all measurements.','']
        baseline=next(r for r in rows if r['name']=='p4096_20k')
        for r in rows:
            if r['checkpoint_step']==20000 and r['particle_count']>4096:
                b=r['information']['observed']['decodable_bits']-baseline['information']['observed']['decodable_bits']
                d=r['density_coverage']['density']-baseline['density_coverage']['density']
                c=r['density_coverage']['coverage']-baseline['density_coverage']['coverage']
                findings.append(f"{r['name']} versus4096 at20k: FID {r['checkpoint_fid50k']-baseline['checkpoint_fid50k']:+.4f}, decodable bits {b:+.4f}, density {d:+.4f}, coverage {c:+.4f}. Different sibling counts have different available-bit budgets; see the table.")
                findings.append('')
        a=next(r for r in rows if r['name']=='p4096_35k');b=next(r for r in rows if r['name']=='p4096_40k')
        findings += [f"4096 from35k to40k: FID {b['checkpoint_fid50k']-a['checkpoint_fid50k']:+.4f}; density {b['density_coverage']['density']-a['density_coverage']['density']:+.4f}; coverage {b['density_coverage']['coverage']-a['density_coverage']['coverage']:+.4f}; decodable bits {b['information']['observed']['decodable_bits']-a['information']['observed']['decodable_bits']:+.4f}.", '',
                     'Recommendation: interpret particle bits jointly with density/coverage and the FID trajectory. Increasing bits alone is insufficient evidence of useful diversity; weak bits can also reflect a limited decoder. Do not promote a checkpoint based solely on these diagnostics. Keep the next training choice focused on particle scaling as requested.', '',
                     'Method references: InfoGAN (https://arxiv.org/abs/1606.03657); density/coverage (https://proceedings.mlr.press/v119/naeem20a.html). No checkpoint or training objective was changed.']
        (report/'FINDINGS.md').write_text('\n'.join(findings)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--analyze-only',action='store_true');p.add_argument('--gpus',default='0,1');a=p.parse_args()
    track='particle_information_smoke' if a.smoke else 'particle_information'
    if not a.analyze_only:
        manifest=prepare(track,a.smoke);run=ROOT/f'runs/cifar_particle_ae/{track}'
        subprocess.run([sys.executable,'-u','experiments/follow_grid.py','--root',str(run),'--log',str(run/'PIPELINE.log'),'--',
                        '--config_manifest',str(manifest),'--gpus',a.gpus,'--workers_per_gpu','1','--python',sys.executable,'--trainer',str(TRAINER)],cwd=ROOT,check=True)
    analyze(track,a.smoke)
