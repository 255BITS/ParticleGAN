"""Render the three measured K3P runs; every frame is an original saved checkpoint.

Run with /tmp/pr38-default-env/bin/python. Requires the local source artifacts
listed below. Does not run training, interpolate samples, or use EMA samples.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
RUN = Path('/ml2/hypergan/gan-attempts/gap-fill-20260925/out')
GRID = Path('/ml2/hypergan/gan-attempts/claude-pool-20260925T063704Z/critic_both_arms/20260925T125747Z-3385271/runs/out-k3p-grid100-1234')
PANELS = [
    ('grid100', 'Grid · 0°', GRID),
    ('rotated100', 'Rotated · 25°', RUN/'k3p-native-rotated100'),
    ('staggered100', 'Staggered rows', RUN/'k3p-native-staggered100'),
]
BG, INK, MUTED, BLUE = '#f6f8fc', '#182338', '#64748b', '#1466d9'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

loaded=[]
for task,title,source in PANELS:
    run=source/'native'/task
    result=json.loads((source/'result.json').read_text())
    summary=json.loads((run/'summary.json').read_text())
    assert result['status']=='PASS' and result['steps']==7000 and summary['config']['seed']==1234
    assert result['coverage']['status']=='PASS' and result['accuracy']['status']=='PASS'
    evals={e['step']:e['metrics'] for l in (run/'events.jsonl').read_text().splitlines()
           if (e:=json.loads(l)).get('event')=='eval' and e.get('model')=='live'}
    snapshots={int(p.stem.split('_')[1]):p for p in (run/'snapshots').glob('step_*.npz')}
    loaded.append(dict(task=task,title=title,source=source,run=run,result=result,evals=evals,snapshots=snapshots))
steps=sorted(loaded[0]['snapshots'])
assert all(sorted(p['snapshots'])==steps for p in loaded)
assert steps[0]==0 and steps[-1]==7000 and len(steps)==34
for name in ['config.json','mechanism.py','latent.py','response.py']:
    assert len({p['result']['candidate_hashes'][name] for p in loaded})==1, name

fig=plt.figure(figsize=(12.8,5.9),dpi=100,facecolor=BG)
fig.text(.036,.945,'K3P · learning 100 Gaussians',fontsize=23,fontweight='bold',color=INK)
fig.text(.036,.900,'Three independently trained layouts  ·  Live generator',fontsize=12,color=MUTED)
step_text=fig.text(.964,.943,'',ha='right',fontsize=17,fontweight='bold',color=INK)
fig.patches.append(Rectangle((.036,.860),.928,.006,transform=fig.transFigure,facecolor='#dbe3ef',edgecolor='none'))
progress=Rectangle((.036,.860),0,.006,transform=fig.transFigure,facecolor=BLUE,edgecolor='none')
fig.patches.append(progress)
axes=[]
for i,p in enumerate(loaded):
    ax=fig.add_axes([.036+i*.318,.200,.292,.580],facecolor='white')
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-6.5,6.5);ax.set_ylim(-6.5,6.5)
    ax.set_xticks([-6,-3,0,3,6]);ax.set_yticks([-6,-3,0,3,6])
    ax.tick_params(length=0,labelsize=9,colors=MUTED,pad=6)
    for spine in ax.spines.values():spine.set_color('#dbe3ef')
    ax.set_title(p['title'],loc='left',pad=13,fontsize=16,fontweight='bold',color=INK)
    target=ax.scatter([],[],s=2.4,c='#9aa9bd',alpha=.66,edgecolors='none',zorder=1)
    live=ax.scatter([],[],s=2.4,c=BLUE,alpha=.72,edgecolors='none',zorder=2)
    metric=fig.text(.036+i*.318,.146,'',fontsize=12,fontweight='medium',color=INK)
    axes.append((target,live,metric))
fig.legend(handles=[Line2D([],[],color='#9aa9bd',marker='o',linestyle='none',markersize=6,label='Target samples'),
                    Line2D([],[],color=BLUE,marker='o',linestyle='none',markersize=6,label='Generated samples')],
           loc='lower left',bbox_to_anchor=(.027,.037),ncol=2,frameon=False,fontsize=11,labelcolor=MUTED)
fig.text(.964,.071,'Seed 1234  ·  4,096 points per layer',ha='right',fontsize=10,color=MUTED)
fig.text(.964,.043,'Metrics: 20,000 samples  ·  Checkpoint playback, not real time',ha='right',fontsize=9,color=MUTED)

frames=[]
frame_records=[]
preview_dir=Path('/ml2/hypergan/gan-attempts/gap-fill-20260925/gif-preview')
preview_dir.mkdir(exist_ok=True)
for step in steps:
    step_text.set_text(f'Step {step:,} / 7,000')
    progress.set_width(.928*step/7000)
    record=dict(step=step,panels=[])
    for p,(target,live,metric) in zip(loaded,axes):
        path=p['snapshots'][step]
        with np.load(path) as data:
            assert data['live'].shape==data['target'].shape==(4096,2)
            assert np.isfinite(data['live']).all() and np.isfinite(data['target']).all()
            target.set_offsets(data['target']);live.set_offsets(data['live'])
        m=p['evals'][step]
        assert m['n']==20000
        metric.set_text(f"Modes  {m['modes']:3d}/100     Precision  {m['precision']:.1%}")
        record['panels'].append(dict(task=p['task'],snapshot=str(path),sha256=sha(path),live_metrics=m))
    fig.canvas.draw()
    frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    frames.append(frame);frame_records.append(record)
    if step in (0,750,7000):frame.save(preview_dir/f'step-{step}.png')
plt.close(fig)
# A common palette keeps target/sample/text colors consistent throughout playback.
palette=Image.new('RGB',(1280,590*3))
for i,index in enumerate([0,steps.index(750),len(frames)-1]):palette.paste(frames[index],(0,590*i))
palette=palette.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
indexed=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
durations=[350]*len(frames);durations[0]=900;durations[-1]=2500
gif=OUT/'k3p-100gaussians-convergence.gif'
indexed[0].save(gif,save_all=True,append_images=indexed[1:],duration=durations,loop=0,optimize=True,disposal=2)
Image.open(preview_dir/'step-7000.png').save(OUT/'k3p-100gaussians-final.png')
manifest=dict(description='Original live-generator snapshots from three independently trained K3P layouts, seed1234. No sample interpolation or retraining; all34 saved checkpoints shown. Fixed equal axes across frames/panels. Metrics use the20000-sample live evaluation, plot uses4096-sample snapshots.',
              renderer_sha256=sha(Path(__file__)),gif_sha256=sha(gif),size_bytes=gif.stat().st_size,dimensions=[1280,590],duration_ms=sum(durations),frame_durations_ms=durations,
              candidate_hashes=loaded[0]['result']['candidate_hashes'],
              source_results=[dict(task=p['task'],result=str(p['source']/'result.json'),sha256=sha(p['source']/'result.json'),events=str(p['run']/'events.jsonl'),events_sha256=sha(p['run']/'events.jsonl'),full_gate=p['result']['status']) for p in loaded],frames=frame_records)
(OUT/'k3p-convergence-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
with Image.open(gif) as im:
    assert im.n_frames==34 and im.size==(1280,590)
print(json.dumps(dict(gif=str(gif),frames=len(frames),bytes=gif.stat().st_size,duration_seconds=sum(durations)/1000)))
