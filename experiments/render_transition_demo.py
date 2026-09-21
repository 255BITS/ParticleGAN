#!/usr/bin/env python
"""Build a self-contained, shareable viewer from completed transition artifacts."""
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/transition/demo'
OUT.mkdir(parents=True, exist_ok=True)
models = {}
records = {}
for model in ('encoder_shared_state', 'encoder_separate'):
    path = ROOT/'results/transition/encoder'/model
    summary = json.loads((path/'summary.json').read_text())
    models[model] = dict(final=summary['final'], inference=summary['inference'], parameters=summary['parameters'])
    for split in ('train', 'test'):
        data = np.load(path/f'{split}_samples.npz')
        inf = np.load(path/f'{split}_inference.npz')
        for group in np.unique(data['group']):
            ids = np.flatnonzero(data['group'] == group)
            key = f'{split}_{group}'
            record = records.setdefault(key, dict(split=split, group=int(group), geom=data['geom'][ids[0]].tolist(),
                         scene=int(group)//10, c=int(data['c'][ids[0]]), tick=int(data['tick'][ids[0]]),
                         real=data['real'][ids[:128]].round(6).tolist(), models={}))
            record['models'][model] = dict(prior=data['x'][ids[:128]].round(6).tolist(),
                synthetic=inf['synthetic'][ids[:128]].round(6).tolist(),
                prediction=inf['prediction'][ids[:128]].round(6).tolist())

# PNG: fixed, disclosed case, shown in context and enlarged to reveal one-step errors.
path = ROOT/'results/transition/encoder/encoder_shared_state'
data = np.load(path/'test_samples.npz'); inf = np.load(path/'test_inference.npz')
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=170)
fig.patch.set_facecolor('#f4f7f8')
colors = dict(state='#1978a5', next='#dc831c', error='#cd4b65')
for row in range(2):
    ids = np.flatnonzero((data['group'] < 5) if row == 0 else (data['group'] == 2))
    if row == 0:
        ids = np.concatenate([np.flatnonzero(data['group']==g)[:48] for g in range(5)])
    else:
        ids = ids[:96]
    sources = [data['real'][ids], data['x'][ids], inf['synthetic'][ids]]
    if row == 1:
        upper = [x[x[:,1] > 0] for x in sources]
        points = np.concatenate([np.concatenate([x[:,:2],x[:,:2]+x[:,2:4],x[:,4:]],0) for x in upper],0)
        lo, hi = points.min(0), points.max(0)
        pad = np.maximum((hi-lo)*.08, .01)
        detail_limits = dict(xlim=(lo[0]-pad[0],hi[0]+pad[0]), ylim=(lo[1]-pad[1],hi[1]+pad[1]))
    for col, (ax, x, title) in enumerate(zip(axes[row], sources, ('Real transitions', 'G1 / G2 / G3', 'G1 / G2 → E → G3'))):
        if row == 1:
            x = x[x[:, 1] > 0]
        ax.set_facecolor('white')
        if row == 0:
            ax.add_patch(Circle((0, 0), .27, facecolor='#edf0f3', edgecolor='#aebbc6', lw=1))
        real = data['real'][ids]
        ax.scatter(real[:,0],real[:,1],s=10,c='#adb7c0',alpha=.25,zorder=1)
        end = x[:,:2]+x[:,2:4]
        ax.quiver(x[:,0],x[:,1],x[:,2],x[:,3],angles='xy',scale_units='xy',scale=1,
                  color=colors['state'],alpha=.42,width=.003,zorder=2)
        ax.add_collection(LineCollection(np.stack([end,x[:,4:]],1),colors=colors['error'],alpha=.5,lw=1,zorder=3))
        ax.scatter(x[:,0],x[:,1],s=9,c=colors['state'],alpha=.65,zorder=4)
        ax.scatter(x[:,4],x[:,5],s=12,c=colors['next'],alpha=.75,zorder=5)
        ax.set_aspect('equal'); ax.grid(alpha=.1)
        for spine in ax.spines.values(): spine.set_edgecolor('#d4dde3')
        if row == 0:
            ax.set(xlim=(-1.15,1.15),ylim=(-.85,.85),title=title,xlabel='x',ylabel='y' if col==0 else '')
        else:
            ax.set(**detail_limits,xlabel='x',ylabel='y' if col==0 else '')
            ax.set_title(f'Midpoint, upper branch • residual {np.linalg.norm(x[:,4:]-end,axis=1).mean():.4f}',fontsize=10)
fig.suptitle('Learning one transition at a time',fontsize=24,x=.065,ha='left',y=.98,weight='bold')
fig.text(.065,.935,'Shared-state encoder • held-out interpolation scene 1 • class 0 (80% upper-route preference)',fontsize=12,color='#4b6170')
fig.text(.065,.897,'Top: independent transitions at five saved times. Bottom: midpoint detail; all three panels use identical axes.',fontsize=10,color='#4b6170')
fig.text(.065,.055,'Blue point + arrow: st and at     Orange point: generated st+1     Red line: disagreement with st + at',fontsize=11)
fig.text(.065,.027,'Original joint SW1: 0.08565 (14.6% better than prior leader). Extrapolation coverage: 0.72%. This is a toy result, not solved dynamics.',fontsize=10,color='#4b6170')
fig.subplots_adjust(left=.065,right=.98,top=.865,bottom=.12,wspace=.24,hspace=.27)
fig.savefig(OUT/'toy_transitions.png',facecolor=fig.get_facecolor())
fig.savefig(OUT/'toy_transitions.pdf',facecolor=fig.get_facecolor())
plt.close(fig)
architecture = (ROOT/'reports/transition/architecture/transition_encoder.svg').read_text()
architecture = architecture[architecture.index('<svg'):]
html = (ROOT/'lib/transition_demo.html').read_text().replace('__DATA__',json.dumps(dict(models=models,records=list(records.values())),separators=(',',':'),allow_nan=False)).replace('__ARCHITECTURE__',architecture)
(OUT/'index.html').write_text(html)
for suffix in ('png','svg','pdf'):
    shutil.copy2(ROOT/f'reports/transition/architecture/transition_encoder.{suffix}',OUT/f'architecture.{suffix}')
(OUT/'README.md').write_text('''# One transition at a time

Open index.html in a browser. It is self-contained, works offline, and includes
both encoder models, both classes, all saved train/test scenes and time slices,
real-input prediction, the architecture diagram, and results/limitations.

The controls change independent saved transition samples, not a rollout. Blue
arrows are physical displacement; orange points are generated next states; red
connectors expose consistency error. Plot axes match across comparison panels.
The enlarged view is explicitly labeled. The gray circle shows the reference
obstacle geometry. Gray points optionally show reference states for comparison.

Share index.html alone, or this bundle. No server, external libraries or private
data are required. toy_transitions.png / .pdf show one disclosed interpolation
case; use the viewer to inspect all cases, especially the extrapolation geometry.
architecture.png / .svg / .pdf explain the model. Nothing has been published.

Suggested description:

“We tried learning a distribution of individual state/action/next-state triples
with three generators sharing a 1,024-component MoG prior. An encoder maps
(state, action) back into that latent space, allowing a synthetic G1/G2 → E → G3
path. Sharing a time-conditioned state discriminator improves joint SW1 by 14.6%
on our fixed route toy benchmark. Coverage and off-distribution action response
remain weak; this adds paired supervision and is not yet a general world model.”

Rebuild from repo root:
.venv/bin/python experiments/render_transition_architecture.py
.venv/bin/python experiments/render_transition_demo.py
''')
with zipfile.ZipFile(OUT/'transition_demo.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('index.html','README.md','toy_transitions.png','toy_transitions.pdf','architecture.png','architecture.svg','architecture.pdf'):
        archive.write(OUT/name,arcname=name)
print(OUT/'index.html')
