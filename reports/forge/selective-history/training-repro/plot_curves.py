"""Render retained scheduled metrics. Does not import any training code."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(os.environ.get('PARTICLEGAN_COMBINATION_ARTIFACT_ROOT', Path(__file__).resolve().parent))
rows=json.loads((root/'curve-observations.json').read_text())
colors={'B':'#374151','T':'#2563eb','W':'#c2410c','S':'#0f766e'}
labels={'B':'B · direction baseline','T':'T · plus transport',
        'W':'W · whole history','S':'S · selective history'}
panels=[('two_pole','mean_abs','Pole: travel from origin',.30,'≥ .30',False),
        ('two_pole','grad_med','Pole: median critic slope',1.,'≤ 1',False),
        ('ring16_acquisition','mass_tv','Ring: mass TV',.15,'≤ .15',False),
        ('ring16_acquisition','component_covariance_error','Ring: full component covariance',.85,'≤ .85',True),
        ('ring16_acquisition','hq','Ring: high-quality fraction',.85,'≥ .85',False),
        ('ring16_acquisition','component_min_eigen_ratio','Ring: minimum eigen ratio',.15,'≥ .15',True)]
fig,axes=plt.subplots(2,3,figsize=(13.8,7.4))
for ax,(task,metric,title,threshold,bound,log) in zip(axes.flat,panels):
    for arm in 'BTWS':
        values=[r for r in rows if r['arm']==arm and r['task']==task]
        if not values: continue
        ax.plot([r['step'] for r in values],[r['metrics'][metric] for r in values],
                label=labels[arm],color=colors[arm],linewidth=1.35,alpha=.9)
    ax.axhline(threshold,color='#7c3aed',linewidth=1,linestyle='--')
    ax.set_title(title+'  ('+bound+')',fontsize=10.5,loc='left')
    ax.set_xlabel('Committed training updates',fontsize=9)
    if log:
        if metric=='component_covariance_error':
            assert all(r['metrics'][metric]>0 for r in rows if r['task']==task)
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog',linthresh=.05)
            ax.set_ylim(bottom=0)
    ax.set_xlim(0,80 if task=='two_pole' else 1600)
    ax.grid(alpha=.18)
    ax.tick_params(labelsize=8)
handles,legends=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legends,loc='upper center',bbox_to_anchor=(.5,.947),ncol=4,frameon=False,fontsize=9)
fig.suptitle('Complete scheduled curves · B / T / W / S',x=.06,ha='left',y=.992,fontsize=15,weight='bold')
fig.text(.06,.018,'Source: retained run.log observations. Dashed lines are original bounds. Full gates also require complete schedules and five terminal passes.',fontsize=8,color='#4b5563')
fig.tight_layout(rect=(.035,.05,.995,.91),h_pad=2.0,w_pad=2.0)
fig.savefig(root/'training-curves.png',dpi=180)
fig.savefig(root/'training-curves.svg')
