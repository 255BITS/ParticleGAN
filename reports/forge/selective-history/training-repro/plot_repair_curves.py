"""Render matched recipe families from retained observations, without training."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(os.environ.get('PARTICLEGAN_COMBINATION_ARTIFACT_ROOT', Path(__file__).resolve().parent))
rows=json.loads((root/'curve-observations.json').read_text())
families=[('Transport family', [('B','Baseline','#4b5563'),('T','Transport','#2563eb'),
                              ('TP','Transport + penalty 2','#7c3aed'),('TD','Transport + half critic','#c2410c')]),
          ('Selective-history family',[('B','Baseline','#4b5563'),('S','Selective history','#0f766e'),
                                       ('SH','Half generator side','#be185d'),('SHD','Half all players','#d97706')])]
panels=[('two_pole','grad_med','Pole: median critic slope',1.,False),
        ('ring16_acquisition','component_covariance_error','Ring: full component covariance',.85,True),
        ('ring16_acquisition','hq','Ring: high-quality fraction',.85,False)]
fig,axes=plt.subplots(3,2,figsize=(13.4,10.5))
for col,(family,arms) in enumerate(families):
    for row,(task,metric,title,bound,log) in enumerate(panels):
        ax=axes[row,col]
        for arm,label,color in arms:
            points=[r for r in rows if r['arm']==arm and r['task']==task]
            if not points:continue
            ax.plot([r['step'] for r in points],[r['metrics'][metric] for r in points],
                    label=label,color=color,linewidth=1.2,alpha=.95)
        ax.axhline(bound,color='#111827',linestyle='--',linewidth=.9)
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_xlabel('Committed training updates',fontsize=9)
        if log:
            assert all(r['metrics'][metric]>0 for r in rows if r['task']==task)
            ax.set_yscale('log')
        else:ax.set_ylim(bottom=0)
        if metric=='hq':ax.set_ylim(0,1.01)
        ax.set_xlim(0,80 if task=='two_pole' else 1600)
        ax.grid(alpha=.18);ax.tick_params(labelsize=8)
    axes[0,col].legend(loc='upper left',fontsize=8,frameon=True,
                       facecolor='white',edgecolor='none',framealpha=.94)
    axes[0,col].text(0,1.18,family,transform=axes[0,col].transAxes,fontsize=13,weight='bold')
fig.suptitle('Bounded repair comparisons · retained scheduled curves',x=.055,ha='left',y=.995,fontsize=15,weight='bold')
fig.text(.055,.015,'Every curve uses retained scored observations. Dashed lines are original bounds; all gates and five terminal passes remain required.',fontsize=8,color='#4b5563')
fig.tight_layout(rect=(.03,.045,.995,.96),h_pad=2.2,w_pad=2)
fig.savefig(root/'repair-training-curves.png',dpi=180)
fig.savefig(root/'repair-training-curves.svg')
