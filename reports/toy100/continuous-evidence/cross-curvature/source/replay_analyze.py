import json,statistics as st
w=json.load(open('/tmp/cgd/replay-warm/cross_replay.json'))
c=json.load(open('/tmp/cgd/replay-cold/cold_mode_hold.json'))
wr={r['step']:r for r in w['dynamics_receipt']['replays']}; cr={r['step']:r for r in c['replays']}
hq={p['step']:p['hq'] for p in w['diagnostic']}
cobs={o['step']:(o.get('modes'),round(o.get('hq',0),3)) for o in c['observations']} if c.get('observations') else {}
def feats(r):
    g=r['players']['g'];d=r['players']['d'];f=r['functional']
    return dict(alpha=r['alpha'],g_own_over_expl=g['own_curvature']/g['explicit'],g_cross_over_expl=g['cross_response']/g['explicit'],
      g_own_over_cross=g['own_curvature']/max(g['cross_response'],1e-30),d_own_over_cross=d['own_curvature']/max(d['cross_response'],1e-30),
      g_joint_res=g['joint_residual']/max(g['explicit'],1e-30),g_cross_res=g['cross_residual']/max(g['explicit'],1e-30),
      joint_rel=r['joint_relative_residual'],cross_rel=r['cross_relative_residual'],
      g_step_over_expl=g['step']/g['explicit'],
      pmax=f['particle_motion_max'],prms=f['particle_motion_rms'],out_before=f['outside_hq_before'],out_after=f['outside_hq_after'],
      dmax_before=f['nearest_distance_max_before'],approach_min=f['approach_min'],pull_mean=f['critic_radial_pull_mean'],pull_min=f['critic_radial_pull_min'],
      corr=f['center_value_occupancy_correlation'],switches=f['mode_switches'])
def group(rows,label):
    F=[feats(r) for r in rows]; keys=F[0].keys()
    print(f'{label:28s} n={len(F):3d} '+' '.join(f"{k}={st.median([x[k] for x in F if x[k] is not None]):.3g}" for k in keys))
print('per-update warm 1001-1012:')
for s in range(1001,1013):
    x=feats(wr[s]); print(s,'hq',hq.get(s),{k:round(v,3) if isinstance(v,float) else v for k,v in x.items()}, 'occ',wr[s]['functional']['occupancy_before'],'worst',wr[s]['functional']['worst_particle'])
group([wr[s] for s in (1002,)],'warm FAIL update 1002')
group([wr[s] for s in range(1001,1011)],'warm transient 1001-1010')
group([wr[s] for s in range(1101,1201)],'warm late pass 1101-1200')
group([cr[s] for s in range(1,51)],'cold acquisition 1-50')
group([cr[s] for s in range(51,401)],'cold acquisition 51-400')
group([cr[s] for s in range(801,1201)],'cold late 801-1200')
print('cold obs',sorted(cobs.items())[-8:])
print('cold occupancy 400/800/1200',[cr[s]['functional']['occupancy_before'] for s in (400,800,1200)], 'outside',[cr[s]['functional']['outside_hq_before'] for s in (400,800,1200)])
print('warm occupancy 1001/1200',[wr[s]['functional']['occupancy_before'] for s in (1001,1200)])
