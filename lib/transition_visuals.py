"""Portable transition scatter/arrow plots and a standalone canvas viewer."""
import json
import hashlib
from pathlib import Path

import numpy as np


def render(out, samples, length, architecture="branches", critic_mode="joint"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = samples["test"]
    group_ids = np.unique(test["group"])
    middle = group_ids[len(group_ids)//8//2]  # first geometry/class, middle time
    mask = test["group"] == middle
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title in zip(axes[:2], ("real", "x"), ("Reference", "Generated")):
        x = test[key][mask][:128]
        ax.quiver(x[:, 0], x[:, 1], x[:, 2], x[:, 3], angles="xy", scale_units="xy", scale=1, alpha=.4)
        ax.scatter(x[:, 4], x[:, 5], s=8, color="tab:orange", label="st+1")
        ax.set(xlabel="x", ylabel="y", title=title, xlim=(-1.2, 1.2), ylim=(-1, 1))
        ax.set_aspect("equal")
        ax.legend()
    x = test["x"]
    r = np.linalg.norm(x[:, 4:]-x[:, :2]-x[:, 2:4], axis=1)
    axes[2].hist(r, bins=60, alpha=.7, color="tab:orange", label="Generated", density=True)
    real = test["real"]
    reference_error = np.linalg.norm(real[:, 4:]-real[:, :2]-real[:, 2:4], axis=1)
    axes[2].axvline(reference_error.mean(), color="tab:blue", label="Reference mean")
    axes[2].set_xlim(0, max(float(r.max())*1.05, 1e-6))
    axes[2].set(xlabel="||st+1 - st - at||", ylabel="Density", title="All test contexts")
    axes[2].legend()
    fig.savefig(out / "transitions.png", dpi=160)
    plt.close(fig)

    records = []
    for split, data in samples.items():
        for group in np.unique(data["group"]):
            ids = np.flatnonzero(data["group"] == group)
            i = ids[0]
            records.append(dict(label=f"{split} geometry {data['geom'][i].round(3).tolist()}, class {int(data['c'][i])}, t={int(data['tick'][i])}/{length-1}",
                                geom=data["geom"][i].tolist(),
                                real=data["real"][ids[:128]].tolist(), fake=data["x"][ids[:128]].tolist()))
    html = """<!doctype html><meta charset="utf-8"><title>Transition GAN</title>
<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#fafafa;color:#222}
canvas{width:100%;background:white;border:1px solid #ddd}select{padding:8px}p{line-height:1.5}</style>
<h1>One transition, three generators</h1>
<p>G1 → st &nbsp; G2 → at &nbsp; G3 → st+1. Shared latent and scene/time context; one D(st, at, st+1).</p>
<select id="scene"></select><p>Left: reference. Right: generated. Blue arrows: st → st + at.
Orange points: predicted st+1. Red lines: disagreement with st + at.</p>
<canvas id="plot" width="1200" height="500"></canvas><p id="stats"></p>
<script>
const records=RECORDS, menu=document.getElementById('scene'), canvas=document.getElementById('plot'), ctx=canvas.getContext('2d');
records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.label;menu.appendChild(o)});
function draw(){const r=records[menu.value];ctx.clearRect(0,0,1200,500);
function panel(points,offset,title){const px=x=>offset+300+220*x,py=y=>250-220*y;
ctx.fillStyle='#222';ctx.font='20px system-ui';ctx.fillText(title,offset+25,30);
ctx.strokeStyle='#999';ctx.beginPath();ctx.arc(px(0),py(r.geom[1]),220*r.geom[2],0,2*Math.PI);ctx.stroke();
for(const [sx,sy,ax,ay,nx,ny] of points){const ex=sx+ax,ey=sy+ay;
ctx.globalAlpha=.4;ctx.strokeStyle='#2875bb';ctx.beginPath();ctx.moveTo(px(sx),py(sy));ctx.lineTo(px(ex),py(ey));ctx.stroke();
const angle=Math.atan2(-ay,ax);ctx.beginPath();ctx.moveTo(px(ex)-4*Math.cos(angle-.5),py(ey)-4*Math.sin(angle-.5));
ctx.lineTo(px(ex),py(ey));ctx.lineTo(px(ex)-4*Math.cos(angle+.5),py(ey)-4*Math.sin(angle+.5));ctx.stroke();
ctx.strokeStyle='#c44';ctx.beginPath();ctx.moveTo(px(ex),py(ey));ctx.lineTo(px(nx),py(ny));ctx.stroke();
ctx.fillStyle='#ed922a';ctx.beginPath();ctx.arc(px(nx),py(ny),2,0,2*Math.PI);ctx.fill();}ctx.globalAlpha=1;}
panel(r.real,0,'Reference');panel(r.fake,600,'Generated');
const err=r.fake.map(x=>Math.hypot(x[4]-x[0]-x[2],x[5]-x[1]-x[3]));
document.getElementById('stats').textContent='Mean disagreement in displayed generated samples: '+(err.reduce((a,b)=>a+b,0)/err.length).toFixed(5);}
menu.onchange=draw;menu.value=records.findIndex(r=>r.label.startsWith('test'));draw();
</script>"""
    if architecture == "monolithic":
        html = html.replace("One transition, three generators", "One transition, one generator")
        html = html.replace("G1 → st &nbsp; G2 → at &nbsp; G3 → st+1. Shared latent and scene/time context; one D(st, at, st+1).",
                            "G → (st, at, st+1). Particle latent and scene/time context; one D(st, at, st+1).")
    if critic_mode == "joint_marginals":
        html = html.replace("one D(st, at, st+1).", "joint D(st, at, st+1) plus conditional state, action and next-state marginal feedback.")
    (out / "viewer.html").write_text(html.replace("RECORDS", json.dumps(records, allow_nan=False)))
    (out / "render_provenance.json").write_text(json.dumps(dict(
        renderer="lib/transition_visuals.py", sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        architecture=architecture, critic_mode=critic_mode, length=length), indent=2)+"\n")
