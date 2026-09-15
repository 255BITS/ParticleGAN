"""Standalone trajectory figures, animation, and an interactive HTML viewer."""
import json

import numpy as np


def render(out, samples, final, probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, PillowWriter

    colors = ["#dc6951", "#277fb5"]

    def scene(ax, geom):
        start, oy, radius = geom
        ax.add_patch(Circle((0, oy), radius, color="#374151", alpha=.85))
        ax.plot([-1.35, -1], [start, start], color="black", lw=3)
        ax.scatter([1], [0], marker="*", c="black", s=45)
        ax.set(xlim=(-1.45, 1.3), ylim=(-1, 1), aspect="equal")
        ax.grid(alpha=.15)

    for split, data in samples.items():
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for k, ax in enumerate(axes.flat):
            mask = data["group"] == k
            geom = data["geom"][mask][0]
            scene(ax, geom)
            for x in data["real"][mask][:24]:
                ax.plot(*x, color="#a7b1b7", lw=.7, alpha=.5)
            for x in data["x"][mask][:48]:
                route = int(x[1, len(x[1])//2] > geom[1])
                ax.plot(*x, color=colors[route], lw=.8, alpha=.5)
            row = final[split]["contexts"][k]
            ax.set_title(f"Scene {k//2+1}, preference {k%2}\nupper {row['upper_observed']:.0%} / target {row['upper_target']:.0%}; valid {row['valid']:.0%}", fontsize=10)
        fig.suptitle(f"{split.capitalize()} geometries: colored = generated; gray = real; black = observed approach")
        fig.tight_layout()
        fig.savefig(out / f"{split}_routes.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, ax in zip(("train", "test"), axes):
        rows = final[split]["contexts"]
        i = np.arange(len(rows))
        ax.bar(i-.18, [r["upper_target"] for r in rows], .36, label="Target")
        ax.bar(i+.18, [r["upper_observed"] for r in rows], .36, label="Generated")
        ax.set(title=f"{split}: upper-route probability (all samples)", ylim=(0, 1), xlabel="Context")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "route_mass.png", dpi=140)
    plt.close(fig)

    if probes is not None:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for k, ax in enumerate(axes):
            scene(ax, probes["geom"][0])
            xx = probes["noise"][k*24:(k+1)*24] if k < 4 else probes["particles"][:24]
            for x in xx:
                ax.plot(*x, alpha=.5, lw=.8)
            ax.set_title(f"Particle {k} at every step\n24 noise draws" if k < 4 else "Fixed diffusion randomness\n24 particle sequences", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "particle_probe.png", dpi=140)
        plt.close(fig)

    data = samples["test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    artists = []
    for k, ax in enumerate(axes):
        mask = data["group"] == k
        scene(ax, data["geom"][mask][0])
        ax.set_title(f"Held-out scene, preference {k}: upper target {[.8, .3][k]:.0%}")
        for x in data["x"][mask][:20]:
            line, = ax.plot([], [], lw=1, alpha=.65)
            dot, = ax.plot([], [], "o", ms=3, color=line.get_color())
            artists.append((line, dot, x))
    def update(frame):
        for line, dot, x in artists:
            line.set_data(x[:, :frame+1])
            dot.set_data(x[0, frame:frame+1], x[1, frame:frame+1])
        return [a for line, dot, _ in artists for a in (line, dot)]
    fig.tight_layout()
    FuncAnimation(fig, update, frames=data["x"].shape[-1], interval=60, blit=True).save(out / "futures.gif", writer=PillowWriter(fps=16))
    plt.close(fig)

    payload = {}
    for split, data in samples.items():
        payload[split] = []
        for k in range(8):
            mask = data["group"] == k
            payload[split].append(dict(geom=data["geom"][mask][0].tolist(),
                                      x=data["x"][mask][:64].round(5).tolist(),
                                      real=data["real"][mask][:32].round(5).tolist(),
                                      metrics=final[split]["contexts"][k]))
    html = '''<!doctype html><meta charset="utf-8"><title>Possible futures</title>
<style>body{font:16px system-ui;background:#f5f7fa;color:#172635;max-width:1050px;margin:30px auto}canvas{background:white;border-radius:12px;width:100%}select,button,input{margin:8px;padding:7px}p{line-height:1.5}.stats{font-variant-numeric:tabular-nums}</style>
<h1>Possible futures</h1><p>Each colored line is one generated future from the same observed approach. Gray lines are real samples. The dark circle is an obstacle; the star is the destination.</p>
<label>Geometry <select id="split"><option value="test">Held out</option><option value="train">Training</option></select></label>
<label>Scene <select id="scene"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
<label>Preference <select id="pref"><option value="0">80% upper</option><option value="1">30% upper</option></select></label>
<button id="play">Play / pause</button><label>Time <input id="frame" type="range" min="0" max="63" value="63"></label>
<canvas id="canvas" width="1000" height="650"></canvas><p id="stats" class="stats"></p>
<p>Validity requires no obstacle intersection, start/end error below 0.1, and distance to the known family of paths below 0.05 RMS. Route frequency alone does not establish validity or diversity. The two preference labels change route probabilities; they do not reveal which route to generate.</p>
<script>const DATA=__DATA__;const $=s=>document.getElementById(s),ctx=$('canvas').getContext('2d');let playing=false;
const X=x=>70+(x+1.45)*310,Y=y=>325-y*285;
function line(x,color,n,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<n;i++){const xx=X(x[0][i]),yy=Y(x[1][i]);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)}ctx.stroke()}
function draw(){const k=2*(Number($('scene').value)-1)+Number($('pref').value),d=DATA[$('split').value][k],n=Number($('frame').value)+1;ctx.clearRect(0,0,1000,650);ctx.fillStyle='#374151';ctx.beginPath();ctx.ellipse(X(0),Y(d.geom[1]),d.geom[2]*310,d.geom[2]*285,0,0,Math.PI*2);ctx.fill();line([[-1.35,-1],[d.geom[0],d.geom[0]]],'#172635',2,5);ctx.font='28px system-ui';ctx.fillText('★',X(1)-14,Y(0)+10);d.real.forEach(x=>line(x,'#aab4bd60',x[0].length));d.x.forEach(x=>{line(x,x[1][32]>d.geom[1]?'#277fb580':'#dc695180',n,1.3)});const m=d.metrics;$('stats').textContent=`Upper route: ${(100*m.upper_observed).toFixed(1)}% (target ${100*m.upper_target}%) · Valid paths: ${(100*m.valid).toFixed(1)}% · Routes covered: ${m.routes_covered}/2`;}
['split','scene','pref','frame'].forEach(id=>$(id).oninput=draw);$('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){$('frame').value=(Number($('frame').value)+1)%64;draw()}},60);draw();</script>'''
    html = html.replace('max="63"', f'max="{samples["test"]["x"].shape[-1]-1}"')
    html = html.replace('value="63"', f'value="{samples["test"]["x"].shape[-1]-1}"')
    html = html.replace('x[1][32]', f'x[1][{samples["test"]["x"].shape[-1]//2}]')
    html = html.replace('%64', f'%{samples["test"]["x"].shape[-1]}')
    (out / "viewer.html").write_text(html.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
