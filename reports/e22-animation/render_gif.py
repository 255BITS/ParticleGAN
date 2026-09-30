"""Render the E22 native100 convergence GIF from record_e22.py outputs (three panels, shared palette).

usage: python reports/e22-animation/render_gif.py e22-100gaussians.gif [--dir .] [--fps 25] [--hold 2.5]
Per frame and task: modes = centres with >= 10 of the 4,096 plotted points as their nearest centre within
3 sigma; hq = share of points within 3 sigma (.09) of a centre (display numbers from the plotted points,
not the harness gates).
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TASKS = ('grid100', 'rotated100', 'staggered100')
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--fps', type=int, default=25); ap.add_argument('--hold', type=float, default=2.5)
ap.add_argument('--colors', type=int, default=64); ap.add_argument('--width', type=int, default=380)
ap.add_argument('--dir', default='.', help='directory holding frames-<task>.npz')
args = ap.parse_args()
assert 100 % args.fps == 0, 'GIF durations use 10 ms units'

data = {t: np.load(f'{args.dir}/frames-{t}.npz') for t in TASKS}
steps = data[TASKS[0]]['steps']
assert all(np.array_equal(data[t]['steps'], steps) for t in TASKS)
W, H = args.width * 3, args.width + 40
fig, axes = plt.subplots(1, 3, figsize=(W / 100, H / 100), facecolor='white')
scatters, titles = {}, {}
for ax, task in zip(axes, TASKS):
    centers = data[task]['centers']
    extent = max(5.3, float(np.abs(centers).max()) + .6)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal', xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_color('#e2e8f0')
    ax.scatter(centers[:, 0], centers[:, 1], s=22, facecolors='none', edgecolors='#94a3b8', linewidths=.5)
    scatters[task] = ax.scatter(np.zeros(1), np.zeros(1), s=.9, alpha=.55, color='#087f8c', rasterized=True)
    titles[task] = ax.set_title('', fontsize=8.5, color='#334155')
fig.suptitle('ParticleGAN · E22 on the native 100-Gaussian problems (no LR schedule)', fontsize=11, color='#0f172a', y=.975)
footer = fig.text(.5, .02, '', ha='center', fontsize=7.5, color='#64748b')
fig.tight_layout(rect=(0, .04, 1, .95), w_pad=.6)


def stats(points, centers):
    d = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    nearest, dist = d.argmin(1), d.min(1)
    hq = dist <= .09
    return int((np.bincount(nearest[hq], minlength=len(centers)) >= 10).sum()), float(hq.mean())


frames = []
for i, step in enumerate(steps):
    for task in TASKS:
        pts = data[task]['frames'][i].astype(np.float32)
        modes, hq = stats(pts, data[task]['centers'])
        scatters[task].set_offsets(pts)
        titles[task].set_text(f'{task} · {modes}/100 modes · {hq:.1%} within 3σ')
    footer.set_text(f'update {int(step):,} / {int(steps[-1]):,} · configs/100gaussians/e22-noout.json · '
                    'served model, seed 1234 · 1 frame = 25 updates')
    fig.canvas.draw()
    frames.append(Image.frombuffer('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB'))
plt.close(fig)

picks = frames[::max(1, len(frames) // 24)]
strip = Image.new('RGB', (frames[0].width, frames[0].height * len(picks)))
for k, frame in enumerate(picks):
    strip.paste(frame, (0, frames[0].height * k))
palette = strip.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT)
indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
frame_ms = 1000 // args.fps
hold_ms = max(round(args.hold * 100) * 10, frame_ms)
indexed[0].save(args.out, save_all=True, append_images=indexed[1:], duration=[frame_ms] * (len(indexed) - 1) + [hold_ms],
                loop=0, optimize=True)
print(json.dumps({'frames': len(frames), 'size': frames[0].size, 'seconds': round((len(frames) - 1) * frame_ms / 1000 + hold_ms / 1000, 1),
                  'final': {t: json.loads(str(data[t]['final'])) for t in TASKS}}))
