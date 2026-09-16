"""Render saved memory-GAN trajectories without training or changing metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


def reference_geometry(clean):
    centers, radii, omega = [], [], []
    for path in clean[:, :32].astype(np.float64):
        origin = path.mean(0)
        xy = path-origin
        fit = np.linalg.lstsq(np.column_stack((2*xy, np.ones(len(xy)))),
                             (xy*xy).sum(1), rcond=None)[0]
        center = origin+fit[:2]
        radius = np.sqrt(fit[2]+np.square(fit[:2]).sum())
        first, second = path[:2]-center
        angle = np.arctan2(first[0]*second[1]-first[1]*second[0], first@second)
        centers.append(center), radii.append(radius), omega.append(angle)
    return np.asarray(centers), np.asarray(radii), np.asarray(omega)


def render(run, out, fps):
    summary = json.loads((run/'summary.json').read_text())
    source = run/'trajectories.npz'
    with np.load(source) as data:
        cold, warm = data['generated'].copy(), data['prefix32'].copy()
        observed, reference = data['observed_prefix32'].copy(), data['continuation_reference'].copy()
    centers, radii, omega = reference_geometry(reference)
    # Choose by reference direction only, not generated appearance or quality.
    indices = [int(np.flatnonzero(omega < 0)[0]), int(np.flatnonzero(omega > 0)[0])]
    steps = min(len(cold[0]), len(warm[0]))
    points = np.concatenate((cold[indices].reshape(-1, 2), warm[indices].reshape(-1, 2),
                             reference[indices].reshape(-1, 2)))
    midpoint = (points.max(0)+points.min(0))/2
    span = max(np.ptp(points, axis=0))*1.14
    limits = [(float(midpoint[d]-span/2), float(midpoint[d]+span/2)) for d in (0, 1)]
    bg, fg, muted = '#101827', '#edf2fa', '#aab9ce'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': fg, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100, facecolor=bg)
    fig.subplots_adjust(left=.08, right=.96, bottom=.10, top=.82, hspace=.28, wspace=.12)
    fig.text(.5, .965, 'Motion without a stable circle', ha='center', fontsize=23, weight='bold')
    fig.text(.5, .929, f"{summary['name']}  |  {summary['steps']:,} training updates  |  saved autonomous rollouts",
             ha='center', color=muted, fontsize=11)
    fig.text(.5, .902, 'Lowest long-run radial error among the new scouts; full-circle success: 0 / 128',
             ha='center', color=muted, fontsize=10)
    progress = fig.text(.5, .863, '', ha='center', fontsize=12)
    fig.text(.5, .047, 'Bright line: last 32 steps   |   Faint line: complete history   |   Dot: current generated point',
             ha='center', color=muted, fontsize=10)
    fig.text(.5, .022, 'Warm panels: blue circle / hollow dot = offline reference; gray line = observed prefix. No expert after handoff.',
             ha='center', color=muted, fontsize=10)
    artists = []
    for row in range(2):
        for col, index in enumerate(indices):
            ax = axes[row, col]
            color = '#4de1c1' if row == 0 else '#ffb55c'
            ax.set_facecolor(bg)
            ax.set(xlim=limits[0], ylim=limits[1], aspect='equal')
            ax.grid(alpha=.10, color=muted)
            for spine in ax.spines.values():
                spine.set_color('#344359')
            ax.set_xlabel('x')
            if col == 0:
                ax.set_ylabel('y')
            mode = 'Cold start: M = 0' if row == 0 else f"After 32 real points: reference {'CW' if omega[index] < 0 else 'CCW'}"
            ax.set_title(f'{mode}\nparticle {index}', color=fg, fontsize=11, pad=9)
            ref_dot = None
            if row:
                theta = np.linspace(0, 2*np.pi, 257)
                circle = centers[index]+radii[index]*np.column_stack((np.cos(theta), np.sin(theta)))
                ax.plot(circle[:, 0], circle[:, 1], color='#70aaff', linewidth=1.7, alpha=.95)
                ax.plot(observed[index, :, 0], observed[index, :, 1], color='#c7d2e2', linewidth=1.3, alpha=.75)
                ref_dot, = ax.plot([], [], 'o', color='#70aaff', markersize=8, markerfacecolor=bg, markeredgewidth=1.8)
            history, = ax.plot([], [], color=color, linewidth=.8, alpha=.20)
            trail, = ax.plot([], [], color=color, linewidth=2.1, alpha=.95)
            dot, = ax.plot([], [], 'o', color=color, markersize=6)
            label = ax.text(.025, .035, '', transform=ax.transAxes, color=fg, fontsize=9,
                            bbox=dict(facecolor=bg, edgecolor='none', alpha=.85, pad=4))
            path = cold[index] if row == 0 else warm[index]
            artists.append((path, index, row, history, trail, dot, ref_dot, label))

    def frame(t):
        progress.set_text(f'Generated step {t+1:,} / {steps:,}     |     playback: {fps} steps per second')
        for path, index, row, history, trail, dot, ref_dot, label in artists:
            past = path[:t+1]
            recent = path[max(0, t-31):t+1]
            history.set_data(past[:, 0], past[:, 1])
            trail.set_data(recent[:, 0], recent[:, 1])
            dot.set_data(path[t:t+1, 0], path[t:t+1, 1])
            distance = np.linalg.norm(np.diff(recent, axis=0), axis=1).mean() if len(recent) > 1 else 0.
            text = f'Mean step length (recent): {distance:.3f}'
            if row:
                ref = reference[index, 32+t]
                ref_dot.set_data([ref[0]], [ref[1]])
                radial = abs(np.linalg.norm(path[t]-centers[index])/radii[index]-1)
                text += f'\nCurrent radial error / radius: {radial:.2f}'
            label.set_text(text)

    out.parent.mkdir(parents=True, exist_ok=True)
    available = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True).stdout
    encoder = 'libx264' if 'libx264' in available else 'libopenh264'
    quality = ['-crf', '22'] if encoder == 'libx264' else ['-rc_mode', 'quality', '-q', '23']
    writer = FFMpegWriter(fps=fps, codec=encoder, extra_args=[*quality, '-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    with writer.saving(fig, str(out), dpi=100):
        for t in range(steps):
            frame(t)
            writer.grab_frame(facecolor=bg)
    fig.savefig(out.with_suffix('.png'), dpi=100, facecolor=bg)
    plt.close(fig)
    manifest = {'model': summary['name'], 'steps': steps, 'fps': fps, 'particles': indices,
                'selection': 'first clockwise and first counterclockwise reference episodes; not selected by generated quality',
                'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'reference_use': 'display only; saved autonomous trajectories are unchanged',
                'axis_limits': limits, 'encoder': encoder}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'video': str(out), **manifest}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=24)
    args = parser.parse_args()
    render(args.run, args.out, args.fps)
