"""Render four saved observations; never sample a model or interpolate points."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import PIL


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def package_sha(path):
    h = hashlib.sha256()
    for source in sorted((path / 'particlegan').rglob('*.py')):
        h.update(str(source.relative_to(path / 'particlegan')).encode() + b'\0' + source.read_bytes() + b'\0')
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    study, out = args.study_root.resolve(), args.output_dir.resolve()
    assert out.is_dir()
    outputs = ['rotated100-shift-comparison.gif', 'rotated100-shift-comparison.png',
               'moving-score-comparison.json', 'moving-score-comparison.md', 'CAPTION.md',
               'README.md', 'GIF-RECEIPT.json']
    assert not any((out / name).exists() for name in outputs), 'retain every prior artifact'
    lanes = [('RA14', 'validation-ra14-r2'), ('RA15', 'validation-ra15')]
    original = Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py')
    source_inputs = [original, Path(__file__).resolve()]
    verdicts, comparison = {}, []
    for label, lane in lanes:
        verdicts[label] = {}
        for task in ('grid100', 'rotated100', 'staggered100'):
            directory = study / lane / 'moving' / task
            vp, lp = directory / 'frames.npz.verdict.json', directory / 'LAUNCH.json'
            verdict = json.loads(vp.read_text())
            launch = json.loads(lp.read_text())
            assert [p['period_end'] for p in verdict['periods']] == [500, 1000, 1500]
            assert [p['target_deg'] for p in verdict['periods']] == [0, 30, 60]
            assert launch['protocol'] == dict(degrees=30, draw=20000, seed=1234,
                                             steps=1500, turn_every=500, turns=2)
            verdicts[label][task] = verdict
            source_inputs.extend([vp, lp, directory / 'COMPLETION.json', directory / 'adapted_runner.py'])
    for task in ('grid100', 'rotated100', 'staggered100'):
        old, new = verdicts['RA14'][task], verdicts['RA15'][task]
        assert old['pre_turn_hq'] == new['pre_turn_hq']
        threshold = .9 * old['pre_turn_hq']
        for i in (1, 2):
            before, after = old['periods'][i], new['periods'][i]
            passed = lambda row: row['modes'] >= 95 and row['hq'] >= threshold
            comparison.append(dict(task=task, step=before['period_end'], added_target_rotation_deg=before['target_deg'],
                step_500_baseline_hq=old['pre_turn_hq'], minimum_hq=threshold, minimum_modes=95,
                RA14=dict(hq=before['hq'], modes=before['modes'], passed=passed(before)),
                RA15=dict(hq=after['hq'], modes=after['modes'], passed=passed(after)),
                hq_change_percentage_points=round(100 * (after['hq'] - before['hq']), 4)))
    assert sum(x['RA14']['passed'] for x in comparison) == 5
    assert all(x['RA15']['passed'] for x in comparison)
    arrays = []
    for label, lane in lanes:
        path = study / lane / 'moving/rotated100/frames.npz'
        source_inputs.append(path)
        with np.load(path, allow_pickle=False) as data:
            arrays.append({key: data[key].copy() for key in ('frames', 'steps', 'centers', 'angles')})
    for data in arrays:
        assert data['frames'].shape == (4, 4096, 2) and data['frames'].dtype == np.float16
        assert np.array_equal(data['steps'], [0, 500, 1000, 1500])
        assert np.allclose(np.degrees(data['angles']), [0., 0., 30., 60.], rtol=0, atol=1e-12)
        assert np.isfinite(data['frames']).all() and np.isfinite(data['centers']).all()
    assert np.array_equal(arrays[0]['centers'], arrays[1]['centers'])
    assert np.array_equal(arrays[0]['frames'][:2], arrays[1]['frames'][:2])
    source_inputs.extend([study / 'configs/RA14-replay.json', study / 'configs/RA15-partial-recovery.json'])
    input_hashes = {str(p): sha(p) for p in sorted(set(source_inputs))}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    images, header_separations = [], []
    point_color, target_color = '#405FD0', '#D0841D'
    stages = {0: 'Initialization', 500: 'Pre-turn baseline',
              1000: 'After the first 30° turn', 1500: 'After the second 30° turn'}
    limits = (-6.8, 6.8)
    for frame, step in enumerate(arrays[0]['steps']):
        step = int(step)
        added = round(math.degrees(float(arrays[0]['angles'][frame])))
        angle = float(arrays[0]['angles'][frame])
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        target = arrays[0]['centers'].astype(np.float64) @ rotation.T
        assert np.max(np.abs(target)) < limits[1]
        fig = plt.figure(figsize=(12, 8), dpi=100, facecolor='#F8FAFD')
        fig.text(.5, .955, '100-Gaussian toy: two distribution shifts', ha='center',
                 va='center', fontsize=22, weight='bold', color='#182238')
        subtitle_text = fig.text(.5, .918, 'rotated100  ·  original seed 1234  ·  30° turns every 500 updates',
                 ha='center', fontsize=12, color='#536078')
        stage_text = fig.text(.5, .872, f'Update {step:,}  |  added target rotation +{added}°  |  {stages[step]}',
                 ha='center', fontsize=16, weight='bold', color='#25334D')
        panel_headers = []
        for which, (label, _) in enumerate(lanes):
            x = .067 if which == 0 else .555
            ax = fig.add_axes([x, .238, .378, .522], facecolor='white')
            points = arrays[which]['frames'][frame].astype(np.float32)
            assert np.max(np.abs(points)) < limits[1], 'every saved point must remain visible'
            ax.scatter(points[:, 0], points[:, 1], s=1.35, c=point_color, alpha=.72,
                       linewidths=0, rasterized=True)
            ax.scatter(target[:, 0], target[:, 1], s=28, facecolors='none', edgecolors=target_color,
                       linewidths=.9)
            ax.set(xlim=limits, ylim=limits, aspect='equal', xlabel='x', ylabel='y')
            ax.set_xticks([-6, -4, -2, 0, 2, 4, 6]); ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
            ax.tick_params(colors='#66718A', labelsize=9, length=3)
            for spine in ax.spines.values():spine.set_color('#CED5E1')
            title = 'RA14 · before recovery fix' if which == 0 else 'RA15 · fresh run with recovery fix'
            title_text = fig.text(x + .378 / 2, .820, title, ha='center', va='center',
                                  fontsize=14, weight='bold', color='#25334D')
            if step == 0:
                text, color = 'Recorded initialization · no gate at update 0', '#65708A'
            else:
                row = next(p for p in verdicts[label]['rotated100']['periods'] if p['period_end'] == step)
                threshold = .9 * verdicts[label]['rotated100']['pre_turn_hq']
                passed = row['modes'] >= 95 and row['hq'] >= threshold
                status = 'BASELINE' if step == 500 else 'PASS' if passed else 'FAIL'
                text = f"HQ {100 * row['hq']:.2f}%  ·  {row['modes']}/100 modes  ·  {status}"
                color = '#65708A' if step == 500 else '#16745D' if passed else '#B12C40'
            score_text = fig.text(x + .378 / 2, .785, text, ha='center', va='center',
                                  fontsize=12, weight='bold', color=color)
            panel_headers.append((title_text, score_text))
        threshold = .9 * verdicts['RA14']['rotated100']['pre_turn_hq']
        fig.text(.5, .160, f'Unchanged gate after each turn: ≥95 modes and HQ ≥{100 * threshold:.3f}%',
                 ha='center', fontsize=13, weight='bold', color='#25334D')
        handles = [Line2D([], [], marker='o', linestyle='none', color=point_color, markersize=5,
                          label='Generated samples (with output noise)'),
                   Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                          markeredgecolor=target_color, markersize=7, label='Current target centers')]
        fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, .115), ncols=2,
                   frameon=False, fontsize=11, handletextpad=.5, columnspacing=2)
        fig.text(.5, .067, 'Plotted: 4,096 saved points per panel.  HQ/modes: separate original 20,000-point gate draws.',
                 ha='center', fontsize=11, color='#536078')
        fig.text(.5, .039, 'Four observed states only; no interpolated training.  Fixed axes across all frames.',
                 ha='center', fontsize=10, color='#65708A')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        extent = lambda text: text.get_window_extent(renderer)
        gaps = [extent(subtitle_text).y0 - extent(stage_text).y1]
        for title_text, score_text in panel_headers:
            gaps.extend([extent(stage_text).y0 - extent(title_text).y1,
                         extent(title_text).y0 - extent(score_text).y1])
        assert min(gaps) >= 4, f'header rows need a readable gap: {gaps}'
        header_separations.append(dict(step=step, minimum_vertical_gap_pixels=min(gaps)))
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert('RGB'))
    preview = out / outputs[1]
    images[-1].save(preview, optimize=True)
    gallery = Image.new('RGB', (300, 200 * len(images)))
    for i, img in enumerate(images):gallery.paste(img.resize((300, 200)), (0, i * 200))
    palette = gallery.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in images]
    durations = [1200, 2200, 2200, 4400]
    gif = out / outputs[0]
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], loop=0,
                       duration=durations, disposal=2, optimize=True)
    assert gif.stat().st_size <= 5 * 1024 * 1024
    with Image.open(gif) as check:
        assert check.n_frames == 4 and check.size == (1200, 800)
    score_doc = dict(task_family='original moving100 two-turn gates', plotting_task='rotated100',
        protocol=dict(seed=1234, steps=1500, degrees_per_turn=30, rotate_every=500,
                      gate_points=20000, saved_plot_points=4096),
        HQ_definition='fraction within Euclidean distance <=0.09 of the nearest current target center',
        modes_definition='number of target centers with >=10 HQ samples in the 20000-point gate draw',
        acceptance='both turns: modes>=95 and HQ>=0.9*the step500 pre-turn baseline HQ',
        RA14_source='validation-ra14-r2; original adapter-corrected failed/passed controls',
        RA15_source='validation-ra15; fresh original-seed full1500-update reruns',
        rows=comparison, uniform_HQ_gain_claimed=False)
    (out / outputs[2]).write_text(json.dumps(score_doc, indent=2, sort_keys=True) + '\n')
    lines = ['# Original moving-task score comparison', '',
        'HQ is the percentage of 20,000 noisy generated samples within Euclidean distance 0.09 of the nearest current target center.',
        'A mode counts when at least 10 of those HQ samples are assigned to it.',
        'Each turn requires at least 95 modes and HQ ≥90% of that task’s update 500 baseline; the same baseline applies to both turns.', '',
        '|Task|Turn / update|HQ minimum|RA14 HQ / modes|RA15 HQ / modes|HQ change|RA14 → RA15|',
        '|---|---|---:|---:|---:|---:|---|']
    for row in comparison:
        a, b = row['RA14'], row['RA15']
        lines.append(f"|{row['task']}|+{row['added_target_rotation_deg']}° / {row['step']}|{100 * row['minimum_hq']:.3f}%|"
            f"{100*a['hq']:.2f}% / {a['modes']}|{100*b['hq']:.2f}% / {b['modes']}|"
            f"{row['hq_change_percentage_points']:+.2f} pp|{'PASS' if a['passed'] else 'FAIL'} → {'PASS' if b['passed'] else 'FAIL'}|")
    lines += ['', 'RA15 repairs the rotated100 second-turn failure. HQ does not improve uniformly: grid and staggered remain passing with some lower scores.', '',
        'Both versions use the original seed 1234, two 30° turns, 1,500 updates, and unchanged scoring and acceptance rules.',
        'These rows label the actual historical sources RA14 and fresh RA15; they are not relabelled as RA17 executions.', '']
    (out / outputs[3]).write_text('\n'.join(lines))
    caption = '''Four observed snapshots from the original rotated100 two-turn experiment (seed 1234).
Dots show 4,096 saved noisy generated samples; open rings show current target centers.
The snapshots are the original float16 observations at updates 0, 500, 1,000 and 1,500.
Target angles are additional rotations from the initial rotated100 geometry.
HQ and mode captions use separate original 20,000-sample gate draws.
No interpolation, extra samples or fresh training was used.

Both runs start from HQ 96.09% at update 500. After the second turn, RA14 scores
85.57% HQ / 99 modes (FAIL); fresh RA15 scores 92.59% HQ / 98 modes (PASS).
The unchanged gate is HQ ≥86.481% and at least 95 modes after both turns.
RA15 fixes this rotated failure; it does not produce uniform HQ gains across
the grid, rotated, and staggered moving tasks. See moving-score-comparison.md.
'''
    (out / outputs[4]).write_text(caption)
    readme = '''# Original moving-distribution observations

![Final recorded comparison](rotated100-shift-comparison.png)

The [GIF](rotated100-shift-comparison.gif) compares saved **RA14** observations
with the **fresh RA15** recovery run of the original `rotated100` task.
The source runs use seed 1234 and 1,500 updates, with two 30° target shifts.
These are the actual historical source labels; the animation is not a new
RA17 execution. The [score comparison](moving-score-comparison.md) includes
both original turns for grid100, rotated100 and staggered100.

## Frames and scores

Only four recorded states appear, at updates **0, 500, 1,000 and 1,500**.
Each panel plots every one of the original **4,096 saved float16 samples**.
Open rings show target centers calculated from the saved initial centers and
saved additional angles **0°, 0°, 30° and 60°**. The initial rotated100 layout
is already rotated 25°, so its absolute target orientations are 25°, 25°, 55°
and 85°. The first two recorded sample arrays are exactly identical across
RA14 and RA15. All frames share the same axes, and every saved point is visible.

HQ and mode labels come from **separate original 20,000-point gate draws**.
HQ measures the fraction within Euclidean distance 0.09 (three target standard
deviations) of the nearest current target center. A mode needs at least 10 HQ
samples. Both turns must retain at least 95 modes and HQ at least 90% of the
update 500 baseline. For rotated100, the baseline is 96.09% and the unchanged
HQ minimum is 86.481%.

After the second turn, RA14 has **85.57% HQ / 99 modes (FAIL)** and fresh RA15
has **92.59% HQ / 98 modes (PASS)**. HQ gains are not uniform across all three
moving tasks; the complete comparison retains the lower grid and staggered
scores. See [CAPTION.md](CAPTION.md) for a compact caption.

There are no diagnostic target-shift frames or interpolated particle moves.
The animation holds each observed state; recovery between snapshots is not
shown. Rendering used no model sampling, fresh training, PyTorch import or GPU.
The 4,096-point plots do not substitute for the original acceptance draws.

## Provenance and reproduction

[GIF-RECEIPT.json](GIF-RECEIPT.json) records the exact input and output SHA256
hashes, original source-package identities, rendering command, versions and
limits. Inputs are retained under these study paths:

- `validation-ra14-r2/moving/rotated100/frames.npz`
- `validation-ra15/moving/rotated100/frames.npz`
- `validation-{ra14-r2,ra15}/moving/{grid100,rotated100,staggered100}/frames.npz.verdict.json`

The renderer also hashes the original gate runner, adapted runners, launch and
completion receipts, and identical config files. It verifies input hashes again
after rendering. This directory contains the visualization and provenance;
the original point clouds remain in the study archive.

To reproduce into a new empty directory with the original archive available:

```sh
mkdir /tmp/rotated100-pr223-render
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python \\
  render_rotated100_comparison.py \\
  --study-root /ml2/hypergan/gan-attempts/feature-cells-generalization-20260930 \\
  --output-dir /tmp/rotated100-pr223-render
```

Earlier layout attempts are retained in `attempt-1/`, `attempt-2/` and
`attempt-3/`, including their exact renderer and any completed receipts.
Their receipts record the original canonical paths; the corresponding source
bytes are now in each attempt's `render_rotated100_comparison.py`. The final
render uses explicit figure coordinates, separates all header rows by at
least four pixels and cleans the prose. Numerical inputs and observations
are unchanged.
'''
    (out / outputs[5]).write_text(readme)
    for path, expected in input_hashes.items():assert sha(Path(path)) == expected, path
    receipt = dict(status='CREATED_FROM_ORIGINAL_RECORDED_OBSERVATIONS',
        command=[sys.executable, str(Path(__file__).resolve()), '--study-root', str(study), '--output-dir', str(out)],
        versions=dict(python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__, Pillow=PIL.__version__),
        input_sha256=input_hashes, input_files_unchanged_after_render=True,
        package_RA14_sha256=package_sha(study / 'pkg-RA14-replay'),
        package_RA15_sha256=package_sha(study / 'pkg-RA15-partial-recovery'),
        source_labels=['RA14', 'fresh RA15'], current_source_execution_claimed=False,
        steps=[0, 500, 1000, 1500], added_target_rotation_deg=[0, 0, 30, 60],
        saved_point_dtype='float16', points_per_panel=4096, samples_visible='all 4096; no point filtering',
        metric_draw_count=20000, metric_source='unchanged original gate verdicts',
        figure_size=[1200, 800], axes_limits=[-6.8, 6.8], same_axes_all_frames=True,
        header_separations=header_separations,
        animation_frames=4, diagnostic_target_shift_frames=0,
        frame_durations_ms=durations, gif_bytes=gif.stat().st_size,
        interpolation=False, offline_model_sampling=False, fresh_training=False, GPU_launch=False,
        torch_imported='torch' in sys.modules, score_comparison_rows=len(comparison),
        limitations=['Only four recorded observations; recovery between snapshots is not shown.',
                     'Saved plotting samples have original float16 precision; gate scores use separate 20k draws.',
                     'RA14/RA15 historical sources are labelled explicitly; this is not a new RA17 run.'],
        output_sha256={name:sha(out / name) for name in outputs if name != 'GIF-RECEIPT.json'})
    assert receipt['torch_imported'] is False
    (out / outputs[6]).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], gif=str(gif), gif_bytes=gif.stat().st_size,
        gif_sha256=sha(gif), preview=str(preview), receipt_sha256=sha(out / outputs[6])), sort_keys=True), flush=True)


if __name__ == '__main__':main()
