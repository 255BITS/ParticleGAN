"""Vector explanations of PR155 E22 and ParticleGAN Atlas; schematic, not data."""
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parent
C = dict(bg='#F4F7FC', ink='#17233C', muted='#52647E', line='#D4DDEB',
         blue='#3766E7', bluebg='#EAF0FF', teal='#007F78', tealbg='#E4F5F0',
         orange='#D87419', orangebg='#FFF2E4', purple='#7750B6', purplebg='#F1EBFA')


def canvas(height=1000):
    fig = plt.figure(figsize=(16, height / 100), facecolor=C['bg'])
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1600); ax.set_ylim(height, 0); ax.axis('off')
    return fig, ax


def text(ax, x, y, value, size=25, color='ink', weight='normal', ha='left', va='top'):
    return ax.text(x, y, value, fontsize=size * .72, color=C.get(color, color),
                   weight=weight, ha=ha, va=va, linespacing=1.4)


def card(ax, x, y, w, h, fill='white', edge=None, radius=24):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=C.get(fill, fill), edgecolor=C.get(edge, edge) if edge else C['line'], linewidth=1.5)
    ax.add_patch(patch); return patch


def arrow(ax, start, end, color='muted', width=2.8, curve=0, dashed=False):
    p = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=20,
        color=C[color], linewidth=width, connectionstyle=f'arc3,rad={curve}',
        linestyle='--' if dashed else '-', shrinkA=0, shrinkB=0)
    ax.add_patch(p)


def points(ax, x, y, w, h, color='blue', n=36, grouped=False):
    # Deterministic schematic positions; no experimental samples or random draws.
    for k in range(n):
        if grouped:
            group = k % 4; a = (k * 2.39996323) % (2 * math.pi)
            px = x + w * (.25 if group % 2 == 0 else .75) + 9 * math.cos(a)
            py = y + h * (.25 if group < 2 else .75) + 9 * math.sin(a)
        else:
            px = x + 12 + (k % 6) * (w - 24) / 5 + 4 * math.sin(k * 3)
            py = y + 12 + (k // 6) * (h - 24) / 5 + 4 * math.cos(k * 2)
        ax.add_patch(Circle((px, py), 4.5, facecolor=C[color], edgecolor='none', alpha=.85))


def export(fig, stem):
    for ext in ('svg', 'png'):
        fig.savefig(ROOT / f'{stem}.{ext}', dpi=200, facecolor=fig.get_facecolor(), pad_inches=0)
    plt.close(fig)


def baseline():
    fig, ax = canvas(1040)
    text(ax, 70, 48, 'E22: a GAN with a learned population', 47, weight='bold')
    text(ax, 70, 120, 'The generator and the latent particles learn together.', 28, 'muted')
    text(ax, 70, 179, 'HOW A SAMPLE IS MADE', 20, 'blue', 'bold')

    card(ax, 70, 228, 320, 258)
    text(ax, 94, 249, 'Latent particles', 31, weight='bold')
    text(ax, 94, 296, 'Trainable starting points', 23, 'muted')
    points(ax, 119, 349, 220, 95, n=36)
    card(ax, 458, 264, 246, 185, 'bluebg')
    text(ax, 581, 312, 'Generator', 34, 'blue', 'bold', ha='center')
    text(ax, 581, 371, 'G(z)', 30, 'blue', ha='center')
    card(ax, 776, 228, 304, 258)
    text(ax, 800, 249, 'Generated samples', 28, weight='bold')
    text(ax, 800, 295, 'What the GAN produces', 23, 'muted')
    points(ax, 819, 344, 219, 102, grouped=True)
    card(ax, 1159, 264, 350, 185, 'orangebg')
    text(ax, 1334, 303, 'Critic', 34, 'orange', 'bold', ha='center')
    text(ax, 1334, 361, 'Scores real and generated', 23, 'ink', ha='center')
    arrow(ax, (390, 356), (456, 356), 'blue')
    arrow(ax, (704, 356), (774, 356), 'blue')
    arrow(ax, (1080, 356), (1157, 356), 'blue')
    text(ax, 416, 377, 'sample', 18, 'muted', ha='center')

    card(ax, 1168, 519, 330, 105)
    text(ax, 1194, 551, 'Real data', 27, 'orange', 'bold')
    for k in range(10):
        ax.add_patch(Circle((1391 + (k % 5) * 16, 547 + (k // 5) * 21), 4.5,
                           facecolor=C['orange'], edgecolor='none'))
    arrow(ax, (1334, 519), (1334, 451), 'orange')

    # Gradients flow back through G to its latent inputs; the diagram is conceptual.
    arrow(ax, (1228, 451), (941, 543), 'purple', curve=-.12, dashed=True)
    arrow(ax, (902, 543), (581, 451), 'purple', curve=-.12, dashed=True)
    arrow(ax, (524, 451), (254, 488), 'purple', curve=-.1, dashed=True)
    text(ax, 708, 571, 'Adversarial gradients update G and the particles', 24,
         'purple', weight='bold', ha='center')

    card(ax, 70, 663, 1438, 265, 'tealbg', edge='teal')
    text(ax, 100, 688, 'E22 adds training feedback around this GAN loop', 31, 'teal', 'bold')
    labels = [
        ('Settle learning rates', 'Use optimizer history,\nwithout a fixed schedule.'),
        ('Rebalance the table', 'Compare real and generated\ncritic features locally.'),
        ('Recover after changes', 'Reopen rates and rescale\noptimizer memory after a shock.'),
    ]
    for k, (title, body) in enumerate(labels):
        x = 100 + k * 465
        text(ax, x, 759, title, 27, weight='bold')
        text(ax, x, 805, body, 23, 'muted')
        if k < 2: ax.plot([x + 432] * 2, [754, 886], color='#B9DDD5', lw=1.5)
    text(ax, 70, 963, 'Already in E22: learned output noise, critic regularization, and averaged inference.',
         24, 'muted')
    text(ax, 70, 1005, 'Concept diagram · E22 from PR155 · dots are schematic', 18, 'muted')
    export(fig, 'e22-explained')


def map_picture(ax, x, y, atlas):
    # Both diagrams use the same conceptual real/generated observations.
    if atlas:
        colors = ['#E4F1EA', '#F9EEE0', '#E8E9FC', '#E4F2F7']
        for k in range(4):
            ax.add_patch(Rectangle((x + (k % 2) * 155, y + (k // 2) * 79), 155, 79,
                                  facecolor=colors[k], edgecolor='white', linewidth=3))
    real = [(46, 35), (201, 35), (46, 114), (201, 114)]
    for cx, cy in real:
        for k in range(5):
            a = k * 2.39996
            ax.add_patch(Circle((x + cx + 10 * math.cos(a), y + cy + 8 * math.sin(a)),
                               4, facecolor=C['orange'], edgecolor='none'))
    for group, n in enumerate((14, 2, 9, 8)):
        cx, cy = real[group]
        # One underfilled group and one displaced group illustrate distinct errors.
        dx = 25 if group == 2 else 0
        for k in range(n):
            a = k * 2.39996
            ax.add_patch(Circle((x + cx + dx + (10 + k % 3) * math.cos(a),
                                y + cy + (9 + k % 2) * math.sin(a)), 3.5,
                               facecolor=C['blue'], edgecolor='none', alpha=.8))
    if not atlas:
        for cx, cy in real:
            ax.add_patch(Circle((x + cx, y + cy), 29, facecolor='none',
                               edgecolor=C['muted'], linewidth=1.7, linestyle='--'))
    else:
        arrow(ax, (x + 87, y + 115), (x + 51, y + 115), 'purple', width=2.5)
        arrow(ax, (x + 112, y + 45), (x + 185, y + 45), 'teal', width=2.5)


def changes():
    fig, ax = canvas(1260)
    text(ax, 70, 48, 'ParticleGAN Atlas: repair the local population', 45, weight='bold')
    text(ax, 70, 118, 'For eligible low-dimensional outputs: check regional counts and average positions.', 26, 'muted')
    text(ax, 70, 161, 'Cells are temporary regions fitted from recent real critic features.', 20, 'muted')

    card(ax, 70, 192, 681, 422)
    card(ax, 784, 192, 724, 422, 'bluebg')
    text(ax, 99, 216, 'E22  /  PR155', 31, weight='bold')
    text(ax, 814, 216, 'Atlas  /  this PR', 31, 'blue', 'bold')
    text(ax, 99, 269, 'Neighbor evidence in critic features', 24, 'muted')
    text(ax, 814, 269, 'A regional map in critic features', 24, 'muted')
    map_picture(ax, 150, 322, False)
    map_picture(ax, 912, 322, True)
    text(ax, 99, 521, 'Ask: is mass missing, excessive, or unsupported?', 22, weight='bold')
    text(ax, 99, 566, 'GAN gradients continue to learn placement.', 22, 'muted')
    text(ax, 814, 521, 'Ask: where is mass missing or misplaced?', 24, weight='bold')
    text(ax, 814, 566, 'Adds separate checks of groups’ output averages.', 22, 'muted')
    ax.add_patch(Circle((1312, 356), 5, facecolor=C['orange'], edgecolor='none'))
    text(ax, 1328, 343, 'real', 20, 'muted')
    ax.add_patch(Circle((1312, 395), 5, facecolor=C['blue'], edgecolor='none'))
    text(ax, 1328, 382, 'generated', 20, 'muted')
    text(ax, 1491, 481, 'Schematic feature map', 17, 'muted', ha='right')

    text(ax, 70, 659, 'THREE LOCAL CONTROL DECISIONS', 20, 'blue', 'bold')
    rows = [
        ('01', 'Counts', 'Allocate more particles\nto underfilled regions.', 'tealbg', 'teal'),
        ('02', 'Placement', 'Correct group averages\nby reallocating particles.', 'purplebg', 'purple'),
        ('03', 'Support', 'Create latent particles\nthat pass real-support checks.', 'orangebg', 'orange'),
    ]
    for k, (number, title, body, fill, color) in enumerate(rows):
        x = 70 + k * 486
        card(ax, x, 707, 463, 218, fill)
        text(ax, x + 26, 728, number, 20, color, 'bold')
        text(ax, x + 26, 764, title, 30, weight='bold')
        text(ax, x + 26, 817, body, 23, 'muted')

    card(ax, 70, 971, 1438, 186)
    text(ax, 99, 991, 'Choose the control path automatically', 29, weight='bold')
    text(ax, 99, 1046, 'Enough particles + ≤8 output coordinates', 24, 'blue', 'bold')
    arrow(ax, (674, 1058), (737, 1058), 'blue')
    text(ax, 764, 1046, 'Feature cells + placement checks', 24, 'blue', 'bold')
    text(ax, 99, 1096, 'Other shapes and custom/routed models retain their existing control paths.', 23, 'muted')
    text(ax, 70, 1192, 'Also added: a settled-game reopen guard and CPU/CUDA checkpoint portability repairs.',
         23, 'muted')
    text(ax, 70, 1230, 'Concept diagram · low-dimensional feature cells are opt-in · no universal speed or quality claim',
         17, 'muted')
    export(fig, 'atlas-changes')


if __name__ == '__main__':
    baseline(); changes()
    print('Wrote two SVG infographics and two high-resolution PNGs.')
