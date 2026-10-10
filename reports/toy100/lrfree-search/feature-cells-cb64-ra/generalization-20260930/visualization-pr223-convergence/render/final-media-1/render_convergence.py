"""Render actual paired E22/Atlas observations; no model imports or interpolation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from PIL import Image


ATLAS = "#176BB3"
E22 = "#68566F"
TARGET = "#C46A13"
INK = "#18263C"
MUTED = "#56667B"
BG = "#F5F8FC"
GREEN = "#176B4A"
SHIFT = "#8B3979"
SIGMA = .03
STEPS = 1500


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rotation(angle):
    c, s = math.cos(float(angle)), math.sin(float(angle))
    return np.array([[c, -s], [s, c]])


@dataclass
class Capture:
    name: str
    frames: np.ndarray
    steps: np.ndarray
    angles: np.ndarray
    centers: np.ndarray
    events: np.ndarray
    hq: np.ndarray
    modes: np.ndarray
    path: Path | None = None

    @classmethod
    def read(cls, name, path):
        with np.load(path, allow_pickle=False) as source:
            data = {k: source[k].copy() for k in source.files}
        required = {"frames", "steps", "angles", "centers", "event_kind", "capture_hq", "capture_modes"}
        if not required <= data.keys():
            raise ValueError(f"{name}: missing fields {sorted(required - data.keys())}")
        x = cls(name, data["frames"], data["steps"], data["angles"], data["centers"],
                data["event_kind"].astype(str), data["capture_hq"], data["capture_modes"], path)
        n = len(x.steps)
        assert n == 153 and x.frames.shape == (n, 4096, 2) and x.centers.shape == (100, 2)
        assert x.frames.dtype == np.float32
        assert np.issubdtype(x.steps.dtype, np.integer) and np.issubdtype(x.modes.dtype, np.integer)
        assert all(v.shape == (n,) for v in (x.angles, x.events, x.hq, x.modes))
        assert np.isfinite(x.frames).all() and np.isfinite(x.centers).all()
        assert np.isfinite(x.hq).all() and np.isfinite(x.angles).all()
        assert (np.diff(x.steps) >= 0).all() and x.steps[0] == 0 and x.steps[-1] == STEPS
        assert ((x.hq >= 0) & (x.hq <= 1)).all() and ((x.modes >= 0) & (x.modes <= 100)).all()
        assert set(x.events) <= {"initial", "update", "target_shift"}
        shifts = np.flatnonzero(x.events == "target_shift")
        assert x.steps[shifts].tolist() == [500, 1000]
        ordinary = x.events != "target_shift"
        assert x.steps[ordinary].tolist() == list(range(0, STEPS + 1, 10))
        assert x.events[0] == "initial" and (x.events[ordinary][1:] == "update").all()
        for i in shifts:
            assert i > 0 and x.steps[i] == x.steps[i - 1]
            assert np.array_equal(x.frames[i], x.frames[i - 1]), "target shifts must retain the observed model cloud"
            assert math.isclose(x.angles[i] - x.angles[i - 1], math.pi / 6, abs_tol=1e-10)
        return x

    @classmethod
    def layout(cls, name):
        a = np.arange(10, dtype=float) - 4.5
        centers = np.stack(np.meshgrid(a, a), -1).reshape(-1, 2) @ rotation(math.radians(25)).T
        return cls(name, np.empty((1, 0, 2)), np.array([0]), np.array([0.]), centers,
                   np.array(["initial"]), np.array([np.nan]), np.array([0]))


def gate_rows(path):
    if path is None:
        return []
    obj = json.loads(path.read_text())
    rows = obj.get("periods", obj.get("gate_rows", []))
    assert [r["period_end"] for r in rows] == [500, 1000, 1500]
    assert [r["target_deg"] for r in rows] == [0, 30, 60]
    assert all(0 <= r["hq"] <= 1 and type(r["modes"]) is int and 0 <= r["modes"] <= 100 for r in rows)
    if "task" in obj:
        assert obj["task"] == "rotated100"
    baseline = rows[0]["hq"]
    if "pre_turn_hq" in obj:
        assert obj["pre_turn_hq"] == baseline
    values = [dict(r, minimum_hq=.9 * baseline,
                   passed=None if r["period_end"] == 500 else (r["modes"] >= 95 and r["hq"] >= .9 * baseline),
                   baseline=r["period_end"] == 500) for r in rows]
    if "status" in obj:
        assert obj["status"] == ("PASS" if all(r["passed"] for r in values[1:]) else "FAIL")
    return values


def box(fig, xywh, color="white", border="#D9E1EB", radius=.012):
    x, y, w, h = xywh
    artist = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=color, edgecolor=border, linewidth=1, zorder=-1)
    fig.add_artist(artist)
    return artist


def txt(fig, x, y, text, size=16, color=INK, weight="normal", ha="left", va="center"):
    return fig.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va=va)


def style_axes(ax, *, ticks=True):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD5E1")
    ax.tick_params(colors=MUTED, labelsize=12, length=3)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def phase(capture, index, gates, layout=False, comparison=None):
    if layout:
        return "Layout preview", "Actual paired captures are being prepared", MUTED
    if capture.events[index] == "target_shift":
        return "Target rotates 30°", "The target jumps; these model samples have not moved", SHIFT
    step = int(capture.steps[index])
    if step <= 100:
        return "Start learning", "Both methods learn from the same observations", ATLAS
    near = [capture.name] if capture.hq[index] >= .9 else []
    if comparison is not None and comparison.hq[index] >= .9:
        near.append(comparison.name)
    if near:
        name = "Both" if len(near) == 2 else near[0]
        color = GREEN if len(near) == 2 else (ATLAS if near[0] == "Atlas" else E22)
        return f"{name} ≥90% near target", "Visual guide on 4,096 points; original gate checks are separate", color
    if capture.angles[index] > 0:
        return "Recovering", "New observations guide the model toward the shifted target", ATLAS
    return "Learning the distribution", "Watch the generated cloud form around the target centers", ATLAS


def draw_frame(atlas, e22, index, gates, *, extent, detail_mode, layout=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "axes.labelcolor": MUTED, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16, 13.8), dpi=100, facecolor=BG)
    step = int(atlas.steps[index])
    target = atlas.centers.astype(float) @ rotation(atlas.angles[index]).T
    title, subtitle, phase_color = phase(atlas, index, gates["Atlas"], layout, comparison=e22)
    if not layout and atlas.events[index] != "target_shift":
        checks = {name: next((r for r in rows if r["period_end"] == step), None)
                  for name, rows in gates.items()}
        if all(checks.values()):
            def score(name):
                r = checks[name]
                return f"{name} {100 * r['hq']:.2f}% / {r['modes']} modes"
            difference = 100 * (checks["Atlas"]["hq"] - checks["E22"]["hq"])
            leader = "Atlas" if difference >= 0 else "E22"
            label = "Baseline" if step == 500 else "Both pass" if all(r["passed"] for r in checks.values()) else "Recorded check"
            title = f"{label} · {leader} +{abs(difference):.2f} pp HQ"
            phase_color = ATLAS if leader == "Atlas" else E22
            subtitle = "20,000 points: " + score("Atlas") + "   ·   " + score("E22")
    txt(fig, .055, .963, "ParticleGAN Atlas", 31, weight="bold")
    txt(fig, .945, .963, "Compared with E22 · PR155", 21, color=E22, ha="right", weight="bold")
    txt(fig, .055, .927, "Learn 100 narrow clusters. Recover when the target rotates.", 21, color=MUTED)
    box(fig, (.055, .870, .89, .038), color="#EAF1FA", border="#DFE8F4")
    txt(fig, .070, .889, title, 21, phase_color, "bold")
    txt(fig, .430, .889, subtitle, 16, color=MUTED)
    txt(fig, .945, .927, f"Update {step:,} / 1,500", 20, ha="right", weight="bold")

    # One large hero view. Reference and local fit are explanatory companions.
    txt(fig, .062, .846, "Atlas", 23, ATLAS, "bold")
    txt(fig, .595, .846, "Generated samples against current targets", 16, MUTED, ha="right")
    hero = fig.add_axes([.06875, .243, .485, .5623])
    hero.set_aspect("equal", adjustable="box")
    style_axes(hero)
    hero.set(xlim=(-extent, extent), ylim=(-extent, extent))
    hero.set_xticks([-6, -3, 0, 3, 6]); hero.set_yticks([-6, -3, 0, 3, 6])
    points = atlas.frames[index]
    hero.scatter(points[:, 0], points[:, 1], s=7.5, c=ATLAS, alpha=.65, linewidths=0, rasterized=True)
    if atlas.events[index] == "target_shift":
        old = atlas.centers.astype(float) @ rotation(atlas.angles[index - 1]).T
        hero.scatter(old[:, 0], old[:, 1], s=44, facecolors="none", edgecolors="#9CA8B8", linewidths=1)
        for k in range(0, 100, 20):
            hero.annotate("", target[k], old[k], arrowprops=dict(arrowstyle="->", color=TARGET, lw=1.2, alpha=.8))
    hero.scatter(target[:, 0], target[:, 1], s=52, facecolors="none", edgecolors=TARGET, linewidths=1.3)
    hero.add_patch(Circle(target[detail_mode], .24, fill=False, color=INK, linewidth=1.5, linestyle="--"))
    hero.annotate("Detail →", target[detail_mode], xytext=(8, 12), textcoords="offset points", fontsize=13,
                  color=INK, weight="bold")

    txt(fig, .645, .846, "E22 reference", 21, E22, "bold")
    small = fig.add_axes([.646, .547, .270, .258])
    small.set_aspect("equal", adjustable="box"); style_axes(small, ticks=False)
    small.set(xlim=(-extent, extent), ylim=(-extent, extent))
    p = e22.frames[index]
    small.scatter(p[:, 0], p[:, 1], s=7, c=E22, marker="^", alpha=.62, linewidths=0, rasterized=True)
    small.scatter(target[:, 0], target[:, 1], s=25, facecolors="none", edgecolors=TARGET, linewidths=.9)

    txt(fig, .645, .520, "Magnified Gaussian fit", 20, weight="bold")
    txt(fig, .645, .495, "One fixed target region · σ = 0.03", 15, color=MUTED)
    local = fig.add_axes([.671, .242, .220, .230])
    style_axes(local); local.set_aspect("equal", adjustable="box")
    center = target[detail_mode]
    radius = .18
    for c, color, marker in ((e22, E22, "^"), (atlas, ATLAS, "o")):
        p = c.frames[index] - center
        keep = (np.abs(p) <= radius).all(1)
        local.scatter(p[keep, 0], p[keep, 1], s=27, c=color, marker=marker, alpha=.56, linewidths=0)
    for r, line in ((SIGMA, "-"), (3 * SIGMA, "--")):
        local.add_patch(Circle((0, 0), r, fill=False, edgecolor=TARGET, lw=1.6, linestyle=line))
    local.scatter([0], [0], marker="+", color=TARGET, s=60, linewidths=1.6)
    local.set(xlim=(-radius, radius), ylim=(-radius, radius))
    local.set_xticks([-.15, 0, .15]); local.set_yticks([-.15, 0, .15])
    local.tick_params(labelsize=11)
    txt(fig, .781, .211, "Target rings: 1σ and 3σ", 14, MUTED, ha="center")

    handles = [Line2D([], [], marker="o", linestyle="none", color=ATLAS, label="Atlas samples", markersize=7),
               Line2D([], [], marker="^", linestyle="none", color=E22, label="E22 samples", markersize=7),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=TARGET,
                          label="Target centers", markersize=8)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.315, .211), ncols=3, frameon=False,
               fontsize=15, handletextpad=.4, columnspacing=1.2)

    # Dense curves describe the visualization draw, not the acceptance draw.
    txt(fig, .060, .180, "Observed learning and recovery", 19, weight="bold")
    txt(fig, .945, .180, "Lines: 4,096-point diagnostics  •  Dots: original 20,000-point checks",
        14, MUTED, ha="right")
    quality = fig.add_axes([.073, .062, .533, .094])
    coverage = fig.add_axes([.680, .062, .261, .094])
    for ax, ylabel in ((quality, "Near target (%)"), (coverage, "Modes / 100")):
        style_axes(ax)
        ax.set(xlim=(0, STEPS), ylim=(0, 105))
        ax.set_xticks([0, 500, 1000, 1500]); ax.set_yticks([0, 50, 100])
        ax.grid(axis="y", color="#E3E8F0", lw=.7)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.axvline(500, color="#B2BDC9", ls=":", lw=1)
        ax.axvline(1000, color="#B2BDC9", ls=":", lw=1)
        ax.axvline(step, color=INK, lw=1.5)
    quality.axhline(90, color="#ADB8C7", linewidth=1, linestyle=":")
    quality.text(25, 93, "90% visual guide", fontsize=10, color=MUTED,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    for c, color in ((e22, E22), (atlas, ATLAS)):
        line, marker = ("--", "^") if c.name == "E22" else ("-", "o")
        quality.plot(c.steps[:index + 1], 100 * c.hq[:index + 1], color=color, lw=2.7, linestyle=line)
        coverage.plot(c.steps[:index + 1], c.modes[:index + 1], color=color, lw=2.4, linestyle=line)
        for row in gates[c.name]:
            if row["period_end"] > step:
                continue
            quality.scatter(row["period_end"], 100 * row["hq"], color=color, marker=marker, s=49, edgecolors="white", linewidths=.8, zorder=4)
            coverage.scatter(row["period_end"], row["modes"], color=color, marker=marker, s=49, edgecolors="white", linewidths=.8, zorder=4)
    txt(fig, .073, .034, "HQ: points within 0.09 of a target. Visual guide: 90% on 4,096 points. Diagnostic mode: ≥10 near-target points.", 13, MUTED)
    gate_note = "Original gate: both turns need ≥95 modes and HQ ≥90% of each method’s step-500 baseline."
    if gates["Atlas"] and gates["E22"]:
        a_min, e_min = (100 * gates[n][0]["minimum_hq"] for n in ("Atlas", "E22"))
        gate_note = (f"Original 20,000-point gate: ≥95 modes; HQ ≥{a_min:.3f}% Atlas / {e_min:.3f}% E22 "
                     "(90% of own step-500 baseline), after both turns.")
    txt(fig, .073, .017, gate_note, 13, MUTED)
    if layout:
        box(fig, (.073, .635, .515, .065), color="#FFFFFF", border="#D7DEE9")
        txt(fig, .330, .669, "DESIGN DRAFT · ACTUAL CAPTURES PENDING", 18, MUTED, "bold", ha="center")
    else:
        current = f"4,096 points: {100 * atlas.hq[index]:.1f}% near target · {int(atlas.modes[index])}/100 modes"
        txt(fig, .311, .824, current, 15, ATLAS, ha="center")
        txt(fig, .781, .824, f"4,096 points: {100 * e22.hq[index]:.1f}% · {int(e22.modes[index])}/100 modes",
            14, E22, ha="center")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    return fig, image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas-capture", type=Path)
    ap.add_argument("--e22-capture", type=Path)
    ap.add_argument("--atlas-gates", type=Path)
    ap.add_argument("--e22-gates", type=Path)
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--detail-mode", type=int, default=55)
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--layout-only", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        ap.error("--output-dir must be a new directory")
    assert 0 <= args.detail_mode < 100
    if args.layout_only:
        atlas, e22 = Capture.layout("Atlas"), Capture.layout("E22")
    else:
        if any(p is None for p in (args.atlas_capture, args.e22_capture, args.atlas_gates, args.e22_gates)):
            ap.error("both actual paired captures and both original gate files are required")
        atlas, e22 = Capture.read("Atlas", args.atlas_capture), Capture.read("E22", args.e22_capture)
        assert np.array_equal(atlas.steps, e22.steps) and np.array_equal(atlas.events, e22.events)
        assert np.allclose(atlas.angles, e22.angles, rtol=0, atol=1e-12)
        assert np.array_equal(atlas.centers, e22.centers)
        assert np.array_equal(atlas.frames[0], e22.frames[0]), "paired original initialization samples differ"
    gates = {"Atlas": gate_rows(args.atlas_gates), "E22": gate_rows(args.e22_gates)}
    inputs = [Path(__file__).resolve()]
    inputs += [p.resolve() for p in (args.atlas_capture, args.e22_capture, args.atlas_gates, args.e22_gates, args.metadata) if p]
    hashes = {str(p): sha(p) for p in inputs}
    if args.metadata:
        json.loads(args.metadata.read_text())
    extreme = max([6.8] + [float(np.abs(c.frames).max()) * 1.03 for c in (atlas, e22) if c.frames.size])
    extent = math.ceil(extreme * 2) / 2
    out.mkdir(parents=True)
    renderer_snapshot = out / "render_convergence.py"
    shutil.copyfile(Path(__file__).resolve(), renderer_snapshot)
    fig, poster = draw_frame(atlas, e22, len(atlas.steps) - 1, gates, extent=extent,
                            detail_mode=args.detail_mode, layout=args.layout_only)
    poster.save(out / "poster.png", optimize=True)
    fig.savefig(out / "poster.svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    outputs = [out / "poster.png", out / "poster.svg", renderer_snapshot]
    durations = []
    if not args.preview_only and not args.layout_only:
        cache = out / "frames"
        cache.mkdir()
        paths = []
        for i, step in enumerate(atlas.steps):
            fig, img = draw_frame(atlas, e22, i, gates, extent=extent, detail_mode=args.detail_mode)
            path = cache / f"{i:04d}.png"
            img.save(path, compress_level=2)
            plt.close(fig)
            paths.append(path)
            hold = 2200 if atlas.events[i] == "target_shift" else 180
            if step == 0: hold = 1600
            if i == len(atlas.steps) - 1: hold = 3600
            if step in (500, 1000) and atlas.events[i] != "target_shift": hold = 900
            durations.append(hold)
            if i % 20 == 0:
                print(json.dumps(dict(event="render_progress", frame=i, total=len(atlas.steps), step=int(step))), flush=True)
        gallery = Image.new("RGB", (400 * 4, 280 * 5), "white")
        for j, i in enumerate(np.linspace(0, len(paths) - 1, 20, dtype=int)):
            with Image.open(paths[int(i)]) as im:
                gallery.paste(im.resize((400, 280)), ((j % 4) * 400, (j // 4) * 280))
        palette = gallery.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        def gif_images():
            for path in paths:
                with Image.open(path) as im:
                    yield im.quantize(palette=palette, dither=Image.Dither.NONE)
        sequence = gif_images()
        first = next(sequence)
        gif = out / "atlas-vs-e22-convergence.gif"
        first.save(gif, save_all=True, append_images=sequence, loop=0, duration=durations,
                   disposal=2, optimize=True)
        outputs.append(gif)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            concat = out / "video-input.txt"
            with concat.open("w") as stream:
                for path, ms in zip(paths, durations):
                    stream.write(f"file '{path.as_posix()}'\nduration {ms / 1000:.3f}\n")
                stream.write(f"file '{paths[-1].as_posix()}'\n")
            video = out / "atlas-vs-e22-convergence.mp4"
            command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat), "-vf", "fps=24,format=yuv420p", "-filter_threads", "2", "-c:v", "libx264", "-threads", "2", "-preset", "medium",
                "-crf", "18", "-movflags", "+faststart", str(video)]
            subprocess.run(command, check=True)
            outputs.append(video)
    assert "torch" not in sys.modules
    assert all(sha(Path(p)) == value for p, value in hashes.items())
    receipt = dict(status="LAYOUT_DRAFT_NO_OBSERVED_DATA" if args.layout_only else "RENDERED_FROM_ACTUAL_PAIRED_CAPTURES",
        inputs_sha256=hashes, outputs_sha256={p.name: sha(p) for p in outputs},
        variants=["E22 (PR155)", "ParticleGAN Atlas"], frame_count=len(atlas.steps),
        steps=atlas.steps.tolist(), events=atlas.events.tolist(), frame_durations_ms=durations,
        visual_samples=4096, visual_mode_rule="At least 10 of 4,096 samples within 0.09 of a target center.",
        official_gate_samples=20000, official_gate_rule="Both turns: modes >= 95 and HQ >= 0.9 * own step-500 baseline.",
        official_gates=gates, chart_lines="Straight lines join measured diagnostics; generated clouds are never interpolated.",
        visual_phase_guide={"HQ": .9, "draw_count": 4096, "purpose": "Presentation guide, separate from original acceptance."},
        detail_mode=args.detail_mode, detail_selected_before_quality=False if args.detail_mode != 55 else True,
        target_sigma=SIGMA, fixed_axes=[-extent, extent], figure_pixels=[1600,1380],
        interpolation=False, target_shift_clouds_unchanged=True, model_sampling=False,
        new_training=False, GPU_launch=False, torch_imported=False,
        preview_only=args.preview_only, metadata_path=str(args.metadata) if args.metadata else None)
    (out / "RENDER-RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=receipt["status"], output_dir=str(out), outputs=[p.name for p in outputs])), flush=True)


if __name__ == "__main__":
    main()
