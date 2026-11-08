"""Contact-sheet GIFs for the Atlas/E22 policy-cloud cohort report.

Every trained attempt wrote ``training.gif`` during its run from its own saved
scoring observations. This module tiles those recorded frames into one GIF per
family and scoring law. It reads only the verified in-run GIF bytes, so it adds
zero optimizer updates and zero sampling draws. Output bytes depend only on the
source GIFs, the tile order and Pillow's encoder, so a rerun writes identical
bytes.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import file_hash

TILE = (216, 144)
CAPTION = 16
COLUMNS = 6
FRAMES = 5
FRAME_MS = 900
LAST_FRAME_MS = 2200
BACKGROUND = (255, 255, 255)
INK = (20, 20, 20)
STATUS_INK = {"PASS": (0, 110, 40), "FAIL": (170, 20, 20)}


def _frames(path):
    from PIL import Image
    with Image.open(path) as image:
        count = getattr(image, "n_frames", 1)
        frames = []
        for index in range(count):
            image.seek(index)
            frames.append(image.convert("RGB"))
    return frames


def _fit(frame):
    from PIL import Image
    width, height = frame.size
    scale = min(TILE[0] / width, TILE[1] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    tile = Image.new("RGB", TILE, BACKGROUND)
    tile.paste(frame.resize(size, Image.Resampling.LANCZOS),
               ((TILE[0] - size[0]) // 2, (TILE[1] - size[1]) // 2))
    return tile


def contact_sheet(tiles, destination):
    """Write one GIF whose frame k shows frame k of every source (held at its end).

    ``tiles`` is an ordered list of ``(caption, status, gif_path)``. Returns the
    frame mapping used for each source so the receipt can state it.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()
    sources = [_frames(path) for _, _, path in tiles]
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    cell = (TILE[0], TILE[1] + CAPTION)
    mapping = []
    for frames in sources:
        count = len(frames)
        mapping.append([round(k * (count - 1) / (FRAMES - 1)) if count > 1 else 0 for k in range(FRAMES)])
    sheets = []
    for k in range(FRAMES):
        sheet = Image.new("RGB", (COLUMNS * cell[0], rows * cell[1]), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for index, ((caption, status, _), frames) in enumerate(zip(tiles, sources)):
            x, y = (index % COLUMNS) * cell[0], (index // COLUMNS) * cell[1]
            width = draw.textlength(status, font=font)
            text, room = caption, cell[0] - width - 12
            while len(text) > 4 and draw.textlength(text, font=font) > room:
                text = text[:-3] + ".."
            draw.text((x + 3, y + 2), text, fill=INK, font=font)
            draw.text((x + cell[0] - width - 4, y + 2), status, fill=STATUS_INK.get(status, INK), font=font)
            sheet.paste(_fit(frames[mapping[index][k]]), (x, y + CAPTION))
        sheets.append(sheet.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    durations = [FRAME_MS] * (FRAMES - 1) + [LAST_FRAME_MS]
    sheets[0].save(destination, save_all=True, append_images=sheets[1:], duration=durations, loop=0,
                   optimize=False, disposal=1)
    return mapping


def render_cohort_media(root, report_directory, trained, *, families, laws, order):
    """One contact sheet per (family, law); returns the media receipt or None.

    ``trained`` maps ``(family, measurement_law, parent)`` to a drained cohort
    job whose ``training.gif`` hash was already verified against its receipt.
    Shared (law-independent) attempts appear on the clean sheet only.
    ``order`` lists parent tests in report order.
    """
    root, out = Path(root), Path(root) / report_directory / "gifs"
    sheets = []
    for family in families:
        for law in laws:
            tiles, sources = [], []
            for parent in order:
                job = trained.get((family, law, parent))
                if job is None and law == "clean":
                    job = trained.get((family, "shared", parent))
                if job is None:
                    continue
                gif = Path(job["directory"]) / "training.gif"
                if not gif.is_file():
                    continue
                caption = parent + (" (shared)" if job["measurement_law"] == "shared" else "")
                tiles.append((caption, job["status"], gif))
                sources.append({"test": parent, "measurement_law": job["measurement_law"], "status": job["status"],
                                "attempt_id": job.get("attempt_id"), "training_gif_sha256": file_hash(gif)})
            if not tiles:
                continue
            path = out / f"{family}-{law.replace('_', '-')}.gif"
            mapping = contact_sheet(tiles, path)
            for source, frames in zip(sources, mapping):
                source["frames_used"] = frames
            sheets.append({"family": family, "law": law, "path": path.relative_to(root).as_posix(),
                           "sha256": file_hash(path), "bytes": path.stat().st_size, "tiles": sources})
    if not sheets:
        return None
    return {"kind": "contact_sheets_of_in_run_training_gifs", "optimizer_updates_added": 0, "sampling_draws_added": 0,
            "renderer_sha256": file_hash(Path(__file__)), "frames_per_sheet": FRAMES, "tile_pixels": list(TILE),
            "note": ("Each tile is the attempt's own training.gif, written during its run from its saved scoring "
                     "observations; the sheet only resamples and tiles those frames."),
            "sheets": sheets}
