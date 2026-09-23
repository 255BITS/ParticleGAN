"""Local, traceable GIF and HTML evidence for paired Lunar flights."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

SIM_FPS = 50
HOLD_TICKS = 3


def episode_summary(ep: dict) -> dict:
    """Small JSON-safe record; never serialize image or state arrays."""
    actions = np.asarray(ep.get("actions", []), dtype=float)
    down = np.asarray(ep.get("downward_main_power", []), dtype=float)
    return {
        "seed": int(ep["seed"]), "outcome": str(ep["outcome"]),
        "steps": int(ep["steps"]), "return": float(ep["return_"]),
        "contact_step": None if ep.get("contact_step") is None else int(ep["contact_step"]),
        "variant": str(ep.get("variant", "unknown")),
        "downward_boost_steps": int(np.count_nonzero(down > 0)),
        "downward_boost_fraction": float(np.mean(down > 0)) if len(down) else 0.,
        "mean_main_command": float(np.mean(actions[:, 0])) if actions.ndim == 2 and len(actions) else None,
    }


def _samples(ep: dict) -> list[int]:
    frames = np.asarray(ep["frames"])
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or len(frames) < 2:
        raise ValueError("render_comparison requires RGB rollout frames including reset and terminal")
    stride = int(ep.get("frame_stride", 3))
    if stride < 1:
        raise ValueError("frame_stride must be positive")
    steps = int(ep["steps"])
    if steps < 1:
        raise ValueError("Episode must include at least one simulator step")
    times = [0] + list(range(stride, steps + 1, stride))
    if times[-1] != steps:
        times.append(steps)
    if len(times) != len(frames):
        raise ValueError("Frame count does not match frame_stride and terminal step")
    return times


def _font(size: int):
    from PIL import ImageFont
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _frame_at(ep: dict, samples: list[int], tick: int):
    from PIL import Image
    index = int(np.searchsorted(samples, tick, side="right") - 1)
    frame = np.asarray(ep["frames"][max(0, index)], dtype=np.uint8)
    image = Image.fromarray(frame[..., :3])
    width = min(430, image.width)
    if image.width != width:
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
    return image


def _down_at(ep: dict, tick: int) -> bool:
    power = np.asarray(ep.get("downward_main_power", []))
    if len(power) == 0 or tick <= 0 or tick > int(ep["steps"]):
        return False
    return bool(power[min(tick - 1, len(power) - 1)] > 0)


def _card(ep: dict, samples: list[int], tick: int, label: str):
    from PIL import Image, ImageDraw
    flight = _frame_at(ep, samples, tick)
    card = Image.new("RGB", (flight.width + 32, flight.height + 112), "#101b2b")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((5, 5, card.width - 6, card.height - 6), radius=14,
                           outline="#32506b", width=2)
    card.paste(flight, (16, 51))
    accent = "#67dbed" if label.lower().startswith("slow") else "#ffb35c"
    draw.text((17, 14), label.upper(), fill=accent, font=_font(20))
    draw.text((card.width - 125, 20), f"T+{tick / SIM_FPS:06.2f}s", fill="#e9f3fa", font=_font(14))
    status = f"SEED {ep['seed']}  |  {str(ep['outcome']).replace('_', ' ').upper()}  |  {ep['steps']} STEPS"
    draw.text((17, flight.height + 60), status, fill="#d6e4f0", font=_font(12))
    down = _down_at(ep, tick)
    draw.text((17, flight.height + 82),
              f"DOWN BOOST {'ACTIVE' if down else 'OFF'}  |  RETURN {float(ep['return_']):+.1f}",
              fill="#ffb35c" if down else "#93a9bd", font=_font(12))
    return card


def _save_gif(path: Path, frames: list, durations_ticks: list[int]) -> None:
    if len(frames) != len(durations_ticks):
        raise ValueError("GIF frames and durations disagree")
    # GIF stores centiseconds; at 50 simulator steps/s every step is exactly 2 cs.
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:],
                   duration=[round(t * 1000 / SIM_FPS) for t in durations_ticks],
                   loop=0, optimize=False, disposal=2)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dashboard(report: dict, slow: dict, fast: dict) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for label, ep in (("Slow", slow), ("Fast", fast)):
        s = episode_summary(ep)
        rows.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in
                    (label, s["outcome"].replace("_", " "), s["steps"], f"{s['return']:+.1f}",
                     s["downward_boost_steps"])) + "</tr>")
    slow_ok = slow["outcome"] == "successful_landing"
    fast_ok = fast["outcome"] == "successful_landing"
    verdict = (f"FAST LANDED {int(slow['steps']) - int(fast['steps'])} STEPS EARLIER"
               if slow_ok and fast_ok and int(fast["steps"]) < int(slow["steps"])
               else "COMPARE SUCCESSFUL LANDINGS BEFORE CLAIMING SPEED")
    metadata = json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False)
    metadata = esc(metadata)
    board = report.get("leaderboard", [])
    if isinstance(board, dict):
        board = [dict(controller=name, **row) for name, row in board.items() if isinstance(row, dict)]
    if isinstance(board, list) and board and all(isinstance(row, dict) for row in board):
        columns = list(dict.fromkeys(key for row in board for key in row))
        head = "".join(f"<th>{esc(key.replace('_', ' '))}</th>" for key in columns)
        body = "".join("<tr>" + "".join(f"<td>{esc(row.get(key, '—'))}</td>" for key in columns)
                       + "</tr>" for row in board)
        board_html = f'<section class="panel"><h2>Flight leaderboard</h2><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>'
    else:
        board_html = ""
    provenance = "".join(
        f"<div><strong>{esc(key.replace('_', ' '))}</strong> · <code>{esc(value)}</code></div>"
        for key, value in report.items()
        if ("checkpoint" in key.lower() or "config" in key.lower() or key.lower() == "winner")
        and isinstance(value, (str, Path)))
    provenance_html = f'<section class="panel"><h2>Run provenance</h2>{provenance}</section>' if provenance else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lunar flight comparison</title><style>
:root {{ color-scheme:dark; font-family:Inter,ui-sans-serif,system-ui,sans-serif; background:#07111d; color:#e9f3fa; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:radial-gradient(circle at 25% 0,#193047 0,transparent 42%),#07111d }}
main {{ max-width:1120px; margin:auto; padding:40px 24px 80px }} .eyebrow {{ color:#69dbea; font-size:12px; letter-spacing:.24em; font-weight:800 }}
h1 {{ font-size:clamp(32px,5vw,58px); margin:10px 0 12px; letter-spacing:-.04em }} p {{ color:#aec4d5; line-height:1.55 }}
.badge {{ display:inline-block; border:1px solid #38768b; border-radius:99px; padding:10px 17px; color:#9ceafa; font-size:13px; font-weight:800; letter-spacing:.06em }}
.panel {{ background:#101d2d; border:1px solid #294257; border-radius:19px; padding:22px; margin-top:24px; box-shadow:0 14px 40px #0005 }}
.hero img {{ display:block; width:100%; max-width:940px; margin:auto; image-rendering:auto }}
.twocol {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px }} .twocol img {{ display:block; width:100% }}
h2 {{ margin:0 0 16px; font-size:20px }} .cyan {{ color:#67dbed }} .orange {{ color:#ffb35c }}
table {{ width:100%; border-collapse:collapse; text-align:left }} th,td {{ padding:12px 10px; border-bottom:1px solid #294257 }} th {{ color:#8aaabd; font-size:12px; text-transform:uppercase; letter-spacing:.08em }}
.scroll {{ overflow-x:auto }} code {{ color:#c5e6ed; overflow-wrap:anywhere }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; color:#a9c7d7; font-size:12px }} summary {{ cursor:pointer; color:#84dae6 }}
@media(max-width:700px) {{ .twocol {{ grid-template-columns:1fr }} main {{ padding:25px 14px 60px }} }}
</style></head><body><main><div class="eyebrow">PARTICLEGAN · LUNAR FLIGHT LAB</div>
<h1>Two flights. One clock.</h1><p>Actual simulator frames on the same reset seed. Every GIF uses 50 simulation steps per second; the final image is held after termination to align the clocks. Orange indicates active downward thrust.</p>
<div class="badge">{esc(verdict)}</div>
<section class="panel hero"><h2>Aligned flight replay</h2><img src="comparison.gif" alt="Slow and fast Lunar flights aligned by simulation time"></section>
<section class="twocol"><div class="panel"><h2 class="cyan">Slow flight</h2><img src="slow.gif" alt="Slow Lunar flight"></div>
<div class="panel"><h2 class="orange">Fast flight</h2><img src="fast.gif" alt="Fast Lunar flight"></div></section>
<section class="panel"><h2>Flight board · seed {esc(slow['seed'])}</h2><table><thead><tr><th>Flight</th><th>Outcome</th><th>Steps</th><th>Return</th><th>Down boost steps</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Step count measures flight duration only for successful landings. A short crash does not count as a fast landing.</p></section>
{board_html}{provenance_html}
<section class="panel"><details><summary>Run report, checkpoints &amp; configuration</summary><pre>{metadata}</pre></details></section>
</main></body></html>"""


def render_comparison(outdir, slow_episode: dict, fast_episode: dict, report: dict) -> dict:
    """Export true-time GIFs and a standalone, local HTML dashboard.

    The two episodes must share a reset seed and simulator variant. Frame time is
    recovered from ``frame_stride`` (default 3), with a shorter final interval
    when termination falls between samples. No frames are sped up or dropped.
    """
    if int(slow_episode["seed"]) != int(fast_episode["seed"]):
        raise ValueError("Comparison requires the same reset seed")
    if slow_episode.get("variant") != fast_episode.get("variant"):
        raise ValueError("Comparison requires the same simulator variant")
    slow_times, fast_times = _samples(slow_episode), _samples(fast_episode)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = (str(report.get("slow_label", "Slow")), str(report.get("fast_label", "Fast")))
    for name, ep, samples, label in (("slow", slow_episode, slow_times, labels[0]),
                                     ("fast", fast_episode, fast_times, labels[1])):
        ticks = samples + [samples[-1] + HOLD_TICKS]
        cards = [_card(ep, samples, tick, label) for tick in samples]
        _save_gif(outdir / f"{name}.gif", cards, np.diff(ticks).tolist())

    from PIL import Image, ImageDraw
    end = max(slow_times[-1], fast_times[-1])
    ticks = sorted(set(slow_times + fast_times + [end + HOLD_TICKS]))
    joined = []
    for tick in ticks[:-1]:
        left = _card(slow_episode, slow_times, tick, labels[0])
        right = _card(fast_episode, fast_times, tick, labels[1])
        canvas = Image.new("RGB", (left.width + right.width + 24, max(left.height, right.height) + 24), "#07111d")
        canvas.paste(left, (8, 12))
        canvas.paste(right, (left.width + 16, 12))
        joined.append(canvas)
    _save_gif(outdir / "comparison.gif", joined, np.diff(ticks).tolist())
    (outdir / "index.html").write_text(_dashboard(report, slow_episode, fast_episode), encoding="utf-8")
    return {"slow_gif": "slow.gif", "fast_gif": "fast.gif",
            "comparison_gif": "comparison.gif", "dashboard": "index.html"}
